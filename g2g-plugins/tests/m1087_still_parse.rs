//! M1087 still-image parsers: `jpegparse` and `pngparse` frame a byte stream
//! back into whole images, and a `.jpg` now types by content because the
//! auto-plugged decode chain splices the parser ahead of the decoder.
//!
//! The framing reference is independent of the parsers' own logic: the JPEG
//! lengths come from the `Content-length` headers of the checked-in multipart
//! MJPEG fixture (ffmpeg's own output), and the PNG geometry comes from the
//! `png` crate's decode of the checked-in still, not from the `IHDR` reader
//! under test.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std,png,mjpeg`.
#![cfg(feature = "std")]

use std::path::PathBuf;

use g2g_core::frame::Frame;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, OutputSink, PipelinePacket, PushOutcome, Rate,
    VideoCodec,
};
use g2g_plugins::registry::default_registry;
use g2g_plugins::stillparse::{JpegParse, PngParse};
use g2g_plugins::typefind::sniff_caps;

/// ffmpeg's multipart MJPEG output: the JPEG framing reference, since each part
/// declares its own byte count in a MIME header.
const MJPEG_FIXTURE: &str = "multipart_64x48_jpeg.mjpg";
/// ffmpeg's PNG stills, in the two colour types our encoder never writes.
const PNG_FIXTURES: [&str; 2] = ["still_64x48_pal8.png", "still_64x48_gray16.png"];

/// Bytes per input buffer when a parser is driven directly: an odd size, so
/// images straddle buffer boundaries.
const CHUNK_LEN: usize = 233;

/// The launch-line legs need a clock and a temporary file, and both exist only
/// when a still-image decoder is compiled in.
#[cfg(any(feature = "png", feature = "mjpeg"))]
struct ZeroClock;
#[cfg(any(feature = "png", feature = "mjpeg"))]
impl g2g_core::PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl Collect {
    fn payloads(&self) -> Vec<Vec<u8>> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => {
                    Some(f.domain.as_system_slice().expect("system frame").to_vec())
                }
                _ => None,
            })
            .collect()
    }

    fn caps(&self) -> Vec<Caps> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn read_fixture(name: &str) -> Vec<u8> {
    std::fs::read(fixture(name)).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

#[cfg(any(feature = "png", feature = "mjpeg"))]
fn temp_path(tag: &str, extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "g2g-m1087-{tag}-{}.{extension}",
        std::process::id()
    ))
}

/// The JPEG bodies of a `multipart/x-mixed-replace` stream, each sized by its
/// own `Content-length` header: the framing reference the parser must match.
fn multipart_bodies(file: &[u8]) -> Vec<Vec<u8>> {
    const HEADER: &str = "content-length:";
    const HEADER_END: &[u8] = b"\r\n\r\n";
    let mut bodies = Vec::new();
    let mut at = 0;
    while at < file.len() {
        let text = String::from_utf8_lossy(&file[at..(at + 512).min(file.len())]).to_lowercase();
        let Some(found) = text.find(HEADER) else {
            break;
        };
        let length: usize = text[found + HEADER.len()..]
            .lines()
            .next()
            .expect("a value line")
            .trim()
            .parse()
            .expect("a byte count");
        let body = file[at..]
            .windows(HEADER_END.len())
            .position(|window| window == HEADER_END)
            .expect("headers end")
            + HEADER_END.len()
            + at;
        bodies.push(file[body..body + length].to_vec());
        at = body + length;
    }
    assert!(!bodies.is_empty(), "the fixture holds MJPEG parts");
    bodies
}

/// Push `bytes` through `element` in `CHUNK_LEN` pieces, then end the stream.
async fn drive<E: AsyncElement>(element: &mut E, bytes: &[u8]) -> Collect {
    let mut out = Collect::default();
    for piece in bytes.chunks(CHUNK_LEN) {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(piece.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        element
            .process(PipelinePacket::DataFrame(frame), &mut out)
            .await
            .expect("the chunk parses");
    }
    element
        .process(PipelinePacket::Eos, &mut out)
        .await
        .expect("the stream ends on an image boundary");
    out
}

/// The geometry a `CapsChanged` declares.
fn declared_geometry(caps: &Caps) -> (u32, u32) {
    let Caps::CompressedVideo {
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        ..
    } = caps
    else {
        panic!("a fixed geometry was declared, got {caps:?}");
    };
    (*width, *height)
}

fn still_caps(codec: VideoCodec, width: u32, height: u32) -> Caps {
    Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        // A still negotiated on its own gets the placeholder rate's floor.
        framerate: Rate::Fixed(1 << 16),
    }
}

// ---------------------------------------------------------------------------
// jpegparse
// ---------------------------------------------------------------------------

#[tokio::test]
async fn jpeg_images_match_the_multipart_part_lengths() {
    let bodies = multipart_bodies(&read_fixture(MJPEG_FIXTURE));
    let dump: Vec<u8> = bodies.concat();

    let mut parser = JpegParse::new();
    parser
        .configure_pipeline(&still_caps(VideoCodec::Mjpeg, 64, 48))
        .expect("JPEG caps");
    let out = drive(&mut parser, &dump).await;

    assert_eq!(
        out.payloads(),
        bodies,
        "every part comes back at the length its MIME header declared"
    );
    assert_eq!(parser.images_emitted() as usize, bodies.len());
}

#[tokio::test]
async fn jpeg_geometry_is_declared_from_the_frame_header() {
    let bodies = multipart_bodies(&read_fixture(MJPEG_FIXTURE));
    let mut parser = JpegParse::new();
    parser
        .configure_pipeline(&still_caps(VideoCodec::Mjpeg, 64, 48))
        .expect("JPEG caps");
    let out = drive(&mut parser, &bodies[0]).await;
    let declared = out.caps();
    assert_eq!(declared.len(), 1, "one declaration for one geometry");
    // The fixture's own name records what ffmpeg encoded.
    assert_eq!(declared_geometry(&declared[0]), (64, 48));
}

/// A JPEG now types by content, and the decode chain splices `jpegparse` ahead
/// of the decoder, so a file read in pieces smaller than the image still
/// decodes.
#[cfg(feature = "mjpeg")]
#[tokio::test]
async fn a_jpeg_file_decodes_through_decodebin_in_small_reads() {
    let bodies = multipart_bodies(&read_fixture(MJPEG_FIXTURE));
    let path = temp_path("still", "jpg");
    std::fs::write(&path, &bodies[0]).expect("the image is written");
    const READ_BYTES: usize = 512;
    assert!(
        bodies[0].len() > READ_BYTES,
        "the read size must split the image"
    );
    let line = format!(
        "filesrc location={} blocksize={READ_BYTES} ! decodebin ! fakesink",
        path.display()
    );
    let reg = default_registry();
    let graph = g2g_core::runtime::parse_launch(&reg, &line)
        .unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = g2g_core::runtime::run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");
    std::fs::remove_file(&path).ok();
    assert_eq!(stats.frames_consumed, 1, "the still decodes");
}

#[test]
fn a_jpeg_types_by_content() {
    let bodies = multipart_bodies(&read_fixture(MJPEG_FIXTURE));
    let caps = sniff_caps(&bodies[0]).expect("a JPEG is typed");
    assert!(
        matches!(
            caps,
            Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                ..
            }
        ),
        "got {caps:?}"
    );
}

// ---------------------------------------------------------------------------
// pngparse
// ---------------------------------------------------------------------------

#[tokio::test]
async fn png_files_frame_back_out_whole() {
    let files: Vec<Vec<u8>> = PNG_FIXTURES.iter().map(|name| read_fixture(name)).collect();
    let dump: Vec<u8> = files.concat();

    let mut parser = PngParse::new();
    parser
        .configure_pipeline(&still_caps(VideoCodec::Png, 64, 48))
        .expect("PNG caps");
    let out = drive(&mut parser, &dump).await;

    assert_eq!(
        out.payloads(),
        files,
        "two files joined in one stream come out as two images"
    );
}

/// The `IHDR` reader and the `png` crate must agree on the geometry.
#[cfg(feature = "png")]
#[tokio::test]
async fn png_geometry_matches_the_decoder() {
    use g2g_plugins::pngdec::PngDec;

    for name in PNG_FIXTURES {
        let file = read_fixture(name);

        let mut parser = PngParse::new();
        parser
            .configure_pipeline(&still_caps(VideoCodec::Png, 64, 48))
            .expect("PNG caps");
        let parsed = drive(&mut parser, &file).await;
        let (width, height) = declared_geometry(&parsed.caps()[0]);

        let mut decoder = PngDec::new();
        decoder
            .configure_pipeline(&still_caps(VideoCodec::Png, width, height))
            .expect("PNG caps");
        let decoded = drive(&mut decoder, &file).await;
        let Caps::RawVideo {
            width: Dim::Fixed(decoded_width),
            height: Dim::Fixed(decoded_height),
            ..
        } = decoded.caps()[0]
        else {
            panic!("the decoder declares a fixed geometry");
        };
        assert_eq!(
            (width, height),
            (decoded_width, decoded_height),
            "{name}: the IHDR reader agrees with the decoder"
        );
    }
}

#[cfg(feature = "png")]
#[tokio::test]
async fn a_png_file_decodes_through_decodebin_in_small_reads() {
    let path = temp_path("still", "png");
    let file = read_fixture(PNG_FIXTURES[0]);
    std::fs::write(&path, &file).expect("the image is written");
    const READ_BYTES: usize = 256;
    assert!(
        file.len() > READ_BYTES,
        "the read size must split the image"
    );
    let line = format!(
        "filesrc location={} blocksize={READ_BYTES} ! decodebin ! fakesink",
        path.display()
    );
    let reg = default_registry();
    let graph = g2g_core::runtime::parse_launch(&reg, &line)
        .unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = g2g_core::runtime::run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");
    std::fs::remove_file(&path).ok();
    assert_eq!(stats.frames_consumed, 1, "the still decodes");
}

/// The auto-plug search itself, so the two elements a still-image decode needs
/// are named: the framer ahead of the decoder.
#[cfg(all(feature = "png", feature = "mjpeg"))]
#[test]
fn decodebin_splices_the_framer_ahead_of_the_decoder() {
    use g2g_core::runtime::{is_raw_video, GraphNode, Registry};
    use g2g_core::Graph;
    use g2g_plugins::fakesink::FakeSink;
    use g2g_plugins::filesrc::FileSrc;

    for (name, codec, parser, decoder) in [
        (
            PNG_FIXTURES[0],
            VideoCodec::Png,
            g2g_core::log::short_type_name::<PngParse>(),
            g2g_core::log::short_type_name::<g2g_plugins::pngdec::PngDec>(),
        ),
        (
            MJPEG_FIXTURE,
            VideoCodec::Mjpeg,
            g2g_core::log::short_type_name::<JpegParse>(),
            g2g_core::log::short_type_name::<g2g_plugins::mjpegdec::MjpegDec>(),
        ),
    ] {
        let caps = still_caps(codec, 64, 48);
        let reg: Registry = default_registry();
        let mut graph: Graph<GraphNode> = Graph::new();
        let src = graph.add_source(GraphNode::source(FileSrc::new(fixture(name), caps.clone())));
        let sink = graph.add_sink(GraphNode::element(FakeSink::new()));
        let inserted = reg
            .decodebin(&mut graph, src, sink, &caps, &is_raw_video, 4)
            .unwrap_or_else(|e| panic!("{name} reaches raw video: {e:?}"));
        let names: Vec<&str> = inserted
            .iter()
            .map(|id| {
                graph
                    .element(*id)
                    .expect("a spliced element")
                    .log_category()
            })
            .collect();
        assert_eq!(names, vec![parser, decoder], "{name}");
    }
}

#[test]
fn gst_ivfparse_points_at_the_demuxer() {
    use g2g_plugins::gst_compat::{gst_equivalent, GstEquivalent};
    let reg = default_registry();
    assert_eq!(
        gst_equivalent(&reg, "ivfparse"),
        GstEquivalent::Renamed("ivfdemux")
    );
}
