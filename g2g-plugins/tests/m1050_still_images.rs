//! M1050 still-image codecs: `PngEnc` / `PngDec` (pure-Rust `png`) and `WebPDec`
//! (pure-Rust `image-webp`), plus the typefind + decodebin routing that gets a
//! `.png` / `.webp` file to them.
//!
//! Validated against ffmpeg as the reference peer in both directions: what
//! `pngenc` writes must decode in ffmpeg to the exact source pixels (PNG is
//! lossless), and what ffmpeg writes must decode through our elements to the
//! same pixels. The committed fixtures are ffmpeg's own output for the PNG
//! colour types the encoder here never produces (16-bit grayscale, palette) and
//! for WebP, which has no encoder here at all.
//!
//! The synthetic pattern is deterministic, so the lossless paths assert against
//! it directly and need no ffmpeg; the ffmpeg comparisons are extra evidence and
//! skip when it is missing.

#![cfg(all(feature = "png", feature = "webp"))]

use std::io::Write;
use std::process::{Command, Stdio};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::pngdec::PngDec;
use g2g_plugins::pngenc::PngEnc;
use g2g_plugins::typefind::{sniff_caps, still_image_caps};
use g2g_plugins::webpdec::WebPDec;

const W: u32 = 64;
const H: u32 = 48;
const RGBA_BYTES: usize = (W * H * 4) as usize;

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");

#[derive(Default)]
struct CaptureSink {
    caps: Vec<Caps>,
    frames: Vec<Vec<u8>>,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            match packet {
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.frames.push(s.to_vec());
                    }
                }
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// The source image: a deterministic RGBA gradient, opaque everywhere.
fn pattern_rgba() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(RGBA_BYTES);
    for y in 0..H {
        for x in 0..W {
            pixels.extend_from_slice(&[
                ((x * 4) % 256) as u8,
                ((y * 5) % 256) as u8,
                (((x + y) * 3) % 256) as u8,
                255,
            ]);
        }
    }
    pixels
}

fn data_frame(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

fn rgba_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

/// Push one buffer through an element and hand back what reached the sink.
async fn run_one<E: AsyncElement>(element: &mut E, bytes: Vec<u8>) -> CaptureSink {
    let mut sink = CaptureSink::default();
    element
        .process(data_frame(bytes), &mut sink)
        .await
        .expect("element accepts the buffer");
    sink
}

/// Encode the pattern with `PngEnc` at `level`.
async fn encode_pattern_png(level: u8) -> Vec<u8> {
    let mut encoder = PngEnc::new().with_compression_level(level);
    encoder
        .configure_pipeline(&rgba_caps())
        .expect("pngenc takes RGBA");
    let sink = run_one(&mut encoder, pattern_rgba()).await;
    assert_eq!(sink.frames.len(), 1, "one image out per image in");
    sink.frames.into_iter().next().unwrap()
}

/// Decode a PNG through the real `PngDec` element, returning the sink.
async fn decode_png(bytes: Vec<u8>) -> CaptureSink {
    let mut decoder = PngDec::new();
    decoder
        .configure_pipeline(&still_image_caps(VideoCodec::Png))
        .expect("pngdec takes image/png");
    run_one(&mut decoder, bytes).await
}

/// Decode a WebP through the real `WebPDec` element, returning the sink.
async fn decode_webp(bytes: Vec<u8>) -> CaptureSink {
    let mut decoder = WebPDec::new();
    decoder
        .configure_pipeline(&still_image_caps(VideoCodec::WebP))
        .expect("webpdec takes image/webp");
    run_one(&mut decoder, bytes).await
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{FIXTURE_DIR}{name}")).expect("committed fixture is readable")
}

/// Run ffmpeg over `stdin`, or `None` when it is missing or fails. Both payloads
/// are a few KB, well inside a pipe buffer, so a single write cannot block.
fn ffmpeg(args: &[&str], stdin: &[u8]) -> Option<Vec<u8>> {
    let mut child = Command::new("ffmpeg")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(stdin).ok()?;
    let out = child.wait_with_output().ok()?;
    out.status.success().then_some(out.stdout)
}

/// ffmpeg's decode of an image file to packed RGBA, the pixel oracle.
fn ffmpeg_to_rgba(image: &[u8]) -> Option<Vec<u8>> {
    ffmpeg(
        &[
            "-v",
            "error",
            "-f",
            "image2pipe",
            "-i",
            "pipe:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "pipe:1",
        ],
        image,
    )
}

/// Mean absolute per-sample difference, for the lossy comparisons.
fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "same sample count");
    let total: u64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| u64::from(x.abs_diff(*y)))
        .sum();
    total as f64 / a.len() as f64
}

fn decoded_geometry(sink: &CaptureSink) -> (u32, u32) {
    let Some(Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        ..
    }) = sink.caps.first()
    else {
        panic!(
            "the decoder announces fixed RGBA geometry, got {:?}",
            sink.caps
        );
    };
    (*w, *h)
}

/// PNG is lossless, so what `pngenc` writes must come back out of `pngdec` as the
/// exact source pixels, at every compression level.
#[tokio::test]
async fn pngenc_pngdec_roundtrip_is_bit_exact_at_every_level() {
    let source = pattern_rgba();
    let mut sizes = Vec::new();
    for level in [0u8, 6, 9] {
        let encoded = encode_pattern_png(level).await;
        assert!(
            encoded.starts_with(&[0x89, b'P', b'N', b'G']),
            "level {level} writes a real PNG signature"
        );
        sizes.push(encoded.len());
        let sink = decode_png(encoded).await;
        assert_eq!(decoded_geometry(&sink), (W, H));
        assert_eq!(
            sink.frames[0], source,
            "level {level} round-trips every pixel"
        );
    }
    // The compression-level property is applied, not just accepted: storing
    // uncompressed is bigger than deflating at the default.
    assert!(
        sizes[0] > sizes[1],
        "compression-level=0 stores, =6 deflates: {sizes:?}"
    );
}

/// The reference peer reads our PNG: ffmpeg's decode of `pngenc` output is the
/// source pixels, byte for byte.
#[tokio::test]
async fn ffmpeg_decodes_pngenc_output_bit_exact() {
    let encoded = encode_pattern_png(6).await;
    let Some(reference) = ffmpeg_to_rgba(&encoded) else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    assert_eq!(reference, pattern_rgba(), "ffmpeg reads back every pixel");
}

/// The reference peer writes a PNG we read: ffmpeg's encode of the same pattern
/// decodes through `pngdec` to the source pixels.
#[tokio::test]
async fn pngdec_decodes_ffmpeg_png_bit_exact() {
    let size = format!("{W}x{H}");
    let Some(encoded) = ffmpeg(
        &[
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &size,
            "-i",
            "pipe:0",
            "-f",
            "image2pipe",
            "-c:v",
            "png",
            "pipe:1",
        ],
        &pattern_rgba(),
    ) else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    let sink = decode_png(encoded).await;
    assert_eq!(decoded_geometry(&sink), (W, H));
    assert_eq!(sink.frames[0], pattern_rgba());
}

/// A 16-bit grayscale PNG decodes rather than being rejected: samples narrow to
/// their high byte and widen across RGB with an opaque alpha. ffmpeg's own
/// gray16 decode is the oracle for the byte that survives.
#[tokio::test]
async fn pngdec_narrows_16_bit_grayscale() {
    let encoded = fixture("still_64x48_gray16.png");
    let sink = decode_png(encoded.clone()).await;
    assert_eq!(decoded_geometry(&sink), (W, H));
    let decoded = &sink.frames[0];
    assert_eq!(decoded.len(), RGBA_BYTES);
    for pixel in decoded.as_chunks::<4>().0 {
        assert_eq!(
            (pixel[0], pixel[1], pixel[2]),
            (pixel[0], pixel[0], pixel[0]),
            "grayscale widens across RGB"
        );
        assert_eq!(pixel[3], 255, "no tRNS chunk means fully opaque");
    }

    let Some(reference) = ffmpeg(
        &[
            "-v",
            "error",
            "-f",
            "image2pipe",
            "-i",
            "pipe:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray16be",
            "pipe:1",
        ],
        &encoded,
    ) else {
        eprintln!("skipping the ffmpeg half: no ffmpeg");
        return;
    };
    let high_bytes: Vec<u8> = reference.as_chunks::<2>().0.iter().map(|s| s[0]).collect();
    let ours: Vec<u8> = decoded.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
    assert_eq!(ours, high_bytes, "STRIP_16 keeps the high byte");
}

/// A paletted PNG expands to RGBA, matching ffmpeg's own palette lookup.
#[tokio::test]
async fn pngdec_expands_a_palette_to_rgba() {
    let encoded = fixture("still_64x48_pal8.png");
    let sink = decode_png(encoded.clone()).await;
    assert_eq!(decoded_geometry(&sink), (W, H));
    assert_eq!(sink.frames[0].len(), RGBA_BYTES);
    let Some(reference) = ffmpeg_to_rgba(&encoded) else {
        eprintln!("skipping the ffmpeg half: no ffmpeg");
        return;
    };
    assert_eq!(sink.frames[0], reference, "same palette lookup as ffmpeg");
}

/// A lossless WebP written by libwebp decodes to the exact source pixels.
#[tokio::test]
async fn webpdec_decodes_lossless_bit_exact() {
    let sink = decode_webp(fixture("still_64x48_lossless.webp")).await;
    assert_eq!(decoded_geometry(&sink), (W, H));
    assert_eq!(
        sink.frames[0],
        pattern_rgba(),
        "VP8L is lossless, so every pixel survives"
    );
}

/// A lossy WebP decodes close to both the source and libwebp's own decode; the
/// two differ only in chroma upsampling, so this is a tolerance, not equality.
#[tokio::test]
async fn webpdec_decodes_lossy_close_to_the_reference() {
    let encoded = fixture("still_64x48_lossy.webp");
    let sink = decode_webp(encoded.clone()).await;
    assert_eq!(decoded_geometry(&sink), (W, H));
    let decoded = &sink.frames[0];
    assert_eq!(decoded.len(), RGBA_BYTES);

    let drift = mean_abs_diff(decoded, &pattern_rgba());
    assert!(
        drift < 12.0,
        "lossy decode tracks the source, drift {drift}"
    );

    let Some(reference) = ffmpeg_to_rgba(&encoded) else {
        eprintln!("skipping the ffmpeg half: no ffmpeg");
        return;
    };
    let vs_libwebp = mean_abs_diff(decoded, &reference);
    assert!(
        vs_libwebp < 3.0,
        "our VP8 decode tracks libwebp's, drift {vs_libwebp}"
    );
}

/// `no-fancy-upsampling` is applied, not just accepted: simple upsampling gives
/// different pixels from the bilinear default on a lossy image.
#[tokio::test]
async fn webpdec_simple_upsampling_changes_lossy_output() {
    let encoded = fixture("still_64x48_lossy.webp");
    let fancy = decode_webp(encoded.clone()).await;

    let mut decoder = WebPDec::new().with_no_fancy_upsampling(true);
    decoder
        .configure_pipeline(&still_image_caps(VideoCodec::WebP))
        .expect("webpdec takes image/webp");
    let simple = run_one(&mut decoder, encoded).await;

    assert_ne!(
        fancy.frames[0], simple.frames[0],
        "the upsampling property reaches the decoder"
    );
    let drift = mean_abs_diff(&fancy.frames[0], &simple.frames[0]);
    assert!(drift < 8.0, "still the same image, drift {drift}");
}

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("g2g-m1050-{tag}-{}.raw", std::process::id()))
}

/// Run a launch line to EOS and hand back what its `filesink` wrote.
async fn run_line_to_file(line: &str, out: &std::path::Path) -> Vec<u8> {
    let _ = std::fs::remove_file(out);
    let registry = g2g_plugins::registry::default_registry();
    let graph = parse_launch(&registry, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let written = std::fs::read(out).expect("the sink wrote a file");
    let _ = std::fs::remove_file(out);
    written
}

/// The whole auto-plug path over a real file: `filesrc ! decodebin` types the
/// bytes by content and plugs the image decoder the registry offers, so the
/// pixels reaching the sink are the ones the element produces on its own, and
/// ffmpeg's decode of the same file where ffmpeg is around.
#[tokio::test]
async fn decodebin_decodes_a_still_image_file() {
    let png = decode_png(fixture("still_64x48_pal8.png")).await;
    let webp = decode_webp(fixture("still_64x48_lossless.webp")).await;

    for (name, fixture_name, element_pixels) in [
        ("png", "still_64x48_pal8.png", &png.frames[0]),
        ("webp", "still_64x48_lossless.webp", &webp.frames[0]),
    ] {
        let source = format!("{FIXTURE_DIR}{fixture_name}");
        let out = temp_path(name);
        let line = format!(
            "filesrc location={source} ! decodebin ! videoconvert \
             ! video/x-raw,format=RGBA ! filesink location={}",
            out.display()
        );
        let decoded = run_line_to_file(&line, &out).await;
        assert_eq!(
            decoded.len(),
            RGBA_BYTES,
            "{fixture_name} decoded to one {W}x{H} RGBA frame"
        );
        assert_eq!(
            &decoded, element_pixels,
            "{fixture_name} auto-plugged to the decoder that types it"
        );
        let Some(reference) = ffmpeg_to_rgba(&fixture(fixture_name)) else {
            continue;
        };
        assert_eq!(decoded, reference, "{fixture_name} matches ffmpeg's decode");
    }
}

/// `pngenc` from a launch line, `compression-level` set as a property: the file it
/// writes is a real PNG that ffmpeg reads back at the source geometry, so the
/// snapshot pipeline works through the registry and not just as an element.
#[tokio::test]
async fn pngenc_snapshots_a_launch_line() {
    let out = temp_path("launch-png");
    let line = format!(
        "videotestsrc num-buffers=1 width={W} height={H} ! videoconvert \
         ! video/x-raw,format=RGBA ! pngenc compression-level=9 ! filesink location={}",
        out.display()
    );
    let written = run_line_to_file(&line, &out).await;
    assert!(
        written.starts_with(&[0x89, b'P', b'N', b'G']),
        "the sink holds one PNG"
    );

    let sink = decode_png(written.clone()).await;
    assert_eq!(decoded_geometry(&sink), (W, H));
    let Some(reference) = ffmpeg_to_rgba(&written) else {
        eprintln!("skipping the ffmpeg half: no ffmpeg");
        return;
    };
    assert_eq!(
        sink.frames[0], reference,
        "ffmpeg reads the launched snapshot as the same pixels"
    );
}

/// Typefind types the real files by content, so `filesrc ! decodebin` can reach
/// the right decoder without an extension to go on.
#[test]
fn typefind_types_the_fixtures_by_content() {
    let mut png = Vec::new();
    png.extend_from_slice(&fixture("still_64x48_pal8.png"));
    assert_eq!(sniff_caps(&png), Some(still_image_caps(VideoCodec::Png)));
    assert_eq!(
        sniff_caps(&fixture("still_64x48_lossless.webp")),
        Some(still_image_caps(VideoCodec::WebP))
    );
    assert_eq!(
        sniff_caps(&fixture("still_64x48_lossy.webp")),
        Some(still_image_caps(VideoCodec::WebP))
    );
}

/// A PNG whose IHDR claims a 200000 x 200000 RGBA image and that carries no
/// pixels: what an attacker writes to make a decoder size a buffer from a
/// header. Dropping the writer closes the file with `IEND`, so the framing is
/// complete and the header does reach the decoder.
fn huge_png_header() -> Vec<u8> {
    const ABSURD_SIDE: u32 = 200_000;
    let mut header = Vec::new();
    let mut encoder = png::Encoder::new(&mut header, ABSURD_SIDE, ABSURD_SIDE);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header().expect("IHDR is written");
    header
}

/// The lossless fixture's VP8L bitstream rewrapped in an extended container
/// whose VP8X canvas claims 20000 x 20000.
fn oversized_canvas_webp(lossless: &[u8]) -> Vec<u8> {
    const ABSURD_SIDE: u32 = 20_000;
    const VP8X_PAYLOAD_LEN: u32 = 10;
    let canvas = (ABSURD_SIDE - 1).to_le_bytes();
    let mut body = Vec::from(*b"WEBPVP8X");
    body.extend_from_slice(&VP8X_PAYLOAD_LEN.to_le_bytes());
    body.extend_from_slice(&[0; 4]); // flags + reserved
    body.extend_from_slice(&canvas[..3]);
    body.extend_from_slice(&canvas[..3]);
    // Everything from the source file's own chunk header onward.
    body.extend_from_slice(&lossless[12..]);

    let mut file = Vec::from(*b"RIFF");
    file.extend_from_slice(&(body.len() as u32).to_le_bytes());
    file.extend_from_slice(&body);
    file
}

/// Push one buffer through a freshly configured `PngDec`, returning what the
/// element did with it.
async fn png_verdict(bytes: Vec<u8>) -> (Result<(), G2gError>, CaptureSink) {
    let mut decoder = PngDec::new();
    decoder
        .configure_pipeline(&still_image_caps(VideoCodec::Png))
        .unwrap();
    let mut sink = CaptureSink::default();
    let verdict = decoder.process(data_frame(bytes), &mut sink).await;
    (verdict, sink)
}

async fn webp_verdict(bytes: Vec<u8>) -> (Result<(), G2gError>, CaptureSink) {
    let mut decoder = WebPDec::new();
    decoder
        .configure_pipeline(&still_image_caps(VideoCodec::WebP))
        .unwrap();
    let mut sink = CaptureSink::default();
    let verdict = decoder.process(data_frame(bytes), &mut sink).await;
    (verdict, sink)
}

/// Absurd and unparseable input fails instead of panicking or allocating on the
/// header's word.
#[tokio::test]
async fn malformed_png_fails_cleanly() {
    // A well-formed 200000 x 200000 IHDR (CRC and all), the size a header alone
    // can claim. The `png` crate parses it happily and would then hand back a
    // 160 GB output-buffer size, so the geometry bound is the only thing
    // standing between the header's word and the allocation.
    let (verdict, sink) = png_verdict(huge_png_header()).await;
    assert!(
        verdict.is_err(),
        "an absurd IHDR is refused, not allocated for"
    );
    assert!(sink.frames.is_empty());

    // Bytes that are not a PNG at all are rejected on the signature, before any
    // buffer is sized.
    let (verdict, sink) = png_verdict(vec![0u8; 512]).await;
    assert!(verdict.is_err());
    assert!(sink.frames.is_empty());
}

#[tokio::test]
async fn malformed_webp_fails_cleanly() {
    // The lossless fixture behind an extended-container header claiming a
    // 20000 x 20000 canvas. `image-webp` accepts that and reports a 1.2 GB
    // output size, so the geometry bound is what keeps a 90-byte file from
    // asking for 1.2 GB.
    let absurd = oversized_canvas_webp(&fixture("still_64x48_lossless.webp"));
    let (verdict, sink) = webp_verdict(absurd).await;
    assert!(
        verdict.is_err(),
        "an absurd VP8X canvas is refused, not allocated for"
    );
    assert!(sink.frames.is_empty());

    let (verdict, sink) = webp_verdict(vec![0u8; 512]).await;
    assert!(verdict.is_err());
    assert!(sink.frames.is_empty());
}

/// A stream that stops mid-image emits nothing and says so at end of stream,
/// rather than decoding a half-received file or dropping it in silence.
#[tokio::test]
async fn a_truncated_image_is_reported_at_end_of_stream() {
    let png = encode_pattern_png(6).await;
    let webp = fixture("still_64x48_lossless.webp");

    let mut decoder = PngDec::new();
    decoder
        .configure_pipeline(&still_image_caps(VideoCodec::Png))
        .unwrap();
    let mut sink = CaptureSink::default();
    decoder
        .process(data_frame(png[..png.len() / 2].to_vec()), &mut sink)
        .await
        .expect("a partial image is not yet an error");
    assert!(sink.frames.is_empty(), "half a PNG decodes to nothing");
    assert!(decoder
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .is_err());

    let mut decoder = WebPDec::new();
    decoder
        .configure_pipeline(&still_image_caps(VideoCodec::WebP))
        .unwrap();
    let mut sink = CaptureSink::default();
    decoder
        .process(data_frame(webp[..webp.len() / 2].to_vec()), &mut sink)
        .await
        .expect("a partial image is not yet an error");
    assert!(sink.frames.is_empty());
    assert!(decoder
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .is_err());
}

/// `filesrc` hands over read-sized chunks, not whole files, so the decoders
/// reassemble: an image split across many buffers decodes exactly once, and two
/// images arriving in one buffer decode as two.
#[tokio::test]
async fn images_are_reassembled_across_buffer_boundaries() {
    const CHUNK: usize = 7;

    let png = encode_pattern_png(6).await;
    let mut decoder = PngDec::new();
    decoder
        .configure_pipeline(&still_image_caps(VideoCodec::Png))
        .unwrap();
    let mut sink = CaptureSink::default();
    for chunk in png.chunks(CHUNK) {
        decoder
            .process(data_frame(chunk.to_vec()), &mut sink)
            .await
            .expect("every chunk is accepted");
    }
    assert_eq!(sink.frames.len(), 1, "one image out of many buffers");
    assert_eq!(sink.frames[0], pattern_rgba());
    assert_eq!(
        sink.caps.len(),
        1,
        "geometry is announced once, not per chunk"
    );

    let mut joined = png.clone();
    joined.extend_from_slice(&png);
    let sink = decode_png(joined).await;
    assert_eq!(
        sink.frames.len(),
        2,
        "two images in one buffer decode as two"
    );
    assert_eq!(sink.caps.len(), 1, "unchanged geometry is not re-announced");
}

/// The single-image JPEG path (GStreamer's `jpegenc`) is the MJPEG encoder, so
/// its output has to satisfy the same reference peer: ffmpeg decodes it to
/// something close to the source.
#[cfg(feature = "mjpeg-encode")]
#[tokio::test]
async fn ffmpeg_decodes_jpegenc_output_within_tolerance() {
    use g2g_plugins::mjpegenc::MjpegEnc;

    let mut encoder = MjpegEnc::new().with_quality(95);
    encoder
        .configure_pipeline(&rgba_caps())
        .expect("jpegenc takes RGBA");
    let sink = run_one(&mut encoder, pattern_rgba()).await;
    let jpeg = &sink.frames[0];
    assert!(jpeg.starts_with(&[0xff, 0xd8, 0xff]), "a real JPEG SOI");

    let Some(reference) = ffmpeg_to_rgba(jpeg) else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    let drift = mean_abs_diff(&reference, &pattern_rgba());
    assert!(drift < 6.0, "quality 95 stays close to the source, {drift}");
}
