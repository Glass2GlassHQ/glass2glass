//! M896 HLS packager sink: `HlsSink` cuts the byte stream a muxer feeds it into
//! media segment files and publishes an `.m3u8` media playlist beside them, for
//! both carriers HLS defines:
//!
//! ```text
//! ... ! tsmux  ! hlssink location=seg%05d.ts  playlist-location=out.m3u8
//! ... ! mp4mux ! hlssink location=seg%05d.m4s init-location=init.mp4
//! ```
//!
//! The segmenter adds and drops nothing, so the init segment plus the segment
//! files concatenate back to the muxer's own byte stream; the playlist is read
//! back with this repo's own `hls` parser. ffmpeg is the reference peer: it
//! demuxes and decodes the published playlist, one packet per access unit.
//!
//! `HlsSink` is `std`-gated, so this file is too.
#![cfg(feature = "std")]

use std::path::{Path, PathBuf};
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement, MultiOutputElement,
    OutputSink, PropValue, PropertySpec, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::hls::{parse, Playlist};
use g2g_plugins::hlssink::HlsSink;
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};
use g2g_plugins::mp4mux::Mp4Mux;
use g2g_plugins::registry::default_registry;
use g2g_plugins::tsmux::TsMux;
use g2g_plugins::tsmuxn::TsMux as TsMuxN;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
/// One frame at 25 fps, a whole number of the muxers' 90 kHz ticks, so the
/// synthetic fixture's segment durations are exact.
const FRAME_25FPS_NS: u64 = 40_000_000;
/// Frames per GOP in the synthetic fixture: exactly 1 s, the target duration the
/// segmenting tests use.
const GOP: usize = 25;

// --- pipeline plumbing ----------------------------------------------------

/// Captures the byte-stream frames an element pushes downstream, timing and all,
/// which is exactly what the runner would hand the next element.
#[derive(Default)]
struct CaptureSink {
    frames: Vec<(Vec<u8>, FrameTiming)>,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames.push((s.to_vec(), f.timing));
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[derive(Default)]
struct PortCapture {
    ports: Vec<Vec<(Vec<u8>, FrameTiming)>>,
}

impl g2g_core::MultiOutputSink for PortCapture {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if self.ports.len() <= port {
                self.ports.resize(port + 1, Vec::new());
            }
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.ports[port].push((s.to_vec(), f.timing));
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }

    fn port_count(&self) -> usize {
        self.ports.len().max(1)
    }
}

struct NullSink;
impl OutputSink for NullSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn frame(bytes: Vec<u8>, timing: FrameTiming) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing,
        0,
    ))
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(25 << 16),
    }
}

fn aac_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 2,
        sample_rate: 48_000,
    }
}

fn bytestream(encoding: ByteStreamEncoding) -> Caps {
    Caps::ByteStream { encoding }
}

/// Access units through `TsMux`, as the frames it pushes downstream.
async fn mux_ts(aus: &[(Vec<u8>, FrameTiming)]) -> Vec<(Vec<u8>, FrameTiming)> {
    let mut mux = TsMux::new();
    mux.configure_pipeline(&h264_caps()).expect("configure");
    let mut sink = CaptureSink::default();
    for (au, timing) in aus {
        mux.process(frame(au.clone(), *timing), &mut sink)
            .await
            .expect("mux");
    }
    mux.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    sink.frames
}

/// The same access units through the multi-track `tsmuxn`, with an AAC AU
/// interleaved mid-frame on a second pad (flagged a sync sample, as every AAC AU
/// is).
async fn mux_ts_av(aus: &[(Vec<u8>, FrameTiming)]) -> Vec<(Vec<u8>, FrameTiming)> {
    let adts: Vec<u8> = vec![0xFF, 0xF1, 0x4C, 0x80, 0x01, 0x00, 0xFC, 0x00];
    let mut mux = TsMuxN::new(2);
    mux.configure_pipeline(0, &h264_caps())
        .expect("configure v");
    mux.configure_pipeline(1, &aac_caps()).expect("configure a");
    let mut sink = CaptureSink::default();
    for (au, timing) in aus {
        mux.process(0, frame(au.clone(), *timing), &mut sink)
            .await
            .expect("mux video");
        let audio = FrameTiming {
            pts_ns: timing.pts_ns + FRAME_25FPS_NS / 2,
            keyframe: true,
            ..FrameTiming::default()
        };
        mux.process(1, frame(adts.clone(), audio), &mut sink)
            .await
            .expect("mux audio");
    }
    for input in 0..2 {
        mux.process(input, PipelinePacket::Eos, &mut sink)
            .await
            .expect("mux eos");
    }
    sink.frames
}

/// Access units through `Mp4Mux` batched into `fragment_ms` fragments.
async fn mux_fmp4(aus: &[(Vec<u8>, FrameTiming)], fragment_ms: u64) -> Vec<(Vec<u8>, FrameTiming)> {
    let mut mux = Mp4Mux::new().with_fragment_duration_ms(fragment_ms);
    mux.configure_pipeline(&h264_caps()).expect("configure");
    let mut sink = CaptureSink::default();
    for (au, timing) in aus {
        mux.process(frame(au.clone(), *timing), &mut sink)
            .await
            .expect("mux");
    }
    mux.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    sink.frames
}

/// Run muxed frames through the sink and close the stream.
async fn package(
    sink: &mut HlsSink,
    encoding: ByteStreamEncoding,
    frames: &[(Vec<u8>, FrameTiming)],
) {
    sink.configure_pipeline(&bytestream(encoding))
        .expect("configure hlssink");
    let mut out = NullSink;
    for (bytes, timing) in frames {
        sink.process(frame(bytes.clone(), *timing), &mut out)
            .await
            .expect("package frame");
    }
    sink.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("package eos");
}

// --- fixtures -------------------------------------------------------------

/// A fresh, empty directory for one test's output.
fn work_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("g2g-m896-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create work dir");
    dir
}

fn at(dir: &Path, name: &str) -> String {
    dir.join(name).to_string_lossy().into_owned()
}

fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}

/// `groups` GOPs of [`GOP`] access units at 25 fps, each opening on an IDR. The
/// muxers never look inside a slice, so synthetic NALs suffice for the
/// segmenting tests; the ffmpeg oracle uses a real encode instead.
fn synthetic_aus(groups: usize) -> Vec<(Vec<u8>, FrameTiming)> {
    let sps: &[u8] = &[0x67, 0x42, 0x00, 0x1e, 0x88];
    let pps: &[u8] = &[0x68, 0xce, 0x3c, 0x80];
    let idr: &[u8] = &[0x65, 0x88, 0x84, 0x00];
    let inter: &[u8] = &[0x41, 0x9a, 0x00];
    let mut aus = Vec::new();
    for g in 0..groups {
        for i in 0..GOP {
            let index = (g * GOP + i) as u64;
            let bytes = if i == 0 {
                annexb(&[sps, pps, idr])
            } else {
                annexb(&[inter])
            };
            aus.push((
                bytes,
                FrameTiming {
                    pts_ns: index * FRAME_25FPS_NS,
                    dts_ns: index * FRAME_25FPS_NS,
                    duration_ns: FRAME_25FPS_NS,
                    keyframe: i == 0,
                    ..FrameTiming::default()
                },
            ));
        }
    }
    aus
}

// --- reference peers ------------------------------------------------------

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
        && Command::new("ffprobe").arg("-version").output().is_ok()
}

/// A 2 s H.264 encode in a fragmented MP4, 15-frame GOPs, as real access units.
async fn real_access_units(dir: &Path) -> Vec<(Vec<u8>, FrameTiming)> {
    let path = dir.join("source.mp4");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg(format!("testsrc2=size={WIDTH}x{HEIGHT}:rate=30:duration=2"))
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0"])
        .args(["-g", "15", "-movflags", "cmaf", "-f", "mp4"])
        .arg(&path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the fixture");
    let file = std::fs::read(&path).expect("read fixture");

    let streams = forwardable_streams(&file);
    let ports: Vec<Mp4Port> = streams
        .iter()
        .map(|s| Mp4Port {
            track_id: s.track_id,
            caps: s.caps.clone(),
        })
        .collect();
    let mut demux = Mp4DemuxN::new(ports);
    demux
        .configure_pipeline(&bytestream(ByteStreamEncoding::IsoBmff))
        .expect("configure demux");
    let mut tap = PortCapture::default();
    tap.ports.resize(streams.len(), Vec::new());
    demux
        .process(frame(file, FrameTiming::default()), &mut tap)
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut tap)
        .await
        .expect("demux eos");
    tap.ports.remove(0)
}

/// ffmpeg reads the playlist end to end (`-f null -`), failing on any demuxer or
/// decoder complaint, and reports how many packets it decoded.
fn ffmpeg_reads_playlist(playlist: &str) -> u64 {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i", playlist, "-f", "null", "-"])
        .output()
        .expect("run ffmpeg");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && err.is_empty(),
        "ffmpeg played {playlist} cleanly: {err}"
    );
    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0"])
        .args(["-count_packets", "-show_entries", "stream=nb_read_packets"])
        .args(["-of", "default=nw=1:nk=1", playlist])
        .output()
        .expect("run ffprobe");
    assert!(
        probe.status.success(),
        "ffprobe read {playlist}: {}",
        String::from_utf8_lossy(&probe.stderr)
    );
    // The HLS demuxer reports the video stream once per program, so the count
    // comes back on more than one line; they agree, take the first.
    String::from_utf8_lossy(&probe.stdout)
        .lines()
        .next()
        .expect("ffprobe reported a packet count")
        .trim()
        .parse()
        .expect("packet count")
}

fn have_gstreamer() -> bool {
    Command::new("gst-launch-1.0")
        .arg("--version")
        .output()
        .is_ok()
}

/// GStreamer's own HLS client (`playbin3` -> `hlsdemux2`) plays the playlist
/// through to EOS. It must be reached by URI: `hlsdemux2` resolves the segment
/// URIs against the manifest's own.
fn gst_plays_playlist(playlist: &str) {
    let out = Command::new("gst-launch-1.0")
        .arg("playbin3")
        .arg(format!("uri=file://{playlist}"))
        .args(["video-sink=fakesink", "audio-sink=fakesink"])
        .output()
        .expect("run gst-launch-1.0");
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.status.success() && log.contains("Got EOS") && !log.contains("ERROR"),
        "gst-launch played {playlist}:\n{log}"
    );
}

// --- helpers on the published output --------------------------------------

/// The published playlist, read back through this repo's own parser.
fn read_playlist(path: &str) -> g2g_plugins::hls::MediaPlaylist {
    let text = std::fs::read_to_string(path).expect("playlist written");
    match parse(&text).expect("playlist parses") {
        Playlist::Media(m) => m,
        Playlist::Master(_) => panic!("hlssink writes a media playlist, got a master:\n{text}"),
    }
}

/// Every listed segment's bytes, concatenated in playlist order.
fn concat_segments(dir: &Path, playlist: &g2g_plugins::hls::MediaPlaylist) -> Vec<u8> {
    let mut out = Vec::new();
    for segment in &playlist.segments {
        out.extend_from_slice(&std::fs::read(dir.join(&segment.uri)).expect("segment file"));
    }
    out
}

fn declares(specs: &[PropertySpec], name: &str) -> bool {
    specs.iter().any(|s| s.name == name)
}

// --- tests ----------------------------------------------------------------

/// Every knob is a launch property that round-trips onto the field, and the
/// element resolves under its own name plus the gst HLS sink names.
#[test]
fn properties_round_trip_and_the_registry_resolves_the_gst_names() {
    let mut sink = HlsSink::new("seg%05d.ts");
    let string_props = [
        ("location", "s%03d.m4s"),
        ("playlist-location", "/tmp/p.m3u8"),
        ("init-location", "/tmp/init.mp4"),
        ("playlist-root", "https://cdn/hls"),
    ];
    for (name, value) in string_props {
        assert!(declares(sink.properties(), name), "{name} is declared");
        sink.set_property(name, PropValue::Str(value.into()))
            .expect("set");
        assert_eq!(
            sink.get_property(name),
            Some(PropValue::Str(value.into())),
            "{name} round-trips"
        );
    }
    for (name, value) in [
        ("target-duration", 6u64),
        ("playlist-length", 3),
        ("max-files", 4),
    ] {
        assert!(declares(sink.properties(), name), "{name} is declared");
        sink.set_property(name, PropValue::Uint(value))
            .expect("set");
        assert_eq!(
            sink.get_property(name),
            Some(PropValue::Uint(value)),
            "{name} round-trips"
        );
    }
    assert!(sink.set_property("nope", PropValue::Uint(1)).is_err());

    let reg = default_registry();
    assert!(reg.make_element("hlssink").is_some(), "registered by name");
    for gst_name in ["hlssink2", "hlssink3", "hlscmafsink"] {
        assert!(
            reg.make_element(gst_name).is_some(),
            "{gst_name} aliases the packager"
        );
    }
}

/// MPEG-TS: segments start at a keyframe, close at the first one past the
/// target, and together are byte for byte the stream `tsmux` produced.
#[tokio::test]
async fn ts_segments_cut_at_keyframes_and_concatenate_to_the_mux_output() {
    let dir = work_dir("ts");
    let playlist_path = at(&dir, "out.m3u8");
    let muxed = mux_ts(&synthetic_aus(2)).await;
    let whole: Vec<u8> = muxed.iter().flat_map(|(b, _)| b.clone()).collect();

    let mut sink = HlsSink::new(at(&dir, "seg%05d.ts"))
        .with_playlist_location(&playlist_path)
        .with_target_duration(1)
        .with_playlist_length(0)
        .with_max_files(0);
    package(&mut sink, ByteStreamEncoding::MpegTs, &muxed).await;

    let playlist = read_playlist(&playlist_path);
    assert_eq!(sink.segments_written(), 2, "two 1 s segments from 2 s");
    assert_eq!(playlist.segments.len(), 2);
    assert_eq!(playlist.media_sequence, 0, "nothing has rolled off yet");
    assert!(playlist.end_list, "EOS closes the playlist for VOD");
    assert_eq!(playlist.map_uri, None, "MPEG-TS needs no init segment");
    assert_eq!(playlist.target_duration_secs, 1);
    for segment in &playlist.segments {
        assert!(
            (segment.duration_ms as i64 - 1000).abs() < 50,
            "segment {} is about the 1 s target: {} ms",
            segment.uri,
            segment.duration_ms
        );
    }
    assert_eq!(
        playlist
            .segments
            .iter()
            .map(|s| s.uri.clone())
            .collect::<Vec<_>>(),
        vec!["seg00000.ts", "seg00001.ts"],
        "playlist names the segments relative to itself"
    );
    assert_eq!(
        concat_segments(&dir, &playlist),
        whole,
        "segmenting adds and drops nothing"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// M908: an A/V multiplex from `tsmuxn` segments the same as the single-track
/// one, because the muxer carries the video pad's sync flag onto its output.
#[tokio::test]
async fn av_multiplex_segments_on_the_video_keyframes() {
    let dir = work_dir("ts-av");
    let playlist_path = at(&dir, "av.m3u8");
    let muxed = mux_ts_av(&synthetic_aus(2)).await;

    let mut sink = HlsSink::new(at(&dir, "av%05d.ts"))
        .with_playlist_location(&playlist_path)
        .with_target_duration(1)
        .with_playlist_length(0)
        .with_max_files(0);
    package(&mut sink, ByteStreamEncoding::MpegTs, &muxed).await;

    assert_eq!(sink.segments_written(), 2, "two 1 s segments, not one blob");
    let playlist = read_playlist(&playlist_path);
    assert_eq!(playlist.segments.len(), 2);
    assert_eq!(
        concat_segments(&dir, &playlist),
        muxed
            .iter()
            .flat_map(|(b, _)| b.clone())
            .collect::<Vec<_>>(),
        "segmenting adds and drops nothing"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// fMP4: the `ftyp`+`moov` init segment is written once and named by
/// `#EXT-X-MAP`, fragments are never split across segments, and init plus
/// segments are byte for byte the stream `mp4mux` produced.
#[tokio::test]
async fn fmp4_segments_carry_an_init_map_and_concatenate_to_the_mux_output() {
    let dir = work_dir("fmp4");
    let playlist_path = at(&dir, "out.m3u8");
    // 400 ms fragments close at the next keyframe, i.e. one 500 ms GOP each, so a
    // 1 s target segment is exactly two fragments.
    let muxed = mux_fmp4(&synthetic_aus(2), 400).await;
    let whole: Vec<u8> = muxed.iter().flat_map(|(b, _)| b.clone()).collect();

    let mut sink = HlsSink::new(at(&dir, "seg%05d.m4s"))
        .with_playlist_location(&playlist_path)
        .with_init_location(at(&dir, "init.mp4"))
        .with_target_duration(1)
        .with_playlist_length(0)
        .with_max_files(0);
    package(&mut sink, ByteStreamEncoding::IsoBmff, &muxed).await;

    let playlist = read_playlist(&playlist_path);
    assert_eq!(playlist.map_uri.as_deref(), Some("init.mp4"));
    assert_eq!(sink.segments_written(), 2, "two 1 s segments from 2 s");
    assert_eq!(playlist.segments.len(), 2);
    for segment in &playlist.segments {
        assert!(
            (segment.duration_ms as i64 - 1000).abs() < 50,
            "segment {} is about the 1 s target: {} ms",
            segment.uri,
            segment.duration_ms
        );
    }

    let init = std::fs::read(dir.join("init.mp4")).expect("init segment written");
    assert_eq!(&init[4..8], b"ftyp", "the init segment opens with ftyp");
    assert!(
        init.windows(4).any(|w| w == b"moov"),
        "and carries the moov"
    );
    // Every media segment starts at a fragment, never mid-`moof`.
    for segment in &playlist.segments {
        let bytes = std::fs::read(dir.join(&segment.uri)).expect("segment file");
        assert_eq!(
            &bytes[4..8],
            b"moof",
            "segment {} opens at a fragment",
            segment.uri
        );
    }
    let mut rebuilt = init;
    rebuilt.extend_from_slice(&concat_segments(&dir, &playlist));
    assert_eq!(rebuilt, whole, "init + segments is the muxer's byte stream");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A live playlist rolls: it lists only `playlist-length` segments, advances
/// `#EXT-X-MEDIA-SEQUENCE` as older ones fall off, and `max-files` deletes the
/// segments that are no longer listed.
#[tokio::test]
async fn rolling_playlist_advances_media_sequence_and_prunes_files() {
    let dir = work_dir("rolling");
    let playlist_path = at(&dir, "live.m3u8");
    let muxed = mux_ts(&synthetic_aus(4)).await;

    let mut sink = HlsSink::new(at(&dir, "live%03d.ts"))
        .with_playlist_location(&playlist_path)
        .with_target_duration(1)
        .with_playlist_length(2)
        .with_max_files(2);
    package(&mut sink, ByteStreamEncoding::MpegTs, &muxed).await;

    assert_eq!(sink.segments_written(), 4, "four 1 s segments from 4 s");
    let playlist = read_playlist(&playlist_path);
    assert_eq!(playlist.segments.len(), 2, "only the window is listed");
    assert_eq!(
        playlist.media_sequence, 2,
        "two segments rolled off the front"
    );
    assert_eq!(
        playlist.segments[0].uri, "live002.ts",
        "the window is the newest segments"
    );
    for segment in &playlist.segments {
        assert!(
            dir.join(&segment.uri).exists(),
            "listed segment {} is on disk",
            segment.uri
        );
    }
    for gone in ["live000.ts", "live001.ts"] {
        assert!(!dir.join(gone).exists(), "{gone} was pruned by max-files");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A caps this sink cannot segment is refused at negotiation rather than
/// producing files it cannot cut.
#[test]
fn rejects_a_container_it_cannot_segment() {
    let mut sink = HlsSink::new("seg%05d.ts");
    assert!(sink
        .configure_pipeline(&bytestream(ByteStreamEncoding::Matroska))
        .is_err());
    assert!(sink.configure_pipeline(&h264_caps()).is_err());
    assert!(sink
        .configure_pipeline(&bytestream(ByteStreamEncoding::MpegTs))
        .is_ok());
}

/// The reference peer: ffmpeg demuxes and decodes both published playlists of a
/// real encode, one packet per access unit and no complaint.
#[tokio::test]
async fn ffmpeg_plays_the_published_playlists() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg on this host");
        return;
    }
    let dir = work_dir("oracle");
    let aus = real_access_units(&dir).await;
    assert_eq!(aus.len(), 60, "2 s at 30 fps");

    let ts_playlist = at(&dir, "ts.m3u8");
    let mut ts_sink = HlsSink::new(at(&dir, "ts%05d.ts"))
        .with_playlist_location(&ts_playlist)
        .with_target_duration(1)
        .with_playlist_length(0)
        .with_max_files(0);
    package(
        &mut ts_sink,
        ByteStreamEncoding::MpegTs,
        &mux_ts(&aus).await,
    )
    .await;

    let fmp4_playlist = at(&dir, "fmp4.m3u8");
    let mut fmp4_sink = HlsSink::new(at(&dir, "f%05d.m4s"))
        .with_playlist_location(&fmp4_playlist)
        .with_init_location(at(&dir, "finit.mp4"))
        .with_target_duration(1)
        .with_playlist_length(0)
        .with_max_files(0);
    package(
        &mut fmp4_sink,
        ByteStreamEncoding::IsoBmff,
        &mux_fmp4(&aus, 400).await,
    )
    .await;

    assert_eq!(ts_sink.segments_written(), 2);
    assert_eq!(fmp4_sink.segments_written(), 2);
    assert_eq!(
        ffmpeg_reads_playlist(&ts_playlist),
        aus.len() as u64,
        "the MPEG-TS playlist decodes to one packet per access unit"
    );
    assert_eq!(
        ffmpeg_reads_playlist(&fmp4_playlist),
        aus.len() as u64,
        "and so does the fMP4 playlist through its EXT-X-MAP"
    );

    persist::record_evidence(
        "hlssink",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264")
            .detail("ffmpeg demuxes and decodes hlssink's MPEG-TS and fMP4 (EXT-X-MAP) playlists"),
    )
    .expect("record oracle evidence");

    if have_gstreamer() {
        for playlist in [&ts_playlist, &fmp4_playlist] {
            gst_plays_playlist(playlist);
        }
        persist::record_evidence(
            "hlssink",
            &Evidence::new(ConformanceDimension::Oracle)
                .peer("gstreamer")
                .codec("h264")
                .detail("gst-launch playbin3 plays hlssink's MPEG-TS and fMP4 playlists to EOS"),
        )
        .expect("record oracle evidence");
    } else {
        eprintln!("skipping the GStreamer leg: no gst-launch-1.0 on this host");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The other half of the loop: g2g's own HLS client reads the published
/// playlist back over HTTP and delivers exactly the bytes `tsmux` produced.
#[cfg(feature = "hls")]
#[tokio::test]
async fn hlssrc_reads_back_the_published_playlist() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::hlssrc::HlsSrc;

    let dir = work_dir("loopback");
    let playlist_path = at(&dir, "loop.m3u8");
    let muxed = mux_ts(&synthetic_aus(2)).await;
    let whole: Vec<u8> = muxed.iter().flat_map(|(b, _)| b.clone()).collect();
    let mut sink = HlsSink::new(at(&dir, "loop%05d.ts"))
        .with_playlist_location(&playlist_path)
        .with_target_duration(1)
        .with_playlist_length(0)
        .with_max_files(0);
    package(&mut sink, ByteStreamEncoding::MpegTs, &muxed).await;

    let mut src = HlsSrc::new(format!("{}/loop.m3u8", serve_dir(&dir)));
    src.configure_pipeline(&bytestream(ByteStreamEncoding::MpegTs))
        .expect("configure hlssrc");
    let mut tap = ByteCapture::default();
    let segments = src
        .run(&mut tap)
        .await
        .expect("hlssrc streams the playlist");

    assert!(tap.eos, "the ENDLIST playlist terminates");
    assert_eq!(segments, 2, "one frame per published segment");
    assert_eq!(tap.body, whole, "the loop is byte-exact");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Serves the files of `dir` over HTTP, returning the base URL. One connection
/// at a time, which is all the segment loop needs.
#[cfg(feature = "hls")]
fn serve_dir(dir: &Path) -> String {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let root = dir.to_path_buf();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { break };
            let mut request = Vec::new();
            let mut byte = [0u8; 1];
            while stream.read(&mut byte).unwrap_or(0) == 1 {
                request.push(byte[0]);
                if request.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            let line = String::from_utf8_lossy(&request);
            let path = line.split_whitespace().nth(1).unwrap_or("");
            // Serve by file name only: no traversal out of the work directory.
            let name = path.rsplit('/').next().unwrap_or("");
            let Ok(body) = std::fs::read(root.join(name)) else {
                let _ = stream.write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
                continue;
            };
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&body);
        }
    });
    format!("http://127.0.0.1:{port}")
}

/// Collects a source's byte stream and whether it ended.
#[cfg(feature = "hls")]
#[derive(Default)]
struct ByteCapture {
    body: Vec<u8>,
    eos: bool,
}

#[cfg(feature = "hls")]
impl OutputSink for ByteCapture {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.body.extend_from_slice(s);
                    }
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}
