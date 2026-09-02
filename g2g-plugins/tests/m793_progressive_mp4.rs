//! M793: `Mp4MuxN`'s progressive (non-fragmented) layout. `fragmented = false`
//! buffers the whole movie and writes `ftyp` + one `mdat` + a `moov` carrying
//! real sample tables (`stts` / `ctts` / `stss` / `stsc` / `stsz` /
//! `stco`) and real `mvhd` / `tkhd` / `mdhd` durations at EOS.
//!
//! This is what M791 measured as missing: `ffprobe` derives a *fragmented*
//! file's duration by summing sample durations and applies an edit list only as
//! a timestamp shift, so the Opus pre-skip trim never showed in the reported
//! number. With a real sample table it does, and a 1.0 s Opus source reports
//! `duration=1.000000` exactly, the same layout ffmpeg's own `-c copy` remux
//! writes.
//!
//! The fragmented layout stays the default and is unchanged; every check here
//! that has a fragmented counterpart asserts the two agree on the media
//! (decoded PCM, video frame count) and differ only in the reported timeline.
#![cfg(feature = "std")]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, G2gError, MultiInputElement,
    MultiOutputElement, MultiOutputSink, OutputSink, PropValue, PushOutcome,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};
use g2g_plugins::mp4muxn::Mp4MuxN;
use g2g_plugins::oggdemux::OggDemux;

const WIDTH: usize = 320;
const HEIGHT: usize = 240;

#[derive(Default)]
struct CaptureSink {
    frames: Vec<(Vec<u8>, FrameTiming)>,
}

impl CaptureSink {
    fn bytes(&self) -> Vec<u8> {
        self.frames.iter().flat_map(|(b, _)| b.clone()).collect()
    }
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

/// Per-port capture of a demuxer's output: the frames and the refined caps (a
/// compressed-audio port negotiates `0/0` and refines at runtime, which the
/// muxer needs to size its `mdhd`).
#[derive(Default)]
struct PortCapture {
    ports: Vec<Vec<(Vec<u8>, FrameTiming)>>,
    caps: Vec<Option<Caps>>,
}

impl MultiOutputSink for PortCapture {
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
                self.caps.resize(port + 1, None);
            }
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.ports[port].push((s.to_vec(), f.timing));
                    }
                }
                PipelinePacket::CapsChanged(c) => self.caps[port] = Some(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }

    fn port_count(&self) -> usize {
        self.ports.len().max(1)
    }
}

/// One input pad's worth of muxer input: the caps it negotiated, the concrete
/// caps a demuxer refined to at runtime (`None` when negotiation was already
/// concrete), and its frames.
struct MuxTrack {
    nego: Caps,
    refined: Option<Caps>,
    frames: Vec<(Vec<u8>, FrameTiming)>,
}

fn frame(data: Vec<u8>, timing: FrameTiming) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        timing,
        0,
    ))
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
        && Command::new("ffprobe").arg("-version").output().is_ok()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m793-{}-{name}", std::process::id()))
}

/// ffprobe's `key=value` lines for one stream selector (`a:0` / `v:0`), plus the
/// container duration under the key `format_duration`.
fn probe(path: &PathBuf, select: &str) -> Vec<(String, String)> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", select])
        .args([
            "-show_entries",
            "stream=codec_name,width,height,channels,sample_rate,duration,start_time,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1",
        ])
        .arg(path)
        .output()
        .expect("run ffprobe");
    assert!(
        out.status.success(),
        "ffprobe read {}: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stderr.is_empty(),
        "ffprobe read {} without complaint: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    // `duration` appears twice (stream then format); the second is the container's.
    let mut seen_stream_duration = false;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().split_once('='))
        .map(|(k, v)| {
            let key = if k == "duration" && seen_stream_duration {
                "format_duration"
            } else {
                if k == "duration" {
                    seen_stream_duration = true;
                }
                k
            };
            (key.to_string(), v.to_string())
        })
        .collect()
}

fn field<'a>(probed: &'a [(String, String)], key: &str) -> &'a str {
    probed
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("ffprobe reported {key}, got {probed:?}"))
}

/// ffmpeg's decode of a file's audio as raw interleaved S16LE, failing the test
/// on any decoder complaint.
fn decode_audio(path: &PathBuf) -> Vec<u8> {
    decode(path, &["-f", "s16le", "-c:a", "pcm_s16le"])
}

/// ffmpeg's decode of a file's video as raw I420, so the byte count divides into
/// a frame count.
fn decode_video(path: &PathBuf) -> Vec<u8> {
    decode(path, &["-f", "rawvideo", "-pix_fmt", "yuv420p"])
}

fn decode(path: &PathBuf, args: &[&str]) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(args)
        .arg("-")
        .output()
        .expect("run ffmpeg");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && err.is_empty(),
        "ffmpeg decoded {} cleanly: {err}",
        path.display()
    );
    assert!(!out.stdout.is_empty(), "ffmpeg decoded something");
    out.stdout
}

/// Mux `tracks` (per input pad: its refined caps, then its frames) into one MP4
/// in the chosen layout, interleaving the pads by PTS the way a runner does.
async fn mux(tracks: &[MuxTrack], fragmented: bool) -> Vec<u8> {
    let mut m = Mp4MuxN::new(tracks.len()).with_fragmented(fragmented);
    let mut sink = CaptureSink::default();
    for (i, t) in tracks.iter().enumerate() {
        m.configure_pipeline(i, &t.nego).expect("configure mp4mux");
    }
    for (i, t) in tracks.iter().enumerate() {
        if let Some(caps) = &t.refined {
            m.process(i, PipelinePacket::CapsChanged(caps.clone()), &mut sink)
                .await
                .expect("caps");
        }
    }
    let mut cursors = vec![0usize; tracks.len()];
    loop {
        let next = (0..tracks.len())
            .filter(|&i| cursors[i] < tracks[i].frames.len())
            .min_by_key(|&i| tracks[i].frames[cursors[i]].1.pts_ns);
        let Some(i) = next else { break };
        let (data, timing) = tracks[i].frames[cursors[i]].clone();
        cursors[i] += 1;
        m.process(i, frame(data, timing), &mut sink)
            .await
            .expect("mux");
    }
    for i in 0..tracks.len() {
        m.process(i, PipelinePacket::Eos, &mut sink)
            .await
            .expect("mux eos");
    }
    sink.bytes()
}

/// Demux an MP4 into per-port frames plus each port's refined caps.
async fn demux_mp4(file: &[u8]) -> (Vec<Caps>, PortCapture) {
    let streams = forwardable_streams(file);
    assert!(!streams.is_empty(), "the file has forwardable tracks");
    let ports: Vec<Mp4Port> = streams
        .iter()
        .map(|s| Mp4Port {
            track_id: s.track_id,
            caps: s.caps.clone(),
        })
        .collect();
    let mut d = Mp4DemuxN::new(ports);
    d.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .expect("configure mp4demux");
    let mut tap = PortCapture::default();
    tap.ports.resize(streams.len(), Vec::new());
    tap.caps.resize(streams.len(), None);
    d.process(frame(file.to_vec(), FrameTiming::default()), &mut tap)
        .await
        .expect("demux");
    d.process(PipelinePacket::Eos, &mut tap)
        .await
        .expect("demux eos");
    (streams.into_iter().map(|s| s.caps).collect(), tap)
}

/// Whether this ffmpeg build has `name` as an encoder, so a codec it cannot
/// author self-skips instead of failing.
fn has_encoder(name: &str) -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains(name))
        .unwrap_or(false)
}

/// ffmpeg-authored 1.0 s A/V MP4. B-frames are off, so these layout checks read
/// a straight timeline where presentation and decode order agree; the reordered
/// case is `m972_progressive_bframes`.
fn author_av(path: &PathBuf, vcodec: &str, acodec: &str) -> Option<Vec<u8>> {
    if !has_encoder(vcodec) || !has_encoder(acodec) {
        eprintln!("skipping: this ffmpeg has no {vcodec} / {acodec} encoder");
        return None;
    }
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i"])
        .arg(format!("testsrc2=size={WIDTH}x{HEIGHT}:rate=30:duration=1"))
        .args(["-f", "lavfi", "-i"])
        .arg("sine=frequency=440:duration=1:sample_rate=48000")
        .args([
            "-c:v", vcodec, "-pix_fmt", "yuv420p", "-bf", "0", "-g", "15",
        ])
        .args(["-c:a", acodec, "-ac", "2", "-ar", "48000"])
        .arg(path)
        .status()
        .expect("run ffmpeg");
    assert!(
        status.success(),
        "ffmpeg authored the {vcodec}+{acodec} A/V fixture"
    );
    Some(std::fs::read(path).expect("read fixture"))
}

/// The headline: a 1.0 s Opus source remuxed progressively reports exactly
/// 1.000000, because the sample table lets ffprobe apply the edit list's
/// pre-skip trim. The fragmented layout of the same frames reports the media
/// span instead, which is the M791 limit this milestone lifts.
#[tokio::test]
async fn progressive_opus_reports_the_exact_trimmed_duration() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("src.opus");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg("sine=frequency=440:duration=1.0:sample_rate=48000")
        .args(["-ac", "2", "-c:a", "libopus"])
        .arg(&src)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the Opus fixture");
    let ogg = std::fs::read(&src).expect("read fixture");

    // oggdemux gives the in-band OpusHead plus the audio packets, the last one
    // short by the granule trim (M791).
    let mut d = OggDemux::new();
    d.set_property("stream", PropValue::Str("opus".into()))
        .expect("stream property");
    d.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    })
    .expect("configure oggdemux");
    let mut demuxed = CaptureSink::default();
    for piece in ogg.chunks(1021) {
        d.process(frame(piece.to_vec(), FrameTiming::default()), &mut demuxed)
            .await
            .expect("demux");
    }
    d.process(PipelinePacket::Eos, &mut demuxed)
        .await
        .expect("demux eos");

    let caps = Caps::Audio {
        format: AudioFormat::Opus,
        channels: 2,
        sample_rate: 48_000,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    };
    let tracks = [MuxTrack {
        nego: caps,
        refined: None,
        frames: demuxed.frames.clone(),
    }];

    let progressive = mux(&tracks, false).await;
    let out = temp_path("out-progressive.mp4");
    std::fs::write(&out, &progressive).expect("write progressive");
    let probed = probe(&out, "a:0");
    println!("ffprobe progressive opus: {probed:?}");
    assert_eq!(field(&probed, "codec_name"), "opus");
    assert_eq!(field(&probed, "channels"), "2");
    assert_eq!(field(&probed, "sample_rate"), "48000");
    assert_eq!(
        field(&probed, "start_time"),
        "0.000000",
        "the edit list puts the first real sample at zero"
    );
    assert_eq!(
        field(&probed, "duration"),
        "1.000000",
        "the trimmed presentation duration, reported exactly"
    );
    assert_eq!(
        field(&probed, "format_duration"),
        "1.000000",
        "and the container agrees"
    );
    assert_eq!(
        field(&probed, "nb_frames"),
        "51",
        "every Opus packet is in the sample table"
    );

    // The fragmented layout of the same frames carries the same media but can
    // only report the untrimmed span (M791).
    let frag = temp_path("out-fragmented.mp4");
    std::fs::write(&frag, mux(&tracks, true).await).expect("write fragmented");
    let frag_probed = probe(&frag, "a:0");
    println!("ffprobe fragmented opus: {frag_probed:?}");
    assert_eq!(field(&frag_probed, "duration"), "1.006500");
    assert_eq!(field(&frag_probed, "start_time"), "-0.006500");

    // Same audio either way.
    assert_eq!(
        decode_audio(&out),
        decode_audio(&frag),
        "the two layouts decode to the same samples"
    );

    persist::record_evidence(
        "mp4mux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("opus")
            .detail("ffprobe reports the exact trimmed duration of a progressive g2g MP4"),
    )
    .expect("record oracle evidence");

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&frag);
}

/// Both layouts of a real A/V movie are fully readable: ffprobe sees the right
/// codecs and geometry, ffmpeg decodes both tracks without complaint, and the
/// media is identical to the fragmented path's. Run for H.264+AAC and
/// H.264+Opus.
async fn assert_av_layouts_agree(vcodec: &str, acodec: &str, probe_video: &str, probe_audio: &str) {
    let src = temp_path(&format!("src-av-{vcodec}-{acodec}.mp4"));
    let Some(file) = author_av(&src, vcodec, acodec) else {
        return;
    };
    let (nego, tap) = demux_mp4(&file).await;
    assert_eq!(nego.len(), 2, "a video and an audio track");

    let tracks: Vec<MuxTrack> = (0..nego.len())
        .map(|i| MuxTrack {
            nego: nego[i].clone(),
            refined: tap.caps[i].clone(),
            frames: tap.ports[i].clone(),
        })
        .collect();

    let prog_path = temp_path(&format!("out-av-{vcodec}-{acodec}-progressive.mp4"));
    let frag_path = temp_path(&format!("out-av-{vcodec}-{acodec}-fragmented.mp4"));
    let progressive = mux(&tracks, false).await;
    std::fs::write(&prog_path, &progressive).expect("write progressive");
    std::fs::write(&frag_path, mux(&tracks, true).await).expect("write fragmented");

    // Layout: one mdat, a moov, and no fragments at all.
    assert_eq!(
        progressive.windows(4).filter(|w| *w == b"moof").count(),
        0,
        "a progressive file carries no fragments"
    );

    let video = probe(&prog_path, "v:0");
    println!("ffprobe progressive {vcodec}+{acodec} video: {video:?}");
    assert_eq!(field(&video, "codec_name"), probe_video);
    assert_eq!(field(&video, "width"), WIDTH.to_string());
    assert_eq!(field(&video, "height"), HEIGHT.to_string());
    assert_eq!(
        field(&video, "nb_frames"),
        tap.ports[0].len().to_string(),
        "every video access unit is in the sample table"
    );
    let audio = probe(&prog_path, "a:0");
    println!("ffprobe progressive {vcodec}+{acodec} audio: {audio:?}");
    assert_eq!(field(&audio, "codec_name"), probe_audio);
    assert_eq!(field(&audio, "channels"), "2");
    assert_eq!(field(&audio, "sample_rate"), "48000");

    // Same media as the fragmented layout, decoded by ffmpeg both times.
    let prog_video = decode_video(&prog_path);
    assert_eq!(
        prog_video,
        decode_video(&frag_path),
        "{vcodec}: the two layouts decode to the same pictures"
    );
    assert_eq!(
        prog_video.len() / (WIDTH * HEIGHT * 3 / 2),
        tap.ports[0].len(),
        "{vcodec}: every muxed picture comes back"
    );
    assert_eq!(
        decode_audio(&prog_path),
        decode_audio(&frag_path),
        "{acodec}: the two layouts decode to the same samples"
    );

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&prog_path);
    let _ = std::fs::remove_file(&frag_path);
}

#[tokio::test]
async fn progressive_h264_aac_reads_and_decodes() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    assert_av_layouts_agree("libx264", "aac", "h264", "aac").await;
}

#[tokio::test]
async fn progressive_h264_opus_reads_and_decodes() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    assert_av_layouts_agree("libx264", "libopus", "h264", "opus").await;
}

/// The rest of the video matrix the muxer writes: the layout is codec-agnostic
/// (only the sample entry differs), so one pass each is enough to catch a codec
/// whose samples the progressive tables would misindex.
#[tokio::test]
async fn progressive_h265_and_av1_read_and_decode() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    assert_av_layouts_agree("libx265", "aac", "hevc", "aac").await;
    assert_av_layouts_agree("libsvtav1", "libopus", "av1", "opus").await;
}

/// g2g reads its own progressive file back: same tracks, same packets, byte for
/// byte, through the sample-table parser rather than the fragment parser.
#[tokio::test]
async fn g2g_demuxes_its_own_progressive_file_packet_exact() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("src-roundtrip.mp4");
    let Some(file) = author_av(&src, "libx264", "aac") else {
        return;
    };
    let (nego, tap) = demux_mp4(&file).await;
    let tracks: Vec<MuxTrack> = (0..nego.len())
        .map(|i| MuxTrack {
            nego: nego[i].clone(),
            refined: tap.caps[i].clone(),
            frames: tap.ports[i].clone(),
        })
        .collect();

    let progressive = mux(&tracks, false).await;
    let (again_nego, again) = demux_mp4(&progressive).await;
    assert_eq!(again_nego.len(), nego.len(), "same track count");
    for port in 0..nego.len() {
        let before: Vec<&Vec<u8>> = tap.ports[port].iter().map(|(b, _)| b).collect();
        let after: Vec<&Vec<u8>> = again.ports[port].iter().map(|(b, _)| b).collect();
        assert_eq!(
            after, before,
            "port {port}: packets survive the progressive remux byte for byte"
        );
    }

    persist::record_evidence(
        "mp4mux",
        &Evidence::new(ConformanceDimension::RoundTrip)
            .codec("h264+aac")
            .detail("mp4demux -> progressive mp4mux -> mp4demux is packet-exact"),
    )
    .expect("record round-trip evidence");

    let _ = std::fs::remove_file(&src);
}
