//! M790: grouped multi-stream Ogg, both directions. `OggDemuxN` splits a file's
//! logical bitstreams onto one output port each, `oggmuxn` writes several
//! streams back into one grouped file, and ffmpeg is the reference peer at both
//! ends: what g2g demuxes re-muxes to the source's samples bit for bit, and what
//! g2g muxes ffprobe reports (and ffmpeg decodes) as the streams that went in.
#![cfg(feature = "std")]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::parse_launch;
use g2g_core::{
    AsyncElement, AudioFormat, Bus, BusMessage, ByteStreamEncoding, Caps, G2gError,
    MultiInputElement, MultiOutputElement, MultiOutputSink, OutputSink, PushOutcome, StreamType,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::ogg::OggCodec;
use g2g_plugins::oggdemux::{OggDemuxN, OggPort};
use g2g_plugins::oggmux::OggMux;
use g2g_plugins::oggmuxn::OggMuxN;
use g2g_plugins::registry::default_registry;

/// Per-port capture of what a [`OggDemuxN`] emitted: the frame payloads with
/// their timing (the muxers read `duration_ns`) and the announced caps.
#[derive(Debug, Default)]
struct PortTap {
    frames: Vec<Vec<(Vec<u8>, FrameTiming)>>,
    caps: Vec<Vec<Caps>>,
}

impl PortTap {
    fn new(ports: usize) -> Self {
        Self {
            frames: (0..ports).map(|_| Vec::new()).collect(),
            caps: (0..ports).map(|_| Vec::new()).collect(),
        }
    }

    fn payloads(&self, port: usize) -> Vec<Vec<u8>> {
        self.frames[port].iter().map(|(b, _)| b.clone()).collect()
    }
}

impl MultiOutputSink for PortTap {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.frames[port].push((s.to_vec(), f.timing));
                    }
                }
                PipelinePacket::CapsChanged(c) => self.caps[port].push(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }

    fn port_count(&self) -> usize {
        self.frames.len()
    }
}

#[derive(Default)]
struct ByteSink {
    bytes: Vec<u8>,
}

impl OutputSink for ByteSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.bytes.extend_from_slice(s);
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn frame(data: Vec<u8>, timing: FrameTiming) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        timing,
        0,
    ))
}

fn ogg_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    }
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
        && Command::new("ffprobe").arg("-version").output().is_ok()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m790-{}-{name}", std::process::id()))
}

/// Author a two-stream Ogg with ffmpeg: two sine tones, one per codec.
fn author_pair(path: &PathBuf, codec0: &str, codec1: &str, rate: u32) -> Vec<u8> {
    let tone = |hz: u32| format!("sine=frequency={hz}:duration=1.0:sample_rate={rate}");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", &tone(440)])
        .args(["-f", "lavfi", "-i", &tone(880)])
        .args(["-map", "0:a", "-map", "1:a", "-ac", "2"])
        .args(["-c:a:0", codec0, "-c:a:1", codec1, "-f", "ogg"])
        .arg(path)
        .status()
        .expect("run ffmpeg");
    assert!(
        status.success(),
        "ffmpeg authored the {codec0}+{codec1} pair"
    );
    std::fs::read(path).expect("read fixture")
}

/// ffprobe's per-stream `(codec_name, channels, sample_rate)`, in stream order.
fn probe_streams(path: &PathBuf) -> Vec<(String, String, String)> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a"])
        .args([
            "-show_entries",
            "stream=codec_name,channels,sample_rate",
            "-of",
            "compact=p=0:nk=0",
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
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|line| {
            let fields: std::collections::BTreeMap<&str, &str> = line
                .trim()
                .split('|')
                .filter_map(|f| f.split_once('='))
                .collect();
            let get = |k: &str| {
                fields
                    .get(k)
                    .unwrap_or_else(|| panic!("ffprobe reported {k} in `{line}`"))
                    .to_string()
            };
            (get("codec_name"), get("channels"), get("sample_rate"))
        })
        .collect()
}

/// ffmpeg's decode of one audio stream of `path` as raw interleaved 16-bit PCM.
/// Fails the test if ffmpeg reports any error, so a container the peer cannot
/// read is caught here rather than silently comparing two empty buffers.
fn decode_stream(path: &PathBuf, index: usize) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-map", &format!("0:a:{index}")])
        .args(["-f", "s16le", "-c:a", "pcm_s16le", "-"])
        .output()
        .expect("run ffmpeg");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && err.is_empty(),
        "ffmpeg decoded stream {index} of {} cleanly: {err}",
        path.display()
    );
    assert!(!out.stdout.is_empty(), "ffmpeg decoded some audio");
    out.stdout
}

/// ffprobe's packet sizes for one audio stream, in stream order. Packaging
/// independent, unlike a decode: it names exactly the elementary-stream packets
/// the reference demuxer found.
fn probe_packet_sizes(path: &PathBuf, index: usize) -> Vec<usize> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", &format!("a:{index}")])
        .args(["-show_entries", "packet=size", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .expect("run ffprobe");
    assert!(out.status.success(), "ffprobe listed packets");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().trim_end_matches(',').parse().ok())
        .collect()
}

/// Every page of an Ogg byte stream as `(serial, header_type)`, in file order.
fn pages(data: &[u8]) -> Vec<(u32, u8)> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at + 27 <= data.len() {
        assert_eq!(&data[at..at + 4], b"OggS", "page at {at}");
        let n = data[at + 26] as usize;
        let body: usize = data[at + 27..at + 27 + n].iter().map(|&s| s as usize).sum();
        out.push((
            u32::from_le_bytes(data[at + 14..at + 18].try_into().unwrap()),
            data[at + 5],
        ));
        at += 27 + n + body;
    }
    out
}

/// Demux a grouped Ogg onto `ports` output ports, returning the tap and the bus.
async fn demux_n(ogg: &[u8], ports: Vec<OggPort>) -> (PortTap, Bus) {
    let (bus, handle) = Bus::new(32);
    let count = ports.len();
    let mut d = OggDemuxN::new(ports).with_bus(handle);
    d.configure_pipeline(&ogg_caps()).expect("configure");
    let mut tap = PortTap::new(count);
    for piece in ogg.chunks(1021) {
        d.process(frame(piece.to_vec(), FrameTiming::default()), &mut tap)
            .await
            .expect("demux");
    }
    d.process(PipelinePacket::Eos, &mut tap)
        .await
        .expect("demux eos");
    (tap, bus)
}

/// Mux one port's frames back into a single-stream Ogg file.
async fn remux_one(tap: &PortTap, port: usize, caps: &Caps) -> Vec<u8> {
    let mut m = OggMux::new();
    m.configure_pipeline(caps).expect("configure oggmux");
    let mut sink = ByteSink::default();
    for (data, timing) in &tap.frames[port] {
        m.process(frame(data.clone(), *timing), &mut sink)
            .await
            .expect("mux");
    }
    m.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    sink.bytes
}

/// Mux every port's frames back into one grouped Ogg file, interleaved by PTS.
async fn remux_grouped(tap: &PortTap, caps: &[Caps]) -> Vec<u8> {
    let mut m = OggMuxN::new(caps.len());
    for (input, c) in caps.iter().enumerate() {
        m.configure_pipeline(input, c).expect("configure oggmuxn");
    }
    let mut sink = ByteSink::default();
    // Feed each pad's frames in order; the muxer's aggregator interleaves them.
    let depth = tap.frames.iter().map(|f| f.len()).max().unwrap_or(0);
    for i in 0..depth {
        for (input, port) in tap.frames.iter().enumerate() {
            if let Some((data, timing)) = port.get(i) {
                m.process(input, frame(data.clone(), *timing), &mut sink)
                    .await
                    .expect("mux");
            }
        }
    }
    for input in 0..caps.len() {
        m.process(input, PipelinePacket::Eos, &mut sink)
            .await
            .expect("mux eos");
    }
    sink.bytes
}

/// The caps of each port, as announced by the demuxer.
fn announced(tap: &PortTap) -> Vec<Caps> {
    tap.caps
        .iter()
        .enumerate()
        .map(|(port, list)| {
            list.last()
                .unwrap_or_else(|| panic!("port {port} announced caps"))
                .clone()
        })
        .collect()
}

/// What one logical bitstream of a fixture should demux to: its parser codec,
/// the caps format the port announces, ffprobe's codec name, and how many
/// in-band codec-config frames precede its audio (1 `OpusHead`, the three Vorbis
/// headers, 1 native `fLaC` block).
#[derive(Debug, Clone, Copy)]
struct StreamExpectation {
    codec: OggCodec,
    format: AudioFormat,
    probe_name: &'static str,
    header_frames: usize,
}

/// Demux an ffmpeg-authored pair, check the announced streams, then prove both
/// directions against ffmpeg: each demuxed stream re-muxed alone, and all of
/// them re-muxed back into one grouped file, decode to the source's samples.
async fn assert_pair_round_trips(
    tag: &str,
    codec0: &str,
    codec1: &str,
    rate: u32,
    expect: [StreamExpectation; 2],
) {
    let src = temp_path(&format!("src-{tag}.ogg"));
    let bytes = author_pair(&src, codec0, codec1, rate);
    let source_pcm = [decode_stream(&src, 0), decode_stream(&src, 1)];

    let ports = Vec::from([
        OggPort::new(0, expect[0].codec),
        OggPort::new(1, expect[1].codec),
    ]);
    let (tap, bus) = demux_n(&bytes, ports).await;

    // Both logical bitstreams landed on their own port, with concrete caps.
    let caps = announced(&tap);
    for (port, want) in expect.iter().enumerate() {
        let Caps::Audio {
            format: got,
            sample_rate,
            channels,
        } = &caps[port]
        else {
            panic!("port {port} announced audio caps, got {:?}", caps[port]);
        };
        assert_eq!(*got, want.format, "{tag}: port {port} codec");
        assert_eq!(*sample_rate, rate, "{tag}: port {port} rate");
        assert_eq!(*channels, 2, "{tag}: port {port} channels");
        // Reference-peer demux oracle: the audio packets this port carried are
        // exactly the ones ffmpeg's own demuxer found in that stream.
        let sizes: Vec<usize> = tap.frames[port][want.header_frames..]
            .iter()
            .map(|(b, _)| b.len())
            .collect();
        assert_eq!(
            sizes,
            probe_packet_sizes(&src, port),
            "{tag}: port {port} packets match ffprobe's list"
        );
    }

    // The announced StreamCollection lists every logical bitstream.
    let mut collection = None;
    while let Some(m) = bus.try_recv() {
        if let BusMessage::StreamCollection(c) = m {
            collection = Some(c);
        }
    }
    let collection = collection.expect("a StreamCollection was posted");
    assert_eq!(collection.streams.len(), 2, "{tag}: both streams announced");
    for (i, stream) in collection.streams.iter().enumerate() {
        assert_eq!(stream.id, format!("ogg-stream-{i}"));
        assert_eq!(stream.stream_type, StreamType::Audio);
    }

    // Each stream also stands alone: re-muxed by itself it is a playable
    // single-stream file ffmpeg reads and decodes. Only the framing is asserted
    // here, because ffmpeg's Vorbis end-of-stream trim depends on how a stream is
    // packaged (its own single-stream copy of this fixture decodes to a different
    // length than the fixture does); the sample-exact comparison belongs on the
    // grouped file below, where the packaging matches.
    for port in 0..2 {
        let single = temp_path(&format!("one-{tag}-{port}.ogg"));
        std::fs::write(&single, remux_one(&tap, port, &caps[port]).await).expect("write");
        let probed = probe_streams(&single);
        println!("ffprobe {tag} stream {port} alone: {probed:?}");
        assert_eq!(probed.len(), 1);
        assert_eq!(
            probed[0].0, expect[port].probe_name,
            "{tag}: port {port} codec"
        );
        assert!(!decode_stream(&single, 0).is_empty());
        let _ = std::fs::remove_file(&single);
    }

    // Direction 2 (mux): all of them back into one grouped file.
    let grouped_bytes = remux_grouped(&tap, &caps).await;
    let pages = pages(&grouped_bytes);
    let bos: Vec<u32> = pages
        .iter()
        .filter(|(_, ht)| ht & 0x02 != 0)
        .map(|(s, _)| *s)
        .collect();
    assert_eq!(bos.len(), 2, "{tag}: one BOS page per stream");
    assert_ne!(bos[0], bos[1], "{tag}: distinct serials");
    assert_eq!(
        pages[..2].iter().map(|(s, _)| *s).collect::<Vec<_>>(),
        bos,
        "{tag}: RFC 3533 grouping, both BOS pages lead"
    );

    let grouped = temp_path(&format!("grouped-{tag}.ogg"));
    std::fs::write(&grouped, &grouped_bytes).expect("write");
    let probed = probe_streams(&grouped);
    println!("ffprobe {tag} grouped: {probed:?}");
    assert_eq!(probed.len(), 2, "{tag}: ffprobe found both streams");
    for (port, (codec, channels, sample_rate)) in probed.iter().enumerate() {
        assert_eq!(
            codec, expect[port].probe_name,
            "{tag}: muxed stream {port} codec"
        );
        assert_eq!(channels, "2", "{tag}: muxed stream {port} channels");
        assert_eq!(
            sample_rate,
            &rate.to_string(),
            "{tag}: muxed stream {port} rate"
        );
        assert_eq!(
            decode_stream(&grouped, port),
            source_pcm[port],
            "{tag}: ffmpeg decodes muxed stream {port} to the source's samples"
        );
    }

    // g2g reads its own grouped output back to the same packets.
    let (again, _) = demux_n(
        &grouped_bytes,
        Vec::from([
            OggPort::new(0, expect[0].codec),
            OggPort::new(1, expect[1].codec),
        ]),
    )
    .await;
    for port in 0..2 {
        assert_eq!(
            again.payloads(port),
            tap.payloads(port),
            "{tag}: stream {port} survives the grouped remux byte for byte"
        );
    }

    persist::record_evidence(
        "oggdemuxn",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec(expect[0].probe_name)
            .detail(
                "each stream of an ffmpeg-authored grouped Ogg demuxes to ffprobe's packet list",
            ),
    )
    .expect("record oracle evidence");
    persist::record_evidence(
        "oggmuxn",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec(expect[1].probe_name)
            .detail("ffmpeg decodes both streams of a g2g-muxed grouped Ogg"),
    )
    .expect("record oracle evidence");
    persist::record_evidence(
        "oggmuxn",
        &Evidence::new(ConformanceDimension::RoundTrip)
            .detail("oggdemuxn -> oggmuxn -> oggdemuxn is packet-exact on both streams"),
    )
    .expect("record round-trip evidence");

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&grouped);
}

/// Two different mappings in one file: Opus and Ogg-FLAC side by side.
#[tokio::test]
async fn mixed_codec_pair_round_trips_through_ffmpeg() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    assert_pair_round_trips(
        "mixed",
        "libopus",
        "flac",
        48_000,
        [
            StreamExpectation {
                codec: OggCodec::Opus,
                format: AudioFormat::Opus,
                probe_name: "opus",
                header_frames: 1,
            },
            StreamExpectation {
                codec: OggCodec::Flac,
                format: AudioFormat::Flac,
                probe_name: "flac",
                header_frames: 1,
            },
        ],
    )
    .await;
}

/// Two streams of the *same* codec, the case that makes routing positional
/// rather than codec-keyed.
#[tokio::test]
async fn same_codec_pair_round_trips_through_ffmpeg() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    assert_pair_round_trips(
        "vorbis",
        "libvorbis",
        "libvorbis",
        44_100,
        [
            StreamExpectation {
                codec: OggCodec::Vorbis,
                format: AudioFormat::Vorbis,
                probe_name: "vorbis",
                header_frames: 3,
            },
            StreamExpectation {
                codec: OggCodec::Vorbis,
                format: AudioFormat::Vorbis,
                probe_name: "vorbis",
                header_frames: 3,
            },
        ],
    )
    .await;
}

/// Per-stream VorbisComment metadata posts as `BusMessage::StreamTag` under the
/// ids the `StreamCollection` announced.
#[tokio::test]
async fn per_stream_tags_post_under_the_collection_ids() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let path = temp_path("tags.ogg");
    let tone = |hz: u32| format!("sine=frequency={hz}:duration=0.4:sample_rate=44100");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", &tone(440)])
        .args(["-f", "lavfi", "-i", &tone(880)])
        .args(["-map", "0:a", "-map", "1:a", "-c:a", "libvorbis"])
        .args(["-metadata:s:a:0", "title=First"])
        .args(["-metadata:s:a:1", "title=Second"])
        .args(["-f", "ogg"])
        .arg(&path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the tagged pair");
    let bytes = std::fs::read(&path).expect("read fixture");

    let (_, bus) = demux_n(
        &bytes,
        Vec::from([
            OggPort::new(0, OggCodec::Vorbis),
            OggPort::new(1, OggCodec::Vorbis),
        ]),
    )
    .await;
    let mut tagged: Vec<(String, String)> = Vec::new();
    while let Some(m) = bus.try_recv() {
        if let BusMessage::StreamTag { stream_id, tags } = m {
            for tag in tags.tags() {
                if let g2g_core::Tag::Title(t) = tag {
                    tagged.push((stream_id.clone(), t.to_string()));
                }
            }
        }
    }
    println!("per-stream tags: {tagged:?}");
    assert_eq!(
        tagged,
        Vec::from([
            ("ogg-stream-0".to_string(), "First".to_string()),
            ("ogg-stream-1".to_string(), "Second".to_string()),
        ]),
        "each stream's title posts under its collection id"
    );
    let _ = std::fs::remove_file(&path);
}

/// The launch fan-out: a named `oggdemux` with two output-pad refs probes the
/// file and builds the multi-output demuxer.
#[test]
fn oggdemux_fans_out_from_a_launch_line() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let path = temp_path("fanout.ogg");
    author_pair(&path, "libvorbis", "libvorbis", 44_100);
    let p = path.display();
    let line = format!(
        "filesrc location={p} ! oggdemux name=d  \
         d.audio_0 ! fakesink  d.audio_1 ! fakesink"
    );
    let reg = default_registry();
    assert!(
        parse_launch(&reg, &line).is_ok(),
        "a two-branch oggdemux fan-out parses: {line}"
    );
    let _ = std::fs::remove_file(&path);
}

/// The `oggmux` name covers both shapes: one input builds the single-stream
/// element, several build the fan-in muxer.
#[test]
fn oggmux_resolves_as_both_element_and_fan_in_muxer() {
    let reg = default_registry();
    assert!(
        reg.make_muxer("oggmux", 2).is_some(),
        "oggmux resolves as a 2-input fan-in muxer"
    );
    assert!(
        parse_launch(
            &reg,
            "filesrc location=a.ogg ! oggdemux ! oggmux ! fakesink"
        )
        .is_ok(),
        "and still as the single-input element"
    );
}
