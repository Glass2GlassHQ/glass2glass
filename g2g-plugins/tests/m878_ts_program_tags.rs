//! M878 - per-program service text on MPEG-TS. A multi-program multiplex carries
//! one SDT entry per program, each with its own `service_name` /
//! `service_provider_name`: `TsMux::with_program_tags` names a program by its
//! `prog-map` number, and `with_tags` still names whichever programs do not. The
//! demuxers post each service as a `BusMessage::Tag` scoped to its
//! `program_number`, so an application can tell the services apart.
//!
//! Four legs: a g2g round trip, the default-service fallback, ffprobe reading the
//! per-program text out of a g2g-muxed stream (the oracle), and g2g demuxing a
//! two-program stream ffmpeg authored with distinct service names. The ffmpeg legs
//! self-skip when the binary is absent.
#![cfg(feature = "std")]

use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::fanout::MultiOutputSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, Bus, BusMessage, ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement,
    MultiOutputElement, OutputSink, PushOutcome, Rate, Tag, TagList, VideoCodec,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::tsdemux::{TsDemuxN, TsStream};
use g2g_plugins::tsmuxn::TsMux;

/// The stream id the mux's PID layout gives program 2's video (inputs 0 and 1,
/// program 1's video + audio, take 256 and 257).
const P2_VIDEO_ID: &str = "mpegts-pid-258";

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn aac_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 2,
        sample_rate: 48_000,
    }
}

#[derive(Default)]
struct Collect {
    bytes: Vec<u8>,
}
impl OutputSink for Collect {
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

/// Counts what each output port received: the frames only prove the demux ran.
#[derive(Default)]
struct PortTap {
    frames: Vec<usize>,
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
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                self.frames[port] += 1;
            }
            Ok(PushOutcome::Accepted)
        })
    }
    fn port_count(&self) -> usize {
        self.frames.len()
    }
}

fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}

fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}

/// A minimal ADTS AAC access unit (7-byte header + payload) at 48 kHz stereo.
fn adts_au(payload: &[u8]) -> Vec<u8> {
    let frame_len = payload.len() + 7;
    let mut au = vec![
        0xFF,
        0xF1,
        (1 << 6) | (3 << 2),
        ((2 & 3) << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F,
        0xFC,
    ];
    au.extend_from_slice(payload);
    au
}

fn tags(list: &[Tag]) -> TagList {
    list.iter().cloned().collect()
}

/// Program 1's own service text: the name under [`Tag::Title`], the provider under
/// ffprobe's `service_provider` key.
fn program1_tags() -> TagList {
    tags(&[
        Tag::Title("G2G News".into()),
        Tag::Other {
            key: "service_provider".into(),
            value: "G2G Broadcasting".into(),
        },
    ])
}

/// Program 2's own service text, plus a program-scoped language: its streams take
/// `spa` without naming it themselves.
fn program2_tags() -> TagList {
    tags(&[
        Tag::Title("G2G Sports".into()),
        Tag::Other {
            key: "service_provider".into(),
            value: "G2G Sports Network".into(),
        },
        Tag::Language("spa".into()),
    ])
}

/// The service text of a program that names none of its own.
fn fallback_tags() -> TagList {
    tags(&[
        Tag::Title("G2G Network".into()),
        Tag::Other {
            key: "service_provider".into(),
            value: "G2G Broadcasting".into(),
        },
    ])
}

/// Mux three elementary streams into a two-program transport stream: program 1
/// carries H.264 + AAC, program 2 a second H.264 stream. `whole` is the
/// muxer-wide service (`with_tags`), `per_program` each program's own.
async fn mux_two_programs(whole: Option<TagList>, per_program: &[(u16, TagList)]) -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    let mut mux = TsMux::new(3).with_program_numbers(&[1, 1, 2]);
    if let Some(whole) = whole {
        mux = mux.with_tags(whole);
    }
    for (program, tags) in per_program {
        mux = mux.with_program_tags(*program, tags.clone());
    }
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    mux.configure_pipeline(2, &h264_caps()).unwrap();

    let mut sink = Collect::default();
    for i in 0..4u64 {
        let base = i * 33_000_000;
        mux.process(0, frame(annexb(&[&sps, &pps, &idr]), base), &mut sink)
            .await
            .unwrap();
        mux.process(
            1,
            frame(adts_au(&[0xA1, 0xA2, 0xA3]), base + 10_000_000),
            &mut sink,
        )
        .await
        .unwrap();
        mux.process(
            2,
            frame(annexb(&[&sps, &pps, &idr]), base + 20_000_000),
            &mut sink,
        )
        .await
        .unwrap();
    }
    for input in 0..3 {
        mux.process(input, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
    }
    sink.bytes
}

fn data_frame(bytes: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

/// Demux `ts` with a `TsDemuxN` selecting `program`, one port per entry of
/// `ports`, and return `(service tags with their program scope, per-stream tags)`.
#[allow(clippy::type_complexity)]
async fn demux_bus_messages(
    ts: &[u8],
    program: u16,
    ports: &[TsStream],
) -> (Vec<(Option<u16>, TagList)>, Vec<(String, TagList)>) {
    let (bus, handle) = Bus::new(64);
    let mut demux = TsDemuxN::new(ports.to_vec())
        .with_program_number(Some(program))
        .with_bus(handle);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        })
        .expect("configure");
    let mut tap = PortTap {
        frames: vec![0; ports.len()],
    };
    demux.process(data_frame(ts), &mut tap).await.unwrap();
    demux.process(PipelinePacket::Eos, &mut tap).await.unwrap();
    assert!(
        tap.frames.iter().all(|&n| n > 0),
        "program {program}'s streams demuxed: {:?}",
        tap.frames
    );

    let (mut services, mut per_stream) = (Vec::new(), Vec::new());
    while let Some(msg) = bus.try_recv() {
        match msg {
            BusMessage::Tag { tags, program } => services.push((program, tags)),
            BusMessage::StreamTag { stream_id, tags } => per_stream.push((stream_id, tags)),
            _ => {}
        }
    }
    (services, per_stream)
}

/// The two programs' service text survives the round trip on its own scope, and a
/// program-scoped language reaches that program's streams.
#[tokio::test]
async fn each_program_carries_its_own_service_text() {
    let ts = mux_two_programs(None, &[(1, program1_tags()), (2, program2_tags())]).await;

    // The SDT describes the whole multiplex, so either program's demuxer reports
    // both services, each on its own scope.
    let expected = vec![
        (Some(1), program1_tags()),
        (
            Some(2),
            tags(&[
                Tag::Title("G2G Sports".into()),
                Tag::Other {
                    key: "service_provider".into(),
                    value: "G2G Sports Network".into(),
                },
            ]),
        ),
    ];
    let (services, per_stream) = demux_bus_messages(&ts, 1, &[TsStream::H264, TsStream::Aac]).await;
    assert_eq!(
        services, expected,
        "one scoped Tag per program, with that program's text"
    );
    assert!(
        per_stream.is_empty(),
        "program 1 named no language: {per_stream:?}"
    );

    let (services, per_stream) = demux_bus_messages(&ts, 2, &[TsStream::H264]).await;
    assert_eq!(services, expected, "the same two services from program 2");
    assert_eq!(
        per_stream,
        vec![(
            P2_VIDEO_ID.to_string(),
            tags(&[Tag::Language("spa".into())])
        )],
        "the program-scoped language lands on that program's stream"
    );
}

/// A program with no service of its own is named by the whole-mux `with_tags`
/// text, the single-service surface a one-program mux uses.
#[tokio::test]
async fn the_whole_mux_service_names_a_program_without_its_own() {
    let ts = mux_two_programs(Some(fallback_tags()), &[(2, program1_tags())]).await;
    let (services, _) = demux_bus_messages(&ts, 1, &[TsStream::H264, TsStream::Aac]).await;
    assert_eq!(
        services,
        vec![(Some(1), fallback_tags()), (Some(2), program1_tags())],
        "program 1 falls back to the whole-mux service, program 2 keeps its own"
    );
}

/// A service for a program no input is in would never reach the SDT, so the
/// builder refuses it instead of dropping it.
#[test]
#[should_panic(expected = "no input in program 7")]
fn a_service_for_an_absent_program_is_refused() {
    let _ = TsMux::new(2)
        .with_program_numbers(&[1, 1])
        .with_program_tags(7, program1_tags());
}

/// The ffprobe oracle: the reference implementation must read the per-program SDT
/// entries g2g writes, reporting each program's own service name and provider, and
/// the program-scoped language on that program's stream. Records peer-tagged
/// `Oracle` evidence for `mpegtsmux` on success (the M619 pattern).
#[tokio::test]
async fn ffprobe_reports_a_distinct_service_per_program() {
    if Command::new("ffprobe").arg("-version").output().is_err() {
        eprintln!("ffprobe not present; skipping the per-program TS tag oracle");
        return;
    }
    // Dedicated freshly-truncated log standalone; append to a shared CI log when
    // $G2G_CONFORMANCE_LOG is already set.
    let external = std::env::var_os("G2G_CONFORMANCE_LOG");
    let log = match &external {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let l = std::env::temp_dir().join("g2g-conformance-m878.tsv");
            std::env::set_var("G2G_CONFORMANCE_LOG", &l);
            let _ = std::fs::remove_file(&l);
            l
        }
    };
    let ts = mux_two_programs(None, &[(1, program1_tags()), (2, program2_tags())]).await;
    let path = std::env::temp_dir().join("g2g-m878-programs.ts");
    std::fs::write(&path, &ts).expect("write ts");

    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_programs",
            "-show_entries",
            "program=program_num:program_tags:program_stream=index:program_stream_tags=language",
            "-of",
            "compact",
        ])
        .arg(&path)
        .output()
        .expect("run ffprobe");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success(), "ffprobe read the stream: {text}");
    let programs: Vec<&str> = text
        .lines()
        .filter(|l| l.contains("program_num="))
        .collect();
    assert_eq!(programs.len(), 2, "both programs reported: {text}");
    assert!(
        programs[0].contains("program_num=1")
            && programs[0].contains("tag:service_name=G2G News")
            && programs[0].contains("tag:service_provider=G2G Broadcasting"),
        "program 1's own service text: {text}"
    );
    assert!(
        programs[1].contains("program_num=2")
            && programs[1].contains("tag:service_name=G2G Sports")
            && programs[1].contains("tag:service_provider=G2G Sports Network"),
        "program 2's own service text, not program 1's: {text}"
    );
    assert!(
        !programs[0].contains("language=spa") && programs[1].contains("language=spa"),
        "the program-scoped language rides only program 2's stream: {text}"
    );

    persist::record_evidence(
        "mpegtsmux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .detail("ffprobe reports a distinct SDT service per program of a multi-program TS"),
    )
    .expect("record oracle evidence");
    let report = persist::full_report();
    assert!(
        report.records.iter().any(|r| r.element == "mpegtsmux"),
        "mpegtsmux present after persisting evidence"
    );

    if external.is_none() {
        let _ = std::fs::remove_file(&log);
    }
    let _ = std::fs::remove_file(&path);
}

/// The reference-peer direction: ffmpeg authors a two-program transport stream
/// whose programs carry distinct service text (`-program` + `-metadata:p:N`), and
/// g2g's demuxer must post each service on its own program scope.
#[tokio::test]
async fn demuxes_a_distinct_service_per_program_from_an_ffmpeg_authored_ts() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ffmpeg not present; skipping the reference-peer multi-program TS demux");
        return;
    }
    let path = std::env::temp_dir().join("g2g-m878-ffmpeg.ts");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=176x144:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=176x144:rate=10:duration=1",
            "-map",
            "0:v",
            "-map",
            "1:v",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-program",
            "program_num=1:st=0",
            "-program",
            "program_num=2:st=1",
            "-metadata:p:0",
            "service_name=Peer News",
            "-metadata:p:0",
            "service_provider=Peer Broadcasting",
            "-metadata:p:1",
            "service_name=Peer Sports",
            "-metadata:p:1",
            "service_provider=Peer Sports Network",
            "-f",
            "mpegts",
        ])
        .arg(&path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the two-program TS");
    let ts = std::fs::read(&path).expect("read reference ts");

    let expected = vec![
        (
            Some(1),
            tags(&[
                Tag::Title("Peer News".into()),
                Tag::Other {
                    key: "service_provider".into(),
                    value: "Peer Broadcasting".into(),
                },
            ]),
        ),
        (
            Some(2),
            tags(&[
                Tag::Title("Peer Sports".into()),
                Tag::Other {
                    key: "service_provider".into(),
                    value: "Peer Sports Network".into(),
                },
            ]),
        ),
    ];
    for program in [1u16, 2] {
        let (services, _) = demux_bus_messages(&ts, program, &[TsStream::H264]).await;
        assert_eq!(
            services, expected,
            "the peer's two services, each on its own program scope"
        );
    }
    let _ = std::fs::remove_file(&path);
}
