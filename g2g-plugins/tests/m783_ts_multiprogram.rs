//! M783 - multi-program MPEG-TS muxing. Three elementary streams are muxed into
//! one transport stream carrying two programs (`prog-map=1,1,2`): program 1 with
//! H.264 + AAC, program 2 with a second H.264 stream. Each program is recovered
//! by demuxing with `program-number`, and `ffprobe` validates the PSI layout.
#![cfg(feature = "std")]

use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, Dim, FrameTiming, G2gError, MemoryDomain,
    MultiInputElement, OutputSink, PipelinePacket, PropValue, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::tsdemux::{TsDemux, TsStream};
use g2g_plugins::tsmuxn::TsMux;

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}
fn aac_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 2,
        sample_rate: 48_000,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}
fn ts_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    }
}

/// Collects everything pushed downstream: the raw bytes and each frame on its own.
#[derive(Default)]
struct CaptureSink {
    bytes: Vec<u8>,
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
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.bytes.extend_from_slice(s);
                    self.frames.push(s.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
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
    let sr_index = 3u8; // 48000
    let channels = 2u8;
    let mut au = vec![
        0xFF,
        0xF1,
        (1 << 6) | (sr_index << 2) | ((channels >> 2) & 1),
        ((channels & 3) << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F,
        0xFC,
    ];
    au.extend_from_slice(payload);
    au
}

/// The access units of one elementary stream, in order.
type Aus = Vec<Vec<u8>>;

/// The access units of each input: program 1 video, program 1 audio, program 2
/// video. The video streams open with SPS + PPS + IDR so ffprobe can identify
/// them.
fn scripted_aus() -> (Aus, Aus, Aus) {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    let video = |tail: u8| annexb(&[&[0x41u8, 0x9a, tail]]);
    (
        vec![annexb(&[&sps, &pps, &idr]), video(0x00), video(0x01)],
        vec![adts_au(&[0xA1, 0xA2, 0xA3]), adts_au(&[0xB4, 0xB5])],
        vec![annexb(&[&sps, &pps, &idr]), video(0x10), video(0x11)],
    )
}

/// Mux the scripted streams into one two-program transport stream, driving the
/// element directly (inputs 0 and 1 in program 1, input 2 in program 2).
async fn mux_two_programs() -> Vec<u8> {
    let (video1, audio1, video2) = scripted_aus();
    let mut mux = TsMux::new(3);
    mux.set_property("prog-map", PropValue::Str("1,1,2".into()))
        .expect("prog-map accepted");
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    mux.configure_pipeline(2, &h264_caps()).unwrap();

    // Round-robin by PTS so the aggregator can release AUs as they arrive.
    let mut sink = CaptureSink::default();
    for i in 0..3usize {
        let base = i as u64 * 33_000_000;
        for (input, aus, offset) in [
            (0usize, &video1, 0u64),
            (1, &audio1, 10_000_000),
            (2, &video2, 20_000_000),
        ] {
            if let Some(au) = aus.get(i) {
                mux.process(input, frame(au.clone(), base + offset), &mut sink)
                    .await
                    .unwrap();
            }
        }
    }
    for input in 0..3 {
        mux.process(input, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
    }
    assert_eq!(mux.emitted(), 8, "all eight AUs muxed (3 + 2 + 3)");
    sink.bytes
}

/// Drive a whole TS byte buffer through a `TsDemux` selecting `stream` within
/// `program`, returning the access units it recovers.
async fn demux(ts: &[u8], program: u16, stream: TsStream) -> Vec<Vec<u8>> {
    let mut demux = TsDemux::new().with_stream(stream);
    demux
        .set_property("program-number", PropValue::Int(program as i64))
        .expect("program-number accepted");
    demux.configure_pipeline(&ts_caps()).unwrap();
    let mut sink = CaptureSink::default();
    let f = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(ts.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    demux
        .process(PipelinePacket::DataFrame(f), &mut sink)
        .await
        .unwrap();
    demux.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    sink.frames
}

#[tokio::test]
async fn prog_map_splits_the_inputs_into_two_recoverable_programs() {
    let (video1, audio1, video2) = scripted_aus();
    let ts = mux_two_programs().await;
    assert_eq!(ts[0], 0x47, "TS sync byte");
    assert_eq!(ts.len() % 188, 0, "whole TS packets");

    assert_eq!(
        demux(&ts, 1, TsStream::H264).await,
        video1,
        "program 1 video recovered"
    );
    assert_eq!(
        demux(&ts, 1, TsStream::Aac).await,
        audio1,
        "program 1 audio recovered"
    );
    assert_eq!(
        demux(&ts, 2, TsStream::H264).await,
        video2,
        "program 2 video recovered"
    );
    // Program 2 has no audio: selecting it recovers nothing.
    assert!(
        demux(&ts, 2, TsStream::Aac).await.is_empty(),
        "program 2 carries no audio stream"
    );
}

#[test]
fn prog_map_round_trips_and_rejects_a_bad_map() {
    let mut mux = TsMux::new(3);
    assert_eq!(
        mux.get_property("prog-map"),
        Some(PropValue::Str("1,1,1".into())),
        "every input is in program 1 by default"
    );
    mux.set_property("prog-map", PropValue::Str(" 1, 1 ,2".into()))
        .unwrap();
    assert_eq!(
        mux.get_property("prog-map"),
        Some(PropValue::Str("1,1,2".into())),
        "the canonical value is the joined list"
    );
    // One entry per input, and each must parse as a program number.
    assert!(mux
        .set_property("prog-map", PropValue::Str("1,2".into()))
        .is_err());
    assert!(mux
        .set_property("prog-map", PropValue::Str("1,x,2".into()))
        .is_err());
    assert_eq!(
        mux.get_property("prog-map"),
        Some(PropValue::Str("1,1,2".into())),
        "a rejected value leaves the map untouched"
    );
}

/// The `(program_number, [codec_name])` pairs ffprobe reports for a TS file. Each
/// `[PROGRAM]` block lists its `program_num` then a `[STREAM]` block per
/// elementary stream; the flat stream list ffprobe prints after the last program
/// is not part of any block and is skipped.
fn ffprobe_programs(path: &std::path::Path) -> Vec<(u32, Vec<String>)> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_programs",
            "-show_entries",
            "program=program_num:stream=codec_name",
            "-of",
            "default=noprint_wrappers=0",
        ])
        .arg(path)
        .output()
        .expect("run ffprobe");
    assert!(
        out.status.success(),
        "ffprobe accepted the native multi-program TS: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let mut programs: Vec<(u32, Vec<String>)> = Vec::new();
    let mut in_program = false;
    for line in text.lines() {
        match line.trim() {
            "[PROGRAM]" => in_program = true,
            "[/PROGRAM]" => in_program = false,
            l if l.starts_with("program_num=") => programs.push((
                l["program_num=".len()..].parse().expect("program number"),
                Vec::new(),
            )),
            l if in_program && l.starts_with("codec_name=") => {
                if let Some(p) = programs.last_mut() {
                    p.1.push(l["codec_name=".len()..].to_string());
                }
            }
            _ => {}
        }
    }
    programs
}

#[tokio::test]
async fn ffprobe_reads_both_programs_and_records_interop_evidence() {
    if Command::new("ffprobe").arg("-version").output().is_err() {
        eprintln!("ffprobe not present; skipping the multi-program TS oracle");
        return;
    }

    // Dedicated freshly-truncated log standalone; append to a shared CI log when
    // $G2G_CONFORMANCE_LOG is already set (assertions search by element name).
    let external = std::env::var_os("G2G_CONFORMANCE_LOG");
    let log = match &external {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let l = std::env::temp_dir().join("g2g-conformance-m783.tsv");
            std::env::set_var("G2G_CONFORMANCE_LOG", &l);
            let _ = std::fs::remove_file(&l);
            l
        }
    };

    let ts = std::env::temp_dir().join("g2g-conformance-m783.ts");
    std::fs::write(&ts, mux_two_programs().await).expect("write ts");
    let programs = ffprobe_programs(&ts);
    assert_eq!(
        programs,
        vec![
            (1, vec!["h264".to_string(), "aac".to_string()]),
            (2, vec!["h264".to_string()]),
        ],
        "ffprobe reads both programs with their own streams"
    );

    persist::record_evidence(
        "mpegtsmux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264")
            .detail("ffprobe reads both programs of the native multi-program MPEG-TS"),
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
    let _ = std::fs::remove_file(&ts);
}
