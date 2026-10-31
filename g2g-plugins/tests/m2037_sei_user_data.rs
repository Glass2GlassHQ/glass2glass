//! M2037: `h264seiinserter` and `h265seiinserter` write each frame's registered
//! and unregistered SEI user data into an ffmpeg-encoded stream, and ffmpeg's
//! `trace_headers` reads every message back byte for byte.
#![cfg(all(feature = "std", feature = "metadata"))]

#[path = "common/ffmpeg_tool.rs"]
mod ffmpeg_tool;
use ffmpeg_tool::have_ffmpeg;

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

use std::path::PathBuf;
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::meta::{SeiUserData, SeiUserDataMeta};
use g2g_core::{AsyncElement, Caps, Dim, Rate, VideoCodec};
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::h265parse::H265Parse;
use g2g_plugins::registry::default_registry;
use g2g_plugins::seiinsert::{H264SeiInserter, H265SeiInserter};

const FRAMES: usize = 5;
const UUID: [u8; 16] = *b"g2g-m2037-sei-id";
// United States, then the ATSC provider code
const T35_PREFIX: [u8; 3] = [0xB5, 0x00, 0x31];

#[derive(Debug, PartialEq, Eq)]
enum Traced {
    Registered(Vec<u8>),
    Unregistered(Vec<u8>),
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m2037-{}-{name}", std::process::id()))
}

fn have_encoder(name: &str) -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains(name))
}

fn encode(encoder: &str, format: &str, out: &PathBuf) -> Vec<u8> {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg("testsrc=size=64x64:rate=10")
        .args([
            "-frames:v",
            &FRAMES.to_string(),
            "-c:v",
            encoder,
            "-f",
            format,
        ])
        .arg(out)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg encoded with {encoder}");
    std::fs::read(out).expect("read the encoded stream")
}

fn frame_messages(index: usize) -> Vec<SeiUserData> {
    let mut t35 = T35_PREFIX.to_vec();
    t35.push(index as u8);
    Vec::from([
        SeiUserData::Registered { t35 },
        // the zero run needs emulation prevention inside the NAL
        SeiUserData::Unregistered {
            uuid: UUID,
            payload: Vec::from([0, 0, 0, index as u8]),
        },
    ])
}

fn expected_trace() -> Vec<Traced> {
    (0..FRAMES)
        .flat_map(frame_messages)
        .map(|message| match message {
            SeiUserData::Registered { t35 } => Traced::Registered(t35),
            SeiUserData::Unregistered { uuid, payload } => {
                Traced::Unregistered([uuid.as_slice(), &payload].concat())
            }
        })
        .collect()
}

fn system_frame(bytes: &[u8]) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    )
}

fn data_frames(sink: CollectSink) -> Vec<Frame> {
    sink.packets
        .into_iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
        .collect()
}

async fn split_and_insert(
    mut parser: impl AsyncElement,
    mut inserter: impl AsyncElement,
    codec: VideoCodec,
    stream: &[u8],
) -> Vec<u8> {
    let caps = Caps::CompressedVideo {
        codec,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    };
    parser.configure_pipeline(&caps).expect("parser configures");
    inserter
        .configure_pipeline(&caps)
        .expect("inserter configures");
    let mut parsed = CollectSink::default();
    parser
        .process(PipelinePacket::DataFrame(system_frame(stream)), &mut parsed)
        .await
        .expect("parse");
    parser
        .process(PipelinePacket::Eos, &mut parsed)
        .await
        .expect("parse eos");
    let access_units = data_frames(parsed);
    assert_eq!(access_units.len(), FRAMES, "{codec:?} access units");

    let mut inserted = CollectSink::default();
    for (index, mut frame) in access_units.into_iter().enumerate() {
        frame.meta.attach(SeiUserDataMeta {
            messages: frame_messages(index),
        });
        inserter
            .process(PipelinePacket::DataFrame(frame), &mut inserted)
            .await
            .expect("insert");
    }
    data_frames(inserted)
        .iter()
        .flat_map(|frame| {
            frame
                .domain
                .as_system_slice()
                .expect("system frame")
                .to_vec()
        })
        .collect()
}

// every user data message ffmpeg's `trace_headers` reports, in stream order
fn trace_user_data(path: &PathBuf) -> Vec<Traced> {
    let output = Command::new("ffmpeg")
        .args(["-loglevel", "debug", "-i"])
        .arg(path)
        .args(["-c", "copy", "-bsf:v", "trace_headers", "-f", "null", "-"])
        .output()
        .expect("run ffmpeg trace_headers");
    let log = String::from_utf8_lossy(&output.stderr);
    let mut messages = Vec::new();
    let mut open: Option<Traced> = None;
    for line in log.lines().filter(|line| line.contains("[trace_headers")) {
        if line.ends_with("User Data Registered ITU-T T.35") {
            messages.extend(open.replace(Traced::Registered(Vec::new())));
            continue;
        }
        if line.ends_with("User Data Unregistered") {
            messages.extend(open.replace(Traced::Unregistered(Vec::new())));
            continue;
        }
        let byte_field = [
            "itu_t_t35_",
            "uuid_iso_iec_11578[",
            "user_data_payload_byte[",
        ]
        .iter()
        .any(|field| line.contains(field));
        match (&mut open, byte_field) {
            (Some(Traced::Registered(bytes) | Traced::Unregistered(bytes)), true) => {
                let value = line.rsplit("= ").next().expect("a traced value");
                bytes.push(value.trim().parse().expect("a byte value"));
            }
            (Some(_), false) => messages.extend(open.take()),
            (None, _) => {}
        }
    }
    messages.extend(open);
    messages
}

fn ours(traced: Vec<Traced>) -> Vec<Traced> {
    traced
        .into_iter()
        .filter(|message| match message {
            Traced::Registered(bytes) => bytes.starts_with(&T35_PREFIX),
            Traced::Unregistered(bytes) => bytes.starts_with(&UUID),
        })
        .collect()
}

#[tokio::test]
async fn ffmpeg_reads_back_the_inserted_user_data() {
    if !have_ffmpeg() {
        eprintln!("skipping m2037: no ffmpeg on PATH");
        return;
    }
    for (encoder, format, codec) in [
        ("libx264", "h264", VideoCodec::H264),
        ("libx265", "hevc", VideoCodec::H265),
    ] {
        if !have_encoder(encoder) {
            eprintln!("skipping m2037 {codec:?}: ffmpeg has no {encoder}");
            continue;
        }
        let source = temp_path(&format!("in.{format}"));
        let written = temp_path(&format!("out.{format}"));
        let stream = encode(encoder, format, &source);
        let bytes = match codec {
            VideoCodec::H265 => {
                split_and_insert(
                    H265Parse::reframing(),
                    H265SeiInserter::new(),
                    codec,
                    &stream,
                )
                .await
            }
            _ => {
                split_and_insert(
                    H264Parse::reframing(),
                    H264SeiInserter::new(),
                    codec,
                    &stream,
                )
                .await
            }
        };
        std::fs::write(&written, bytes).expect("write the inserted stream");
        let traced = trace_user_data(&written);
        let _ = std::fs::remove_file(&source);
        let _ = std::fs::remove_file(&written);
        assert_eq!(ours(traced), expected_trace(), "{codec:?}");
    }
}

#[test]
fn both_inserters_build_from_the_registry() {
    let registry = default_registry();
    for name in ["h264seiinserter", "h265seiinserter"] {
        assert!(registry.make_element(name).is_some(), "`{name}` builds");
    }
}
