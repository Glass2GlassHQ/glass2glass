//! M767: Opus in MP4 demuxes. The muxer already wrote the `Opus` sample entry
//! and `dOps`; now the demux side recognizes it, so an Opus track surfaces as
//! `Caps::Audio { Opus }` with the raw packets forwarded verbatim (no ADTS
//! framing, Opus needs no out-of-band config). Round-trips our own muxer and
//! demuxes an ffmpeg-authored file, comparing per-packet sizes against
//! ffprobe's sample table read.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, ByteStreamEncoding, Caps, G2gError, MultiInputElement, MultiOutputElement,
    MultiOutputSink, OutputSink, PushOutcome,
};
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};
use g2g_plugins::mp4muxn::Mp4MuxN;

fn opus_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: 2,
        sample_rate: 48000,
    }
}

#[derive(Default)]
struct CaptureSink {
    bytes: Vec<u8>,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.bytes.extend_from_slice(s);
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[derive(Default)]
struct PortCapture {
    frames: Vec<Vec<u8>>,
    caps: Vec<Caps>,
}
impl MultiOutputSink for PortCapture {
    fn push_to<'a>(
        &'a mut self,
        _port: usize,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.frames.push(s.to_vec());
                    }
                }
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }

    fn port_count(&self) -> usize {
        1
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

/// Demux every audio packet of `file` through `Mp4DemuxN` and return
/// (announced caps, packets).
async fn demux_opus(file: &[u8]) -> (Vec<Caps>, Vec<Vec<u8>>) {
    let streams = forwardable_streams(file);
    assert_eq!(streams.len(), 1, "one Opus track discovered");
    assert!(
        matches!(
            streams[0].caps,
            Caps::Audio {
                format: AudioFormat::Opus,
                ..
            }
        ),
        "discovered as Opus, got {:?}",
        streams[0].caps
    );
    let ports = vec![Mp4Port {
        track_id: streams[0].track_id,
        caps: streams[0].caps.clone(),
    }];
    let mut demux = Mp4DemuxN::new(ports);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        })
        .expect("configure");
    let mut tap = PortCapture::default();
    demux
        .process(frame(file.to_vec(), 0), &mut tap)
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut tap)
        .await
        .expect("eos");
    (tap.caps, tap.frames)
}

#[tokio::test]
async fn opus_round_trips_through_mp4() {
    // Three fake Opus packets (a real TOC byte, then arbitrary payload); the
    // demux must hand them back verbatim.
    let packets: Vec<Vec<u8>> = vec![
        vec![0xFC, 1, 2, 3, 4],
        vec![0xFC, 5, 6, 7],
        vec![0xFC, 8, 9, 10, 11, 12],
    ];

    let mut mux = Mp4MuxN::new(1);
    mux.configure_pipeline(0, &opus_caps()).unwrap();
    let mut sink = CaptureSink::default();
    for (i, p) in packets.iter().enumerate() {
        mux.process(0, frame(p.clone(), i as u64 * 20_000_000), &mut sink)
            .await
            .unwrap();
    }
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    let file = sink.bytes;
    assert!(!file.is_empty(), "muxer produced a file");

    let (caps, frames) = demux_opus(&file).await;
    assert!(
        caps.iter().any(|c| matches!(
            c,
            Caps::Audio {
                format: AudioFormat::Opus,
                channels: 2,
                sample_rate: 48000,
            }
        )),
        "concrete Opus caps announced, got {caps:?}"
    );
    assert_eq!(frames, packets, "packets recovered verbatim");
}

#[tokio::test]
async fn ffmpeg_authored_opus_mp4_demuxes() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }
    let dir = std::env::temp_dir();
    let path = dir.join(format!("g2g-m767-{}.mp4", std::process::id()));
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.5:sample_rate=48000",
            "-c:a",
            "libopus",
            "-b:a",
            "64k",
        ])
        .arg(&path)
        .status()
        .expect("run ffmpeg");
    if !status.success() {
        eprintln!("skipping: this ffmpeg cannot encode libopus into mp4");
        return;
    }
    let file = std::fs::read(&path).unwrap();

    // ffprobe's packet sizes are the sample-table ground truth.
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "packet=size",
            "-of",
            "csv=p=0",
        ])
        .arg(&path)
        .output()
        .expect("run ffprobe");
    // A packet with side data prints a trailing empty CSV column: strip it.
    let expected_sizes: Vec<usize> = String::from_utf8_lossy(&probe.stdout)
        .lines()
        .filter_map(|l| l.trim().trim_end_matches(',').parse().ok())
        .collect();
    assert!(!expected_sizes.is_empty(), "ffprobe reports packets");
    let _ = std::fs::remove_file(&path);

    let (caps, frames) = demux_opus(&file).await;
    assert!(
        caps.iter().any(|c| matches!(
            c,
            Caps::Audio {
                format: AudioFormat::Opus,
                sample_rate: 48000,
                ..
            }
        )),
        "Opus caps at 48 kHz announced, got {caps:?}"
    );
    let sizes: Vec<usize> = frames.iter().map(Vec::len).collect();
    assert_eq!(
        sizes, expected_sizes,
        "per-packet sizes match ffprobe's sample-table read"
    );
}
