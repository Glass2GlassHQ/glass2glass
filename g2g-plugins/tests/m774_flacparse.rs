//! M774: the native FLAC parse layer. `flacparse` takes a bare `.flac` byte
//! stream (arbitrary chunks), announces the STREAMINFO caps, forwards the
//! `fLaC` header block in-band, and splits CRC-validated frames. The framing
//! is oracled against ffprobe's packet list for the same file (sizes and
//! sample-accurate timestamps), and `filesrc` types `.flac` by extension so a
//! launch line reaches the parser.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AsyncElement, AudioFormat, Caps, G2gError, OutputSink, PushOutcome};
use g2g_plugins::flacparse::FlacParse;

#[derive(Default)]
struct CaptureSink {
    caps: Vec<Caps>,
    frames: Vec<(u64, Vec<u8>)>,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.frames.push((f.timing.pts_ns, s.to_vec()));
                    }
                }
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn chunk(data: &[u8], pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.to_vec().into_boxed_slice())),
        FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}

/// One ffprobe packet: `(pts in 1/44100, size)`.
type ProbedPacket = (u64, usize);

/// A real 0.5 s 44.1 kHz stereo tone encoded to `.flac` by ffmpeg, plus
/// ffprobe's packet list as the framing oracle, or `None` when the host cannot.
fn encode_flac(tag: &str) -> Option<(Vec<u8>, Vec<ProbedPacket>)> {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return None;
    }
    let path = std::env::temp_dir().join(format!("g2g-m774-{tag}-{}.flac", std::process::id()));
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.5:sample_rate=44100",
            "-ac",
            "2",
            "-c:a",
            "flac",
        ])
        .arg(&path)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&path).ok()?;

    // ffprobe's packet list is the framing oracle, captured while the file exists.
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "packet=pts,size",
            "-of",
            "csv=p=0",
        ])
        .arg(&path)
        .output()
        .ok()?;
    let _ = std::fs::remove_file(&path);
    let expected: Vec<ProbedPacket> = String::from_utf8_lossy(&probe.stdout)
        .lines()
        .filter_map(|l| {
            let mut it = l.trim().trim_end_matches(',').split(',');
            let pts: u64 = it.next()?.parse().ok()?;
            let size: usize = it.next()?.parse().ok()?;
            Some((pts, size))
        })
        .collect();
    Some((bytes, expected))
}

#[tokio::test]
async fn frames_match_ffprobe_packets() {
    let Some((flac, expected)) = encode_flac("frames") else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    assert!(!expected.is_empty(), "ffprobe reports packets");

    let mut parse = FlacParse::new();
    parse
        .configure_pipeline(&Caps::Audio {
            format: AudioFormat::Flac,
            channels: 0,
            sample_rate: 0,
        })
        .expect("configure");
    let mut sink = CaptureSink::default();
    // Odd-sized chunks, so frames straddle input boundaries.
    for piece in flac.chunks(997) {
        parse
            .process(chunk(piece, 0), &mut sink)
            .await
            .expect("parse");
    }
    parse
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos flush");

    // STREAMINFO caps announced before any data.
    assert_eq!(
        sink.caps.first(),
        Some(&Caps::Audio {
            format: AudioFormat::Flac,
            channels: 2,
            sample_rate: 44_100,
        }),
        "STREAMINFO caps"
    );

    // Frame 0 is the in-band header block (the decoder's extradata).
    let (first_pts, header) = &sink.frames[0];
    assert!(header.starts_with(b"fLaC"), "header block forwarded first");
    assert_eq!(*first_pts, 0);

    // Every media frame matches ffprobe's packet list: same count, sizes, and
    // sample-accurate timestamps (ffprobe pts is in 1/44100).
    let media = &sink.frames[1..];
    assert_eq!(media.len(), expected.len(), "frame count matches ffprobe");
    for (i, ((pts_ns, data), (ref_pts, ref_size))) in media.iter().zip(&expected).enumerate() {
        assert_eq!(data.len(), *ref_size, "frame {i} size matches ffprobe");
        let ref_ns = *ref_pts as u128 * 1_000_000_000 / 44_100;
        assert_eq!(*pts_ns as u128, ref_ns, "frame {i} pts matches ffprobe");
    }
}

#[tokio::test]
async fn launch_line_types_flac_and_parses() {
    let Some((flac, _)) = encode_flac("launch") else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    let path = std::env::temp_dir().join(format!("g2g-m774-launch-{}.flac", std::process::id()));
    std::fs::write(&path, &flac).unwrap();

    // `.flac` types by extension, `flacparse` is a launch element, and the
    // chain negotiates and runs end to end.
    struct ZeroClock;
    impl g2g_core::PipelineClock for ZeroClock {
        fn now_ns(&self) -> u64 {
            0
        }
    }
    let reg = g2g_plugins::registry::default_registry();
    let line = format!("filesrc location={} ! flacparse ! fakesink", path.display());
    let graph =
        g2g_core::runtime::parse_launch(&reg, &line).expect("launch line parses and negotiates");
    let stats = g2g_core::runtime::run_graph(graph, &ZeroClock, 4)
        .await
        .expect("pipeline runs");
    let _ = std::fs::remove_file(&path);
    assert!(
        stats.frames_consumed > 2,
        "header + frames flowed to the sink, got {}",
        stats.frames_consumed
    );
}
