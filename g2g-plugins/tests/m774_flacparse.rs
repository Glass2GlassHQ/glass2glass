//! M774: the native FLAC parse layer. `flacparse` takes a bare `.flac` byte
//! stream (arbitrary chunks), announces the STREAMINFO caps, forwards the
//! `fLaC` header block in-band, and splits CRC-validated frames. The framing
//! is oracled against ffprobe's packet list for the same file (sizes and
//! sample-accurate timestamps), and `filesrc` types `.flac` by extension so a
//! launch line reaches the parser.
#![cfg(feature = "std")]

use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::time::ticks_to_ns;
use g2g_core::{AsyncElement, AudioFormat, Caps, G2gError, OutputSink, PushOutcome};
use g2g_plugins::flacparse::FlacParse;

#[derive(Default)]
struct CaptureSink {
    caps: Vec<Caps>,
    frames: Vec<(u64, Vec<u8>)>,
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
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
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
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
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
        assert_eq!(
            *pts_ns,
            ticks_to_ns(*ref_pts, 44_100),
            "frame {i} pts matches ffprobe"
        );
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

const SYNTH_RATE: u64 = 44_100;
const SYNTH_BLOCK: usize = 16;
// block size code 6 (8-bit size after the number), rate code 9 (44.1 kHz)
const SYNTH_SIZE_AND_RATE: u8 = 0x69;
// mono, 8-bit samples
const SYNTH_CHANNELS_AND_DEPTH: u8 = 0x02;
const VERBATIM_SUBFRAME: u8 = 0x02;

fn flac_crc8(data: &[u8]) -> u8 {
    data.iter().fold(0u8, |crc, &b| {
        (0..8).fold(crc ^ b, |c, _| {
            if c & 0x80 != 0 {
                (c << 1) ^ 0x07
            } else {
                c << 1
            }
        })
    })
}

fn flac_crc16(data: &[u8]) -> u16 {
    data.iter().fold(0u16, |crc, &b| {
        (0..8).fold(crc ^ (u16::from(b) << 8), |c, _| {
            if c & 0x8000 != 0 {
                (c << 1) ^ 0x8005
            } else {
                c << 1
            }
        })
    })
}

fn frame_header(number: u8) -> Vec<u8> {
    let mut header = vec![
        0xFF,
        0xF8,
        SYNTH_SIZE_AND_RATE,
        SYNTH_CHANNELS_AND_DEPTH,
        number,
        (SYNTH_BLOCK - 1) as u8,
    ];
    header.push(flac_crc8(&header));
    header
}

// a mono 8-bit verbatim frame, with a valid header for frame 5 planted in its samples
fn synth_frame(number: u8) -> Vec<u8> {
    let mut frame = frame_header(number);
    frame.push(VERBATIM_SUBFRAME);
    let mut samples = vec![0x10u8; SYNTH_BLOCK];
    let alias = frame_header(5);
    samples[2..2 + alias.len()].copy_from_slice(&alias);
    frame.extend_from_slice(&samples);
    frame.extend_from_slice(&flac_crc16(&frame).to_be_bytes());
    frame
}

fn synth_stream_header() -> Vec<u8> {
    let mut header = b"fLaC".to_vec();
    header.extend_from_slice(&[0x80, 0, 0, 34]);
    let block = (SYNTH_BLOCK as u16).to_be_bytes();
    header.extend_from_slice(&block);
    header.extend_from_slice(&block);
    header.extend_from_slice(&[0; 6]);
    let bits_per_sample = 8u64;
    let packed = (SYNTH_RATE << 44) | ((bits_per_sample - 1) << 36);
    header.extend_from_slice(&packed.to_be_bytes());
    header.extend_from_slice(&[0; 16]);
    header
}

async fn parse_frames(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut parse = FlacParse::new();
    parse
        .configure_pipeline(&Caps::Audio {
            format: AudioFormat::Flac,
            channels: 0,
            sample_rate: 0,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .expect("configure");
    let mut sink = CaptureSink::default();
    parse
        .process(chunk(stream, 0), &mut sink)
        .await
        .expect("parse");
    parse
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    sink.frames.into_iter().skip(1).map(|(_, f)| f).collect()
}

#[tokio::test]
async fn a_sync_aliased_in_the_samples_does_not_split_a_frame() {
    let frames: Vec<Vec<u8>> = (0..3).map(synth_frame).collect();
    let stream = [synth_stream_header(), frames.concat()].concat();
    assert_eq!(parse_frames(&stream).await, frames);
}

#[tokio::test]
async fn a_truncated_last_frame_is_dropped() {
    let frames: Vec<Vec<u8>> = (0..2).map(synth_frame).collect();
    let mut stream = [synth_stream_header(), frames.concat()].concat();
    stream.truncate(stream.len() - 1);
    assert_eq!(parse_frames(&stream).await, frames[..1]);
}

#[tokio::test]
async fn a_flush_restarts_timestamps_at_the_demuxer_landing() {
    let frame_pts =
        |number: u8| u64::from(number) * SYNTH_BLOCK as u64 * 1_000_000_000 / SYNTH_RATE;
    let mut parse = FlacParse::new();
    parse
        .configure_pipeline(&Caps::Audio {
            format: AudioFormat::Flac,
            channels: 0,
            sample_rate: 0,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .expect("configure");
    let mut sink = CaptureSink::default();
    // a demuxer sends the header and then one timed frame per packet, and sends the header again after a seek
    let mut feed = vec![chunk(&synth_stream_header(), 0)];
    feed.extend((0..4).map(|number| chunk(&synth_frame(number), frame_pts(number))));
    feed.push(PipelinePacket::Flush);
    feed.push(chunk(&synth_stream_header(), 0));
    let landing = 1..4;
    feed.extend(
        landing
            .clone()
            .map(|number| chunk(&synth_frame(number), frame_pts(number))),
    );
    feed.push(PipelinePacket::Eos);
    for packet in feed {
        let flushed = matches!(packet, PipelinePacket::Flush);
        parse.process(packet, &mut sink).await.expect("parse");
        if flushed {
            sink.frames.clear();
        }
    }
    let after_flush: Vec<u64> = sink.frames.iter().skip(1).map(|(pts, _)| *pts).collect();
    assert_eq!(after_flush, landing.map(frame_pts).collect::<Vec<_>>());
}
