//! M750: Opus decode applies the container trims. An Opus elementary stream
//! carries encoder lookahead (pre-skip) at the head and codec padding at the
//! tail; `OggDemux` + `OpusDec` must discard both so the decoded PCM has the
//! same sample count as ffmpeg / gstreamer.
//!
//! Opus decode is deterministic, so the mono fixture is a hard bit-exact oracle:
//! its decoded PCM hashes identically to `ffmpeg -c:a libopus`. Stereo differs
//! from ffmpeg by at most 1 LSB (g2g decodes via libopus' int16 API, ffmpeg via
//! its float API + convert; gstreamer shows the same class of difference), so the
//! stereo check is the exact sample count, which is what this milestone fixes.

#![cfg(all(feature = "opus", feature = "std"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, PushOutcome,
};
use g2g_plugins::oggdemux::OggDemux;
use g2g_plugins::opusdec::OpusDec;
use g2g_plugins::opusenc::OpusEnc;
use g2g_plugins::opusparse::OPUS_RATE_HZ;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// FNV-1a 64: a dependency-free stable hash for the checked-in reference PCM.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name)
}

/// Decode `fixture` to raw S16LE via the exact repro line and return the bytes.
async fn decode_pcm(fixture_name: &str, channels: u8) -> Vec<u8> {
    let out = std::env::temp_dir().join(format!(
        "g2g_m750_{}_{}.raw",
        std::process::id(),
        fixture_name
    ));
    let line = format!(
        "filesrc location={src} ! decodebin ! audioconvert ! \
         audio/x-raw,format=S16LE,rate=48000,channels={ch} ! filesink location={out}",
        src = fixture(fixture_name),
        ch = channels,
        out = out.display(),
    );
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).expect("pipeline parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("pipeline runs");
    let bytes = std::fs::read(&out).expect("output written");
    let _ = std::fs::remove_file(&out);
    bytes
}

#[tokio::test]
async fn mono_fixture_is_bit_exact_after_trim() {
    // 0.25 s of 440 Hz mono at 48 kHz: exactly 12000 samples once pre-skip and
    // end padding are removed. The hash matches `ffmpeg -c:a libopus` decode.
    let pcm = decode_pcm("opus_mono_48k.opus", 1).await;
    assert_eq!(
        pcm.len(),
        12_000 * 2,
        "12000 mono samples, no pre-skip/padding"
    );
    assert_eq!(
        fnv1a(&pcm),
        0xa989_609d_8af3_d090,
        "decoded PCM is bit-identical to the ffmpeg reference"
    );
}

#[tokio::test]
async fn stereo_fixture_has_exact_sample_count() {
    // Stereo decode differs from ffmpeg by <=1 LSB (int16 vs float decode API),
    // so assert the exact sample count, which is the milestone's fix.
    let pcm = decode_pcm("opus_stereo_48k.opus", 2).await;
    assert_eq!(
        pcm.len(),
        12_000 * 2 * 2,
        "12000 samples/channel, no pre-skip/padding"
    );
}

// -- Malformed-input handling ------------------------------------------------

#[derive(Default)]
struct CaptureSink {
    frames: Vec<Vec<u8>>,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames.push(s.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// One Ogg page carrying `packets` for `serial`, with an explicit `granule`.
fn page(header_type: u8, serial: u32, seq: u32, granule: u64, packets: &[&[u8]]) -> Vec<u8> {
    let mut table = Vec::new();
    let mut body = Vec::new();
    for p in packets {
        let mut n = p.len();
        loop {
            let seg = n.min(255);
            table.push(seg as u8);
            n -= seg;
            if seg < 255 {
                break;
            }
        }
        body.extend_from_slice(p);
    }
    let mut out = b"OggS".to_vec();
    out.push(0);
    out.push(header_type);
    out.extend_from_slice(&granule.to_le_bytes());
    out.extend_from_slice(&serial.to_le_bytes());
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.push(table.len() as u8);
    out.extend_from_slice(&table);
    out.extend_from_slice(&body);
    out
}

fn opus_head(channels: u8, pre_skip: u16) -> Vec<u8> {
    let mut h = b"OpusHead".to_vec();
    h.push(1);
    h.push(channels);
    h.extend_from_slice(&pre_skip.to_le_bytes());
    h.extend_from_slice(&48_000u32.to_le_bytes());
    h.extend_from_slice(&[0, 0, 0]);
    h
}

/// A bogus pre-skip larger than the whole stream and an end granule smaller than
/// the decoded sample count must not underflow, panic, or over-read: the demuxer
/// clamps the keep count with saturating math and drops fully-padded packets.
#[tokio::test]
async fn malformed_preskip_and_granule_are_clamped() {
    let serial = 42;
    let mut stream = Vec::new();
    // pre-skip 0xFFFF (far beyond the stream), granule 1 on the EOS page (far
    // below the ~1440 decoded samples of the three 480-sample SILK-NB packets).
    stream.extend_from_slice(&page(0x02, serial, 0, 0, &[&opus_head(1, 0xFFFF)]));
    stream.extend_from_slice(&page(0x00, serial, 1, 0, &[b"OpusTags\0\0\0\0"]));
    stream.extend_from_slice(&page(
        0x04, // end-of-stream
        serial,
        2,
        1, // granule below the decoded count
        &[&[0x08, 0xA0], &[0x08, 0xA1], &[0x08, 0xA2]],
    ));

    let mut demux = OggDemux::new();
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Ogg,
        })
        .unwrap();
    let mut sink = CaptureSink::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(stream.into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    // Must not panic on the underflowing granule / oversized pre-skip.
    demux
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .unwrap();
    demux.process(PipelinePacket::Eos, &mut sink).await.unwrap();

    // The OpusHead is forwarded; audio packets past granule 1 are pure padding
    // and dropped, leaving a bounded, header-only-or-shorter output.
    assert!(
        sink.frames.first().map(|f| f.starts_with(b"OpusHead")) == Some(true),
        "OpusHead forwarded in-band"
    );
    assert!(
        sink.frames.len() <= 2,
        "padding packets beyond the tiny granule are dropped, got {}",
        sink.frames.len()
    );
}

/// A decoder handed an oversized pre-skip via `OpusHead` trims the whole frame to
/// nothing rather than panicking or over-reading, and a bogus huge `duration_ns`
/// saturates to the full frame.
#[tokio::test]
async fn decoder_clamps_oversized_preskip_and_duration() {
    // A real 20 ms mono packet from the encoder.
    let mut enc = OpusEnc::new().with_bitrate(64_000);
    enc.configure_pipeline(&Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: 1,
        sample_rate: OPUS_RATE_HZ,
    })
    .unwrap();
    let n = (OPUS_RATE_HZ as usize * 20) / 1000;
    let pcm: Vec<u8> = (0..n)
        .flat_map(|i| (((i as f32 * 0.1).sin() * 8000.0) as i16).to_le_bytes())
        .collect();
    #[derive(Default)]
    struct Grab {
        packets: Vec<Vec<u8>>,
    }
    impl OutputSink for Grab {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(f) = packet {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.packets.push(s.to_vec());
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }
    let mut g = Grab::default();
    enc.process(
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(pcm.into_boxed_slice())),
            FrameTiming::default(),
            0,
        )),
        &mut g,
    )
    .await
    .unwrap();
    enc.process(PipelinePacket::Eos, &mut g).await.unwrap();
    let packet = g.packets.pop().expect("one encoded packet");

    let mut dec = OpusDec::new();
    dec.configure_pipeline(&Caps::Audio {
        format: AudioFormat::Opus,
        channels: 1,
        sample_rate: OPUS_RATE_HZ,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    // Oversized pre-skip via OpusHead: the whole 960-sample frame is inside the
    // skip window, so it decodes to no PCM (consumed, not emitted, no panic).
    dec.process(
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(
                opus_head(1, 0xFFFF).into_boxed_slice(),
            )),
            FrameTiming::default(),
            0,
        )),
        &mut sink,
    )
    .await
    .unwrap();
    dec.process(
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(packet.clone().into_boxed_slice())),
            // A bogus huge duration must saturate, not overflow.
            FrameTiming {
                duration_ns: u64::MAX,
                ..FrameTiming::default()
            },
            1,
        )),
        &mut sink,
    )
    .await
    .unwrap();
    assert!(
        sink.frames.is_empty(),
        "a frame wholly inside an oversized pre-skip yields no PCM"
    );
}
