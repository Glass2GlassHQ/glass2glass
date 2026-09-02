//! M857: `OpusDec` in-band FEC. An Opus stream encoded with FEC carries a
//! low-bitrate redundant copy (LBRR) of every frame inside the *next* packet, so
//! a decoder that lost one packet can rebuild it from the packet that follows
//! instead of concealing it blind (M829 PLC).
//!
//! The checks are comparative: the same lossy stream is decoded twice, once with
//! `use-inband-fec` and once without, and the recovered gap must sit far closer
//! to what the intact stream decodes to than concealment does. Timing is
//! unchanged either way: the fill covers exactly the lost span.

#![cfg(feature = "opus")]

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PropError, PropValue, PushOutcome,
};
use g2g_plugins::opusdec::OpusDec;
use g2g_plugins::opusenc::{OpusEnc, OpusFrameSize};
use g2g_plugins::opusparse::OPUS_RATE_HZ;

/// Mono: FEC lives in the SILK layer, and one channel keeps the bitrate the
/// redundant copy has to fit into comfortably low.
const CHANNELS: u8 = 1;
const BYTES_PER_SAMPLE: usize = CHANNELS as usize * 2;
const FRAME_SIZE: OpusFrameSize = OpusFrameSize::Ms20;
/// Low enough that libopus codes in SILK, where LBRR exists at all.
const BITRATE: i32 = 24_000;
/// Loss the encoder codes for. Above ~8 % it also keeps libopus out of
/// CELT-only mode, which has no FEC.
const LOSS_PERCENT: u8 = 30;

#[derive(Clone, Debug)]
struct Pkt {
    data: Vec<u8>,
    pts_ns: u64,
    dur_ns: u64,
}

#[derive(Debug)]
struct Out {
    pts_ns: u64,
    dur_ns: u64,
    pcm: Vec<u8>,
}

impl Out {
    fn samples(&self) -> u64 {
        (self.pcm.len() / BYTES_PER_SAMPLE) as u64
    }
}

#[derive(Default)]
struct PcmSink(Vec<Out>);

impl OutputSink for PcmSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.0.push(Out {
                        pts_ns: f.timing.pts_ns,
                        dur_ns: f.timing.duration_ns,
                        pcm: s.to_vec(),
                    });
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// A rising harmonic chirp: the content changes every frame, so concealment
/// (which extrapolates the previous frame) cannot stand in for the real audio.
fn chirp_pcm(n: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(n * BYTES_PER_SAMPLE);
    let mut phase = 0f32;
    for i in 0..n {
        let t = i as f32 / n as f32;
        let hz = 200.0 * 8.0f32.powf(t);
        phase += core::f32::consts::TAU * hz / OPUS_RATE_HZ as f32;
        let s = (phase.sin() * 9_000.0 + (2.0 * phase).sin() * 3_000.0) as i16;
        for _ in 0..CHANNELS {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
    }
    bytes
}

fn opus_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: CHANNELS,
        sample_rate: OPUS_RATE_HZ,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// Encode the chirp into `count` packets, with or without in-band FEC.
async fn encode(fec: bool, count: usize) -> Vec<Vec<u8>> {
    #[derive(Default)]
    struct Capture(Vec<Vec<u8>>);
    impl OutputSink for Capture {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                if let PipelinePacket::DataFrame(f) = packet {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.0.push(s.to_vec());
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    let mut enc = OpusEnc::new()
        .with_bitrate(BITRATE)
        .with_frame_size(FRAME_SIZE)
        .with_inband_fec(fec)
        .with_packet_loss_percentage(if fec { LOSS_PERCENT } else { 0 });
    enc.configure_pipeline(&Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: CHANNELS,
        sample_rate: OPUS_RATE_HZ,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    })
    .unwrap();
    assert_eq!(enc.inband_fec(), fec, "libopus took the FEC setting");

    let pcm = chirp_pcm(FRAME_SIZE.samples() * count);
    let mut sink = Capture::default();
    // Feed frame by frame: libopus decides per packet whether to spend bits on
    // the redundant copy, exactly as it would on a live stream.
    for (i, chunk) in pcm
        .chunks(FRAME_SIZE.samples() * BYTES_PER_SAMPLE)
        .enumerate()
    {
        enc.process(
            PipelinePacket::DataFrame(Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(chunk.to_vec().into_boxed_slice())),
                FrameTiming {
                    pts_ns: i as u64 * FRAME_SIZE.nanos(),
                    ..FrameTiming::default()
                },
                i as u64,
            )),
            &mut sink,
        )
        .await
        .unwrap();
    }
    assert_eq!(sink.0.len(), count, "one packet per fed frame");
    sink.0
}

/// Lay packets out back to back at the frame cadence.
fn timeline(packets: Vec<Vec<u8>>) -> Vec<Pkt> {
    packets
        .into_iter()
        .enumerate()
        .map(|(i, data)| Pkt {
            data,
            pts_ns: i as u64 * FRAME_SIZE.nanos(),
            dur_ns: FRAME_SIZE.nanos(),
        })
        .collect()
}

async fn decode(plc: bool, fec: bool, packets: &[Pkt]) -> Vec<Out> {
    let mut dec = OpusDec::new().with_plc(plc).with_inband_fec(fec);
    dec.configure_pipeline(&opus_caps()).unwrap();
    let mut sink = PcmSink::default();
    for p in packets {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(p.data.clone().into_boxed_slice())),
            FrameTiming {
                pts_ns: p.pts_ns,
                duration_ns: p.dur_ns,
                ..FrameTiming::default()
            },
            0,
        );
        dec.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
    }
    sink.0
}

fn samples_of(pcm: &[u8]) -> Vec<i16> {
    pcm.as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Signal-to-noise ratio in dB of `test` against `reference`, over the samples
/// they share. Higher is closer to the audio that was actually lost.
fn snr_db(reference: &[u8], test: &[u8]) -> f64 {
    let r = samples_of(reference);
    let t = samples_of(test);
    assert_eq!(r.len(), t.len(), "windows compare sample for sample");
    let signal: f64 = r.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    let noise: f64 = r
        .iter()
        .zip(t.iter())
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    10.0 * (signal / noise.max(1e-9)).log10()
}

fn assert_contiguous(outs: &[Out]) {
    for (i, o) in outs.iter().enumerate() {
        assert_eq!(
            o.dur_ns,
            o.samples() * 1_000_000_000 / OPUS_RATE_HZ as u64,
            "frame {i}: declared duration matches its sample count"
        );
        if i > 0 {
            let prev = &outs[i - 1];
            assert_eq!(o.pts_ns, prev.pts_ns + prev.dur_ns, "frame {i} continues");
        }
    }
}

/// The FEC properties round-trip through both halves of the property pair and
/// reject values of the wrong type or out of range.
#[test]
fn fec_properties_round_trip() {
    let mut d = OpusDec::new();
    assert!(d.properties().iter().any(|p| p.name == "use-inband-fec"));
    assert_eq!(
        d.get_property("use-inband-fec"),
        Some(PropValue::Bool(false))
    );
    d.set_property("use-inband-fec", PropValue::Bool(true))
        .unwrap();
    assert!(
        d.inband_fec(),
        "the property reaches the field decode reads"
    );
    assert_eq!(
        d.set_property("use-inband-fec", PropValue::Uint(1)),
        Err(PropError::Type)
    );
    assert!(OpusDec::new().with_inband_fec(true).inband_fec());

    let mut e = OpusEnc::new();
    for name in ["inband-fec", "packet-loss-percentage"] {
        assert!(
            e.properties().iter().any(|p| p.name == name),
            "{name} is declared, so parse_launch can find its kind"
        );
    }
    assert_eq!(e.get_property("inband-fec"), Some(PropValue::Bool(false)));
    assert_eq!(
        e.get_property("packet-loss-percentage"),
        Some(PropValue::Uint(0))
    );
    e.set_property("inband-fec", PropValue::Bool(true)).unwrap();
    e.set_property("packet-loss-percentage", PropValue::Uint(25))
        .unwrap();
    assert_eq!(e.get_property("inband-fec"), Some(PropValue::Bool(true)));
    assert_eq!(
        e.get_property("packet-loss-percentage"),
        Some(PropValue::Uint(25))
    );
    assert_eq!(
        e.set_property("packet-loss-percentage", PropValue::Uint(101)),
        Err(PropError::Value),
        "a percentage over 100 is rejected, not clamped"
    );
    assert_eq!(
        e.set_property("inband-fec", PropValue::Str("true".into())),
        Err(PropError::Type)
    );

    // The builders are the same knobs, and libopus confirms them once the
    // encoder exists.
    let mut e = OpusEnc::new()
        .with_inband_fec(true)
        .with_packet_loss_percentage(20);
    e.configure_pipeline(&Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: CHANNELS,
        sample_rate: OPUS_RATE_HZ,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    })
    .unwrap();
    assert!(e.inband_fec(), "libopus reports FEC on");
    assert_eq!(e.packet_loss_percentage(), 20);
}

/// The core case: one packet dropped from an FEC-encoded stream. With
/// `use-inband-fec` the gap is rebuilt from the redundant copy in the next
/// packet, and that reconstruction is far closer to the lost audio than the
/// concealment the same stream gets without it. Timing is identical either way.
#[tokio::test]
async fn a_lost_packet_is_rebuilt_from_the_next_packets_redundant_copy() {
    let packets = encode(true, 10).await;
    assert!(
        packets.iter().all(|p| p[0] >> 3 < 16),
        "the encoder stayed in SILK/hybrid, the only modes that carry LBRR"
    );
    let stream = timeline(packets);

    let intact = decode(true, true, &stream).await;
    assert_eq!(
        intact.len(),
        10,
        "the reference decode is one frame per packet"
    );

    let mut lossy = stream.clone();
    let lost = lossy.remove(4);

    let plc = decode(true, false, &lossy).await;
    let fec = decode(true, true, &lossy).await;

    // (a) timing: both fills cover exactly the lost span, and the stream comes
    // out with the same frames and sample count as if nothing was lost.
    for (name, outs) in [("plc", &plc), ("fec", &fec)] {
        assert_eq!(
            outs.len(),
            intact.len(),
            "{name}: no frame invented or lost"
        );
        assert_contiguous(outs);
        assert_eq!(
            outs[4].pts_ns, lost.pts_ns,
            "{name}: fill starts at the hole"
        );
        assert_eq!(outs[4].dur_ns, lost.dur_ns, "{name}: fill covers the hole");
        for (i, (o, r)) in outs.iter().zip(intact.iter()).enumerate() {
            assert_eq!(o.pts_ns, r.pts_ns, "{name}: frame {i} keeps its PTS");
            assert_eq!(o.samples(), r.samples(), "{name}: frame {i} sample count");
        }
    }

    // (b) quality: measured against what the intact stream decodes to over the
    // same window, the FEC rebuild must beat blind concealment outright.
    let reference = &intact[4].pcm;
    let plc_snr = snr_db(reference, &plc[4].pcm);
    let fec_snr = snr_db(reference, &fec[4].pcm);
    assert!(
        fec_snr > plc_snr + 6.0,
        "FEC rebuild ({fec_snr:.1} dB) must be well clear of concealment ({plc_snr:.1} dB)"
    );
    assert!(
        fec_snr > 5.0,
        "the rebuilt frame tracks the lost audio ({fec_snr:.1} dB)"
    );

    // The packets around the gap decode identically: FEC only supplies the fill.
    assert_eq!(
        plc[..4].iter().map(|o| &o.pcm).collect::<Vec<_>>(),
        fec[..4].iter().map(|o| &o.pcm).collect::<Vec<_>>()
    );
}

/// Two packets in a row lost: the redundant copy in the arriving packet reaches
/// back one frame, so the tail of the gap is rebuilt and the head stays
/// concealment. The fill still covers the whole hole.
#[tokio::test]
async fn a_two_packet_gap_rebuilds_only_its_last_frame() {
    let stream = timeline(encode(true, 10).await);
    let intact = decode(true, true, &stream).await;

    let mut lossy = stream.clone();
    lossy.remove(4);
    lossy.remove(4);

    let plc = decode(true, false, &lossy).await;
    let fec = decode(true, true, &lossy).await;
    for outs in [&plc, &fec] {
        assert_contiguous(outs);
        assert_eq!(
            outs[4].dur_ns,
            2 * FRAME_SIZE.nanos(),
            "the fill spans both lost packets"
        );
    }

    let frame_bytes = FRAME_SIZE.samples() * BYTES_PER_SAMPLE;
    let reference: Vec<u8> = intact[4]
        .pcm
        .iter()
        .chain(intact[5].pcm.iter())
        .copied()
        .collect();
    let plc_tail = snr_db(&reference[frame_bytes..], &plc[4].pcm[frame_bytes..]);
    let fec_tail = snr_db(&reference[frame_bytes..], &fec[4].pcm[frame_bytes..]);
    // A smaller margin than a single-packet loss: the rebuild runs off a decoder
    // state that spent the previous frame concealing, so it recovers less.
    assert!(
        fec_tail > plc_tail + 3.0,
        "the last lost frame is rebuilt ({fec_tail:.1} dB) rather than concealed ({plc_tail:.1} dB)"
    );
    assert_eq!(
        plc[4].pcm[..frame_bytes],
        fec[4].pcm[..frame_bytes],
        "the frame before it has no redundant copy anywhere: pure concealment either way"
    );
}

/// A stream encoded without FEC has no redundant copy to find. Asking for it
/// must still decode: libopus falls back to concealment, so the output matches
/// the PLC path frame for frame.
#[tokio::test]
async fn a_stream_without_fec_falls_back_to_concealment() {
    let stream = timeline(encode(false, 6).await);
    let mut lossy = stream.clone();
    lossy.remove(2);

    let plc = decode(true, false, &lossy).await;
    let fec = decode(true, true, &lossy).await;
    assert_eq!(fec.len(), plc.len());
    assert_contiguous(&fec);
    for (a, b) in fec.iter().zip(plc.iter()) {
        assert_eq!(a.pts_ns, b.pts_ns);
        assert_eq!(a.samples(), b.samples());
    }
}

/// With no loss, `use-inband-fec` changes nothing: same frames, same bytes.
#[tokio::test]
async fn an_intact_stream_decodes_the_same_with_fec_on() {
    let stream = timeline(encode(true, 6).await);
    let off = decode(true, false, &stream).await;
    let on = decode(true, true, &stream).await;
    assert_eq!(on.len(), off.len());
    for (a, b) in on.iter().zip(off.iter()) {
        assert_eq!(a.pts_ns, b.pts_ns);
        assert_eq!(a.pcm, b.pcm);
    }
}
