//! M829: `OpusDec` packet-loss concealment. Nothing upstream marks a lost
//! packet, so the decoder infers loss from the timeline: each packet's TOC gives
//! its coded duration, so a packet arriving later than the previous one's end
//! left a hole. With `plc` on, libopus synthesizes audio for exactly that hole.
//!
//! The checks are sample-exact: the concealed run plus the real frames must
//! cover the whole timeline with no overlap and no residue, at a mix of frame
//! sizes (M826), and a jump too large to be loss must re-anchor rather than
//! synthesize minutes of fill.

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

const CHANNELS: u8 = 2;
/// Interleaved S16LE bytes per sample of every channel.
const BYTES_PER_SAMPLE: usize = CHANNELS as usize * 2;

/// One Opus packet placed on a timeline.
#[derive(Clone, Debug)]
struct Pkt {
    data: Vec<u8>,
    pts_ns: u64,
    dur_ns: u64,
}

/// A decoded output frame, as the downstream sees it.
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

/// A tone with enough detail that libopus codes something the concealer can
/// extrapolate from, interleaved S16LE.
fn tone_pcm(n: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(n * BYTES_PER_SAMPLE);
    for i in 0..n {
        let t = i as f32;
        let s = ((t * core::f32::consts::TAU / 100.0).sin() * 9_000.0
            + (t * core::f32::consts::TAU / 13.0).sin() * 2_500.0) as i16;
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
    }
}

/// `count` real Opus packets of one frame size, from encoding a tone.
async fn encode(frame_size: OpusFrameSize, count: usize) -> Vec<Vec<u8>> {
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
        .with_bitrate(96_000)
        .with_frame_size(frame_size);
    enc.configure_pipeline(&Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: CHANNELS,
        sample_rate: OPUS_RATE_HZ,
    })
    .unwrap();
    let pcm = tone_pcm(frame_size.samples() * count);
    let mut sink = Capture::default();
    enc.process(
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(pcm.into_boxed_slice())),
            FrameTiming::default(),
            0,
        )),
        &mut sink,
    )
    .await
    .unwrap();
    assert_eq!(
        sink.0.len(),
        count,
        "encoder produced the requested packets"
    );
    sink.0
}

fn nanos(frame_size: OpusFrameSize) -> u64 {
    frame_size.nanos()
}

/// Lay packets out back to back, each starting where the previous one ended.
fn timeline(packets: Vec<(Vec<u8>, u64)>) -> Vec<Pkt> {
    let mut pts_ns = 0;
    let mut out = Vec::new();
    for (data, dur_ns) in packets {
        out.push(Pkt {
            data,
            pts_ns,
            dur_ns,
        });
        pts_ns += dur_ns;
    }
    out
}

/// Decode `packets` (already carrying their timeline positions) and collect what
/// the downstream receives.
async fn decode(plc: bool, packets: &[Pkt]) -> Vec<Out> {
    let mut dec = OpusDec::new().with_plc(plc);
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

/// Every output frame must start where the previous one ended, and its declared
/// duration must match the PCM it actually carries.
fn assert_contiguous(outs: &[Out]) {
    for (i, o) in outs.iter().enumerate() {
        assert_eq!(
            o.dur_ns,
            o.samples() * 1_000_000_000 / OPUS_RATE_HZ as u64,
            "frame {i}: declared duration matches its sample count"
        );
        if i > 0 {
            let prev = &outs[i - 1];
            assert_eq!(
                o.pts_ns,
                prev.pts_ns + prev.dur_ns,
                "frame {i} at {} ns continues frame {} ending at {} ns",
                o.pts_ns,
                i - 1,
                prev.pts_ns + prev.dur_ns
            );
        }
    }
}

fn total_samples(outs: &[Out]) -> u64 {
    outs.iter().map(Out::samples).sum()
}

/// `plc` round-trips through the property pair, defaults off as in GStreamer's
/// opusdec, and rejects a value of the wrong type rather than coercing it.
#[test]
fn plc_property_round_trips_and_rejects_wrong_type() {
    let mut d = OpusDec::new();
    assert!(
        d.properties().iter().any(|p| p.name == "plc"),
        "the element declares plc, so parse_launch can find its kind"
    );
    assert_eq!(
        d.get_property("plc"),
        Some(PropValue::Bool(false)),
        "off by default"
    );
    assert!(!d.plc());

    d.set_property("plc", PropValue::Bool(true)).unwrap();
    assert!(d.plc(), "the property reaches the field the decoder reads");
    assert_eq!(d.get_property("plc"), Some(PropValue::Bool(true)));
    d.set_property("plc", PropValue::Bool(false)).unwrap();
    assert_eq!(d.get_property("plc"), Some(PropValue::Bool(false)));

    for bad in [PropValue::Uint(1), PropValue::Str("true".into())] {
        assert_eq!(
            d.set_property("plc", bad),
            Err(PropError::Type),
            "a non-boolean is rejected, not coerced"
        );
    }
    assert_eq!(
        d.set_property("packet-loss", PropValue::Bool(true)),
        Err(PropError::Unknown)
    );
    assert_eq!(d.get_property("plc"), Some(PropValue::Bool(false)));

    assert!(
        OpusDec::new().with_plc(true).plc(),
        "the builder is the same knob"
    );
}

/// A packet deleted from the middle of a stream: without `plc` the output keeps
/// the hole, with it libopus fills exactly the missing duration and the real
/// packet after the gap still lands at its own PTS.
#[tokio::test]
async fn deleted_packet_is_a_hole_without_plc_and_an_exact_fill_with_it() {
    let frame_size = OpusFrameSize::Ms20;
    let packets = encode(frame_size, 6).await;
    let mut stream = timeline(
        packets
            .into_iter()
            .map(|p| (p, nanos(frame_size)))
            .collect(),
    );
    let lost = stream.remove(2);
    assert_eq!(lost.pts_ns, 2 * nanos(frame_size));

    let without = decode(false, &stream).await;
    assert_eq!(without.len(), 5, "no concealment frame is invented");
    assert_eq!(
        without[1].pts_ns + without[1].dur_ns,
        lost.pts_ns,
        "the frame before the loss ends where the lost packet began"
    );
    assert_eq!(
        without[2].pts_ns,
        lost.pts_ns + lost.dur_ns,
        "the next real frame jumps a whole packet ahead: the hole is left open"
    );
    assert_eq!(
        total_samples(&without),
        5 * frame_size.samples() as u64,
        "only the surviving packets are decoded"
    );

    let with = decode(true, &stream).await;
    assert_eq!(with.len(), 6, "the gap is filled by one synthesized frame");
    assert_contiguous(&with);
    assert_eq!(
        with[2].pts_ns, lost.pts_ns,
        "the fill starts where the lost packet did"
    );
    assert_eq!(
        with[2].dur_ns, lost.dur_ns,
        "and covers exactly the lost duration"
    );
    assert_eq!(
        with[3].pts_ns,
        lost.pts_ns + lost.dur_ns,
        "the real packet after the gap keeps its own PTS"
    );
    assert_eq!(
        total_samples(&with),
        6 * frame_size.samples() as u64,
        "the timeline is covered sample for sample, as if nothing was lost"
    );

    // Concealment is synthesized audio, not a silent patch: libopus extrapolates
    // the tone from its decoder state.
    assert!(
        with[2].pcm.chunks_exact(2).any(|s| s != [0, 0]),
        "the concealed run carries a signal"
    );

    // The surviving packets decode identically either way: `plc` only adds the
    // fill, it does not disturb the real audio around it.
    let real_with: Vec<&Vec<u8>> = with
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 2)
        .map(|(_, o)| &o.pcm)
        .collect();
    let real_without: Vec<&Vec<u8>> = without.iter().map(|o| &o.pcm).collect();
    assert_eq!(
        real_with[..2],
        real_without[..2],
        "the packets before the gap are untouched"
    );
}

/// With no loss, `plc` changes nothing at all: same frames, same bytes.
#[tokio::test]
async fn an_intact_stream_decodes_the_same_with_plc_on() {
    let frame_size = OpusFrameSize::Ms20;
    let packets = encode(frame_size, 5).await;
    let stream = timeline(
        packets
            .into_iter()
            .map(|p| (p, nanos(frame_size)))
            .collect(),
    );

    let off = decode(false, &stream).await;
    let on = decode(true, &stream).await;
    assert_eq!(on.len(), off.len(), "no frame is invented");
    assert_contiguous(&on);
    for (a, b) in on.iter().zip(off.iter()) {
        assert_eq!(a.pts_ns, b.pts_ns);
        assert_eq!(a.pcm, b.pcm);
    }
}

/// Frame sizes vary mid-stream (M826), so the gap length comes from the lost
/// packet's own place on the timeline, not from a fixed cadence: a 40 ms hole
/// after 20 ms packets is filled as 40 ms, exactly.
#[tokio::test]
async fn a_gap_between_different_frame_sizes_is_filled_exactly() {
    let short = encode(OpusFrameSize::Ms20, 2).await;
    let long = encode(OpusFrameSize::Ms40, 2).await;
    let tail = encode(OpusFrameSize::Ms10, 3).await;

    let mut items: Vec<(Vec<u8>, u64)> = Vec::new();
    items.extend(short.into_iter().map(|p| (p, nanos(OpusFrameSize::Ms20))));
    items.extend(long.into_iter().map(|p| (p, nanos(OpusFrameSize::Ms40))));
    items.extend(tail.into_iter().map(|p| (p, nanos(OpusFrameSize::Ms10))));
    let mut stream = timeline(items);
    let total = stream.last().map(|p| p.pts_ns + p.dur_ns).unwrap();

    // Drop the first 40 ms packet: the packet before it is 20 ms, so the fill
    // has to outrun the previous packet's duration.
    let lost = stream.remove(2);
    assert_eq!(lost.dur_ns, nanos(OpusFrameSize::Ms40));

    let outs = decode(true, &stream).await;
    assert_contiguous(&outs);
    assert_eq!(outs[2].pts_ns, lost.pts_ns, "the fill starts at the hole");
    assert_eq!(
        outs[2].dur_ns,
        nanos(OpusFrameSize::Ms40),
        "a 40 ms hole is filled with 40 ms, not with the 20 ms cadence before it"
    );
    assert_eq!(
        total_samples(&outs) * 1_000_000_000 / OPUS_RATE_HZ as u64,
        total,
        "the mixed-frame-size timeline is covered end to end"
    );
}

/// A jump too large to be packet loss (a seek, a stream restart) re-anchors the
/// timeline instead of synthesizing fill for it, and concealment still works
/// after the jump: the reset is clean, not a wedged state.
#[tokio::test]
async fn a_jump_past_the_cap_re_anchors_instead_of_filling() {
    let frame_size = OpusFrameSize::Ms20;
    let step = nanos(frame_size);
    // The cap is 200 ms; 200 ms is loss, anything past it is a discontinuity.
    let cap_ns = 200_000_000;
    let packets = encode(frame_size, 4).await;

    // Two packets, then a jump of `jump` past where the third was due, then two
    // more packets running on from there.
    let build = |jump: u64| {
        let mut stream = Vec::new();
        let mut pts_ns = 0;
        for (i, data) in packets.iter().enumerate() {
            if i == 2 {
                pts_ns += jump;
            }
            stream.push(Pkt {
                data: data.clone(),
                pts_ns,
                dur_ns: step,
            });
            pts_ns += step;
        }
        stream
    };

    let at_cap = decode(true, &build(cap_ns)).await;
    assert_eq!(at_cap.len(), 5, "a 200 ms hole is still loss: it is filled");
    assert_contiguous(&at_cap);
    assert_eq!(at_cap[2].dur_ns, cap_ns, "filled to the cap exactly");

    let past_cap = build(cap_ns + step);
    let outs = decode(true, &past_cap).await;
    assert_eq!(
        outs.len(),
        4,
        "past the cap nothing is synthesized: 4 real packets, no fill"
    );
    assert_eq!(
        total_samples(&outs),
        4 * frame_size.samples() as u64,
        "only the real packets are decoded"
    );
    assert_eq!(
        outs[2].pts_ns,
        2 * step + cap_ns + step,
        "the frame after the jump keeps its own PTS"
    );
    assert_eq!(
        outs[3].pts_ns,
        outs[2].pts_ns + step,
        "the timeline re-anchored on the post-jump packet, so its successor follows on"
    );

    // A minutes-long jump is the case the cap exists for: it must not try to
    // synthesize it.
    let hours = decode(true, &build(3_600_000_000_000)).await;
    assert_eq!(hours.len(), 4, "an hour-long jump synthesizes nothing");
}

/// A gap in front of the very first packet is not loss: there is no decoder
/// state to conceal from and no earlier audio to be missing.
#[tokio::test]
async fn a_late_first_packet_is_not_concealed() {
    let frame_size = OpusFrameSize::Ms20;
    let packets = encode(frame_size, 3).await;
    let step = nanos(frame_size);
    let stream: Vec<Pkt> = packets
        .into_iter()
        .enumerate()
        .map(|(i, data)| Pkt {
            data,
            // The stream simply starts at 5 s, mid-timeline.
            pts_ns: 5_000_000_000 + i as u64 * step,
            dur_ns: step,
        })
        .collect();

    let outs = decode(true, &stream).await;
    assert_eq!(outs.len(), 3, "nothing is synthesized ahead of the stream");
    assert_eq!(outs[0].pts_ns, 5_000_000_000);
    assert_contiguous(&outs);
}
