//! M1176: the two `no_std` members of the record set.
//!
//! - `trim` keeps the frames of its range, rebases them onto its start, and ends
//!   the stream at `stop`, dropping what follows.
//! - `textdigest` joins the text frames of a window into one frame per window,
//!   and flushes the window still open at EOS.

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    ByteStreamEncoding, Caps, G2gError, OutputSink, PropValue, PushOutcome, TextFormat,
};
use g2g_plugins::textdigest::TextDigest;
use g2g_plugins::trim::Trim;

const NS_PER_SECOND: u64 = 1_000_000_000;

#[derive(Default)]
struct RecordingSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for RecordingSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.packets.push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl RecordingSink {
    fn frames(&self) -> Vec<&Frame> {
        self.packets
            .iter()
            .filter_map(|packet| match packet {
                PipelinePacket::DataFrame(frame) => Some(frame),
                _ => None,
            })
            .collect()
    }

    fn ended(&self) -> bool {
        self.packets
            .iter()
            .any(|packet| matches!(packet, PipelinePacket::Eos))
    }
}

fn frame(bytes: &[u8], pts_ns: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        sequence: pts_ns,
        meta: Default::default(),
    }
}

/// The stream `trim` is fed: one frame every tenth of a second, each holding its
/// own index so a kept frame is identifiable after the rebase.
const TRIM_FRAMES: u64 = 8;
const TRIM_FRAME_PERIOD_NS: u64 = NS_PER_SECOND / 10;
const TRIM_START_NS: u64 = 2 * TRIM_FRAME_PERIOD_NS;
const TRIM_STOP_NS: u64 = 5 * TRIM_FRAME_PERIOD_NS;

#[tokio::test]
async fn trim_keeps_the_range_and_rebases_it() {
    let mut trim = Trim::new();
    trim.set_property("start", PropValue::Uint(TRIM_START_NS))
        .expect("start is settable");
    trim.set_property("stop", PropValue::Uint(TRIM_STOP_NS))
        .expect("stop is settable");
    trim.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Raw,
    })
    .expect("trim takes any caps");

    let mut out = RecordingSink::default();
    for index in 0..TRIM_FRAMES {
        trim.process(
            PipelinePacket::DataFrame(frame(&[index as u8], index * TRIM_FRAME_PERIOD_NS)),
            &mut out,
        )
        .await
        .expect("trim handles every frame");
    }

    let kept: Vec<(u64, u64, u8)> = out
        .frames()
        .iter()
        .map(|frame| {
            let bytes = frame.domain.as_system_slice().expect("system bytes");
            (frame.timing.pts_ns, frame.timing.dts_ns, bytes[0])
        })
        .collect();
    let expected: Vec<(u64, u64, u8)> = (0..TRIM_FRAMES)
        .map(|index| index * TRIM_FRAME_PERIOD_NS)
        .filter(|pts| *pts >= TRIM_START_NS && *pts < TRIM_STOP_NS)
        .map(|pts| {
            (
                pts - TRIM_START_NS,
                pts - TRIM_START_NS,
                (pts / TRIM_FRAME_PERIOD_NS) as u8,
            )
        })
        .collect();
    assert_eq!(kept, expected, "the range comes out rebased onto its start");
    assert_eq!(trim.kept(), expected.len() as u64);
    assert!(out.ended(), "the frame at stop ends the stream");
}

/// The cues the digest is built from: `(pts seconds, text)`. The first three
/// share a window; the fourth opens the next one.
const CUES: &[(f64, &str)] = &[
    (0.0, "gate opened"),
    (1.0, "person at the door"),
    (2.0, "person left"),
    (11.0, "gate closed"),
];
const DIGEST_WINDOW_SECONDS: f64 = 10.0;

#[tokio::test]
async fn textdigest_joins_a_window_and_flushes_at_eos() {
    let mut digest = TextDigest::new();
    digest
        .set_property("window-seconds", PropValue::Double(DIGEST_WINDOW_SECONDS))
        .expect("the window is settable");
    let caps = Caps::Text {
        format: TextFormat::Utf8,
    };
    digest
        .configure_pipeline(&caps)
        .expect("the digest takes UTF-8 text");

    let mut out = RecordingSink::default();
    for (seconds, text) in CUES {
        digest
            .process(
                PipelinePacket::DataFrame(frame(
                    text.as_bytes(),
                    (seconds * NS_PER_SECOND as f64) as u64,
                )),
                &mut out,
            )
            .await
            .expect("the digest takes the cue");
    }
    // Nothing comes out until a cue past the window closes it.
    assert_eq!(out.frames().len(), 1, "one window, one frame");
    digest
        .process(PipelinePacket::Eos, &mut out)
        .await
        .expect("the open window is flushed");

    let digests = out.frames();
    assert_eq!(digests.len(), 2);
    let (closed, flushed) = (digests[0], digests[1]);
    let text = |frame: &Frame| {
        String::from_utf8(
            frame
                .domain
                .as_system_slice()
                .expect("system bytes")
                .to_vec(),
        )
        .expect("the digest is UTF-8")
    };
    let first_window: Vec<&(f64, &str)> = CUES
        .iter()
        .filter(|(seconds, _)| *seconds < DIGEST_WINDOW_SECONDS)
        .collect();
    let closed_text = text(closed);
    let lines: Vec<&str> = closed_text.lines().collect();
    assert_eq!(lines.len(), first_window.len(), "one line per cue");
    for (line, (seconds, cue)) in lines.iter().zip(&first_window) {
        assert_eq!(*line, format!("At {seconds:.1}s: {cue}"));
    }
    assert_eq!(
        closed.timing.pts_ns,
        (CUES[0].0 * NS_PER_SECOND as f64) as u64,
        "the digest is stamped at its window's start"
    );
    assert_eq!(
        closed.timing.duration_ns,
        (DIGEST_WINDOW_SECONDS * NS_PER_SECOND as f64) as u64
    );
    let last = CUES.last().expect("a cue past the window");
    assert_eq!(text(flushed), format!("At {:.1}s: {}", last.0, last.1));
    assert_eq!(
        flushed.timing.pts_ns,
        (last.0 * NS_PER_SECOND as f64) as u64,
        "the next window starts at the cue that closed the last one"
    );
}
