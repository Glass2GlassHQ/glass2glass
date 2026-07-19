//! Localhost P2P data-channel loopback for the native str0m data-channel
//! elements ([`WebRtcDataSrc`] / [`WebRtcDataSink`]). Two independent channels
//! carry application bytes in opposite directions: each side sends a fixed set
//! of messages (including one 60 KiB payload that SCTP must fragment across many
//! DATA chunks and reassemble), and we assert the other side received every
//! message intact and in order.
//!
//! Runs fully on localhost UDP with synthesized payloads (no fixture, no media
//! server), so it is a default CI gate under the `webrtc` feature. Deterministic:
//! the channel is reliable + ordered, so all messages arrive; the sink sends an
//! end-of-stream marker after the data, which terminates the receiving source.

#![cfg(all(target_os = "linux", feature = "webrtc"))]

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline, SourceLoop};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, ConfigureOutcome, FrameTiming,
    G2gError, MemoryDomain, OutputSink, PipelineClock, PipelinePacket,
};
use g2g_plugins::webrtcdata::{WebRtcDataSink, WebRtcDataSrc, MAX_MESSAGE_SIZE};
use g2g_plugins::webrtcduplex::{SdpChannel, SignalRole};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    }
}

/// Deterministic message set for one side. Content is a per-message pattern so a
/// received copy can be compared byte-for-byte. Sizes mix sub-MTU, several-MTU,
/// and one 60 KiB payload (proving multi-fragment SCTP reassembly, just under the
/// 64 KiB max). `salt` distinguishes the two directions.
fn messages(salt: u8) -> Vec<Vec<u8>> {
    let sizes = [1usize, 10, 1500, 100, 5000, 60_000, 42];
    sizes
        .iter()
        .enumerate()
        .map(|(k, &len)| {
            (0..len)
                .map(|j| (((j as u32 + (k as u32) * 7 + salt as u32) % 251) as u8) + 1)
                .collect::<Vec<u8>>()
        })
        .collect()
}

/// Source that emits a fixed set of `messages` as `ByteStream` frames, then EOS.
struct PayloadSrc {
    msgs: Vec<Vec<u8>>,
}

impl SourceLoop for PayloadSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(caps()))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(caps())).await?;
            let mut seq = 0u64;
            for msg in &self.msgs {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(msg.clone().into_boxed_slice())),
                    FrameTiming {
                        pts_ns: seq,
                        ..FrameTiming::default()
                    },
                    seq,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
                seq += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

/// Sink that records each received message's bytes into a shared vec.
struct CollectingSink {
    got: Arc<Mutex<Vec<Vec<u8>>>>,
}

impl AsyncElement for CollectingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                if let MemoryDomain::System(slice) = &frame.domain {
                    self.got.lock().unwrap().push(slice.as_slice().to_vec());
                }
            }
            Ok(())
        })
    }
}

/// Drive one direction: `PayloadSrc -> WebRtcDataSink(offerer)` on the send peer,
/// `WebRtcDataSrc(answerer) -> CollectingSink` on the recv peer, over one SDP
/// channel pair. Returns what the recv side collected.
async fn run_one_direction(salt: u8) -> Vec<Vec<u8>> {
    let (off_chan, ans_chan) = SdpChannel::pair();
    let msgs = messages(salt);
    let got = Arc::new(Mutex::new(Vec::new()));

    let sender = {
        let msgs = msgs.clone();
        async move {
            let mut src = PayloadSrc { msgs };
            let mut sink = WebRtcDataSink::new(SignalRole::Offerer, off_chan)
                .with_linger(Duration::from_millis(500));
            let clock = ZeroClock;
            run_simple_pipeline(&mut src, &mut sink, &clock, 8).await
        }
    };
    let receiver = {
        let got = got.clone();
        async move {
            let mut src = WebRtcDataSrc::new(SignalRole::Answerer, ans_chan);
            let mut sink = CollectingSink { got };
            let clock = ZeroClock;
            run_simple_pipeline(&mut src, &mut sink, &clock, 8).await
        }
    };

    let (s, r) = tokio::time::timeout(Duration::from_secs(30), async {
        tokio::join!(sender, receiver)
    })
    .await
    .expect("data-channel loopback completes in time");
    s.expect("sender pipeline ok");
    r.expect("receiver pipeline ok");

    Arc::try_unwrap(got).unwrap().into_inner().unwrap()
}

#[tokio::test]
async fn data_channel_delivers_messages_intact_and_ordered() {
    let expected = messages(1);
    let got = run_one_direction(1).await;
    assert_eq!(
        got.len(),
        expected.len(),
        "received {} messages, expected {}",
        got.len(),
        expected.len()
    );
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.len(),
            e.len(),
            "message {i} length {} != expected {}",
            g.len(),
            e.len()
        );
        assert!(g == e, "message {i} content differs");
    }
    // The 60 KiB payload (index 5) round-tripped, proving SCTP fragmentation +
    // reassembly under the 64 KiB message cap.
    assert!(expected[5].len() < MAX_MESSAGE_SIZE);
    assert_eq!(got[5].len(), 60_000);
    eprintln!(
        "data channel: received {} messages, largest {} bytes",
        got.len(),
        got.iter().map(|m| m.len()).max().unwrap()
    );
}

#[tokio::test]
async fn data_channel_is_bidirectional() {
    // Two independent channels in opposite directions; each side's messages
    // arrive on the other, intact and in order.
    let (a_to_b, b_to_a) = tokio::join!(run_one_direction(1), run_one_direction(2));
    assert_eq!(a_to_b, messages(1), "A->B mismatch");
    assert_eq!(b_to_a, messages(2), "B->A mismatch");
    eprintln!(
        "bidirectional: A->B {} msgs, B->A {} msgs",
        a_to_b.len(),
        b_to_a.len()
    );
}
