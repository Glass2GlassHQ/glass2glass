//! M653: receive direction on-MCU. The RX chain, `RtpSrc -> JitterBuffer ->
//! G.711 decode`, reconstructs an ordered media stream from a jumbled RTP wire
//! (reorder, duplicate, loss) on mock peripherals. The jitter buffer's reorder /
//! dedup / loss logic and the RTP parse + lend are the real units; only the
//! network receiver is a mock.

use std::cell::RefCell;
use std::future::Future;
use std::rc::Rc;

use g2g_core::error::G2gError;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::mediaclock::MediaClock;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::rtp::RtpHeader;
use g2g_core::staticelem::Chain;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{run_source_transform_sink, StaticSink, StaticTransform};
use g2g_mcu::{G711Dec, JitterBuffer, Law, PacketReceiver, RtpSrc};

fn block_on<F: Future>(fut: F) -> F::Output {
    g2g_core::drive_ready(fut).expect("the static chain never suspends")
}

fn leaked_ring<const N: usize, const B: usize>() -> &'static StaticLendRing<N, B> {
    Box::leak(Box::new(StaticLendRing::new()))
}

/// Build an input frame carrying `payload` at RTP sequence `seq` (the shape
/// `RtpSrc` produces), lending a leaked buffer (host-test convenience).
fn frame(seq: u64, payload: &[u8]) -> Frame {
    let leaked: &'static [u8] = Box::leak(payload.to_vec().into_boxed_slice());
    // SAFETY: the leaked buffer is 'static and never mutated; the lent slice
    // covers exactly its bytes and needs no reclamation (free = None).
    let slice = unsafe {
        SystemSlice::from_foreign(leaked.as_ptr(), leaked.len(), None, core::ptr::null_mut())
    };
    Frame::new(
        MemoryDomain::System(slice),
        FrameTiming {
            pts_ns: seq,
            ..FrameTiming::default()
        },
        seq,
    )
}

/// Push one packet into the jitter buffer and return the emitted sequence, if any.
fn push<const N: usize, const B: usize>(
    jb: &mut JitterBuffer<N, B>,
    seq: u64,
    payload: &[u8],
) -> Option<u64> {
    block_on(jb.process(frame(seq, payload)))
        .expect("jitter process")
        .map(|f| f.sequence)
}

#[test]
fn in_order_stream_plays_out_after_a_prime_latency() {
    let mut jb: JitterBuffer<8, 2> = JitterBuffer::new(3);
    let mut out = Vec::new();
    for seq in 0u64..8 {
        if let Some(s) = push(&mut jb, seq, &[seq as u8, 0]) {
            out.push(s);
        }
    }
    // Depth 3: the first two pushes buffer (jitter cushion), then playout tracks
    // the input in order. All emitted sequences are contiguous and ascending.
    assert_eq!(
        out,
        vec![0, 1, 2, 3, 4, 5],
        "in-order playout, two-frame prime latency"
    );
    assert_eq!(jb.lost(), 0);
    assert_eq!(jb.reordered(), 0);
    assert_eq!(
        jb.buffered(),
        2,
        "depth-1 packets remain buffered at the tail"
    );
}

#[test]
fn reordered_packets_are_played_back_in_sequence() {
    let mut jb: JitterBuffer<8, 2> = JitterBuffer::new(3);
    // Swap 3<->4 and 6<->7 on the wire; the buffer must reorder them.
    let wire = [0u64, 1, 2, 4, 3, 5, 7, 6, 8, 9];
    let mut out = Vec::new();
    for &seq in &wire {
        if let Some(s) = push(&mut jb, seq, &[seq as u8, 0]) {
            out.push(s);
        }
    }
    assert_eq!(
        out,
        vec![0, 1, 2, 3, 4, 5, 6, 7],
        "reordered pairs emitted in order"
    );
    assert_eq!(jb.lost(), 0, "reorder within the window is not loss");
    assert!(
        jb.reordered() >= 2,
        "the two late arrivals were counted as reordered"
    );
}

#[test]
fn duplicate_packets_are_dropped() {
    let mut jb: JitterBuffer<8, 2> = JitterBuffer::new(2);
    let wire = [0u64, 1, 2, 2, 3, 4, 5]; // seq 2 duplicated
    let mut out = Vec::new();
    for &seq in &wire {
        if let Some(s) = push(&mut jb, seq, &[seq as u8, 0]) {
            out.push(s);
        }
    }
    assert_eq!(jb.duplicates(), 1, "the repeated sequence was dropped once");
    // Output stays a contiguous ascending run despite the duplicate.
    let mut prev = None;
    for &s in &out {
        if let Some(p) = prev {
            assert_eq!(s, p + 1, "no duplicate reached playout");
        }
        prev = Some(s);
    }
}

#[test]
fn late_packet_below_the_play_cursor_is_dropped() {
    let mut jb: JitterBuffer<8, 2> = JitterBuffer::new(2);
    // Prime and play a few, then a straggler for an already-played sequence.
    for seq in 0u64..5 {
        push(&mut jb, seq, &[seq as u8, 0]);
    }
    // seq 0 has long since played; it is dropped as late and counted. (The
    // process tick may still flush a legitimately-pending head, but the late
    // straggler itself never reaches playout.)
    let emitted = push(&mut jb, 0, &[0, 0]);
    assert_ne!(emitted, Some(0), "the late straggler never plays out");
    assert!(jb.late() >= 1, "the straggler was counted late");
}

#[test]
fn a_lost_packet_yields_a_gap_and_does_not_stall() {
    let mut jb: JitterBuffer<8, 2> = JitterBuffer::new(3);
    // Sequence 4 never arrives; the stream must continue past the gap.
    let wire = [0u64, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12];
    let mut out = Vec::new();
    for &seq in &wire {
        if let Some(s) = push(&mut jb, seq, &[seq as u8, 0]) {
            out.push(s);
        }
    }
    assert!(!out.contains(&4), "the lost sequence 4 never played");
    assert_eq!(jb.lost(), 1, "exactly one loss declared");
    // Everything after the gap is still delivered in order.
    let after: Vec<u64> = out.iter().copied().filter(|&s| s > 4).collect();
    let mut prev = 4;
    for s in after {
        assert!(s == prev + 1, "contiguous playout resumes after the gap");
        prev = s;
    }
}

// --- RX flagship: RtpSrc -> JitterBuffer -> G.711 decode over a jumbled wire ---

const SP: usize = 8; // samples per packet
const NFRAMES: u64 = 16;
const DEPTH: u32 = 3;

/// A tone-ish PCM frame for sequence `k` (a per-packet ramp).
fn pcm_frame(k: u64) -> [i16; SP] {
    let mut f = [0i16; SP];
    for (i, s) in f.iter_mut().enumerate() {
        *s = (((k.wrapping_add(i as u64) & 0x3f) as i32 - 32) * 200) as i16;
    }
    f
}

/// The mu-law payload for packet `k` (what a G.711 sender puts on the wire).
fn mulaw_payload(k: u64) -> Vec<u8> {
    pcm_frame(k).iter().map(|&s| Law::Mulaw.encode(s)).collect()
}

/// One RTP/PCMU datagram for sequence `k` (the shape `RtpSink` emits, whose wire
/// is separately ffmpeg-validated in M643).
fn rtp_datagram(k: u64) -> Vec<u8> {
    let header = RtpHeader {
        payload_type: 0, // PCMU
        marker: k == 0,
        sequence: k as u16,
        timestamp: (k * SP as u64) as u32, // 8 samples/packet at the sample clock
        ssrc: 0xDEAD_BEEF,
    };
    let mut d = header.to_bytes().to_vec();
    d.extend_from_slice(&mulaw_payload(k));
    d
}

/// A mock network receiver that hands out a queued list of datagrams in order,
/// erroring when drained (the RX source stops via its frame limit first).
struct QueuedReceiver {
    datagrams: Rc<RefCell<std::collections::VecDeque<Vec<u8>>>>,
}
impl PacketReceiver for QueuedReceiver {
    async fn recv(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        let mut q = self.datagrams.borrow_mut();
        let Some(d) = q.pop_front() else {
            return Err(G2gError::Shutdown);
        };
        let n = d.len().min(buf.len());
        buf[..n].copy_from_slice(&d[..n]);
        Ok(n)
    }
}

/// Collects decoded PCM bytes, in playout order.
struct PcmCollector {
    bytes: Vec<u8>,
    frames: u32,
}
impl StaticSink for PcmCollector {
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
        if let MemoryDomain::System(s) = &frame.domain {
            self.bytes.extend_from_slice(s.as_slice());
        }
        self.frames += 1;
        Ok(())
    }
}

struct JbStats {
    lost: u32,
    reordered: u32,
    duplicates: u32,
}

fn run_rx(wire: Vec<Vec<u8>>) -> (Vec<u8>, u32, JbStats) {
    let accepted = wire.len() as u32;
    let q = Rc::new(RefCell::new(
        wire.into_iter().collect::<std::collections::VecDeque<_>>(),
    ));
    let recv = QueuedReceiver { datagrams: q };

    let rtp_ring: &'static StaticLendRing<4, SP> = leaked_ring(); // mu-law payload
    let dec_ring: &'static StaticLendRing<4, { SP * 2 }> = leaked_ring(); // S16 PCM
    let src = RtpSrc::<_, 4, SP, { 12 + SP }>::new(recv, rtp_ring, MediaClock::audio(8000))
        .with_payload_type(0)
        .with_frame_limit(accepted);
    // `&mut jb` keeps the buffer so its reception counters can be read after.
    let mut jb: JitterBuffer<8, SP> = JitterBuffer::new(DEPTH);
    let dec = G711Dec::new(Law::Mulaw, dec_ring);
    let mut sink = PcmCollector {
        bytes: Vec::new(),
        frames: 0,
    };
    block_on(run_source_transform_sink(
        src,
        Chain(&mut jb, dec),
        &mut sink,
    ))
    .expect("rx pipeline");
    let stats = JbStats {
        lost: jb.lost(),
        reordered: jb.reordered(),
        duplicates: jb.duplicates(),
    };
    (sink.bytes, sink.frames, stats)
}

/// The decoded PCM the ordered, lossless stream produces for sequences `0..upto`.
fn reference_pcm(upto: u64) -> Vec<u8> {
    let mut out = Vec::new();
    for k in 0..upto {
        for &m in &mulaw_payload(k) {
            out.extend_from_slice(&Law::Mulaw.decode(m).to_le_bytes());
        }
    }
    out
}

#[test]
fn rx_flagship_reconstructs_pcm_from_a_reordered_wire() {
    // Build the ordered wire, then perturb it by swapping adjacent pairs (a
    // reorder distance the depth-3 buffer tolerates). No loss, so playout must
    // reproduce the ordered decode exactly, minus the tail still buffered at EOS
    // (depth - 1 packets).
    let mut wire: Vec<Vec<u8>> = (0..NFRAMES).map(rtp_datagram).collect();
    wire.swap(4, 5);
    wire.swap(9, 10);

    let (pcm, frames, stats) = run_rx(wire);
    let delivered = NFRAMES - (DEPTH as u64 - 1);
    assert_eq!(
        frames as u64, delivered,
        "all but the depth-1 buffered tail played out"
    );
    assert_eq!(stats.lost, 0, "reorder within the window is not loss");
    assert!(
        stats.reordered >= 2,
        "the two swapped pairs were counted reordered"
    );
    assert_eq!(
        pcm,
        reference_pcm(delivered),
        "reordered wire decodes to the ordered stream"
    );
}

#[test]
fn rx_flagship_dedups_a_duplicated_wire() {
    // A duplicate packet must be dropped and must not corrupt the decode.
    let mut wire: Vec<Vec<u8>> = (0..NFRAMES).map(rtp_datagram).collect();
    wire.insert(3, rtp_datagram(2)); // duplicate seq 2 shortly after the original

    let (pcm, frames, stats) = run_rx(wire);
    assert_eq!(
        stats.duplicates, 1,
        "the duplicate was dropped exactly once"
    );
    assert_eq!(stats.lost, 0);
    // Whatever count played out, it is the ordered decode of that many leading
    // frames (the duplicate neither corrupted nor reordered the stream).
    assert_eq!(
        pcm,
        reference_pcm(frames as u64),
        "dedup leaves the ordered stream intact"
    );
}

#[test]
fn rx_flagship_survives_packet_loss() {
    // Drop sequence 8 from the wire entirely; the stream must continue and the
    // decoded PCM must match the ordered stream with that one frame missing.
    let wire: Vec<Vec<u8>> = (0..NFRAMES).filter(|&k| k != 8).map(rtp_datagram).collect();
    let (pcm, frames, stats) = run_rx(wire);
    assert_eq!(
        stats.lost, 1,
        "exactly the one dropped packet was declared lost"
    );
    assert!(frames >= 1, "playout continued past the loss");
    // The decoded stream is the ordered decode with frame 8 excised (and the
    // buffered tail withheld). Reconstruct that expectation.
    let mut expected = Vec::new();
    let mut played = 0u64;
    for k in 0..NFRAMES {
        if k == 8 {
            continue; // lost, no PCM
        }
        // Stop where the real run stops (its buffered tail is withheld).
        if played >= frames as u64 {
            break;
        }
        for &m in &mulaw_payload(k) {
            expected.extend_from_slice(&Law::Mulaw.decode(m).to_le_bytes());
        }
        played += 1;
    }
    assert_eq!(
        pcm, expected,
        "loss leaves a clean gap; surrounding frames intact and in order"
    );
}
