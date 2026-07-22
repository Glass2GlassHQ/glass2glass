//! Receive-direction proof (M653): the RX chain on the emulated Cortex-M4. A
//! mock network receiver hands the pipeline a *reordered* stream of RTP/PCMU
//! datagrams; `RtpSrc -> JitterBuffer -> G.711 decode -> hash` must reconstruct
//! the ordered PCM. Correctness is proved by equivalence to an independent
//! ordered decode, using an ORDER-SENSITIVE rolling hash (a plain checksum sum
//! is order-insensitive and would not catch a reordering bug): the pipeline's
//! playout-order hash must equal the reference's in-order hash, which holds only
//! if the jitter buffer put every packet back in sequence.
//!
//! `tools/qemu-check.sh` boots this and asserts the banner + exit, upgrading the
//! host-side RX tests to "runs on the Cortex-M ISA".

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hio};

use g2g_core::error::G2gError;
use g2g_core::mediaclock::MediaClock;
use g2g_core::rtp::RtpHeader;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{drive_ready, run_source_transform_sink, Chain, Frame, MemoryDomain, StaticSink};
use g2g_mcu::{G711Dec, JitterBuffer, Law, PacketReceiver, RtpSrc};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// Samples (and mu-law payload bytes) per packet.
const SP: usize = 8;
/// Number of distinct RTP packets on the wire.
const NF: usize = 16;
/// Jitter-buffer prime depth; depth - 1 packets remain buffered at end of stream.
const DEPTH: u32 = 3;
/// Delivered frames = distinct - (depth - 1).
const DELIVERED: u64 = NF as u64 - (DEPTH as u64 - 1);

/// The wire arrival order: adjacent-pair swaps, a reorder distance the depth-3
/// buffer tolerates, so no packet is lost and playout is the ordered stream.
const ORDER: [u16; NF] = [0, 1, 2, 4, 3, 5, 6, 8, 7, 9, 10, 11, 13, 12, 14, 15];

/// Deterministic PCM for packet `seq` (a per-packet ramp).
fn pcm_sample(seq: u64, i: usize) -> i16 {
    (((seq.wrapping_add(i as u64) & 0x3f) as i32 - 32) * 200) as i16
}

/// Order-sensitive rolling hash: reordering the byte stream changes the result.
fn roll(acc: u64, byte: u8) -> u64 {
    acc.wrapping_mul(1_000_003).wrapping_add(byte as u64)
}

/// A mock receiver that emits the `ORDER`-permuted RTP/PCMU datagrams, building
/// each one on demand into the caller's buffer (no allocation).
struct ShuffledReceiver {
    idx: usize,
}
impl PacketReceiver for ShuffledReceiver {
    async fn recv(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        let Some(&seq) = ORDER.get(self.idx) else {
            return Err(G2gError::Shutdown); // drained; the source's frame limit stops first
        };
        self.idx += 1;
        let header = RtpHeader {
            payload_type: 0, // PCMU
            marker: seq == 0,
            sequence: seq,
            timestamp: (seq as u32).wrapping_mul(SP as u32),
            ssrc: 0x0102_0304,
        };
        let hdr = header.to_bytes();
        let total = hdr.len() + SP;
        if buf.len() < total {
            return Err(G2gError::CapsMismatch);
        }
        if let Some(dst) = buf.get_mut(..hdr.len()) {
            dst.copy_from_slice(&hdr);
        }
        for i in 0..SP {
            let m = Law::Mulaw.encode(pcm_sample(seq as u64, i));
            if let Some(cell) = buf.get_mut(hdr.len() + i) {
                *cell = m;
            }
        }
        Ok(total)
    }
}

/// Hashes decoded PCM bytes in playout order and counts frames.
struct HashSink {
    acc: u64,
    count: u32,
}
impl StaticSink for HashSink {
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
        if let Some(s) = frame.domain.as_system_slice() {
            for &b in s.as_slice() {
                self.acc = roll(self.acc, b);
            }
        }
        self.count = self.count.wrapping_add(1);
        Ok(())
    }
}

/// The reference: the in-order decode of the first `DELIVERED` frames, hashed
/// the same way, computed directly (no pipeline).
fn reference() -> u64 {
    let mut acc = 0u64;
    for seq in 0..DELIVERED {
        for i in 0..SP {
            let m = Law::Mulaw.encode(pcm_sample(seq, i));
            let pcm = Law::Mulaw.decode(m);
            for &b in &pcm.to_le_bytes() {
                acc = roll(acc, b);
            }
        }
    }
    acc
}

#[entry]
fn main() -> ! {
    let want = reference();

    let rtp_ring: StaticLendRing<4, SP> = StaticLendRing::new();
    let dec_ring: StaticLendRing<4, { SP * 2 }> = StaticLendRing::new();
    // SAFETY: both rings outlive the pipeline run below (dropped after it ends).
    let src = unsafe {
        RtpSrc::<_, 4, SP, { 12 + SP }>::with_ring(
            ShuffledReceiver { idx: 0 },
            &rtp_ring,
            MediaClock::audio(8000),
        )
    }
    .with_payload_type(0)
    .with_frame_limit(NF as u32);
    let mut jb: JitterBuffer<8, SP> = JitterBuffer::new(DEPTH);
    // SAFETY: as above.
    let dec = unsafe { G711Dec::with_ring(Law::Mulaw, &dec_ring) };
    let mut sink = HashSink { acc: 0, count: 0 };

    let _ = drive_ready(run_source_transform_sink(src, Chain(&mut jb, dec), &mut sink));

    let ok = sink.count as u64 == DELIVERED && sink.acc == want && jb.lost() == 0 && jb.reordered() >= 2;

    if let Ok(mut out) = hio::hstdout() {
        let mut line = [0u8; 96];
        let mut pos = 0;
        put_str(&mut line, &mut pos, "g2g-rx: played=");
        put_u32(&mut line, &mut pos, sink.count);
        put_str(&mut line, &mut pos, " reordered=");
        put_u32(&mut line, &mut pos, jb.reordered());
        put_str(&mut line, &mut pos, " lost=");
        put_u32(&mut line, &mut pos, jb.lost());
        put_str(&mut line, &mut pos, if ok { " OK\n" } else { " FAIL\n" });
        let _ = out.write_all(line.get(..pos).unwrap_or(&[]));
    }

    debug::exit(if ok { debug::EXIT_SUCCESS } else { debug::EXIT_FAILURE });
    loop {}
}

/// Append `v` in decimal to `buf` at `pos` (no `core::fmt`).
fn put_u32(buf: &mut [u8], pos: &mut usize, v: u32) {
    let mut digits = [0u8; 10];
    let mut n = 0;
    let mut v = v;
    loop {
        if let Some(d) = digits.get_mut(n) {
            *d = b'0' + (v % 10) as u8;
        }
        n += 1;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    while n > 0 {
        n -= 1;
        if let (Some(dst), Some(&src)) = (buf.get_mut(*pos), digits.get(n)) {
            *dst = src;
            *pos += 1;
        }
    }
}

fn put_str(buf: &mut [u8], pos: &mut usize, s: &str) {
    for &b in s.as_bytes() {
        if let Some(dst) = buf.get_mut(*pos) {
            *dst = b;
            *pos += 1;
        }
    }
}
