// RTP jitter buffer over untrusted packets: length-prefixed (u16 BE) datagrams
// pushed with a monotonic clock, interleaved with pops / gap queries, so RTP
// header parsing and the reorder / deadline bookkeeping run over attacker bytes.
#![no_main]
use libfuzzer_sys::fuzz_target;
use g2g_plugins::rtpjitter::{JitterConfig, RtpJitterBuffer};

fuzz_target!(|data: &[u8]| {
    let mut jb = RtpJitterBuffer::new(JitterConfig::new(50, 128));
    let mut i = 0;
    let mut now: u64 = 0;
    while i + 2 <= data.len() {
        let len = u16::from_be_bytes([data[i], data[i + 1]]) as usize;
        i += 2;
        let end = (i + len).min(data.len());
        jb.push(&data[i..end], now);
        let _ = jb.pop(now);
        let _ = jb.missing_seqs();
        let _ = jb.next_deadline_ns(now);
        now = now.wrapping_add(1_000_000);
        i = end;
    }
});
