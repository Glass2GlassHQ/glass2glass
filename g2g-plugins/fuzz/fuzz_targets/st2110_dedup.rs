#![no_main]
// ST 2110-7 seamless dedup: parses the RTP header of each redundant packet.
use libfuzzer_sys::fuzz_target;
use g2g_plugins::st2110dup::SeamlessDedup;

fuzz_target!(|data: &[u8]| {
    let mut d = SeamlessDedup::new();
    let mut i = 0;
    while i + 2 <= data.len() {
        let len = u16::from_be_bytes([data[i], data[i + 1]]) as usize;
        i += 2;
        let end = (i + len).min(data.len());
        let _ = d.accept(&data[i..end]);
        i = end;
    }
});
