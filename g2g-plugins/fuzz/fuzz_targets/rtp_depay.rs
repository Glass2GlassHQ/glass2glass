#![no_main]
// H.264 RTP depayloader over untrusted packets. The input is split into
// length-prefixed packets (u16 BE length) so cross-packet FU-A / STAP-A
// reassembly is exercised, not just single-packet header parsing.
use libfuzzer_sys::fuzz_target;
use g2g_plugins::rtpdepay::RtpH264Depayloader;

fuzz_target!(|data: &[u8]| {
    let mut d = RtpH264Depayloader::new();
    let mut i = 0;
    while i + 2 <= data.len() {
        let len = u16::from_be_bytes([data[i], data[i + 1]]) as usize;
        i += 2;
        let end = (i + len).min(data.len());
        let _ = d.depacketize(&data[i..end]);
        i = end;
    }
});
