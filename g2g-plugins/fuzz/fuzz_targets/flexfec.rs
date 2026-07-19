#![no_main]
// FlexFEC recovery over untrusted packets: the stateless recover_packet plus the
// stateful decoder fed length-prefixed packets (seq parity picks media vs fec).
use libfuzzer_sys::fuzz_target;
use g2g_plugins::flexfec::{recover_packet, FlexFecDecoder};

fuzz_target!(|data: &[u8]| {
    let _ = recover_packet(data, &[]);
    let mut dec = FlexFecDecoder::new(64);
    let mut i = 0;
    let mut seq = 0u16;
    while i + 2 <= data.len() {
        let len = u16::from_be_bytes([data[i], data[i + 1]]) as usize;
        i += 2;
        let end = (i + len).min(data.len());
        let pkt = &data[i..end];
        if seq & 1 == 0 {
            dec.push_media(seq, pkt);
        } else {
            dec.push_fec(pkt);
        }
        seq = seq.wrapping_add(1);
        i = end;
    }
});
