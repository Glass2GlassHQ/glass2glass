#![no_main]
// ULPFEC recovery over untrusted RTP: feed length-prefixed (u16 BE) packets to
// the FEC decoder, alternating media / repair, so protected-sequence parsing and
// single-loss recovery run over attacker bytes. Also drives the raw recover fn.
use libfuzzer_sys::fuzz_target;
use g2g_plugins::ulpfec::{recover_packet, FecDecoder};

fuzz_target!(|data: &[u8]| {
    let mut dec = FecDecoder::new(64);
    let mut i = 0;
    let mut seq: u16 = 0;
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
        let _ = dec.take_recovered();
        seq = seq.wrapping_add(1);
        i = end;
    }
    let _ = recover_packet(data, &[]);
});
