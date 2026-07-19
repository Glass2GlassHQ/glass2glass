#![no_main]
// ST 2110-40 / SMPTE ST 291 ancillary-data depacketization (RFC 8331) over RTP:
// hand-written MSB-first 10-bit-word bit reader + parity / checksum over an
// attacker-controlled datagram.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = g2g_plugins::st2110anc::St2110AncDepacketizer::new().depacketize(data);
});
