#![no_main]
// RTP retransmission (RFC 4588) packet parsing: header offset walk (CSRC + one-
// byte extension), RTX unwrap (OSN strip), and re-wrap over attacker bytes.
use libfuzzer_sys::fuzz_target;
use g2g_plugins::rtx::{build_rtx_packet, parse_rtx_packet, rtp_payload_offset};

fuzz_target!(|data: &[u8]| {
    let _ = rtp_payload_offset(data);
    let _ = parse_rtx_packet(data, 96, 0x1234_5678);
    let _ = build_rtx_packet(data, 97, 0x8765_4321, 42);
});
