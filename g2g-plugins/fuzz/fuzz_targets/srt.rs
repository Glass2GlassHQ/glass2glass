#![no_main]
// SRT (Secure Reliable Transport) packet parsing: control packets (handshake CIF,
// NAK range list, KM) and data-packet headers over an attacker-controlled UDP
// datagram.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = g2g_plugins::srt::parse_control(data);
    let _ = g2g_plugins::srt::parse_data_packet(data);
});
