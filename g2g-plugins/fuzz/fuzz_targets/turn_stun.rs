#![no_main]
// TURN / STUN message parsing over untrusted UDP: ChannelData + DATA-INDICATION
// framing and the XOR-PEER / XOR-MAPPED-ADDRESS + ERROR-CODE attribute walk the
// WebRTC relay data plane runs on inbound datagrams (hand-rolled, RFC 5766/8489).
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    g2g_plugins::turn_fuzz_parse(data);
    g2g_plugins::stun_fuzz_parse(data);
});
