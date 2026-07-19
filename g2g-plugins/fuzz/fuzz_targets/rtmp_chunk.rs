#![no_main]
// RTMP server-side chunk-stream reassembly + AMF0 command parsing: the remote
// surface a malicious publisher reaches after the (uncrypto) handshake. The
// shim forces the post-handshake Streaming state and feeds attacker bytes.
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    g2g_plugins::rtmp::RtmpSession::fuzz_feed_chunk_stream(data);
});
