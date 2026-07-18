#![no_main]
// MPEG-TS demux over untrusted input: PAT / PMT / PES parsing. The input is
// sliced into 188-byte transport packets (the fixed TS packet size) and each is
// fed to the demuxer, so PSI section reassembly and PES parsing get exercised.
use libfuzzer_sys::fuzz_target;
use g2g_plugins::mpegts::TsDemuxer;

fuzz_target!(|data: &[u8]| {
    let mut d = TsDemuxer::new();
    for pkt in data.chunks(188) {
        d.push_packet(pkt);
    }
});
