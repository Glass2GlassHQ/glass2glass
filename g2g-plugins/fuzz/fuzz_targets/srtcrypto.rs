#![no_main]
// SRT keying-material message parsing: the KM control message a peer sends to
// establish the stream encryption key (header layout, wrapped-key length, salt),
// unwrapped under a fixed passphrase.
use libfuzzer_sys::fuzz_target;
use g2g_plugins::srtcrypto::SrtCrypto;

fuzz_target!(|data: &[u8]| {
    let _ = SrtCrypto::km_kk(data);
    let _ = SrtCrypto::from_km(data, "fuzz-passphrase");
});
