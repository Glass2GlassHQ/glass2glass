//! M1042: `moqtsrc`, `moqtsessionsrc` and `webrtcwhepsessionsrc` spell
//! `num-buffers` the way gst `basesrc` does, so -1 runs until the stream ends, n
//! emits exactly n, and 0 emits nothing and goes straight to EOS. They used to
//! take an unsigned count with 0 meaning unlimited, which is the M1040
//! conversion the three were left out of.
//!
//! The run tests need no relay and no SFU: a zero limit ends the source before
//! it connects, which is the point. They are wrapped in a timeout, since the old
//! reading of 0 blocks on the network forever.
//!
//! ```sh
//! cargo test -p g2g-plugins --features moqt,webrtc \
//!     --test m1042_moqt_whep_num_buffers
//! ```

#![cfg(any(feature = "moqt", feature = "webrtc"))]

mod numbuffers_common;

use core::time::Duration;

use g2g_core::PropValue;
use numbuffers_common::assert_num_buffers_round_trips;

/// How long a zero-limit `run` may take before we call it hung. Generous: the
/// correct path does no IO at all.
const ZERO_LIMIT_DEADLINE: Duration = Duration::from_secs(5);

#[cfg(feature = "moqt")]
#[test]
fn moqtsrc_num_buffers_round_trips() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::moqtsrc::MoqtSrc;
    assert_num_buffers_round_trips!(MoqtSrc::new("https://127.0.0.1:4443/", "g2g"));
}

#[cfg(feature = "moqt")]
#[test]
fn moqtsessionsrc_num_buffers_round_trips() {
    use g2g_core::MultiOutputSource;
    use g2g_plugins::moqtsessionsrc::MoqtSessionSrc;
    assert_num_buffers_round_trips!(MoqtSessionSrc::new("https://127.0.0.1:4443/", "g2g"));
}

#[cfg(feature = "webrtc")]
#[test]
fn whepsessionsrc_num_buffers_round_trips() {
    use g2g_core::MultiOutputSource;
    use g2g_plugins::webrtcwhepsession::WebRtcWhepSessionSrc;
    assert_num_buffers_round_trips!(WebRtcWhepSessionSrc::new("https://sfu.invalid/whep"));
}

#[cfg(feature = "moqt")]
#[tokio::test]
async fn moqtsrc_num_buffers_zero_emits_only_eos() {
    use crate::numbuffers_common::{assert_only_eos, Collect};
    use g2g_core::runtime::SourceLoop;
    use g2g_core::{ByteStreamEncoding, Caps};
    use g2g_plugins::moqtsrc::MoqtSrc;

    let mut src = MoqtSrc::new("https://127.0.0.1:4443/", "g2g").with_num_buffers(0);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .expect("moqtsrc caps");

    let mut out = Collect::default();
    let emitted = tokio::time::timeout(ZERO_LIMIT_DEADLINE, src.run(&mut out))
        .await
        .expect("a zero limit must not wait on the relay")
        .expect("run");
    assert_only_eos(&out, emitted);
}

#[cfg(feature = "moqt")]
#[tokio::test]
async fn moqtsessionsrc_num_buffers_zero_emits_only_eos() {
    use crate::numbuffers_common::{assert_only_eos_on_every_pad, CollectPorts};
    use g2g_core::MultiOutputSource;
    use g2g_plugins::moqtsessionsrc::MoqtSessionSrc;

    let mut src = MoqtSessionSrc::new("https://127.0.0.1:4443/", "g2g")
        .with_outputs(2)
        .with_tracks("video.m4s,audio.m4s");
    src.set_property("num-buffers", PropValue::Int(0)).unwrap();

    let mut out = CollectPorts::new(src.output_count());
    let emitted = tokio::time::timeout(ZERO_LIMIT_DEADLINE, src.run(&mut out))
        .await
        .expect("a zero limit must not wait on the relay")
        .expect("run");
    assert_only_eos_on_every_pad(&out, emitted);
}

#[cfg(feature = "webrtc")]
#[tokio::test]
async fn whepsessionsrc_num_buffers_zero_emits_only_eos() {
    use crate::numbuffers_common::{assert_only_eos_on_every_pad, CollectPorts};
    use g2g_core::MultiOutputSource;
    use g2g_plugins::webrtcwhepsession::WebRtcWhepSessionSrc;

    let mut src = WebRtcWhepSessionSrc::new("https://sfu.invalid/whep");
    src.set_property("num-buffers", PropValue::Int(0)).unwrap();

    let mut out = CollectPorts::new(src.output_count());
    let emitted = tokio::time::timeout(ZERO_LIMIT_DEADLINE, src.run(&mut out))
        .await
        .expect("a zero limit must not POST to the SFU")
        .expect("run");
    assert_only_eos_on_every_pad(&out, emitted);
}
