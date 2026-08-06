//! M915: interop against a public MoQ Transport relay.
//!
//! `mp4mux ! moqtsink` -> a relay on the public internet -> `moqtsrc`, the same
//! round trip `m903` runs against a locally spawned `moq-relay-ietf`, but
//! against an implementation that is not Cloudflare's: Meta's moxygen
//! (`https://fb.mvfst.net:9448/moq-relay`, which advertises drafts 14, 16 and
//! 18).
//!
//! Opt-in, because it reaches the internet: set `G2G_MOQT_PUBLIC_RELAY=1` to
//! run it, and `G2G_MOQT_PUBLIC_URL` to point it somewhere else. Without the
//! variable it prints a line saying so and passes.
#![cfg(feature = "moqt")]

use std::time::{Duration, Instant};

use g2g_core::element::AsyncElement;
use g2g_core::frame::PipelinePacket;
use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, PropValue};

use g2g_plugins::moqtsink::MoqtSink;
use g2g_plugins::moqtsrc::MoqtSrc;

mod moqt_common;
use moqt_common::{assert_ordered_fragments, CaptureSink, NullOut, VideoMuxer};

/// The relay to probe. Its certificate is publicly trusted, so no hashes.
const DEFAULT_URL: &str = "https://fb.mvfst.net:9448/moq-relay";

/// Frames the subscriber asks for: the init segment plus a few fragments.
const FRAMES_WANTED: u64 = 6;

fn bmff_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    }
}

/// Publish into the relay and read the broadcast back out of it, on the draft
/// the `versions` offer selects.
async fn round_trip(url: &str, versions: &str) {
    // A namespace nobody else on a public relay is using.
    let namespace = format!(
        "g2g-probe-{}-{versions}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_millis()
    );

    let mut sink = MoqtSink::new(url, &namespace);
    sink.set_property("versions", PropValue::Str(String::from(versions)))
        .expect("versions");
    sink.configure_pipeline(&bmff_caps())
        .expect("moqtsink caps");

    let mut src = MoqtSrc::new(url, &namespace).with_num_buffers(FRAMES_WANTED);
    SourceLoop::set_property(&mut src, "versions", PropValue::Str(String::from(versions)))
        .expect("versions");
    src.configure_pipeline(&bmff_caps()).expect("moqtsrc caps");

    // The publisher keeps going until the subscriber has what it asked for; a
    // relay only forwards what arrives after the subscription is established.
    let done = std::cell::Cell::new(false);
    let publish = async {
        let mut mux = VideoMuxer::new(0);
        let mut published = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(40);
        while !done.get() && Instant::now() < deadline {
            published.extend(mux.step(&mut sink).await);
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        sink.process(PipelinePacket::Eos, &mut NullOut)
            .await
            .expect("clean end of stream");
        published
    };

    let mut captured = CaptureSink::default();
    let subscribe = async {
        tokio::time::sleep(Duration::from_millis(2000)).await;
        let emitted = src.run(&mut captured).await;
        done.set(true);
        emitted
    };

    let (published, emitted) = tokio::join!(publish, subscribe);
    let emitted = emitted.unwrap_or_else(|e| panic!("{versions}: subscribe and play: {e:?}"));
    assert_eq!(emitted, FRAMES_WANTED, "{versions}");
    assert_ordered_fragments(&captured.frames, &published);
    eprintln!("{url} draft {versions}: {emitted} frames round-tripped");
}

#[tokio::test]
async fn a_public_relay_round_trips_the_broadcast() {
    if std::env::var("G2G_MOQT_PUBLIC_RELAY").is_err() {
        eprintln!(
            "SKIP: set G2G_MOQT_PUBLIC_RELAY=1 to probe {DEFAULT_URL} (this test reaches the \
             public internet); G2G_MOQT_PUBLIC_URL points it elsewhere."
        );
        return;
    }
    let url = std::env::var("G2G_MOQT_PUBLIC_URL").unwrap_or_else(|_| String::from(DEFAULT_URL));
    // The default offer first, then draft-18 alone, so the report says which
    // drafts the relay actually served.
    round_trip(&url, "18,16").await;
    round_trip(&url, "18").await;
}
