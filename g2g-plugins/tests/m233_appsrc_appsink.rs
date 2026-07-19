//! M233 appsrc / appsink: the application feeds buffers into a pipeline and
//! receives them at the other end, by name through the launch DSL. Pre-fills the
//! feed before running so the whole thing drives on one thread (no background
//! pusher needed); the C ABI test covers the cross-thread path.
//!
//! `default_registry` is `std`-gated, so this file is too.
#![cfg(feature = "std")]

use std::ffi::c_void;
use std::sync::Mutex;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::appsink::set_appsink_callback;
use g2g_plugins::appsrc::register_appsrc;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[derive(Default)]
struct Rec {
    frames: Vec<(Vec<u8>, u64)>,
    eos: bool,
}

extern "C" fn record(data: *const u8, len: usize, pts_ns: u64, user: *mut c_void) {
    // SAFETY: `user` is the &Mutex<Rec> registered below, alive for the run.
    let rec = unsafe { &*(user as *const Mutex<Rec>) };
    let mut g = rec.lock().unwrap();
    if data.is_null() && len == 0 {
        g.eos = true;
        return;
    }
    // SAFETY: appsink passes `len` readable bytes for the call.
    let bytes = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
    g.frames.push((bytes, pts_ns));
}

#[tokio::test]
async fn appsrc_feeds_appsink_through_the_dsl() {
    let rec = Box::new(Mutex::new(Rec::default()));
    let user = (&*rec as *const Mutex<Rec>) as *mut c_void;
    set_appsink_callback("m233out", record, user);

    // Pre-fill the feed: three 2x2 RGBA buffers then EOS, all buffered, so the
    // source consumes them without a concurrent producer.
    let feed = register_appsrc("m233in");
    for i in 0u8..3 {
        assert!(feed.push(&[i; 16], u64::from(i) * 1_000));
    }
    feed.end_of_stream();

    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "appsrc channel=m233in caps=video/x-raw,format=RGBA,width=2,height=2,framerate=30/1 \
         ! appsink channel=m233out",
    )
    .expect("appsrc ! appsink parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("pipeline runs");

    assert_eq!(
        stats.frames_emitted, 3,
        "appsrc emitted every pushed buffer"
    );
    assert_eq!(stats.frames_consumed, 3, "appsink consumed them");

    let g = rec.lock().unwrap();
    assert_eq!(g.frames.len(), 3);
    assert_eq!(g.frames[0].0, vec![0u8; 16], "bytes round-tripped");
    assert_eq!(g.frames[1].1, 1_000, "pts carried through");
    assert!(g.eos, "appsink saw EOS");
}
