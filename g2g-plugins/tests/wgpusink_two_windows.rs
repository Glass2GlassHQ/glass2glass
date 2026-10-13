#![cfg(all(target_os = "linux", feature = "wgpu-present"))]

use g2g_core::metrics::monotonic_ns;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::gpu::GpuContext;
use g2g_plugins::registry::default_registry;

const SHORT_WINDOW_FRAMES: u32 = 5;
const LONG_WINDOW_FRAMES: u32 = 45;
const LINK_CAPACITY: usize = 2;

struct MonotonicClock;
impl PipelineClock for MonotonicClock {
    fn now_ns(&self) -> u64 {
        monotonic_ns()
    }
}

#[tokio::test]
async fn a_wgpusink_closing_at_eos_leaves_another_presenting() {
    if std::env::var_os("WAYLAND_DISPLAY").is_none() {
        eprintln!("skipping: no WAYLAND_DISPLAY in env (run under a Wayland session)");
        return;
    }
    if let Err(e) = GpuContext::headless().await {
        eprintln!("skipping: no wgpu adapter here ({e:?})");
        return;
    }

    let line = format!(
        "videotestsrc num-buffers={SHORT_WINDOW_FRAMES} ! wgpusink title=g2g-wgpusink-closes-first \
         videotestsrc num-buffers={LONG_WINDOW_FRAMES} ! wgpusink title=g2g-wgpusink-keeps-presenting"
    );
    let registry = default_registry();
    let graph = parse_launch(&registry, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &MonotonicClock, LINK_CAPACITY)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
}
