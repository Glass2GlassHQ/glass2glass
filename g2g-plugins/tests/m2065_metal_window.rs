#[cfg(target_os = "macos")]
fn main() {
    use g2g_core::runtime::{parse_launch, run_graph};
    use g2g_core::PipelineClock;
    use g2g_plugins::metalvideosink::{run_with_appkit_main_loop, MetalVideoSink};
    use g2g_plugins::registry::default_registry;

    struct ZeroClock;
    impl PipelineClock for ZeroClock {
        fn now_ns(&self) -> u64 {
            0
        }
    }

    const FRAMES: u64 = 10;

    if !MetalVideoSink::device_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    run_with_appkit_main_loop(|| {
        let reg = default_registry();
        let line = format!(
            "videotestsrc num-buffers={FRAMES} ! video/x-raw,format=NV12,width=320,height=240 ! metalvideosink"
        );
        let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line:?} parse: {e:?}"));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let stats = runtime
            .block_on(run_graph(graph, &ZeroClock, 4))
            .unwrap_or_else(|e| panic!("{line:?} runs: {e:?}"));
        assert_eq!(stats.frames_consumed, FRAMES, "{line}");
        println!("m2065: {FRAMES} frames presented in an AppKit window");
    });
}

#[cfg(not(target_os = "macos"))]
fn main() {}
