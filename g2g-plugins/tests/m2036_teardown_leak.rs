//! M2036: building, running and dropping pipelines many times returns the heap,
//! the open file descriptors and the thread count to where they started.
#![cfg(all(
    target_os = "linux",
    feature = "std",
    feature = "multi-thread",
    feature = "udp-egress"
))]

use core::sync::atomic::{AtomicIsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};
use std::net::UdpSocket;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use g2g_core::runtime::{parse_launch, run_graph, run_graph_threaded, Registry};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;
use g2g_plugins::TokioThreadSpawner;

static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);

struct LiveBytes;

// SAFETY: forwards to `System` unchanged, only adding up the sizes first.
unsafe impl GlobalAlloc for LiveBytes {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE_BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: `layout` is the valid layout forwarded from our caller.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: `ptr` / `layout` come from a prior `alloc` of this allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE_BYTES.fetch_add(
            new_size as isize - layout.size() as isize,
            Ordering::Relaxed,
        );
        // SAFETY: `ptr` / `layout` come from a prior `alloc`; `new_size` is the
        // caller's valid request, forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: LiveBytes = LiveBytes;

const LINK_CAPACITY: usize = 4;
const WARMUP_CYCLES: usize = 20;
const MEASURED_CYCLES: usize = 200;
// a leak of eight bytes a cycle exceeds this
const MAX_HEAP_GROWTH_BYTES: isize = 1024;
const TS_PACKET_BYTES: usize = 188;
const INPUT_TS_PACKETS: usize = 256;
// sync byte, then PID 0x1FFF (null packet) with payload only
const TS_NULL_PACKET_HEADER: [u8; 4] = [0x47, 0x1F, 0xFF, 0x10];
// arm threads exit just after the run returns its stats
const THREAD_EXIT_DEADLINE: Duration = Duration::from_secs(5);

struct ZeroClock;

impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m2036-{tag}-{}.bin", std::process::id()))
}

fn open_file_descriptors() -> usize {
    std::fs::read_dir("/proc/self/fd")
        .expect("read /proc/self/fd")
        .count()
}

fn threads() -> usize {
    std::fs::read_dir("/proc/self/task")
        .expect("read /proc/self/task")
        .count()
}

fn threads_once_settled(limit: usize) -> usize {
    let deadline = Instant::now() + THREAD_EXIT_DEADLINE;
    while threads() > limit && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    threads()
}

struct Pipelines {
    cooperative: Vec<String>,
    threaded: String,
}

async fn run_each(registry: &Registry, pipelines: &Pipelines) {
    for line in &pipelines.cooperative {
        let graph = parse_launch(registry, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
        run_graph(graph, &ZeroClock, LINK_CAPACITY)
            .await
            .unwrap_or_else(|e| panic!("`{line}` runs: {e:?}"));
    }
    let line = &pipelines.threaded;
    let graph = parse_launch(registry, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph_threaded(graph, &ZeroClock, LINK_CAPACITY, &TokioThreadSpawner)
        .await
        .unwrap_or_else(|e| panic!("`{line}` runs threaded: {e:?}"));
}

#[tokio::test]
async fn repeated_runs_return_heap_descriptors_and_threads() {
    let input = temp_path("in");
    let output = temp_path("out");
    let mut null_packet = [0xFF; TS_PACKET_BYTES];
    null_packet[..TS_NULL_PACKET_HEADER.len()].copy_from_slice(&TS_NULL_PACKET_HEADER);
    std::fs::write(&input, null_packet.repeat(INPUT_TS_PACKETS)).expect("write input file");
    let receiver = UdpSocket::bind("127.0.0.1:0").expect("bind the udp receiver");
    let port = receiver.local_addr().expect("receiver address").port();

    let tee = "videotestsrc num-buffers=4 ! tee name=t ! fakesink t. ! fakesink".to_string();
    let pipelines = Pipelines {
        cooperative: Vec::from([
            tee.clone(),
            format!(
                "filesrc location={} bytestream-format=mpegts blocksize=4096 ! filesink location={}",
                input.display(),
                output.display()
            ),
            format!(
                "filesrc location={} bytestream-format=mpegts blocksize=1316 ! udpsink host=127.0.0.1 port={port}",
                input.display()
            ),
        ]),
        threaded: tee,
    };
    let registry = default_registry();
    let threads_before_runs = threads();

    for _ in 0..WARMUP_CYCLES {
        run_each(&registry, &pipelines).await;
    }
    let start_threads = threads_once_settled(threads_before_runs);
    let start_descriptors = open_file_descriptors();
    let start_bytes = LIVE_BYTES.load(Ordering::Relaxed);

    for _ in 0..MEASURED_CYCLES {
        run_each(&registry, &pipelines).await;
    }
    let end_threads = threads_once_settled(start_threads);
    let end_descriptors = open_file_descriptors();
    let heap_growth = LIVE_BYTES.load(Ordering::Relaxed) - start_bytes;

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
    assert_eq!(
        end_descriptors, start_descriptors,
        "open file descriptors after {MEASURED_CYCLES} runs"
    );
    assert_eq!(
        end_threads, start_threads,
        "threads after {MEASURED_CYCLES} runs"
    );
    assert!(
        heap_growth <= MAX_HEAP_GROWTH_BYTES,
        "the heap grew {heap_growth} bytes over {MEASURED_CYCLES} runs"
    );
}
