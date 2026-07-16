//! Benchmark for the runner loop's inner transport (M286): the bounded per-edge
//! channel every frame crosses. The full `run_graph` paces to PTS (wall-clock
//! bound, unsuitable for a microbench), so this isolates the hot mechanism, a
//! producer filling a bounded channel while a consumer drains it, the
//! backpressure path that dominates steady-state glass-to-glass latency.
//!
//! Run with `cargo xtask bench` (or `cargo bench --manifest-path
//! g2g-bench/Cargo.toml --bench runner`).

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use g2g_core::runtime::bounded;

/// Frames pushed per iteration.
const N: u64 = 4096;

fn bench_channel(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("build current-thread runtime");

    let mut group = c.benchmark_group("runner");
    group.throughput(Throughput::Elements(N));

    // Sweep the link capacity: a live (2) vs a throughput (8) edge depth, to
    // show the backpressure round-trip cost at each.
    for cap in [2usize, 8] {
        group.bench_function(format!("bounded_channel_cap{cap}"), |b| {
            b.to_async(&rt).iter(|| async move {
                let (tx, rx) = bounded::<u64>(cap);
                // join! drives producer + consumer on one task: the producer
                // parks when full, the consumer's recv wakes it, exactly the
                // runner's single-edge backpressure handshake.
                let producer = async {
                    for i in 0..N {
                        tx.send(i).await.expect("receiver alive");
                    }
                };
                let consumer = async {
                    let mut sum = 0u64;
                    for _ in 0..N {
                        sum += rx.recv().await.expect("sender alive");
                    }
                    sum
                };
                let (_, sum) = tokio::join!(producer, consumer);
                black_box(sum)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_channel);
criterion_main!(benches);
