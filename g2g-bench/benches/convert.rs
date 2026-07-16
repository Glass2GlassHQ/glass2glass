//! Benchmarks for the raw-video frame-convert hot path (M284): the per-pixel
//! `videoconvert::convert` dispatch a software pipeline runs on every frame.
//! Guards regressions in the conversion inner loops (RGBA<->NV12/I420 and the
//! NV12<->I420 chroma repack) at 1080p.
//!
//! Run with `cargo xtask bench` (or `cargo bench --manifest-path
//! g2g-bench/Cargo.toml --bench convert`).

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use g2g_core::RawVideoFormat;
use g2g_plugins::videoconvert::convert;

const W: usize = 1920;
const H: usize = 1080;

/// A source buffer big enough for any of the formats benched (RGBA is the
/// largest at 4 bytes/px).
fn src_buf() -> Vec<u8> {
    let mut v = vec![0u8; W * H * 4];
    // A non-uniform pattern so the compiler can't fold the conversion away.
    for (i, b) in v.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    v
}

fn bench_convert(c: &mut Criterion) {
    let src = src_buf();
    let mut group = c.benchmark_group("videoconvert_1080p");
    // One "element" per benchmark is one frame; report pixels/s.
    group.throughput(Throughput::Elements((W * H) as u64));

    let cases = [
        ("rgba_to_nv12", RawVideoFormat::Rgba8, RawVideoFormat::Nv12),
        ("rgba_to_i420", RawVideoFormat::Rgba8, RawVideoFormat::I420),
        ("nv12_to_rgba", RawVideoFormat::Nv12, RawVideoFormat::Rgba8),
        ("nv12_to_i420", RawVideoFormat::Nv12, RawVideoFormat::I420),
    ];
    for (name, from, to) in cases {
        group.bench_function(name, |b| {
            b.iter(|| convert(black_box(&src), black_box(from), black_box(to), W, H))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_convert);
criterion_main!(benches);
