//! Benchmarks for the caps-negotiation hot path (M284), the latency-critical,
//! hardest code in the framework. Guards regressions in the caps algebra
//! (`intersect` / `fixate`), the linear solver, and the DAG solver.
//!
//! Run with `cargo xtask bench` (or `cargo bench --manifest-path
//! g2g-bench/Cargo.toml --bench caps`).

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use g2g_core::caps::{Caps, CapsSet};
use g2g_core::format_element::CapsConstraint;
use g2g_core::graph::Graph;
use g2g_core::runtime::solver::{solve_graph, solve_linear, NodeConstraint};
use g2g_core::{Dim, RawVideoFormat, Rate, VideoCodec};

fn raw(fmt: RawVideoFormat, w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: fmt,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn raw_ranged(fmt: RawVideoFormat) -> Caps {
    Caps::RawVideo {
        format: fmt,
        width: Dim::Range { min: 16, max: 8192 },
        height: Dim::Range { min: 16, max: 8192 },
        framerate: Rate::Any,
    }
}

fn h264(w: u32, h: u32) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn bench_caps_algebra(c: &mut Criterion) {
    // A ranged producer meeting a fixed request: the common narrowing.
    let ranged = raw_ranged(RawVideoFormat::Nv12);
    let fixed = raw(RawVideoFormat::Nv12, 1920, 1080);
    c.bench_function("caps_intersect_range_fixed", |b| {
        b.iter(|| black_box(&ranged).intersect(black_box(&fixed)))
    });

    // A multi-format set narrowed against a single format, then fixated, the
    // per-link work the solver repeats.
    let set = CapsSet::from_alternatives(vec![
        raw_ranged(RawVideoFormat::Rgba8),
        raw_ranged(RawVideoFormat::Nv12),
        raw_ranged(RawVideoFormat::I420),
        raw_ranged(RawVideoFormat::Bgra8),
    ]);
    let want = CapsSet::one(raw(RawVideoFormat::Nv12, 1280, 720));
    c.bench_function("capsset_intersect_then_fixate", |b| {
        b.iter(|| black_box(&set).intersect(black_box(&want)).fixate())
    });
}

fn bench_solver(c: &mut Criterion) {
    // Linear chain: source(H264) -> decode(H264->NV12) -> sink(NV12), the
    // canonical decode pipeline negotiation.
    let nv12 = raw(RawVideoFormat::Nv12, 1920, 1080);
    let lin: Vec<CapsConstraint> = vec![
        CapsConstraint::Produces(CapsSet::one(h264(1920, 1080))),
        CapsConstraint::DerivedOutput(Box::new({
            let nv12 = nv12.clone();
            move |_in: &Caps| CapsSet::one(nv12.clone())
        })),
        CapsConstraint::Accepts(CapsSet::one(nv12.clone())),
    ];
    let refs: Vec<&CapsConstraint> = lin.iter().collect();
    c.bench_function("solve_linear_decode_chain", |b| {
        b.iter(|| solve_linear(black_box(&refs)).unwrap())
    });

    // The same chain as a DAG through the arc-consistency graph solver.
    let mut g: Graph<()> = Graph::new();
    let src = g.add_source(());
    let tx = g.add_transform(());
    let sink = g.add_sink(());
    g.link(src, tx).unwrap();
    g.link(tx, sink).unwrap();
    let vg = g.finish().unwrap();
    c.bench_function("solve_graph_decode_chain", |b| {
        b.iter(|| {
            // Rebuild constraints each iter: the DerivedOutput box is not Clone.
            let cs: Vec<NodeConstraint> = vec![
                NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(h264(1920, 1080)))),
                NodeConstraint::Element(CapsConstraint::DerivedOutput(Box::new({
                    let nv12 = nv12.clone();
                    move |_in: &Caps| CapsSet::one(nv12.clone())
                }))),
                NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(nv12.clone()))),
            ];
            solve_graph(black_box(&vg), black_box(&cs)).unwrap()
        })
    });
}

criterion_group!(benches, bench_caps_algebra, bench_solver);
criterion_main!(benches);
