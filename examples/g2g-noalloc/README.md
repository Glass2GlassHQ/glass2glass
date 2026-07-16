# g2g-noalloc: the link-time no-heap + panic-free proof

This standalone `no_std` staticlib is the machine-checkable form of glass2glass's
flagship embedded guarantee: **a whole media pipeline that links with no heap and
no reachable panic.**

It wraps a real `source -> transform -> sink` pipeline (the shared
[`noalloc-pipeline`](../noalloc-pipeline) crate, also booted on an emulated
Cortex-M by [`g2g-qemu`](../g2g-qemu)) built with the static element model
([`g2g_core::staticelem`](../../g2g-core/src/staticelem.rs)) over a
[`StaticLendRing`](../../g2g-core/src/staticpool.rs) capture ring, driven to
completion by a bare `no_std` single-poll executor. Every stage is a concrete
type, so the chain monomorphizes to unboxed `async` state machines: no `dyn`, no
`Box`, no allocation.

Note the `staticlib`-only crate-type: adding `rlib` here silently disables fat
LTO on the archive, which un-eliminates core's panic/fmt machinery and fails
the symbol checks (that is why the pipeline is shared via `noalloc-pipeline`
instead).

The no-heap proof rests on two facts:

1. `g2g-core` is depended on with `default-features = false`, so the `alloc` crate
   is **not a dependency** at all. Any `Box`/`Vec`/`String` on a reachable path
   would be a compile error.
2. This crate defines **no `#[global_allocator]`**. If any reachable code needed
   the heap, the build would fail for want of an allocator.

So the fact that it compiles and links for `thumbv7em-none-eabihf`, and that the
linked archive references **zero allocator symbols**, is the guarantee that a full
g2g pipeline runs heap-free on a Cortex-M.

The panic-free proof (M626) goes one step further: every reachable path avoids
unwrap / slice-index / overflow panics, and the single-poll executor lets the
optimizer discharge the compiler's resumed-after-completion guard, so the
optimized archive contains **zero `core::panicking` symbols**. The mandatory
`#[panic_handler]` is provably dead code: nothing can call it. On a safety /
no-heap part, the pipeline cannot halt through the panic machinery.

## Run the check

```sh
tools/noalloc-check.sh
```

It builds this crate for `thumbv7em-none-eabihf` and asserts the archive contains
no allocator symbols (`__rust_alloc`, `_ZN5alloc`, `handle_alloc_error`, ...) and
no panic symbols (`panic_fmt`, `panic_bounds_check`, ...) while the pipeline entry
point (`g2g_noalloc_run`) *is* present (so the code was really emitted, not
eliminated). It then builds the crate for the host and runs the pipeline for real
through [`host-harness.c`](host-harness.c) (64 frames, checksum asserted), so the
symbol proofs are about code that demonstrably executes to completion.

## Footprint report

```sh
tools/footprint-report.sh
```

links this crate into a real gc-sectioned ELF and reports the numbers an MCU
integrator budgets against, enforced as CI regression budgets: **ROM 3067 bytes,
static RAM 0 bytes, worst-case stack 1468 bytes** for the whole pipeline
*including the real camera seam and SPI display elements* (the entry frame
holds the capture ring + the monomorphized pipeline state machine, so the
stack number is the true per-pipeline RAM). The stack bound is computed from the disassembly call
graph by [`tools/footprint.py`](../../tools/footprint.py).

The runtime complement (the same static model driving 100k frames with a counting
allocator observing zero allocations) is
[`g2g-plugins/tests/m624_static_pipeline_noalloc.rs`](../../g2g-plugins/tests/m624_static_pipeline_noalloc.rs).

Kept out of the normal workspace build (see the root `Cargo.toml` `exclude`); it is
built only by the check script / CI.
