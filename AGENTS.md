# AGENTS.md

Guidance for AI agents working in this repository. Read this before making changes.

## What this is

`glass2glass` (`g2g`) is a Rust multimedia pipeline framework, GStreamer-like but
with a statically typed, `no_std + alloc` core. Graphs are composed from typed
elements rather than runtime string-keyed factories. Full design: `DESIGN.md`.

## Workspace

Cargo workspace (`resolver = "2"`, edition 2021, MSRV 1.75, stable toolchain):

| Crate | Role |
| :--- | :--- |
| `g2g-core` | Core traits, `Frame`/`PipelinePacket`, `Caps` algebra, clock, memory domains, runtime. `no_std + alloc` baseline. |
| `g2g-plugins` | Standard source/sink/transform elements. OS-coupled ones gated behind features. |
| `g2g-ml` | ML inference elements (Burn / ORT) + multi-stream batching sink. |
| `g2g-bridge` | Interop layer. |

## Conventions

- **`no_std + alloc` baseline.** `g2g-core` and `g2g-plugins` are `#![no_std]`
  with `extern crate alloc`. Anything needing the OS (network, COM, GPU) lives
  behind a cargo feature that implies `std`. Do not add `std` use to the
  baseline path.
- **Feature + target gating.** OS-coupled deps are `optional = true`, pulled in
  by a named feature. Platform-specific deps go under
  `[target.'cfg(...)'.dependencies]` (e.g. the Windows-only `windows` crate
  under `cfg(windows)`), and the module is `#[cfg(all(target_os = "...", feature = "..."))]`.
- **Elements** implement `AsyncElement` (transform/sink) or `SourceLoop`
  (source) from `g2g-core`. Pattern: `intercept_caps` (negotiation),
  `configure_pipeline` (accept absolute caps), `process`/`run` (async work).
  Study `g2g-plugins/src/h264parse.rs` and `rtspsrc.rs` as references.
- **Properties: expose every meaningful knob.** Any `with_*` builder / setting a
  real pipeline would tune (bitrate, location, port, device, latency, ...) must
  also be a runtime property, so a `gst-launch` line can set it. Override
  `properties()` (a `&'static [PropertySpec]`, kebab-case names matching the
  GStreamer element where one exists), and handle each name in `set_property` /
  `get_property`. `parse_launch` looks the name up in `properties()` for its
  `PropKind`, then calls `set_property`, so both halves are required. Match the
  GStreamer unit/semantics (e.g. `bitrate` is `Uint` bits/second). Never accept a
  property you then ignore: only expose behavior the element actually applies.
  Study `g2g-plugins/src/ffmpegenc.rs` and the `m454_element_properties.rs` test.
- **Caps refinement** flows at runtime via `PipelinePacket::CapsChanged`, not
  just at negotiation. Emit it before the first affected `DataFrame` and on any
  mid-stream change; suppress re-emission when unchanged.
- **Parsers / demuxers: never trust the stream.** Counts, lengths, offsets,
  and dimensions read from a bitstream or container are attacker-controlled.
  Validate before use and fold arithmetic with checked / saturating ops, so
  malformed input fails the parse (returns `None` / an error) rather than
  panicking, overflowing, or allocating on a bogus length. Study
  `h264parse.rs` (saturating SPS geometry) and `fmp4.rs` (bounded box / sample
  counts) as references.
- **Comments:** preserve existing comments; update them if the logic changes;
  only delete when the whole block goes. Comment unusual code, not the obvious.
  No em dash (`—`) in comments; use `,` `:` or `()`.
- **Unsafe:** workspace lints set `unsafe_op_in_unsafe_fn = "deny"` and
  `undocumented_unsafe_blocks = "warn"`. Every `unsafe { }` needs a `// SAFETY:`
  comment. Types need `Debug` (`missing_debug_implementations = "warn"`).
- **Commits:** keep messages terse, a subject line plus at most a one or two
  line body. No `Co-Authored-By` trailer, ever: do not add Claude / Opus / any
  AI assistant as a co-author. Same for `CHANGELOG.md`, one terse line per
  milestone.

## Build & test (PowerShell)

```powershell
cargo check --workspace                              # default (no_std) build
cargo test --workspace                               # default test suite
cargo clippy --workspace --all-targets               # lints

# Feature-gated elements:
cargo test  -p g2g-plugins --features rtsp
cargo test  -p g2g-plugins --features mf-decode      # Windows only
cargo clippy -p g2g-plugins --features mf-decode --all-targets
```

Integration tests live in `g2g-plugins/tests/` and `g2g-core/tests/`, one file
per milestone (e.g. `m10_muxer.rs`).

## Testing policy

Test important features, not coverage. A test must run the real unit (import and
call it); mock only external boundaries (network, COM, GPU), never the code
under test. Every test needs an assertion that fails if the feature breaks.

## Milestones

Work is tracked by milestone `Mn`. The high-level roadmap is the top of
`DESIGN_TODO.md`; `DESIGN.md` §4.10 maps the architectural tracks to their spec
sections. Record each milestone in `CHANGELOG.md` under `## Unreleased`.
Pre-release `0.2.0` (tagged, not published to crates.io). Stability tiers and the
versioning policy live in `STABILITY.md`.

`DESIGN_TODO.md` is a terse catalogue of outstanding tasks only: no comparison
to GStreamer, no list of historical accomplishments. When a task is done, remove
it from `DESIGN_TODO.md` (do not leave "DONE" notes); if it established
architecture worth keeping, document that in `DESIGN.md` instead. `DESIGN.md`
describes only the current design.

## Platform notes

- **Windows decode (`mf-decode`):** `MfDecode` wraps the Media Foundation H.264
  Decoder MFT (`IMFTransform`) via the `windows` crate. COM is MTA; the element
  is thread-affine and intended for a single-thread executor (it asserts `Send`
  under a documented contract so the `multi-thread` runner accepts it). To verify
  the API surface, grep the fetched crate source under
  `~/.cargo/registry/src/.../windows-0.62.*/src/Windows/Win32/Media/MediaFoundation/mod.rs`
  rather than guessing signatures.

- **Android elements** (`mediacodec` / `mediacodec-wgpu` / `aaudio` / `camera2`):
  CI only cross-compiles them (`cargo check --target aarch64-linux-android`);
  real validation is on a device. See the "Android on-device testing" section in
  `README.md` for the build/push/run recipe (`tools/android-*-smoke.sh`). Key
  agent-facing caveats: a bare native binary has no binder threadpool (Codec2
  needs it to allocate the decoder's graphic buffers, so the decode probes dlsym
  `ABinderProcess_startThreadPool` from `libbinder_ndk.so`); the permission-gated
  capture paths (mic = `RECORD_AUDIO`, camera = `CAMERA`) and a true on-screen
  `SurfaceView` present cannot run from `/data/local/tmp` and need an APK harness,
  so those probes report the denial and assert only the parts they can check.
