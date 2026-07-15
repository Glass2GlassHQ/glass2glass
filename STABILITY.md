# API Stability Audit & Policy

Status: **pre-1.0, tagged `0.2.0`**. This document is both the audit that answers
"when is `g2g` ready for a 1.0 version indicator" and the **stability policy now in
effect** (see "Versioning & MSRV policy" below). **Update (M561):** the
`#[non_exhaustive]` pass has landed - the growth-prone vocabulary enums are now
annotated, so adding a variant is no longer a breaking change. Gate item 1 is done
and the policy is published (`0.2.0`); the remaining gates are adoption. `1.0` is an
*API-stability promise* (semver), not a feature-count claim, so the gating question
is: **can we freeze the public vocabulary and treat changes to it as breaking?**

## TL;DR

- The **execution surface is already stable in practice.** `AsyncElement` /
  `OutputSink` / `Frame` / `PipelinePacket` / `MemoryDomain` have not changed shape
  in ~1–2 weeks while ~200 milestones' worth of work landed on top of them
  (element.rs untouched since M354; frame.rs since M226). That is the hard part
  and it is essentially done.
- The **blocker was not churn, it was extensibility.** Until M561, none of the
  public vocabulary enums (`Caps`, `RawVideoFormat`, `AudioFormat`, `VideoCodec`,
  `MemoryDomain`, `G2gError`, `PipelinePacket`, ...) were `#[non_exhaustive]`, and
  the caps vocabulary still gains variants regularly (new formats, `Text`, audio
  channel modes landed as recently as M400–M479). With exhaustive enums, **every
  new variant is a breaking change** for any downstream `match`. That single fact
  is what would force major-version bumps and was the main reason to hold 1.0.
- The cheap, high-leverage fix, **now done (M561)**: the growth-prone vocabulary
  enums are marked `#[non_exhaustive]`, converting their ongoing growth from
  breaking to non-breaking and removing the biggest 1.0 obstacle.
- Recommendation: **cut a tagged `0.2` now**, apply the `#[non_exhaustive]` +
  stability-split work below, land one real external adopter, and reserve `1.0`
  for when the vocabulary is frozen-and-extensible and someone depends on it.
  Consider `g2g-core` reaching 1.0 ahead of the sprawling plugin set.

## Churn evidence (g2g-core public-surface files)

Measured at 2026-07-06 over the last 200 commits (870 total in the repo).

| File | commits/200 | last change | reading |
| :--- | ---: | :--- | :--- |
| `element.rs` (traits) | 26 | 2026-06-28 (M354) | **stable** ~1 wk, all recent work built on it unchanged |
| `frame.rs` (Frame/Packet) | 8 | 2026-06-23 (M226) | **stable** ~2 wk |
| `memory.rs` (domains) | 17 | 2026-06-28 (M354) | **stable** ~1 wk |
| `error.rs` (G2gError) | 6 | 2026-06-28 (M343) | stable |
| `format_element.rs` (CapsConstraint) | 8 | 2026-06-23 (M227) | stable |
| `property.rs` (props) | 3 | 2026-06-25 | stable |
| `wire.rs` (codec) | 3 | 2026-07-06 (M557) | additive only; already `WIRE_VERSION`-gated |
| `graph.rs` (Bin/Graph) | 12 | 2026-07-01 (M485) | additive, still moving |
| `caps.rs` (vocabulary) | 25 | 2026-07-01 (M479) | **live edge** — variants still being added |

The pattern: the *traits and carriers* have converged; the *vocabulary* (caps.rs)
is the only core file that keeps changing, and its changes are additive
(new variants), which is exactly the case `#[non_exhaustive]` exists to make safe.

## The `#[non_exhaustive]` gap (the pivotal finding) — CLOSED (M561)

These public vocabulary enums are now `#[non_exhaustive]`:

- `caps::{Caps, RawVideoFormat, AudioFormat, VideoCodec, ByteStreamEncoding, TextFormat, TensorDType, TensorLayout}`
- `memory::{MemoryDomain, MemoryDomainKind}`
- `error::{G2gError, HardwareError}`
- `frame::PipelinePacket`
- `property::{PropKind, PropValue}`

`#[non_exhaustive]` forces downstream `match`es to carry a `_` arm, so M561 was a
**workspace-wide mechanical change** touching ~100 match sites across every
feature and platform target. The convention applied: a pass-through transform's
`PipelinePacket` match forwards an unknown ordered control packet downstream
unchanged (`other => out.push(other)...`), matching the `Segment` variant's
documented "elements forward it downstream unchanged" contract; a terminal sink
no-ops; a format/codec namer mirrors its existing fallback (error / `None` /
`unreachable!` when the value can only be a variant the element itself produced).
Matches inside `g2g-core` are unaffected (the attribute has no effect in-crate).

Deliberately-closed enums (left exhaustive, complete by design, and now annotated
as such at their definition sites): `ConfigureOutcome`, `PadDir` / `PadDirection`,
`ClockPriority`, `LinkPolicy`, `SeekType`. Adding a variant to one of these is a
deliberate breaking change; keeping them exhaustive is what makes that a compile
error at every use site rather than a silent fall-through.

## Proposed stability tiers

**Tier 1 — Stable (freeze at 1.0, semver-covered).** The `g2g-core` execution
vocabulary, once the `#[non_exhaustive]` pass lands:
`AsyncElement`, `SourceLoop`, `OutputSink`, `ConfigureOutcome`, `Reconfigure`;
`Frame`, `FrameTiming`, `PipelinePacket`; `Caps`, `CapsSet`, `Dim`, `Rate` + the
format enums; `MemoryDomain`, `MemoryDomainKind`, the `Owned*` payloads + keep-alive
traits; `G2gError`; `PropertySpec` / `PropKind` / `PropValue`; the `wire` codec
(already versioned); `CapsConstraint`.

**Tier 2 — Provisional (public, but may change in a minor pre-2.0; document as
such).** Negotiation coupling internals (`format_element` beyond `CapsConstraint`),
`graph`/`Bin`/`ValidatedGraph`, `pad_template`, the `runtime` runners, `fanout` /
`ReverseChannel`, `slot` (`dyn-slot`). These are newer or still gaining shape
(graph.rs, fanout) and should not block a core 1.0.

**Tier 3 — Experimental (feature-gated, explicitly no stability promise).** All
OS-/GPU-coupled elements in `g2g-plugins`, `g2g-ml`, the bridges, and the
language bindings. Anything currently "validated on host, not CI" ships here until
CI covers it. Correction (M561): the earlier "`macOS VtDecode` is compile-pending
/ never built" claim is outdated — the `features (macos)` CI job compiles
`vtdecode,vtencode` on `macos-latest` on every push (green as of M561), just as
`features (android)` cross-compiles `mediacodec`/`aaudio`/`camera2`. So the
compile gate is met on both; what these share with the rest of Tier 3 is that
their *runtime* path is device/host-validated, not CI-run. The real 1.0 action is
therefore the routine Tier-3 one (mark as experimental, no stability promise),
not "build a platform never compiled".

## Crate → tier mapping

The type tiers above translate to per-crate promises:

| Crate | Tier | Promise |
| :--- | :--- | :--- |
| `g2g-core` | **1** | The semver-covered surface. Breaking changes to the frozen vocabulary bump the major version once at 1.0; until then, minor. |
| `g2g-plugins` | **3** | Standard elements. Feature-gated / OS-coupled paths are experimental; the pure-Rust element set is provisional (Tier 2 in spirit) but not yet frozen. |
| `g2g-plugin` | **2** | The dynamic-plugin SDK (`declare_plugin!` + ABI tag). ABI is versioned separately via its own tag. |
| `g2g-ml`, `g2g-bridge` | **3** | Experimental; no stability promise pre-1.0. |
| `g2g-capi`, `g2g-pyapi`, `g2g-python` | **3** | Bindings; surface tracks `g2g-core` but is itself experimental. |
| `xtask`, `g2g-bench`, `g2g-web` | n/a | Dev/demo crates, `publish = false`. |

## Versioning & MSRV policy

- **Scheme.** Semver. Pre-1.0, breaking changes to any tier bump the **minor**
  version (0.x → 0.(x+1)); additive changes bump the **patch**. At 1.0, Tier-1
  breaking changes bump the **major**; Tier-2/3 may still break in a minor with a
  `CHANGELOG` note.
- **What "breaking" means for Tier 1.** Removing/renaming a public item, changing a
  signature, or adding a variant to a *deliberately-closed* (exhaustive) enum
  (`ConfigureOutcome`, `PadDir`/`PadDirection`, `ClockPriority`, `LinkPolicy`,
  `SeekType`). Adding a variant to a `#[non_exhaustive]` vocabulary enum is
  **not** breaking, by design.
- **Feature flags** are additive and not part of the semver contract; enabling one
  may pull in `std` / OS deps. Default (`no_std + alloc`) is the stable baseline.
- **MSRV.** Currently Rust 1.75 (`rust-version` in `[workspace.package]`). An MSRV
  bump is a minor-version change and called out in `CHANGELOG.md`. We do not raise
  MSRV in a patch release.
- **The `wire` codec** carries its own `WIRE_VERSION`; on-wire format changes are
  gated there independently of the crate version.

## 1.0 gate checklist

1. ~~**`#[non_exhaustive]` pass** on the Tier-1 vocabulary enums + a note on the
   deliberately-closed ones.~~ **DONE (M561)** — full default suite + clippy
   `--all-targets`, the CI linux/GPU/Android feature sets green; Windows/macOS
   arms added statically (per-platform CI verifies).
2. ~~**Freeze Tier 1**; publish this policy (which crates/types are Tier 1/2/3, MSRV
   policy, how versions bump).~~ **DONE (0.2.0)** — the crate→tier mapping and the
   versioning/MSRV policy above are now in effect; the traits are de-facto frozen
   since M354.
3. **Claims match validation**: every advertised capability is either in CI or
   marked experimental. (The macOS-unbuilt worry is resolved: CI compiles VtDecode
   on macOS - see the Tier 3 correction above. What remains is labelling the
   host-/device-only-*runtime*-validated items as experimental.)
4. **One real external adopter** — the Rerun/`re_video` upstream (currently a
   fork) or a bindings/embedded design partner. 1.0 without a consumer is a
   promise nobody asked for.
5. ~~**Intermediate release first**: tag `0.2`.~~ **DONE (0.2.0)** — tagged
   in-repo (not yet published to crates.io); the crates carry publish metadata
   (description/license) and are crates.io-ready when an adopter warrants it.
   `g2g-core` may reach 1.0 before `g2g-plugins`.

Realistic read: the codebase is capability-rich and the traits have converged, so
1.0 is gated on *policy + extensibility + adoption*, not features. With gates 1, 2,
and 5 closed, the nearest concrete path is now: land the Rerun upstream (or
equivalent adopter) → `g2g-core` 1.0.
