# Rerun upstream: where we stand

Status snapshot for the `re_video` hardware-decode contribution. The technical PoC
is **done and green**; this file records the plan for opening a PR so it can be
picked up cold.

## Update 2026-07-06: lead PR prepared and verified

Step 2 of the plan (the smallest standalone PR, `re_renderer::adopt_external_texture`)
is isolated, rebased onto current Rerun `main`, and verified. It is **not pushed** and
**no PR/RFC text is written** (that prose is owned by the author).

- **Patch:** `adopt-external-texture.patch` (this dir), a `git apply`-able diff over
  Rerun `main` at `981d09f507650834709f8110e1034c590f06d084`. Two files only
  (`re_renderer/src/wgpu_resources/{texture_pool,dynamic_resource_pool}.rs`); names no
  decoder; genericised the PoC's doc comments so nothing references glass2glass.
- **Contents:** `GpuTexturePool::adopt_external_texture` + the `GpuTextureInternal.external`
  destroy-skip + `DynamicResourcePool::alloc_external`, plus two tests: a CI-safe
  pool-level unit test (`alloc_external_registers_resolvable_handle`, no GPU) and a
  GPU behavioural test (`adopt_external_texture_resolves_and_survives_reclaim`) that
  proves the adopted texture is not destroyed on reclaim.
- **Verified** against the pinned `1.92.0` toolchain: `cargo fmt --check` clean,
  `cargo clippy -p re_renderer --lib --tests` clean, both tests pass on the lavapipe
  software adapter (so they are CI-portable, not 3060-only).
- **To open the PR:** apply the patch onto a fresh `main` clone (or a fork), push, and
  write the PR title/description. Do the RFC discussion (step 1) first.

## What is built and proven (RTX 3060)

The generalization is complete: `re_video` defines a **vendor-neutral** hardware-decode
extension point with **zero glass2glass dependency**, and the g2g binding lives in a
separate out-of-tree crate. All four tests pass; the patch applies cleanly to the pinned
Rerun commit; g2g-side docs are committed to `master`. See `README.md` here for the full
technical write-up and reproduce steps. The Rerun-side code lives (uncommitted, PoC) in
`/home/aaron/src/rerun-poc` at pin `ef9d94e9cf1af999a114bb0b815abcd3f0c0c94c`, captured
by `re_video-g2g-vulkan.patch`.

Three logically separable pieces, in increasing coupling:

| Piece | Coupling | Upstreamability |
| :--- | :--- | :--- |
| `re_renderer::adopt_external_texture` + `alloc_external` + `external` flag | **None** (names no decoder) | Standalone small PR, mergeable on its own merit |
| `re_video` registry (`register_hw_video_decoder`, `HwDecoderAttempt`, `DecodeError::HwDecoder`) + generic `GpuVideoFrame` behind `gpu-textures` | None (no g2g dep) | The RFC core; needs maintainer buy-in |
| `re_video_g2g` crate (the actual g2g Vulkan backend) | Path-deps glass2glass | **Not upstreamed** — reference implementer, stays out of tree |

## Rerun's contribution rules (checked July 2026)

- **No AI-generated-code policy exists** — neither `CONTRIBUTING.md` (main) nor the PR
  template mentions AI/LLM/generated code. No disclosure requirement, no ban.
- **The binding constraint is change size.** Acceptable PRs are either small (≤ ~100
  lines) or larger changes **discussed with a maintainer first** (issue or Discord).
  *"PRs containing large undiscussed changes may be closed without comment."*
- Trunk-based, small short-lived branches, draft PRs encouraged for early feedback.

Our change spans three crates + a new crate — squarely "large, must be discussed first".

## Plan for opening the PR (do this, in order)

1. **Open an RFC discussion first** (GitHub issue, or their Discord): propose the
   `re_video` hardware-decode registry + `GpuVideoFrame` extension point. Frame it as
   "let out-of-tree crates supply hardware decoders (Vulkan Video / VideoToolbox / NVDEC)
   without `re_video` depending on them", motivated by their known-slow ffmpeg-CLI native
   decode (issue #9815) and the WebCodecs-parity story. Link this PoC + results.
2. **Lead with the smallest standalone PR:** `re_renderer::adopt_external_texture` (the
   generic external-texture adoption, no decoder mention, ~60 lines). It stands on its own
   and de-risks the review relationship before the larger RFC lands. **PREPARED + verified
   against current `main`** (see the 2026-07-06 update above); only the push + PR text remain.
3. **Then the registry + `GpuVideoFrame`** as the RFC's reference PR, with `re_video_g2g`
   shown as the out-of-tree implementer (not submitted).
4. Keep `re_video_g2g` and the glass2glass path-deps entirely out of the PR.

## Open items owed on the Rerun side (not blockers for the RFC)

- Negotiate the Tier A/B (readback vs zero-copy) choice from which device `re_renderer`
  renders on, replacing the `set_prefer_gpu_textures` PoC setter.
- Wire texture adoption into Rerun's real video space-view (the PoC proves the render path
  with a standalone offscreen `ViewBuilder`).
- Device-identity note for the RFC: zero-copy requires the renderer to run on the *decode*
  device (g2g creates it, with video-decode + compute queue families a render-only device
  omits); on a split decode/display-GPU host a cross-device copy is unavoidable (Tier A).

## Make-or-break invariant

`re_video::GpuVideoFrame` (Rerun's workspace wgpu, 29.0) and `re_video_g2g` (glass2glass
wgpu, 29.x via `g2g-plugins`) must resolve to **one** wgpu crate version for the decoded
texture handle to cross with no copy. Verified: both unify to `wgpu 29.0.3`. If Rerun and
g2g ever move to semver-incompatible wgpu majors, Tier B stops compiling until realigned.
