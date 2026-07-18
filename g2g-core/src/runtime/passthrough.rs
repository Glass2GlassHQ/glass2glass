//! Passthrough-field negotiation helpers for the solver's mid-stream re-solve
//! (DESIGN.md 4.13.4). A transform declares which caps fields it forwards
//! unchanged (`PassthroughFields`); these functions couple those fields across a
//! link (`couple_*`), project a feasible input from an output (`project_*`), and
//! discover / verify a mask by probing an element's caps closure (`discover_*`,
//! `verify_*`). Solver-only, so they live under the runtime tree rather than in
//! the data-plane `caps.rs`; the two `project_*` functions serve the std graph
//! runner's backward-feasibility sweep and stay `std`-gated.

use crate::caps::{
    intersect_sample_rate, AudioFormat, Caps, CapsSet, Dim, PassthroughFields, Rate, RawVideoFormat,
    VideoCodec,
};
// Only the std-gated `project_passthrough` widens sample_rate back to ANY.
#[cfg(feature = "std")]
use crate::caps::ANY_SAMPLE_RATE;

/// Narrow `input` by intersecting each *passthrough* field against the
/// corresponding field of `pin` (the field-level backward coupling: e.g.
/// `Range(1..MAX) ∩ Fixed(160) = Fixed(160)`). Retarget fields (not in `mask`)
/// are left as `input` carries them, since the transform sets them
/// independently of its input. Same media variant required; `None` if a
/// passthrough field has no overlap (the alternative dies) or the variants
/// differ. Used by the solver's `DerivedCoupled` backward sweep.
pub(crate) fn couple_passthrough(input: &Caps, pin: &Caps, mask: PassthroughFields) -> Option<Caps> {
    match (input, pin) {
        (
            Caps::RawVideo { format: fi, width: wi, height: hi, framerate: ri },
            Caps::RawVideo { format: fp, width: wp, height: hp, framerate: rp },
        ) => {
            let format = if mask.format {
                if fi != fp {
                    return None;
                }
                *fi
            } else {
                *fi
            };
            let width = if mask.width { wi.intersect(wp)? } else { wi.clone() };
            let height = if mask.height { hi.intersect(hp)? } else { hi.clone() };
            let framerate = if mask.framerate { ri.intersect(rp)? } else { ri.clone() };
            Some(Caps::RawVideo { format, width, height, framerate })
        }
        (
            Caps::CompressedVideo { codec: ci, width: wi, height: hi, framerate: ri },
            Caps::CompressedVideo { codec: cp, width: wp, height: hp, framerate: rp },
        ) => {
            let codec = if mask.format {
                if ci != cp {
                    return None;
                }
                *ci
            } else {
                *ci
            };
            let width = if mask.width { wi.intersect(wp)? } else { wi.clone() };
            let height = if mask.height { hi.intersect(hp)? } else { hi.clone() };
            let framerate = if mask.framerate { ri.intersect(rp)? } else { ri.clone() };
            Some(Caps::CompressedVideo { codec, width, height, framerate })
        }
        (
            Caps::Audio { format: fi, channels: ci, sample_rate: si },
            Caps::Audio { format: fp, channels: cp, sample_rate: sp },
        ) => {
            let format = if mask.format {
                if fi != fp {
                    return None;
                }
                *fi
            } else {
                *fi
            };
            let channels = if mask.channels {
                if ci != cp {
                    return None;
                }
                *ci
            } else {
                *ci
            };
            let sample_rate =
                if mask.sample_rate { intersect_sample_rate(*si, *sp)? } else { *si };
            Some(Caps::Audio { format, channels, sample_rate })
        }
        _ => None,
    }
}

/// Like [`couple_passthrough`], but tolerates a *variant change* across the
/// transform (a decoder `CompressedVideo -> RawVideo`, an encoder the reverse),
/// for the discovered-passthrough backward coupling of a plain `DerivedOutput`.
/// Same-variant inputs defer to [`couple_passthrough`] (exact field coupling,
/// including `format`/`channels`/`sample_rate`). Across the two video variants
/// only the shared geometry / framerate fields can couple (the `format` slot is a
/// codec vs raw-format boundary, so it is never a passthrough field there); the
/// input keeps its own variant and scalar identity. `None` if a masked shared
/// field has no overlap, or for a cross-variant pair with no shared geometry.
pub(crate) fn couple_passthrough_derived(input: &Caps, pin: &Caps, mask: PassthroughFields) -> Option<Caps> {
    match (input, pin) {
        (Caps::RawVideo { .. }, Caps::RawVideo { .. })
        | (Caps::CompressedVideo { .. }, Caps::CompressedVideo { .. })
        | (Caps::Audio { .. }, Caps::Audio { .. }) => return couple_passthrough(input, pin, mask),
        _ => {}
    }
    // Cross video-variant: couple the geometry / rate both carry, keep `input`'s
    // variant + scalar identity (`format`/`codec` is retargeted across a codec
    // boundary, so `mask.format` is not applied here).
    let (wi, hi, ri) = (geo_width(input)?, geo_height(input)?, geo_rate(input)?);
    let (wp, hp, rp) = (geo_width(pin)?, geo_height(pin)?, geo_rate(pin)?);
    let width = if mask.width { wi.intersect(wp)? } else { wi.clone() };
    let height = if mask.height { hi.intersect(hp)? } else { hi.clone() };
    let framerate = if mask.framerate { ri.intersect(rp)? } else { ri.clone() };
    match input {
        Caps::RawVideo { format, .. } => Some(Caps::RawVideo { format: *format, width, height, framerate }),
        Caps::CompressedVideo { codec, .. } => {
            Some(Caps::CompressedVideo { codec: *codec, width, height, framerate })
        }
        _ => None,
    }
}

/// Project an output-side feasible `out` onto the *input* side of a
/// `DerivedCoupled` transform: keep passthrough fields, widen each retargeted
/// field to "anything the transform can take" (`Dim`/`Rate` -> `Any`,
/// `sample_rate` -> [`ANY_SAMPLE_RATE`]). Returns `None` when a retargeted field
/// is a non-rangeable scalar (`format` / `codec` / `channels`) with no wildcard,
/// i.e. the input feasibility can't be expressed as a single `Caps` (the solver
/// then imposes no upstream feasibility constraint, the status quo). Used by
/// `backward_feasible` for the mid-stream snapshot.
#[cfg(feature = "std")]
pub(crate) fn project_passthrough(out: &Caps, mask: PassthroughFields) -> Option<Caps> {
    match out {
        Caps::RawVideo { format, width, height, framerate } => {
            if !mask.format {
                return None; // retargeted format has no wildcard
            }
            Some(Caps::RawVideo {
                format: *format,
                width: if mask.width { width.clone() } else { Dim::Any },
                height: if mask.height { height.clone() } else { Dim::Any },
                framerate: if mask.framerate { framerate.clone() } else { Rate::Any },
            })
        }
        Caps::CompressedVideo { codec, width, height, framerate } => {
            if !mask.format {
                return None;
            }
            Some(Caps::CompressedVideo {
                codec: *codec,
                width: if mask.width { width.clone() } else { Dim::Any },
                height: if mask.height { height.clone() } else { Dim::Any },
                framerate: if mask.framerate { framerate.clone() } else { Rate::Any },
            })
        }
        Caps::Audio { format, channels, sample_rate } => {
            if !mask.format || !mask.channels {
                return None; // no format / channel wildcard
            }
            Some(Caps::Audio {
                format: *format,
                channels: *channels,
                sample_rate: if mask.sample_rate { *sample_rate } else { ANY_SAMPLE_RATE },
            })
        }
        _ => None,
    }
}

/// Project an output-side feasible `out` onto the *input* side of a plain
/// `DerivedOutput` for the mid-stream snapshot ([`backward_feasible`]). Unlike
/// [`couple_passthrough_derived`] (the full-chain coupling, which keeps the input
/// sample's own value on a non-passthrough field), this *widens* every
/// non-passthrough geometry / rate field to `Any`: the transform re-derives that
/// field from whatever input it receives mid-stream, so the input edge must stay
/// unconstrained on it. Freezing it to the startup sample (the M258 v1 behaviour)
/// made the snapshot reject a legitimately re-derived mid-stream geometry, the
/// Caps-β forward gap.
///
/// Same-variant transforms defer to [`project_passthrough`] (which already widens
/// retargeted fields and rejects a non-rangeable retargeted scalar). Across the
/// decoder / encoder variant change, the passthrough geometry / rate fields take
/// the downstream value from `out` while the non-passthrough fields widen to
/// `Any`; `sample` supplies the input variant and its scalar identity (codec /
/// format), which `out` cannot give.
#[cfg(feature = "std")]
pub(crate) fn project_passthrough_derived(
    sample: &Caps,
    out: &Caps,
    mask: PassthroughFields,
) -> Option<Caps> {
    match (sample, out) {
        (Caps::RawVideo { .. }, Caps::RawVideo { .. })
        | (Caps::CompressedVideo { .. }, Caps::CompressedVideo { .. })
        | (Caps::Audio { .. }, Caps::Audio { .. }) => return project_passthrough(out, mask),
        _ => {}
    }
    // Cross video-variant (decoder / encoder): passthrough geometry / rate take the
    // downstream value, the rest widen to `Any`; keep `sample`'s variant + scalar id.
    let (wp, hp, rp) = (geo_width(out)?, geo_height(out)?, geo_rate(out)?);
    let width = if mask.width { wp.clone() } else { Dim::Any };
    let height = if mask.height { hp.clone() } else { Dim::Any };
    let framerate = if mask.framerate { rp.clone() } else { Rate::Any };
    match sample {
        Caps::RawVideo { format, .. } => Some(Caps::RawVideo { format: *format, width, height, framerate }),
        Caps::CompressedVideo { codec, .. } => {
            Some(Caps::CompressedVideo { codec: *codec, width, height, framerate })
        }
        _ => None,
    }
}

/// The fields [`discover_passthrough`] probes for, one per [`PassthroughFields`]
/// flag.
#[derive(Clone, Copy)]
enum ProbeField {
    Width,
    Height,
    Framerate,
    Format,
    Channels,
    SampleRate,
}

/// Probe a `DerivedOutput`-style closure to discover which caps fields it passes
/// through unchanged (output field tracks input field), so the solver can couple
/// those fields backward the same way a declared
/// [`DerivedCoupled`](crate::format_element::CapsConstraint::DerivedCoupled) mask
/// does, the "invertible fields of a `DerivedOutput`". `f` is not analytically
/// invertible, but it is evaluable, so a field's behaviour is read off two
/// concrete probes.
///
/// Conservative by construction: a field is marked passthrough only when two
/// distinct concrete probes *both* show the closure's single, same-shaped output
/// field equal to the probed input field. A closure that rejects a probe, fixes
/// the field (a retargeted decoder format), or returns multiple/ambiguous outputs
/// yields `false` for that field, so discovery never invents coupling that is not
/// there (a wrong `true` would narrow the input incorrectly). `sample` is a
/// representative input alternative; its geometry is concretised first so a
/// `Range`/`Any` input field does not confuse the equality test.
pub(crate) fn discover_passthrough(f: &dyn Fn(&Caps) -> CapsSet, sample: &Caps) -> PassthroughFields {
    let base = concrete_probe_base(sample);
    // Soundness gate: a field is probed by *varying* it, so a closure that is
    // multi-valued on the sample's own identity (e.g. a converter that offers
    // `{passthrough, retargeted}` for the sample's format but is coincidentally
    // single-valued at the probe values) would be mis-read as passthrough. Per-
    // field equality alone can't see that, so require the closure to be single-
    // valued on the sample's representative input before trusting any field: a
    // genuinely ambiguous transform has no well-defined per-field passthrough.
    if single_out(f, &base).is_none() {
        return PassthroughFields::NONE;
    }
    PassthroughFields {
        width: probe_field(f, &base, ProbeField::Width),
        height: probe_field(f, &base, ProbeField::Height),
        framerate: probe_field(f, &base, ProbeField::Framerate),
        format: probe_field(f, &base, ProbeField::Format),
        channels: probe_field(f, &base, ProbeField::Channels),
        sample_rate: probe_field(f, &base, ProbeField::SampleRate),
    }
}

/// Soundness check for a [`DerivedCoupled`](crate::format_element::CapsConstraint::DerivedCoupled)
/// transform: every field its `passthrough` mask declares must genuinely be
/// passed through by its `derive` closure, i.e. for the concrete input `sample`
/// *every* output alternative repeats that field unchanged. The mask and the
/// closure are two sources of truth for the same fact (which fields couple
/// backward), and a mask that claims a field the closure actually retargets is
/// unsound: the solver would narrow the input on a field the transform rewrites.
/// This catches that drift (driven from a `debug_assert!` on the solver's
/// forward-derivation path), and unlike [`discover_passthrough`] it stays correct
/// for the multi-valued closures `DerivedCoupled` exists for (it checks the
/// declared fields across *all* alternatives rather than requiring a single
/// output). A closure that rejects `sample` (empty output) has nothing to verify
/// and passes; only the unsound direction (declared-but-not-honoured) fails. The
/// conservative reverse (a field the closure passes through but the mask omits)
/// is sound, just a missed coupling, so it is not flagged.
pub(crate) fn verify_passthrough_sound(
    f: &dyn Fn(&Caps) -> CapsSet,
    passthrough: PassthroughFields,
    sample: &Caps,
) -> bool {
    let out = f(sample);
    if out.alternatives().is_empty() {
        return true;
    }
    let declared = [
        (passthrough.format, ProbeField::Format),
        (passthrough.width, ProbeField::Width),
        (passthrough.height, ProbeField::Height),
        (passthrough.framerate, ProbeField::Framerate),
        (passthrough.channels, ProbeField::Channels),
        (passthrough.sample_rate, ProbeField::SampleRate),
    ];
    for (claimed, field) in declared {
        if claimed && !out.alternatives().iter().all(|alt| field_eq(alt, sample, field)) {
            return false;
        }
    }
    true
}

/// Concretise `sample`'s ranged geometry/rate to fixed sentinels so the closure
/// is probed on concrete inputs (a `Range`/`Any` input field would otherwise
/// make the output-equals-input test ambiguous). Scalar identity (format / codec
/// / channels) is kept from `sample`, since the closure may key on it.
fn concrete_probe_base(sample: &Caps) -> Caps {
    match sample {
        Caps::RawVideo { format, .. } => Caps::RawVideo {
            format: *format,
            width: Dim::Fixed(64),
            height: Dim::Fixed(64),
            framerate: Rate::Fixed(30 << 16),
        },
        Caps::CompressedVideo { codec, .. } => Caps::CompressedVideo {
            codec: *codec,
            width: Dim::Fixed(64),
            height: Dim::Fixed(64),
            framerate: Rate::Fixed(30 << 16),
        },
        Caps::Audio { format, .. } => Caps::Audio { format: *format, channels: 2, sample_rate: 48_000 },
        other => other.clone(),
    }
}

/// True when `f` passes `field` through: two concrete probes that differ only in
/// `field` each produce a single output whose `field` equals the probe's.
fn probe_field(f: &dyn Fn(&Caps) -> CapsSet, base: &Caps, field: ProbeField) -> bool {
    let (Some(p0), Some(p1)) = (set_probe(base, field, false), set_probe(base, field, true)) else {
        return false;
    };
    let (Some(o0), Some(o1)) = (single_out(f, &p0), single_out(f, &p1)) else {
        return false;
    };
    field_eq(&o0, &p0, field) && field_eq(&o1, &p1, field)
}

/// The single output of `f(input)`, or `None` if it produced zero or several
/// alternatives (discovery stays conservative on ambiguous closures).
fn single_out(f: &dyn Fn(&Caps) -> CapsSet, input: &Caps) -> Option<Caps> {
    let set = f(input);
    match set.alternatives() {
        [only] => Some(only.clone()),
        _ => None,
    }
}

/// `base` with `field` set to probe value 0 (`hi = false`) or 1 (`hi = true`),
/// or `None` if `base`'s variant has no such field.
fn set_probe(base: &Caps, field: ProbeField, hi: bool) -> Option<Caps> {
    let mut c = base.clone();
    match (&mut c, field) {
        (Caps::RawVideo { width, .. }, ProbeField::Width)
        | (Caps::CompressedVideo { width, .. }, ProbeField::Width) => {
            *width = Dim::Fixed(if hi { 128 } else { 64 });
        }
        (Caps::RawVideo { height, .. }, ProbeField::Height)
        | (Caps::CompressedVideo { height, .. }, ProbeField::Height) => {
            *height = Dim::Fixed(if hi { 128 } else { 64 });
        }
        (Caps::RawVideo { framerate, .. }, ProbeField::Framerate)
        | (Caps::CompressedVideo { framerate, .. }, ProbeField::Framerate) => {
            *framerate = Rate::Fixed(if hi { 60 << 16 } else { 30 << 16 });
        }
        (Caps::RawVideo { format, .. }, ProbeField::Format) => {
            *format = if hi { RawVideoFormat::I420 } else { RawVideoFormat::Nv12 };
        }
        (Caps::CompressedVideo { codec, .. }, ProbeField::Format) => {
            *codec = if hi { VideoCodec::H265 } else { VideoCodec::H264 };
        }
        (Caps::Audio { format, .. }, ProbeField::Format) => {
            *format = if hi { AudioFormat::PcmF32Le } else { AudioFormat::PcmS16Le };
        }
        (Caps::Audio { channels, .. }, ProbeField::Channels) => {
            *channels = if hi { 1 } else { 2 };
        }
        (Caps::Audio { sample_rate, .. }, ProbeField::SampleRate) => {
            *sample_rate = if hi { 44_100 } else { 48_000 };
        }
        _ => return None,
    }
    Some(c)
}

/// True when `out`'s `field` equals `inp`'s. Geometry/rate compare across
/// variants (both `RawVideo` and `CompressedVideo` carry them); the scalar
/// identity / channels / sample_rate require the same variant.
fn field_eq(out: &Caps, inp: &Caps, field: ProbeField) -> bool {
    match field {
        ProbeField::Width => geo_width(out).zip(geo_width(inp)).is_some_and(|(a, b)| a == b),
        ProbeField::Height => geo_height(out).zip(geo_height(inp)).is_some_and(|(a, b)| a == b),
        ProbeField::Framerate => geo_rate(out).zip(geo_rate(inp)).is_some_and(|(a, b)| a == b),
        ProbeField::Format => match (out, inp) {
            (Caps::RawVideo { format: a, .. }, Caps::RawVideo { format: b, .. }) => a == b,
            (Caps::CompressedVideo { codec: a, .. }, Caps::CompressedVideo { codec: b, .. }) => a == b,
            (Caps::Audio { format: a, .. }, Caps::Audio { format: b, .. }) => a == b,
            _ => false,
        },
        ProbeField::Channels => match (out, inp) {
            (Caps::Audio { channels: a, .. }, Caps::Audio { channels: b, .. }) => a == b,
            _ => false,
        },
        ProbeField::SampleRate => match (out, inp) {
            (Caps::Audio { sample_rate: a, .. }, Caps::Audio { sample_rate: b, .. }) => a == b,
            _ => false,
        },
    }
}

fn geo_width(c: &Caps) -> Option<&Dim> {
    match c {
        Caps::RawVideo { width, .. } | Caps::CompressedVideo { width, .. } => Some(width),
        _ => None,
    }
}

fn geo_height(c: &Caps) -> Option<&Dim> {
    match c {
        Caps::RawVideo { height, .. } | Caps::CompressedVideo { height, .. } => Some(height),
        _ => None,
    }
}

fn geo_rate(c: &Caps) -> Option<&Rate> {
    match c {
        Caps::RawVideo { framerate, .. } | Caps::CompressedVideo { framerate, .. } => Some(framerate),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    fn video(width: Dim, height: Dim, framerate: Rate) -> Caps {
        Caps::RawVideo { format: RawVideoFormat::Rgba8, width, height, framerate }
    }

    #[test]
    fn discover_passthrough_decoder_geometry_and_framerate() {
        // H264 -> Nv12: geometry + framerate copied through, format retargeted.
        let dec = |input: &Caps| match input {
            Caps::CompressedVideo { width, height, framerate, .. } => CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        };
        let sample = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let pt = discover_passthrough(&dec, &sample);
        assert!(pt.width && pt.height && pt.framerate, "geometry + rate copied through");
        assert!(!pt.format, "codec -> format is retargeted, not passthrough");
    }

    #[test]
    fn discover_passthrough_none_for_fixed_output() {
        // Output ignores the input (fixed dims): nothing invertible to discover.
        let dec = |_: &Caps| {
            CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Fixed(30 << 16),
            })
        };
        let sample = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(discover_passthrough(&dec, &sample), PassthroughFields::NONE);
    }

    #[test]
    fn discover_passthrough_identity_convert_all_fields() {
        // RawVideo -> RawVideo identity: every probed field passes through.
        let id = |input: &Caps| CapsSet::one(input.clone());
        let pt = discover_passthrough(&id, &video(Dim::Any, Dim::Any, Rate::Any));
        assert!(pt.width && pt.height && pt.framerate && pt.format);
    }

    #[test]
    fn discover_passthrough_scaler_retargets_geometry_only() {
        // A scaler fixes output geometry but keeps format + framerate: those two
        // are passthrough, width/height are not.
        let scale = |input: &Caps| match input {
            Caps::RawVideo { format, framerate, .. } => CapsSet::one(Caps::RawVideo {
                format: *format,
                width: Dim::Fixed(320),
                height: Dim::Fixed(240),
                framerate: framerate.clone(),
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        };
        let pt = discover_passthrough(&scale, &video(Dim::Any, Dim::Any, Rate::Any));
        assert!(pt.format && pt.framerate, "format + rate kept");
        assert!(!pt.width && !pt.height, "geometry is retargeted by the scaler");
    }

    #[test]
    fn discover_passthrough_none_for_multivalued_closure() {
        // A converter that offers {passthrough, retargeted-NV12} for an RGBA input
        // is multi-valued on its own sample, but coincidentally single-valued at
        // the format-probe values (Nv12 / I420, neither in `from`). Per-field
        // probing alone would mis-read `format` as passthrough and then drop the
        // RGBA input when coupling it against an NV12 pin (the M257 startup-failure
        // bug). The single-valued gate on the sample makes discovery bail to NONE.
        let from = [RawVideoFormat::Rgba8];
        let conv = move |input: &Caps| {
            let mut alts = vec![input.clone()];
            if let Caps::RawVideo { format, width, height, framerate } = input {
                if from.contains(format) {
                    alts.push(Caps::RawVideo {
                        format: RawVideoFormat::Nv12,
                        width: width.clone(),
                        height: height.clone(),
                        framerate: framerate.clone(),
                    });
                }
            }
            CapsSet::from_alternatives(alts)
        };
        let sample = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
        };
        assert_eq!(discover_passthrough(&conv, &sample), PassthroughFields::NONE);
    }

    #[test]
    fn verify_passthrough_sound_accepts_honoured_mask() {
        // A scaler keeps format + framerate, retargets geometry. A mask declaring
        // exactly the honoured fields is sound, even though the closure is
        // multi-valued (passthrough + scalable range), which `discover_passthrough`
        // could not verify.
        let scale = |input: &Caps| match input {
            Caps::RawVideo { format, width, height, framerate } => CapsSet::from_alternatives(vec![
                Caps::RawVideo {
                    format: *format,
                    width: width.clone(),
                    height: height.clone(),
                    framerate: framerate.clone(),
                },
                Caps::RawVideo {
                    format: *format,
                    width: Dim::Range { min: 1, max: 8192 },
                    height: Dim::Range { min: 1, max: 8192 },
                    framerate: framerate.clone(),
                },
            ]),
            _ => CapsSet::from_alternatives(Vec::new()),
        };
        let sample = video(Dim::Fixed(640), Dim::Fixed(480), Rate::Fixed(30 << 16));
        let honoured = PassthroughFields::NONE.with_format().with_framerate();
        assert!(
            verify_passthrough_sound(&scale, honoured, &sample),
            "format + framerate are genuinely passed through in every alternative"
        );
    }

    #[test]
    fn verify_passthrough_sound_rejects_overclaiming_mask() {
        // The same scaler, but a mask that also claims `width` passthrough: the
        // closure retargets width (one alternative is a Range, not the input's
        // Fixed), so the mask is unsound and the guard catches it.
        let scale = |input: &Caps| match input {
            Caps::RawVideo { format, framerate, .. } => CapsSet::from_alternatives(vec![
                Caps::RawVideo {
                    format: *format,
                    width: Dim::Range { min: 1, max: 8192 },
                    height: Dim::Range { min: 1, max: 8192 },
                    framerate: framerate.clone(),
                },
            ]),
            _ => CapsSet::from_alternatives(Vec::new()),
        };
        let sample = video(Dim::Fixed(640), Dim::Fixed(480), Rate::Fixed(30 << 16));
        let overclaim = PassthroughFields::NONE.with_format().with_framerate().with_width();
        assert!(
            !verify_passthrough_sound(&scale, overclaim, &sample),
            "claiming width passthrough when the closure retargets it is unsound"
        );
    }

    #[test]
    fn verify_passthrough_sound_passes_when_closure_rejects_input() {
        // A closure that rejects the sample (empty output) has nothing to verify,
        // so any mask is vacuously sound (the solve fails loud elsewhere).
        let reject = |_: &Caps| CapsSet::from_alternatives(Vec::new());
        let sample = video(Dim::Fixed(640), Dim::Fixed(480), Rate::Fixed(30 << 16));
        let all = PassthroughFields::NONE
            .with_format()
            .with_width()
            .with_height()
            .with_framerate()
            .with_channels()
            .with_sample_rate();
        assert!(verify_passthrough_sound(&reject, all, &sample));
    }
}
