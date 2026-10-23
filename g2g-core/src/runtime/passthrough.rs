//! Passthrough-field negotiation helpers for the solver's mid-stream re-solve
//! (design/caps.md). A transform declares which caps fields it forwards
//! unchanged (`PassthroughFields`); these functions couple those fields across a
//! link (`couple_*`), project a feasible input from an output (`project_*`), and
//! discover a mask by probing an element's caps closure (`discover_*`) where the
//! element declares none. Solver-only, so they live under the runtime tree rather
//! than in the data-plane `caps.rs`. `project_passthrough` also serves every
//! runner's mid-stream forward resolve; `project_passthrough_derived` serves only
//! the std graph runner's backward-feasibility sweep and stays `std`-gated.

#[cfg(feature = "std")]
use crate::caps::ANY_SAMPLE_RATE;
use crate::caps::{
    intersect_channels, intersect_sample_rate, AudioFormat, Caps, CapsSet, Colorimetry, Dim,
    Interlace, PassthroughFields, Rate, RawVideoFormat, VideoCodec,
};

/// Narrow `input` by intersecting each *passthrough* field against the
/// corresponding field of `pin` (the field-level backward coupling: e.g.
/// `Range(1..MAX) ∩ Fixed(160) = Fixed(160)`). Retarget fields (not in `mask`)
/// are left as `input` carries them, since the transform sets them
/// independently of its input. Same media variant required; `None` if a
/// passthrough field has no overlap (the alternative dies) or the variants
/// differ. Used by the solver's `DerivedFields` backward sweep.
pub(crate) fn couple_passthrough(
    input: &Caps,
    pin: &Caps,
    mask: PassthroughFields,
) -> Option<Caps> {
    match (input, pin) {
        (
            Caps::RawVideo {
                format: fi,
                width: wi,
                height: hi,
                framerate: ri,
                interlace: ii,
                colorimetry: cli,
            },
            Caps::RawVideo {
                format: fp,
                width: wp,
                height: hp,
                framerate: rp,
                ..
            },
        ) => {
            let format = if mask.format {
                if fi != fp {
                    return None;
                }
                *fi
            } else {
                *fi
            };
            let width = if mask.width {
                wi.intersect(wp)?
            } else {
                wi.clone()
            };
            let height = if mask.height {
                hi.intersect(hp)?
            } else {
                hi.clone()
            };
            let framerate = if mask.framerate {
                ri.intersect(rp)?
            } else {
                ri.clone()
            };
            Some(Caps::RawVideo {
                format,
                width,
                height,
                framerate,
                // Interlace is a runtime-refined hint, not a maskable field:
                // coupling it backward could only kill a link over a value the
                // decoder has not produced yet, so the input keeps its own.
                interlace: *ii,
                // Colorimetry is the same kind of runtime-refined hint.
                colorimetry: *cli,
            })
        }
        (
            Caps::CompressedVideo {
                codec: ci,
                width: wi,
                height: hi,
                framerate: ri,
                colorimetry: cli,
                alignment: ai,
            },
            Caps::CompressedVideo {
                codec: cp,
                width: wp,
                height: hp,
                framerate: rp,
                ..
            },
        ) => {
            let codec = if mask.format {
                if ci != cp {
                    return None;
                }
                *ci
            } else {
                *ci
            };
            let width = if mask.width {
                wi.intersect(wp)?
            } else {
                wi.clone()
            };
            let height = if mask.height {
                hi.intersect(hp)?
            } else {
                hi.clone()
            };
            let framerate = if mask.framerate {
                ri.intersect(rp)?
            } else {
                ri.clone()
            };
            Some(Caps::CompressedVideo {
                codec,
                width,
                height,
                framerate,
                // Kept from the input like RawVideo's (see there).
                colorimetry: *cli,
                alignment: *ai,
            })
        }
        (
            Caps::Audio {
                format: fi,
                channels: ci,
                sample_rate: si,
                ..
            },
            Caps::Audio {
                format: fp,
                channels: cp,
                sample_rate: sp,
                ..
            },
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
                // `ANY_CHANNELS` (0) is a wildcard: an unknown input channel count
                // couples to a concrete downstream pin (mirrors `Caps::intersect`).
                intersect_channels(*ci, *cp)?
            } else {
                *ci
            };
            let sample_rate = if mask.sample_rate {
                intersect_sample_rate(*si, *sp)?
            } else {
                *si
            };
            Some(Caps::Audio {
                format,
                channels,
                sample_rate,
                channel_layout: crate::ChannelLayout::UNSPECIFIED,
            })
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
pub(crate) fn couple_passthrough_derived(
    input: &Caps,
    pin: &Caps,
    mask: PassthroughFields,
) -> Option<Caps> {
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
    let width = if mask.width {
        wi.intersect(wp)?
    } else {
        wi.clone()
    };
    let height = if mask.height {
        hi.intersect(hp)?
    } else {
        hi.clone()
    };
    let framerate = if mask.framerate {
        ri.intersect(rp)?
    } else {
        ri.clone()
    };
    match input {
        Caps::RawVideo {
            format,
            interlace,
            colorimetry,
            ..
        } => Some(Caps::RawVideo {
            format: *format,
            width,
            height,
            framerate,
            interlace: *interlace,
            colorimetry: *colorimetry,
        }),
        Caps::CompressedVideo {
            codec, colorimetry, ..
        } => Some(Caps::CompressedVideo {
            codec: *codec,
            width,
            height,
            framerate,
            colorimetry: *colorimetry,
            alignment: crate::BitstreamAlignment::Unspecified,
        }),
        _ => None,
    }
}

/// Project an output-side feasible `out` onto the *input* side of a
/// `DerivedFields` transform: keep passthrough fields, widen each retargeted
/// field to "anything the transform can take" (`Dim`/`Rate` -> `Any`,
/// `sample_rate` -> [`ANY_SAMPLE_RATE`]). Returns `None` when a retargeted field
/// is a non-rangeable scalar (`format` / `codec` / `channels`) with no wildcard,
/// i.e. the input feasibility can't be expressed as a single `Caps` (the solver
/// then imposes no upstream feasibility constraint, the status quo). Used by
/// `backward_feasible` for the mid-stream snapshot and by
/// `resolve_forward_output` to widen a previous output into the shape to keep.
#[cfg(feature = "std")]
pub(crate) fn project_passthrough(out: &Caps, mask: PassthroughFields) -> Option<Caps> {
    match out {
        Caps::RawVideo {
            format,
            width,
            height,
            framerate,
            ..
        } => {
            if !mask.format {
                return None; // retargeted format has no wildcard
            }
            Some(Caps::RawVideo {
                format: *format,
                width: if mask.width { width.clone() } else { Dim::Any },
                height: if mask.height {
                    height.clone()
                } else {
                    Dim::Any
                },
                framerate: if mask.framerate {
                    framerate.clone()
                } else {
                    Rate::Any
                },
                // A feasibility projection imposes no scan constraint upstream,
                // and no colorimetry one either.
                interlace: Interlace::Any,
                colorimetry: Colorimetry::UNKNOWN,
            })
        }
        Caps::CompressedVideo {
            codec,
            width,
            height,
            framerate,
            ..
        } => {
            if !mask.format {
                return None;
            }
            Some(Caps::CompressedVideo {
                codec: *codec,
                width: if mask.width { width.clone() } else { Dim::Any },
                height: if mask.height {
                    height.clone()
                } else {
                    Dim::Any
                },
                framerate: if mask.framerate {
                    framerate.clone()
                } else {
                    Rate::Any
                },
                colorimetry: Colorimetry::UNKNOWN,
                alignment: crate::BitstreamAlignment::Unspecified,
            })
        }
        Caps::Audio {
            format,
            channels,
            sample_rate,
            ..
        } => {
            if !mask.format || !mask.channels {
                return None; // no format / channel wildcard
            }
            Some(Caps::Audio {
                format: *format,
                channels: *channels,
                sample_rate: if mask.sample_rate {
                    *sample_rate
                } else {
                    ANY_SAMPLE_RATE
                },
                channel_layout: crate::ChannelLayout::UNSPECIFIED,
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
    let framerate = if mask.framerate {
        rp.clone()
    } else {
        Rate::Any
    };
    match sample {
        Caps::RawVideo { format, .. } => Some(Caps::RawVideo {
            format: *format,
            width,
            height,
            framerate,
            interlace: Interlace::Any,
            colorimetry: Colorimetry::UNKNOWN,
        }),
        Caps::CompressedVideo { codec, .. } => Some(Caps::CompressedVideo {
            codec: *codec,
            width,
            height,
            framerate,
            colorimetry: Colorimetry::UNKNOWN,
            alignment: crate::BitstreamAlignment::Unspecified,
        }),
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
/// [`DerivedFields`](crate::format_element::CapsConstraint::DerivedFields) mask
/// does, the "invertible fields of a `DerivedOutput`". `f` is not analytically
/// invertible, but it is evaluable, so a field's behaviour is read off two
/// concrete probes.
///
/// Conservative by construction: a field is marked passthrough only when the
/// sample and two distinct concrete probes *all* show every output alternative's
/// field equal to the probed input field. A closure that rejects a probe, fixes
/// the field (a retargeted decoder format), or retargets it on any one of several
/// output alternatives yields `false` for that field, so discovery never invents
/// coupling that is not there (a wrong `true` would narrow the input
/// incorrectly). A decoder offering several output formats still passes its
/// geometry and rate through on each of them. `sample` is a
/// representative input alternative; its geometry is concretised first so a
/// `Range`/`Any` input field does not confuse the equality test.
pub(crate) fn discover_passthrough(
    f: &dyn Fn(&Caps) -> CapsSet,
    sample: &Caps,
) -> PassthroughFields {
    let base = concrete_probe_base(sample);
    PassthroughFields {
        width: probe_field(f, &base, ProbeField::Width),
        height: probe_field(f, &base, ProbeField::Height),
        framerate: probe_field(f, &base, ProbeField::Framerate),
        format: probe_field(f, &base, ProbeField::Format),
        channels: probe_field(f, &base, ProbeField::Channels),
        sample_rate: probe_field(f, &base, ProbeField::SampleRate),
    }
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
            interlace: Interlace::Any,
            colorimetry: Colorimetry::UNKNOWN,
        },
        Caps::CompressedVideo { codec, .. } => Caps::CompressedVideo {
            codec: *codec,
            width: Dim::Fixed(64),
            height: Dim::Fixed(64),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: Colorimetry::UNKNOWN,
            alignment: crate::BitstreamAlignment::Unspecified,
        },
        Caps::Audio { format, .. } => Caps::Audio {
            format: *format,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: crate::ChannelLayout::UNSPECIFIED,
        },
        other => other.clone(),
    }
}

/// True when `f` passes `field` through: on the sample's own input and on two
/// concrete probes that differ only in `field`, every output alternative's
/// `field` equals the input's.
fn probe_field(f: &dyn Fn(&Caps) -> CapsSet, base: &Caps, field: ProbeField) -> bool {
    let (Some(p0), Some(p1)) = (set_probe(base, field, false), set_probe(base, field, true)) else {
        return false;
    };
    [base, &p0, &p1]
        .iter()
        .all(|input| every_output_keeps_field(f, input, field))
}

// a converter offering {passthrough, retargeted} keeps the field on one alternative only
fn every_output_keeps_field(f: &dyn Fn(&Caps) -> CapsSet, input: &Caps, field: ProbeField) -> bool {
    let outputs = f(input);
    !outputs.is_empty()
        && outputs
            .alternatives()
            .iter()
            .all(|output| field_eq(output, input, field))
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
            *format = if hi {
                RawVideoFormat::I420
            } else {
                RawVideoFormat::Nv12
            };
        }
        (Caps::CompressedVideo { codec, .. }, ProbeField::Format) => {
            *codec = if hi {
                VideoCodec::H265
            } else {
                VideoCodec::H264
            };
        }
        (Caps::Audio { format, .. }, ProbeField::Format) => {
            *format = if hi {
                AudioFormat::PcmF32Le
            } else {
                AudioFormat::PcmS16Le
            };
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
        ProbeField::Width => geo_width(out)
            .zip(geo_width(inp))
            .is_some_and(|(a, b)| a == b),
        ProbeField::Height => geo_height(out)
            .zip(geo_height(inp))
            .is_some_and(|(a, b)| a == b),
        ProbeField::Framerate => geo_rate(out)
            .zip(geo_rate(inp))
            .is_some_and(|(a, b)| a == b),
        ProbeField::Format => match (out, inp) {
            (Caps::RawVideo { format: a, .. }, Caps::RawVideo { format: b, .. }) => a == b,
            (Caps::CompressedVideo { codec: a, .. }, Caps::CompressedVideo { codec: b, .. }) => {
                a == b
            }
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
        Caps::RawVideo { framerate, .. } | Caps::CompressedVideo { framerate, .. } => {
            Some(framerate)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    fn video(width: Dim, height: Dim, framerate: Rate) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width,
            height,
            framerate,
            interlace: crate::Interlace::Any,
            colorimetry: crate::Colorimetry::UNKNOWN,
        }
    }

    #[test]
    fn discover_passthrough_decoder_geometry_and_framerate() {
        // H264 -> Nv12: geometry + framerate copied through, format retargeted.
        let dec = |input: &Caps| match input {
            Caps::CompressedVideo {
                width,
                height,
                framerate,
                ..
            } => CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
                interlace: crate::Interlace::Any,
                colorimetry: crate::Colorimetry::UNKNOWN,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        };
        let sample = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: crate::Colorimetry::UNKNOWN,
            alignment: crate::BitstreamAlignment::Unspecified,
        };
        let pt = discover_passthrough(&dec, &sample);
        assert!(
            pt.width && pt.height && pt.framerate,
            "geometry + rate copied through"
        );
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
                interlace: crate::Interlace::Any,
                colorimetry: crate::Colorimetry::UNKNOWN,
            })
        };
        let sample = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: crate::Colorimetry::UNKNOWN,
            alignment: crate::BitstreamAlignment::Unspecified,
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
            Caps::RawVideo {
                format, framerate, ..
            } => CapsSet::one(Caps::RawVideo {
                format: *format,
                width: Dim::Fixed(320),
                height: Dim::Fixed(240),
                framerate: framerate.clone(),
                interlace: crate::Interlace::Any,
                colorimetry: crate::Colorimetry::UNKNOWN,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        };
        let pt = discover_passthrough(&scale, &video(Dim::Any, Dim::Any, Rate::Any));
        assert!(pt.format && pt.framerate, "format + rate kept");
        assert!(
            !pt.width && !pt.height,
            "geometry is retargeted by the scaler"
        );
    }

    #[test]
    fn discover_passthrough_multivalued_closure_retargets_format() {
        // A converter that offers {passthrough, retargeted-NV12} for an RGBA input
        // is multi-valued on its own sample, but coincidentally single-valued at
        // the format-probe values (Nv12 / I420, neither in `from`). Probing only
        // those would mis-read `format` as passthrough and then drop the RGBA
        // input when coupling it against an NV12 pin (the M257 startup-failure
        // bug). The sample's NV12 alternative marks `format` retargeted, while
        // geometry and rate, kept on both alternatives, still couple.
        let from = [RawVideoFormat::Rgba8];
        let conv = move |input: &Caps| {
            let mut alts = vec![input.clone()];
            if let Caps::RawVideo {
                format,
                width,
                height,
                framerate,
                ..
            } = input
            {
                if from.contains(format) {
                    alts.push(Caps::RawVideo {
                        format: RawVideoFormat::Nv12,
                        width: width.clone(),
                        height: height.clone(),
                        framerate: framerate.clone(),
                        interlace: crate::Interlace::Any,
                        colorimetry: crate::Colorimetry::UNKNOWN,
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
            interlace: crate::Interlace::Any,
            colorimetry: crate::Colorimetry::UNKNOWN,
        };
        let pt = discover_passthrough(&conv, &sample);
        assert!(!pt.format, "the NV12 alternative retargets the format");
        assert!(
            pt.width && pt.height && pt.framerate,
            "every alternative keeps geometry + rate"
        );
    }

    #[test]
    fn discover_passthrough_couples_geometry_of_a_multi_format_decoder() {
        // A decoder offering two output formats (the bit depth decides which)
        // copies geometry and rate onto both, so a downstream geometry pin couples.
        let dec = |input: &Caps| match input {
            Caps::CompressedVideo {
                width,
                height,
                framerate,
                ..
            } => CapsSet::from_alternatives(
                [RawVideoFormat::Nv12, RawVideoFormat::P010]
                    .into_iter()
                    .map(|format| Caps::RawVideo {
                        format,
                        width: width.clone(),
                        height: height.clone(),
                        framerate: framerate.clone(),
                        interlace: crate::Interlace::Any,
                        colorimetry: crate::Colorimetry::UNKNOWN,
                    })
                    .collect(),
            ),
            _ => CapsSet::from_alternatives(Vec::new()),
        };
        let sample = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: crate::Colorimetry::UNKNOWN,
            alignment: crate::BitstreamAlignment::Unspecified,
        };
        let pt = discover_passthrough(&dec, &sample);
        assert!(pt.width && pt.height && pt.framerate);
        assert!(!pt.format, "codec -> format is retargeted, not passthrough");
    }
}
