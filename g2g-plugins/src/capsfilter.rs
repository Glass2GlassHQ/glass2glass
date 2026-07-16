//! Caps filter: a pass-through transform that forces a negotiation-time
//! narrowing (DESIGN.md §4.13.1). Data flows through unchanged;
//! the element's only job is to constrain the link to a specific
//! `CapsSet` so the solver narrows the chain to it.
//!
//! Native constraint is `Identity(set)`: input == output, both drawn from
//! the filter set. Insert one anywhere a downstream peer is too permissive
//! (e.g. an `AcceptsAny` sink) and you need to pin a concrete format.
//!
//! Per the transform contract (see `run_source_transform_sink`), this
//! element does NOT emit `Eos` itself — the runner forwards the EOS
//! sentinel after `process(Eos)` returns.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
    VideoCodec,
};

#[derive(Debug)]
pub struct CapsFilter {
    filter: CapsSet,
    /// The `caps` property string, kept so `get_property` round-trips it.
    caps_str: String,
    forwarded: u64,
    configured: bool,
}

impl Default for CapsFilter {
    /// An empty filter (accepts nothing) until the `caps` property is set; the
    /// `parse_launch` / registry path always sets it before negotiation.
    fn default() -> Self {
        Self::from_set(CapsSet::from_alternatives(Vec::new()))
    }
}

impl CapsFilter {
    /// Filter to a single concrete description (the common case: force
    /// one format / geometry).
    pub fn new(caps: Caps) -> Self {
        Self::from_set(CapsSet::one(caps))
    }

    /// Filter to a preference-ordered set of alternatives.
    pub fn from_set(filter: CapsSet) -> Self {
        Self {
            filter,
            caps_str: String::new(),
            forwarded: 0,
            configured: false,
        }
    }

    pub fn forwarded(&self) -> u64 {
        self.forwarded
    }
}

impl AsyncElement for CapsFilter {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Legacy / mixed-cascade path: narrow upstream against the filter,
        // honoring the set's preference order. The native solver uses the
        // `Identity` constraint below instead.
        for alt in self.filter.alternatives() {
            if let Ok(narrowed) = upstream_caps.intersect(alt) {
                return Ok(narrowed);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    /// Native pass-through constraint pinned to the filter set. The solver
    /// couples input and output links and narrows both to this set.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(self.filter.clone())
    }

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        // The solver should only ever hand us caps the filter accepts;
        // fail loud if it didn't (a negotiation bug, not a runtime state).
        if !self.filter.accepts(absolute_caps) {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(f) => {
                    self.forwarded += 1;
                    out.push(PipelinePacket::DataFrame(f)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // Enforce the filter mid-stream too: a change that the
                    // filter rejects is a pipeline error, surfaced loud.
                    if !self.filter.accepts(&c) {
                        return Err(G2gError::CapsMismatch);
                    }
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        CAPSFILTER_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "caps" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                let set = parse_caps_set(s).ok_or(PropError::Value)?;
                if set.alternatives().is_empty() {
                    return Err(PropError::Value);
                }
                self.filter = set;
                self.caps_str = s.into();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "caps" if !self.caps_str.is_empty() => Some(PropValue::Str(self.caps_str.clone())),
            _ => None,
        }
    }
}

/// `CapsFilter`'s settable properties (M117).
static CAPSFILTER_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "caps",
    PropKind::Str,
    "caps to filter to, gst-launch syntax: e.g. video/x-raw,format=nv12,width=320,height=240",
)];

/// The raw pixel formats a format-less `video/x-raw` expands to (M184). Order is
/// the preference the solver fixates by when several survive; in practice the
/// upstream format narrows it to one.
const RAW_VIDEO_FORMATS: [RawVideoFormat; 5] = [
    RawVideoFormat::Nv12,
    RawVideoFormat::I420,
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
    RawVideoFormat::Yuyv,
];

/// The raw sample formats a format-less `audio/x-raw` expands to (M184).
const RAW_AUDIO_FORMATS: [AudioFormat; 2] = [AudioFormat::PcmS16Le, AudioFormat::PcmF32Le];

/// A parsed caps field value: a fixed scalar (`width=640`), a `[min,max]` range
/// (`width=[1,1920]`), or a `{a,b,...}` list (`format={I420,NV12}`). A range maps
/// to `Dim::Range` / `Rate::Range` within one caps; a list expands to alternatives
/// in the returned `CapsSet` (the gst idiom, so a launch caps filter constrains
/// negotiation without over-fixing).
enum FieldVal<'a> {
    One(&'a str),
    Range(&'a str, &'a str),
    List(Vec<&'a str>),
}

/// Split on top-level commas only, so the commas inside a `[..]` range or `{..}`
/// list are not mistaken for field separators.
fn split_top_commas(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '[' | '{' => depth += 1,
            ']' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

fn parse_field_val(v: &str) -> FieldVal<'_> {
    let v = v.trim();
    if let Some(inner) = v.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let p = split_top_commas(inner);
        // gst ranges are `[min,max]`; a third `step` element is ignored.
        if p.len() >= 2 {
            return FieldVal::Range(p[0].trim(), p[1].trim());
        }
    }
    if let Some(inner) = v.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
        return FieldVal::List(
            split_top_commas(inner).into_iter().map(str::trim).filter(|s| !s.is_empty()).collect(),
        );
    }
    FieldVal::One(v)
}

// Expand a dimension field into its constraint(s): a fixed value or `Any`, a
// `Range`, or (for a list) one `Fixed` per alternative. A present-but-unparseable
// fixed value stays lenient (`Any`), as before; a range / list must parse or the
// whole caps is rejected (`None`).
fn expand_dim(fv: Option<&FieldVal>) -> Option<Vec<Dim>> {
    Some(match fv {
        None => alloc::vec![Dim::Any],
        Some(FieldVal::One(s)) => alloc::vec![s.parse::<u32>().map_or(Dim::Any, Dim::Fixed)],
        Some(FieldVal::Range(a, b)) => {
            alloc::vec![Dim::Range { min: a.parse().ok()?, max: b.parse().ok()? }]
        }
        Some(FieldVal::List(xs)) => {
            xs.iter().map(|x| x.parse::<u32>().ok().map(Dim::Fixed)).collect::<Option<Vec<_>>>()?
        }
    })
}

fn expand_rate(fv: Option<&FieldVal>) -> Option<Vec<Rate>> {
    Some(match fv {
        None => alloc::vec![Rate::Any],
        Some(FieldVal::One(s)) => alloc::vec![parse_rate(s).unwrap_or(Rate::Any)],
        Some(FieldVal::Range(a, b)) => {
            alloc::vec![Rate::Range { min_q16: rate_q16(a)?, max_q16: rate_q16(b)? }]
        }
        Some(FieldVal::List(xs)) => xs.iter().map(|x| parse_rate(x)).collect::<Option<Vec<_>>>()?,
    })
}

fn expand_raw_format(fv: Option<&FieldVal>) -> Option<Vec<RawVideoFormat>> {
    Some(match fv {
        None => RAW_VIDEO_FORMATS.to_vec(),
        Some(FieldVal::One(s)) => alloc::vec![parse_raw_format(s)?],
        Some(FieldVal::List(xs)) => {
            xs.iter().map(|x| parse_raw_format(x)).collect::<Option<Vec<_>>>()?
        }
        Some(FieldVal::Range(..)) => return None, // a format range is meaningless
    })
}

fn expand_audio_format(fv: Option<&FieldVal>) -> Option<Vec<AudioFormat>> {
    Some(match fv {
        None => RAW_AUDIO_FORMATS.to_vec(),
        Some(FieldVal::One(s)) => alloc::vec![parse_audio_format(s)?],
        Some(FieldVal::List(xs)) => {
            xs.iter().map(|x| parse_audio_format(x)).collect::<Option<Vec<_>>>()?
        }
        Some(FieldVal::Range(..)) => return None,
    })
}

// `Caps::Audio` holds scalar channels (u8) / sample_rate (u32) with no range
// type, so a range is rejected; a list expands to alternatives.
fn expand_u8(fv: Option<&FieldVal>, default: u8) -> Option<Vec<u8>> {
    Some(match fv {
        None => alloc::vec![default],
        Some(FieldVal::One(s)) => alloc::vec![s.parse().unwrap_or(default)],
        Some(FieldVal::List(xs)) => xs.iter().map(|x| x.parse::<u8>().ok()).collect::<Option<Vec<_>>>()?,
        Some(FieldVal::Range(..)) => return None,
    })
}

fn expand_u32(fv: Option<&FieldVal>, default: u32) -> Option<Vec<u32>> {
    Some(match fv {
        None => alloc::vec![default],
        Some(FieldVal::One(s)) => alloc::vec![s.parse().unwrap_or(default)],
        Some(FieldVal::List(xs)) => xs.iter().map(|x| x.parse::<u32>().ok()).collect::<Option<Vec<_>>>()?,
        Some(FieldVal::Range(..)) => return None,
    })
}

/// Parse a `gst-launch` caps description (`media/type,field=value,...`) into a
/// [`CapsSet`]. Field values may be fixed (`width=640`), a `[min,max]` range
/// (`width=[1,1920]`, mapped to `Dim::Range` / `Rate::Range`), or a `{a,b,...}`
/// list (`format={I420,NV12}`, expanded to alternatives). A `video/x-raw` /
/// `audio/x-raw` with no `format` expands to all supported raw formats at the
/// given geometry (the gst-idiomatic format-less caps). `None` on an unknown
/// media type or an unparseable range / list. Format values are case-insensitive
/// (GStreamer's uppercase or the historical lowercase, M182).
pub fn parse_caps_set(desc: &str) -> Option<CapsSet> {
    let mut parts = split_top_commas(desc).into_iter();
    let media = parts.next()?.trim();
    let fields: Vec<(&str, FieldVal)> =
        parts.filter_map(|p| p.split_once('=')).map(|(k, v)| (k.trim(), parse_field_val(v))).collect();
    let fv = |key: &str| fields.iter().find(|(k, _)| *k == key).map(|(_, v)| v);

    // Cartesian product of the list-valued fields; range fields stay as one
    // `Range` inside each alternative.
    let compressed_set = |codec: VideoCodec| -> Option<CapsSet> {
        let (widths, heights, rates) = (expand_dim(fv("width"))?, expand_dim(fv("height"))?, expand_rate(fv("framerate"))?);
        let mut alts = Vec::new();
        for w in &widths {
            for h in &heights {
                for r in &rates {
                    alts.push(compressed(codec, w.clone(), h.clone(), r.clone()));
                }
            }
        }
        Some(CapsSet::from_alternatives(alts))
    };
    let audio_set = |formats: &[AudioFormat]| -> Option<CapsSet> {
        let (channels, rates) = (expand_u8(fv("channels"), 2)?, expand_u32(fv("rate"), 48_000)?);
        let mut alts = Vec::new();
        for &format in formats {
            for &ch in &channels {
                for &sr in &rates {
                    alts.push(Caps::Audio { format, channels: ch, sample_rate: sr });
                }
            }
        }
        Some(CapsSet::from_alternatives(alts))
    };

    match media {
        "video/x-raw" => {
            let (formats, widths, heights, rates) = (
                expand_raw_format(fv("format"))?,
                expand_dim(fv("width"))?,
                expand_dim(fv("height"))?,
                expand_rate(fv("framerate"))?,
            );
            let mut alts = Vec::new();
            for &format in &formats {
                for w in &widths {
                    for h in &heights {
                        for r in &rates {
                            alts.push(Caps::RawVideo {
                                format,
                                width: w.clone(),
                                height: h.clone(),
                                framerate: r.clone(),
                            });
                        }
                    }
                }
            }
            Some(CapsSet::from_alternatives(alts))
        }
        "audio/x-raw" => audio_set(&expand_audio_format(fv("format"))?),
        "video/x-h264" => compressed_set(VideoCodec::H264),
        "video/x-h265" => compressed_set(VideoCodec::H265),
        "video/x-vp8" => compressed_set(VideoCodec::Vp8),
        "video/x-vp9" => compressed_set(VideoCodec::Vp9),
        "video/x-av1" => compressed_set(VideoCodec::Av1),
        "image/jpeg" => compressed_set(VideoCodec::Mjpeg),
        "audio/x-opus" => audio_set(&[AudioFormat::Opus]),
        // gst names AAC `audio/mpeg` (with mpegversion=4, which we don't require).
        "audio/mpeg" => audio_set(&[AudioFormat::Aac]),
        _ => None,
    }
}

/// Parse a `gst-launch` caps description into a single concrete [`Caps`]. Returns
/// `None` when the description expands to more than one alternative (a
/// format-less raw caps, see [`parse_caps_set`]) or is unparseable.
pub fn parse_caps(desc: &str) -> Option<Caps> {
    let set = parse_caps_set(desc)?;
    match set.alternatives() {
        [only] => Some(only.clone()),
        _ => None,
    }
}

fn compressed(codec: VideoCodec, width: Dim, height: Dim, framerate: Rate) -> Caps {
    Caps::CompressedVideo { codec, width, height, framerate }
}

// GStreamer caps name formats uppercase (NV12, RGBA, YUY2, S16LE); accept any
// case and the historical lowercase spellings so both port.
pub(crate) fn parse_raw_format(s: &str) -> Option<RawVideoFormat> {
    Some(match s.to_ascii_lowercase().as_str() {
        "rgba" => RawVideoFormat::Rgba8,
        "bgra" => RawVideoFormat::Bgra8,
        "nv12" => RawVideoFormat::Nv12,
        "i420" => RawVideoFormat::I420,
        "yuyv" | "yuy2" => RawVideoFormat::Yuyv,
        "i420_10le" => RawVideoFormat::I420p10,
        "i420_12le" => RawVideoFormat::I420p12,
        "y42b" => RawVideoFormat::I422,
        "i422_10le" => RawVideoFormat::I422p10,
        "i422_12le" => RawVideoFormat::I422p12,
        "y444" => RawVideoFormat::I444,
        "y444_10le" => RawVideoFormat::I444p10,
        "y444_12le" => RawVideoFormat::I444p12,
        _ => return None,
    })
}

fn parse_audio_format(s: &str) -> Option<AudioFormat> {
    Some(match s.to_ascii_lowercase().as_str() {
        "s16le" => AudioFormat::PcmS16Le,
        "f32le" => AudioFormat::PcmF32Le,
        _ => return None,
    })
}

/// Parse a framerate `num/den` (or bare integer) into a Q16 fixed-point value.
/// Shared by [`parse_rate`] and the `[min,max]` framerate-range expansion.
fn rate_q16(s: &str) -> Option<u32> {
    Some(match s.trim().split_once('/') {
        Some((n, d)) => {
            let n: u64 = n.trim().parse().ok()?;
            let d: u64 = d.trim().parse().ok()?;
            if d == 0 {
                return None;
            }
            ((n << 16) / d) as u32
        }
        None => (s.trim().parse::<u64>().ok()? << 16) as u32,
    })
}

/// Parse a framerate `num/den` (or bare integer) into a Q16 [`Rate::Fixed`].
fn parse_rate(s: &str) -> Option<Rate> {
    rate_q16(s).map(Rate::Fixed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{Dim, Rate, VideoCodec, RawVideoFormat};

    fn nv12(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        }
    }

    #[test]
    fn parses_high_bit_depth_and_alt_chroma_format_names() {
        // The GStreamer `format=` strings for the planar high-bit-depth / 4:2:2 /
        // 4:4:4 family resolve to the right variant (case-insensitively).
        for (s, want) in [
            ("I420_10LE", RawVideoFormat::I420p10),
            ("i420_12le", RawVideoFormat::I420p12),
            ("Y42B", RawVideoFormat::I422),
            ("I422_10LE", RawVideoFormat::I422p10),
            ("I422_12LE", RawVideoFormat::I422p12),
            ("Y444", RawVideoFormat::I444),
            ("Y444_10LE", RawVideoFormat::I444p10),
            ("Y444_12LE", RawVideoFormat::I444p12),
        ] {
            assert_eq!(parse_raw_format(s), Some(want), "format string {s}");
        }
    }

    #[test]
    fn caps_constraint_is_identity_of_filter() {
        let f = CapsFilter::new(nv12(1920, 1080));
        let CapsConstraint::Identity(set) = f.caps_constraint_as_transform() else {
            panic!("expected Identity");
        };
        assert_eq!(set.alternatives(), &[nv12(1920, 1080)]);
    }

    #[test]
    fn intercept_narrows_compatible_upstream() {
        // Filter on NV12/any-dims narrows an any-dims upstream to itself
        // and rejects a different format.
        let f = CapsFilter::new(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        });
        assert_eq!(f.intercept_caps(&nv12(1280, 720)), Ok(nv12(1280, 720)));

        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(f.intercept_caps(&h264), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn configure_rejects_caps_outside_filter() {
        let mut f = CapsFilter::new(nv12(1920, 1080));
        assert!(f.configure_pipeline(&nv12(1920, 1080)).is_ok());

        let mut g = CapsFilter::new(nv12(1920, 1080));
        assert_eq!(
            g.configure_pipeline(&nv12(1280, 720)).err(),
            Some(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn parse_caps_raw_video() {
        assert_eq!(
            parse_caps("video/x-raw,format=nv12,width=320,height=240,framerate=30/1"),
            Some(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(320),
                height: Dim::Fixed(240),
                framerate: Rate::Fixed(30 << 16),
            })
        );
        // Omitted dims default to Any; a missing format is rejected.
        assert_eq!(
            parse_caps("video/x-raw,format=rgba"),
            Some(Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            })
        );
        // `parse_caps` yields a single Caps, so a format-less (multi-format) raw
        // description is `None` here; `parse_caps_set` expands it instead.
        assert_eq!(parse_caps("video/x-raw,width=320"), None, "format-less is not a single caps");
    }

    #[test]
    fn parse_caps_set_expands_format_less_raw_video() {
        // No `format` -> all supported pixel formats at the fixed geometry (M184).
        let set = parse_caps_set("video/x-raw,width=160,height=120").expect("parses");
        assert_eq!(set.alternatives().len(), RAW_VIDEO_FORMATS.len());
        assert!(set.alternatives().iter().all(|c| matches!(
            c,
            Caps::RawVideo { width: Dim::Fixed(160), height: Dim::Fixed(120), .. }
        )));
        // Every supported format is represented at that geometry.
        for fmt in RAW_VIDEO_FORMATS {
            assert!(set.alternatives().iter().any(|c| matches!(
                c,
                Caps::RawVideo { format, .. } if *format == fmt
            )));
        }
        // A pinned format still yields exactly one alternative.
        assert_eq!(parse_caps_set("video/x-raw,format=NV12").unwrap().alternatives().len(), 1);
        // Format-less audio expands to the raw sample formats.
        assert_eq!(
            parse_caps_set("audio/x-raw,channels=2").unwrap().alternatives().len(),
            RAW_AUDIO_FORMATS.len()
        );
    }

    #[test]
    fn parse_caps_compressed_and_audio() {
        assert!(matches!(
            parse_caps("video/x-h264,width=1920,height=1080"),
            Some(Caps::CompressedVideo { codec: VideoCodec::H264, .. })
        ));
        assert!(matches!(
            parse_caps("video/x-vp9"),
            Some(Caps::CompressedVideo { codec: VideoCodec::Vp9, .. })
        ));
        assert_eq!(
            parse_caps("audio/x-opus,channels=2,rate=48000"),
            Some(Caps::Audio { format: g2g_core::AudioFormat::Opus, channels: 2, sample_rate: 48_000 })
        );
        assert_eq!(parse_caps("video/x-bogus"), None);
    }

    #[test]
    fn parse_caps_range_maps_to_dim_and_rate_range() {
        // `[min,max]` on width/height -> Dim::Range in one caps (not an expansion).
        let set = parse_caps_set("video/x-raw,format=nv12,width=[1,1920],height=[1,1080]").unwrap();
        assert_eq!(set.alternatives().len(), 1);
        let Caps::RawVideo { width, height, .. } = &set.alternatives()[0] else { panic!() };
        assert_eq!(*width, Dim::Range { min: 1, max: 1920 });
        assert_eq!(*height, Dim::Range { min: 1, max: 1080 });
        // A framerate range maps to Rate::Range.
        let set = parse_caps_set("video/x-h264,framerate=[0/1,60/1]").unwrap();
        let Caps::CompressedVideo { framerate, .. } = &set.alternatives()[0] else { panic!() };
        assert!(matches!(framerate, Rate::Range { .. }), "got {framerate:?}");
    }

    #[test]
    fn parse_caps_list_expands_to_alternatives() {
        // `format={I420,NV12}` -> two alternatives, geometry fixed on both.
        let set = parse_caps_set("video/x-raw,format={I420,NV12},width=640,height=480").unwrap();
        let fmts: Vec<RawVideoFormat> = set
            .alternatives()
            .iter()
            .map(|c| match c {
                Caps::RawVideo { format, width, .. } => {
                    assert_eq!(*width, Dim::Fixed(640));
                    *format
                }
                _ => panic!("raw video"),
            })
            .collect();
        assert_eq!(fmts.len(), 2);
        assert!(fmts.contains(&RawVideoFormat::I420) && fmts.contains(&RawVideoFormat::Nv12));
        // A width list expands too (cartesian with format): {640,1280} x one format.
        let set = parse_caps_set("video/x-raw,format=nv12,width={640,1280}").unwrap();
        assert_eq!(set.alternatives().len(), 2);
        // A malformed range fails the whole caps (rejected, not silently dropped).
        assert!(parse_caps_set("video/x-raw,width=[a,b]").is_none());
    }

    #[test]
    fn caps_property_round_trips_and_drives_filter() {
        let desc = "video/x-raw,format=nv12,width=320,height=240";
        let mut f = CapsFilter::default();
        f.set_property("caps", PropValue::Str(desc.into())).unwrap();
        assert_eq!(f.get_property("caps"), Some(PropValue::Str(desc.into())));

        let CapsConstraint::Identity(set) = f.caps_constraint_as_transform() else {
            panic!("expected Identity");
        };
        assert_eq!(set.alternatives(), &[nv12(320, 240)]);

        assert_eq!(f.set_property("caps", PropValue::Str("nonsense".into())), Err(PropError::Value));
    }
}
