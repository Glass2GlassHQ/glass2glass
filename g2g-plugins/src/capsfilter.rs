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

/// Parse a `gst-launch` caps description (`media/type,field=value,...`) into a
/// [`CapsSet`]. A `video/x-raw` or `audio/x-raw` with no `format` field expands
/// to all supported raw formats at the given geometry (the gst-idiomatic
/// format-less caps), so a downstream capsfilter can pin geometry while leaving
/// the format to negotiation. Other media types yield a single-alternative set.
/// `None` on an unknown media type or an unparseable field. Format values are
/// case-insensitive (GStreamer's uppercase or the historical lowercase, M182).
pub fn parse_caps_set(desc: &str) -> Option<CapsSet> {
    let mut parts = desc.split(',');
    let media = parts.next()?.trim();
    let fields: Vec<(&str, &str)> =
        parts.filter_map(|p| p.trim().split_once('=')).map(|(k, v)| (k.trim(), v.trim())).collect();

    let dim = |key: &str| field(&fields, key).and_then(|v| v.parse::<u32>().ok()).map_or(Dim::Any, Dim::Fixed);
    let framerate = field(&fields, "framerate").and_then(parse_rate).unwrap_or(Rate::Any);

    let one = |caps: Caps| Some(CapsSet::one(caps));
    match media {
        "video/x-raw" => {
            let (width, height) = (dim("width"), dim("height"));
            match field(&fields, "format") {
                Some(f) => one(Caps::RawVideo {
                    format: parse_raw_format(f)?,
                    width,
                    height,
                    framerate,
                }),
                // Format-less: any supported pixel format at this geometry.
                None => Some(CapsSet::from_alternatives(
                    RAW_VIDEO_FORMATS
                        .iter()
                        .map(|&format| Caps::RawVideo {
                            format,
                            width: width.clone(),
                            height: height.clone(),
                            framerate: framerate.clone(),
                        })
                        .collect(),
                )),
            }
        }
        "audio/x-raw" => match field(&fields, "format") {
            Some(f) => one(audio(parse_audio_format(f)?, &fields)),
            // Format-less: any supported sample format at these channels / rate.
            None => Some(CapsSet::from_alternatives(
                RAW_AUDIO_FORMATS.iter().map(|&format| audio(format, &fields)).collect(),
            )),
        },
        "video/x-h264" => one(compressed(VideoCodec::H264, dim("width"), dim("height"), framerate)),
        "video/x-h265" => one(compressed(VideoCodec::H265, dim("width"), dim("height"), framerate)),
        "video/x-vp8" => one(compressed(VideoCodec::Vp8, dim("width"), dim("height"), framerate)),
        "video/x-vp9" => one(compressed(VideoCodec::Vp9, dim("width"), dim("height"), framerate)),
        "video/x-av1" => one(compressed(VideoCodec::Av1, dim("width"), dim("height"), framerate)),
        "image/jpeg" => one(compressed(VideoCodec::Mjpeg, dim("width"), dim("height"), framerate)),
        "audio/x-opus" => one(audio(AudioFormat::Opus, &fields)),
        // gst names AAC `audio/mpeg` (with mpegversion=4, which we don't require).
        "audio/mpeg" => one(audio(AudioFormat::Aac, &fields)),
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

fn field<'a>(fields: &[(&'a str, &'a str)], key: &str) -> Option<&'a str> {
    fields.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

fn compressed(codec: VideoCodec, width: Dim, height: Dim, framerate: Rate) -> Caps {
    Caps::CompressedVideo { codec, width, height, framerate }
}

fn audio(format: AudioFormat, fields: &[(&str, &str)]) -> Caps {
    Caps::Audio {
        format,
        channels: field(fields, "channels").and_then(|c| c.parse().ok()).unwrap_or(2),
        sample_rate: field(fields, "rate").and_then(|r| r.parse().ok()).unwrap_or(48_000),
    }
}

// GStreamer caps name formats uppercase (NV12, RGBA, YUY2, S16LE); accept any
// case and the historical lowercase spellings so both port.
fn parse_raw_format(s: &str) -> Option<RawVideoFormat> {
    Some(match s.to_ascii_lowercase().as_str() {
        "rgba" => RawVideoFormat::Rgba8,
        "bgra" => RawVideoFormat::Bgra8,
        "nv12" => RawVideoFormat::Nv12,
        "i420" => RawVideoFormat::I420,
        "yuyv" | "yuy2" => RawVideoFormat::Yuyv,
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

/// Parse a framerate `num/den` (or bare integer) into a Q16 [`Rate::Fixed`].
fn parse_rate(s: &str) -> Option<Rate> {
    let q16 = match s.split_once('/') {
        Some((n, d)) => {
            let n: u64 = n.trim().parse().ok()?;
            let d: u64 = d.trim().parse().ok()?;
            if d == 0 {
                return None;
            }
            ((n << 16) / d) as u32
        }
        None => (s.trim().parse::<u64>().ok()? << 16) as u32,
    };
    Some(Rate::Fixed(q16))
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
