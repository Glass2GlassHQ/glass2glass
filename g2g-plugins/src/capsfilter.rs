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
                let caps = parse_caps(s).ok_or(PropError::Value)?;
                self.filter = CapsSet::one(caps);
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

/// Parse a `gst-launch` caps description (`media/type,field=value,...`) into a
/// [`Caps`]. `None` on an unknown media type or a missing / unparseable required
/// field. Field values use the lowercase convention of the element format
/// properties (`format=nv12`), not the GStreamer uppercase fourcc.
pub fn parse_caps(desc: &str) -> Option<Caps> {
    let mut parts = desc.split(',');
    let media = parts.next()?.trim();
    let fields: Vec<(&str, &str)> =
        parts.filter_map(|p| p.trim().split_once('=')).map(|(k, v)| (k.trim(), v.trim())).collect();

    let dim = |key: &str| field(&fields, key).and_then(|v| v.parse::<u32>().ok()).map_or(Dim::Any, Dim::Fixed);
    let framerate = field(&fields, "framerate").and_then(parse_rate).unwrap_or(Rate::Any);

    match media {
        "video/x-raw" => Some(Caps::RawVideo {
            format: parse_raw_format(field(&fields, "format")?)?,
            width: dim("width"),
            height: dim("height"),
            framerate,
        }),
        "video/x-h264" => Some(compressed(VideoCodec::H264, dim("width"), dim("height"), framerate)),
        "video/x-h265" => Some(compressed(VideoCodec::H265, dim("width"), dim("height"), framerate)),
        "video/x-vp8" => Some(compressed(VideoCodec::Vp8, dim("width"), dim("height"), framerate)),
        "video/x-vp9" => Some(compressed(VideoCodec::Vp9, dim("width"), dim("height"), framerate)),
        "video/x-av1" => Some(compressed(VideoCodec::Av1, dim("width"), dim("height"), framerate)),
        "image/jpeg" => Some(compressed(VideoCodec::Mjpeg, dim("width"), dim("height"), framerate)),
        "audio/x-opus" => Some(audio(AudioFormat::Opus, &fields)),
        // gst names AAC `audio/mpeg` (with mpegversion=4, which we don't require).
        "audio/mpeg" => Some(audio(AudioFormat::Aac, &fields)),
        "audio/x-raw" => Some(audio(parse_audio_format(field(&fields, "format")?)?, &fields)),
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

fn parse_raw_format(s: &str) -> Option<RawVideoFormat> {
    Some(match s {
        "rgba" => RawVideoFormat::Rgba8,
        "bgra" => RawVideoFormat::Bgra8,
        "nv12" => RawVideoFormat::Nv12,
        "i420" => RawVideoFormat::I420,
        "yuyv" | "yuy2" => RawVideoFormat::Yuyv,
        _ => return None,
    })
}

fn parse_audio_format(s: &str) -> Option<AudioFormat> {
    Some(match s {
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
        assert_eq!(parse_caps("video/x-raw,width=320"), None, "raw video needs a format");
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
