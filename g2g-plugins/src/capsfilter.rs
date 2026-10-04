//! Caps filter: a pass-through transform that forces a negotiation-time
//! narrowing (design/caps.md). Data flows through unchanged;
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
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

/// # Example
///
/// ```no_run
/// use g2g_core::{AudioFormat, Caps, ChannelLayout};
/// use g2g_plugins::capsfilter::CapsFilter;
///
/// let format = AudioFormat::PcmS16Le;
/// let element = CapsFilter::new(Caps::Audio {
///     format,
///     channels: 2,
///     sample_rate: 48_000,
///     channel_layout: ChannelLayout::UNSPECIFIED,
/// });
/// ```
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
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Caps filter",
            "Generic",
            "Restricts the caps that may negotiate across a link",
            "g2g",
        )
    }
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
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

    fn output_domain_follows_input(&self) -> bool {
        true
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
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

/// Parse a `gst-launch` caps description into a [`CapsSet`]. The parser itself
/// lives in `g2g-core` next to its inverse, [`CapsSet::from_gst_string`].
pub fn parse_caps_set(desc: &str) -> Option<CapsSet> {
    CapsSet::from_gst_string(desc)
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

const DMABUF_CAPS_FEATURE: &str = "(memory:DMABuf)";

const FORMAT_FIELD: &str = "format=";

const DRM_FORMAT_FIELD: &str = "drm-format=";

const DMA_DRM_FORMAT: &str = "DMA_DRM";

// GStreamer's drm-format fourcc for each format dmabuftowgpu imports
const DRM_FOURCC_FORMATS: &[(&str, &str)] = &[
    ("AB24", "RGBA"),
    ("AR24", "BGRA"),
    ("NV12", "NV12"),
    ("P010", "P010_10LE"),
    ("YU12", "I420"),
    ("YUYV", "YUY2"),
];

/// Turn a serialized `GstCaps` into the form g2g's caps reader and launch DSL
/// accept: `video/x-raw, format=(string)RGBA, width=(int)1280, ...` ->
/// `video/x-raw,format=RGBA,width=1280,...`. Two transforms:
///
/// - drop every `(type)` annotation in the fields (a media-caps value never
///   legitimately contains parentheses), keeping a caps feature on the media
///   type (`video/x-raw(memory:DMABuf)`) for [`read_memory_feature`] to read, and
/// - drop all whitespace, because the launch DSL tokenizes on spaces, so a
///   `caps=` value with the spaces GStreamer inserts after commas would split
///   into separate launch tokens (PORTING.md: no quoted values with spaces).
///
/// Fields g2g does not model (`multiview-mode`, `pixel-aspect-ratio`, ...) are
/// carried through harmlessly; the caps reader ignores unknown fields.
pub fn normalize_gst_caps(s: &str) -> String {
    let (media_type, fields) = s.split_once(',').unwrap_or((s, ""));
    let mut out: String = media_type
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect();
    if !fields.is_empty() {
        out.push(',');
    }
    let mut depth = 0u32;
    for ch in fields.chars() {
        match ch {
            '(' => depth += 1,
            ')' if depth > 0 => depth -= 1,
            _ if depth > 0 => {}
            _ if ch.is_whitespace() => {}
            _ => out.push(ch),
        }
    }
    out
}

/// A normalized GStreamer caps string with its `memory:DMABuf` feature read off.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GstCapsMemory {
    /// The caps without the feature, in the form [`parse_caps`] reads.
    pub caps: String,
    pub dmabuf: bool,
}

/// Read the `memory:DMABuf` feature off a [`normalize_gst_caps`] string. A
/// GStreamer 1.24+ `format=DMA_DRM, drm-format=<fourcc>` is read as the pixel
/// format it names. `None` for dma-buf caps whose `drm-format` names no format
/// `dmabuftowgpu` imports, or carries a non-linear modifier.
pub fn read_memory_feature(caps: &str) -> Option<GstCapsMemory> {
    let (media_type, fields) = caps.split_once(',').unwrap_or((caps, ""));
    let Some(media_type) = media_type.strip_suffix(DMABUF_CAPS_FEATURE) else {
        return Some(GstCapsMemory {
            caps: caps.into(),
            dmabuf: false,
        });
    };
    let drm_format = fields
        .split(',')
        .find_map(|field| field.strip_prefix(DRM_FORMAT_FIELD));
    let mut rewritten = alloc::vec![String::from(media_type)];
    for field in fields.split(',').filter(|field| !field.is_empty()) {
        if field.starts_with(DRM_FORMAT_FIELD) {
            continue;
        }
        if field.strip_prefix(FORMAT_FIELD) != Some(DMA_DRM_FORMAT) {
            rewritten.push(field.into());
            continue;
        }
        // a modifier suffix (a tiled layout) matches no entry
        let (_, format) = DRM_FOURCC_FORMATS
            .iter()
            .find(|(known, _)| Some(*known) == drm_format)?;
        rewritten.push(alloc::format!("{FORMAT_FIELD}{format}"));
    }
    Some(GstCapsMemory {
        caps: rewritten.join(","),
        dmabuf: true,
    })
}

/// `caps` as GStreamer 1.24+ spells dma-buf video: the `memory:DMABuf`
/// feature with `format=DMA_DRM, drm-format=<fourcc>`, the inverse of
/// [`read_memory_feature`]. `None` for anything but raw video in a format
/// with a linear drm fourcc.
pub fn dmabuf_gst_caps(caps: &Caps) -> Option<String> {
    let Caps::RawVideo { .. } = caps else {
        return None;
    };
    let plain = caps.to_gst_string();
    let (media_type, fields) = plain.split_once(',')?;
    let mut rewritten = alloc::vec![alloc::format!("{media_type}{DMABUF_CAPS_FEATURE}")];
    for field in fields.split(',') {
        let Some(format) = field.strip_prefix(FORMAT_FIELD) else {
            rewritten.push(field.into());
            continue;
        };
        let (fourcc, _) = DRM_FOURCC_FORMATS
            .iter()
            .find(|(_, known)| *known == format)?;
        rewritten.push(alloc::format!("{FORMAT_FIELD}{DMA_DRM_FORMAT}"));
        rewritten.push(alloc::format!("{DRM_FORMAT_FIELD}{fourcc}"));
    }
    Some(rewritten.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{Dim, Rate, RawVideoFormat, VideoCodec};

    fn nv12(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
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
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        });
        assert_eq!(f.intercept_caps(&nv12(1280, 720)), Ok(nv12(1280, 720)));

        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
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
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN
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
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN
            })
        );
        // `parse_caps` yields a single Caps, so a format-less (multi-format) raw
        // description is `None` here; `parse_caps_set` expands it instead.
        assert_eq!(
            parse_caps("video/x-raw,width=320"),
            None,
            "format-less is not a single caps"
        );
    }

    #[test]
    fn parse_caps_compressed_and_audio() {
        assert!(matches!(
            parse_caps("video/x-h264,width=1920,height=1080"),
            Some(Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            })
        ));
        assert!(matches!(
            parse_caps("video/x-vp9"),
            Some(Caps::CompressedVideo {
                codec: VideoCodec::Vp9,
                ..
            })
        ));
        assert_eq!(
            parse_caps("audio/x-opus,channels=2,rate=48000"),
            Some(Caps::Audio {
                format: g2g_core::AudioFormat::Opus,
                channels: 2,
                sample_rate: 48_000,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED
            })
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

        assert_eq!(
            f.set_property("caps", PropValue::Str("nonsense".into())),
            Err(PropError::Value)
        );
    }

    #[test]
    fn keeps_the_caps_feature_and_drops_type_annotations() {
        let serialized = "video/x-raw(memory:DMABuf), format=(string)DMA_DRM, width=(int)64, drm-format=(string)AB24";
        assert_eq!(
            normalize_gst_caps(serialized),
            "video/x-raw(memory:DMABuf),format=DMA_DRM,width=64,drm-format=AB24"
        );
    }

    #[test]
    fn dmabuf_caps_read_back_as_what_was_written() {
        let caps = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(64),
            height: Dim::Fixed(48),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        let written = dmabuf_gst_caps(&caps).expect("RGBA has a drm fourcc");
        let read = read_memory_feature(&written).expect("the fourcc is known");
        assert!(read.dmabuf);
        assert_eq!(parse_caps(&read.caps), Some(caps));
    }

    #[test]
    fn plain_caps_carry_no_memory_feature() {
        let plain = "video/x-raw,format=NV12,width=2,height=2";
        assert_eq!(
            read_memory_feature(plain),
            Some(GstCapsMemory {
                caps: plain.into(),
                dmabuf: false
            })
        );
    }
}
