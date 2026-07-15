//! The catalog of MCU static elements the compiler understands, and the frame
//! geometry each imposes. Every kind here maps to a real `g2g-mcu` heap-free
//! static element; the geometry rules are how the compiler sizes each ring
//! exactly (the number the hand-written flagship graph hard-codes) and how it
//! rejects a mis-wired graph (an encoder fed 32-bit slots, a mixer whose two
//! inputs disagree) before a line of Rust is emitted.

use std::collections::BTreeMap;

use crate::model::{Node, Scalar};
use crate::CompileError;

/// A frame's format on one link. Audio and raster (video / display) frames
/// size their rings differently, so geometry is a sum: the compiler is not
/// audio-specific, only its first catalog kinds were. Together with the
/// document's `frame_ns` this fixes the per-frame byte count, and thus the
/// ring size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Geometry {
    /// Interleaved PCM: sampling rate, bytes per sample, channels.
    Audio { sample_rate: u32, width: u8, channels: u8 },
    /// A raster frame: pixel dimensions and bytes per pixel.
    Raster { width_px: u32, height_px: u32, bpp: u8 },
}

/// Samples per channel in one audio frame, requiring the rate and period to
/// divide evenly (a fractional frame would tear samples across packets).
pub(crate) fn samples_per_frame(sample_rate: u32, frame_ns: u64) -> Result<u32, CompileError> {
    match u64::from(sample_rate).checked_mul(frame_ns) {
        Some(n) if n % 1_000_000_000 == 0 => Ok((n / 1_000_000_000) as u32),
        _ => Err(CompileError::FractionalFrame { rate: sample_rate, frame_ns }),
    }
}

impl Geometry {
    /// Bytes in one frame at this geometry (the ring size for a lending node).
    pub(crate) fn ring_bytes(&self, frame_ns: u64) -> Result<usize, CompileError> {
        match self {
            Geometry::Audio { sample_rate, width, channels } => {
                let samples = samples_per_frame(*sample_rate, frame_ns)?;
                Ok(samples as usize * *channels as usize * *width as usize)
            }
            Geometry::Raster { width_px, height_px, bpp } => {
                Ok(*width_px as usize * *height_px as usize * *bpp as usize)
            }
        }
    }
}

/// How a node sits in the graph, which decides how the runner drives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    Source,
    Transform,
    /// A two-input fan-in (the mixer): the graph's one join point.
    FanIn,
    Sink,
}

/// A resolved catalog element: the validated kind plus enough to codegen it.
/// The kinds span audio (`PcmConvert` / `Resample` / `Mixer` / `G711Enc` /
/// `RtpSink`) and raster / display (`SpiDisplaySink`); `GrabberSrc` is the
/// shared byte-capture seam, audio or video by its geometry.
#[derive(Debug, Clone)]
pub(crate) enum Kind {
    GrabberSrc { geom: Geometry },
    PcmConvert,
    Resample { from: u32, to: u32 },
    Mixer { gain_a: i16, gain_b: i16 },
    G711Enc { law: Law },
    RtpSink { clock_rate: u32, payload_type: u8, ssrc: u32, sequence: u16 },
    SpiDisplaySink { driver: Driver, width_px: u32, height_px: u32 },
}

/// G.711 companding law (mirrors `g2g_mcu::Law`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Law {
    Mulaw,
    Alaw,
}

impl Law {
    /// The `g2g_mcu::Law` variant path this emits.
    pub(crate) fn variant(self) -> &'static str {
        match self {
            Law::Mulaw => "Law::Mulaw",
            Law::Alaw => "Law::Alaw",
        }
    }
}

/// The panel controller a `SpiDisplaySink` drives (mirrors the `g2g_mcu`
/// constructors).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Driver {
    St7789,
    Ili9341,
}

impl Driver {
    /// The `SpiDisplaySink` constructor name this emits.
    pub(crate) fn ctor(self) -> &'static str {
        match self {
            Driver::St7789 => "st7789",
            Driver::Ili9341 => "ili9341",
        }
    }
}

impl Kind {
    /// This kind's role in the graph.
    pub(crate) fn role(&self) -> Role {
        match self {
            Kind::GrabberSrc { .. } => Role::Source,
            Kind::PcmConvert | Kind::Resample { .. } | Kind::G711Enc { .. } => Role::Transform,
            Kind::Mixer { .. } => Role::FanIn,
            Kind::RtpSink { .. } | Kind::SpiDisplaySink { .. } => Role::Sink,
        }
    }

    /// The output geometry given the input link's geometry (`None` for a
    /// source, which has none). Enforces each element's input contract.
    pub(crate) fn output_geometry(
        &self,
        id: &str,
        input: Option<Geometry>,
    ) -> Result<Option<Geometry>, CompileError> {
        let need = |g: Option<Geometry>| g.ok_or_else(|| CompileError::MissingInput(id.into()));
        let bad = |detail: &str| CompileError::BadGeometry { node: id.into(), detail: detail.into() };
        // Unpack an audio input link or fail: the PCM elements are audio-only.
        let audio = |g: Geometry| -> Result<(u32, u8, u8), CompileError> {
            match g {
                Geometry::Audio { sample_rate, width, channels } => Ok((sample_rate, width, channels)),
                Geometry::Raster { .. } => Err(bad("expected an audio input link")),
            }
        };
        match self {
            Kind::GrabberSrc { geom } => Ok(Some(*geom)),
            Kind::PcmConvert => {
                let (rate, width, channels) = audio(need(input)?)?;
                if width != 4 {
                    return Err(bad("pcmconvert needs 32-bit (width 4) input slots"));
                }
                Ok(Some(Geometry::Audio { sample_rate: rate, width: 2, channels }))
            }
            Kind::Resample { from, to } => {
                let (rate, width, channels) = audio(need(input)?)?;
                if rate != *from {
                    return Err(bad("resample `from` rate does not match the input link"));
                }
                Ok(Some(Geometry::Audio { sample_rate: *to, width, channels }))
            }
            Kind::Mixer { .. } => {
                audio(need(input)?)?;
                Ok(Some(need(input)?))
            }
            Kind::G711Enc { .. } => {
                let (rate, width, channels) = audio(need(input)?)?;
                if width != 2 {
                    return Err(bad("g711enc needs 16-bit (width 2) PCM input"));
                }
                Ok(Some(Geometry::Audio { sample_rate: rate, width: 1, channels }))
            }
            Kind::RtpSink { .. } => {
                audio(need(input)?)?;
                Ok(None)
            }
            Kind::SpiDisplaySink { width_px, height_px, .. } => {
                match need(input)? {
                    Geometry::Raster { width_px: w, height_px: h, bpp } => {
                        if bpp != 4 {
                            return Err(bad("spidisplaysink needs RGBA (bpp 4) input"));
                        }
                        if w != *width_px || h != *height_px {
                            return Err(bad("spidisplaysink input dimensions do not match the panel"));
                        }
                        Ok(None)
                    }
                    Geometry::Audio { .. } => Err(bad("spidisplaysink needs a raster (video) input")),
                }
            }
        }
    }
}

/// Read an integer property, or a default when absent.
fn int_prop(props: &BTreeMap<String, Scalar>, key: &str, default: i64) -> Result<i64, CompileError> {
    match props.get(key) {
        None => Ok(default),
        Some(v) => v.as_int().ok_or_else(|| CompileError::BadProp {
            key: key.into(),
            detail: "expected an integer".into(),
        }),
    }
}

/// Read a required integer property.
fn req_int(props: &BTreeMap<String, Scalar>, key: &str) -> Result<i64, CompileError> {
    props
        .get(key)
        .ok_or_else(|| CompileError::MissingProp(key.into()))?
        .as_int()
        .ok_or_else(|| CompileError::BadProp { key: key.into(), detail: "expected an integer".into() })
}

fn as_rate(v: i64, key: &str) -> Result<u32, CompileError> {
    match v {
        8000 | 16000 | 48000 => Ok(v as u32),
        _ => Err(CompileError::BadProp {
            key: key.into(),
            detail: "sample rate must be 8000, 16000, or 48000".into(),
        }),
    }
}

fn as_small<T: TryFrom<i64>>(v: i64, key: &str, ty: &str) -> Result<T, CompileError> {
    T::try_from(v).map_err(|_| CompileError::BadProp {
        key: key.into(),
        detail: format!("out of range for {ty}"),
    })
}

fn bad_prop(key: &str, detail: impl Into<String>) -> CompileError {
    CompileError::BadProp { key: key.into(), detail: detail.into() }
}

/// Bytes per pixel from a raster `format` property (default RGBA8888).
fn pixel_bpp(props: &BTreeMap<String, Scalar>) -> Result<u8, CompileError> {
    match props.get("format").and_then(Scalar::as_str) {
        None | Some("rgba8888") | Some("rgba") => Ok(4),
        Some("rgb565") => Ok(2),
        Some("gray8") | Some("gray") => Ok(1),
        Some(other) => Err(bad_prop("format", format!("unknown pixel format `{other}`"))),
    }
}

/// Resolve one document node into a validated catalog [`Kind`].
pub(crate) fn resolve(node: &Node) -> Result<Kind, CompileError> {
    let p = &node.props;
    match node.element.as_str() {
        "grabbersrc" => {
            // One byte-capture seam, audio or video by its props: `sample-rate`
            // picks an audio capture, `width-px` / `height-px` a raster one.
            if p.contains_key("sample-rate") {
                let sample_rate = as_rate(req_int(p, "sample-rate")?, "sample-rate")?;
                let width = as_small(int_prop(p, "width", 2)?, "width", "u8")?;
                let channels = as_small(int_prop(p, "channels", 1)?, "channels", "u8")?;
                if width != 2 && width != 4 {
                    return Err(bad_prop("width", "capture width must be 2 (S16) or 4 (S32 slot)"));
                }
                if channels == 0 {
                    return Err(bad_prop("channels", "channels must be >= 1"));
                }
                Ok(Kind::GrabberSrc { geom: Geometry::Audio { sample_rate, width, channels } })
            } else {
                let width_px = as_small::<u32>(req_int(p, "width-px")?, "width-px", "u32")?;
                let height_px = as_small::<u32>(req_int(p, "height-px")?, "height-px", "u32")?;
                let bpp = pixel_bpp(p)?;
                if width_px == 0 || height_px == 0 {
                    return Err(bad_prop("width-px", "pixel dimensions must be >= 1"));
                }
                Ok(Kind::GrabberSrc { geom: Geometry::Raster { width_px, height_px, bpp } })
            }
        }
        "pcmconvert" => Ok(Kind::PcmConvert),
        "resample" => {
            let from = as_rate(req_int(p, "from")?, "from")?;
            let to = as_rate(req_int(p, "to")?, "to")?;
            if from == to {
                return Err(CompileError::BadProp {
                    key: "to".into(),
                    detail: "resample from and to rates are equal (drop the node)".into(),
                });
            }
            Ok(Kind::Resample { from, to })
        }
        "mixer" => {
            let gain_a = as_small(int_prop(p, "gain-a", 16384)?, "gain-a", "i16")?;
            let gain_b = as_small(int_prop(p, "gain-b", 16384)?, "gain-b", "i16")?;
            Ok(Kind::Mixer { gain_a, gain_b })
        }
        "g711enc" => {
            let law = match p.get("law").and_then(Scalar::as_str) {
                None | Some("mulaw") => Law::Mulaw,
                Some("alaw") => Law::Alaw,
                Some(other) => {
                    return Err(CompileError::BadProp {
                        key: "law".into(),
                        detail: format!("unknown law `{other}` (mulaw | alaw)"),
                    })
                }
            };
            Ok(Kind::G711Enc { law })
        }
        "rtpsink" => {
            let clock_rate = as_rate(int_prop(p, "clock-rate", 8000)?, "clock-rate")?;
            let payload_type = as_small(int_prop(p, "payload-type", 0)?, "payload-type", "u8")?;
            let ssrc = as_small::<u32>(req_int(p, "ssrc")?, "ssrc", "u32")?;
            let sequence = as_small(int_prop(p, "sequence", 0)?, "sequence", "u16")?;
            Ok(Kind::RtpSink { clock_rate, payload_type, ssrc, sequence })
        }
        "spidisplaysink" => {
            let driver = match p.get("driver").and_then(Scalar::as_str) {
                None | Some("st7789") => Driver::St7789,
                Some("ili9341") => Driver::Ili9341,
                Some(other) => {
                    return Err(bad_prop("driver", format!("unknown driver `{other}` (st7789 | ili9341)")))
                }
            };
            let width_px = as_small::<u32>(req_int(p, "width-px")?, "width-px", "u32")?;
            let height_px = as_small::<u32>(req_int(p, "height-px")?, "height-px", "u32")?;
            Ok(Kind::SpiDisplaySink { driver, width_px, height_px })
        }
        other => Err(CompileError::UnknownElement(other.into())),
    }
}
