//! Shared PipeWire video helpers for
//! [`PipeWireVideoSrc`](crate::pipewirevideosrc): the SPA <-> `RawVideoFormat`
//! mapping, the `EnumFormat` pod a video stream connects with, and the tight
//! plane layout used to size and de-stride a captured buffer. The video sibling
//! of [`pwaudio`](crate::pwaudio). Linux-only (`pipewire` feature).

use alloc::vec::Vec;

use pipewire::spa;
use spa::param::format::{FormatProperties, MediaSubtype, MediaType};
use spa::param::video::{VideoFormat, VideoInfoRaw};
use spa::param::ParamType;
use spa::pod::serialize::PodSerializer;
use spa::pod::{object, property, ChoiceValue, Property, Value};
use spa::utils::{Choice, ChoiceEnum, ChoiceFlags, Fraction, Id, Rectangle, SpaTypes};

use g2g_core::{Dim, G2gError, Rate, RawVideoFormat};

/// The raw formats the capture element handles, in preference order. The
/// connect pod offers them as an enum choice and the negotiated one is mapped
/// back; anything else fails the negotiation. All are 8-bit and describable by
/// [`PlaneLayout`].
const FORMATS: [(RawVideoFormat, VideoFormat); 5] = [
    (RawVideoFormat::I420, VideoFormat::I420),
    (RawVideoFormat::Nv12, VideoFormat::NV12),
    (RawVideoFormat::Yuyv, VideoFormat::YUY2),
    (RawVideoFormat::Rgba8, VideoFormat::RGBA),
    (RawVideoFormat::Bgra8, VideoFormat::BGRA),
];

/// Upper bound on a negotiated dimension. The daemon's numbers are not trusted,
/// so a frame wider / taller than this is rejected instead of sizing an
/// allocation from it.
pub(crate) const MAX_DIM: u32 = 16_384;

/// The formats a `PipeWireVideoSrc` pad template advertises.
pub(crate) fn supported_formats() -> impl Iterator<Item = RawVideoFormat> {
    FORMATS.iter().map(|(g2g, _)| *g2g)
}

/// The SPA format for one of our raw formats, or `None` when the element cannot
/// carry it.
pub(crate) fn spa_format(format: RawVideoFormat) -> Option<VideoFormat> {
    FORMATS
        .iter()
        .find(|(g2g, _)| *g2g == format)
        .map(|(_, spa)| *spa)
}

/// Our raw format for a negotiated SPA format. `None` for anything outside
/// [`FORMATS`] (including `Unknown` / `Encoded`), which fails the capture rather
/// than being reinterpreted as some default.
pub(crate) fn g2g_format(format: VideoFormat) -> Option<RawVideoFormat> {
    FORMATS
        .iter()
        .find(|(_, spa)| *spa == format)
        .map(|(g2g, _)| *g2g)
}

/// A negotiated raw video format: what the node actually settled on, parsed out
/// of the `Format` param.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VideoInfo {
    pub format: RawVideoFormat,
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_denom: u32,
}

impl VideoInfo {
    /// Read a parsed `VideoInfoRaw` (the fixated `Format` param). Rejects a
    /// format we cannot carry and geometry we will not allocate from.
    pub(crate) fn from_spa(info: &VideoInfoRaw) -> Result<Self, G2gError> {
        let format = g2g_format(info.format()).ok_or(G2gError::CapsMismatch)?;
        let size = info.size();
        if size.width == 0 || size.height == 0 || size.width > MAX_DIM || size.height > MAX_DIM {
            return Err(G2gError::CapsMismatch);
        }
        let rate = info.framerate();
        Ok(Self {
            format,
            width: size.width,
            height: size.height,
            fps_num: rate.num,
            fps_denom: rate.denom,
        })
    }

    pub(crate) fn caps(&self) -> g2g_core::Caps {
        g2g_core::Caps::RawVideo {
            format: self.format,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(rate_q16(self.fps_num, self.fps_denom)),
            interlace: g2g_core::Interlace::Any,
        }
    }

    /// Nominal frame duration. Zero when the node reports a variable rate
    /// (`0/1`), which leaves the frames un-paced rather than inventing a period.
    pub(crate) fn frame_period_ns(&self) -> u64 {
        if self.fps_num == 0 || self.fps_denom == 0 {
            return 0;
        }
        u64::from(self.fps_denom) * 1_000_000_000 / u64::from(self.fps_num)
    }
}

/// A framerate fraction as the Q16 fixed-point fps [`Rate`] carries.
pub(crate) fn rate_q16(num: u32, denom: u32) -> u32 {
    if denom == 0 {
        return 0;
    }
    u32::try_from((u64::from(num) << 16) / u64::from(denom)).unwrap_or(u32::MAX)
}

/// Serialize the `EnumFormat` pod a capture stream connects with: our formats as
/// an enum choice with `preferred` first, and the requested geometry / rate as
/// the default of an open range so a node with its own fixed mode still
/// negotiates (the result arrives via `param_changed`).
///
/// With `pinned` the choice offers `preferred` alone, so a node that cannot
/// produce it fails the negotiation instead of settling on another format.
///
/// The returned bytes back a `Pod::from_bytes` at the call site (kept there so
/// the borrow lives as long as the `connect` call needs it).
pub(crate) fn format_pod_bytes(
    preferred: RawVideoFormat,
    pinned: bool,
    width: u32,
    height: u32,
    fps: u32,
) -> Result<Vec<u8>, G2gError> {
    let preferred = spa_format(preferred).ok_or(G2gError::CapsMismatch)?;
    let obj = object! {
        SpaTypes::ObjectParamFormat,
        ParamType::EnumFormat,
        property!(FormatProperties::MediaType, Id, MediaType::Video),
        property!(FormatProperties::MediaSubtype, Id, MediaSubtype::Raw),
        format_choice(preferred, pinned),
        property!(
            FormatProperties::VideoSize,
            Choice,
            Range,
            Rectangle,
            Rectangle { width, height },
            Rectangle {
                width: 1,
                height: 1
            },
            Rectangle {
                width: MAX_DIM,
                height: MAX_DIM
            }
        ),
        property!(
            FormatProperties::VideoFramerate,
            Choice,
            Range,
            Fraction,
            Fraction { num: fps, denom: 1 },
            Fraction { num: 0, denom: 1 },
            Fraction {
                num: 1_000,
                denom: 1
            }
        ),
    };
    Ok(
        PodSerializer::serialize(std::io::Cursor::new(Vec::new()), &Value::Object(obj))
            .expect("serialize SPA video format pod")
            .0
            .into_inner(),
    )
}

/// A fixated `Format` param, the shape the daemon answers negotiation with.
/// Test-only: it lets the `param_changed` parse run without a live node.
#[cfg(test)]
pub(crate) fn fixed_format_pod_bytes(
    format: VideoFormat,
    width: u32,
    height: u32,
    fps: u32,
) -> Vec<u8> {
    let obj = object! {
        SpaTypes::ObjectParamFormat,
        ParamType::Format,
        property!(FormatProperties::MediaType, Id, MediaType::Video),
        property!(FormatProperties::MediaSubtype, Id, MediaSubtype::Raw),
        property!(FormatProperties::VideoFormat, Id, format),
        property!(FormatProperties::VideoSize, Rectangle, Rectangle { width, height }),
        property!(
            FormatProperties::VideoFramerate,
            Fraction,
            Fraction { num: fps, denom: 1 }
        ),
    };
    PodSerializer::serialize(std::io::Cursor::new(Vec::new()), &Value::Object(obj))
        .expect("serialize fixated SPA video format pod")
        .0
        .into_inner()
}

/// The `VideoFormat` enum choice, `preferred` first (it doubles as the choice
/// default) then the rest of [`FORMATS`], or `preferred` alone when `pinned`.
/// Built by hand rather than with `property!` because the alternatives come from
/// the table.
fn format_choice(preferred: VideoFormat, pinned: bool) -> Property {
    let mut alternatives = Vec::with_capacity(FORMATS.len());
    alternatives.push(Id(preferred.as_raw()));
    if !pinned {
        alternatives.extend(
            FORMATS
                .iter()
                .filter(|(_, spa)| *spa != preferred)
                .map(|(_, spa)| Id(spa.as_raw())),
        );
    }
    Property::new(
        FormatProperties::VideoFormat.as_raw(),
        Value::Choice(ChoiceValue::Id(Choice(
            ChoiceFlags::empty(),
            ChoiceEnum::Enum {
                default: Id(preferred.as_raw()),
                alternatives,
            },
        ))),
    )
}

/// Tight plane geometry of a captured frame: `(row bytes, rows, stride shift)`
/// per plane, where the shift derives a plane's stride from plane 0's (the only
/// stride PipeWire reports for a single mapped block).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlaneLayout {
    planes: [(usize, usize, u32); 3],
    count: usize,
}

impl PlaneLayout {
    /// Layout of `format` at `width` x `height`, or `None` for geometry we will
    /// not size a buffer from.
    pub(crate) fn new(format: RawVideoFormat, width: u32, height: u32) -> Option<Self> {
        if width == 0 || height == 0 || width > MAX_DIM || height > MAX_DIM {
            return None;
        }
        let (w, h) = (width as usize, height as usize);
        // rounded up, so an odd-sized frame keeps a full chroma row / column
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let (planes, count) = match format {
            RawVideoFormat::I420 => ([(w, h, 0), (cw, ch, 1), (cw, ch, 1)], 3),
            RawVideoFormat::Nv12 => ([(w, h, 0), (cw * 2, ch, 0), (0, 0, 0)], 2),
            RawVideoFormat::Yuyv => ([(w * 2, h, 0), (0, 0, 0), (0, 0, 0)], 1),
            RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8 => {
                ([(w * 4, h, 0), (0, 0, 0), (0, 0, 0)], 1)
            }
            _ => return None,
        };
        Some(Self { planes, count })
    }

    /// Bytes of a tightly packed frame in this layout.
    pub(crate) fn frame_bytes(&self) -> usize {
        self.planes[..self.count]
            .iter()
            .map(|(row_bytes, rows, _)| row_bytes * rows)
            .sum()
    }

    /// Append the frame in `src` to `dst`, tightly packed. `stride` is plane 0's
    /// row stride from the buffer chunk (0 means "tight"). Every read is bounded
    /// against `src`, so a chunk that disagrees with the negotiated geometry
    /// yields `None` instead of a short or out-of-range copy. `dst` may hold a
    /// partial frame on `None`, so discard it.
    pub(crate) fn copy_tight(&self, src: &[u8], stride: usize, dst: &mut Vec<u8>) -> Option<()> {
        let mut plane_start = 0usize;
        for &(row_bytes, rows, shift) in &self.planes[..self.count] {
            let stride = if stride == 0 {
                row_bytes
            } else {
                stride >> shift
            };
            if stride < row_bytes {
                return None;
            }
            for row in 0..rows {
                let start = plane_start.checked_add(stride.checked_mul(row)?)?;
                let end = start.checked_add(row_bytes)?;
                if end > src.len() {
                    return None;
                }
                dst.extend_from_slice(&src[start..end]);
            }
            plane_start = plane_start.checked_add(stride.checked_mul(rows)?)?;
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spa::pod::deserialize::PodDeserializer;
    use spa::pod::Pod;

    /// Serialize the connect pod and read it back: the preferred format leads the
    /// enum choice (a peer takes the first alternative it can do), the rest of
    /// the table follows, and the requested geometry / rate are the range
    /// defaults.
    /// The format alternatives a serialized connect pod offers, in order.
    fn pod_format_alternatives(bytes: &[u8]) -> (Id, Vec<Id>) {
        let (_, value) =
            PodDeserializer::deserialize_any_from(bytes).expect("our pod deserializes");
        let Value::Object(obj) = value else {
            panic!("expected an object pod");
        };
        let value = obj
            .properties
            .iter()
            .find(|p| p.key == FormatProperties::VideoFormat.as_raw())
            .map(|p| p.value.clone())
            .expect("pod carries a format");
        let Value::Choice(ChoiceValue::Id(Choice(
            _,
            ChoiceEnum::Enum {
                default,
                alternatives,
            },
        ))) = value
        else {
            panic!("format is an enum choice");
        };
        (default, alternatives)
    }

    /// A pinned format is the only alternative on offer, so a node that cannot
    /// produce it fails negotiation instead of picking another entry.
    #[test]
    fn a_pinned_format_is_the_only_alternative() {
        for (g2g, spa_fmt) in FORMATS {
            let bytes = format_pod_bytes(g2g, true, 320, 240, 30).expect("supported format");
            let (default, alternatives) = pod_format_alternatives(&bytes);
            assert_eq!(default, Id(spa_fmt.as_raw()));
            assert_eq!(alternatives, Vec::from([Id(spa_fmt.as_raw())]));
        }
    }

    #[test]
    fn enum_format_pod_round_trips() {
        for (g2g, spa_fmt) in FORMATS {
            let bytes = format_pod_bytes(g2g, false, 320, 240, 30).expect("supported format");
            let (_, value) =
                PodDeserializer::deserialize_any_from(&bytes).expect("our pod deserializes");
            let Value::Object(obj) = value else {
                panic!("expected an object pod");
            };
            assert_eq!(obj.type_, SpaTypes::ObjectParamFormat.as_raw());
            assert_eq!(obj.id, ParamType::EnumFormat.as_raw());

            let prop = |key: FormatProperties| {
                obj.properties
                    .iter()
                    .find(|p| p.key == key.as_raw())
                    .map(|p| p.value.clone())
                    .unwrap_or_else(|| panic!("pod carries {key:?}"))
            };
            assert_eq!(
                prop(FormatProperties::MediaType),
                Value::Id(Id(MediaType::Video.as_raw()))
            );
            assert_eq!(
                prop(FormatProperties::MediaSubtype),
                Value::Id(Id(MediaSubtype::Raw.as_raw()))
            );

            let mut expected: Vec<Id> = Vec::from([Id(spa_fmt.as_raw())]);
            expected.extend(
                FORMATS
                    .iter()
                    .filter(|(_, s)| *s != spa_fmt)
                    .map(|(_, s)| Id(s.as_raw())),
            );
            let (default, alternatives) = pod_format_alternatives(&bytes);
            assert_eq!(default, Id(spa_fmt.as_raw()));
            assert_eq!(alternatives, expected);

            let Value::Choice(ChoiceValue::Rectangle(Choice(
                _,
                ChoiceEnum::Range { default, min, max },
            ))) = prop(FormatProperties::VideoSize)
            else {
                panic!("size is a range choice");
            };
            assert_eq!(
                default,
                Rectangle {
                    width: 320,
                    height: 240
                }
            );
            assert_eq!(
                min,
                Rectangle {
                    width: 1,
                    height: 1
                }
            );
            assert_eq!(
                max,
                Rectangle {
                    width: MAX_DIM,
                    height: MAX_DIM
                }
            );

            let Value::Choice(ChoiceValue::Fraction(Choice(_, ChoiceEnum::Range { default, .. }))) =
                prop(FormatProperties::VideoFramerate)
            else {
                panic!("framerate is a range choice");
            };
            assert_eq!(default, Fraction { num: 30, denom: 1 });
        }
    }

    /// The fixated `Format` param the daemon answers with parses back into the
    /// negotiated info the capture path uses. `spa_format_video_raw_parse` reads
    /// fixed values, not the choices our `EnumFormat` offers, so this is the
    /// param shape `param_changed` really sees.
    #[test]
    fn fixated_format_parses_into_negotiated_info() {
        for (g2g, spa_fmt) in FORMATS {
            let bytes = fixed_format_pod_bytes(spa_fmt, 320, 240, 30);
            let pod = Pod::from_bytes(&bytes).expect("pod bytes parse");
            let mut parsed = VideoInfoRaw::new();
            parsed.parse(pod).expect("spa parses the fixated format");
            assert_eq!(parsed.format(), spa_fmt);
            assert_eq!(
                VideoInfo::from_spa(&parsed),
                Ok(VideoInfo {
                    format: g2g,
                    width: 320,
                    height: 240,
                    fps_num: 30,
                    fps_denom: 1,
                })
            );
        }
    }

    #[test]
    fn q16_rate_and_frame_period_follow_the_fraction() {
        assert_eq!(rate_q16(30, 1), 30 << 16);
        assert_eq!(rate_q16(30_000, 1001), (((30_000u64) << 16) / 1001) as u32);
        assert_eq!(rate_q16(30, 0), 0);
        let info = |num, denom| VideoInfo {
            format: RawVideoFormat::I420,
            width: 320,
            height: 240,
            fps_num: num,
            fps_denom: denom,
        };
        assert_eq!(info(25, 1).frame_period_ns(), 40_000_000);
        // a variable-rate node reports 0/1: no invented period
        assert_eq!(info(0, 1).frame_period_ns(), 0);
    }

    #[test]
    fn a_format_the_element_cannot_carry_has_no_pod() {
        assert_eq!(
            format_pod_bytes(RawVideoFormat::P010, false, 320, 240, 30),
            Err(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn spa_mapping_is_a_bijection_over_the_table() {
        for (g2g, spa_fmt) in FORMATS {
            assert_eq!(spa_format(g2g), Some(spa_fmt));
            assert_eq!(g2g_format(spa_fmt), Some(g2g));
        }
        // outside the table: no silent default in either direction
        assert_eq!(g2g_format(VideoFormat::Unknown), None);
        assert_eq!(g2g_format(VideoFormat::YV12), None);
        assert_eq!(spa_format(RawVideoFormat::I422), None);
    }

    #[test]
    fn unusable_negotiated_info_is_rejected() {
        let mut info = VideoInfoRaw::new();
        info.set_format(VideoFormat::I420);
        info.set_size(Rectangle {
            width: 0,
            height: 240,
        });
        assert_eq!(VideoInfo::from_spa(&info), Err(G2gError::CapsMismatch));
        info.set_size(Rectangle {
            width: MAX_DIM + 1,
            height: 240,
        });
        assert_eq!(VideoInfo::from_spa(&info), Err(G2gError::CapsMismatch));
        // a format we do not carry fails even with sane geometry
        info.set_format(VideoFormat::YV12);
        info.set_size(Rectangle {
            width: 320,
            height: 240,
        });
        assert_eq!(VideoInfo::from_spa(&info), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn plane_layout_sizes_each_format() {
        let bytes = |f, w, h| PlaneLayout::new(f, w, h).map(|l| l.frame_bytes());
        assert_eq!(bytes(RawVideoFormat::I420, 320, 240), Some(115_200));
        assert_eq!(bytes(RawVideoFormat::Nv12, 320, 240), Some(115_200));
        assert_eq!(bytes(RawVideoFormat::Yuyv, 320, 240), Some(153_600));
        assert_eq!(bytes(RawVideoFormat::Rgba8, 320, 240), Some(307_200));
        // odd geometry rounds chroma up: 3x3 luma + 2x2x2 chroma
        assert_eq!(bytes(RawVideoFormat::I420, 3, 3), Some(9 + 8));
        assert_eq!(bytes(RawVideoFormat::I420, 0, 240), None);
        assert_eq!(bytes(RawVideoFormat::I420, 320, MAX_DIM + 1), None);
        assert_eq!(bytes(RawVideoFormat::P010, 320, 240), None);
    }

    #[test]
    fn copy_tight_de_strides_padded_rows() {
        // 2x2 I420 with 4-byte padded luma rows (chroma stride 2 for 1 sample).
        let layout = PlaneLayout::new(RawVideoFormat::I420, 2, 2).unwrap();
        assert_eq!(layout.frame_bytes(), 6);
        let src = [
            1, 2, 0xff, 0xff, // Y row 0 + pad
            3, 4, 0xff, 0xff, // Y row 1 + pad
            5, 0xff, // U row 0 + pad
            6, 0xff, // V row 0 + pad
        ];
        let mut dst = Vec::new();
        layout.copy_tight(&src, 4, &mut dst).expect("copy fits");
        assert_eq!(dst, [1, 2, 3, 4, 5, 6]);

        // stride 0 means tight
        let mut tight = Vec::new();
        layout
            .copy_tight(&[1, 2, 3, 4, 5, 6], 0, &mut tight)
            .expect("tight copy fits");
        assert_eq!(tight, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn copy_tight_rejects_a_chunk_that_disagrees_with_the_geometry() {
        let layout = PlaneLayout::new(RawVideoFormat::I420, 2, 2).unwrap();
        let mut dst = Vec::new();
        // one byte short of the tight frame
        assert_eq!(layout.copy_tight(&[1, 2, 3, 4, 5], 0, &mut dst), None);
        // stride narrower than a row of luma
        assert_eq!(layout.copy_tight(&[0; 64], 1, &mut dst), None);
        // a stride that overflows the offset arithmetic
        assert_eq!(layout.copy_tight(&[0; 64], usize::MAX, &mut dst), None);
    }
}
