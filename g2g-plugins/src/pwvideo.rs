//! Shared PipeWire video helpers for
//! [`PipeWireVideoSrc`](crate::pipewirevideosrc): the SPA <-> `RawVideoFormat`
//! mapping, the `EnumFormat` pod a video stream connects with, the `Buffers` pod
//! the dma-buf path asks for (plus the block validation that path needs), and the
//! tight plane layout used to size and de-stride a mapped buffer. The video
//! sibling of [`pwaudio`](crate::pwaudio). Linux-only (`pipewire` feature).

use alloc::vec::Vec;

use pipewire::spa;
use spa::sys as spa_sys;

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

/// The formats that arrive as a single plane, so a dma-buf carrying one is one
/// block: the only shape [`MemoryDomain::DmaBuf`](g2g_core::MemoryDomain) carries
/// (one fd, one stride, one offset).
pub(crate) fn single_plane_formats() -> impl Iterator<Item = RawVideoFormat> {
    supported_formats().filter(|f| single_plane_row_bytes(*f, 1).is_some())
}

/// Row bytes of `format` at `width` when the format is a single plane, else
/// `None`. Derived from [`PlaneLayout`] so the two cannot disagree on which
/// formats are planar.
pub(crate) fn single_plane_row_bytes(format: RawVideoFormat, width: u32) -> Option<usize> {
    let layout = PlaneLayout::new(format, width, 1)?;
    (layout.count() == 1).then(|| layout.first_row_bytes())
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

/// Which formats a connect pod offers behind its preferred one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FormatOffer {
    /// Every format the element can carry.
    All,
    /// Only the single-plane ones (the dma-buf capture path).
    SinglePlane,
    /// None: the caller pinned the preferred format, so a node that cannot
    /// produce it fails the negotiation.
    PreferredOnly,
}

impl FormatOffer {
    fn includes(self, format: VideoFormat) -> bool {
        match self {
            Self::All => true,
            Self::SinglePlane => single_plane_formats().any(|f| spa_format(f) == Some(format)),
            Self::PreferredOnly => false,
        }
    }
}

/// Serialize the `EnumFormat` pod a capture stream connects with: the `offer`
/// formats as an enum choice with `preferred` first, and the requested geometry /
/// rate as the default of an open range so a node with its own fixed mode still
/// negotiates (the result arrives via `param_changed`).
///
/// The returned bytes back a `Pod::from_bytes` at the call site (kept there so
/// the borrow lives as long as the `connect` call needs it).
pub(crate) fn format_pod_bytes(
    preferred: RawVideoFormat,
    offer: FormatOffer,
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
        format_choice(preferred, offer),
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
/// default) then whichever of [`FORMATS`] the `offer` includes. Built by hand
/// rather than with `property!` because the alternatives come from the table.
fn format_choice(preferred: VideoFormat, offer: FormatOffer) -> Property {
    let mut alternatives = Vec::with_capacity(FORMATS.len());
    alternatives.push(Id(preferred.as_raw()));
    alternatives.extend(
        FORMATS
            .iter()
            .filter(|(_, spa)| *spa != preferred && offer.includes(*spa))
            .map(|(_, spa)| Id(spa.as_raw())),
    );
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

/// Buffer count asked for on the dma-buf path: the element holds a buffer for as
/// long as downstream references the frame, so a handful of spares keeps the
/// producer from running dry while a frame is in flight.
const DMABUF_BUFFERS: (i32, i32, i32) = (8, 2, 16);

/// Serialize the `Buffers` param that asks the producer for dma-buf memory: one
/// block per buffer (all [`MemoryDomain::DmaBuf`](g2g_core::MemoryDomain) carries
/// is a single fd) and dma-buf as the only accepted data type, so a producer that
/// has no dma-buf to give fails the negotiation instead of handing back mapped
/// memory the element would have to copy. Announced from `param_changed`, before
/// the buffers are allocated.
pub(crate) fn dmabuf_buffers_pod_bytes() -> Vec<u8> {
    let (default, min, max) = DMABUF_BUFFERS;
    let obj = object! {
        SpaTypes::ObjectParamBuffers,
        ParamType::Buffers,
        Property::new(
            spa_sys::SPA_PARAM_BUFFERS_dataType,
            Value::Choice(ChoiceValue::Int(Choice(
                ChoiceFlags::empty(),
                ChoiceEnum::Flags {
                    default: 1 << spa_sys::SPA_DATA_DmaBuf,
                    flags: Vec::new(),
                },
            ))),
        ),
        Property::new(spa_sys::SPA_PARAM_BUFFERS_blocks, Value::Int(1)),
        Property::new(
            spa_sys::SPA_PARAM_BUFFERS_buffers,
            Value::Choice(ChoiceValue::Int(Choice(
                ChoiceFlags::empty(),
                ChoiceEnum::Range { default, min, max },
            ))),
        ),
    };
    PodSerializer::serialize(std::io::Cursor::new(Vec::new()), &Value::Object(obj))
        .expect("serialize SPA dma-buf buffers pod")
        .0
        .into_inner()
}

/// The part of an `spa_data` block the dma-buf path reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DataBlock {
    pub data_type: u32,
    pub fd: i64,
    pub offset: u32,
    pub size: u32,
    pub stride: i32,
    pub maxsize: u32,
}

/// What the dma-buf capture path makes of a dequeued buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DmaBufFrame {
    /// The descriptor to share downstream.
    Ready { fd: i32, stride: u32, offset: u32 },
    /// The node produced nothing this tick (an empty chunk), not an error.
    Empty,
    /// Not one dma-buf block holding a frame of the negotiated format, so the
    /// capture fails instead of handing downstream a descriptor it cannot read.
    Unusable,
}

/// Reduce a dequeued buffer's blocks to the descriptor
/// [`MemoryDomain::DmaBuf`](g2g_core::MemoryDomain) carries.
pub(crate) fn dmabuf_frame(blocks: &[DataBlock], info: &VideoInfo) -> DmaBufFrame {
    let [block] = blocks else {
        return DmaBufFrame::Unusable;
    };
    if block.data_type != spa_sys::SPA_DATA_DmaBuf {
        return DmaBufFrame::Unusable;
    }
    if block.size == 0 {
        return DmaBufFrame::Empty;
    }
    match dmabuf_descriptor(block, info) {
        Some((fd, stride, offset)) => DmaBufFrame::Ready { fd, stride, offset },
        None => DmaBufFrame::Unusable,
    }
}

/// The `(fd, stride, offset)` of a dma-buf block, or `None` when it does not hold
/// a whole frame of the negotiated format. The daemon's numbers are not trusted:
/// the fd has to be a real descriptor and the block has to cover the frame at the
/// stride it reports.
fn dmabuf_descriptor(block: &DataBlock, info: &VideoInfo) -> Option<(i32, u32, u32)> {
    let fd = i32::try_from(block.fd).ok()?;
    if fd < 0 {
        return None;
    }
    let row_bytes = u32::try_from(single_plane_row_bytes(info.format, info.width)?).ok()?;
    // a producer that reports no stride packs its rows tightly
    let stride = match block.stride {
        0 => row_bytes,
        s => u32::try_from(s).ok()?,
    };
    if stride < row_bytes {
        return None;
    }
    let end = stride.checked_mul(info.height)?.checked_add(block.offset)?;
    (end <= block.maxsize).then_some((fd, stride, block.offset))
}

/// Tight plane geometry of a captured frame, the shape a mapped buffer is read
/// against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlaneLayout {
    format: RawVideoFormat,
    width: usize,
    height: usize,
}

impl PlaneLayout {
    /// Layout of `format` at `width` x `height`, or `None` for a format this
    /// element does not carry and geometry we will not size a buffer from.
    pub(crate) fn new(format: RawVideoFormat, width: u32, height: u32) -> Option<Self> {
        if width == 0 || height == 0 || width > MAX_DIM || height > MAX_DIM {
            return None;
        }
        supported_formats().any(|f| f == format).then_some(Self {
            format,
            width: width as usize,
            height: height as usize,
        })
    }

    /// Per-plane `(row bytes, rows, stride shift)`, where the shift derives a
    /// plane's stride from plane 0's (the only stride PipeWire reports for a
    /// single mapped block).
    fn planes(&self) -> Vec<(usize, usize, u32)> {
        crate::paddedrows::plane_shapes_with_stride_shift(self.format, self.width, self.height)
    }

    /// Bytes of a tightly packed frame in this layout.
    pub(crate) fn frame_bytes(&self) -> usize {
        self.planes()
            .iter()
            .map(|(row_bytes, rows, _)| row_bytes * rows)
            .sum()
    }

    /// How many planes the frame has.
    pub(crate) fn count(&self) -> usize {
        self.planes().len()
    }

    /// Row bytes of plane 0.
    pub(crate) fn first_row_bytes(&self) -> usize {
        self.planes()[0].0
    }

    /// Append the frame in `src` to `dst`, tightly packed. `stride` is plane 0's
    /// row stride from the buffer chunk (0 means "tight"). Every read is bounded
    /// against `src`, so a chunk that disagrees with the negotiated geometry
    /// yields `None` instead of a short or out-of-range copy. `dst` may hold a
    /// partial frame on `None`, so discard it.
    pub(crate) fn copy_tight(&self, src: &[u8], stride: usize, dst: &mut Vec<u8>) -> Option<()> {
        crate::paddedrows::pack_tight(src, self.format, self.width, self.height, stride, dst)
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
            let bytes = format_pod_bytes(g2g, FormatOffer::PreferredOnly, 320, 240, 30)
                .expect("supported format");
            let (default, alternatives) = pod_format_alternatives(&bytes);
            assert_eq!(default, Id(spa_fmt.as_raw()));
            assert_eq!(alternatives, Vec::from([Id(spa_fmt.as_raw())]));
        }
    }

    #[test]
    fn enum_format_pod_round_trips() {
        for (g2g, spa_fmt) in FORMATS {
            let bytes =
                format_pod_bytes(g2g, FormatOffer::All, 320, 240, 30).expect("supported format");
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
            format_pod_bytes(RawVideoFormat::P010, FormatOffer::All, 320, 240, 30),
            Err(G2gError::CapsMismatch)
        );
    }

    /// The dma-buf path offers single-plane formats alone: a planar frame arrives
    /// as one SPA block per plane, and the `DmaBuf` domain carries a single fd.
    #[test]
    fn the_single_plane_offer_drops_the_planar_formats() {
        assert_eq!(
            single_plane_formats().collect::<Vec<_>>(),
            Vec::from([
                RawVideoFormat::Yuyv,
                RawVideoFormat::Rgba8,
                RawVideoFormat::Bgra8
            ])
        );
        let bytes = format_pod_bytes(
            RawVideoFormat::Bgra8,
            FormatOffer::SinglePlane,
            320,
            240,
            30,
        )
        .expect("supported format");
        let (default, alternatives) = pod_format_alternatives(&bytes);
        assert_eq!(default, Id(VideoFormat::BGRA.as_raw()));
        assert_eq!(
            alternatives,
            Vec::from([
                Id(VideoFormat::BGRA.as_raw()),
                Id(VideoFormat::YUY2.as_raw()),
                Id(VideoFormat::RGBA.as_raw()),
            ])
        );
        assert_eq!(
            single_plane_row_bytes(RawVideoFormat::Bgra8, 320),
            Some(1280)
        );
        assert_eq!(single_plane_row_bytes(RawVideoFormat::Yuyv, 320), Some(640));
        assert_eq!(single_plane_row_bytes(RawVideoFormat::Nv12, 320), None);
    }

    /// The `Buffers` param the dma-buf path announces: dma-buf as the only
    /// accepted memory type, one block per buffer.
    #[test]
    fn the_dmabuf_buffers_pod_demands_dmabuf_memory() {
        let bytes = dmabuf_buffers_pod_bytes();
        let (_, value) =
            PodDeserializer::deserialize_any_from(&bytes).expect("our pod deserializes");
        let Value::Object(obj) = value else {
            panic!("expected an object pod");
        };
        assert_eq!(obj.type_, SpaTypes::ObjectParamBuffers.as_raw());
        assert_eq!(obj.id, ParamType::Buffers.as_raw());
        let prop = |key: u32| {
            obj.properties
                .iter()
                .find(|p| p.key == key)
                .map(|p| p.value.clone())
                .unwrap_or_else(|| panic!("pod carries {key}"))
        };
        let Value::Choice(ChoiceValue::Int(Choice(_, ChoiceEnum::Flags { default, .. }))) =
            prop(spa_sys::SPA_PARAM_BUFFERS_dataType)
        else {
            panic!("dataType is a flags choice");
        };
        assert_eq!(default, 1 << spa_sys::SPA_DATA_DmaBuf);
        // mapped memory is not on offer: no silent copy path
        assert_eq!(default & (1 << spa_sys::SPA_DATA_MemPtr), 0);
        assert_eq!(default & (1 << spa_sys::SPA_DATA_MemFd), 0);
        assert_eq!(prop(spa_sys::SPA_PARAM_BUFFERS_blocks), Value::Int(1));
        let Value::Choice(ChoiceValue::Int(Choice(_, ChoiceEnum::Range { default, min, .. }))) =
            prop(spa_sys::SPA_PARAM_BUFFERS_buffers)
        else {
            panic!("buffers is a range choice");
        };
        assert_eq!((default, min), (DMABUF_BUFFERS.0, DMABUF_BUFFERS.1));
    }

    /// M1059: the descriptor a dma-buf block yields is what the emitted frame
    /// declares, so a consumer that maps the buffer finds the producer's rows
    /// without going through the domain type.
    #[cfg(feature = "metadata")]
    #[test]
    fn a_dmabuf_descriptor_becomes_the_frames_plane_layout() {
        let info = VideoInfo {
            format: RawVideoFormat::Bgra8,
            width: 4,
            height: 2,
            fps_num: 30,
            fps_denom: 1,
        };
        let block = DataBlock {
            data_type: spa_sys::SPA_DATA_DmaBuf,
            fd: 7,
            offset: 8,
            size: 56,
            stride: 24,
            maxsize: 64,
        };
        let DmaBufFrame::Ready { stride, offset, .. } = dmabuf_frame(&[block], &info) else {
            panic!("a usable dma-buf block");
        };
        let layout = crate::paddedrows::padded_plane_layout(
            info.format,
            info.width as usize,
            info.height as usize,
            offset as usize,
            stride as usize,
        )
        .expect("a layout for the descriptor");
        assert_eq!(layout.count(), 1, "one dma-buf block is one plane");
        assert_eq!(
            layout.plane(0),
            Some(g2g_core::meta::Plane {
                offset: 8,
                stride: 24
            })
        );
        // row 1 sits one pitch past the block offset, not one row width
        assert_eq!(layout.row_range(0, 1, 16), Some(32..48));
    }

    /// A dequeued dma-buf buffer yields the descriptor to share downstream, and
    /// anything that is not one dma-buf block covering the frame is rejected
    /// instead of being handed on.
    #[test]
    fn a_dmabuf_block_is_validated_against_the_negotiated_format() {
        let info = VideoInfo {
            format: RawVideoFormat::Bgra8,
            width: 4,
            height: 2,
            fps_num: 30,
            fps_denom: 1,
        };
        let good = DataBlock {
            data_type: spa_sys::SPA_DATA_DmaBuf,
            fd: 7,
            offset: 0,
            size: 32,
            stride: 16,
            maxsize: 32,
        };
        assert_eq!(
            dmabuf_frame(&[good], &info),
            DmaBufFrame::Ready {
                fd: 7,
                stride: 16,
                offset: 0
            }
        );
        // a padded stride is honoured as long as the block covers it
        assert_eq!(
            dmabuf_frame(
                &[DataBlock {
                    stride: 24,
                    maxsize: 64,
                    offset: 8,
                    ..good
                }],
                &info
            ),
            DmaBufFrame::Ready {
                fd: 7,
                stride: 24,
                offset: 8
            }
        );
        // no stride reported means tight rows
        assert_eq!(
            dmabuf_frame(&[DataBlock { stride: 0, ..good }], &info),
            DmaBufFrame::Ready {
                fd: 7,
                stride: 16,
                offset: 0
            }
        );
        // an empty chunk is a tick, not a failure
        assert_eq!(
            dmabuf_frame(&[DataBlock { size: 0, ..good }], &info),
            DmaBufFrame::Empty
        );
        // mapped memory, a missing fd, a stride narrower than a row, a block that
        // does not cover the frame, and a planar format all fail loudly
        for bad in [
            DataBlock {
                data_type: spa_sys::SPA_DATA_MemFd,
                ..good
            },
            DataBlock { fd: -1, ..good },
            DataBlock { stride: 8, ..good },
            DataBlock {
                maxsize: 31,
                ..good
            },
            DataBlock { offset: 8, ..good },
        ] {
            assert_eq!(
                dmabuf_frame(&[bad], &info),
                DmaBufFrame::Unusable,
                "{bad:?}"
            );
        }
        assert_eq!(dmabuf_frame(&[], &info), DmaBufFrame::Unusable);
        assert_eq!(dmabuf_frame(&[good, good], &info), DmaBufFrame::Unusable);
        assert_eq!(
            dmabuf_frame(
                &[good],
                &VideoInfo {
                    format: RawVideoFormat::Nv12,
                    ..info
                }
            ),
            DmaBufFrame::Unusable
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
