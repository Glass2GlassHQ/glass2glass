use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AnalyticsMeta, AsyncElement, Caps, CapsConstraint, CapsTransform, ConfigureOutcome, Dim,
    ElementMetadata, FieldTransform, G2gError, MemoryDomain, OutputSink, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Rate, RawVideoFormat, RawVideoShape,
};

use crate::audiofx::double_in_range;
use crate::mathf::{exp, round_to_i64};
use crate::pixel::{even_dims_required, frame_byte_size};
use crate::videocrop::crop;
use crate::videoscale::{scale, FORMATS};

const NS_PER_SECOND: f64 = 1e9;
const NS_PER_MILLISECOND: u64 = 1_000_000;

const DEFAULT_DEAD_ZONE: f64 = 0.1;
const DEFAULT_MAX_PAN_SPEED: f64 = 0.5;
const DEFAULT_SMOOTHING_TIME_MS: u64 = 300;

static TRACKCROP_PROPERTIES: &[PropertySpec] = &[
    PropertySpec::new("width", PropKind::Uint, "output width in pixels"),
    PropertySpec::new("height", PropKind::Uint, "output height in pixels"),
    PropertySpec::new(
        "dead-zone",
        PropKind::Double,
        "distance from the crop centre, as a fraction of the crop size, within which target movement leaves the crop still",
    )
    .with_range("0", "0.5")
    .with_default("0.1"),
    PropertySpec::new(
        "max-pan-speed",
        PropKind::Double,
        "fastest crop movement, in input frame widths (or heights) per second",
    )
    .with_default("0.5"),
    PropertySpec::new(
        "smoothing-time",
        PropKind::Uint,
        "time constant in milliseconds of the low-pass filter on the crop centre, 0 for none",
    )
    .with_default("300"),
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CameraLimits {
    pub dead_zone: f64,
    pub max_pan_speed: f64,
    pub smoothing_time_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CropGeometry {
    pub frame_width: f64,
    pub frame_height: f64,
    pub crop_width: f64,
    pub crop_height: f64,
}

impl CropGeometry {
    pub(crate) fn clamp_centre(&self, centre: Point) -> Point {
        Point {
            x: clamp_axis(centre.x, self.crop_width, self.frame_width),
            y: clamp_axis(centre.y, self.crop_height, self.frame_height),
        }
    }
}

fn clamp_axis(centre: f64, crop_size: f64, frame_size: f64) -> f64 {
    centre.clamp(crop_size / 2.0, frame_size - crop_size / 2.0)
}

fn next_axis_centre(
    centre: f64,
    target: f64,
    crop_size: f64,
    frame_size: f64,
    elapsed_ns: u64,
    limits: &CameraLimits,
) -> f64 {
    let dead_zone = limits.dead_zone * crop_size;
    let offset = target - centre;
    let desired = if offset > dead_zone {
        target - dead_zone
    } else if offset < -dead_zone {
        target + dead_zone
    } else {
        centre
    };
    let follow = if limits.smoothing_time_ns == 0 {
        1.0
    } else {
        1.0 - exp(-(elapsed_ns as f64) / limits.smoothing_time_ns as f64)
    };
    let max_step = limits.max_pan_speed * frame_size * elapsed_ns as f64 / NS_PER_SECOND;
    let step = ((desired - centre) * follow).clamp(-max_step, max_step);
    clamp_axis(centre + step, crop_size, frame_size)
}

pub(crate) fn next_camera_centre(
    centre: Point,
    target: Point,
    geometry: &CropGeometry,
    elapsed_ns: u64,
    limits: &CameraLimits,
) -> Point {
    Point {
        x: next_axis_centre(
            centre.x,
            target.x,
            geometry.crop_width,
            geometry.frame_width,
            elapsed_ns,
            limits,
        ),
        y: next_axis_centre(
            centre.y,
            target.y,
            geometry.crop_height,
            geometry.frame_height,
            elapsed_ns,
            limits,
        ),
    }
}

#[derive(Debug, Clone, PartialEq)]
struct RegionTarget {
    centre: Point,
    ids: Vec<u32>,
}

fn region_target(meta: &AnalyticsMeta) -> Option<RegionTarget> {
    let mut rois = meta.rois().peekable();
    rois.peek()?;
    let mut ids = Vec::new();
    let (mut left, mut top) = (f32::MAX, f32::MAX);
    let (mut right, mut bottom) = (f32::MIN, f32::MIN);
    for roi in rois {
        ids.push(roi.id);
        left = left.min(roi.bbox.x);
        top = top.min(roi.bbox.y);
        right = right.max(roi.bbox.x + roi.bbox.w);
        bottom = bottom.max(roi.bbox.y + roi.bbox.h);
    }
    Some(RegionTarget {
        centre: Point {
            x: f64::from(left + right) / 2.0,
            y: f64::from(top + bottom) / 2.0,
        },
        ids,
    })
}

fn crop_size(format: RawVideoFormat, input: (u32, u32), output: (u32, u32)) -> (u32, u32) {
    let (in_w, in_h) = (u64::from(input.0), u64::from(input.1));
    let (out_w, out_h) = (u64::from(output.0), u64::from(output.1));
    let (w, h) = if in_w * out_h >= in_h * out_w {
        (in_h * out_w / out_h, in_h)
    } else {
        (in_w, in_w * out_h / out_w)
    };
    let (even_width, even_height) = even_dims_required(format);
    let align = |value: u64, even: bool| if even { value & !1 } else { value };
    (align(w, even_width) as u32, align(h, even_height) as u32)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct InputStream {
    format: RawVideoFormat,
    width: u32,
    height: u32,
    crop_width: u32,
    crop_height: u32,
}

impl InputStream {
    fn geometry(&self) -> CropGeometry {
        CropGeometry {
            frame_width: f64::from(self.width),
            frame_height: f64::from(self.height),
            crop_width: f64::from(self.crop_width),
            crop_height: f64::from(self.crop_height),
        }
    }

    fn crop_origin(&self, centre: Point) -> (u32, u32) {
        let (even_width, even_height) = even_dims_required(self.format);
        let origin = |centre: f64, crop: u32, frame: u32, even: bool| {
            let start = round_to_i64(centre - f64::from(crop) / 2.0)
                .clamp(0, i64::from(frame - crop)) as u32;
            if even {
                start & !1
            } else {
                start
            }
        };
        (
            origin(centre.x, self.crop_width, self.width, even_width),
            origin(centre.y, self.crop_height, self.height, even_height),
        )
    }
}

#[derive(Debug)]
pub struct TrackCrop {
    output_width: u32,
    output_height: u32,
    limits: CameraLimits,
    input: Option<InputStream>,
    framerate: Rate,
    camera_centre: Option<Point>,
    target: Option<Point>,
    followed_regions: Vec<u32>,
    last_pts_ns: Option<u64>,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for TrackCrop {
    fn default() -> Self {
        Self {
            output_width: 0,
            output_height: 0,
            limits: CameraLimits {
                dead_zone: DEFAULT_DEAD_ZONE,
                max_pan_speed: DEFAULT_MAX_PAN_SPEED,
                smoothing_time_ns: DEFAULT_SMOOTHING_TIME_MS * NS_PER_MILLISECOND,
            },
            input: None,
            framerate: Rate::Any,
            camera_centre: None,
            target: None,
            followed_regions: Vec::new(),
            last_pts_ns: None,
            last_caps: None,
            emitted: 0,
        }
    }
}

impl TrackCrop {
    pub fn new(output_width: u32, output_height: u32) -> Self {
        Self {
            output_width,
            output_height,
            ..Self::default()
        }
    }

    fn reset_camera(&mut self) {
        self.camera_centre = None;
        self.target = None;
        self.followed_regions.clear();
        self.last_pts_ns = None;
    }

    fn move_camera(&mut self, frame: &Frame, input: &InputStream) -> Point {
        let geometry = input.geometry();
        let pts_ns = frame.timing.pts_ns;
        let elapsed_ns = self
            .last_pts_ns
            .map_or(0, |last| pts_ns.saturating_sub(last));
        self.last_pts_ns = Some(pts_ns);

        let regions = frame.meta.get::<AnalyticsMeta>().and_then(region_target);
        let mut jump = false;
        if let Some(regions) = regions {
            let new_subject = !regions
                .ids
                .iter()
                .any(|id| self.followed_regions.contains(id));
            jump = new_subject || self.camera_centre.is_none();
            self.followed_regions = regions.ids;
            self.target = Some(Point {
                x: regions.centre.x * geometry.frame_width,
                y: regions.centre.y * geometry.frame_height,
            });
        }

        let frame_centre = Point {
            x: geometry.frame_width / 2.0,
            y: geometry.frame_height / 2.0,
        };
        let centre = match (self.camera_centre, self.target) {
            (_, Some(target)) if jump => geometry.clamp_centre(target),
            (Some(centre), Some(target)) => {
                next_camera_centre(centre, target, &geometry, elapsed_ns, &self.limits)
            }
            (Some(centre), None) => centre,
            (None, _) => geometry.clamp_centre(frame_centre),
        };
        self.camera_centre = Some(centre);
        centre
    }

    fn output_caps_for(&self, format: RawVideoFormat) -> Caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(self.output_width),
            height: Dim::Fixed(self.output_height),
            framerate: self.framerate.clone(),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }
}

impl AsyncElement for TrackCrop {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        for format in FORMATS {
            let candidate = Caps::RawVideo {
                format,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            };
            if let Ok(narrowed) = upstream_caps.intersect(&candidate) {
                return Ok(narrowed);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let output_size_set = self.output_width > 0 && self.output_height > 0;
        let accept = if output_size_set {
            FORMATS.to_vec()
        } else {
            Vec::new()
        };
        CapsConstraint::DerivedFields(CapsTransform::RawVideo {
            accept,
            produce: Vec::new(),
            shapes: vec![RawVideoShape::PASSTHROUGH
                .with_width(FieldTransform::Fixed(Dim::Fixed(self.output_width)))
                .with_height(FieldTransform::Fixed(Dim::Fixed(self.output_height)))],
        })
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.output_width == 0 || self.output_height == 0 {
            return Err(G2gError::NotConfigured);
        }
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !FORMATS.contains(format) || *width == 0 || *height == 0 {
            return Err(G2gError::CapsMismatch);
        }
        let (crop_width, crop_height) = crop_size(
            *format,
            (*width, *height),
            (self.output_width, self.output_height),
        );
        if crop_width == 0 || crop_height == 0 {
            return Err(G2gError::CapsMismatch);
        }
        let input = InputStream {
            format: *format,
            width: *width,
            height: *height,
            crop_width,
            crop_height,
        };
        if self.input != Some(input) {
            self.reset_camera();
        }
        self.input = Some(input);
        self.framerate = framerate.clone();
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let input = self.input.ok_or(G2gError::NotConfigured)?;
                    let src = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    if src.len() < frame_byte_size(input.format, input.width, input.height) {
                        return Err(G2gError::CapsMismatch);
                    }
                    let centre = self.move_camera(&frame, &input);
                    let (x, y) = input.crop_origin(centre);
                    let cropped = crop(
                        src,
                        input.format,
                        (input.width as usize, input.height as usize),
                        (
                            x as usize,
                            y as usize,
                            input.crop_width as usize,
                            input.crop_height as usize,
                        ),
                    );
                    let scaled = scale(
                        &cropped,
                        input.format,
                        input.crop_width as usize,
                        input.crop_height as usize,
                        self.output_width as usize,
                        self.output_height as usize,
                    );
                    let caps = self.output_caps_for(input.format);
                    if self.last_caps.as_ref() != Some(&caps) {
                        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_caps = Some(caps);
                    }
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(scaled)),
                        timing: frame.timing,
                        sequence: self.emitted,
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                // g2g's transform arm sends this element its own solved output caps.
                PipelinePacket::CapsChanged(caps) => {
                    out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                    self.last_caps = Some(caps);
                }
                PipelinePacket::Flush => {
                    self.reset_camera();
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // The runner's transform arm forwards EOS.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        TRACKCROP_PROPERTIES
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Tracking crop",
            "Filter/Effect/Video",
            "Crops a fixed-size window that follows a detected object and scales it to the output size",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "width" => self.output_width = value_as_u32(value)?,
            "height" => self.output_height = value_as_u32(value)?,
            "dead-zone" => self.limits.dead_zone = double_in_range(value, 0.0, 0.5)?,
            "max-pan-speed" => self.limits.max_pan_speed = double_in_range(value, 0.0, f64::MAX)?,
            "smoothing-time" => {
                let milliseconds = value.as_uint().ok_or(PropError::Type)?;
                self.limits.smoothing_time_ns = milliseconds
                    .checked_mul(NS_PER_MILLISECOND)
                    .ok_or(PropError::Value)?;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "width" => Some(PropValue::Uint(u64::from(self.output_width))),
            "height" => Some(PropValue::Uint(u64::from(self.output_height))),
            "dead-zone" => Some(PropValue::Double(self.limits.dead_zone)),
            "max-pan-speed" => Some(PropValue::Double(self.limits.max_pan_speed)),
            "smoothing-time" => Some(PropValue::Uint(
                self.limits.smoothing_time_ns / NS_PER_MILLISECOND,
            )),
            _ => None,
        }
    }
}

fn value_as_u32(value: PropValue) -> Result<u32, PropError> {
    let value = value.as_uint().ok_or(PropError::Type)?;
    u32::try_from(value).map_err(|_| PropError::Value)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME_PERIOD_NS: u64 = 40_000_000;

    fn landscape_to_portrait() -> CropGeometry {
        CropGeometry {
            frame_width: 1920.0,
            frame_height: 1080.0,
            crop_width: 608.0,
            crop_height: 1080.0,
        }
    }

    fn limits() -> CameraLimits {
        CameraLimits {
            dead_zone: DEFAULT_DEAD_ZONE,
            max_pan_speed: DEFAULT_MAX_PAN_SPEED,
            smoothing_time_ns: DEFAULT_SMOOTHING_TIME_MS * NS_PER_MILLISECOND,
        }
    }

    #[test]
    fn a_target_inside_the_dead_zone_leaves_the_crop_still() {
        let geometry = landscape_to_portrait();
        let centre = Point { x: 960.0, y: 540.0 };
        let inside = Point {
            x: centre.x + DEFAULT_DEAD_ZONE * geometry.crop_width - 1.0,
            y: centre.y,
        };
        let next = next_camera_centre(centre, inside, &geometry, FRAME_PERIOD_NS, &limits());
        assert_eq!(next, centre);
    }

    #[test]
    fn one_step_never_exceeds_the_pan_speed() {
        let geometry = landscape_to_portrait();
        let unsmoothed = CameraLimits {
            smoothing_time_ns: 0,
            ..limits()
        };
        let centre = Point { x: 400.0, y: 540.0 };
        let far = Point {
            x: 1600.0,
            y: 540.0,
        };
        let next = next_camera_centre(centre, far, &geometry, FRAME_PERIOD_NS, &unsmoothed);
        let max_step =
            DEFAULT_MAX_PAN_SPEED * geometry.frame_width * FRAME_PERIOD_NS as f64 / NS_PER_SECOND;
        assert!((next.x - centre.x - max_step).abs() < 1e-9, "{next:?}");
    }

    #[test]
    fn the_crop_stays_inside_the_frame_at_both_edges() {
        let geometry = landscape_to_portrait();
        let unlimited = CameraLimits {
            dead_zone: 0.0,
            max_pan_speed: f64::MAX,
            smoothing_time_ns: 0,
        };
        let centre = Point { x: 960.0, y: 540.0 };
        let left = next_camera_centre(
            centre,
            Point { x: 0.0, y: 0.0 },
            &geometry,
            FRAME_PERIOD_NS,
            &unlimited,
        );
        let right = next_camera_centre(
            centre,
            Point {
                x: 1920.0,
                y: 1080.0,
            },
            &geometry,
            FRAME_PERIOD_NS,
            &unlimited,
        );
        assert_eq!(left, Point { x: 304.0, y: 540.0 });
        assert_eq!(
            right,
            Point {
                x: 1616.0,
                y: 540.0
            }
        );
    }

    #[test]
    fn the_crop_aspect_matches_the_output_aspect() {
        assert_eq!(
            crop_size(RawVideoFormat::I420, (1920, 1080), (1080, 1920)),
            (606, 1080)
        );
        assert_eq!(
            crop_size(RawVideoFormat::Rgba8, (720, 1280), (1080, 1080)),
            (720, 720)
        );
    }

    fn frame_with_regions(pts_ns: u64, regions: &[(f32, u32)]) -> Frame {
        let mut meta = AnalyticsMeta::new();
        for &(x, id) in regions {
            meta.push(g2g_core::AnalyticsNode::Roi(g2g_core::Roi {
                bbox: g2g_core::BBox {
                    x,
                    y: 0.5,
                    w: 0.0,
                    h: 0.0,
                },
                id,
                label: 0,
            }));
        }
        let mut frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; 4].into_boxed_slice())),
            g2g_core::frame::FrameTiming::default(),
            0,
        );
        frame.timing.pts_ns = pts_ns;
        frame.meta.attach(meta);
        frame
    }

    #[test]
    fn the_crop_jumps_only_to_a_new_subject() {
        let mut element = TrackCrop::new(1080, 1920);
        let caps = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        element.configure_pipeline(&caps).unwrap();
        let input = element.input.unwrap();
        let geometry = input.geometry();

        let first = element.move_camera(&frame_with_regions(0, &[(0.25, 1)]), &input);
        assert_eq!(
            first.x,
            0.25 * geometry.frame_width,
            "the first region is jumped to"
        );

        // the same subject moving far pans instead
        let panned = element.move_camera(&frame_with_regions(FRAME_PERIOD_NS, &[(0.5, 1)]), &input);
        assert!(
            panned.x > first.x && panned.x < 0.5 * geometry.frame_width,
            "{panned:?}"
        );

        let switched = element.move_camera(
            &frame_with_regions(2 * FRAME_PERIOD_NS, &[(0.75, 2)]),
            &input,
        );
        assert_eq!(
            switched.x,
            0.75 * geometry.frame_width,
            "a new subject is jumped to"
        );
    }

    #[test]
    fn several_regions_are_framed_together() {
        let mut meta = AnalyticsMeta::new();
        for (x, id) in [(0.25, 1), (0.5, 2)] {
            meta.push(g2g_core::AnalyticsNode::Roi(g2g_core::Roi {
                bbox: g2g_core::BBox {
                    x,
                    y: 0.25,
                    w: 0.125,
                    h: 0.5,
                },
                id,
                label: 0,
            }));
        }
        let target = region_target(&meta).unwrap();
        assert_eq!(target.centre, Point { x: 0.4375, y: 0.5 });
        assert_eq!(target.ids, [1, 2]);
    }
}
