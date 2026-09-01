//! ONNX Runtime instance-segmentation element (M992, DESIGN.md §5.3): the in-tree
//! producer of the `Segmentation` / `Roi` analytics nodes.
//!
//! `OrtSegmentation` runs a YOLO `-seg` model (Ultralytics YOLOv8-seg /
//! YOLO11-seg) and attaches the decoded instances to the frame it forwards
//! unchanged, so the picture and its masks travel together to an overlay or a
//! recorder. Unlike `OrtInference -> DetectionPostprocess`, the model's two
//! outputs (boxes + mask coefficients, and the mask prototypes) never leave the
//! element: a tensor frame carries one tensor, and the masks need both. The decode
//! itself is [`crate::segmentation`], reusable without an inference engine.
//!
//! Input pad, like `OrtInference`: RGBA at the model's geometry (normalized on the
//! CPU here), or an already-normalized f32 NCHW `[1, 3, H, W]` tensor under
//! [`with_tensor_input`](OrtSegmentation::with_tensor_input) (e.g. from
//! `WgpuPreprocess`). The output pad equals the input: this is an identity
//! transform that adds metadata.
//!
//! Model contract (checked when the model loads, fails loud):
//! - one f32 input, rank-4 `[N, 3, H, W]`, `N` 1 or dynamic, static `H` / `W`
//! - exactly two f32 outputs with static dims: a rank-3 `[1, 4 + C + M, A]` box
//!   output and a rank-4 `[1, M, mh, mw]` prototype output, `C >= 1`
//!
//! The `model` property loads the session after construction, so a launch line can
//! say `ortsegment model=yolo11n-seg.onnx mask-threshold=0.6`.

use core::future::Future;
use core::pin::Pin;

use ::ort::session::Session;
use ::ort::value::Tensor;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, HardwareError,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
    TensorDType, TensorLayout, TensorShape,
};

use crate::ortinfer::{
    f32_tensor_dims, input_geometry, input_tensor_dims, ort_err, rgba_to_chw, static_output_dims,
    tensor_bytes_to_chw,
};
use crate::segmentation::{decode_instances, SegmentationGeometry, SegmentationThresholds};

/// A loaded segmentation session plus the geometry read off its two outputs.
/// Separate from the element so the element exists before a model does: a launch
/// line builds `ortsegment` bare and the `model` property loads this.
#[derive(Debug)]
struct SegModel {
    session: Session,
    input_name: String,
    /// The rank-3 `[1, 4 + C + M, A]` output.
    box_name: String,
    /// The rank-4 `[1, M, mh, mw]` prototype output.
    proto_name: String,
    geometry: SegmentationGeometry,
}

/// # Example
///
/// ```no_run
/// use g2g_ml::ortsegment::OrtSegmentation;
///
/// // gst-launch style: ortsegment model=yolo11n-seg.onnx
/// let element = OrtSegmentation::from_file("yolo11n-seg.onnx").unwrap();
/// ```
#[derive(Debug)]
pub struct OrtSegmentation {
    /// `None` until a model is loaded; negotiation and `process` fail with
    /// `NotConfigured` while it is.
    model: Option<SegModel>,
    /// Path the `model` property loaded from, for read-back.
    model_path: Option<String>,
    /// When set, the input pad is a preprocessed NCHW `Caps::Tensor` fed straight
    /// to the session, not RGBA normalized on the CPU.
    tensor_input: bool,
    thresholds: SegmentationThresholds,
    configured: bool,
    emitted: u64,
}

impl Default for OrtSegmentation {
    fn default() -> Self {
        Self::new()
    }
}

impl OrtSegmentation {
    /// A model-less element, the text-construction path: `ortsegment
    /// model=yolo11n-seg.onnx` builds this and then loads the model through the
    /// `model` property.
    pub fn new() -> Self {
        Self {
            model: None,
            model_path: None,
            tensor_input: false,
            thresholds: SegmentationThresholds::default(),
            configured: false,
            emitted: 0,
        }
    }

    /// Build a session from in-memory ONNX model bytes and validate the model
    /// contract (see module docs).
    pub fn from_memory(model_bytes: &[u8]) -> Result<Self, G2gError> {
        let mut builder = Session::builder().map_err(ort_err)?;
        let session = builder.commit_from_memory(model_bytes).map_err(ort_err)?;
        let mut element = Self::new();
        element.model = Some(SegModel::load(session)?);
        Ok(element)
    }

    /// As [`from_memory`](Self::from_memory), reading the model from a file.
    pub fn from_file(path: &str) -> Result<Self, G2gError> {
        let mut element = Self::new();
        element.load_model(path)?;
        Ok(element)
    }

    /// Load (or replace) the model from a file, validating the same contract the
    /// constructors do. The tensor-input mode and thresholds are preserved, so the
    /// properties can be set in any order.
    pub fn load_model(&mut self, path: &str) -> Result<(), G2gError> {
        let mut builder = Session::builder().map_err(ort_err)?;
        let session = builder.commit_from_file(path).map_err(ort_err)?;
        self.model = Some(SegModel::load(session)?);
        self.model_path = Some(path.to_owned());
        Ok(())
    }

    /// Accept an already-normalized f32 NCHW `[1, 3, H, W]` tensor input instead of
    /// RGBA, feeding it straight to the session with no CPU normalize.
    pub fn with_tensor_input(mut self) -> Self {
        self.tensor_input = true;
        self
    }

    /// Set the confidence / IoU / mask-coverage cutoffs the decode applies.
    pub fn with_thresholds(mut self, thresholds: SegmentationThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// The model's expected input geometry, `(width, height)`; `(0, 0)` before a
    /// model is loaded.
    pub fn input_dims(&self) -> (u32, u32) {
        self.model.as_ref().map_or((0, 0), |m| {
            (m.geometry.input_width, m.geometry.input_height)
        })
    }

    /// The geometry read off the loaded model's two outputs; `None` before a model
    /// is loaded. The prototype grid is the mask resolution consumers see.
    pub fn geometry(&self) -> Option<SegmentationGeometry> {
        self.model.as_ref().map(|m| m.geometry)
    }

    /// Count of frames processed (one analytics graph per input frame).
    pub fn processed_count(&self) -> u64 {
        self.emitted
    }

    fn supported_input(&self) -> Option<Caps> {
        let model = self.model.as_ref()?;
        let (width, height) = (model.geometry.input_width, model.geometry.input_height);
        Some(if self.tensor_input {
            Caps::Tensor {
                dtype: TensorDType::F32,
                shape: TensorShape::new([1, 3, height, width]),
                layout: TensorLayout::Nchw,
            }
        } else {
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(width),
                height: Dim::Fixed(height),
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            }
        })
    }
}

impl SegModel {
    fn load(session: Session) -> Result<Self, G2gError> {
        let [input] = session.inputs() else {
            return Err(G2gError::CapsMismatch);
        };
        let [first, second] = session.outputs() else {
            return Err(G2gError::CapsMismatch);
        };
        let input_name = input.name().to_owned();
        let (input_dims, input_dtype) = input_tensor_dims(input.dtype())?;
        // The masks are decoded from f32 prototypes; a quantized-input export is
        // not this element's model.
        if input_dtype != TensorDType::F32 {
            return Err(G2gError::CapsMismatch);
        }
        let (input_width, input_height) = input_geometry(&input_dims)?;

        // The two outputs are told apart by rank, not by export name: rank 3 is the
        // box output, rank 4 the prototypes.
        let candidates = [
            (first.name().to_owned(), f32_tensor_dims(first.dtype())?),
            (second.name().to_owned(), f32_tensor_dims(second.dtype())?),
        ];
        let [(box_name, box_dims), (proto_name, proto_dims)] = match candidates {
            [a, b] if a.1.len() == 3 && b.1.len() == 4 => [a, b],
            [a, b] if a.1.len() == 4 && b.1.len() == 3 => [b, a],
            _ => return Err(G2gError::CapsMismatch),
        };
        let box_shape = static_output_dims(&box_dims)?;
        let proto_shape = static_output_dims(&proto_dims)?;
        let [1, channels, anchors] = *box_shape.dims() else {
            return Err(G2gError::CapsMismatch);
        };
        let [1, prototypes, proto_height, proto_width] = *proto_shape.dims() else {
            return Err(G2gError::CapsMismatch);
        };
        // channels = 4 box + C classes + M mask coefficients, so a model whose two
        // outputs disagree about M leaves no room for a class.
        let classes = (channels as usize)
            .checked_sub(4 + prototypes as usize)
            .filter(|c| *c >= 1)
            .ok_or(G2gError::CapsMismatch)?;

        Ok(Self {
            session,
            input_name,
            box_name,
            proto_name,
            geometry: SegmentationGeometry {
                input_width,
                input_height,
                anchors: anchors as usize,
                classes,
                prototypes: prototypes as usize,
                proto_width: proto_width as usize,
                proto_height: proto_height as usize,
            },
        })
    }

    /// Run the session on a `[1, 3, H, W]` f32 plane, returning the box output and
    /// the prototype output as f32 values.
    fn run(&mut self, chw: Vec<f32>) -> Result<(Vec<f32>, Vec<f32>), G2gError> {
        let shape = vec![
            1i64,
            3,
            self.geometry.input_height as i64,
            self.geometry.input_width as i64,
        ];
        let tensor = Tensor::from_array((shape, chw)).map_err(ort_err)?;
        let outputs = self
            .session
            .run(::ort::inputs![self.input_name.as_str() => tensor])
            .map_err(ort_err)?;
        let extract = |name: &str| -> Result<Vec<f32>, G2gError> {
            let value = outputs
                .get(name)
                .ok_or(G2gError::Hardware(HardwareError::Other))?;
            let (_, data) = value.try_extract_tensor::<f32>().map_err(ort_err)?;
            Ok(data.to_vec())
        };
        Ok((extract(&self.box_name)?, extract(&self.proto_name)?))
    }
}

impl AsyncElement for OrtSegmentation {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        let supported = self.supported_input().ok_or(G2gError::NotConfigured)?;
        upstream_caps.intersect(&supported)
    }

    /// Identity: the frame passes through unchanged, only metadata is added. With
    /// no model loaded the set is empty, so negotiation fails instead of guessing
    /// an input geometry.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(match self.supported_input() {
            Some(caps) => CapsSet::one(caps),
            None => CapsSet::from_alternatives(Vec::new()),
        })
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let supported = self.supported_input().ok_or(G2gError::NotConfigured)?;
        absolute_caps.intersect(&supported)?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        ORT_SEGMENT_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "model" => {
                let path = value.as_str().ok_or(PropError::Type)?;
                // A path that does not load (missing, not ONNX, or not a two-output
                // segmentation export) is a bad value, so the line fails loud.
                self.load_model(path).map_err(|_| PropError::Value)
            }
            "tensor-input" => {
                self.tensor_input = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "conf-threshold" | "iou-threshold" | "mask-threshold" => {
                let v = value.as_double().ok_or(PropError::Type)?;
                // All three are scores in [0, 1]; outside it they would suppress
                // every instance or none.
                if !(0.0..=1.0).contains(&v) {
                    return Err(PropError::Value);
                }
                match name {
                    "conf-threshold" => self.thresholds.confidence = v as f32,
                    "iou-threshold" => self.thresholds.iou = v as f32,
                    _ => self.thresholds.coverage = v as f32,
                }
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "model" => Some(PropValue::Str(self.model_path.clone().unwrap_or_default())),
            "tensor-input" => Some(PropValue::Bool(self.tensor_input)),
            "conf-threshold" => Some(PropValue::Double(self.thresholds.confidence as f64)),
            "iou-threshold" => Some(PropValue::Double(self.thresholds.iou as f64)),
            "mask-threshold" => Some(PropValue::Double(self.thresholds.coverage as f64)),
            _ => None,
        }
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
                PipelinePacket::DataFrame(mut frame) => {
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let (width, height) = self.input_dims();
                    let chw = if self.tensor_input {
                        tensor_bytes_to_chw(slice, width, height)?
                    } else {
                        rgba_to_chw(slice, width, height)?
                    };
                    let thresholds = self.thresholds;
                    let model = self.model.as_mut().ok_or(G2gError::NotConfigured)?;
                    let geometry = model.geometry;
                    let (boxes, protos) = model.run(chw)?;
                    let analytics = decode_instances(&geometry, &thresholds, &boxes, &protos);
                    // Attach the graph and forward the (unchanged) frame, so the
                    // picture and its masks reach the overlay together.
                    frame.meta.attach(analytics);
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                // Identity caps: the runner's transform arm pushes our pre-fixed
                // output caps here, which equal the input, so forward unchanged.
                PipelinePacket::CapsChanged(caps) => {
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                // Drop EOS: the runner's transform arm forwards it; re-pushing it
                // here double-pushes onto a full link (the project EOS contract).
                PipelinePacket::Eos => {}
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // PipelinePacket is non_exhaustive: forward variants added since
                // unchanged.
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Settable properties: the model to load, the input-pad mode, and the three
/// decode cutoffs, so a `gst-launch` line can tune the producer without the
/// builders.
static ORT_SEGMENT_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "model",
        PropKind::Str,
        "path to the ONNX instance-segmentation model file",
    ),
    PropertySpec::new(
        "tensor-input",
        PropKind::Bool,
        "take a preprocessed f32 NCHW tensor instead of RGBA video",
    ),
    PropertySpec::new(
        "conf-threshold",
        PropKind::Double,
        "minimum class score to emit an instance, 0..1",
    ),
    PropertySpec::new(
        "iou-threshold",
        PropKind::Double,
        "IoU above which a lower-scoring same-class box is suppressed, 0..1",
    ),
    PropertySpec::new(
        "mask-threshold",
        PropKind::Double,
        "minimum mask probability for a sample to count as covered, 0..1",
    ),
];

/// The RGBA input pad is the static superset (the model's geometry narrows it at
/// instance time), and the output pad equals it.
#[cfg(feature = "launch")]
impl g2g_core::PadTemplates for OrtSegmentation {
    fn pad_templates() -> Vec<g2g_core::PadTemplate> {
        let any_rgba = CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        });
        Vec::from([
            g2g_core::PadTemplate::sink(any_rgba.clone()),
            g2g_core::PadTemplate::source(any_rgba),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negotiation_and_process_fail_without_a_model() {
        let mut element = OrtSegmentation::new();
        assert_eq!(element.geometry(), None);
        assert_eq!(element.input_dims(), (0, 0));
        let rgba = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(640),
            height: Dim::Fixed(640),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert_eq!(
            element.intercept_caps(&rgba),
            Err(G2gError::NotConfigured),
            "no model, no negotiation"
        );
        assert_eq!(
            element.configure_pipeline(&rgba).unwrap_err(),
            G2gError::NotConfigured
        );
        assert!(matches!(
            element.caps_constraint_as_transform(),
            CapsConstraint::Identity(set) if set.is_empty()
        ));
    }

    #[test]
    fn thresholds_are_properties_bounded_to_scores() {
        let mut element = OrtSegmentation::new();
        for name in ["conf-threshold", "iou-threshold", "mask-threshold"] {
            // 0.625 is exact in binary, so the f32 round-trip is comparable.
            element
                .set_property(name, PropValue::Double(0.625))
                .unwrap();
            assert_eq!(element.get_property(name), Some(PropValue::Double(0.625)));
            assert_eq!(
                element.set_property(name, PropValue::Double(1.5)),
                Err(PropError::Value)
            );
            assert_eq!(
                element.set_property(name, PropValue::Bool(true)),
                Err(PropError::Type)
            );
        }
        element
            .set_property("tensor-input", PropValue::Bool(true))
            .unwrap();
        assert_eq!(
            element.get_property("tensor-input"),
            Some(PropValue::Bool(true))
        );
        assert_eq!(
            element.set_property("model", PropValue::Str("/nonexistent.onnx".into())),
            Err(PropError::Value)
        );
    }
}
