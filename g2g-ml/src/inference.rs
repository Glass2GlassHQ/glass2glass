use core::fmt::Debug;
use core::future::Future;
use core::pin::Pin;

use crate::tensor_bytes::{rgba_to_nchw, tensor_bytes_to_chw};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementBound, G2gError,
    MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate,
    RawVideoFormat, TensorDType, TensorLayout, TensorShape,
};

// one channel per RGB plane of the `[1, 3, H, W]` model input
const INPUT_CHANNELS: usize = 3;

pub trait InferenceBackend: Default + Debug + ElementBound {
    type Model: LoadedModel + Debug + ElementBound;

    // every property the element declares, the base ones included
    const PROPERTIES: &'static [PropertySpec];

    fn load(&self, path: &str) -> Result<Self::Model, G2gError>;

    // backend-specific names only, anything else is `PropError::Unknown`
    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError>;

    fn get_property(&self, name: &str) -> Option<PropValue>;
}

pub trait LoadedModel {
    fn geometry(&self) -> &ModelGeometry;

    // a `[1, 3, H, W]` f32 plane in, the first output's little-endian f32 bytes and dims out
    fn run_f32(&mut self, chw: Vec<f32>) -> Result<(Box<[u8]>, Vec<u32>), G2gError>;

    // exactly `3 * H * W` u8 / i8 values, one byte each
    fn run_int(&mut self, bytes: &[u8]) -> Result<(Box<[u8]>, Vec<u32>), G2gError>;
}

#[derive(Debug, Clone, Copy)]
pub struct ModelGeometry {
    /// Model input geometry (static `W x H` from the `[N, 3, H, W]` input).
    pub width: u32,
    pub height: u32,
    /// The model's input element type: `F32` (the RGBA / f32-tensor path) or `U8`
    /// / `I8` (a quantized model, fed an integer tensor straight through, M442 the
    /// `TensorConvert::quantize` output). RGBA mode requires `F32`.
    pub input_dtype: TensorDType,
    /// Static output shape (a dynamic leading batch dim coerced to 1).
    pub output_shape: TensorShape,
}

#[derive(Debug)]
pub struct Inference<B: InferenceBackend> {
    /// `None` until a model is loaded; negotiation and `process` fail with
    /// `NotConfigured` while it is.
    pub(crate) model: Option<B::Model>,
    /// Path the `model` property loaded from, for read-back.
    model_path: Option<String>,
    /// When set, the input pad is a preprocessed NCHW `Caps::Tensor` fed straight
    /// to the session, not RGBA normalized on the CPU.
    tensor_input: bool,
    backend: B,
    /// When set, the video frame goes out unchanged with the model output
    /// attached as a `TensorMeta` instead of becoming the tensor frame.
    #[cfg(feature = "analytics")]
    attach_tensor: bool,
    /// Name this element's output is attached under, so a frame carrying
    /// several models' outputs keeps them apart. Empty for a single model.
    #[cfg(feature = "analytics")]
    tensor_name: String,
    /// Name of the metadata a frame must carry to be inferred on: a blob header,
    /// or `detections` for any detection. Empty runs the model on every frame.
    only_on: g2g_core::OnlyOnGate,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl<B: InferenceBackend> Default for Inference<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: InferenceBackend> Inference<B> {
    /// A model-less element, the text-construction path: `ortinfer
    /// model=yolov8n.onnx` builds this and then loads the model through the
    /// `model` property. Until one is loaded, negotiation and `process` fail with
    /// [`G2gError::NotConfigured`].
    pub fn new() -> Self {
        Self {
            model: None,
            model_path: None,
            tensor_input: false,
            backend: B::default(),
            #[cfg(feature = "analytics")]
            attach_tensor: false,
            #[cfg(feature = "analytics")]
            tensor_name: String::new(),
            only_on: g2g_core::OnlyOnGate::default(),
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    /// Reads the model from a file, validating the model contract.
    pub fn from_file(path: &str) -> Result<Self, G2gError> {
        let mut element = Self::new();
        element.load_model(path)?;
        Ok(element)
    }

    /// Load (or replace) the model from a file, validating the same v1 contract
    /// the constructors do. The tensor-input mode is preserved, so `tensor-input`
    /// and `model` can be set in either order.
    pub fn load_model(&mut self, path: &str) -> Result<(), G2gError> {
        self.model = Some(self.backend.load(path)?);
        self.model_path = Some(path.to_owned());
        Ok(())
    }

    /// The model's expected input geometry, `(width, height)`; `(0, 0)` before a
    /// model is loaded.
    pub fn input_dims(&self) -> (u32, u32) {
        self.geometry()
            .map_or((0, 0), |geometry| (geometry.width, geometry.height))
    }

    /// The model's static output tensor dims; empty before a model is loaded.
    pub fn output_shape(&self) -> &[u32] {
        self.model
            .as_ref()
            .map_or(&[][..], |m| m.geometry().output_shape.dims())
    }

    /// Count of tensor `DataFrame`s pushed downstream. Useful in tests.
    pub fn inferred_count(&self) -> u64 {
        self.emitted
    }

    /// Accept an already-normalized f32 NCHW `[1, 3, H, W]` tensor input
    /// (e.g. from `WgpuPreprocess` / `WebGPUPreprocess`) instead of RGBA,
    /// feeding it straight to the session with no CPU normalize. The model
    /// geometry is unchanged.
    pub fn with_tensor_input(mut self) -> Self {
        self.tensor_input = true;
        self
    }

    /// Keep the video frame on the wire and attach the model output to it as a
    /// [`TensorMeta`](g2g_core::TensorMeta) instead of emitting the tensor as
    /// the frame. Lets `ortinfer ! detectionpostprocess ! analyticsoverlay` run
    /// as one straight chain, since the frame that reaches the overlay is still
    /// the picture the model saw.
    #[cfg(feature = "analytics")]
    pub fn with_attach_tensor(mut self) -> Self {
        self.attach_tensor = true;
        self
    }

    /// Name this element's output is attached under, so a second model on the
    /// same frame does not collide with it. Only needed with more than one.
    #[cfg(feature = "analytics")]
    pub fn with_tensor_name(mut self, name: &str) -> Self {
        self.tensor_name = String::from(name);
        self
    }

    fn geometry(&self) -> Option<ModelGeometry> {
        self.model.as_ref().map(|m| *m.geometry())
    }

    fn supported_input(&self) -> Option<Caps> {
        let m = self.geometry()?;
        Some(if self.tensor_input {
            Caps::Tensor {
                // The model's own input dtype: f32, or u8 / i8 for a quantized
                // model fed an already-quantized tensor (M442).
                dtype: m.input_dtype,
                shape: TensorShape::new([1, 3, m.height, m.width]),
                layout: TensorLayout::Nchw,
            }
        } else {
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(m.width),
                height: Dim::Fixed(m.height),
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            }
        })
    }

    fn output_caps(&self) -> Option<Caps> {
        Some(Caps::Tensor {
            dtype: TensorDType::F32,
            shape: self.geometry()?.output_shape,
            // rank-4 outputs are channel-first by the input convention;
            // for other ranks the layout tag is nominal.
            layout: TensorLayout::Nchw,
        })
    }
}

impl<B: InferenceBackend> AsyncElement for Inference<B> {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        let supported = self.supported_input().ok_or(G2gError::NotConfigured)?;
        upstream_caps.intersect(&supported)
    }

    /// Native `DerivedOutput`: RGBA at the model's geometry in, the model's
    /// static tensor caps out. Non-matching input yields an empty set, so
    /// the solver rejects it at negotiation time. With no model loaded the set
    /// is empty for every input, so negotiation fails instead of guessing caps.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        #[cfg(feature = "analytics")]
        let attach = self.attach_tensor;
        #[cfg(not(feature = "analytics"))]
        let attach = false;
        let loaded = self.supported_input().zip(self.output_caps());
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match &loaded {
            Some((supported, out)) => match input.intersect(supported) {
                // attach-tensor keeps the picture: identity output, the model
                // output rides along as meta.
                Ok(narrowed) => CapsSet::one(if attach { narrowed } else { out.clone() }),
                Err(_) => CapsSet::from_alternatives(Vec::new()),
            },
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // A gated frame goes downstream exactly as it arrived, which matches the
        // negotiated output caps only while this element keeps the picture on the
        // wire. Emitting the tensor as the frame makes the two disagree.
        #[cfg(feature = "analytics")]
        if self.only_on.is_set() && !self.attach_tensor {
            return Err(G2gError::CapsMismatch);
        }
        let geometry = self.geometry().ok_or(G2gError::NotConfigured)?;
        // RGBA mode normalizes to f32, so it only feeds an f32 model; a quantized
        // (u8 / i8) model must take a pre-quantized tensor via `with_tensor_input`.
        if !self.tensor_input && geometry.input_dtype != TensorDType::F32 {
            return Err(G2gError::CapsMismatch);
        }
        let supported = self.supported_input().ok_or(G2gError::NotConfigured)?;
        absolute_caps.intersect(&supported)?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        B::PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "model" => {
                let path = value.as_str().ok_or(PropError::Type)?;
                // A path that does not load (missing, not a model, or violating
                // the v1 contract) is a bad value, so the launch line fails loud.
                self.load_model(path).map_err(|_| PropError::Value)
            }
            "tensor-input" => {
                self.tensor_input = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            #[cfg(feature = "analytics")]
            "attach-tensor" => {
                self.attach_tensor = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            #[cfg(feature = "analytics")]
            "tensor-name" => {
                self.tensor_name = String::from(value.as_str().ok_or(PropError::Type)?);
                Ok(())
            }
            g2g_core::OnlyOnGate::PROPERTY => self.only_on.set(&value),
            _ => {
                self.backend.set_property(name, value)?;
                // Set after `model=`, so rebuild the model on the new backend
                // setting rather than making the launch line's property order matter.
                if let Some(path) = self.model_path.clone() {
                    self.load_model(&path).map_err(|_| PropError::Value)?;
                }
                Ok(())
            }
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "model" => Some(PropValue::Str(self.model_path.clone().unwrap_or_default())),
            "tensor-input" => Some(PropValue::Bool(self.tensor_input)),
            #[cfg(feature = "analytics")]
            "attach-tensor" => Some(PropValue::Bool(self.attach_tensor)),
            #[cfg(feature = "analytics")]
            "tensor-name" => Some(PropValue::Str(self.tensor_name.clone())),
            g2g_core::OnlyOnGate::PROPERTY => Some(self.only_on.value()),
            _ => self.backend.get_property(name),
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
                PipelinePacket::DataFrame(frame) => {
                    if self.only_on.forwards_untouched(&frame.meta) {
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                        return Ok(());
                    }
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let tensor_input = self.tensor_input;
                    let model = self.model.as_mut().ok_or(G2gError::NotConfigured)?;
                    let geometry = *model.geometry();
                    let (bytes, dims) = if tensor_input {
                        if geometry.input_dtype == TensorDType::F32 {
                            model.run_f32(tensor_bytes_to_chw(
                                slice,
                                geometry.width,
                                geometry.height,
                            )?)?
                        } else {
                            model.run_int(int_tensor_bytes(slice, &geometry)?)?
                        }
                    } else {
                        // RGBA8 -> normalized f32 NCHW RGB, then run the model.
                        model.run_f32(rgba_to_nchw(slice, geometry.width, geometry.height)?)?
                    };
                    let shape = TensorShape::from_slice(&dims).ok_or(G2gError::CapsMismatch)?;
                    #[cfg(feature = "analytics")]
                    if self.attach_tensor {
                        // Caps are identity here, so no CapsChanged: the frame
                        // going out is the one that came in, carrying the model
                        // output for a post-processor downstream.
                        let mut frame = frame;
                        let produced = g2g_core::NamedTensor {
                            name: self.tensor_name.clone(),
                            dtype: TensorDType::F32,
                            shape,
                            layout: TensorLayout::Nchw,
                            data: bytes.into_vec(),
                        };
                        // Add to whatever an earlier inference stage already put
                        // on this frame rather than displacing it.
                        match frame.meta.get_mut::<g2g_core::TensorMeta>() {
                            Some(tensors) => tensors.push(produced),
                            None => {
                                let mut tensors = g2g_core::TensorMeta::new();
                                tensors.push(produced);
                                frame.meta.attach(tensors);
                            }
                        }
                        self.emitted += 1;
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                        return Ok(());
                    }
                    let new_caps = Caps::Tensor {
                        dtype: TensorDType::F32,
                        shape,
                        layout: TensorLayout::Nchw,
                    };
                    if self.last_caps.as_ref() != Some(&new_caps) {
                        out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                            .await?;
                        self.last_caps = Some(new_caps);
                    }
                    let tensor_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes)),
                        // inference is per-frame: the tensor inherits the
                        // source frame's timing so latency stays traceable.
                        timing: frame.timing,
                        sequence: self.emitted,
                        meta: Default::default(),
                    };
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(tensor_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // Runner contract (the videoconvert / videoscale convention):
                    // the transform arm calls `configure_pipeline` (input) then
                    // `configure_output` (output) and pushes this packet carrying
                    // the pre-fixed forward *output* caps, NOT a new input. So `c`
                    // is our output tensor (the model's), not something to validate
                    // against `supported_input` (a real model's output never
                    // matches its input, so the old `c.intersect(supported_input)?`
                    // hard-errored whenever ORT was an interior transform). Forward
                    // it so a strict downstream sees the tensor caps before the
                    // first frame, and record `last_caps` to suppress the data
                    // path's duplicate emit.
                    self.last_caps = Some(c.clone());
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is a timing marker: forward unchanged.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // stateless per-frame inference: nothing to drain.
                PipelinePacket::Eos => {}
                // PipelinePacket is non_exhaustive: forward variants added
                // since (a metadata-carrying packet, etc.) unchanged.
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

// the leading `[1, 3, H, W]` one-byte elements of a quantized tensor frame
fn int_tensor_bytes<'a>(bytes: &'a [u8], geometry: &ModelGeometry) -> Result<&'a [u8], G2gError> {
    let plane = (geometry.width as usize)
        .checked_mul(geometry.height as usize)
        .ok_or(G2gError::CapsMismatch)?;
    let n = plane
        .checked_mul(INPUT_CHANNELS)
        .ok_or(G2gError::CapsMismatch)?;
    bytes.get(..n).ok_or(G2gError::CapsMismatch)
}

/// Settable properties: the model file to load and the input-pad mode, so a
/// `gst-launch` line can build the element without the Rust constructors.
pub const MODEL_PROP: PropertySpec =
    PropertySpec::new("model", PropKind::Str, "path to the model file");
pub const TENSOR_INPUT_PROP: PropertySpec = PropertySpec::new(
    "tensor-input",
    PropKind::Bool,
    "take a preprocessed f32 NCHW tensor instead of RGBA video",
);
#[cfg(feature = "analytics")]
pub const ATTACH_TENSOR_PROP: PropertySpec = PropertySpec::new(
    "attach-tensor",
    PropKind::Bool,
    "forward the video frame with the model output attached as metadata, instead of emitting the tensor as the frame",
);
#[cfg(feature = "analytics")]
pub const TENSOR_NAME_PROP: PropertySpec = PropertySpec::new(
    "tensor-name",
    PropKind::Str,
    "name the attached output is stored under, to keep several models on one frame apart",
);

pub const ONLY_ON_PROP: PropertySpec = g2g_core::OnlyOnGate::property_spec(
    "Run only on frames carrying this blob, or any detection when set to detections. Other frames pass through untouched, so this needs attach-tensor=true.",
);

// the properties `Inference` handles itself, which every backend's list starts with
#[cfg(feature = "analytics")]
pub const BASE_PROPERTIES: &[PropertySpec] = &[
    MODEL_PROP,
    TENSOR_INPUT_PROP,
    ATTACH_TENSOR_PROP,
    TENSOR_NAME_PROP,
    ONLY_ON_PROP,
];
#[cfg(not(feature = "analytics"))]
pub const BASE_PROPERTIES: &[PropertySpec] = &[MODEL_PROP, TENSOR_INPUT_PROP, ONLY_ON_PROP];

/// The RGBA input pad is the static superset (the model's geometry narrows it at
/// instance time). No source template: the output tensor's shape is the loaded
/// model's, so there is nothing static to advertise.
#[cfg(feature = "launch")]
impl<B: InferenceBackend> g2g_core::PadTemplates for Inference<B> {
    fn pad_templates() -> Vec<g2g_core::PadTemplate> {
        Vec::from([g2g_core::PadTemplate::sink(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lists_every_base_property<B: InferenceBackend>() {
        for base in BASE_PROPERTIES {
            assert!(
                B::PROPERTIES.iter().any(|p| p.name == base.name),
                "{} is missing from the backend's properties",
                base.name
            );
        }
    }

    #[cfg(feature = "ort")]
    #[test]
    fn ort_lists_every_base_property() {
        lists_every_base_property::<crate::ortinfer::OrtBackend>();
    }

    #[cfg(feature = "executorch")]
    #[test]
    fn executorch_lists_every_base_property() {
        lists_every_base_property::<crate::executorchinfer::ExecutorchBackend>();
    }

    /// M1178: `only-on` forwards a frame that does not carry the named blob, with
    /// no session run, which is also why this needs no model file: without the
    /// gate the same frame fails with `NotConfigured` (the last case below). It
    /// is refused outright unless `attach-tensor` keeps the picture on the wire,
    /// since a forwarded frame has to match the negotiated output caps.
    #[cfg(feature = "analytics")]
    fn only_on_forwards_a_frame_without_the_blob<B: InferenceBackend>() {
        use g2g_core::{BlobMeta, Dim, FrameTiming, PropError, PushOutcome, Rate};

        const GATE: &str = "alert";
        const OTHER: &str = "embedding";

        #[derive(Default)]
        struct CollectSink {
            packets: Vec<PipelinePacket>,
        }

        impl OutputSink for CollectSink {
            fn poll_push(
                &mut self,
                _cx: &mut core::task::Context<'_>,
                packet_slot: &mut Option<PipelinePacket>,
            ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
                let packet = packet_slot.take().expect("poll_push without a packet");
                self.packets.push(packet);
                core::task::Poll::Ready(Ok(PushOutcome::Accepted))
            }
        }

        fn frame_carrying(header: &str) -> Frame {
            let mut frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(
                    vec![0u8; 4].into_boxed_slice(),
                )),
                timing: FrameTiming::default(),
                sequence: 7,
                meta: Default::default(),
            };
            let mut blobs = BlobMeta::new();
            blobs.push(header, Vec::from(b"x".as_slice()));
            frame.meta.attach(blobs);
            frame
        }

        let mut element = Inference::<B>::new();
        assert!(element.properties().iter().any(|p| p.name == "only-on"));
        element
            .set_property("only-on", PropValue::Str(String::from(GATE)))
            .unwrap();
        assert_eq!(
            element.get_property("only-on"),
            Some(PropValue::Str(String::from(GATE)))
        );
        assert_eq!(
            element.set_property("only-on", PropValue::Bool(true)),
            Err(PropError::Type)
        );

        // Emitting the tensor as the frame would put a forwarded picture on a
        // pad negotiated for the model's output, so the pair is refused.
        let rgba = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(1),
            height: Dim::Fixed(1),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert_eq!(
            element.configure_pipeline(&rgba).err(),
            Some(G2gError::CapsMismatch),
            "only-on needs attach-tensor=true"
        );
        element
            .set_property("attach-tensor", PropValue::Bool(true))
            .unwrap();

        // A session needs a model file, which this test has none of. The gate
        // sits ahead of the session, so set the flag `process` checks.
        element.configured = true;

        assert!(element
            .only_on
            .forwards_untouched(&frame_carrying(OTHER).meta));
        assert!(!element
            .only_on
            .forwards_untouched(&frame_carrying(GATE).meta));

        let mut sink = CollectSink::default();
        g2g_core::runtime::block_on(
            element.process(PipelinePacket::DataFrame(frame_carrying(OTHER)), &mut sink),
        )
        .unwrap();
        let [PipelinePacket::DataFrame(forwarded)] = &sink.packets[..] else {
            panic!("the gated frame should go downstream on its own");
        };
        assert_eq!(forwarded.sequence, 7, "forwarded untouched, not renumbered");
        assert!(forwarded.meta.get::<BlobMeta>().is_some(), "metadata kept");
        assert_eq!(element.inferred_count(), 0, "no session ran");

        // The frame the gate admits reaches the session, which there is none of.
        assert_eq!(
            g2g_core::runtime::block_on(
                element.process(PipelinePacket::DataFrame(frame_carrying(GATE)), &mut sink)
            ),
            Err(G2gError::NotConfigured)
        );
    }

    #[cfg(all(feature = "ort", feature = "analytics"))]
    #[test]
    fn ort_only_on_forwards_a_frame_without_the_blob() {
        only_on_forwards_a_frame_without_the_blob::<crate::ortinfer::OrtBackend>();
    }

    #[cfg(all(feature = "executorch", feature = "analytics"))]
    #[test]
    fn executorch_only_on_forwards_a_frame_without_the_blob() {
        only_on_forwards_a_frame_without_the_blob::<crate::executorchinfer::ExecutorchBackend>();
    }
}
