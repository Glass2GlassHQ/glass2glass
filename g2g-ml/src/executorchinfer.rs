//! ExecuTorch inference element (`executorch` backend, design/ml.md).
//!
//! `ExecutorchInference` is the shared [`Inference`] element on
//! [`ExecutorchBackend`]: it runs a PyTorch program through the ExecuTorch
//! runtime on its portable CPU kernels, with the caps, the tensor-input mode and
//! the properties of `ortinfer` (minus `execution-provider`).
//!
//! Model contract (checked when the model loads, fails loud): one `forward`
//! method with exactly one `[1, 3, H, W]` f32 / u8 / i8 tensor input at static
//! `H` / `W`, and exactly one f32 tensor output with static dims. A `.pte` file
//! comes from `torch.export` then `to_edge_transform_and_lower().to_executorch()`.
//!
//! The `executorch` crate links the ExecuTorch 1.5.1 C++ static libraries rather
//! than building them: `tools/executorch-build.sh` builds them, and the build
//! finds them through `EXECUTORCH_RS_EXECUTORCH_LIB_DIR`.

use executorch::evalue::{IntoEValue, Tag};
use executorch::module::Module;
use executorch::ndarray::Array;
use executorch::tensor::{Scalar, ScalarType, TensorAny, TensorPtr};

use crate::inference::{Inference, InferenceBackend, LoadedModel, ModelGeometry, BASE_PROPERTIES};
use crate::model_dims::{input_geometry, static_output_dims};
use crate::tensor_bytes::f32_le_bytes;

use g2g_core::{G2gError, HardwareError, PropError, PropValue, PropertySpec, TensorDType};

const FORWARD_METHOD: &str = "forward";
const LOG_CATEGORY: &str = "ExecutorchInference";
const INPUT_BATCH: usize = 1;
const INPUT_CHANNELS: usize = 3;

#[derive(Debug, Default)]
pub struct ExecutorchBackend;

/// # Example
///
/// ```no_run
/// use g2g_ml::executorchinfer::ExecutorchInference;
///
/// // gst-launch style: executorchinfer model=model.pte
/// let element = ExecutorchInference::from_file("model.pte").unwrap();
/// ```
pub type ExecutorchInference = Inference<ExecutorchBackend>;

pub struct ExecutorchModel {
    module: Module<'static>,
    geometry: ModelGeometry,
}

// `Module` has no `Debug` impl
impl core::fmt::Debug for ExecutorchModel {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ExecutorchModel")
            .field("geometry", &self.geometry)
            .finish_non_exhaustive()
    }
}

impl InferenceBackend for ExecutorchBackend {
    type Model = ExecutorchModel;

    const PROPERTIES: &'static [PropertySpec] = BASE_PROPERTIES;

    fn load(&self, path: &str) -> Result<ExecutorchModel, G2gError> {
        let mut module = Module::new(path);
        // `Module::new` is lazy, this is what reads and verifies the program
        let geometry = model_geometry(&mut module)?;
        Ok(ExecutorchModel { module, geometry })
    }

    fn set_property(&mut self, _name: &str, _value: PropValue) -> Result<(), PropError> {
        Err(PropError::Unknown)
    }

    fn get_property(&self, _name: &str) -> Option<PropValue> {
        None
    }
}

fn model_geometry(module: &mut Module<'static>) -> Result<ModelGeometry, G2gError> {
    let meta = module.method_meta(FORWARD_METHOD).map_err(executorch_err)?;
    if meta.num_inputs() != 1 || meta.num_outputs() != 1 {
        return Err(G2gError::CapsMismatch);
    }
    if meta.input_tag(0).map_err(executorch_err)? != Tag::Tensor {
        return Err(G2gError::CapsMismatch);
    }
    let input = meta.input_tensor_meta(0).map_err(executorch_err)?;
    let input_dims: Vec<i64> = input.sizes().iter().map(|&d| i64::from(d)).collect();
    let (width, height) = input_geometry(&input_dims)?;
    let input_dtype = match input.scalar_type() {
        ScalarType::Float => TensorDType::F32,
        ScalarType::Byte => TensorDType::U8,
        ScalarType::Char => TensorDType::I8,
        _ => return Err(G2gError::CapsMismatch),
    };
    let output = meta.output_tensor_meta(0).map_err(executorch_err)?;
    if output.scalar_type() != ScalarType::Float {
        return Err(G2gError::CapsMismatch);
    }
    let output_dims: Vec<i64> = output.sizes().iter().map(|&d| i64::from(d)).collect();
    Ok(ModelGeometry {
        width,
        height,
        input_dtype,
        output_shape: static_output_dims(&output_dims)?,
    })
}

impl ExecutorchModel {
    fn run<S: Scalar>(&mut self, values: Vec<S>) -> Result<(Box<[u8]>, Vec<u32>), G2gError> {
        let shape = (
            INPUT_BATCH,
            INPUT_CHANNELS,
            self.geometry.height as usize,
            self.geometry.width as usize,
        );
        let array = Array::from_shape_vec(shape, values).map_err(|_| G2gError::CapsMismatch)?;
        let input = TensorPtr::from_array(array).map_err(executorch_err)?;
        let outputs = self
            .module
            .forward(&[input.into_evalue()])
            .map_err(executorch_err)?;
        let [output] = &outputs[..] else {
            return Err(G2gError::CapsMismatch);
        };
        let output = TensorAny::try_from(output)
            .map_err(|_| G2gError::CapsMismatch)?
            .try_into_typed::<f32>()
            .map_err(|_| G2gError::CapsMismatch)?;
        let dims: Vec<u32> = output.sizes().iter().map(|&d| d.max(0) as u32).collect();
        let values: Vec<f32> = output.as_array_dyn().iter().copied().collect();
        Ok((f32_le_bytes(&values).into_boxed_slice(), dims))
    }
}

impl LoadedModel for ExecutorchModel {
    fn geometry(&self) -> &ModelGeometry {
        &self.geometry
    }

    fn run_f32(&mut self, chw: Vec<f32>) -> Result<(Box<[u8]>, Vec<u32>), G2gError> {
        self.run(chw)
    }

    fn run_int(&mut self, bytes: &[u8]) -> Result<(Box<[u8]>, Vec<u32>), G2gError> {
        match self.geometry.input_dtype {
            TensorDType::U8 => self.run(bytes.to_vec()),
            TensorDType::I8 => self.run(bytes.iter().map(|&b| b as i8).collect::<Vec<i8>>()),
            _ => Err(G2gError::CapsMismatch),
        }
    }
}

// a launch line only reports a bad `model` value, so name the runtime's error here
fn executorch_err(error: executorch::Error) -> G2gError {
    g2g_core::g2g_error!(
        g2g_core::log::Target::category(LOG_CATEGORY),
        "ExecuTorch call failed: {error}"
    );
    G2gError::Hardware(HardwareError::Other)
}
