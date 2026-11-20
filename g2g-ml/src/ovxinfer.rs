// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_plugins::openvx::{
    NetworkBinaryGraph, OpenVxDataType, OpenVxQuantization, OpenVxTensorInfo,
};

use crate::inference::{Inference, InferenceBackend, LoadedModel, ModelGeometry, BASE_PROPERTIES};
use crate::safetensors::f16_to_f32;
use crate::tensor_bytes::{f32_le_bytes, RGB_CHANNELS};

use g2g_core::{
    G2gError, PropError, PropValue, PropertySpec, TensorDType, TensorLayout, TensorShape,
};

const BATCH: usize = 1;

#[derive(Debug, Default)]
pub struct OvxBackend;

pub type OvxInference = Inference<OvxBackend>;

#[derive(Debug)]
pub struct OvxModel {
    graph: NetworkBinaryGraph,
    geometry: ModelGeometry,
    output: OpenVxTensorInfo,
}

impl InferenceBackend for OvxBackend {
    type Model = OvxModel;

    const PROPERTIES: &'static [PropertySpec] = BASE_PROPERTIES;

    fn load(&self, path: &str) -> Result<OvxModel, G2gError> {
        let graph = NetworkBinaryGraph::load(path)?;
        let inputs: Vec<&OpenVxTensorInfo> = graph.inputs().collect();
        let outputs: Vec<&OpenVxTensorInfo> = graph.outputs().collect();
        let ([input], [output]) = (&inputs[..], &outputs[..]) else {
            return Err(G2gError::CapsMismatch);
        };
        let geometry = model_geometry(input, output)?;
        let output = (*output).clone();
        Ok(OvxModel {
            graph,
            geometry,
            output,
        })
    }

    fn set_property(&mut self, _name: &str, _value: PropValue) -> Result<(), PropError> {
        Err(PropError::Unknown)
    }

    fn get_property(&self, _name: &str) -> Option<PropValue> {
        None
    }
}

fn model_geometry(
    input: &OpenVxTensorInfo,
    output: &OpenVxTensorInfo,
) -> Result<ModelGeometry, G2gError> {
    let (width, height, input_layout) = match input.dims[..] {
        [RGB_CHANNELS, width, height, BATCH] => (width, height, TensorLayout::Nhwc),
        [width, height, RGB_CHANNELS, BATCH] => (width, height, TensorLayout::Nchw),
        _ => return Err(G2gError::CapsMismatch),
    };
    let input_dtype = match input.data_type {
        OpenVxDataType::U8 => TensorDType::U8,
        OpenVxDataType::I8 => TensorDType::I8,
        OpenVxDataType::F32 => TensorDType::F32,
        OpenVxDataType::I16 | OpenVxDataType::F16 => return Err(G2gError::CapsMismatch),
    };
    let output_dims: Vec<u32> = output
        .dims
        .iter()
        .rev()
        .map(|&dim| u32::try_from(dim).map_err(|_| G2gError::CapsMismatch))
        .collect::<Result<_, _>>()?;
    let to_u32 = |dim: usize| u32::try_from(dim).map_err(|_| G2gError::CapsMismatch);
    Ok(ModelGeometry {
        width: to_u32(width)?,
        height: to_u32(height)?,
        input_dtype,
        input_layout,
        output_shape: TensorShape::from_slice(&output_dims).ok_or(G2gError::CapsMismatch)?,
    })
}

fn dequantize(bytes: &[u8], info: &OpenVxTensorInfo) -> Result<Vec<f32>, G2gError> {
    let values = match (info.data_type, info.quantization) {
        (OpenVxDataType::U8, OpenVxQuantization::Affine { scale, zero_point }) => bytes
            .iter()
            .map(|&value| (i32::from(value) - zero_point) as f32 * scale)
            .collect(),
        (OpenVxDataType::I8, OpenVxQuantization::Affine { scale, zero_point }) => bytes
            .iter()
            .map(|&value| (i32::from(value as i8) - zero_point) as f32 * scale)
            .collect(),
        (OpenVxDataType::I8, OpenVxQuantization::DynamicFixedPoint { fraction_bits }) => bytes
            .iter()
            .map(|&value| f32::from(value as i8) / 2f32.powi(i32::from(fraction_bits)))
            .collect(),
        (OpenVxDataType::I16, OpenVxQuantization::DynamicFixedPoint { fraction_bits }) => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f32::from(i16::from_le_bytes(*pair)) / 2f32.powi(i32::from(fraction_bits)))
            .collect(),
        (OpenVxDataType::F16, OpenVxQuantization::None) => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f16_to_f32(u16::from_le_bytes(*pair)))
            .collect(),
        (OpenVxDataType::F32, OpenVxQuantization::None) => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect(),
        _ => return Err(G2gError::CapsMismatch),
    };
    Ok(values)
}

impl OvxModel {
    fn run(&mut self, input: &[u8]) -> Result<(Box<[u8]>, Vec<u32>), G2gError> {
        self.graph.run(&[input])?;
        let values = dequantize(&self.graph.output(0)?, &self.output)?;
        let dims = self.geometry.output_shape.dims().to_vec();
        Ok((f32_le_bytes(&values).into_boxed_slice(), dims))
    }
}

impl LoadedModel for OvxModel {
    fn geometry(&self) -> &ModelGeometry {
        &self.geometry
    }

    fn run_f32(&mut self, chw: Vec<f32>) -> Result<(Box<[u8]>, Vec<u32>), G2gError> {
        if self.geometry.input_dtype != TensorDType::F32 {
            return Err(G2gError::CapsMismatch);
        }
        let values = match self.geometry.input_layout {
            TensorLayout::Nhwc => {
                let plane = chw.len() / RGB_CHANNELS;
                (0..plane)
                    .flat_map(|index| (0..RGB_CHANNELS).map(move |channel| channel * plane + index))
                    .map(|position| chw[position])
                    .collect()
            }
            _ => chw,
        };
        self.run(&f32_le_bytes(&values))
    }

    fn run_int(&mut self, bytes: &[u8]) -> Result<(Box<[u8]>, Vec<u32>), G2gError> {
        self.run(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(
        dims: &[usize],
        data_type: OpenVxDataType,
        quantization: OpenVxQuantization,
    ) -> OpenVxTensorInfo {
        OpenVxTensorInfo {
            dims: dims.to_vec(),
            data_type,
            quantization,
        }
    }

    const CLASSES: usize = 1000;

    #[test]
    fn channels_innermost_is_an_nhwc_model() {
        let input = tensor(
            &[3, 224, 128, 1],
            OpenVxDataType::U8,
            OpenVxQuantization::None,
        );
        let output = tensor(&[CLASSES, 1], OpenVxDataType::F16, OpenVxQuantization::None);
        let geometry = model_geometry(&input, &output).unwrap();
        assert_eq!(
            (
                geometry.width,
                geometry.height,
                geometry.input_layout,
                geometry.input_dtype
            ),
            (224, 128, TensorLayout::Nhwc, TensorDType::U8)
        );
        assert_eq!(geometry.output_shape.dims(), &[1, CLASSES as u32]);
    }

    #[test]
    fn channels_outermost_is_an_nchw_model() {
        let input = tensor(
            &[224, 128, 3, 1],
            OpenVxDataType::F32,
            OpenVxQuantization::None,
        );
        let output = tensor(&[CLASSES, 1], OpenVxDataType::F32, OpenVxQuantization::None);
        let geometry = model_geometry(&input, &output).unwrap();
        assert_eq!(
            (geometry.width, geometry.height, geometry.input_layout),
            (224, 128, TensorLayout::Nchw)
        );
    }

    #[test]
    fn quantized_outputs_dequantize_to_f32() {
        let affine = tensor(
            &[2],
            OpenVxDataType::U8,
            OpenVxQuantization::Affine {
                scale: 0.5,
                zero_point: 10,
            },
        );
        assert_eq!(dequantize(&[10, 14], &affine).unwrap(), [0.0, 2.0]);
        let fixed_point = tensor(
            &[2],
            OpenVxDataType::I8,
            OpenVxQuantization::DynamicFixedPoint { fraction_bits: 2 },
        );
        assert_eq!(
            dequantize(&[4, (-8i8) as u8], &fixed_point).unwrap(),
            [1.0, -2.0]
        );
        // 0x3c00 is 1.0 and 0xc000 is -2.0 in IEEE half precision
        let half = tensor(&[2], OpenVxDataType::F16, OpenVxQuantization::None);
        assert_eq!(
            dequantize(&[0x00, 0x3c, 0x00, 0xc0], &half).unwrap(),
            [1.0, -2.0]
        );
    }
}
