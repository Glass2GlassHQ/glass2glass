// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;
use core::ffi::{c_char, c_void, CStr};
use core::ptr::null_mut;
use std::ffi::CString;

use g2g_core::{g2g_error, G2gError, HardwareError};

const LIBOPENVX: &str = "libOpenVX.so.1";
const VX_SUCCESS: i32 = 0;
const VX_FAILURE: i32 = -1;

// enum values from Khronos' vx_types.h and vx_khr_nn.h as VeriSilicon's SDK headers set them
const VX_KERNEL_PARAMETERS: i32 = 0x80400;
const VX_PARAMETER_DIRECTION: i32 = 0x80501;
const VX_PARAMETER_META_FORMAT: i32 = 0x80505;
const VX_REFERENCE_TYPE: i32 = 0x80001;
const VX_TENSOR_NUMBER_OF_DIMS: i32 = 0x81500;
const VX_TENSOR_DIMS: i32 = 0x81501;
const VX_TENSOR_DATA_TYPE: i32 = 0x81502;
const VX_TENSOR_FIXED_POINT_POSITION: i32 = 0x81503;
const VX_TENSOR_QUANT_FORMAT: i32 = 0x78_1500;
const VX_TENSOR_ZERO_POINT: i32 = 0x78_1501;
const VX_TENSOR_SCALE: i32 = 0x78_1502;
const VX_INPUT: i32 = 0;
const VX_TYPE_TENSOR: i32 = 0x815;
const VX_TYPE_INT8: i32 = 0x2;
const VX_TYPE_UINT8: i32 = 0x3;
const VX_TYPE_INT16: i32 = 0x4;
const VX_TYPE_FLOAT32: i32 = 0xa;
const VX_TYPE_FLOAT16: i32 = 0xf;
const VX_QUANT_NONE: i32 = 0;
const VX_QUANT_DYNAMIC_FIXED_POINT: i32 = 1;
const VX_QUANT_AFFINE_SCALE: i32 = 2;
const VX_READ_ONLY: i32 = 0x1_1001;
const VX_WRITE_ONLY: i32 = 0x1_1002;
const VX_MEMORY_TYPE_HOST: i32 = 0xe001;
// VeriSilicon's import type for a network binary graph read from a file
const VX_VIVANTE_IMPORT_KERNEL_FROM_FILE: &CStr = c"vx_vivante_file";
// the rank limit of VeriSilicon's tensors (VSI_NN_MAX_DIM_NUM)
const MAX_TENSOR_DIMS: usize = 6;

type Handle = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct AffineQuantization {
    scale: f32,
    zero_point: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PerChannelQuantization {
    channel_dim: u32,
    scale_count: u32,
    scales: *mut f32,
    zero_point_count: u32,
    zero_points: *mut i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
union QuantizationParameters {
    fixed_point_position: i8,
    affine: AffineQuantization,
    per_channel: PerChannelQuantization,
}

#[repr(C)]
struct TensorCreateParameters {
    number_of_dims: u32,
    sizes: *const u32,
    data_format: i32,
    quant_format: i32,
    quant_data: QuantizationParameters,
}

// the size the C compiler gives vx_tensor_create_params_t on 64-bit Linux
const _: () = assert!(core::mem::size_of::<TensorCreateParameters>() == 56);

crate::shared_library::shared_library! {
    struct OpenVxObjects = LIBOPENVX;
    when missing return null_mut();
    fn vx_create_context = "vxCreateContext"() -> Handle;
    fn vx_create_graph = "vxCreateGraph"(context: Handle) -> Handle;
    fn vx_import_kernel_from_url = "vxImportKernelFromURL"(
        context: Handle,
        import_type: *const c_char,
        url: *const c_char,
    ) -> Handle;
    fn vx_create_generic_node = "vxCreateGenericNode"(graph: Handle, kernel: Handle) -> Handle;
    fn vx_get_kernel_parameter_by_index = "vxGetKernelParameterByIndex"(kernel: Handle, index: u32) -> Handle;
    fn vx_create_tensor2 = "vxCreateTensor2"(
        context: Handle,
        parameters: *const TensorCreateParameters,
        size: usize,
    ) -> Handle;
}

crate::shared_library::shared_library! {
    struct OpenVxStatuses = LIBOPENVX;
    when missing return VX_FAILURE;
    fn vx_get_status = "vxGetStatus"(reference: Handle) -> i32;
    fn vx_query_kernel = "vxQueryKernel"(kernel: Handle, attribute: i32, value: *mut c_void, size: usize) -> i32;
    fn vx_query_parameter = "vxQueryParameter"(
        parameter: Handle,
        attribute: i32,
        value: *mut c_void,
        size: usize,
    ) -> i32;
    fn vx_query_meta_format_attribute = "vxQueryMetaFormatAttribute"(
        meta: Handle,
        attribute: i32,
        value: *mut c_void,
        size: usize,
    ) -> i32;
    fn vx_set_parameter_by_index = "vxSetParameterByIndex"(node: Handle, index: u32, value: Handle) -> i32;
    fn vx_verify_graph = "vxVerifyGraph"(graph: Handle) -> i32;
    fn vx_process_graph = "vxProcessGraph"(graph: Handle) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn vx_copy_tensor_patch = "vxCopyTensorPatch"(
        tensor: Handle,
        number_of_dims: usize,
        view_start: *const usize,
        view_end: *const usize,
        user_stride: *const usize,
        user_pointer: *mut c_void,
        usage: i32,
        memory_type: i32,
    ) -> i32;
    fn vx_release_parameter = "vxReleaseParameter"(parameter: *mut Handle) -> i32;
    fn vx_release_tensor = "vxReleaseTensor"(tensor: *mut Handle) -> i32;
    fn vx_release_node = "vxReleaseNode"(node: *mut Handle) -> i32;
    fn vx_release_kernel = "vxReleaseKernel"(kernel: *mut Handle) -> i32;
    fn vx_release_graph = "vxReleaseGraph"(graph: *mut Handle) -> i32;
    fn vx_release_context = "vxReleaseContext"(context: *mut Handle) -> i32;
}

pub fn available() -> bool {
    OpenVxObjects::get().is_some() && OpenVxStatuses::get().is_some()
}

fn check(call: &str, status: i32) -> Result<(), G2gError> {
    if status == VX_SUCCESS {
        return Ok(());
    }
    if !available() {
        return Err(G2gError::Hardware(HardwareError::LibraryNotFound(
            LIBOPENVX,
        )));
    }
    g2g_error!(
        g2g_core::log::Target::category("openvx"),
        "{call} returned {status}"
    );
    Err(G2gError::Hardware(HardwareError::OpenVx(status)))
}

fn checked_object(call: &str, handle: Handle) -> Result<Handle, G2gError> {
    if handle.is_null() {
        check(call, VX_FAILURE)?;
    }
    // SAFETY: `handle` is a non-null reference the OpenVX call just returned.
    check(call, unsafe { vx_get_status(handle) })?;
    Ok(handle)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenVxDataType {
    U8,
    I8,
    I16,
    F16,
    F32,
}

impl OpenVxDataType {
    fn from_vx(data_type: i32) -> Option<Self> {
        match data_type {
            VX_TYPE_UINT8 => Some(Self::U8),
            VX_TYPE_INT8 => Some(Self::I8),
            VX_TYPE_INT16 => Some(Self::I16),
            VX_TYPE_FLOAT16 => Some(Self::F16),
            VX_TYPE_FLOAT32 => Some(Self::F32),
            _ => None,
        }
    }

    fn vx(self) -> i32 {
        match self {
            Self::U8 => VX_TYPE_UINT8,
            Self::I8 => VX_TYPE_INT8,
            Self::I16 => VX_TYPE_INT16,
            Self::F16 => VX_TYPE_FLOAT16,
            Self::F32 => VX_TYPE_FLOAT32,
        }
    }

    pub fn element_size(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::I16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpenVxQuantization {
    None,
    DynamicFixedPoint { fraction_bits: i8 },
    Affine { scale: f32, zero_point: i32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpenVxTensorInfo {
    // innermost first, the OpenVX order: an NHWC image is `[C, W, H, N]`
    pub dims: Vec<usize>,
    pub data_type: OpenVxDataType,
    pub quantization: OpenVxQuantization,
}

impl OpenVxTensorInfo {
    pub fn byte_len(&self) -> Option<usize> {
        self.dims
            .iter()
            .try_fold(self.data_type.element_size(), |len, &dim| {
                len.checked_mul(dim)
            })
    }

    fn copy_strides(&self) -> Vec<usize> {
        let mut strides = Vec::with_capacity(self.dims.len());
        let mut stride = self.data_type.element_size();
        for dim in &self.dims {
            strides.push(stride);
            stride = stride.saturating_mul(*dim);
        }
        strides
    }
}

#[derive(Debug)]
struct BoundTensor {
    handle: Handle,
    info: OpenVxTensorInfo,
}

// a VeriSilicon network binary graph (`.nb`) is a model compiled for a Vivante NPU
#[derive(Debug)]
pub struct NetworkBinaryGraph {
    context: Handle,
    graph: Handle,
    kernel: Handle,
    node: Handle,
    inputs: Vec<BoundTensor>,
    outputs: Vec<BoundTensor>,
}

// SAFETY: OpenVX handles have no thread affinity, and the graph is used by one thread at a time
// because every call takes `&mut self` or owns it.
unsafe impl Send for NetworkBinaryGraph {}

fn query_meta_format(
    meta: Handle,
    attribute: i32,
    value: *mut c_void,
    size: usize,
) -> Result<(), G2gError> {
    // SAFETY: `value` points to `size` writable bytes, the type OpenVX documents for `attribute`.
    check("vxQueryMetaFormatAttribute", unsafe {
        vx_query_meta_format_attribute(meta, attribute, value, size)
    })
}

fn tensor_info(meta: Handle) -> Result<OpenVxTensorInfo, G2gError> {
    let mut rank: usize = 0;
    query_meta_format(
        meta,
        VX_TENSOR_NUMBER_OF_DIMS,
        (&raw mut rank).cast(),
        size_of::<usize>(),
    )?;
    if rank == 0 || rank > MAX_TENSOR_DIMS {
        return Err(G2gError::CapsMismatch);
    }
    let mut sizes = [0usize; MAX_TENSOR_DIMS];
    query_meta_format(
        meta,
        VX_TENSOR_DIMS,
        sizes.as_mut_ptr().cast(),
        size_of_val(&sizes),
    )?;
    let mut data_type: i32 = 0;
    query_meta_format(
        meta,
        VX_TENSOR_DATA_TYPE,
        (&raw mut data_type).cast(),
        size_of::<i32>(),
    )?;
    let data_type = OpenVxDataType::from_vx(data_type).ok_or(G2gError::CapsMismatch)?;
    let mut quant_format: i32 = 0;
    query_meta_format(
        meta,
        VX_TENSOR_QUANT_FORMAT,
        (&raw mut quant_format).cast(),
        size_of::<i32>(),
    )?;
    let quantization = match quant_format {
        VX_QUANT_NONE => OpenVxQuantization::None,
        VX_QUANT_DYNAMIC_FIXED_POINT => {
            let mut fraction_bits: i8 = 0;
            query_meta_format(
                meta,
                VX_TENSOR_FIXED_POINT_POSITION,
                (&raw mut fraction_bits).cast(),
                size_of::<i8>(),
            )?;
            OpenVxQuantization::DynamicFixedPoint { fraction_bits }
        }
        VX_QUANT_AFFINE_SCALE => {
            let mut zero_point: i32 = 0;
            query_meta_format(
                meta,
                VX_TENSOR_ZERO_POINT,
                (&raw mut zero_point).cast(),
                size_of::<i32>(),
            )?;
            let mut scale: f32 = 0.0;
            query_meta_format(
                meta,
                VX_TENSOR_SCALE,
                (&raw mut scale).cast(),
                size_of::<f32>(),
            )?;
            OpenVxQuantization::Affine { scale, zero_point }
        }
        _ => return Err(G2gError::CapsMismatch),
    };
    let info = OpenVxTensorInfo {
        dims: sizes[..rank].to_vec(),
        data_type,
        quantization,
    };
    info.byte_len().ok_or(G2gError::CapsMismatch)?;
    Ok(info)
}

// the direction and shape of a kernel parameter, released before it returns
fn parameter_info(kernel: Handle, index: u32) -> Result<(bool, OpenVxTensorInfo), G2gError> {
    // SAFETY: `kernel` is a live kernel and `index` is below its parameter count.
    let mut parameter = checked_object("vxGetKernelParameterByIndex", unsafe {
        vx_get_kernel_parameter_by_index(kernel, index)
    })?;
    let described = (|| {
        let mut direction: i32 = 0;
        // SAFETY: `direction` is the vx_enum OpenVX writes for this attribute.
        check("vxQueryParameter", unsafe {
            vx_query_parameter(
                parameter,
                VX_PARAMETER_DIRECTION,
                (&raw mut direction).cast(),
                size_of::<i32>(),
            )
        })?;
        let mut meta: Handle = null_mut();
        // SAFETY: `meta` is the vx_meta_format handle OpenVX writes for this attribute.
        check("vxQueryParameter", unsafe {
            vx_query_parameter(
                parameter,
                VX_PARAMETER_META_FORMAT,
                (&raw mut meta).cast(),
                size_of::<Handle>(),
            )
        })?;
        let mut reference_type: i32 = 0;
        query_meta_format(
            meta,
            VX_REFERENCE_TYPE,
            (&raw mut reference_type).cast(),
            size_of::<i32>(),
        )?;
        if reference_type != VX_TYPE_TENSOR {
            return Err(G2gError::CapsMismatch);
        }
        Ok((direction == VX_INPUT, tensor_info(meta)?))
    })();
    // SAFETY: `parameter` is the live parameter fetched above, released once.
    unsafe { vx_release_parameter(&raw mut parameter) };
    described
}

fn create_tensor(context: Handle, info: &OpenVxTensorInfo) -> Result<Handle, G2gError> {
    let sizes: Vec<u32> = info
        .dims
        .iter()
        .map(|&dim| u32::try_from(dim).map_err(|_| G2gError::CapsMismatch))
        .collect::<Result<_, _>>()?;
    let (quant_format, quant_data) = match info.quantization {
        OpenVxQuantization::None => (
            VX_QUANT_NONE,
            QuantizationParameters {
                fixed_point_position: 0,
            },
        ),
        OpenVxQuantization::DynamicFixedPoint { fraction_bits } => (
            VX_QUANT_DYNAMIC_FIXED_POINT,
            QuantizationParameters {
                fixed_point_position: fraction_bits,
            },
        ),
        OpenVxQuantization::Affine { scale, zero_point } => (
            VX_QUANT_AFFINE_SCALE,
            QuantizationParameters {
                affine: AffineQuantization { scale, zero_point },
            },
        ),
    };
    let parameters = TensorCreateParameters {
        number_of_dims: sizes.len() as u32,
        sizes: sizes.as_ptr(),
        data_format: info.data_type.vx(),
        quant_format,
        quant_data,
    };
    // SAFETY: `parameters` and the `sizes` it points to outlive the call, which copies them.
    checked_object("vxCreateTensor2", unsafe {
        vx_create_tensor2(
            context,
            &raw const parameters,
            size_of::<TensorCreateParameters>(),
        )
    })
}

fn copy_tensor(tensor: &BoundTensor, bytes: *mut c_void, usage: i32) -> Result<(), G2gError> {
    let start = [0usize; MAX_TENSOR_DIMS];
    let strides = tensor.info.copy_strides();
    // SAFETY: `bytes` covers the whole tensor (callers check `byte_len`), and the
    // start, end and stride arrays each hold one entry per tensor dimension.
    check("vxCopyTensorPatch", unsafe {
        vx_copy_tensor_patch(
            tensor.handle,
            tensor.info.dims.len(),
            start.as_ptr(),
            tensor.info.dims.as_ptr(),
            strides.as_ptr(),
            bytes,
            usage,
            VX_MEMORY_TYPE_HOST,
        )
    })
}

impl NetworkBinaryGraph {
    pub fn load(path: &str) -> Result<Self, G2gError> {
        let url = CString::new(path).map_err(|_| G2gError::CapsMismatch)?;
        let mut loaded = Self {
            // SAFETY: vxCreateContext takes no arguments.
            context: checked_object("vxCreateContext", unsafe { vx_create_context() })?,
            graph: null_mut(),
            kernel: null_mut(),
            node: null_mut(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        };
        // SAFETY: the context is live.
        loaded.graph = checked_object("vxCreateGraph", unsafe { vx_create_graph(loaded.context) })?;
        // SAFETY: both strings are NUL-terminated and outlive the call.
        loaded.kernel = checked_object("vxImportKernelFromURL", unsafe {
            vx_import_kernel_from_url(
                loaded.context,
                VX_VIVANTE_IMPORT_KERNEL_FROM_FILE.as_ptr(),
                url.as_ptr(),
            )
        })?;
        // SAFETY: the graph and kernel are live.
        loaded.node = checked_object("vxCreateGenericNode", unsafe {
            vx_create_generic_node(loaded.graph, loaded.kernel)
        })?;
        let mut parameter_count: u32 = 0;
        // SAFETY: `parameter_count` is the vx_uint32 OpenVX writes for this attribute.
        check("vxQueryKernel", unsafe {
            vx_query_kernel(
                loaded.kernel,
                VX_KERNEL_PARAMETERS,
                (&raw mut parameter_count).cast(),
                size_of::<u32>(),
            )
        })?;
        for index in 0..parameter_count {
            let (is_input, info) = parameter_info(loaded.kernel, index)?;
            let handle = create_tensor(loaded.context, &info)?;
            let tensor = BoundTensor { handle, info };
            let bound = if is_input {
                &mut loaded.inputs
            } else {
                &mut loaded.outputs
            };
            bound.push(tensor);
            // SAFETY: the node and tensor are live, and `index` is a parameter of the node's kernel.
            check("vxSetParameterByIndex", unsafe {
                vx_set_parameter_by_index(loaded.node, index, handle)
            })?;
        }
        // SAFETY: the graph is live and every node parameter is bound.
        check("vxVerifyGraph", unsafe { vx_verify_graph(loaded.graph) })?;
        Ok(loaded)
    }

    pub fn inputs(&self) -> impl Iterator<Item = &OpenVxTensorInfo> {
        self.inputs.iter().map(|tensor| &tensor.info)
    }

    pub fn outputs(&self) -> impl Iterator<Item = &OpenVxTensorInfo> {
        self.outputs.iter().map(|tensor| &tensor.info)
    }

    pub fn run(&mut self, inputs: &[&[u8]]) -> Result<(), G2gError> {
        if inputs.len() != self.inputs.len() {
            return Err(G2gError::CapsMismatch);
        }
        for (tensor, bytes) in self.inputs.iter().zip(inputs) {
            if Some(bytes.len()) != tensor.info.byte_len() {
                return Err(G2gError::CapsMismatch);
            }
            // vxCopyTensorPatch only reads the host buffer when writing to the tensor
            copy_tensor(tensor, bytes.as_ptr().cast_mut().cast(), VX_WRITE_ONLY)?;
        }
        // SAFETY: the graph is live and verified.
        check("vxProcessGraph", unsafe { vx_process_graph(self.graph) })
    }

    pub fn output(&self, index: usize) -> Result<Vec<u8>, G2gError> {
        let tensor = self.outputs.get(index).ok_or(G2gError::CapsMismatch)?;
        let mut bytes = alloc::vec![0u8; tensor.info.byte_len().ok_or(G2gError::CapsMismatch)?];
        copy_tensor(tensor, bytes.as_mut_ptr().cast(), VX_READ_ONLY)?;
        Ok(bytes)
    }
}

impl Drop for NetworkBinaryGraph {
    fn drop(&mut self) {
        for tensor in self.inputs.iter_mut().chain(self.outputs.iter_mut()) {
            // SAFETY: each tensor was created by `load` and is released once.
            unsafe { vx_release_tensor(&raw mut tensor.handle) };
        }
        // SAFETY: each handle is null or a live object created by `load`, released once, children first.
        unsafe {
            if !self.node.is_null() {
                vx_release_node(&raw mut self.node);
            }
            if !self.kernel.is_null() {
                vx_release_kernel(&raw mut self.kernel);
            }
            if !self.graph.is_null() {
                vx_release_graph(&raw mut self.graph);
            }
            if !self.context.is_null() {
                vx_release_context(&raw mut self.context);
            }
        }
    }
}
