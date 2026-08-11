//! GPU zero-copy handoff: NV12 CUDA planes described to Python through
//! `__cuda_array_interface__` v3 (M984, `python` feature).
//!
//! A [`MemoryDomain::Cuda`](g2g_core::MemoryDomain::Cuda) frame carries device
//! pointers, not bytes, so the buffer protocol the System path uses cannot
//! describe it. Each semi-planar plane becomes a [`CudaPlane`] whose
//! `__cuda_array_interface__` dict names the device pointer, the plane shape and
//! the producer's row pitch as strides, so `cupy.asarray(plane)` aliases the
//! decoder's surface with no PCIe round-trip. `torch.as_tensor` refuses a
//! read-only CAI export (see [`PLANE_READ_ONLY`]), so torch's path here is
//! DLPack.
//!
//! The same plane also exports DLPack (M986): `__dlpack__` hands out a capsule
//! over the same device pointer, for the frameworks that prefer it (torch's
//! `from_dlpack`, `cupy.from_dlpack`). DLPack carries a device and a stream in
//! the protocol, which CAI does not.
//!
//! CUDA-context caveat: neither protocol carries a CUDA context. The pointers are
//! valid only in the context the producer decoded into, exposed as the plane's
//! `cuda_context` property; a consumer must already be in that context or push
//! it (`cuCtxPushCurrent`) before touching the memory. cupy and torch use the
//! device's primary context, so a producer that decoded into its own driver
//! context needs the Python side to push `cuda_context` itself.

use core::ffi::{c_void, CStr};
use core::ptr::addr_of_mut;

use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use std::sync::Arc;

use g2g_core::memory::{CudaKeepAlive, OwnedCudaBuffer};
use g2g_core::RawVideoFormat;

/// `__cuda_array_interface__` version this describes.
const CAI_VERSION: u32 = 3;

/// CAI's `data` read-only flag. The device memory belongs to the producer (a
/// decoder surface), and a teed frame hands the same pointers to every branch
/// under a read-only-sharing guarantee; unlike the System path there is no
/// copy-on-write to fall back on here, so a plane is exported read-only. cupy
/// treats the flag as advisory and aliases anyway; torch's CAI importer refuses
/// a read-only export outright, so a torch consumer takes the DLPack path.
const PLANE_READ_ONLY: bool = true;

/// DLPack, as the `dlpack.h` the installed consumers bundle defines it (v1.0;
/// cupy ships it under `cupy/_core/include/cupy/_dlpack/`). The struct layouts
/// are asserted by size below.
const DLPACK_MAJOR: u32 = 1;
const DLPACK_MINOR: u32 = 0;
/// `kDLCUDA`.
const DLPACK_DEVICE_CUDA: i32 = 2;
/// `kDLUInt`: both semi-planar formats are unsigned samples.
const DLPACK_CODE_UINT: u8 = 1;
/// `DLPACK_FLAG_BITMASK_READ_ONLY`, the versioned struct's spelling of the
/// read-only export [`PLANE_READ_ONLY`] declares on the CAI side.
const DLPACK_FLAG_READ_ONLY: u64 = 1;
/// Capsule names the protocol fixes. A consumer that takes ownership renames the
/// capsule to `used_dltensor*`, which is how [`drop_unconsumed_legacy`] knows
/// whether the tensor is still ours to free.
const CAPSULE_LEGACY: &CStr = c"dltensor";
const CAPSULE_VERSIONED: &CStr = c"dltensor_versioned";
/// Rank of the widest plane (interleaved chroma), so shape / stride storage is
/// inline rather than a second allocation.
const MAX_RANK: usize = 3;

#[repr(C)]
#[derive(Debug)]
struct DlPackVersion {
    major: u32,
    minor: u32,
}

#[repr(C)]
#[derive(Debug)]
struct DlDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
#[derive(Debug)]
struct DlDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
#[derive(Debug)]
struct DlTensor {
    data: *mut c_void,
    device: DlDevice,
    ndim: i32,
    dtype: DlDataType,
    /// Both point into the owning [`Exported`], fixed up once it is boxed.
    shape: *mut i64,
    /// In *elements*, not bytes (unlike CAI).
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
#[derive(Debug)]
struct DlManagedTensor {
    dl_tensor: DlTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DlManagedTensor)>,
}

#[repr(C)]
#[derive(Debug)]
struct DlManagedTensorVersioned {
    version: DlPackVersion,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DlManagedTensorVersioned)>,
    flags: u64,
    dl_tensor: DlTensor,
}

const _: () = assert!(core::mem::size_of::<DlTensor>() == 48);
const _: () = assert!(core::mem::size_of::<DlManagedTensor>() == 64);
const _: () = assert!(core::mem::size_of::<DlManagedTensorVersioned>() == 80);

/// Backing store for one exported tensor: the managed tensor the capsule points
/// at, the shape / stride arrays it points into, and a handle on the plane, so a
/// consumer that keeps the tensor keeps the plane alive too and the host's
/// post-call refcount check catches it.
#[derive(Debug)]
struct Exported<T> {
    managed: T,
    shape: [i64; MAX_RANK],
    strides: [i64; MAX_RANK],
    #[allow(dead_code, reason = "held only to keep the exporting plane alive")]
    plane: Py<CudaPlane>,
}

/// One plane of a GPU-resident frame, described to Python by CAI v3 and DLPack.
/// Holds only the device pointer and the plane's layout, so it is inert on the
/// Rust side: nothing here touches CUDA. Valid for the duration of one
/// `g2g_process_cuda` call (see the retention guard in [`crate::host`]).
#[pyclass(frozen, module = "g2g")]
#[derive(Debug)]
pub(crate) struct CudaPlane {
    device_ptr: u64,
    shape: Vec<usize>,
    /// Byte strides, outermost first: the producer's row pitch, then the
    /// element stride(s) within a row.
    strides: Vec<usize>,
    /// Bytes per sample: 1 for NV12, 2 for P010.
    sample_bytes: usize,
    context: u64,
    /// Ordinal of the CUDA device the pointer lives on, as the producer
    /// reported it. DLPack carries it; CAI has no field for it.
    device_ordinal: i32,
}

impl CudaPlane {
    /// The CAI type string for this plane's samples: unsigned integers, and
    /// little-endian once they are wider than a byte.
    fn typestr(&self) -> &'static str {
        if self.sample_bytes == 1 {
            "|u1"
        } else {
            "<u2"
        }
    }

    /// This plane as a DLPack tensor description, with `shape` / `strides` left
    /// null for the caller to point at its own storage. `None` when a row pitch
    /// is not a whole number of samples, since DLPack strides count elements and
    /// there is no honest way to express that.
    fn dl_tensor(&self) -> Option<(DlTensor, [i64; MAX_RANK], [i64; MAX_RANK])> {
        let mut shape = [0i64; MAX_RANK];
        let mut strides = [0i64; MAX_RANK];
        for (slot, value) in shape.iter_mut().zip(&self.shape) {
            *slot = *value as i64;
        }
        for (slot, bytes) in strides.iter_mut().zip(&self.strides) {
            if bytes % self.sample_bytes != 0 {
                return None;
            }
            *slot = (bytes / self.sample_bytes) as i64;
        }
        let tensor = DlTensor {
            data: self.device_ptr as *mut c_void,
            device: DlDevice {
                device_type: DLPACK_DEVICE_CUDA,
                device_id: self.device_ordinal,
            },
            ndim: self.shape.len() as i32,
            dtype: DlDataType {
                code: DLPACK_CODE_UINT,
                bits: (self.sample_bytes * 8) as u8,
                lanes: 1,
            },
            shape: core::ptr::null_mut(),
            strides: core::ptr::null_mut(),
            byte_offset: 0,
        };
        Some((tensor, shape, strides))
    }
}

/// The tensor's deleter, called by the consumer when it is done with the tensor
/// (or by our capsule destructor when nobody took it): reclaim the box
/// `manager_ctx` points at, which drops the plane handle with it.
///
/// # Safety
/// `managed` must be a tensor this module exported and not yet freed.
unsafe extern "C" fn drop_legacy(managed: *mut DlManagedTensor) {
    // SAFETY: the caller guarantees `managed` is one of our live exports, whose
    // `manager_ctx` is the `Box::into_raw` pointer of its own backing store.
    unsafe {
        let context = (*managed).manager_ctx as *mut Exported<DlManagedTensor>;
        drop(Box::from_raw(context));
    }
}

/// # Safety
/// As [`drop_legacy`], for the versioned struct.
unsafe extern "C" fn drop_versioned(managed: *mut DlManagedTensorVersioned) {
    // SAFETY: as `drop_legacy`.
    unsafe {
        let context = (*managed).manager_ctx as *mut Exported<DlManagedTensorVersioned>;
        drop(Box::from_raw(context));
    }
}

/// Capsule destructor: free the tensor only while the capsule still carries the
/// unconsumed name. A consumer that took ownership renamed it and calls the
/// tensor's own deleter when it is done, so touching it here would double-free.
///
/// # Safety
/// Called by CPython with the capsule being destroyed.
unsafe extern "C" fn drop_unconsumed_legacy(capsule: *mut ffi::PyObject) {
    // SAFETY: CPython hands us the capsule it is destroying; `PyCapsule_GetPointer`
    // under the matching name yields the pointer we stored, or null (with an error
    // set, which is cleared by the caller's destructor protocol).
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, CAPSULE_LEGACY.as_ptr()) == 1 {
            let managed =
                ffi::PyCapsule_GetPointer(capsule, CAPSULE_LEGACY.as_ptr()) as *mut DlManagedTensor;
            if let Some(deleter) = managed.as_ref().and_then(|m| m.deleter) {
                deleter(managed);
            }
        }
    }
}

/// # Safety
/// As [`drop_unconsumed_legacy`], for the versioned capsule name.
unsafe extern "C" fn drop_unconsumed_versioned(capsule: *mut ffi::PyObject) {
    // SAFETY: as `drop_unconsumed_legacy`.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, CAPSULE_VERSIONED.as_ptr()) == 1 {
            let managed = ffi::PyCapsule_GetPointer(capsule, CAPSULE_VERSIONED.as_ptr())
                as *mut DlManagedTensorVersioned;
            if let Some(deleter) = managed.as_ref().and_then(|m| m.deleter) {
                deleter(managed);
            }
        }
    }
}

/// Wrap a boxed export in its capsule, or reclaim the box if CPython refuses.
///
/// # Safety
/// `managed` must be the address of `raw`'s managed tensor, whose shape, strides
/// and `manager_ctx` are already filled in, and `name` / `destructor` must be the
/// matching pair for its struct.
unsafe fn capsule<'py, T>(
    py: Python<'py>,
    raw: *mut Exported<T>,
    managed: *mut c_void,
    name: &CStr,
    destructor: unsafe extern "C" fn(*mut ffi::PyObject),
) -> PyResult<Bound<'py, PyAny>> {
    // SAFETY: `managed` points into the live box `raw`, and the capsule keeps it
    // reachable until a deleter reclaims it.
    let object = unsafe { ffi::PyCapsule_New(managed, name.as_ptr(), Some(destructor)) };
    // SAFETY: `object` is a new reference, or null with an error set.
    match unsafe { Bound::from_owned_ptr_or_opt(py, object) } {
        Some(capsule) => Ok(capsule),
        // Nothing took ownership of the box, so reclaim it here.
        None => {
            // SAFETY: the capsule was not created, so `raw` is still the only
            // pointer to the box.
            drop(unsafe { Box::from_raw(raw) });
            Err(PyErr::fetch(py))
        }
    }
}

/// Export as the pre-1.0 `DLManagedTensor` capsule, for a consumer that asked for
/// no version (or a version below 1.0).
fn export_legacy<'py>(
    py: Python<'py>,
    plane: Py<CudaPlane>,
    tensor: DlTensor,
    shape: [i64; MAX_RANK],
    strides: [i64; MAX_RANK],
) -> PyResult<Bound<'py, PyAny>> {
    let raw = Box::into_raw(Box::new(Exported {
        managed: DlManagedTensor {
            dl_tensor: tensor,
            manager_ctx: core::ptr::null_mut(),
            deleter: Some(drop_legacy),
        },
        shape,
        strides,
        plane,
    }));
    // SAFETY: `raw` is a live, uniquely owned box, so pointers into it stay valid
    // until its deleter reclaims it; the tensor's array pointers must point at the
    // box's own storage and `manager_ctx` back at the box itself.
    unsafe {
        (*raw).managed.dl_tensor.shape = addr_of_mut!((*raw).shape) as *mut i64;
        (*raw).managed.dl_tensor.strides = addr_of_mut!((*raw).strides) as *mut i64;
        (*raw).managed.manager_ctx = raw as *mut c_void;
        let managed = addr_of_mut!((*raw).managed) as *mut c_void;
        capsule(py, raw, managed, CAPSULE_LEGACY, drop_unconsumed_legacy)
    }
}

/// Export as the current `DLManagedTensorVersioned` capsule, which also carries
/// the read-only flag.
fn export_versioned<'py>(
    py: Python<'py>,
    plane: Py<CudaPlane>,
    tensor: DlTensor,
    shape: [i64; MAX_RANK],
    strides: [i64; MAX_RANK],
) -> PyResult<Bound<'py, PyAny>> {
    let raw = Box::into_raw(Box::new(Exported {
        managed: DlManagedTensorVersioned {
            version: DlPackVersion {
                major: DLPACK_MAJOR,
                minor: DLPACK_MINOR,
            },
            manager_ctx: core::ptr::null_mut(),
            deleter: Some(drop_versioned),
            flags: DLPACK_FLAG_READ_ONLY,
            dl_tensor: tensor,
        },
        shape,
        strides,
        plane,
    }));
    // SAFETY: as `export_legacy`.
    unsafe {
        (*raw).managed.dl_tensor.shape = addr_of_mut!((*raw).shape) as *mut i64;
        (*raw).managed.dl_tensor.strides = addr_of_mut!((*raw).strides) as *mut i64;
        (*raw).managed.manager_ctx = raw as *mut c_void;
        let managed = addr_of_mut!((*raw).managed) as *mut c_void;
        capsule(
            py,
            raw,
            managed,
            CAPSULE_VERSIONED,
            drop_unconsumed_versioned,
        )
    }
}

#[pymethods]
impl CudaPlane {
    /// The CAI v3 description of this plane: `shape`, `typestr`, `data`
    /// (device pointer, read-only flag), byte `strides`, `version`, and a
    /// `stream` of `None` (the g2g CUDA domain carries no stream; a producer
    /// hands the frame over once the decode into it has completed, so a
    /// consumer needs no cross-stream synchronization).
    #[getter(__cuda_array_interface__)]
    fn cuda_array_interface<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("shape", PyTuple::new(py, &self.shape)?)?;
        dict.set_item("typestr", self.typestr())?;
        dict.set_item("data", (self.device_ptr, PLANE_READ_ONLY))?;
        dict.set_item("strides", PyTuple::new(py, &self.strides)?)?;
        dict.set_item("version", CAI_VERSION)?;
        dict.set_item("stream", py.None())?;
        Ok(dict)
    }

    /// The `CUcontext` (as an int) the device pointer is valid in, so a consumer
    /// outside the producing context can make it current before use. Zero when
    /// the producer left it unset.
    #[getter]
    fn cuda_context(&self) -> u64 {
        self.context
    }

    /// This plane as a DLPack capsule over the same device pointer, for
    /// `torch.from_dlpack` / `cupy.from_dlpack`.
    ///
    /// `max_version` picks the struct: a consumer that asks for 1.0 or newer gets
    /// the versioned tensor (which carries the read-only flag), one that asks for
    /// nothing gets the pre-1.0 one. `stream` is ignored: the g2g CUDA domain
    /// carries no stream and a producer hands a frame over complete, so there is
    /// nothing for the consumer's stream to wait on. A `dl_device` other than this
    /// plane's, or `copy=True`, is refused rather than silently ignored, since
    /// this export cannot move or duplicate the producer's memory.
    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        slf: &Bound<'py, Self>,
        stream: Option<Bound<'py, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = stream;
        if copy == Some(true) {
            return Err(PyBufferError::new_err(
                "g2g.CudaPlane cannot copy the producer's device memory",
            ));
        }
        let plane = slf.get();
        if let Some(device) = dl_device {
            if device != (DLPACK_DEVICE_CUDA, plane.device_ordinal) {
                let ordinal = plane.device_ordinal;
                return Err(PyBufferError::new_err(format!(
                    "g2g.CudaPlane lives on CUDA device {ordinal}, not {device:?}"
                )));
            }
        }
        let Some((tensor, shape, strides)) = plane.dl_tensor() else {
            return Err(PyBufferError::new_err(
                "row pitch is not a whole number of samples, so DLPack element strides cannot describe this plane",
            ));
        };
        let py = slf.py();
        let handle = slf.clone().unbind();
        match max_version {
            Some((major, _)) if major >= DLPACK_MAJOR => {
                export_versioned(py, handle, tensor, shape, strides)
            }
            _ => export_legacy(py, handle, tensor, shape, strides),
        }
    }

    /// The DLPack device this plane's memory lives on: `(kDLCUDA, ordinal)`.
    fn __dlpack_device__(&self) -> (i32, i32) {
        (DLPACK_DEVICE_CUDA, self.device_ordinal)
    }
}

/// Describe a semi-planar GPU frame as its two planes: luma `(height, width)`
/// and interleaved chroma `(height/2, width/2, 2)`, both strided by the
/// producer's row pitch. `None` for a format the CUDA domain does not carry
/// (only NV12 and its 10-bit P010 sibling are semi-planar).
pub(crate) fn nv12_planes(
    fmt: RawVideoFormat,
    buf: &OwnedCudaBuffer,
) -> Option<(CudaPlane, CudaPlane)> {
    if !matches!(fmt, RawVideoFormat::Nv12 | RawVideoFormat::P010) {
        return None;
    }
    let sample = fmt.bytes_per_sample();
    let (width, height) = (buf.width as usize, buf.height as usize);
    let luma = CudaPlane {
        device_ptr: buf.luma_ptr,
        shape: vec![height, width],
        strides: vec![buf.luma_pitch as usize, sample],
        sample_bytes: sample,
        context: buf.context,
        device_ordinal: buf.device_ordinal,
    };
    let chroma = CudaPlane {
        device_ptr: buf.chroma_ptr,
        shape: vec![height.div_ceil(2), width.div_ceil(2), 2],
        strides: vec![buf.chroma_pitch as usize, 2 * sample, sample],
        sample_bytes: sample,
        context: buf.context,
        device_ordinal: buf.device_ordinal,
    };
    Some((luma, chroma))
}

/// Device memory a hosted Python source produced, kept alive by holding the
/// objects that own it (a cupy / torch array per plane) for as long as the frame
/// references their pointers. Dropped off the worker thread is fine: pyo3 defers
/// the decref until a thread attaches again.
#[derive(Debug)]
struct PyOwnedSurface {
    #[allow(dead_code, reason = "held only to keep the device memory alive")]
    planes: [Py<PyAny>; 2],
}

impl CudaKeepAlive for PyOwnedSurface {}

/// One plane of a surface a Python source handed back, as read from its CAI dict.
struct ProducedPlane {
    device_ptr: u64,
    pitch: u32,
}

/// Read a plane a Python source produced: it must export CAI, at the shape and
/// sample size the negotiated caps call for, packed within each row (a g2g CUDA
/// frame is semi-planar NV12 / P010, which is what every downstream consumer
/// reads), with only the row pitch free. Anything else is refused rather than
/// handed downstream as a device pointer nobody can read correctly.
fn read_produced_plane(
    object: &Bound<'_, PyAny>,
    expected_shape: &[usize],
    sample_bytes: usize,
) -> PyResult<ProducedPlane> {
    let cai = object.getattr("__cuda_array_interface__")?;
    let shape: Vec<usize> = cai.get_item("shape")?.extract()?;
    if shape != expected_shape {
        return Err(PyBufferError::new_err(format!(
            "produced plane has shape {shape:?}, expected {expected_shape:?}"
        )));
    }
    let typestr: String = cai.get_item("typestr")?.extract()?;
    let expected_typestr = if sample_bytes == 1 { "|u1" } else { "<u2" };
    if typestr != expected_typestr {
        return Err(PyBufferError::new_err(format!(
            "produced plane has samples of type {typestr}, expected {expected_typestr}"
        )));
    }
    let (device_ptr, _read_only): (u64, bool) = cai.get_item("data")?.extract()?;
    if device_ptr == 0 {
        return Err(PyBufferError::new_err("produced plane has a null pointer"));
    }

    // Rows are `shape[1..]` samples wide; without strides the plane is packed, so
    // that width is also its pitch.
    let packed_row: usize = shape[1..].iter().product::<usize>() * sample_bytes;
    let reported: Option<Vec<usize>> = cai.get_item("strides")?.extract()?;
    let pitch = match reported {
        None => packed_row,
        Some(reported) if reported.len() == shape.len() => {
            // Only the row stride is free; every stride inside a row must be the
            // packed one.
            let mut expected = reported.clone();
            for axis in 1..shape.len() {
                expected[axis] = shape[axis + 1..].iter().product::<usize>() * sample_bytes;
            }
            if reported != expected {
                return Err(PyBufferError::new_err(format!(
                    "produced plane must be packed within each row, got strides {reported:?}"
                )));
            }
            reported[0]
        }
        Some(reported) => {
            return Err(PyBufferError::new_err(format!(
                "produced plane has {} strides for {} axes",
                reported.len(),
                shape.len()
            )))
        }
    };
    if pitch < packed_row || u32::try_from(pitch).is_err() {
        return Err(PyBufferError::new_err(format!(
            "produced plane has an unusable row pitch of {pitch} bytes"
        )));
    }
    Ok(ProducedPlane {
        device_ptr,
        pitch: pitch as u32,
    })
}

/// Wrap the two planes a Python source produced as a CUDA-domain buffer, holding
/// the producing objects so the device memory outlives the frame. `context` is the
/// `CUcontext` the source reported (zero when it reported none), which a
/// downstream element that must push the producing context needs, and
/// `device_ordinal` the device it allocated on.
pub(crate) fn produced_cuda_buffer(
    luma: &Bound<'_, PyAny>,
    chroma: &Bound<'_, PyAny>,
    fmt: RawVideoFormat,
    width: u32,
    height: u32,
    context: u64,
    device_ordinal: i32,
) -> PyResult<OwnedCudaBuffer> {
    if !matches!(fmt, RawVideoFormat::Nv12 | RawVideoFormat::P010) {
        return Err(PyBufferError::new_err(
            "a GPU-resident frame must be semi-planar (NV12 or P010)",
        ));
    }
    let sample = fmt.bytes_per_sample();
    let (w, h) = (width as usize, height as usize);
    let y = read_produced_plane(luma, &[h, w], sample)?;
    let uv = read_produced_plane(chroma, &[h.div_ceil(2), w.div_ceil(2), 2], sample)?;
    Ok(OwnedCudaBuffer::new(
        y.device_ptr,
        uv.device_ptr,
        y.pitch,
        uv.pitch,
        width,
        height,
        context,
        device_ordinal,
        Arc::new(PyOwnedSurface {
            planes: [luma.clone().unbind(), chroma.clone().unbind()],
        }),
    ))
}

/// The plane layout is derived from pointers and pitches alone, so these tests
/// need no CUDA: the fake device pointers are never dereferenced.
#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::memory::CudaKeepAlive;
    use std::sync::Arc;

    #[derive(Debug)]
    struct NoOwner;
    impl CudaKeepAlive for NoOwner {}

    fn owner() -> Arc<dyn CudaKeepAlive> {
        Arc::new(NoOwner)
    }

    const FAKE_LUMA: u64 = 0xdead_0000;
    const FAKE_CHROMA: u64 = 0xdead_8000;

    /// The device the fixtures allocate on. Not 0, so a plane reporting a fixed
    /// device rather than the buffer's fails.
    const FAKE_DEVICE: i32 = 1;

    /// A 1920x1080 surface at the 2048-byte pitch a decoder would align to, so
    /// pitch != width and the strides cannot be confused with a packed layout.
    fn pitched(fmt: RawVideoFormat) -> (CudaPlane, CudaPlane) {
        let buf = OwnedCudaBuffer::new(
            FAKE_LUMA,
            FAKE_CHROMA,
            2048,
            2048,
            1920,
            1080,
            0x1234,
            FAKE_DEVICE,
            owner(),
        );
        nv12_planes(fmt, &buf).expect("NV12 / P010 are semi-planar")
    }

    /// The CAI dict as Rust values: shape, byte strides, typestr, `data`, version.
    struct Described {
        shape: Vec<usize>,
        strides: Vec<usize>,
        typestr: String,
        data: (u64, bool),
        version: u32,
    }

    fn describe(plane: &CudaPlane) -> Described {
        Python::attach(|py| {
            let dict = plane.cuda_array_interface(py).unwrap();
            let get = |key: &str| dict.get_item(key).unwrap().expect("CAI key present");
            assert!(get("stream").is_none(), "no stream synchronization claimed");
            Described {
                shape: get("shape").extract().unwrap(),
                strides: get("strides").extract().unwrap(),
                typestr: get("typestr").extract().unwrap(),
                data: get("data").extract().unwrap(),
                version: get("version").extract().unwrap(),
            }
        })
    }

    #[test]
    fn nv12_planes_describe_the_pitched_layout() {
        let (luma, chroma) = pitched(RawVideoFormat::Nv12);

        let y = describe(&luma);
        assert_eq!(y.shape, vec![1080, 1920]);
        assert_eq!(
            y.strides,
            vec![2048, 1],
            "row stride is the pitch, not width"
        );
        assert_eq!(y.typestr, "|u1");
        assert_eq!(y.data, (FAKE_LUMA, true));
        assert_eq!(y.version, 3);

        let uv = describe(&chroma);
        assert_eq!(uv.shape, vec![540, 960, 2], "interleaved UV pairs");
        assert_eq!(uv.strides, vec![2048, 2, 1]);
        assert_eq!(uv.typestr, "|u1");
        assert_eq!(uv.data, (FAKE_CHROMA, true));
    }

    #[test]
    fn p010_planes_are_16_bit_samples() {
        let (luma, chroma) = pitched(RawVideoFormat::P010);

        let y = describe(&luma);
        assert_eq!(y.shape, vec![1080, 1920]);
        assert_eq!(y.strides, vec![2048, 2]);
        assert_eq!(y.typestr, "<u2");

        let uv = describe(&chroma);
        assert_eq!(uv.shape, vec![540, 960, 2]);
        assert_eq!(uv.strides, vec![2048, 4, 2]);
    }

    #[test]
    fn odd_dimensions_round_the_chroma_plane_up() {
        let buf = OwnedCudaBuffer::new(FAKE_LUMA, FAKE_CHROMA, 64, 64, 33, 17, 0, 0, owner());
        let (_, chroma) = nv12_planes(RawVideoFormat::Nv12, &buf).unwrap();
        assert_eq!(describe(&chroma).shape, vec![9, 17, 2]);
    }

    #[test]
    fn non_semi_planar_format_has_no_planes() {
        let buf = OwnedCudaBuffer::new(
            FAKE_LUMA,
            FAKE_CHROMA,
            2048,
            2048,
            1920,
            1080,
            0,
            0,
            owner(),
        );
        assert!(nv12_planes(RawVideoFormat::Rgba8, &buf).is_none());
        assert!(nv12_planes(RawVideoFormat::I420, &buf).is_none());
    }

    #[test]
    fn context_reaches_python() {
        let (luma, _) = pitched(RawVideoFormat::Nv12);
        assert_eq!(luma.cuda_context(), 0x1234);
    }

    #[test]
    fn dlpack_reports_the_producers_device_ordinal() {
        let (luma, chroma) = pitched(RawVideoFormat::Nv12);
        for plane in [&luma, &chroma] {
            assert_eq!(
                plane.__dlpack_device__(),
                (DLPACK_DEVICE_CUDA, FAKE_DEVICE),
                "kDLCUDA on the device the producer allocated on"
            );
            let (tensor, _, _) = plane.dl_tensor().expect("pitch is a whole sample count");
            assert_eq!(tensor.device.device_id, FAKE_DEVICE);
        }
    }
}
