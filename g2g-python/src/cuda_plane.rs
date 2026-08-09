//! GPU zero-copy handoff: NV12 CUDA planes described to Python through
//! `__cuda_array_interface__` v3 (M984, `python` feature).
//!
//! A [`MemoryDomain::Cuda`](g2g_core::MemoryDomain::Cuda) frame carries device
//! pointers, not bytes, so the buffer protocol the System path uses cannot
//! describe it. Each semi-planar plane becomes a [`CudaPlane`] whose
//! `__cuda_array_interface__` dict names the device pointer, the plane shape and
//! the producer's row pitch as strides, so `cupy.asarray(plane)` or
//! `torch.as_tensor(plane, device="cuda")` alias the decoder's surface with no
//! PCIe round-trip.
//!
//! CUDA-context caveat: CAI carries no CUDA context. The pointers are valid only
//! in the context the producer decoded into, exposed as the plane's
//! `cuda_context` property; a consumer must already be in that context or push
//! it (`cuCtxPushCurrent`) before touching the memory. cupy and torch use the
//! device's primary context, so a producer that decoded into its own driver
//! context needs the Python side to push `cuda_context` itself. DLPack is the
//! cross-framework alternative that does carry a device and stream contract; it
//! is not implemented here.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use g2g_core::memory::OwnedCudaBuffer;
use g2g_core::RawVideoFormat;

/// `__cuda_array_interface__` version this describes.
const CAI_VERSION: u32 = 3;

/// CAI's `data` read-only flag. The device memory belongs to the producer (a
/// decoder surface), and a teed frame hands the same pointers to every branch
/// under a read-only-sharing guarantee; unlike the System path there is no
/// copy-on-write to fall back on here, so a plane is exported read-only. The
/// flag is advisory: cupy and torch do not enforce it.
const PLANE_READ_ONLY: bool = true;

/// One plane of a GPU-resident frame, described to Python by CAI v3. Holds only
/// the device pointer and the plane's layout, so it is inert on the Rust side:
/// nothing here touches CUDA. Valid for the duration of one
/// `g2g_process_cuda` call (see the retention guard in [`crate::host`]).
#[pyclass(frozen, module = "g2g")]
#[derive(Debug)]
pub(crate) struct CudaPlane {
    device_ptr: u64,
    shape: Vec<usize>,
    /// Byte strides, outermost first: the producer's row pitch, then the
    /// element stride(s) within a row.
    strides: Vec<usize>,
    typestr: &'static str,
    context: u64,
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
        dict.set_item("typestr", self.typestr)?;
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
}

/// Describe a semi-planar GPU frame as its two planes: luma `(height, width)`
/// and interleaved chroma `(height/2, width/2, 2)`, both strided by the
/// producer's row pitch. `None` for a format the CUDA domain does not carry
/// (only NV12 and its 10-bit P010 sibling are semi-planar).
pub(crate) fn nv12_planes(
    fmt: RawVideoFormat,
    buf: &OwnedCudaBuffer,
) -> Option<(CudaPlane, CudaPlane)> {
    let typestr = match fmt {
        RawVideoFormat::Nv12 => "|u1",
        RawVideoFormat::P010 => "<u2",
        _ => return None,
    };
    let sample = fmt.bytes_per_sample();
    let (width, height) = (buf.width as usize, buf.height as usize);
    let luma = CudaPlane {
        device_ptr: buf.luma_ptr,
        shape: vec![height, width],
        strides: vec![buf.luma_pitch as usize, sample],
        typestr,
        context: buf.context,
    };
    let chroma = CudaPlane {
        device_ptr: buf.chroma_ptr,
        shape: vec![height.div_ceil(2), width.div_ceil(2), 2],
        strides: vec![buf.chroma_pitch as usize, 2 * sample, sample],
        typestr,
        context: buf.context,
    };
    Some((luma, chroma))
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
        let buf = OwnedCudaBuffer::new(FAKE_LUMA, FAKE_CHROMA, 64, 64, 33, 17, 0, owner());
        let (_, chroma) = nv12_planes(RawVideoFormat::Nv12, &buf).unwrap();
        assert_eq!(describe(&chroma).shape, vec![9, 17, 2]);
    }

    #[test]
    fn non_semi_planar_format_has_no_planes() {
        let buf = OwnedCudaBuffer::new(FAKE_LUMA, FAKE_CHROMA, 2048, 2048, 1920, 1080, 0, owner());
        assert!(nv12_planes(RawVideoFormat::Rgba8, &buf).is_none());
        assert!(nv12_planes(RawVideoFormat::I420, &buf).is_none());
    }

    #[test]
    fn context_reaches_python() {
        let (luma, _) = pitched(RawVideoFormat::Nv12);
        assert_eq!(luma.cuda_context(), 0x1234);
    }
}
