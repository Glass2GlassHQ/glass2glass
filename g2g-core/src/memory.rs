use alloc::boxed::Box;
use alloc::sync::Arc;

#[cfg(feature = "runtime")]
use crate::pool::PooledBuffer;
use crate::tensor::TensorView;

#[derive(Debug)]
pub enum MemoryDomain {
    System(SystemSlice),
    /// Shared-CPU strided buffer: reference-counted bytes plus a [`TensorView`]
    /// describing how to read them (M180). A layout-preserving transform (flip,
    /// transpose, crop) hands the frame downstream by composing strides on the
    /// *same* `Arc` allocation, so zero bytes are copied. The system-memory
    /// analog of the per-plane stride metadata the GPU domains (eg
    /// [`OwnedCudaBuffer`]) already carry. A consumer that needs contiguous
    /// bytes calls [`SystemView::materialize`]; a stride-aware consumer reads
    /// the view directly.
    SystemView(SystemView),
    DmaBuf(OwnedDmaBuf),
    VulkanTexture(OwnedVulkanTexture),
    WebGPUBuffer(OwnedWebGPUBuffer),
    /// NVIDIA CUDA device memory. Carries raw device pointers (the decoded
    /// frame stays on the GPU), so a downstream GPU consumer can use it with
    /// no device->host copy. The backing allocation is owned elsewhere (eg an
    /// ffmpeg `CUDA`-hwframe `AVFrame`); see [`OwnedCudaBuffer`].
    Cuda(OwnedCudaBuffer),
    /// Direct3D 11 texture (Windows GPU memory). The decoded frame stays in a
    /// `ID3D11Texture2D` so a DXGI / D3D11 consumer (a swapchain present sink)
    /// uses it without a GPU->CPU copy. The texture is owned elsewhere (eg a
    /// Media Foundation `IMFDXGIBuffer` from a DXVA decoder); see
    /// [`OwnedD3D11Texture`]. The Windows analog of [`MemoryDomain::Cuda`].
    D3D11Texture(OwnedD3D11Texture),
    /// A decoded picture left as a browser `VideoFrame`, to be imported into
    /// WebGPU as a `GPUExternalTexture` and sampled on the GPU (browser/wasm),
    /// so a WebCodecs-decoded frame never round-trips to CPU. The frame is
    /// owned elsewhere (a `web_sys::VideoFrame`); see
    /// [`OwnedWebGPUExternalTexture`]. The browser analog of
    /// [`MemoryDomain::D3D11Texture`].
    WebGPUExternalTexture(OwnedWebGPUExternalTexture),
    /// A picture rendered into a native wgpu GPU texture (desktop Vulkan / Metal
    /// / D3D12). The render-side analog of the decode-side CUDA / D3D11 domains:
    /// a GPU element (eg the Vello analytics overlay) draws straight into a
    /// `wgpu::Texture` and forwards the frame with no GPU->CPU copy, so a GPU
    /// sink presents it directly. `g2g-core` never links wgpu, so the texture is
    /// owned by a [`WgpuKeepAlive`] the producing element boxes; a consumer that
    /// links wgpu recovers it via [`WgpuKeepAlive::as_any`]. See
    /// [`OwnedWgpuTexture`].
    WgpuTexture(OwnedWgpuTexture),
}

/// The memory domain of a [`MemoryDomain`] without its payload. Used by the
/// allocation query (M12) so a consumer can name the kind of memory it wants
/// allocated without holding a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MemoryDomainKind {
    #[default]
    System,
    SystemView,
    DmaBuf,
    VulkanTexture,
    WebGPUBuffer,
    Cuda,
    D3D11Texture,
    WebGPUExternalTexture,
    WgpuTexture,
}

impl MemoryDomain {
    /// The payload-free discriminant of this domain.
    pub fn kind(&self) -> MemoryDomainKind {
        match self {
            MemoryDomain::System(_) => MemoryDomainKind::System,
            MemoryDomain::SystemView(_) => MemoryDomainKind::SystemView,
            MemoryDomain::DmaBuf(_) => MemoryDomainKind::DmaBuf,
            MemoryDomain::VulkanTexture(_) => MemoryDomainKind::VulkanTexture,
            MemoryDomain::WebGPUBuffer(_) => MemoryDomainKind::WebGPUBuffer,
            MemoryDomain::Cuda(_) => MemoryDomainKind::Cuda,
            MemoryDomain::D3D11Texture(_) => MemoryDomainKind::D3D11Texture,
            MemoryDomain::WebGPUExternalTexture(_) => MemoryDomainKind::WebGPUExternalTexture,
            MemoryDomain::WgpuTexture(_) => MemoryDomainKind::WgpuTexture,
        }
    }
}

/// CPU-memory slice. The backing buffer may be a freshly-allocated `Box<[u8]>`
/// (`from_boxed`) or a pool-recycled buffer (`from_pool`). Dropping the
/// `SystemSlice` releases the underlying storage — in the pooled case, the
/// buffer returns to its pool automatically.
#[derive(Debug)]
pub struct SystemSlice {
    inner: SystemSliceInner,
}

#[derive(Debug)]
enum SystemSliceInner {
    Owned(Box<[u8]>),
    // a pool buffer may be larger than the frame, so the valid payload length
    // is carried explicitly rather than inferred from the buffer capacity.
    #[cfg(feature = "runtime")]
    Pooled { buffer: PooledBuffer<Box<[u8]>>, len: usize },
}

impl SystemSlice {
    pub fn from_boxed(bytes: Box<[u8]>) -> Self {
        Self { inner: SystemSliceInner::Owned(bytes) }
    }

    /// Wrap a pooled buffer, exposing only its first `len` bytes (the valid
    /// frame payload). The backing buffer may be larger than `len`.
    #[cfg(feature = "runtime")]
    pub fn from_pool(buffer: PooledBuffer<Box<[u8]>>, len: usize) -> Self {
        debug_assert!(len <= buffer.as_ref().len(), "valid len exceeds pool buffer");
        Self { inner: SystemSliceInner::Pooled { buffer, len } }
    }

    pub fn as_slice(&self) -> &[u8] {
        match &self.inner {
            SystemSliceInner::Owned(b) => b,
            #[cfg(feature = "runtime")]
            SystemSliceInner::Pooled { buffer, len } => &buffer.as_ref()[..*len],
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match &mut self.inner {
            SystemSliceInner::Owned(b) => b,
            #[cfg(feature = "runtime")]
            SystemSliceInner::Pooled { buffer, len } => &mut buffer.as_mut()[..*len],
        }
    }
}

/// Shared-CPU strided buffer (M180): an `Arc<[u8]>` backing plus a
/// [`TensorView`] over it. The payload of [`MemoryDomain::SystemView`]. Cloning
/// it (or composing a new view, eg a flip) shares the same allocation, so a
/// layout-preserving transform copies nothing. Two `SystemView`s alias the same
/// bytes iff `Arc::ptr_eq(a.backing(), b.backing())` (the zero-copy witness used
/// in tests).
#[derive(Debug, Clone)]
pub struct SystemView {
    backing: Arc<[u8]>,
    view: TensorView,
}

impl SystemView {
    pub fn new(backing: Arc<[u8]>, view: TensorView) -> Self {
        Self { backing, view }
    }

    /// The shared backing buffer (the whole allocation the view indexes into).
    pub fn backing(&self) -> &Arc<[u8]> {
        &self.backing
    }

    /// The strided view describing how to read [`backing`](Self::backing).
    pub fn view(&self) -> &TensorView {
        &self.view
    }

    /// Materialize the view into a fresh dense row-major buffer: the one copy a
    /// strided chain pays, and only when a consumer needs contiguous bytes.
    pub fn materialize(&self) -> Box<[u8]> {
        self.view.materialize(&self.backing)
    }
}

#[derive(Debug)]
pub struct OwnedDmaBuf {
    fd: i32,
    pub stride: u32,
    pub offset: u32,
}

impl OwnedDmaBuf {
    /// # Safety
    /// `fd` must be a valid DMABUF descriptor with no other owner; the caller
    /// transfers ownership to this struct.
    pub unsafe fn from_raw(fd: i32, stride: u32, offset: u32) -> Self {
        Self { fd, stride, offset }
    }

    pub fn as_raw(&self) -> i32 {
        self.fd
    }
}

impl Drop for OwnedDmaBuf {
    fn drop(&mut self) {
        // DMABUF is Linux-only. On std+linux, close the fd. On other targets
        // (Wasm, RTOS without libc) we leak; a custom close hook registered
        // by the owning BufferPool is the planned no_std story.
        #[cfg(all(target_os = "linux", feature = "std"))]
        {
            extern "C" {
                fn close(fd: i32) -> i32;
            }
            // SAFETY: `from_raw` is the only constructor and is `unsafe`,
            // requiring callers to certify sole ownership of the fd.
            unsafe {
                close(self.fd);
            }
        }
    }
}

#[derive(Debug)]
pub struct OwnedVulkanTexture {
    pub handle: u64,
    pub allocation_id: u64,
}

#[derive(Debug)]
pub struct OwnedWebGPUBuffer {
    pub buffer_id: u64,
}

/// A decoded NV12 picture left in CUDA device memory. Holds the two NV12
/// plane device pointers (luma Y, interleaved chroma UV) with their row
/// pitches, the CUDA context they are valid in, and a keep-alive owner that
/// pins the backing allocation for as long as the pointers are referenced.
///
/// The device memory itself is not owned by this struct: an ffmpeg
/// `CUDA`-hwframe decoder owns it as part of an `AVFrame`. `g2g-core` cannot
/// link CUDA, so the producing element hands over its owning handle boxed as
/// a [`CudaKeepAlive`]; dropping this buffer drops that box, releasing the
/// frame back to its hwframe pool. The pointers stay valid for exactly the
/// lifetime of the keep-alive.
#[derive(Debug)]
pub struct OwnedCudaBuffer {
    /// `CUdeviceptr` (as `u64`) of the luma (Y) plane.
    pub luma_ptr: u64,
    /// `CUdeviceptr` (as `u64`) of the interleaved chroma (UV, NV12) plane.
    pub chroma_ptr: u64,
    /// Row pitch in bytes of the luma plane (>= width; the decoder aligns it).
    pub luma_pitch: u32,
    /// Row pitch in bytes of the chroma plane.
    pub chroma_pitch: u32,
    /// Visible picture dimensions in pixels.
    pub width: u32,
    pub height: u32,
    /// `CUcontext` (as `u64`) the pointers are valid in. A consumer pushes
    /// this context (`cuCtxPushCurrent`) before touching the memory.
    pub context: u64,
    /// Pins the backing allocation (eg the decoder's `AVFrame`) for the life
    /// of the pointers. Dropping it releases the allocation.
    keep_alive: Box<dyn CudaKeepAlive>,
}

impl OwnedCudaBuffer {
    /// Wrap CUDA device pointers with the owner that keeps them valid.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        luma_ptr: u64,
        chroma_ptr: u64,
        luma_pitch: u32,
        chroma_pitch: u32,
        width: u32,
        height: u32,
        context: u64,
        keep_alive: Box<dyn CudaKeepAlive>,
    ) -> Self {
        Self {
            luma_ptr,
            chroma_ptr,
            luma_pitch,
            chroma_pitch,
            width,
            height,
            context,
            keep_alive,
        }
    }

    /// The keep-alive owner, exposed so a consumer that imports the memory
    /// into another API (eg CUDA external memory) can take shared ownership.
    pub fn keep_alive(&self) -> &dyn CudaKeepAlive {
        self.keep_alive.as_ref()
    }
}

/// Owner token kept alongside an [`OwnedCudaBuffer`]'s device pointers. The
/// CUDA memory is owned by the producing element (typically an ffmpeg
/// `AVFrame` from a `CUDA` hwframe pool); `g2g-core` cannot link CUDA, so the
/// element boxes its owning handle as this trait object. Dropping the box
/// releases the backing allocation. `Send` so a frame can cross the runner's
/// worker-thread boundaries like every other domain.
pub trait CudaKeepAlive: core::fmt::Debug + Send {}

/// A decoded picture left in a Direct3D 11 texture (Windows GPU memory). Holds
/// the `ID3D11Texture2D` pointer, the subresource index of this frame within
/// it (DXVA decoders commonly hand out one texture *array* whose subresources
/// are the decoded surfaces), the visible dims, the DXGI format, the D3D11
/// device the texture belongs to, and a keep-alive owner that pins the texture
/// for as long as the pointer is referenced.
///
/// The texture is not owned by this struct: a Media Foundation `IMFDXGIBuffer`
/// (from a DXVA `IMFTransform`) owns it. `g2g-core` cannot link the `windows`
/// crate (it is `no_std`), so the producing element boxes its owning handle as
/// a [`D3D11KeepAlive`] trait object; dropping this buffer drops the box and
/// releases the sample back to the decoder. The pointer stays valid for
/// exactly the keep-alive's lifetime. The Windows analog of [`OwnedCudaBuffer`].
#[derive(Debug)]
pub struct OwnedD3D11Texture {
    /// `ID3D11Texture2D` (as `u64`) backing this frame.
    pub texture: u64,
    /// Index of this frame's subresource within `texture` (0 for a
    /// single-surface texture; the decode slot for a texture array).
    pub subresource: u32,
    /// Visible picture dimensions in pixels.
    pub width: u32,
    pub height: u32,
    /// `DXGI_FORMAT` (as `u32`) of the texture, eg `DXGI_FORMAT_NV12` (103).
    pub dxgi_format: u32,
    /// `ID3D11Device` (as `u64`) the texture belongs to. A consumer must use
    /// this device (or one sharing its adapter) to read the texture.
    pub device: u64,
    /// Pins the backing texture (eg the decoder's `IMFSample`) for the life of
    /// the pointer. Dropping it releases the allocation.
    keep_alive: Box<dyn D3D11KeepAlive>,
}

impl OwnedD3D11Texture {
    /// Wrap a D3D11 texture pointer with the owner that keeps it valid.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        texture: u64,
        subresource: u32,
        width: u32,
        height: u32,
        dxgi_format: u32,
        device: u64,
        keep_alive: Box<dyn D3D11KeepAlive>,
    ) -> Self {
        Self {
            texture,
            subresource,
            width,
            height,
            dxgi_format,
            device,
            keep_alive,
        }
    }

    /// The keep-alive owner, exposed so a consumer that imports the texture
    /// into another API (eg a Direct3D swapchain) can take shared ownership.
    pub fn keep_alive(&self) -> &dyn D3D11KeepAlive {
        self.keep_alive.as_ref()
    }
}

/// Owner token kept alongside an [`OwnedD3D11Texture`]'s texture pointer. The
/// texture is owned by the producing element (typically a Media Foundation
/// `IMFSample` / `IMFDXGIBuffer` from a DXVA decoder); `g2g-core` cannot link
/// the `windows` crate, so the element boxes its owning handle as this trait
/// object. Dropping the box releases the texture. `Send` so a frame can cross
/// the runner's worker-thread boundaries like every other domain.
pub trait D3D11KeepAlive: core::fmt::Debug + Send {}

/// A decoded picture left as a browser `VideoFrame`, to be imported into
/// WebGPU as a `GPUExternalTexture` (`device.importExternalTexture`) and
/// sampled in a compute or render pass, so a WebCodecs-decoded frame is
/// preprocessed and run through inference without ever copying to CPU. A
/// `VideoFrame`-sourced external texture stays valid until the frame is
/// closed, so the keep-alive owns the frame and closes it on drop.
///
/// The `VideoFrame` is a JS handle `g2g-core` cannot name (it is `no_std`
/// and never links `web-sys`), so the producing element (a WebCodecs
/// decoder) boxes the owner as a [`WebGPUKeepAlive`]. Unlike the CUDA / D3D11
/// domains, whose payload is a raw pointer this struct carries directly, the
/// payload here lives inside the owner, so a consumer recovers it by
/// downcasting [`WebGPUKeepAlive::as_any`]. The browser analog of
/// [`OwnedD3D11Texture`] / [`OwnedCudaBuffer`].
#[derive(Debug)]
pub struct OwnedWebGPUExternalTexture {
    /// Visible picture dimensions in pixels.
    pub width: u32,
    pub height: u32,
    /// Owns the backing `VideoFrame` for the life of the imported texture;
    /// dropping it closes the frame and frees the decoder's output slot.
    keep_alive: Box<dyn WebGPUKeepAlive>,
}

impl OwnedWebGPUExternalTexture {
    /// Wrap a decoded frame's dimensions with the owner that keeps the
    /// backing `VideoFrame` alive.
    pub fn new(width: u32, height: u32, keep_alive: Box<dyn WebGPUKeepAlive>) -> Self {
        Self { width, height, keep_alive }
    }

    /// The keep-alive owner, for a consumer to downcast via
    /// [`WebGPUKeepAlive::as_any`] and recover the `VideoFrame` to import, or
    /// to take shared ownership.
    pub fn keep_alive(&self) -> &dyn WebGPUKeepAlive {
        self.keep_alive.as_ref()
    }
}

/// Owner token kept alongside an [`OwnedWebGPUExternalTexture`]. Owns the
/// browser `VideoFrame`; the producing element boxes its handle as this trait
/// object and a consumer that can link `web-sys` downcasts via
/// [`as_any`](Self::as_any) to recover the frame for `importExternalTexture`.
/// Dropping the box closes the frame. `Send` so a frame can cross the
/// runner's worker boundaries like every other domain; on the single-threaded
/// wasm target the concrete owner asserts `Send` under that contract (the
/// frame never actually crosses a thread), as the `D3D11KeepAlive` owners do
/// for their COM handles.
pub trait WebGPUKeepAlive: core::fmt::Debug + Send {
    /// Recover the concrete owner so a consumer can extract the `VideoFrame`.
    /// Mirrors the raw-pointer access the CUDA / D3D11 domains expose
    /// directly; here the payload lives in the owner, so downcast is the route.
    fn as_any(&self) -> &dyn core::any::Any;
}

/// A picture rendered into a native wgpu GPU texture (the payload of
/// [`MemoryDomain::WgpuTexture`]). Carries the visible dimensions; the
/// `wgpu::Texture` itself lives inside the [`WgpuKeepAlive`] owner because
/// `g2g-core` never links wgpu. The desktop render-side analog of
/// [`OwnedWebGPUExternalTexture`] (which is the browser/import side).
#[derive(Debug)]
pub struct OwnedWgpuTexture {
    /// Visible picture dimensions in pixels.
    pub width: u32,
    pub height: u32,
    /// Owns the backing `wgpu::Texture` for as long as the frame is referenced.
    keep_alive: Box<dyn WgpuKeepAlive>,
}

impl OwnedWgpuTexture {
    /// Wrap a rendered texture's dimensions with the owner that keeps the
    /// backing `wgpu::Texture` alive.
    pub fn new(width: u32, height: u32, keep_alive: Box<dyn WgpuKeepAlive>) -> Self {
        Self { width, height, keep_alive }
    }

    /// The keep-alive owner, for a consumer that links wgpu to downcast via
    /// [`WgpuKeepAlive::as_any`] and recover the `wgpu::Texture` to present, or
    /// to take shared ownership.
    pub fn keep_alive(&self) -> &dyn WgpuKeepAlive {
        self.keep_alive.as_ref()
    }
}

/// Owner token kept alongside an [`OwnedWgpuTexture`]. Owns the backing
/// `wgpu::Texture`; the producing element boxes its handle as this trait object
/// and a consumer that links wgpu downcasts via [`as_any`](Self::as_any) to
/// recover the texture for presentation or further GPU work. Dropping the box
/// releases the texture. `Send + Sync` because `wgpu::Texture` is, so a frame
/// crosses the multi-thread runner's worker boundaries like every other domain.
pub trait WgpuKeepAlive: core::fmt::Debug + Send + Sync {
    /// Recover the concrete owner so a consumer can extract the `wgpu::Texture`.
    fn as_any(&self) -> &dyn core::any::Any;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicBool, Ordering};

    /// Stands in for a producer's owning handle (eg an ffmpeg `AVFrame`):
    /// flips a shared flag on drop so the test can prove the keep-alive owner
    /// is released exactly when the buffer is.
    #[derive(Debug)]
    struct FlagOnDrop(Arc<AtomicBool>);
    impl Drop for FlagOnDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    impl CudaKeepAlive for FlagOnDrop {}
    impl D3D11KeepAlive for FlagOnDrop {}
    impl WebGPUKeepAlive for FlagOnDrop {
        fn as_any(&self) -> &dyn core::any::Any {
            self
        }
    }

    #[test]
    fn cuda_domain_reports_cuda_kind() {
        let dropped = Arc::new(AtomicBool::new(false));
        let buf = OwnedCudaBuffer::new(
            0x1000,
            0x2000,
            2048,
            2048,
            1920,
            1080,
            0xC0FFEE,
            Box::new(FlagOnDrop(dropped.clone())),
        );
        let domain = MemoryDomain::Cuda(buf);
        assert_eq!(domain.kind(), MemoryDomainKind::Cuda);
    }

    #[test]
    fn dropping_cuda_buffer_releases_keep_alive() {
        let dropped = Arc::new(AtomicBool::new(false));
        let buf = OwnedCudaBuffer::new(
            0x1000,
            0x2000,
            2048,
            2048,
            1920,
            1080,
            0xC0FFEE,
            Box::new(FlagOnDrop(dropped.clone())),
        );
        assert!(!dropped.load(Ordering::SeqCst), "owner alive while buffer held");
        assert_eq!(buf.luma_ptr, 0x1000);
        assert_eq!(buf.chroma_ptr, 0x2000);
        drop(buf);
        assert!(
            dropped.load(Ordering::SeqCst),
            "dropping the buffer must release the backing allocation"
        );
    }

    #[test]
    fn d3d11_domain_reports_d3d11_kind() {
        let dropped = Arc::new(AtomicBool::new(false));
        let tex = OwnedD3D11Texture::new(
            0xDEAD_BEEF,
            2,
            1920,
            1080,
            103, // DXGI_FORMAT_NV12
            0xD3D1CE,
            Box::new(FlagOnDrop(dropped.clone())),
        );
        let domain = MemoryDomain::D3D11Texture(tex);
        assert_eq!(domain.kind(), MemoryDomainKind::D3D11Texture);
    }

    #[test]
    fn dropping_d3d11_texture_releases_keep_alive() {
        let dropped = Arc::new(AtomicBool::new(false));
        let tex = OwnedD3D11Texture::new(
            0xDEAD_BEEF,
            2,
            1920,
            1080,
            103,
            0xD3D1CE,
            Box::new(FlagOnDrop(dropped.clone())),
        );
        assert!(!dropped.load(Ordering::SeqCst), "owner alive while texture held");
        assert_eq!(tex.texture, 0xDEAD_BEEF);
        assert_eq!(tex.subresource, 2);
        drop(tex);
        assert!(
            dropped.load(Ordering::SeqCst),
            "dropping the texture must release the backing allocation"
        );
    }

    #[test]
    fn webgpu_external_texture_reports_kind() {
        let dropped = Arc::new(AtomicBool::new(false));
        let tex = OwnedWebGPUExternalTexture::new(640, 480, Box::new(FlagOnDrop(dropped)));
        assert_eq!(tex.width, 640);
        assert_eq!(tex.height, 480);
        let domain = MemoryDomain::WebGPUExternalTexture(tex);
        assert_eq!(domain.kind(), MemoryDomainKind::WebGPUExternalTexture);
    }

    #[test]
    fn dropping_webgpu_external_texture_closes_frame() {
        let dropped = Arc::new(AtomicBool::new(false));
        let tex = OwnedWebGPUExternalTexture::new(1280, 720, Box::new(FlagOnDrop(dropped.clone())));
        assert!(!dropped.load(Ordering::SeqCst), "owner alive while texture held");
        drop(tex);
        assert!(
            dropped.load(Ordering::SeqCst),
            "dropping the texture must close the backing VideoFrame"
        );
    }

    #[test]
    fn webgpu_keep_alive_downcasts_to_concrete_owner() {
        // a consumer that can link web-sys recovers the concrete VideoFrame
        // owner through as_any to call importExternalTexture.
        let dropped = Arc::new(AtomicBool::new(false));
        let tex = OwnedWebGPUExternalTexture::new(16, 16, Box::new(FlagOnDrop(dropped)));
        assert!(tex.keep_alive().as_any().downcast_ref::<FlagOnDrop>().is_some());
    }
}
