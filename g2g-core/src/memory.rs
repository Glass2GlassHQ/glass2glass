#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(feature = "alloc")]
use alloc::sync::Arc;
use core::ffi::c_void;

#[cfg(feature = "runtime")]
use crate::pool::PooledBuffer;
#[cfg(feature = "alloc")]
use crate::tensor::TensorView;

#[derive(Debug)]
#[non_exhaustive]
pub enum MemoryDomain {
    System(SystemSlice),
    // Every domain below is heap-backed (a `Box`/`Arc` handle or keep-alive), so
    // it is gated behind `alloc`. The heap-free MCU build carries only `System`,
    // which the `StaticLendRing` fills via `SystemSlice::from_foreign` (zero-copy).
    /// Shared-CPU strided buffer: reference-counted bytes plus a [`TensorView`]
    /// describing how to read them (M180). A layout-preserving transform (flip,
    /// transpose, crop) hands the frame downstream by composing strides on the
    /// *same* `Arc` allocation, so zero bytes are copied. The system-memory
    /// analog of the per-plane stride metadata the GPU domains (eg
    /// [`OwnedCudaBuffer`]) already carry. A consumer that needs contiguous
    /// bytes calls [`SystemView::materialize`]; a stride-aware consumer reads
    /// the view directly.
    #[cfg(feature = "alloc")]
    SystemView(SystemView),
    #[cfg(feature = "alloc")]
    DmaBuf(OwnedDmaBuf),
    #[cfg(feature = "alloc")]
    VulkanTexture(OwnedVulkanTexture),
    #[cfg(feature = "alloc")]
    WebGPUBuffer(OwnedWebGPUBuffer),
    /// NVIDIA CUDA device memory. Carries raw device pointers (the decoded
    /// frame stays on the GPU), so a downstream GPU consumer can use it with
    /// no device->host copy. The backing allocation is owned elsewhere (eg an
    /// ffmpeg `CUDA`-hwframe `AVFrame`); see [`OwnedCudaBuffer`].
    #[cfg(feature = "alloc")]
    Cuda(OwnedCudaBuffer),
    /// Direct3D 11 texture (Windows GPU memory). The decoded frame stays in a
    /// `ID3D11Texture2D` so a DXGI / D3D11 consumer (a swapchain present sink)
    /// uses it without a GPU->CPU copy. The texture is owned elsewhere (eg a
    /// Media Foundation `IMFDXGIBuffer` from a DXVA decoder); see
    /// [`OwnedD3D11Texture`]. The Windows analog of [`MemoryDomain::Cuda`].
    #[cfg(feature = "alloc")]
    D3D11Texture(OwnedD3D11Texture),
    /// A decoded picture left as a browser `VideoFrame`, to be imported into
    /// WebGPU as a `GPUExternalTexture` and sampled on the GPU (browser/wasm),
    /// so a WebCodecs-decoded frame never round-trips to CPU. The frame is
    /// owned elsewhere (a `web_sys::VideoFrame`); see
    /// [`OwnedWebGPUExternalTexture`]. The browser analog of
    /// [`MemoryDomain::D3D11Texture`].
    #[cfg(feature = "alloc")]
    WebGPUExternalTexture(OwnedWebGPUExternalTexture),
    /// A picture rendered into a native wgpu GPU texture (desktop Vulkan / Metal
    /// / D3D12). The render-side analog of the decode-side CUDA / D3D11 domains:
    /// a GPU element (eg the Vello analytics overlay) draws straight into a
    /// `wgpu::Texture` and forwards the frame with no GPU->CPU copy, so a GPU
    /// sink presents it directly. `g2g-core` never links wgpu, so the texture is
    /// owned by a [`WgpuKeepAlive`] the producing element boxes; a consumer that
    /// links wgpu recovers it via [`WgpuKeepAlive::as_any`]. See
    /// [`OwnedWgpuTexture`].
    #[cfg(feature = "alloc")]
    WgpuTexture(OwnedWgpuTexture),
    /// A GPU-resident tensor (or other linear data) in a native wgpu storage
    /// buffer (M215). The buffer analog of [`WgpuTexture`](Self::WgpuTexture): a
    /// GPU element (eg `WgpuPreprocess` in GPU-output mode) leaves a compute
    /// shader's output in a `wgpu::Buffer` rather than reading it back to the
    /// CPU, so a downstream GPU consumer reads it with no GPU->CPU copy. Owned by
    /// a [`WgpuBufferKeepAlive`] (g2g-core never links wgpu); a consumer recovers
    /// it via [`WgpuBufferKeepAlive::as_any`]. See [`OwnedWgpuBuffer`].
    #[cfg(feature = "alloc")]
    WgpuBuffer(OwnedWgpuBuffer),
}

/// The memory domain of a [`MemoryDomain`] without its payload. Used by the
/// allocation query (M12) so a consumer can name the kind of memory it wants
/// allocated without holding a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
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
    WgpuBuffer,
}

impl MemoryDomainKind {
    /// Stable bit index for [`DomainSet`]. Keep in sync with the enum (one bit
    /// per variant, 10 variants fit a `u16`).
    const fn bit_index(self) -> u16 {
        match self {
            MemoryDomainKind::System => 0,
            MemoryDomainKind::SystemView => 1,
            MemoryDomainKind::DmaBuf => 2,
            MemoryDomainKind::VulkanTexture => 3,
            MemoryDomainKind::WebGPUBuffer => 4,
            MemoryDomainKind::Cuda => 5,
            MemoryDomainKind::D3D11Texture => 6,
            MemoryDomainKind::WebGPUExternalTexture => 7,
            MemoryDomainKind::WgpuTexture => 8,
            MemoryDomainKind::WgpuBuffer => 9,
        }
    }
}

/// Preference order for picking a single domain out of a [`DomainSet`]: the
/// zero-copy GPU-resident domains first, plain `System` bytes last. So when a
/// producer and its consumer(s) both accept several domains, the negotiation
/// keeps the frame on the device rather than falling back to a host copy.
const DOMAIN_PREFERENCE: [MemoryDomainKind; 10] = [
    MemoryDomainKind::Cuda,
    MemoryDomainKind::D3D11Texture,
    MemoryDomainKind::VulkanTexture,
    MemoryDomainKind::WgpuTexture,
    MemoryDomainKind::WgpuBuffer,
    MemoryDomainKind::WebGPUExternalTexture,
    MemoryDomainKind::WebGPUBuffer,
    MemoryDomainKind::DmaBuf,
    MemoryDomainKind::SystemView,
    MemoryDomainKind::System,
];

/// A set of [`MemoryDomainKind`]s a producer can emit or a consumer can accept,
/// packed into a bitmask. The allocation query negotiates a single concrete
/// domain by intersecting the producer's capability set with the consumer's
/// acceptance set and picking the most-preferred survivor
/// ([`preferred`](Self::preferred)). A single-domain producer/consumer (the
/// default) is just [`only`](Self::only), so the negotiation reduces to today's
/// exact-match behavior; a multi-domain element (a decoder that can deliver to
/// System or stay resident on the GPU) names every domain it can satisfy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainSet(u16);

impl DomainSet {
    /// The empty set: no domain. The intersection result that signals an
    /// irreconcilable conflict (producer and consumer share no domain).
    pub const EMPTY: Self = Self(0);

    /// Every domain. The default *input* acceptance of an element (it imposes no
    /// domain requirement on its upstream), so the converter auto-plug only acts
    /// on elements that declare a narrower `input_domains`.
    pub const ALL: Self = Self(0x03ff); // 10 variants -> low 10 bits

    /// Iterate the member domains in preference order (GPU-resident first).
    pub fn iter(self) -> impl Iterator<Item = MemoryDomainKind> {
        DOMAIN_PREFERENCE.into_iter().filter(move |&k| self.contains(k))
    }

    /// The singleton set holding just `k`. The default capability/acceptance of
    /// every element, derived from its single `output_memory()`.
    pub const fn only(k: MemoryDomainKind) -> Self {
        Self(1 << k.bit_index())
    }

    /// This set with `k` added.
    pub const fn with(self, k: MemoryDomainKind) -> Self {
        Self(self.0 | (1 << k.bit_index()))
    }

    /// Whether `k` is a member.
    pub const fn contains(self, k: MemoryDomainKind) -> bool {
        self.0 & (1 << k.bit_index()) != 0
    }

    /// The set of domains in both `self` and `other` (the domains a producer and
    /// consumer can agree on).
    pub const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// The set of domains in either `self` or `other`.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether the set holds no domain.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The most-preferred member per [`DOMAIN_PREFERENCE`] (GPU-resident before
    /// `System`), or `None` if the set is empty. The single concrete domain the
    /// negotiation settles on.
    pub fn preferred(self) -> Option<MemoryDomainKind> {
        DOMAIN_PREFERENCE.iter().copied().find(|&k| self.contains(k))
    }
}

impl MemoryDomain {
    /// The payload-free discriminant of this domain.
    pub fn kind(&self) -> MemoryDomainKind {
        match self {
            MemoryDomain::System(_) => MemoryDomainKind::System,
            #[cfg(feature = "alloc")]
            MemoryDomain::SystemView(_) => MemoryDomainKind::SystemView,
            #[cfg(feature = "alloc")]
            MemoryDomain::DmaBuf(_) => MemoryDomainKind::DmaBuf,
            #[cfg(feature = "alloc")]
            MemoryDomain::VulkanTexture(_) => MemoryDomainKind::VulkanTexture,
            #[cfg(feature = "alloc")]
            MemoryDomain::WebGPUBuffer(_) => MemoryDomainKind::WebGPUBuffer,
            #[cfg(feature = "alloc")]
            MemoryDomain::Cuda(_) => MemoryDomainKind::Cuda,
            #[cfg(feature = "alloc")]
            MemoryDomain::D3D11Texture(_) => MemoryDomainKind::D3D11Texture,
            #[cfg(feature = "alloc")]
            MemoryDomain::WebGPUExternalTexture(_) => MemoryDomainKind::WebGPUExternalTexture,
            #[cfg(feature = "alloc")]
            MemoryDomain::WgpuTexture(_) => MemoryDomainKind::WgpuTexture,
            #[cfg(feature = "alloc")]
            MemoryDomain::WgpuBuffer(_) => MemoryDomainKind::WgpuBuffer,
        }
    }

    /// Produce a second handle to this frame's memory for a fan-out branch
    /// (M213, M250). A **zero-copy** reference-count bump for every domain: the
    /// GPU domains and the shared-CPU [`SystemView`] are handle-shared; owned-CPU
    /// [`System`](Self::System) bytes are too, **provided** [`make_shareable`]
    /// has run first (the tee does this once before fanning out). Without that
    /// pre-share, `System` falls back to a deep copy (nothing to refcount yet).
    ///
    /// Read-only fan-out: branches must treat the shared memory as immutable. A
    /// branch that needs to mutate copies first (as the per-frame metadata does
    /// copy-on-write, and `SystemSlice::as_mut_slice` does for shared bytes), so
    /// the shares never alias a mutation.
    ///
    /// [`make_shareable`]: Self::make_shareable
    #[cfg(feature = "alloc")]
    pub fn share(&self) -> MemoryDomain {
        match self {
            // CPU bytes: a refcount bump if pre-shared, else a deep copy.
            MemoryDomain::System(s) => MemoryDomain::System(s.share_handle()),
            // Everything else is refcounted/handle-shared: clone is cheap.
            MemoryDomain::SystemView(v) => MemoryDomain::SystemView(v.clone()),
            MemoryDomain::DmaBuf(d) => MemoryDomain::DmaBuf(d.clone()),
            MemoryDomain::VulkanTexture(t) => MemoryDomain::VulkanTexture(t.clone()),
            MemoryDomain::WebGPUBuffer(b) => MemoryDomain::WebGPUBuffer(b.clone()),
            MemoryDomain::Cuda(c) => MemoryDomain::Cuda(c.clone()),
            MemoryDomain::D3D11Texture(t) => MemoryDomain::D3D11Texture(t.clone()),
            MemoryDomain::WebGPUExternalTexture(t) => {
                MemoryDomain::WebGPUExternalTexture(t.clone())
            }
            MemoryDomain::WgpuTexture(t) => MemoryDomain::WgpuTexture(t.clone()),
            MemoryDomain::WgpuBuffer(b) => MemoryDomain::WgpuBuffer(b.clone()),
        }
    }

    /// Prepare this frame's memory to be fanned out zero-copy (M250): convert
    /// owned-CPU [`System`](Self::System) bytes into a refcounted shareable handle
    /// once, so the subsequent per-branch [`share`](Self::share) calls are
    /// refcount bumps rather than deep copies. The other domains are already
    /// handle-shared, so this is a no-op for them. Called once by the tee before
    /// it broadcasts.
    #[cfg(feature = "alloc")]
    pub fn make_shareable(&mut self) {
        if let MemoryDomain::System(s) = self {
            s.make_shared();
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
    #[cfg(feature = "alloc")]
    Owned(Box<[u8]>),
    // a pool buffer may be larger than the frame, so the valid payload length
    // is carried explicitly rather than inferred from the buffer capacity.
    #[cfg(feature = "runtime")]
    Pooled { buffer: PooledBuffer<Box<[u8]>>, len: usize },
    // Foreign-owned CPU bytes lent to the pipeline zero-copy (M234), e.g. an
    // application buffer through the C ABI. Freed via the callback on drop. The
    // only variant in the heap-free build: the `StaticLendRing` lends a static
    // slot's bytes this way, so the MCU data path allocates nothing.
    Foreign(ForeignSlice),
    // Refcounted owned bytes shared across tee branches (M250). `Arc<Box<[u8]>>`
    // (not `Arc<[u8]>`) so `make_shared` *moves* the existing `Box` in with no
    // byte copy; `share_handle` then hands each branch a refcount bump. Read-only
    // while shared: `as_mut_slice` copies out first (copy-on-write).
    #[cfg(feature = "alloc")]
    Shared(Arc<Box<[u8]>>),
    // The pooled analog: a shared pooled buffer, returned to its pool once the
    // last branch drops.
    #[cfg(feature = "runtime")]
    SharedPooled { buffer: Arc<PooledBuffer<Box<[u8]>>>, len: usize },
}

/// An empty owned boxed slice, used as a `mem::replace` placeholder while an
/// inner buffer is moved out.
#[cfg(feature = "alloc")]
fn empty_boxed() -> Box<[u8]> {
    alloc::vec::Vec::new().into_boxed_slice()
}

impl SystemSlice {
    #[cfg(feature = "alloc")]
    pub fn from_boxed(bytes: Box<[u8]>) -> Self {
        Self { inner: SystemSliceInner::Owned(bytes) }
    }

    /// Wrap foreign-owned CPU bytes zero-copy: no copy is made, the pipeline
    /// reads `ptr[..len]` directly, and on drop `free(user)` is invoked (if
    /// `free` is `Some`) to hand the buffer back to its owner. A mutating
    /// consumer ([`as_mut_slice`](Self::as_mut_slice)) transparently copies the
    /// bytes out first, so the lend stays read-only.
    ///
    /// # Safety
    /// `ptr` must point to `len` bytes that stay valid and unmodified by the
    /// lender until `free` runs; `free`/`user` must be safe to invoke from the
    /// pipeline's thread (the lend contract). A `None` `free` means the lender
    /// guarantees the buffer outlives the pipeline (no reclamation needed).
    pub unsafe fn from_foreign(
        ptr: *const u8,
        len: usize,
        free: Option<unsafe extern "C" fn(*mut c_void)>,
        user: *mut c_void,
    ) -> Self {
        Self {
            inner: SystemSliceInner::Foreign(ForeignSlice {
                ptr: ptr as usize,
                len,
                free,
                user: user as usize,
            }),
        }
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
            #[cfg(feature = "alloc")]
            SystemSliceInner::Owned(b) => b,
            #[cfg(feature = "runtime")]
            SystemSliceInner::Pooled { buffer, len } => &buffer.as_ref()[..*len],
            SystemSliceInner::Foreign(f) => f.as_slice(),
            #[cfg(feature = "alloc")]
            SystemSliceInner::Shared(arc) => arc,
            #[cfg(feature = "runtime")]
            SystemSliceInner::SharedPooled { buffer, len } => &buffer.as_ref().as_ref()[..*len],
        }
    }

    /// A mutable view of the bytes. Only meaningful when a heap is available (an
    /// owned / copy-on-write buffer); the heap-free build lends read-only static
    /// bytes via [`from_foreign`](Self::from_foreign), mutating the ring slot
    /// before the lend instead.
    #[cfg(feature = "alloc")]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // A lent or tee-shared buffer is read-only: copy-on-write to owned bytes
        // first, so an in-place transform never writes the lender's memory or a
        // sibling branch's shared frame. A uniquely-held `Shared` Arc is
        // reclaimed without a copy.
        let needs_cow = matches!(
            self.inner,
            SystemSliceInner::Foreign(_) | SystemSliceInner::Shared(_)
        ) || {
            #[cfg(feature = "runtime")]
            {
                matches!(self.inner, SystemSliceInner::SharedPooled { .. })
            }
            #[cfg(not(feature = "runtime"))]
            {
                false
            }
        };
        if needs_cow {
            let taken = core::mem::replace(&mut self.inner, SystemSliceInner::Owned(empty_boxed()));
            self.inner = match taken {
                SystemSliceInner::Foreign(f) => {
                    SystemSliceInner::Owned(f.as_slice().to_vec().into_boxed_slice())
                }
                // Unique share: reclaim the boxed bytes with no copy; otherwise
                // (a sibling branch still holds it) deep-copy.
                SystemSliceInner::Shared(arc) => match Arc::try_unwrap(arc) {
                    Ok(b) => SystemSliceInner::Owned(b),
                    Err(arc) => SystemSliceInner::Owned(arc.as_ref().clone()),
                },
                #[cfg(feature = "runtime")]
                SystemSliceInner::SharedPooled { buffer, len } => match Arc::try_unwrap(buffer) {
                    // Unique pooled share: reclaim the pooled buffer mutably with
                    // no copy (and keep it pooled); otherwise deep-copy.
                    Ok(buffer) => SystemSliceInner::Pooled { buffer, len },
                    Err(arc) => SystemSliceInner::Owned(
                        arc.as_ref().as_ref()[..len].to_vec().into_boxed_slice(),
                    ),
                },
                _ => unreachable!("needs_cow only set for Foreign / Shared / SharedPooled"),
            };
        }
        match &mut self.inner {
            SystemSliceInner::Owned(b) => b,
            #[cfg(feature = "runtime")]
            SystemSliceInner::Pooled { buffer, len } => &mut buffer.as_mut()[..*len],
            _ => unreachable!("Foreign / Shared / SharedPooled converted to Owned above"),
        }
    }

    /// Convert this slice into a refcounted shareable handle in place (M250), so
    /// a tee can hand each branch a second handle via [`share_handle`] with no
    /// byte copy. Owned bytes (and a pooled buffer) *move* into an `Arc`; a
    /// foreign lent buffer is copied out first (the lend stays single-owner,
    /// read-only). Already-shared slices are a no-op. Idempotent.
    #[cfg(feature = "alloc")]
    pub(crate) fn make_shared(&mut self) {
        match &self.inner {
            SystemSliceInner::Shared(_) => return,
            #[cfg(feature = "runtime")]
            SystemSliceInner::SharedPooled { .. } => return,
            _ => {}
        }
        let taken = core::mem::replace(&mut self.inner, SystemSliceInner::Owned(empty_boxed()));
        self.inner = match taken {
            SystemSliceInner::Owned(b) => SystemSliceInner::Shared(Arc::new(b)),
            #[cfg(feature = "runtime")]
            SystemSliceInner::Pooled { buffer, len } => {
                SystemSliceInner::SharedPooled { buffer: Arc::new(buffer), len }
            }
            SystemSliceInner::Foreign(f) => {
                SystemSliceInner::Shared(Arc::new(f.as_slice().to_vec().into_boxed_slice()))
            }
            already => already,
        };
    }

    /// A second handle to this slice for a fan-out branch. Zero-copy (an `Arc`
    /// refcount bump) when [`make_shared`] has run; a deep copy otherwise (the
    /// caller did not pre-share, so there is nothing to refcount).
    #[cfg(feature = "alloc")]
    pub(crate) fn share_handle(&self) -> SystemSlice {
        let inner = match &self.inner {
            SystemSliceInner::Shared(arc) => SystemSliceInner::Shared(arc.clone()),
            #[cfg(feature = "runtime")]
            SystemSliceInner::SharedPooled { buffer, len } => {
                SystemSliceInner::SharedPooled { buffer: buffer.clone(), len: *len }
            }
            _ => SystemSliceInner::Owned(self.as_slice().to_vec().into_boxed_slice()),
        };
        SystemSlice { inner }
    }
}

/// Foreign-owned CPU bytes lent to the pipeline (M234). Pointers are stored as
/// `usize` so the type is `Send`/`Sync` without an `unsafe impl`, the same
/// convention [`OwnedCudaBuffer`] uses for device pointers; the lender's
/// contract (documented on [`SystemSlice::from_foreign`]) certifies the bytes
/// are valid and safe to read from the pipeline thread. Dropping it invokes the
/// free callback, returning the buffer to its owner.
#[derive(Debug)]
pub struct ForeignSlice {
    ptr: usize,
    len: usize,
    free: Option<unsafe extern "C" fn(*mut c_void)>,
    user: usize,
}

impl ForeignSlice {
    fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: the lend contract (see `SystemSlice::from_foreign`) guarantees
        // `ptr` covers `len` valid bytes, unmodified for this slice's lifetime.
        unsafe { core::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

impl Drop for ForeignSlice {
    fn drop(&mut self) {
        if let Some(free) = self.free {
            // SAFETY: the lend contract certifies `free(user)` is safe to call
            // once, here, to reclaim the buffer. Called exactly once (on drop).
            unsafe { free(self.user as *mut c_void) };
        }
    }
}

/// Shared-CPU strided buffer (M180): an `Arc<[u8]>` backing plus a
/// [`TensorView`] over it. The payload of [`MemoryDomain::SystemView`]. Cloning
/// it (or composing a new view, eg a flip) shares the same allocation, so a
/// layout-preserving transform copies nothing. Two `SystemView`s alias the same
/// bytes iff `Arc::ptr_eq(a.backing(), b.backing())` (the zero-copy witness used
/// in tests).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct SystemView {
    backing: Arc<[u8]>,
    view: TensorView,
}

#[cfg(feature = "alloc")]
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

/// `Clone` is a refcount bump on the shared fd (M213): a tee branch references
/// the same DMABUF, and the fd is closed once when the last share drops.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct OwnedDmaBuf {
    /// The fd's owner is reference-counted, so the buffer fans out through a tee
    /// without dup-ing the descriptor; the last drop closes it.
    fd: Arc<DmaBufFd>,
    pub stride: u32,
    pub offset: u32,
    /// Optional cross-process GPU-completion sync (M562): a shared
    /// [`SyncFd`] (an exported timeline-semaphore fd, one per stream) plus the
    /// timeline value this frame's producer GPU work signals on completion. When
    /// present, a GPU consumer imports the fd once and host-waits `value` before
    /// reading the buffer, so the *producer* need not block on the copy
    /// (`WgpuToDmaBuf` / `DmaBufToWgpu`). `None` means the buffer is already
    /// complete (the producer synchronised itself, e.g. a CPU / capture dma-buf).
    sync: Option<(SyncFd, u64)>,
}

/// A shared, reference-counted GPU-completion sync fd: an exported timeline
/// semaphore a producer signals when a frame's GPU work finishes. The producer
/// exports it *once* per stream and attaches a [`Clone`] to every frame via
/// [`OwnedDmaBuf::with_sync`]; the fd is closed exactly once, on the last drop.
/// A consumer recovers it with [`OwnedDmaBuf::sync_fd`] / [`OwnedDmaBuf::sync`].
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct SyncFd(Arc<DmaBufFd>);

#[cfg(feature = "alloc")]
impl SyncFd {
    /// # Safety
    /// `fd` must be a valid semaphore descriptor (e.g. a `VK_KHR_external_semaphore_fd`
    /// export) with no other owner; the caller transfers ownership.
    pub unsafe fn from_raw(fd: i32) -> Self {
        Self(Arc::new(DmaBufFd(fd)))
    }

    /// The raw fd, valid for as long as this (or a clone) is held.
    pub fn as_raw(&self) -> i32 {
        self.0 .0
    }
}

/// Sole owner of a DMABUF descriptor; closes it on drop. Held behind an `Arc` in
/// [`OwnedDmaBuf`] so the descriptor is shared, not duplicated, across a tee.
#[cfg(feature = "alloc")]
#[derive(Debug)]
struct DmaBufFd(i32);

#[cfg(feature = "alloc")]
impl Drop for DmaBufFd {
    fn drop(&mut self) {
        // DMABUF is Linux-only. On std+linux, close the fd. On other targets
        // (Wasm, RTOS without libc) we leak; a custom close hook registered
        // by the owning BufferPool is the planned no_std story.
        #[cfg(all(target_os = "linux", feature = "std"))]
        {
            extern "C" {
                fn close(fd: i32) -> i32;
            }
            // SAFETY: `OwnedDmaBuf::from_raw` is the only constructor and is
            // `unsafe`, requiring callers to certify sole ownership of the fd;
            // the `Arc` ensures this runs exactly once, on the last share.
            unsafe {
                close(self.0);
            }
        }
    }
}

#[cfg(feature = "alloc")]
impl OwnedDmaBuf {
    /// # Safety
    /// `fd` must be a valid DMABUF descriptor with no other owner; the caller
    /// transfers ownership to this struct.
    pub unsafe fn from_raw(fd: i32, stride: u32, offset: u32) -> Self {
        Self { fd: Arc::new(DmaBufFd(fd)), stride, offset, sync: None }
    }

    pub fn as_raw(&self) -> i32 {
        self.fd.0
    }

    /// Attach a GPU-completion [`SyncFd`] and the timeline `value` this frame's
    /// producer work signals. A consumer host-waits `value` on the imported
    /// semaphore before reading the buffer. Safe: [`SyncFd`] already owns the fd.
    pub fn with_sync(mut self, sync: SyncFd, value: u64) -> Self {
        self.sync = Some((sync, value));
        self
    }

    /// The raw sync (timeline-semaphore) fd, if this frame carries one.
    pub fn sync_fd(&self) -> Option<i32> {
        self.sync.as_ref().map(|(s, _)| s.as_raw())
    }

    /// The timeline value to wait for before reading this buffer, if synced.
    pub fn sync_value(&self) -> Option<u64> {
        self.sync.as_ref().map(|(_, v)| *v)
    }

    /// The shared sync handle + value, if any, so a transport can re-share the one
    /// stream semaphore across the frames it reconstructs.
    pub fn sync(&self) -> Option<(SyncFd, u64)> {
        self.sync.clone()
    }
}

/// `Clone` shares the handle (it is an opaque id the producer owns); a tee
/// branch references the same texture (M213).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct OwnedVulkanTexture {
    pub handle: u64,
    pub allocation_id: u64,
}

/// `Clone` shares the buffer id (M213).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
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
/// `Clone` is a zero-copy refcount bump (the device memory is shared, not
/// copied), so the frame can fan out through a tee to several GPU consumers
/// (M213); see [`MemoryDomain::share`].
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
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
    /// of the pointers. Reference-counted (`Arc`), so a tee branch shares the
    /// allocation rather than copying it; the last drop releases it.
    keep_alive: Arc<dyn CudaKeepAlive>,
}

#[cfg(feature = "alloc")]
impl OwnedCudaBuffer {
    /// Wrap CUDA device pointers with the owner that keeps them valid. The owner
    /// is `Arc`-held so the frame is cheaply shareable across a tee (M213).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        luma_ptr: u64,
        chroma_ptr: u64,
        luma_pitch: u32,
        chroma_pitch: u32,
        width: u32,
        height: u32,
        context: u64,
        keep_alive: Arc<dyn CudaKeepAlive>,
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

    /// A shared-ownership clone of the keep-alive owner. A consumer that outlives
    /// individual frames but depends on the producer's resources (a CUDA encode
    /// session holding the producer's context across frames) clones this to pin
    /// the context / allocation for its own lifetime, so producer teardown cannot
    /// race ahead of it.
    pub fn keep_alive_arc(&self) -> Arc<dyn CudaKeepAlive> {
        Arc::clone(&self.keep_alive)
    }
}

/// Owner token kept alongside an [`OwnedCudaBuffer`]'s device pointers. The
/// CUDA memory is owned by the producing element (typically an ffmpeg
/// `AVFrame` from a `CUDA` hwframe pool); `g2g-core` cannot link CUDA, so the
/// element boxes its owning handle as this trait object. Dropping the box
/// releases the backing allocation. `Send + Sync` so a frame can cross the
/// runner's worker-thread boundaries and, after a tee, be read concurrently by
/// several GPU consumer branches (M213); the producing element certifies the
/// device memory is safe for concurrent read-only access (immutable decoded
/// surface), the same contract under which the owners assert `Send`.
#[cfg(feature = "alloc")]
pub trait CudaKeepAlive: core::fmt::Debug + Send + Sync {}

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
/// `Clone` is a zero-copy refcount bump (the texture is shared, not copied), so
/// the frame can fan out through a tee (M213); see [`MemoryDomain::share`].
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
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
    /// the pointer. Reference-counted (`Arc`), so a tee branch shares the
    /// texture rather than copying it; the last drop releases it.
    keep_alive: Arc<dyn D3D11KeepAlive>,
}

#[cfg(feature = "alloc")]
impl OwnedD3D11Texture {
    /// Wrap a D3D11 texture pointer with the owner that keeps it valid. The owner
    /// is `Arc`-held so the frame is cheaply shareable across a tee (M213).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        texture: u64,
        subresource: u32,
        width: u32,
        height: u32,
        dxgi_format: u32,
        device: u64,
        keep_alive: Arc<dyn D3D11KeepAlive>,
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
/// object. Dropping the box releases the texture. `Send + Sync` so a frame can
/// cross the runner's worker-thread boundaries and, after a tee, be read
/// concurrently by several GPU consumer branches (M213); the producing element
/// certifies the texture is safe for concurrent read-only access, the same
/// contract under which the owners assert `Send`.
#[cfg(feature = "alloc")]
pub trait D3D11KeepAlive: core::fmt::Debug + Send + Sync {}

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
/// `Clone` is a zero-copy refcount bump (the `VideoFrame` is shared, not
/// copied), so the frame can fan out through a tee (M213); see
/// [`MemoryDomain::share`].
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct OwnedWebGPUExternalTexture {
    /// Visible picture dimensions in pixels.
    pub width: u32,
    pub height: u32,
    /// Owns the backing `VideoFrame` for the life of the imported texture;
    /// reference-counted (`Arc`), so a tee branch shares the frame rather than
    /// closing it early; the last drop closes it and frees the decoder slot.
    keep_alive: Arc<dyn WebGPUKeepAlive>,
}

#[cfg(feature = "alloc")]
impl OwnedWebGPUExternalTexture {
    /// Wrap a decoded frame's dimensions with the owner that keeps the
    /// backing `VideoFrame` alive. `Arc`-held so the frame is shareable (M213).
    pub fn new(width: u32, height: u32, keep_alive: Arc<dyn WebGPUKeepAlive>) -> Self {
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
/// for their COM handles. `Sync` too, so a tee branch can read it concurrently
/// (M213), under the same single-threaded-wasm / documented-contract rationale.
#[cfg(feature = "alloc")]
pub trait WebGPUKeepAlive: core::fmt::Debug + Send + Sync {
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
/// `Clone` is a zero-copy refcount bump (the `wgpu::Texture` is shared, not
/// copied), so the rendered frame can fan out through a tee (M213); see
/// [`MemoryDomain::share`].
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct OwnedWgpuTexture {
    /// Visible picture dimensions in pixels.
    pub width: u32,
    pub height: u32,
    /// Owns the backing `wgpu::Texture` for as long as the frame is referenced;
    /// reference-counted (`Arc`) so a tee branch shares it rather than copying.
    keep_alive: Arc<dyn WgpuKeepAlive>,
}

#[cfg(feature = "alloc")]
impl OwnedWgpuTexture {
    /// Wrap a rendered texture's dimensions with the owner that keeps the
    /// backing `wgpu::Texture` alive. `Arc`-held so the frame is shareable (M213).
    pub fn new(width: u32, height: u32, keep_alive: Arc<dyn WgpuKeepAlive>) -> Self {
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
#[cfg(feature = "alloc")]
pub trait WgpuKeepAlive: core::fmt::Debug + Send + Sync {
    /// Recover the concrete owner so a consumer can extract the `wgpu::Texture`.
    fn as_any(&self) -> &dyn core::any::Any;
}

/// A GPU-resident linear buffer (the payload of [`MemoryDomain::WgpuBuffer`],
/// M215): a `wgpu::Buffer` holding a tensor or other linear data a compute
/// shader produced. Carries the valid payload length in bytes; the
/// `wgpu::Buffer` itself lives inside the [`WgpuBufferKeepAlive`] owner because
/// `g2g-core` never links wgpu. The buffer analog of [`OwnedWgpuTexture`].
/// `Clone` is a zero-copy refcount bump (M213).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct OwnedWgpuBuffer {
    /// Valid payload length in bytes (the buffer may be padded larger).
    pub len: usize,
    /// Owns the backing `wgpu::Buffer` (and the device needed to read it) for as
    /// long as the frame is referenced; reference-counted so a tee branch shares
    /// it rather than copying.
    keep_alive: Arc<dyn WgpuBufferKeepAlive>,
}

#[cfg(feature = "alloc")]
impl OwnedWgpuBuffer {
    /// Wrap a GPU buffer's payload length with the owner that keeps the backing
    /// `wgpu::Buffer` alive. `Arc`-held so the frame is shareable (M213).
    pub fn new(len: usize, keep_alive: Arc<dyn WgpuBufferKeepAlive>) -> Self {
        Self { len, keep_alive }
    }

    /// The keep-alive owner, for a consumer that links wgpu to downcast via
    /// [`WgpuBufferKeepAlive::as_any`] and recover the `wgpu::Buffer` for further
    /// GPU work, or to read it back to the CPU.
    pub fn keep_alive(&self) -> &dyn WgpuBufferKeepAlive {
        self.keep_alive.as_ref()
    }
}

/// Owner token kept alongside an [`OwnedWgpuBuffer`]. Owns the backing
/// `wgpu::Buffer` (and typically the device needed to map it); the producing
/// element boxes its handle as this trait object and a consumer that links wgpu
/// downcasts via [`as_any`](Self::as_any) to recover the buffer for further GPU
/// work or a read-back. Dropping the last reference releases the buffer.
/// `Send + Sync` because `wgpu::Buffer` is, so the frame crosses the multi-thread
/// runner's worker boundaries and fans out through a tee like every other domain.
#[cfg(feature = "alloc")]
pub trait WgpuBufferKeepAlive: core::fmt::Debug + Send + Sync {
    /// Recover the concrete owner so a consumer can extract the `wgpu::Buffer`.
    fn as_any(&self) -> &dyn core::any::Any;
}

#[cfg(all(test, feature = "alloc"))]
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
            Arc::new(FlagOnDrop(dropped.clone())),
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
            Arc::new(FlagOnDrop(dropped.clone())),
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
            Arc::new(FlagOnDrop(dropped.clone())),
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
            Arc::new(FlagOnDrop(dropped.clone())),
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
        let tex = OwnedWebGPUExternalTexture::new(640, 480, Arc::new(FlagOnDrop(dropped)));
        assert_eq!(tex.width, 640);
        assert_eq!(tex.height, 480);
        let domain = MemoryDomain::WebGPUExternalTexture(tex);
        assert_eq!(domain.kind(), MemoryDomainKind::WebGPUExternalTexture);
    }

    #[test]
    fn dropping_webgpu_external_texture_closes_frame() {
        let dropped = Arc::new(AtomicBool::new(false));
        let tex = OwnedWebGPUExternalTexture::new(1280, 720, Arc::new(FlagOnDrop(dropped.clone())));
        assert!(!dropped.load(Ordering::SeqCst), "owner alive while texture held");
        drop(tex);
        assert!(
            dropped.load(Ordering::SeqCst),
            "dropping the texture must close the backing VideoFrame"
        );
    }

    #[test]
    fn sharing_a_gpu_domain_is_a_refcount_bump_not_a_copy() {
        // M213: share() on a GPU domain hands a second branch the SAME backing
        // allocation (refcount bump), so the keep-alive releases exactly once,
        // only after BOTH shares drop, never twice (which a copy would imply).
        let dropped = Arc::new(AtomicBool::new(false));
        let buf = OwnedCudaBuffer::new(
            0x1000, 0x2000, 2048, 2048, 1920, 1080, 0xC0FFEE,
            Arc::new(FlagOnDrop(dropped.clone())),
        );
        let original = MemoryDomain::Cuda(buf);
        let branch = original.share();
        // Both shares point at the same pointers: zero device-memory copy.
        match (&original, &branch) {
            (MemoryDomain::Cuda(a), MemoryDomain::Cuda(b)) => {
                assert_eq!(a.luma_ptr, b.luma_ptr);
                assert_eq!(a.chroma_ptr, b.chroma_ptr);
            }
            _ => panic!("share preserved the domain"),
        }
        assert!(!dropped.load(Ordering::SeqCst), "keep-alive held by both shares");
        drop(branch);
        assert!(!dropped.load(Ordering::SeqCst), "still held by the original");
        drop(original);
        assert!(dropped.load(Ordering::SeqCst), "released once the last share drops");
    }

    #[test]
    fn sharing_system_bytes_after_make_shareable_is_zero_copy() {
        // M250: once made shareable, a tee hands each branch a handle to the SAME
        // backing bytes (a refcount bump), not a copy: identical data pointers.
        let mut original =
            MemoryDomain::System(SystemSlice::from_boxed(alloc::vec![9u8; 64].into_boxed_slice()));
        let orig_ptr = match &original {
            MemoryDomain::System(s) => s.as_slice().as_ptr(),
            _ => unreachable!(),
        };
        original.make_shareable();
        let branch = original.share();
        let (a, b) = match (&original, &branch) {
            (MemoryDomain::System(a), MemoryDomain::System(b)) => (a, b),
            _ => panic!("share preserves the domain"),
        };
        assert_eq!(a.as_slice().as_ptr(), b.as_slice().as_ptr(), "branches share one buffer");
        // The move into the Arc preserved the original allocation (no copy).
        assert_eq!(a.as_slice().as_ptr(), orig_ptr, "make_shareable moved, did not copy");
        assert_eq!(b.as_slice(), &[9u8; 64], "shared bytes intact");
    }

    #[test]
    fn mutating_a_shared_branch_copies_on_write() {
        // A branch that mutates must not write the sibling's bytes: as_mut_slice
        // copies out first while the share is still held.
        let mut original =
            MemoryDomain::System(SystemSlice::from_boxed(alloc::vec![1u8; 8].into_boxed_slice()));
        original.make_shareable();
        let mut branch = original.share();
        let MemoryDomain::System(branch_slice) = &mut branch else { unreachable!() };
        branch_slice.as_mut_slice()[0] = 42;
        assert_eq!(branch_slice.as_slice()[0], 42, "branch sees its own write");
        match &original {
            MemoryDomain::System(o) => {
                assert_eq!(o.as_slice()[0], 1, "original untouched by the branch's write");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn webgpu_keep_alive_downcasts_to_concrete_owner() {
        // a consumer that can link web-sys recovers the concrete VideoFrame
        // owner through as_any to call importExternalTexture.
        let dropped = Arc::new(AtomicBool::new(false));
        let tex = OwnedWebGPUExternalTexture::new(16, 16, Arc::new(FlagOnDrop(dropped)));
        assert!(tex.keep_alive().as_any().downcast_ref::<FlagOnDrop>().is_some());
    }

    /// Increments the `AtomicUsize` at `user`, standing in for an application's
    /// buffer-reclaim notify.
    extern "C" fn count_free(user: *mut c_void) {
        // SAFETY: the tests pass a live &AtomicUsize as `user`.
        unsafe { &*(user as *const core::sync::atomic::AtomicUsize) }
            .fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn foreign_slice_reads_in_place_and_frees_once_on_drop() {
        use core::sync::atomic::AtomicUsize;
        let buf = [1u8, 2, 3, 4];
        let frees = AtomicUsize::new(0);
        let user = &frees as *const AtomicUsize as *mut c_void;
        // SAFETY: `buf` outlives the slice; `count_free` is safe to call once.
        let s = unsafe { SystemSlice::from_foreign(buf.as_ptr(), buf.len(), Some(count_free), user) };
        // Zero-copy read: the slice points at the same bytes (same address).
        assert_eq!(s.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(s.as_slice().as_ptr(), buf.as_ptr(), "read in place, no copy");
        assert_eq!(frees.load(Ordering::SeqCst), 0, "not freed while live");
        drop(s);
        assert_eq!(frees.load(Ordering::SeqCst), 1, "freed exactly once on drop");
    }

    #[test]
    fn foreign_slice_copies_out_on_mutation() {
        use core::sync::atomic::AtomicUsize;
        let buf = [5u8; 4];
        let frees = AtomicUsize::new(0);
        let user = &frees as *const AtomicUsize as *mut c_void;
        // SAFETY: `buf` outlives the slice; `count_free` is safe to call once.
        let mut s =
            unsafe { SystemSlice::from_foreign(buf.as_ptr(), buf.len(), Some(count_free), user) };
        // Mutating returns the lend (free fires) and switches to an owned copy.
        s.as_mut_slice()[0] = 9;
        assert_eq!(frees.load(Ordering::SeqCst), 1, "CoW released the lend");
        assert_eq!(s.as_slice(), &[9, 5, 5, 5]);
        assert_ne!(s.as_slice().as_ptr(), buf.as_ptr(), "now an owned copy");
        drop(s);
        assert_eq!(frees.load(Ordering::SeqCst), 1, "no double free after CoW");
    }
}
