# Hosted-element fixtures for the GPU zero-copy paths: elements that take a
# Cuda-domain frame through `g2g_process_cuda(luma, chroma, w, h, meta)` (M984),
# its batch and produce siblings, and the DLPack export (M986).
#
# The probes that only read a plane's description never touch device memory, so
# they run with no GPU present. The cupy classes need a real device and are
# skipped by the Rust side when there is none.

# What the last g2g_process_cuda call saw, read back by the Rust test.
OBSERVED = {}

# Device surfaces allocated for a test, kept alive here for as long as the
# interpreter lives: the g2g Frame holds only the raw device pointer.
SURFACES = []

CHROMA_VALUE = 0x80
MARKER_VALUE = 0xA5


def _luma_ramp(cupy, height, width):
    """The producer's luma pattern, computed the same way on both sides."""
    rows = cupy.arange(height, dtype=cupy.uint32)[:, None]
    columns = cupy.arange(width, dtype=cupy.uint32)[None, :]
    return ((rows * 7 + columns) % 256).astype(cupy.uint8)


class CudaProbe:
    """Records both planes' CAI descriptions and stages a classification, so the
    Rust side sees both the handoff shape and that metadata still flows."""

    def g2g_process_cuda(self, luma, chroma, width, height, meta):
        OBSERVED["luma"] = luma.__cuda_array_interface__
        OBSERVED["chroma"] = chroma.__cuda_array_interface__
        OBSERVED["context"] = luma.cuda_context
        OBSERVED["geometry"] = (width, height)
        meta.add_classification(3, 0.5)


class CpuOnly:
    """Defines only the System-memory entry point, so a GPU-resident frame has
    nowhere to go: the host must refuse the frame rather than guess."""

    def g2g_process(self, buf, width, height, fmt, meta):
        pass


class PlaneRetainer:
    """Misbehaves: keeps a plane past the call, whose device pointer dangles once
    the frame is released downstream. The host must reject this."""

    def g2g_process_cuda(self, luma, chroma, width, height, meta):
        self.kept = luma


class DlpackShapeProbe:
    """Reads the DLPack surface without touching device memory: which capsule each
    `max_version` yields, the device tuple, and that an impossible request (a copy,
    or another device) is refused rather than quietly ignored. Runs with no GPU."""

    def g2g_process_cuda(self, luma, chroma, width, height, meta):
        import ctypes

        OBSERVED["dlpack_device"] = luma.__dlpack_device__()
        versioned = luma.__dlpack__(max_version=(1, 0))
        legacy = luma.__dlpack__()
        OBSERVED["versioned_capsule_name"] = _capsule_name(ctypes, versioned)
        OBSERVED["legacy_capsule_name"] = _capsule_name(ctypes, legacy)
        del versioned, legacy

        OBSERVED["copy_refused"] = _refuses(luma.__dlpack__, copy=True)
        OBSERVED["other_device_refused"] = _refuses(luma.__dlpack__, dl_device=(2, 7))


def _capsule_name(ctypes, capsule):
    """The name the producer gave a capsule, which is what a DLPack consumer keys
    off (and renames when it takes ownership)."""
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.restype = ctypes.c_char_p
    get_name.argtypes = [ctypes.py_object]
    return get_name(capsule).decode()


def _refuses(export, **kwargs):
    try:
        export(**kwargs)
    except BufferError:
        return True
    return False


class DlpackConsumer:
    """Maps both planes into cupy through DLPack (`cupy.from_dlpack`) and checks
    they alias the producer's surface, then takes an unconsumed capsule and drops
    it, which is the path the capsule destructor has to free itself."""

    def g2g_process_cuda(self, luma, chroma, width, height, meta):
        import cupy

        OBSERVED["dlpack_device"] = luma.__dlpack_device__()
        y = cupy.from_dlpack(luma)
        uv = cupy.from_dlpack(chroma)
        OBSERVED["dlpack_luma_ptr"] = int(y.data.ptr)
        OBSERVED["dlpack_chroma_ptr"] = int(uv.data.ptr)
        OBSERVED["dlpack_luma_shape"] = tuple(y.shape)
        OBSERVED["dlpack_luma_strides"] = tuple(y.strides)
        OBSERVED["dlpack_chroma_shape"] = tuple(uv.shape)
        OBSERVED["dlpack_pattern_matches"] = bool(
            (y == _luma_ramp(cupy, height, width)).all()
        )
        del y, uv
        # A capsule nobody consumes: the host's destructor has to free it.
        unconsumed = luma.__dlpack__()
        OBSERVED["unconsumed_capsule"] = type(unconsumed).__name__
        del unconsumed


class CudaBatchProbe:
    """The GPU batch shape: one (luma, chroma) pair per contributing input.
    Records what each pair described, and stages one detection labelled with the
    batch size so the Rust side sees the whole batch reached one call."""

    def g2g_process_cuda_batch(self, planes, width, height, meta):
        OBSERVED["batch_size"] = len(planes)
        OBSERVED["batch_luma_ptrs"] = [
            int(luma.__cuda_array_interface__["data"][0]) for luma, _ in planes
        ]
        OBSERVED["batch_chroma_ptrs"] = [
            int(chroma.__cuda_array_interface__["data"][0]) for _, chroma in planes
        ]
        OBSERVED["batch_geometry"] = (width, height)
        meta.add_classification(len(planes), 1.0)


class CudaSource:
    """A GPU source: allocates each surface itself with cupy and hands back the two
    planes, which is how a Python source produces device-resident frames (this crate
    cannot allocate device memory). Ends the stream after `frames` surfaces."""

    def __init__(self):
        self.frames = 2
        self.produced = 0
        self.surfaces = []

    def g2g_produce_cuda(self, width, height, meta):
        if self.produced >= self.frames:
            return None
        import cupy

        pitch = width + 64  # deliberately not the packed width
        surface = cupy.zeros((height + height // 2, pitch), dtype=cupy.uint8)
        surface[:height, :width] = self.produced + 1
        surface[height:, :width] = CHROMA_VALUE
        luma = surface[:height, :width]
        # A strided view cannot be reshaped without copying, so build the
        # interleaved chroma view over the same memory directly.
        chroma = cupy.ndarray(
            (height // 2, width // 2, 2),
            dtype=cupy.uint8,
            memptr=surface.data + pitch * height,
            strides=(pitch, 2, 1),
        )
        cupy.cuda.runtime.deviceSynchronize()
        # Keep the allocation (and the views' base) alive past the call: the frame
        # holds only the device pointers.
        self.surfaces.append(surface)
        self.produced += 1
        OBSERVED["produced_ptr"] = int(surface.data.ptr)
        OBSERVED["produced_pitch"] = pitch
        meta.add_classification(self.produced, 1.0)
        return luma, chroma


class BadCudaSource:
    """Returns a plane of the wrong shape, which the host must refuse rather than
    hand a mis-described device pointer downstream."""

    def g2g_produce_cuda(self, width, height, meta):
        import cupy

        wrong = cupy.zeros((height, width // 2), dtype=cupy.uint8)
        return wrong, wrong


def allocate_nv12(width, height, pitch):
    """Allocate an NV12 surface in device memory laid out like a decoder's (both
    planes in one allocation, chroma following luma after `height` pitched rows),
    fill luma with the ramp and chroma with a flat grey, and return
    `(device_pointer, pitch)`. None when there is no usable CUDA device, which
    the caller treats as "skip".
    """
    try:
        import cupy
    except Exception as e:  # cupy not installed
        OBSERVED["allocate_error"] = repr(e)
        return None
    try:
        surface = cupy.zeros((height + height // 2, pitch), dtype=cupy.uint8)
        surface[:height, :width] = _luma_ramp(cupy, height, width)
        surface[height:, :width] = CHROMA_VALUE
        cupy.cuda.runtime.deviceSynchronize()
    except Exception as e:  # no device, no driver, out of memory
        OBSERVED["allocate_error"] = repr(e)
        return None
    SURFACES.append(surface)
    return int(surface.data.ptr), pitch


def read_surface(row, column):
    """Read one luma byte through the *producer's* array, to see whether a write
    made through a CAI-derived array landed in the same device memory."""
    import cupy

    cupy.cuda.runtime.deviceSynchronize()
    return int(SURFACES[-1][row, column])


class CupyConsumer:
    """Maps both planes into cupy and checks they alias the producer's surface:
    the device pointer is identical, the pitched strides survive, and the
    producer's pattern reads back. Then writes one marker byte through the luma
    array so the producer side can confirm the write is visible in its own array.

    The plane is exported read-only as advice only (cupy does not enforce it);
    the marker write is a deliberate probe that no copy sits in between.
    """

    def g2g_process_cuda(self, luma, chroma, width, height, meta):
        import cupy

        y = cupy.asarray(luma)
        uv = cupy.asarray(chroma)
        OBSERVED["luma_ptr"] = int(y.data.ptr)
        OBSERVED["chroma_ptr"] = int(uv.data.ptr)
        OBSERVED["cai_ptr"] = int(luma.__cuda_array_interface__["data"][0])
        OBSERVED["luma_shape"] = tuple(y.shape)
        OBSERVED["luma_strides"] = tuple(y.strides)
        OBSERVED["chroma_shape"] = tuple(uv.shape)
        OBSERVED["pattern_matches"] = bool((y == _luma_ramp(cupy, height, width)).all())
        OBSERVED["chroma_matches"] = bool((uv == CHROMA_VALUE).all())
        y[1, 2] = MARKER_VALUE
        cupy.cuda.runtime.deviceSynchronize()
