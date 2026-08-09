# Hosted-element fixtures for the M984 GPU zero-copy path: elements that take a
# Cuda-domain frame through `g2g_process_cuda(luma, chroma, w, h, meta)`.
#
# `CudaProbe` and the misbehaving classes only read the CAI dict, never the
# device memory, so they run with no GPU present. `CupyConsumer` needs a real
# device and is skipped by the Rust side when `allocate_nv12` returns None.

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
