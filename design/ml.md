# Machine learning

GPU tensor preprocessing, the inference backends, multi-stream batching, and the
per-frame metadata that carries detections through the graph to an overlay. Part
of the design in [README.md](README.md).

To prevent GPU-to-CPU synchronization stalls, tensor execution happens directly
inside the VRAM domain. ML elements are `AsyncElement` implementations like any
other: they negotiate `Caps::RawVideo` on the input pad and `Caps::Tensor` on the
output pad.

## Inline tensor preprocessing

The ML element sits in the same memory domain context as the hardware decoder.
When a `MemoryDomain::DmaBuf` frame arrives at the ML element, the memory handle
is bound directly as a texture inside a `wgpu` compute pipeline, an inline
compute shader converts colour spaces and performs normalization scales directly
in graphics memory, and the resulting tensor handle is emitted as a
`Frame { domain: VulkanTexture(..), caps: Caps::Tensor { .. }, .. }` submitted
straight to the inference backend.

`WgpuPreprocess` (`g2g-ml/src/wgpupreprocess.rs`, `wgpu` feature) is the
compute-shader half. An NV12 frame is converted and normalized in a wgpu compute
shader to a `Caps::Tensor { F32, [1,3,H,W], Nchw }`, the same contract
`OrtInference` builds on the CPU. The default system-memory variant uploads NV12
to a storage buffer and reads the f32 tensor back to `MemoryDomain::System`.

`with_gpu_output` instead leaves the tensor in a `wgpu::Buffer` and emits
`MemoryDomain::WgpuBuffer`, an on-device GPU-to-GPU copy into a fresh per-frame
buffer with no map or read-back in the element, so a downstream GPU consumer
reads it on-device and a CPU consumer pays the deferred read-back via the buffer
owner. This removes the output-side GPU-to-CPU copy. `WgpuInference` is the
consumer that binds the resulting buffer on-device, so `preprocess -> infer`
keeps the tensor on the GPU.

Surface-import input closes the other end. When the NV12 frame arrives already
GPU-resident as a `MemoryDomain::WgpuTexture`, a `WgpuNv12Texture` keep-alive
wrapping an R8Uint texture of `width x height*3/2` in standard NV12 byte layout,
the element adopts that texture's device and samples it with `textureLoad`
straight into the compute pass, with no CPU upload, bit-identical to the
storage-buffer path.

### DMA-BUF import

DMA-BUF import (`dmabuf-wgpu` feature, Linux) is the same idea for a
`MemoryDomain::DmaBuf` frame from a capture or decode path. The element opens a
device carrying `VK_KHR_external_memory_fd` and
`VK_EXT_external_memory_dma_buf` on the first such frame and binds the imported
buffer as the compute pass's input, sharing the importer
(`g2g_plugins::dmabufwgpu::DmaBufImporter`, which also honours a producer's
timeline semaphore) with the `dmabuftowgpu` element rather than repeating the
handshake.

The frame's row stride and plane offset reach the shader in the dims uniform, so
a padded capture buffer is read in place with no repack, and the tensor is
bit-identical to the same pixels uploaded from system memory, validated on an RTX
3060 including a padded stride. NV12 and packed YUYV both have a compute shader,
sharing every line but the fetch of one pixel's Y, Cb and Cr, YUYV because that
is what a UVC webcam captures, so a camera reaches the tensor with no
`videoconvert` in front.

Which GPU the import opens on is a real choice, not a default. A discrete GPU
binds only GPU-visible dma-bufs, while a CPU-backed one, udmabuf or a USB
webcam's capture buffer, binds on an integrated GPU, whose memory is the same
system RAM. `ImportAdapter`, the `import-adapter` property on both
`wgpupreprocess` and `dmabuftowgpu`, picks between them, `high-performance` by
default because that is what a GPU-exported dma-buf needs, and `integrated`
searches the enumerated Vulkan adapters by device type. Either way an fd the
driver cannot bind reports `UnsupportedDomain` and the caller falls back to the
upload path.

The live camera case is validated on this host
(`m993_camera_dmabuf_preprocess`): `v4l2src io-mode=dmabuf` at 640x480 YUYV into
`WgpuPreprocess` on the AMD integrated GPU gives the same tensor as that same
captured frame taken through the copy path, and the same frame on the RTX 3060
refuses the fd, which is the choice being real. With both ends GPU-resident,
`capture or decode -> WgpuPreprocess -> WgpuInference` runs with the pixels never
touching the CPU.

### CUDA and wgpu interop

`CudaToWgpu` (`g2g-plugins/src/cudawgpu.rs`) joins the NVDEC decode side to this
surface-import path. There is no portable call to share a CUDA pointer with wgpu,
so the bridge allocates an exportable Vulkan image
(`VK_KHR_external_memory_fd`, wrapped as a `wgpu::Texture` via wgpu-hal), CUDA
imports the same memory by FD with `cuImportExternalMemory` and copies the NVDEC
NV12 planes into it device to device, and the wgpu device travels on the frame's
keep-alive so `WgpuPreprocess` adopts it, the device-identity pattern. The whole
`NVDEC -> CudaToWgpu -> WgpuPreprocess -> WgpuInference` chain is validated on an
RTX 3060, matching a CPU reference with no PCIe download.

Shared images are recycled from a reuse pool. The Vulkan image, its CUDA import
and the `wgpu::Texture` are allocated once and returned to a free list when the
downstream frame is released, through a drop guard on the emitted keep-alive, so
per frame only the two device-to-device plane copies and a sync run. A recycled
entry is drained with `Device::poll` before reuse, since a wgpu submission may
still sample it. The pool cut the bridge step about 2.6x at 1080p, p50 0.38 ms
pooled against 0.98 ms per-frame-allocated.

`WgpuToCuda` closes the encode side. A renderer writes a packed-RGBA
`wgpu::Texture` on FD-exportable Vulkan memory (`export_rgba_image` and
`wrap_rgba_as_texture`, the `R8G8B8A8` mirror), CUDA imports it as a 4-channel
array, and `to_cuda_frame` copies it device to device into a linear
`CUdeviceptr` emitted as a `MemoryDomain::Cuda` `Rgba8` frame that `NvEnc`
registers as `ABGR`. So a GPU render reaches the H.264 encoder with no
device-to-host read-back, validated on an RTX 3060 by the `wgpu_to_cuda` test.

This is the zero-copy egress for server-side rendering and cloud gaming, and the
`bevy-g2g` crate's `RemoteRenderPlugins` is the packaged Bevy proof: Bevy renders
on the interop device, g2g copies the target through `WgpuToCuda`, and `NvEnc`
emits H.264 without a full-frame download, egressing to WHIP, WebRTC or a file.
Without an NVIDIA GPU the plugin falls back to a GPU-to-CPU readback plus libx264
encode so the same app streams on any adapter.

The crate completes the remote-rendering loop with a WebSocket input backchannel,
where viewer keyboard and mouse are injected as ordinary Bevy input messages,
since a WebRTC data channel cannot reach the publisher through a WHIP or WHEP
server with the viewer being a separate peer connection, and a windowed mode,
where the scene camera renders to the stream texture and the window shows it
through a fullscreen UI mirror, so desktop view and stream are the same pixels.
The bridge retains its own CUDA primary context, on the GPU the interop device
selects, and owns the exportable render-target texture.

## Inference backends

g2g avoids bundling heavy, unsafe proprietary C++ engines. The `g2g-ml` crate
provides wrapper elements targeting two execution paradigms.

### Burn

`g2g-ml::burn`, for embedded, wasm and RTOS, leverages the pure-Rust Burn
framework with a `wgpu` backend, compiling ONNX workflows into type-safe,
compile-time Rust graphics shaders. `BurnInference` (`g2g-ml/src/burninfer.rs`,
`burn` feature) is the wgpu-backend inference element over the `RawVideo` to
`Tensor` contract, driving an `input * W + b` linear layer on any Vulkan, Metal,
DX12 or WebGPU adapter.

An ONNX topology runs through the same element, but the import is build-time.
`burn-onnx`, what `burn-import` 0.21 forwards to, generates a burn `Module` plus
an embedded burnpack weight blob from the `.onnx` at compile time, so there is no
runtime graph loader to hand the file to.

The seam is the `BurnModule` trait: one forward pass from the `[1, 3, H, W]` NCHW
f32 tensor the element normalizes to `[1, N]` logits. The importing crate
implements it over its generated `Model<Wgpu>` and passes it to
`BurnInference::module`, which then drives it frame by frame exactly like the
built-in linear layer. A forward pass whose output is not the declared
`num_outputs` fails the frame, so the emitted `Caps::Tensor` cannot lie.

Because the codegen crate drags burn's whole dependency tree into any lockfile
that resolves it, the worked case is a workspace-excluded standalone crate,
`examples/g2g-onnx-import`: a
`Conv2d -> BatchNorm -> ReLU -> global average pool -> linear` graph whose logits
match the ONNX Runtime reference for the same frame on the RTX 3060.

Attention imports through the same seam. The standard-domain ONNX `Attention` op,
opset 23, one node for a whole multi-head block, is lowered by `burn-onnx` onto
`burn::tensor::module::attention`, so the GPU runs burn's own attention kernel
rather than a hand-unrolled matmul and softmax chain, validated on the 3060 by a
second fixture in that crate: pixels as a token sequence into multi-head
self-attention, mean pool, then linear. Because that node is opaque in the graph,
the fixture generator folds the attention formula in numpy and asserts ONNX
Runtime agrees before emitting the reference logits, so the reference is not ORT
agreeing with itself. This is the topology half of the Burn story, the
counterpart of the runtime `safetensors` weight import below.

### ONNX Runtime

`g2g-ml::ort`, for the high-performance server, wraps ONNX Runtime bindings to
pass underlying memory domains to hardware-specific execution paths (CUDA,
TensorRT, DirectML, Apple CoreML) natively. Each execution provider is a
constructor variant on `OrtInference` that registers the EP ahead of the CPU
fallback. Registration is best-effort, so the session keeps running on CPU when
the device is absent. On the desktop that is `from_memory_with_cuda` and
`from_memory_with_directml`.

On the Android edge there is `from_memory_with_nnapi`, the system NeuralNetworks
API over NPU, GPU or DSP, `from_memory_with_xnnpack`, ARM-optimized CPU, and
`from_memory_for_android`, which registers NNAPI then XNNPACK then the default
CPU EP in one call so ORT assigns each node to the first provider that supports
it, the MediaPipe delegate-with-fallback shape. The `nnapi` and `xnnpack`
features link symbols only the Android ONNX Runtime build carries, so they are
Android-target features that a host build or CI never enables. The EP stack is
validated on a device: `tools/android-nnapi-smoke.sh` runs
`g2g-ml/tests/android_nnapi_probe.rs` from `/data/local/tmp`, with a
binder-threadpool shim for the vendor NNAPI HAL, output byte-exact with the CPU
reference.

Edge TPU offload is proven. An int8 QDQ Conv-then-ReLU fixture run through
`from_memory_for_android` is placed on `NnapiExecutionProvider`, read from ORT's
profiling JSON, and on a Pixel 10a (Tensor G4) the DarwiNN HAL log confirms the
Edge TPU compiled and executed it, the `/dev/edgetpu core0` firmware load. The
float-typed input-boundary `QuantizeLinear` is the one op the TPU declines,
correctly delegated to CPU (`tools/android-nnapi-conv-smoke.sh`, which also greps
the `darwinn` logcat to disambiguate the TPU from other NNAPI accelerators).

A uint8-input variant of the model, with the boundary `QuantizeLinear` removed
and the graph input retyped to uint8, runs entirely on the TPU, every node on
`NnapiExecutionProvider` with nothing on the CPU, and the DarwiNN log confirms
`Ops supported = ..., not supported = 0` and
`compilation finished successfully on google-edgetpu`.

The f32 to uint8 quantization that feeds such a model is `TensorConvert`
(`g2g-plugins`), the tensor-domain sibling of `VideoConvert`. It quantizes an f32
tensor to int8 or uint8 (`q = round(x / scale) + zero_point`, clamped) or
dequantizes the inverse, with shape and layout passing through. So
`preprocess -> TensorConvert(quantize) -> inference` keeps the boundary quantize
out of the model, leaving the whole inference graph accelerator-eligible.
`TensorConvert` also transposes NCHW to NHWC and back and narrows and widens
between f32 and F16 in the same pass, so a model that wants `NHWC uint8`, as
NNAPI and TFLite do, is fed straight from an `NCHW f32` source. `OrtInference`
itself accepts the integer input: `from_session` reads the model's input element
type and `with_tensor_input` on a u8 or i8 model feeds the quantized tensor
straight to the session, while RGBA mode stays f32-only.

The whole chain is validated live on the device.
`Camera2Src -> TensorConvert(quantize) -> OrtInference(uint8)` runs a real camera
frame onto the Edge TPU (`tools/android-camera-tpu-smoke.sh`), and on a Pixel 10a
the logcat shows `accelerator name: EDGETPU` and
`compilation finished successfully on google-edgetpu`. This is the g2g answer to
an edge framework that moves inference between CPU and accelerator, demonstrated
end to end on real hardware.

The same constructor shape extends to the other vendor accelerators:
`from_memory_with_qnn` for the Qualcomm AI Engine Direct, the Hexagon NPU and
Adreno GPU on Snapdragon and the alternative to reaching the Hexagon through
NNAPI, and `from_memory_with_coreml` for the Apple Neural Engine and GPU on macOS
and iOS, each behind a target-only feature like `nnapi` that a host build never
links. Both compile for their target. This is the heterogeneous-device story: a
desktop NVIDIA box, a Windows D3D12 GPU, an Android phone NPU, and the Qualcomm
and Apple NPUs all run the same element with the EP picked per platform, the
architectural answer to MediaPipe's runtime CPU and GPU delegate switch.

### WgpuInference

`WgpuInference` (`g2g-ml/src/wgpuinfer.rs`, `wgpu` feature) is the GPU-resident
counterpart of `BurnInference`: a raw wgpu compute pass that binds the
GPU-resident tensor `WgpuPreprocess::with_gpu_output` produced directly, rather
than taking `RawVideo` in System memory and uploading.

It runs one of a small op zoo on that tensor, selected at construction, each its
own WGSL shader behind the shared device-adopt, dispatch and read-back
machinery:

- `linear`, the original `input * W + b` matmul
- `conv2d`, a same-padding stride-1 2D convolution over the `[1, Cin, H, W]` NCHW
  tensor with `[Cout, Cin, KH, KW]` weights, leaving a `[1, Cout, H, W]` feature
  map
- `relu` and `sigmoid`, the elementwise activations
- `maxpool2d` and `avgpool2d`, spatial pooling

The weighted ops, linear and conv2d, bind a 5-entry group of meta, input,
weights, bias and out, while the weightless ops, activation and pooling, bind a
3-entry group of meta, input and out, the bind-group layout following the active
shader.

The conv is the keystone that lets the chain run an actual CNN layer rather than
just a final classifier, the activation is the nonlinearity that keeps stacked
convs from collapsing to one linear map, and the pool is the spatial
downsampler. Chained GPU-resident as `conv2d -> relu -> maxpool`, each in
`with_gpu_output` mode so the data never leaves the device between layers, they
are a real small-CNN body, validated on the RTX 3060 against a CPU reference
folding the same ops (`conv2d_reference`, `relu_reference`,
`maxpool2d_reference`) over the exact tensor the GPU preprocess produced.

Trained weights are imported at runtime from a `safetensors` file via a
dependency-free reader (`g2g-ml::safetensors`, a focused parser for the format's
`u64` length plus JSON-subset header plus raw tensor bytes, with no `serde` and
no `safetensors` crate). `conv2d_from_safetensors` reads the
`[Cout, Cin, KH, KW]` weight and `[Cout]` bias by name and infers the kernel
dims, so picking a different trained checkpoint is parsing a different file while
the layer topology stays this compiled element. This is the weights half. Truly
dynamic graphs at runtime are the `ort` backend's job, and `burn-onnx` build-time
codegen is the Burn-side topology path.

It owns no device. Because a `wgpu::Buffer` is bindable only on the device that
created it, the element adopts the producer's device and queue, carried by the
incoming `WgpuBufferOwner`, on the first frame and submits its compute on the
producer's queue, which orders it after the producer's work with no fence or
read-back. The logits are read back to `MemoryDomain::System` by default or left
GPU-resident under `with_gpu_output` for a downstream GPU consumer. A burn or ort
consumer cannot do this zero-copy: their tensor handles are opaque, with no
foreign-buffer adopt, and run on their own device, so they would force the
GPU-to-CPU-to-GPU round-trip the GPU-resident preprocess and inference paths
exist to delete.

## Batching

`g2g-ml::batcher` provides a lock-free, multi-channel execution sink that groups
separate asynchronous video input streams into a single hardware tensor execution
array:

```
[ Camera Stream 1 ] ──► Async Channel ──┐
[ Camera Stream 2 ] ──► Async Channel ──┼─► [ Bounded Batcher ] ──► [ GPU Tensor Core ]
[ Camera Stream 3 ] ──► Async Channel ──┘     (Select / Timeout)
```

## Per-frame metadata

Inference output is only useful once it is structured and travels with the
picture.

The metadata system is `g2g-core::meta` (`metadata` feature). The `Frame` carries
a `FrameMetaSet`, a list of typed `FrameMeta` trait objects, the GstMeta analog,
with attach, typed-get and iterate plus a
`propagate(Transform) -> Propagation` survival contract, so a re-encode drops
pixel-derived meta while a scale, crop or copy keeps it. It is off by default, so
the RTOS baseline pays nothing and `FrameMetaSet` is a ZST.

The standard `AnalyticsMeta` is the `GstAnalyticsRelationMeta` analog: a relation
graph of `ObjectDetection`, `Classification` and `Tracking` nodes plus directed
edges, so a detector to tracker to classifier to overlay chain reads results by
node kind and traversal instead of re-deriving joins through tensor offsets.
Bounding boxes are normalized to `[0,1]`, so they survive a downstream resample
without a coordinate rewrite.

### Producers

`g2g-ml::DetectionPostprocess` (`analytics` feature) decodes a YOLOv8-style
`[1, 4+C, A]` output tensor, applying a confidence threshold and per-class NMS,
into `ObjectDetection`s, attaches an `AnalyticsMeta`, and forwards the frame.

`g2g-ml::OrtSegmentation` (`ort` plus `analytics`) runs a YOLO `-seg` export,
Ultralytics YOLOv8-seg or YOLO11-seg, and attaches `Segmentation` plus `Roi`
nodes to the frame it forwards, an identity transform that adds metadata, so the
picture and its masks reach an overlay together. Both of the model's outputs stay
inside the element, unlike the detection split of
`OrtInference -> DetectionPostprocess`, because a tensor frame carries one tensor
and a mask needs both the box-plus-coefficient output `[1, 4+C+M, A]` and the
prototype planes `[1, M, mh, mw]`.

An instance's mask is the coefficient-weighted prototype sum through a sigmoid,
read over the instance's box at prototype resolution, so a consumer places sample
`i` of `mask.width()` at `bbox.x + (i + 0.5) / mask.width() * bbox.w` and needs
nothing else. The `Roi` is the mask-tight sub-box, the region an encoder or
tracker should treat specially, related to its `Segmentation` by `Contains`. The
decode is pure Rust (`g2g-ml::segmentation`), so an `ort-web` caller in the
browser that already holds both outputs reuses it without an element.

### Metadata through fan-out and transforms

`FrameMetaSet` holds each `FrameMeta` as an `Arc<dyn FrameMeta>` and is `Clone`,
so a tee clone shares the analytics graph by refcount rather than dropping it: the
graph runner's `try_clone_packet` carries `frame.meta.clone()`, landing the same
`AnalyticsMeta` on both branches of a `decode -> tee -> {detect, video}` diamond.
Mutation is copy-on-write via `FrameMeta::clone_box`, the GstMeta `copy_func`
analog: `FrameMetaSet::get_mut` deep-copies a shared entry before the mutable
borrow, so a branch editing its analytics never aliases the sibling. It is still
a ZST no-op when the `metadata` feature is off.

Fan-out shares the same frame, so meta rides for free. A transform that emits a
new frame, videoscale, videoconvert, videocrop or a re-encode, would otherwise
drop it. An element declares `AsyncElement::meta_transform() -> Option<Transform>`
and when it returns `Some(t)` the graph runner clones the input frame's
`FrameMetaSet`, applies `propagate(t)`, and stashes the survivors on the transform
arm's output adapter, which attaches them to any outgoing `DataFrame` whose own
meta is empty. Element-authored meta is never overwritten, and a Drop verdict that
empties the set clears the stash so nothing stale leaks.

`None`, the default, opts out: a pass-through that forwards the same frame already
carries its meta, and an element that produces none wants nothing added. The stash
is recomputed per input frame, so association is exact for a 1-in-1-out transform
and most-recent-input for a pipelined one such as an encoder with lookahead. The
standard elements declare the obvious mapping: videoconvert `Copy`, videoscale
`Scale`, videocrop `Crop`, and the software video encoders (av1enc, vpxenc,
ffmpegenc, mjpegenc) `Encode`. It is still a no-op when the `metadata` feature is
off, since the method and stash are cfg'd out and the baseline build is
byte-identical.

### Metadata on demand

`meta_transform` moves metadata that already exists.
`AsyncElement::meta_requests() -> MetaRequests` is how a consumer says which
metadata it wants to exist in the first place, the GStreamer allocation-query
`add_meta` analog. `MetaRequests` is a fixed-capacity `Copy` set of
`(TypeId, RequestPolicy)` entries, four of them, sorted so equality is
order-independent, carried as a field of `AllocationParams`, so the demand travels
on the allocation cascade that already runs sink to source. A producer reads
`params.meta_requests.wants::<T>()` in `configure_allocation` and can then skip
work nobody reads.

Downstream demand also crosses a fan-in's output boundary, where its pool
parameters deliberately do not, because a compositor writes the frames the demand
describes. An element with no pool requirement of its own still forwards the
demand as `AllocationParams::meta_demand`, which accepts every memory domain and
which the source-side reconciliation skips, so a metadata request can never decide
a producer's memory domain. A request is a hint, never a guarantee, so a consumer
still handles a frame arriving without the meta. With nothing declared the cascade
is byte-identical to before, and without the `metadata` feature `MetaRequests` is
a ZST empty set.

Every request carries a `RequestPolicy`, because two kinds of metadata combine
differently when several consumers read one producer's frames. `AnyConsumer`, the
default from `request::<T>()`, covers metadata whose attachment costs a consumer
that did not ask nothing, so one asking consumer is enough and the demands union:
`AnalyticsMeta`, `CaptionMeta`, `TimecodeMeta`. `EveryConsumer`, from
`request_from_every_consumer::<T>()`, covers metadata whose honouring changes the
buffer, so a consumer that did not ask would misread it, and the demand only
stands where every consumer asked.

Two folds implement this: `join_branches` at a tee, where a branch that proposed
nothing is still a branch that asked for nothing and vetoes, and `carry_upstream`
at each hop, where the producer's frames pass through that element first so a hop
that does not share the request vetoes it exactly as a sibling branch does. The
strictest policy wins when two elements ask for one meta differently. A demand
that dies leaves the cascade as it found it: a proposal carrying neither pool
constraints nor demand collapses back to none.

### PlaneLayout

`PlaneLayout` is the first meta produced on demand and the `GstVideoMeta` analog:
per-plane byte offset and row stride, up to four planes with every derived offset
checked, for a raw frame whose rows are padded. Without it a raw frame is assumed
tightly packed, so a producer whose rows are not, a GPU readback at the API's
256-byte row alignment or a capture driver's `bytesperline`, has to repack them
row by row.

`WgpuCompositor` asks `wants::<PlaneLayout>()` when the cascade configures its
output: when a consumer downstream requested one it hands over the canvas as the
GPU wrote it and declares the pitch, and the per-frame repack disappears.
`V4l2Src`, `PipeWireVideoSrc` and `VaapiDec` ask the same question of the same
cascade. The capture sources hand over the driver's or daemon's mapped buffer at
its `bytesperline`, deriving each later plane's stride from plane 0's, the same
pitch for NV12's interleaved chroma and half of it for I420's, and the VAAPI
decoder copies the surface out at its own `y_pitch` and `uv_pitch` in one pass per
plane rather than row by row.

A frame in a dma-buf carries the layout whether or not anyone asked, since its
rows sit at the producer's pitch either way and a consumer that maps the buffer
has no other way to find them. `VideoConvert` is that consumer: it requests the
layout and reads a packed RGBA or BGRA input's rows where they lie, and a padded
planar input it packs out first, which is correct and costs what the producer
skipped.

It is the `EveryConsumer` request the policy above exists for. `VideoConvert` asks
with `request_from_every_consumer`, so any consumer or hop that would take the
padded rows for tightly packed ones vetoes the padding and the producer repacks as
it always did. The meta is dropped by every `meta_transform`, since an element
only declares one when it writes a new buffer, and a tee's clone shares the
described buffer and keeps it.

### OrientationMeta

`OrientationMeta` is one member of `Orientation`, the dihedral group of the square
(four rotations, four mirrors, with `compose`, `inverse` and `swaps_dims`), saying
how the buffer as stored has to be turned to be shown the right way up. It exists
so a rotation upstream of a display sink that can turn a picture for free does not
have to remap every pixel. The turn is relative to the buffer, so a consumer
working in display coordinates applies it itself. It survives a scale or a colour
convert (`Propagation::Keep`) and dies under a crop, whose rectangle is chosen in
the coordinates the turn has not been applied to yet.

A sink that can apply it says so with `AsyncElement::absorbs_orientation`, and
each runner's sink arm sends `Reconfigure::AbsorbOrientation` up that sink's input
link while the arms are still being wired, before any frame is pulled. The
advertisement travels the reverse channel the same way a keyframe request does,
but the relay decision is per variant (`ReconfigureAnswered`): a transform relays
it toward the source unless `handles_orientation` says it answers the signal
itself. `VideoFlip` answers it, being the element the advertisement is aimed at,
and `VideoCrop` answers it to stop it, since a crop rectangle means something else
once the picture is turned. Relaying costs one push per hop, so a transform
between the flip and the sink delays the switch by the frames already in flight.
Nothing is lost, they arrive rotated.

`VideoFlip` sees the advertisement as `PushOutcome::Reconfigure` from its own
push. The pre-send check holds that packet back rather than enqueuing it, so the
flip re-announces the output caps, now the input's with no dimension swap, and
sends the packet again as a descriptor: the buffer goes through in the same memory
domain with `OrientationMeta` attached, composed with any turn already on the
input. Negotiation still solves for the swapped shape, which the runtime
`CapsChanged` corrects on the mid-stream re-solve path.

Two rules follow from the hold-back. A `Reconfigure::AbsorbOrientation` only ever
surfaces from the pre-send check, since a post-send one is held for the next push
or the producer would resend a packet that already crossed, and an `Eos` is never
held back at all, since nothing sends one twice.

`WaylandSink` is the sink that absorbs today. It maps the descriptor to a
`wl_surface::set_buffer_transform` argument, re-issued only when the turn changes,
and swaps the window's size hint for a turn that transposes the picture, with the
buffer and its damage staying in buffer coordinates. The argument is the inverse
of the descriptor, because `set_buffer_transform` names the transform already
applied to the buffer and the compositor applies its inverse. `KmsSink` and
`CudaKmsSink` do not advertise, so a flip in front of them keeps realizing the
rotation.

## The analytics overlay

The visible end of the detector chain reads the `AnalyticsMeta` carried onto the
display frame through the fan-out path and draws it, so
`decode -> tee -> {detect, video} -> overlay -> display` works.

There are three shapes in one shared palette: a detection box as a solid outline
in its class colour, an instance segmentation as a translucent fill of its mask
under `mask-alpha`, and a region of interest as a dashed rectangle. A mask spans
exactly its instance's box at the model's own grid resolution, which is the whole
placement rule either backend needs, and an ROI takes the palette slot of the
segmentation that `Contains` it, so a mask and its tight box read as one instance
rather than two findings.

Two backends. The CPU `g2g-plugins::analyticsoverlay::AnalyticsOverlay`
(`analytics` feature) paints onto RGBA8 with the compositor's integer source-over
blend, the `no_std` baseline. The GPU `vellooverlay::VelloAnalyticsOverlay`
(`vello-overlay` feature) strokes antialiased boxes and scales each mask on as an
alpha image fill over a full-frame image with the Vello GPU 2D renderer, emitting
the result in the `MemoryDomain::WgpuTexture` domain.

That domain, an `OwnedWgpuTexture` whose `wgpu::Texture` lives in a
`WgpuKeepAlive` owner since `g2g-core` never links wgpu, is the render-side analog
of the decode-side CUDA and D3D11 texture domains: the rendered frame stays on the
GPU with no readback, so a GPU sink presents it directly.

## The GPU sink

`g2g-plugins::wgpusink::WgpuSink` (`wgpu-sink`) is that consumer. It presents a
`WgpuTexture` frame by sampling it in a small fullscreen blit pass onto its
target, an owned offscreen texture for render-to-texture and screenshots, or a
caller-built `wgpu::Surface` for an on-screen window, again with no readback.

Because a wgpu texture is bound to its device, the overlay and the sink share one
device through a cloneable `gpu::GpuContext`, the overlay's `with_context` and the
sink's constructors, and the producer's texture is recovered by the sink through
the shared `gpu::WgpuTextureKeepAlive` type. This closes the analytics path end to
end: `decode -> tee -> {detect, video} -> overlay -> WgpuSink`, detections
rendered on the GPU reaching the display with no system-memory round-trip.

Window and event-loop ownership stay with the application, since wgpu surfaces are
built from a window handle and must drive the app's event loop, so the sink
presents to a surface the app supplies rather than opening its own window. The app
also owns the resize event and forwards it as `WgpuSink::resize(width, height)`,
which reconfigures the swapchain or reallocates the offscreen texture at the new
size. The frame's negotiated geometry is untouched and the blit just scales it to
whatever the target now is.

### Bring your own device

The same `GpuContext` sharing extends one step further out, to an embedding
application that already owns a `wgpu::Device`, a game engine, a Bevy or Tauri
app, or an editor's renderer. `GpuContext::from_wgpu(instance, adapter, device,
queue)` wraps the embedder's device instead of opening one, so every GPU element
produces textures on that device.

A decoded frame's `MemoryDomain::WgpuTexture` is then a first-class object in the
embedder's own render graph, recovered with `gpu::texture_of` and bindable
directly, sampled onto a 3D surface or composited in the UI, with no second
device, no surface hand-off and no copy, the opposite of `for_surface` where g2g
opens the device. This is the integration path for the lightweight-app and engine
use case where the application drives rendering and g2g is just the pipeline that
hands it textures, validated on the RTX 3060 where a texture produced through a
`from_wgpu` context reads back correctly on the embedder's own device handles. The
frame still flows to the app through any sink, including the `appsink` pull
channel, which carries a GPU-domain `Frame` unchanged.

The `bevy-g2g` crate's `VideoPlayerPlugin` is the packaged proof. A stock Bevy
app's render device is adopted into `from_wgpu` in the plugin's `finish`, a
`filesrc -> h264parse -> ffmpegdec -> videoconvert -> vello overlay -> appsink`
pipeline lands each decoded frame in a `wgpu::Texture` on Bevy's device, and the
plugin registers it as a render-world `GpuImage` and binds it to the material of
every `VideoScreen`-tagged mesh, through an sRGB view, the overlay's texture
listing `Rgba8UnormSrgb` in `view_formats` for exactly this. It is the mirror of
the crate's streaming side, which renders in Bevy and encodes in g2g.

### Presenting on the producer's device

A GPU decoder cannot be handed a device: Vulkan Video decode needs queues and
extensions wgpu never asks for, so `VulkanVideoDec` opens its own, and its
textures bind to no other. A launch line has no application to pass a
`GpuContext` between the two, so the decoder publishes its own
(`gpu::publish_producer_context`, only when the device's swapchain extension is
enabled) and a windowed sink builds its surface on that instance and presents from
that device (`gpu::present_on_producer_device`, shared by every wgpu display
sink), falling back to opening its own device when nothing published or the
published one cannot drive this display.

The decode device is opened once per codec and kept across the repeated
`configure_pipeline` a launch line does, since a second device would leave the
sink presenting from one nothing produces on. So `filesrc ! decodebin ! wgpusink`
decodes on the GPU and presents the frame where it already lies, with no
application code.
