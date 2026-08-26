// Shared scaffolding for the dma-buf -> WgpuPreprocess tests (included via
// `include!` by m990 and m993; not a test crate itself, so plain comments only).
// Needs `dmabuf-wgpu`, and the including file to be Linux-gated.


use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, OutputSink, PushOutcome, Rate, RawVideoFormat,
};
use g2g_ml::wgpupreprocess::WgpuPreprocess;
use g2g_plugins::dmabufwgpu::ImportAdapter;
use g2g_plugins::wgpudmabuf::WgpuToDmaBuf;

// parallel per-test device creation intermittently segfaults in the NVIDIA driver
// (the recorded wgpu gotcha), so each GPU test takes this for its whole body.
static GPU_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

impl Collect {
    fn frame(&mut self) -> Frame {
        self.packets
            .drain(..)
            .find_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .expect("element pushed a tensor frame")
    }
}

fn raw_caps(format: RawVideoFormat, w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

fn raw_geometry(caps: &Caps) -> (RawVideoFormat, u32, u32) {
    match caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } => (*format, *w, *h),
        other => panic!("not fixed raw video caps: {other:?}"),
    }
}

fn frame_f32(frame: &Frame) -> Vec<f32> {
    let slice = frame
        .domain
        .as_system_slice()
        .expect("tensor frame is System memory");
    slice
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

// Export `bytes` as a GPU-allocated dma-buf frame at `caps`: the stand-in capture
// producer. The export element derives the row stride from the caps width, so
// passing a wider caps than the picture yields a padded frame. `None` when no GPU
// can export.
async fn export_dmabuf(bytes: &[u8], caps: &Caps) -> Option<Frame> {
    let (format, width, _) = raw_geometry(caps);
    let mut export = WgpuToDmaBuf::new();
    let (device, queue) = export.gpu().await.ok()?;
    let source = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dmabuf-export-src"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&source, 0, bytes);

    export.configure_pipeline(caps).expect("export configure");
    let mut sink = Collect::default();
    export
        .process(
            PipelinePacket::DataFrame(Frame {
                domain: MemoryDomain::WgpuBuffer(WgpuToDmaBuf::wrap_buffer(
                    &device,
                    source,
                    bytes.len(),
                )),
                timing: FrameTiming {
                    pts_ns: 7_000,
                    ..FrameTiming::default()
                },
                sequence: 3,
                meta: Default::default(),
            }),
            &mut sink,
        )
        .await
        .expect("export process");
    let frame = sink.frame();
    let MemoryDomain::DmaBuf(dmabuf) = &frame.domain else {
        panic!("export produced a dma-buf frame");
    };
    assert_eq!(
        dmabuf.stride,
        format.row_stride(width).expect("exportable format"),
        "export used the caps stride"
    );
    Some(frame)
}

// Run a dma-buf frame through `WgpuPreprocess` configured for `caps`, importing on
// `adapter`. `None` when the driver cannot bind the fd (skip), which the element
// reports as `UnsupportedDomain`.
async fn preprocess_dmabuf(
    frame: Frame,
    caps: &Caps,
    adapter: ImportAdapter,
    gpu_output: bool,
) -> Option<Frame> {
    let mut element = WgpuPreprocess::new().with_import_adapter(adapter);
    if gpu_output {
        element = element.with_gpu_output();
    }
    element.configure_pipeline(caps).expect("preprocess configure");
    let mut sink = Collect::default();
    match element
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
    {
        Ok(()) => {}
        Err(G2gError::UnsupportedDomain) => {
            eprintln!("SKIP: dma-buf import unsupported on this driver / adapter");
            return None;
        }
        Err(e) => panic!("dma-buf preprocess failed: {e:?}"),
    }
    assert_eq!(element.emitted(), 1, "one tensor per frame");
    Some(sink.frame())
}

// The same pixels through the System upload path, for a bit-exact comparison
// against the import path (same shader, so any difference is the import's).
// `pixels` must be tightly packed at `caps`'s geometry.
async fn preprocess_system(caps: &Caps, pixels: Vec<u8>) -> Vec<f32> {
    let mut element = WgpuPreprocess::new();
    element.configure_pipeline(caps).expect("preprocess configure");
    let mut sink = Collect::default();
    element
        .process(
            PipelinePacket::DataFrame(Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(pixels.into_boxed_slice())),
                timing: FrameTiming::default(),
                sequence: 0,
                meta: Default::default(),
            }),
            &mut sink,
        )
        .await
        .expect("system preprocess");
    frame_f32(&sink.frame())
}

// Repack a frame laid out at `stride` bytes per row into the tight layout the host
// reference and the System upload path take. `rows` is every row the format
// carries (NV12's chroma region included).
fn repack_tight(padded: &[u8], stride: usize, row_bytes: usize, rows: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(row_bytes * rows);
    for row in 0..rows {
        out.extend_from_slice(&padded[row * stride..row * stride + row_bytes]);
    }
    out
}
