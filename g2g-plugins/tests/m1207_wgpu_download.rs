#![cfg(feature = "wgpu-sink")]

use std::sync::Arc;

use g2g_core::memory::{MemoryDomain, OwnedWgpuBuffer, WgpuBufferKeepAlive};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, Caps, Dim, Frame, FrameTiming, G2gError, OutputSink, PipelineClock,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::appsink::{register_appsink_pull, Pull};
use g2g_plugins::appsrc::register_appsrc;
use g2g_plugins::registry::default_registry;
use g2g_plugins::wgpudownload::WgpuDownload;

// 37 RGBA pixels are 148 bytes, short of the 256-byte row a texture copy pads to.
const WIDTH: u32 = 37;
const HEIGHT: u32 = 5;
const RGBA_BYTES_PER_PIXEL: usize = 4;
const OPAQUE: u8 = 255;

// Opening wgpu devices concurrently crashes some drivers.
static GPU_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn has_adapter() -> bool {
    wgpu::Instance::default()
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .is_ok()
}

fn opaque_pattern() -> Vec<u8> {
    let len = WIDTH as usize * HEIGHT as usize * RGBA_BYTES_PER_PIXEL;
    (0..len)
        .map(
            |i| match i % RGBA_BYTES_PER_PIXEL == RGBA_BYTES_PER_PIXEL - 1 {
                true => OPAQUE,
                false => (i.wrapping_mul(7).wrapping_add(3)) as u8,
            },
        )
        .collect()
}

fn rgba_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

async fn through_gpu_compositor(name: &str, tail: &str, pixels: &[u8]) -> Frame {
    let in_channel = format!("{name}_in");
    let out_channel = format!("{name}_out");
    let feed = register_appsrc(&in_channel);
    assert!(feed.push(pixels, 0));
    feed.end_of_stream();
    let pull = register_appsink_pull(&out_channel);

    let reg = default_registry();
    let line = format!(
        "appsrc channel={in_channel} caps=video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},framerate=30/1 \
         ! wgpucompositor width={WIDTH} height={HEIGHT} gpu-output=true ! {tail} channel={out_channel}"
    );
    let graph = parse_launch(&reg, &line).expect("parses");
    run_graph(graph, &ZeroClock, 4).await.expect("runs");
    match pull.try_pull() {
        Pull::Frame(frame) => frame,
        other => panic!("no frame reached the appsink: {other:?}"),
    }
}

fn system_bytes(frame: &Frame) -> &[u8] {
    frame
        .domain
        .as_system_slice()
        .unwrap_or_else(|| panic!("expected system memory, got {:?}", frame.domain.kind()))
}

#[tokio::test]
async fn texture_reads_back_with_row_padding_stripped() {
    let _gpu = GPU_LOCK.lock().await;
    if !has_adapter().await {
        eprintln!("no wgpu adapter; skipping");
        return;
    }
    let pixels = opaque_pattern();
    let frame =
        through_gpu_compositor("download_explicit", "wgpudownload ! appsink", &pixels).await;
    assert_eq!(system_bytes(&frame), pixels.as_slice());
}

#[tokio::test]
async fn auto_plug_downloads_for_a_dmabuf_or_system_sink() {
    let _gpu = GPU_LOCK.lock().await;
    if !has_adapter().await {
        eprintln!("no wgpu adapter; skipping");
        return;
    }
    let pixels = opaque_pattern();
    let frame = through_gpu_compositor(
        "download_spliced",
        "appsink input-domains=dmabuf,system",
        &pixels,
    )
    .await;
    assert_eq!(system_bytes(&frame), pixels.as_slice());
}

#[tokio::test]
async fn auto_plug_downloads_through_a_chain_of_capsfilters() {
    let _gpu = GPU_LOCK.lock().await;
    if !has_adapter().await {
        eprintln!("no wgpu adapter; skipping");
        return;
    }
    let pixels = opaque_pattern();
    let rgba = format!("video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT}");
    let frame = through_gpu_compositor(
        "download_through_capsfilters",
        &format!("capsfilter caps={rgba} ! {rgba} ! appsink input-domains=system"),
        &pixels,
    )
    .await;
    assert_eq!(system_bytes(&frame), pixels.as_slice());
}

#[derive(Default)]
struct Capture {
    frame: Option<Frame>,
}

impl OutputSink for Capture {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
            self.frame = Some(frame);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

async fn download(domain: MemoryDomain) -> Result<Frame, G2gError> {
    let mut element = WgpuDownload::new();
    element.configure_pipeline(&rgba_caps())?;
    let mut capture = Capture::default();
    element
        .process(
            PipelinePacket::DataFrame(Frame::new(domain, FrameTiming::default(), 0)),
            &mut capture,
        )
        .await?;
    Ok(capture.frame.expect("the element pushed a frame"))
}

#[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
async fn plain_buffer(pixels: &[u8]) -> Option<OwnedWgpuBuffer> {
    use g2g_plugins::wgpudmabuf::WgpuToDmaBuf;
    let (device, queue) = WgpuToDmaBuf::new().gpu().await.ok()?;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("download-source"),
        size: pixels.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, pixels);
    Some(WgpuToDmaBuf::wrap_buffer(
        &device,
        &queue,
        buffer,
        pixels.len(),
    ))
}

#[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
#[tokio::test]
async fn plain_buffer_reads_back() {
    let _gpu = GPU_LOCK.lock().await;
    let pixels = opaque_pattern();
    let Some(buffer) = plain_buffer(&pixels).await else {
        eprintln!("no Vulkan export device; skipping");
        return;
    };
    let frame = download(MemoryDomain::WgpuBuffer(buffer))
        .await
        .expect("downloads");
    assert_eq!(system_bytes(&frame), pixels.as_slice());
}

#[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
#[tokio::test]
async fn imported_dmabuf_buffer_reads_back() {
    use g2g_plugins::dmabufwgpu::DmaBufToWgpu;
    use g2g_plugins::wgpudmabuf::WgpuToDmaBuf;

    let _gpu = GPU_LOCK.lock().await;
    let pixels = opaque_pattern();
    let mut export = WgpuToDmaBuf::new();
    let Ok((device, queue)) = export.gpu().await else {
        eprintln!("no Vulkan export device; skipping");
        return;
    };
    let source = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("export-source"),
        size: pixels.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&source, 0, &pixels);
    export
        .configure_pipeline(&rgba_caps())
        .expect("export configures");
    let mut exported = Capture::default();
    export
        .process(
            PipelinePacket::DataFrame(Frame::new(
                MemoryDomain::WgpuBuffer(WgpuToDmaBuf::wrap_buffer(
                    &device,
                    &queue,
                    source,
                    pixels.len(),
                )),
                FrameTiming::default(),
                0,
            )),
            &mut exported,
        )
        .await
        .expect("exports");

    let mut import = DmaBufToWgpu::new();
    import
        .configure_pipeline(&rgba_caps())
        .expect("import configures");
    let mut imported = Capture::default();
    import
        .process(
            PipelinePacket::DataFrame(exported.frame.expect("exported a frame")),
            &mut imported,
        )
        .await
        .expect("imports");
    let gpu_frame = imported.frame.expect("imported a frame");
    assert!(matches!(gpu_frame.domain, MemoryDomain::WgpuBuffer(_)));

    let frame = download(gpu_frame.domain).await.expect("downloads");
    assert_eq!(system_bytes(&frame), pixels.as_slice());
}

#[derive(Debug)]
struct ForeignOwner;

impl WgpuBufferKeepAlive for ForeignOwner {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

#[tokio::test]
async fn unknown_owner_fails_loud() {
    let domain = MemoryDomain::WgpuBuffer(OwnedWgpuBuffer::new(
        RGBA_BYTES_PER_PIXEL,
        Arc::new(ForeignOwner),
    ));
    let result = download(domain).await;
    assert!(matches!(result, Err(G2gError::UnsupportedDomain)));
}
