#![cfg(all(target_os = "linux", feature = "wgpu-sink", feature = "dmabuf-wgpu"))]

use g2g_core::memory::{MemoryDomain, OwnedDmaBuf, OwnedWgpuBuffer};
use g2g_core::meta::{Plane, PlaneLayout};
use g2g_core::{
    AsyncElement, Caps, Dim, Frame, FrameTiming, G2gError, OutputSink, PipelinePacket, PushOutcome,
    Rate, RawVideoFormat,
};
use g2g_plugins::dmabufwgpu::DmaBufToWgpu;
use g2g_plugins::wgpudmabuf::WgpuToDmaBuf;
use g2g_plugins::wgpudownload::WgpuDownload;

const WIDTH: u32 = 6;
const HEIGHT: u32 = 4;
// 3 rows of 6 bytes is not a whole number of 4-byte words
const UNALIGNED_HEIGHT: u32 = 2;
const ODD_WIDTH: u32 = 37;
const ODD_HEIGHT: u32 = 5;
const ROW_PADDING: usize = 3;
const NV12_STRIDE: usize = 16;
const P010_STRIDE: usize = 32;
const LEADING_BYTES: usize = 64;
const PADDING: u8 = 0xee;

// Opening wgpu devices concurrently crashes some drivers.
static GPU_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

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

fn caps(format: RawVideoFormat, width: u32, height: u32) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

// A semi-planar frame: luma rows, then half as many interleaved chroma rows.
struct SemiPlanar {
    format: RawVideoFormat,
    width: u32,
    stride: usize,
    leading: usize,
    height: u32,
}

impl SemiPlanar {
    // `(row bytes, rows)` of the luma and the chroma plane
    fn planes(&self) -> [(usize, usize); 2] {
        [0, 1].map(|plane| {
            (
                self.format.plane_stride(plane, self.width).unwrap() as usize,
                self.format.plane_rows(plane, self.height).unwrap() as usize,
            )
        })
    }

    fn caps(&self) -> Caps {
        caps(self.format, self.width, self.height)
    }

    fn layout(&self) -> PlaneLayout {
        let chroma_offset = self.leading + self.stride * self.planes()[0].1;
        PlaneLayout::new(&[
            Plane {
                offset: self.leading,
                stride: self.stride,
            },
            Plane {
                offset: chroma_offset,
                stride: self.stride,
            },
        ])
        .expect("two planes")
    }

    fn padded_and_tight(&self) -> (Vec<u8>, Vec<u8>) {
        let mut padded = vec![PADDING; self.leading];
        let mut tight = Vec::new();
        for (plane, (row_bytes, rows)) in self.planes().into_iter().enumerate() {
            for row in 0..rows {
                let pixels: Vec<u8> = (0..row_bytes)
                    .map(|column| (plane * 101 + row * 13 + column * 7 + 1) as u8)
                    .collect();
                padded.extend_from_slice(&pixels);
                padded.resize(padded.len() + self.stride - row_bytes, PADDING);
                tight.extend_from_slice(&pixels);
            }
        }
        (padded, tight)
    }
}

const NV12: SemiPlanar = SemiPlanar {
    format: RawVideoFormat::Nv12,
    width: WIDTH,
    stride: NV12_STRIDE,
    leading: 0,
    height: HEIGHT,
};

const P010: SemiPlanar = SemiPlanar {
    format: RawVideoFormat::P010,
    width: WIDTH,
    stride: P010_STRIDE,
    leading: 0,
    height: HEIGHT,
};

async fn push_through(
    element: &mut impl AsyncElement,
    frame_caps: &Caps,
    domain: MemoryDomain,
) -> Result<Frame, G2gError> {
    element.configure_pipeline(frame_caps)?;
    let mut capture = Capture::default();
    element
        .process(
            PipelinePacket::DataFrame(Frame::new(domain, FrameTiming::default(), 0)),
            &mut capture,
        )
        .await?;
    Ok(capture.frame.expect("the element pushed a frame"))
}

fn system_bytes(frame: &Frame) -> Vec<u8> {
    frame
        .domain
        .as_system_slice()
        .unwrap_or_else(|| panic!("expected system memory, got {:?}", frame.domain.kind()))
        .to_vec()
}

fn source_buffer(device: &wgpu::Device, bytes: &[u8]) -> wgpu::Buffer {
    // A buffer cannot be created mapped at a size wgpu could not copy whole.
    let size = (bytes.len() as u64).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("padded-source"),
        size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    buffer
        .slice(..)
        .get_mapped_range_mut()
        .slice(..bytes.len())
        .copy_from_slice(bytes);
    buffer.unmap();
    buffer
}

async fn download_laid_out(shape: SemiPlanar) -> Option<(Vec<u8>, Vec<u8>)> {
    let (device, queue) = WgpuToDmaBuf::new().gpu().await.ok()?;
    let (padded, tight) = shape.padded_and_tight();
    let buffer = source_buffer(&device, &padded);
    let owned = WgpuToDmaBuf::wrap_buffer(&device, &queue, buffer, padded.len())
        .with_plane_layout(shape.layout());
    let frame = push_through(
        &mut WgpuDownload::new(),
        &shape.caps(),
        MemoryDomain::WgpuBuffer(owned),
    )
    .await
    .expect("downloads");
    Some((system_bytes(&frame), tight))
}

#[tokio::test]
async fn padded_nv12_buffer_downloads_tight() {
    let _gpu = GPU_LOCK.lock().await;
    let shape = SemiPlanar {
        leading: LEADING_BYTES,
        ..NV12
    };
    let Some((downloaded, tight)) = download_laid_out(shape).await else {
        eprintln!("no Vulkan export device; skipping");
        return;
    };
    assert_eq!(downloaded, tight);
}

#[tokio::test]
async fn padded_p010_buffer_downloads_tight() {
    let _gpu = GPU_LOCK.lock().await;
    let shape = SemiPlanar {
        leading: LEADING_BYTES,
        ..P010
    };
    let Some((downloaded, tight)) = download_laid_out(shape).await else {
        eprintln!("no Vulkan export device; skipping");
        return;
    };
    assert_eq!(downloaded, tight);
}

#[tokio::test]
async fn padded_odd_width_semi_planar_buffers_download_every_chroma_pair() {
    let _gpu = GPU_LOCK.lock().await;
    for format in [RawVideoFormat::Nv12, RawVideoFormat::P010] {
        let shape = SemiPlanar {
            format,
            width: ODD_WIDTH,
            stride: format.plane_stride(1, ODD_WIDTH).unwrap() as usize + ROW_PADDING,
            leading: LEADING_BYTES,
            height: ODD_HEIGHT,
        };
        let Some((downloaded, tight)) = download_laid_out(shape).await else {
            eprintln!("no Vulkan export device; skipping");
            return;
        };
        assert_eq!(
            Some(tight.len() as u64),
            format.unpadded_frame_bytes(ODD_WIDTH, ODD_HEIGHT),
            "{format:?}"
        );
        assert_eq!(downloaded, tight, "{format:?}");
    }
}

// NV12 one stride wide has exactly the padded frame's bytes.
async fn padded_dmabuf(padded: &[u8], stride: usize) -> Option<OwnedDmaBuf> {
    let mut export = WgpuToDmaBuf::new();
    let (device, queue) = export.gpu().await.ok()?;
    let rows = padded.len() / stride;
    let luma_rows = rows * 2 / 3;
    let buffer = source_buffer(&device, padded);
    let frame = push_through(
        &mut export,
        &caps(RawVideoFormat::Nv12, stride as u32, luma_rows as u32),
        MemoryDomain::WgpuBuffer(WgpuToDmaBuf::wrap_buffer(
            &device,
            &queue,
            buffer,
            padded.len(),
        )),
    )
    .await
    .expect("exports");
    let MemoryDomain::DmaBuf(dmabuf) = frame.domain else {
        panic!("the export did not emit a dma-buf");
    };
    assert_eq!(dmabuf.stride as usize, stride);
    Some(dmabuf)
}

async fn import_and_download(shape: SemiPlanar) -> Option<(Vec<u8>, Vec<u8>)> {
    let (padded, tight) = shape.padded_and_tight();
    let dmabuf = padded_dmabuf(&padded, shape.stride).await?;
    let frame_caps = shape.caps();
    let imported = push_through(
        &mut DmaBufToWgpu::new(),
        &frame_caps,
        MemoryDomain::DmaBuf(dmabuf),
    )
    .await
    .expect("imports");
    let MemoryDomain::WgpuBuffer(owned) = &imported.domain else {
        panic!("the import did not emit a wgpu buffer");
    };
    assert_eq!(owned.plane_layout(), Some(&shape.layout()));
    let frame = push_through(&mut WgpuDownload::new(), &frame_caps, imported.domain)
        .await
        .expect("downloads");
    Some((system_bytes(&frame), tight))
}

#[tokio::test]
async fn padded_nv12_dmabuf_imports_and_downloads_tight() {
    let _gpu = GPU_LOCK.lock().await;
    let Some((downloaded, tight)) = import_and_download(NV12).await else {
        eprintln!("no Vulkan export device; skipping");
        return;
    };
    assert_eq!(downloaded, tight);
}

#[tokio::test]
async fn padded_p010_dmabuf_imports_and_downloads_tight() {
    let _gpu = GPU_LOCK.lock().await;
    let Some((downloaded, tight)) = import_and_download(P010).await else {
        eprintln!("no Vulkan export device; skipping");
        return;
    };
    assert_eq!(downloaded, tight);
}

#[tokio::test]
async fn imported_buffer_exports_on_the_import_device_in_its_layout() {
    let _gpu = GPU_LOCK.lock().await;
    let (padded, tight) = NV12.padded_and_tight();
    let Some(dmabuf) = padded_dmabuf(&padded, NV12.stride).await else {
        eprintln!("no Vulkan export device; skipping");
        return;
    };
    let frame_caps = caps(RawVideoFormat::Nv12, WIDTH, HEIGHT);
    let mut import = DmaBufToWgpu::new();
    let imported = push_through(&mut import, &frame_caps, MemoryDomain::DmaBuf(dmabuf))
        .await
        .expect("imports");

    let mut export = WgpuToDmaBuf::new();
    let exported = push_through(&mut export, &frame_caps, imported.domain)
        .await
        .expect("exports from the import device");
    let MemoryDomain::DmaBuf(reexported) = exported.domain else {
        panic!("the export did not emit a dma-buf");
    };
    assert_eq!(reexported.stride as usize, NV12.stride);
    assert_eq!(reexported.offset, 0);

    let reimported = push_through(
        &mut DmaBufToWgpu::new(),
        &frame_caps,
        MemoryDomain::DmaBuf(reexported),
    )
    .await
    .expect("re-imports");
    let frame = push_through(&mut WgpuDownload::new(), &frame_caps, reimported.domain)
        .await
        .expect("downloads");
    assert_eq!(system_bytes(&frame), tight);
}

async fn plain_vulkan_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .ok()?;
    adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .ok()
}

fn tight_nv12_frame(device: &wgpu::Device, queue: &wgpu::Queue) -> (OwnedWgpuBuffer, Vec<u8>) {
    let (_, tight) = SemiPlanar {
        stride: NV12.planes()[0].0,
        ..NV12
    }
    .padded_and_tight();
    let buffer = source_buffer(device, &tight);
    (
        WgpuToDmaBuf::wrap_buffer(device, queue, buffer, tight.len()),
        tight,
    )
}

#[tokio::test]
async fn a_plain_wgpu_device_exports_without_the_semaphore() {
    let _gpu = GPU_LOCK.lock().await;
    let Some((device, queue)) = plain_vulkan_device().await else {
        eprintln!("no Vulkan adapter; skipping");
        return;
    };
    let frame_caps = caps(RawVideoFormat::Nv12, WIDTH, HEIGHT);
    let (owned, tight) = tight_nv12_frame(&device, &queue);
    let exported = push_through(
        &mut WgpuToDmaBuf::new(),
        &frame_caps,
        MemoryDomain::WgpuBuffer(owned),
    )
    .await
    .expect("exports from a device opened with no extra extensions");
    let reimported = push_through(&mut DmaBufToWgpu::new(), &frame_caps, exported.domain)
        .await
        .expect("re-imports");
    let frame = push_through(&mut WgpuDownload::new(), &frame_caps, reimported.domain)
        .await
        .expect("downloads");
    assert_eq!(system_bytes(&frame), tight);

    let (owned, _) = tight_nv12_frame(&device, &queue);
    let refused = push_through(
        &mut WgpuToDmaBuf::new().with_external_semaphore(true),
        &frame_caps,
        MemoryDomain::WgpuBuffer(owned),
    )
    .await;
    assert!(matches!(refused, Err(G2gError::UnsupportedDomain)));
}

#[tokio::test]
async fn unaligned_tight_nv12_exports_unchanged() {
    let _gpu = GPU_LOCK.lock().await;
    let shape = SemiPlanar {
        stride: NV12.planes()[0].0,
        height: UNALIGNED_HEIGHT,
        ..NV12
    };
    let (_, tight) = shape.padded_and_tight();
    assert_eq!(
        Some(tight.len() as u64),
        RawVideoFormat::Nv12.unpadded_frame_bytes(WIDTH, UNALIGNED_HEIGHT)
    );
    assert_ne!(tight.len() as u64 % wgpu::COPY_BUFFER_ALIGNMENT, 0);
    let mut export = WgpuToDmaBuf::new();
    let Ok((device, queue)) = export.gpu().await else {
        eprintln!("no Vulkan export device; skipping");
        return;
    };
    let buffer = source_buffer(&device, &tight);
    let frame_caps = shape.caps();
    let exported = push_through(
        &mut export,
        &frame_caps,
        MemoryDomain::WgpuBuffer(WgpuToDmaBuf::wrap_buffer(
            &device,
            &queue,
            buffer,
            tight.len(),
        )),
    )
    .await
    .expect("exports");
    let reimported = push_through(&mut DmaBufToWgpu::new(), &frame_caps, exported.domain)
        .await
        .expect("re-imports");
    let frame = push_through(&mut WgpuDownload::new(), &frame_caps, reimported.domain)
        .await
        .expect("downloads");
    assert_eq!(system_bytes(&frame), tight);
}

// one dma-buf stride cannot describe luma rows one byte narrower than the chroma rows
#[test]
fn odd_width_tight_nv12_export_is_refused() {
    let frame_caps = caps(RawVideoFormat::Nv12, ODD_WIDTH, HEIGHT);
    assert!(matches!(
        WgpuToDmaBuf::new().configure_pipeline(&frame_caps),
        Err(G2gError::CapsMismatch)
    ));
}

struct ZeroClock;

impl g2g_core::PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

static SOURCE_DMABUF: std::sync::Mutex<Option<OwnedDmaBuf>> = std::sync::Mutex::new(None);

// appsrc declares system memory only, so it cannot feed `dmabuftowgpu` on a launch line.
#[derive(Debug)]
struct DmaBufSource;

impl g2g_core::runtime::SourceLoop for DmaBufSource {
    type RunFuture<'a> =
        core::pin::Pin<Box<dyn core::future::Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(caps(RawVideoFormat::Nv12, WIDTH, HEIGHT)))
    }

    fn configure_pipeline(
        &mut self,
        _absolute_caps: &Caps,
    ) -> Result<g2g_core::ConfigureOutcome, G2gError> {
        Ok(g2g_core::ConfigureOutcome::Accepted)
    }

    fn output_memory(&self) -> g2g_core::MemoryDomainKind {
        g2g_core::MemoryDomainKind::DmaBuf
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let dmabuf = SOURCE_DMABUF
                .lock()
                .unwrap()
                .take()
                .expect("a dma-buf to emit");
            let frame = Frame::new(MemoryDomain::DmaBuf(dmabuf), FrameTiming::default(), 0);
            out.push(PipelinePacket::DataFrame(frame)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

#[tokio::test]
async fn a_dmabuf_sink_gets_the_export_spliced_after_a_gpu_buffer() {
    use g2g_core::runtime::SourceFactory;
    use g2g_plugins::appsink::{register_appsink_pull, Pull};

    let _gpu = GPU_LOCK.lock().await;
    let (padded, tight) = NV12.padded_and_tight();
    let Some(dmabuf) = padded_dmabuf(&padded, NV12.stride).await else {
        eprintln!("no Vulkan export device; skipping");
        return;
    };
    *SOURCE_DMABUF.lock().unwrap() = Some(dmabuf);
    let pull = register_appsink_pull("m1211_export_out");
    let line = "m1211dmabufsrc ! dmabuftowgpu ! appsink input-domains=dmabuf,system channel=m1211_export_out";
    let mut registry = g2g_plugins::registry::default_registry();
    registry.register_source(SourceFactory::new(
        "m1211dmabufsrc",
        caps(RawVideoFormat::Nv12, WIDTH, HEIGHT),
        || Box::new(DmaBufSource),
    ));
    let graph = g2g_core::runtime::parse_launch(&registry, line).expect("parses");
    g2g_core::runtime::run_graph(graph, &ZeroClock, 4)
        .await
        .expect("runs");
    let Pull::Frame(frame) = pull.try_pull() else {
        panic!("no frame reached the appsink");
    };
    let MemoryDomain::DmaBuf(exported) = &frame.domain else {
        panic!("expected a dma-buf, got {:?}", frame.domain.kind());
    };
    assert_eq!(exported.stride as usize, NV12.stride);

    let frame_caps = caps(RawVideoFormat::Nv12, WIDTH, HEIGHT);
    let reimported = push_through(&mut DmaBufToWgpu::new(), &frame_caps, frame.domain)
        .await
        .expect("re-imports");
    let downloaded = push_through(&mut WgpuDownload::new(), &frame_caps, reimported.domain)
        .await
        .expect("downloads");
    assert_eq!(system_bytes(&downloaded), tight);
}
