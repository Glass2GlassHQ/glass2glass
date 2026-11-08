#![cfg(all(target_os = "linux", feature = "hip"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/udmabuf.rs"]
mod udmabuf;

use std::sync::{Mutex, PoisonError};

use collect_sink::CollectSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{GpuVendor, MemoryDomain, OwnedDmaBuf};
use g2g_core::runtime::block_on;
use g2g_core::{AsyncElement, Caps, Colorimetry, Dim, Interlace, PropValue, Rate, RawVideoFormat};
use g2g_plugins::hip::HipDevice;
use g2g_plugins::hipdownload::HipDownload;
use g2g_plugins::hipupload::HipUpload;
use udmabuf::udmabuf_of;

const WIDTH: usize = 64;
const HEIGHT: usize = 48;
const PADDED_STRIDE: usize = 128;
const PAD_BYTE: u8 = 0xee;
const PAGE: usize = 4096;
// the open-fd count is process-wide, so these tests take turns
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn nv12_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(WIDTH as u32),
        height: Dim::Fixed(HEIGHT as u32),
        framerate: Rate::Any,
        interlace: Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
    }
}

fn tight_nv12(seed: usize) -> Vec<u8> {
    (0..WIDTH * HEIGHT * 3 / 2)
        .map(|i| ((i * 7 + seed * 31) % 251) as u8)
        .collect()
}

fn device_present(vendor: GpuVendor) -> bool {
    match HipDevice::open(vendor, 0) {
        Ok(_) => true,
        Err(error) => {
            eprintln!("skipping: no {vendor:?} HIP device 0: {error:?}");
            false
        }
    }
}

fn run<E: AsyncElement>(element: &mut E, frame: Frame) -> Frame {
    let mut sink = CollectSink::default();
    block_on(element.process(PipelinePacket::DataFrame(frame), &mut sink)).unwrap();
    sink.packets
        .into_iter()
        .find_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
        .expect("a frame")
}

fn hipupload(vendor: &str) -> HipUpload {
    let mut upload = HipUpload::new();
    upload
        .set_property("vendor", PropValue::Str(vendor.into()))
        .unwrap();
    upload.configure_pipeline(&nv12_caps()).unwrap();
    upload
}

fn download(hip: &Frame) -> Vec<u8> {
    let mut download = HipDownload::new();
    download.configure_pipeline(&nv12_caps()).unwrap();
    let frame = Frame::new(hip.domain.share(), FrameTiming::default(), 0);
    let system = run(&mut download, frame);
    system.domain.as_system_slice().expect("system").to_vec()
}

fn upload_dmabuf(upload: &mut HipUpload, dmabuf: &OwnedDmaBuf) -> Frame {
    let frame = Frame::new(
        MemoryDomain::DmaBuf(dmabuf.clone()),
        FrameTiming::default(),
        0,
    );
    let hip = run(upload, frame);
    assert!(matches!(hip.domain, MemoryDomain::Hip(_)), "a Hip frame");
    hip
}

// the tight frame's rows at `PADDED_STRIDE` in a udmabuf, which hip does not import
fn padded_udmabuf(tight: &[u8]) -> Option<OwnedDmaBuf> {
    let rows = HEIGHT * 3 / 2;
    let mut padded = vec![PAD_BYTE; (rows * PADDED_STRIDE).next_multiple_of(PAGE)];
    for (row, line) in tight.chunks(WIDTH).enumerate() {
        padded[row * PADDED_STRIDE..][..WIDTH].copy_from_slice(line);
    }
    let mut dmabuf = udmabuf_of(&padded)?;
    dmabuf.stride = PADDED_STRIDE as u32;
    Some(dmabuf)
}

fn copies_a_padded_udmabuf(vendor: GpuVendor, vendor_name: &str) {
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner);
    if !device_present(vendor) {
        return;
    }
    let picture = tight_nv12(1);
    let Some(dmabuf) = padded_udmabuf(&picture) else {
        eprintln!("skipping: /dev/udmabuf is not available");
        return;
    };
    let hip = upload_dmabuf(&mut hipupload(vendor_name), &dmabuf);
    assert!(download(&hip) == picture, "the padded rows come back tight");
}

#[test]
fn amd_copies_a_dmabuf_hip_cannot_import() {
    copies_a_padded_udmabuf(GpuVendor::Amd, "amd");
}

#[test]
fn nvidia_copies_a_dmabuf_through_the_cpu() {
    copies_a_padded_udmabuf(GpuVendor::Nvidia, "nvidia");
}

#[derive(Debug)]
struct ZeroClock;

impl g2g_core::PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[test]
fn a_launch_line_splices_hipupload_after_a_dmabuf_producer() {
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner);
    if !device_present(GpuVendor::Amd) && !device_present(GpuVendor::Nvidia) {
        return;
    }
    let picture = tight_nv12(3);
    let mut pages = picture.clone();
    pages.resize(picture.len().next_multiple_of(PAGE), PAD_BYTE);
    let Some(mut dmabuf) = udmabuf_of(&pages) else {
        eprintln!("skipping: /dev/udmabuf is not available");
        return;
    };
    dmabuf.stride = WIDTH as u32;
    let feed = g2g_plugins::appsrc::register_appsrc("m2107_dmabuf_in");
    assert!(feed.push_dmabuf(dmabuf, 0));
    feed.end_of_stream();
    let pull = g2g_plugins::appsink::register_appsink_pull("m2107_dmabuf_out");
    let line = format!(
        "appsrc channel=m2107_dmabuf_in output-domains=dmabuf \
         caps=video/x-raw,format=NV12,width={WIDTH},height={HEIGHT},framerate=30/1 \
         ! hipconvertscale ! hipdownload ! appsink channel=m2107_dmabuf_out"
    );
    let registry = g2g_plugins::registry::default_registry();
    let graph = g2g_core::runtime::parse_launch(&registry, &line).expect("parses");
    block_on(g2g_core::runtime::run_graph(graph, &ZeroClock, 4)).expect("runs");
    let g2g_plugins::appsink::Pull::Frame(frame) = pull.try_pull() else {
        panic!("no frame reached the appsink");
    };
    assert_eq!(
        frame.domain.as_system_slice().expect("a system frame"),
        picture
    );
}

#[cfg(feature = "dmabuf-wgpu")]
mod amdgpu_exports {
    use std::time::{Duration, Instant};

    use g2g_core::memory::SystemSlice;
    use g2g_plugins::wgpudmabuf::WgpuToDmaBuf;

    use super::*;

    const AMD_PCI_VENDOR: u32 = 0x1002;

    // `pictures` as dma-bufs the radeon's vulkan driver exported, `None` off an AMD GPU
    fn exported(pictures: &[Vec<u8>]) -> Option<(WgpuToDmaBuf, Vec<OwnedDmaBuf>)> {
        let mut export = WgpuToDmaBuf::new();
        let (device, queue) = block_on(export.gpu()).ok()?;
        if device.adapter_info().vendor != AMD_PCI_VENDOR {
            eprintln!("skipping: the vulkan export device is not an AMD GPU");
            return None;
        }
        export.configure_pipeline(&nv12_caps()).unwrap();
        let dmabufs = pictures
            .iter()
            .map(|picture| {
                let source = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("m2107-source"),
                    size: picture.len() as u64,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&source, 0, picture);
                let buffer = WgpuToDmaBuf::wrap_buffer(&device, &queue, source, picture.len());
                let frame = Frame::new(MemoryDomain::WgpuBuffer(buffer), FrameTiming::default(), 0);
                match run(&mut export, frame).domain {
                    MemoryDomain::DmaBuf(dmabuf) => dmabuf,
                    _ => panic!("WgpuToDmaBuf emits a dma-buf"),
                }
            })
            .collect();
        Some((export, dmabufs))
    }

    fn open_fds() -> usize {
        std::fs::read_dir("/proc/self/fd").unwrap().count()
    }

    #[test]
    fn amd_imports_each_pooled_dmabuf_once_and_closes_its_fds() {
        let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner);
        if !device_present(GpuVendor::Amd) {
            return;
        }
        let pictures = [tight_nv12(4), tight_nv12(5)];
        let Some((_export, pool)) = exported(&pictures) else {
            return;
        };
        // the HIP runtime keeps the fds it opens for a stream, so a first element opens them
        let mut warm_up = hipupload("amd");
        let system =
            MemoryDomain::System(SystemSlice::from_boxed(tight_nv12(0).into_boxed_slice()));
        download(&run(
            &mut warm_up,
            Frame::new(system, FrameTiming::default(), 0),
        ));
        let baseline = open_fds();
        let mut upload = hipupload("amd");
        for dmabuf in &pool {
            upload_dmabuf(&mut upload, dmabuf);
        }
        let settled = open_fds();
        assert_eq!(
            settled,
            baseline + pool.len(),
            "each buffer is imported once, holding its fd, rather than copied"
        );
        for round in 0..20 {
            let index = round % pool.len();
            let hip = upload_dmabuf(&mut upload, &pool[index]);
            assert!(download(&hip) == pictures[index], "round {round}");
        }
        assert_eq!(open_fds(), settled, "a cached buffer was imported again");
        drop(upload);
        // the release thread drops the imports once their frames' events complete
        let deadline = Instant::now() + Duration::from_secs(2);
        while open_fds() > baseline && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(open_fds(), baseline, "the imports kept fds open");
    }
}
