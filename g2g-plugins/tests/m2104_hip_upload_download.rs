#![cfg(all(target_os = "linux", feature = "hip"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use collect_sink::CollectSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{DomainSet, GpuVendor, MemoryDomain, MemoryDomainKind, SystemSlice};
use g2g_core::runtime::block_on;
use g2g_core::{
    AllocationParams, AsyncElement, Caps, Colorimetry, Dim, Interlace, PropValue, Rate,
    RawVideoFormat,
};
use g2g_plugins::cuda::{CudaDownload, CudaUpload};
use g2g_plugins::hip::{
    HipDevice, HipEvent, HipKernelModule, HipMemory, HipStream, KernelArgument,
};
use g2g_plugins::hipdownload::HipDownload;
use g2g_plugins::hipupload::HipUpload;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const FRAMES: u64 = 3;
const COPY_SOURCE: &str = r#"
extern "C" __global__ void copy_bytes(const unsigned char* input, unsigned char* output, unsigned int count) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input[index];
    }
}
"#;
const BLOCK: u32 = 256;

fn nv12_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Any,
        interlace: Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
    }
}

fn nv12_frame(sequence: u64) -> Vec<u8> {
    let size = (WIDTH * HEIGHT * 3 / 2) as usize;
    (0..size)
        .map(|index| (index as u64 * 7 + sequence * 31) as u8)
        .collect()
}

fn run<E: AsyncElement>(element: &mut E, packet: PipelinePacket) -> Vec<PipelinePacket> {
    let mut sink = CollectSink::default();
    block_on(element.process(packet, &mut sink)).unwrap();
    sink.packets
}

fn only_frame(packets: Vec<PipelinePacket>) -> Frame {
    match <[PipelinePacket; 1]>::try_from(packets) {
        Ok([PipelinePacket::DataFrame(frame)]) => frame,
        other => panic!("expected one frame, got {other:?}"),
    }
}

fn system_frame(bytes: Vec<u8>, sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        sequence,
    ))
}

fn system_bytes(frame: &Frame) -> &[u8] {
    frame.domain.as_system_slice().expect("a system frame")
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

fn system_frames_round_trip(vendor: GpuVendor) {
    if !device_present(vendor) {
        return;
    }
    let vendor_name = match vendor {
        GpuVendor::Amd => "amd",
        GpuVendor::Nvidia => "nvidia",
    };
    let mut upload = HipUpload::new();
    upload
        .set_property("vendor", PropValue::Str(vendor_name.into()))
        .unwrap();
    upload.configure_pipeline(&nv12_caps()).unwrap();
    let mut download = HipDownload::new();
    download.configure_pipeline(&nv12_caps()).unwrap();

    for sequence in 0..FRAMES {
        let bytes = nv12_frame(sequence);
        let uploaded = only_frame(run(&mut upload, system_frame(bytes.clone(), sequence)));
        let MemoryDomain::Hip(buffer) = &uploaded.domain else {
            panic!(
                "{vendor:?}: upload left a {:?} frame",
                uploaded.domain.kind()
            );
        };
        assert_eq!(buffer.vendor, vendor);
        let downloaded = only_frame(run(&mut download, PipelinePacket::DataFrame(uploaded)));
        assert!(
            system_bytes(&downloaded) == bytes,
            "{vendor:?}: frame {sequence} changed on the way through HIP"
        );
    }
}

#[test]
fn amd_system_frames_round_trip() {
    system_frames_round_trip(GpuVendor::Amd);
}

#[test]
fn nvidia_system_frames_round_trip() {
    system_frames_round_trip(GpuVendor::Nvidia);
}

fn kernel_copy(
    device: &std::sync::Arc<HipDevice>,
    input: &g2g_core::memory::OwnedHipBuffer,
) -> Vec<u8> {
    let module = match HipKernelModule::compile(device, COPY_SOURCE, &[]) {
        Ok(module) => module,
        Err(error) => panic!("compile the copy kernel: {error:?}"),
    };
    let kernel = module.kernel("copy_bytes").unwrap();
    let stream = HipStream::new(device).unwrap();
    let mut output = HipMemory::allocate(device, input.len).unwrap();
    let count = input.len as u32;
    stream
        .launch(
            &kernel,
            [count.div_ceil(BLOCK), 1, 1],
            [BLOCK, 1, 1],
            &[
                KernelArgument::Read(input),
                KernelArgument::Write(&mut output),
                KernelArgument::U32(count),
            ],
        )
        .unwrap();
    let ready = HipEvent::new(device).unwrap();
    stream.record(&ready).unwrap();
    let copied = output.into_buffer(ready, input.len).unwrap();
    let mut bytes = vec![0u8; input.len];
    stream.download_blocking(&copied, &mut bytes).unwrap();
    bytes
}

#[test]
fn nvidia_cuda_frames_cross_without_a_copy() {
    if !device_present(GpuVendor::Nvidia) {
        return;
    }
    let bytes = nv12_frame(0);
    let mut cuda_upload = CudaUpload::new();
    cuda_upload.configure_pipeline(&nv12_caps()).unwrap();
    let cuda_frame = only_frame(run(&mut cuda_upload, system_frame(bytes.clone(), 0)));
    let MemoryDomain::Cuda(cuda) = &cuda_frame.domain else {
        panic!("CudaUpload left a {:?} frame", cuda_frame.domain.kind());
    };
    let (luma_ptr, luma_pitch) = (cuda.luma_ptr, cuda.luma_pitch as usize);

    let mut upload = HipUpload::new();
    upload.configure_pipeline(&nv12_caps()).unwrap();
    let hip_frame = only_frame(run(&mut upload, PipelinePacket::DataFrame(cuda_frame)));
    let MemoryDomain::Hip(buffer) = &hip_frame.domain else {
        panic!("upload left a {:?} frame", hip_frame.domain.kind());
    };
    assert_eq!(buffer.vendor, GpuVendor::Nvidia);
    assert_eq!(
        buffer.device_pointer, luma_ptr,
        "the Hip frame is the CUDA memory"
    );
    assert_eq!(
        buffer.plane_layout().unwrap().plane(0).unwrap().stride,
        luma_pitch
    );

    let device = HipDevice::open(GpuVendor::Nvidia, 0).unwrap();
    let mut wrapped_bytes = vec![0u8; buffer.len];
    HipStream::new(&device)
        .unwrap()
        .download_blocking(buffer, &mut wrapped_bytes)
        .unwrap();
    assert!(
        kernel_copy(&device, buffer) == wrapped_bytes,
        "a kernel read different bytes from the wrapped CUDA frame"
    );

    let mut to_system = HipDownload::new();
    to_system.configure_pipeline(&nv12_caps()).unwrap();
    let system = only_frame(run(
        &mut to_system,
        PipelinePacket::DataFrame(Frame::new(
            hip_frame.domain.share(),
            FrameTiming::default(),
            0,
        )),
    ));
    assert!(
        system_bytes(&system) == bytes,
        "the wrapped frame downloads to its pixels"
    );

    let mut to_cuda = HipDownload::new();
    to_cuda.configure_allocation(&AllocationParams {
        domain: MemoryDomainKind::Cuda,
        accepts: DomainSet::only(MemoryDomainKind::Cuda),
        ..AllocationParams::default()
    });
    assert_eq!(to_cuda.output_memory(), MemoryDomainKind::Cuda);
    to_cuda.configure_pipeline(&nv12_caps()).unwrap();
    let cuda_again = only_frame(run(&mut to_cuda, PipelinePacket::DataFrame(hip_frame)));
    let MemoryDomain::Cuda(cuda) = &cuda_again.domain else {
        panic!("download left a {:?} frame", cuda_again.domain.kind());
    };
    assert_eq!(cuda.luma_ptr, luma_ptr, "the CUDA frame is the Hip memory");
    let mut cuda_download = CudaDownload::new();
    cuda_download.configure_pipeline(&nv12_caps()).unwrap();
    let system = only_frame(run(
        &mut cuda_download,
        PipelinePacket::DataFrame(cuda_again),
    ));
    assert!(
        system_bytes(&system) == bytes,
        "the CUDA frame reads back its pixels"
    );
}

#[test]
fn hipdownload_settles_on_cuda_only_for_a_cuda_only_consumer() {
    let settled = |accepts: DomainSet| {
        let mut download = HipDownload::new();
        download.configure_allocation(&AllocationParams {
            accepts,
            ..AllocationParams::default()
        });
        download.output_memory()
    };
    assert_eq!(settled(DomainSet::ALL), MemoryDomainKind::System);
    assert_eq!(
        settled(DomainSet::only(MemoryDomainKind::Cuda)),
        MemoryDomainKind::Cuda
    );
    assert_eq!(
        settled(DomainSet::only(MemoryDomainKind::Cuda).with(MemoryDomainKind::System)),
        MemoryDomainKind::System
    );
}

#[derive(Debug)]
struct ZeroClock;

impl g2g_core::PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[test]
fn a_launch_line_splices_hipupload_in_front_of_hipdownload() {
    if !device_present(GpuVendor::Amd) && !device_present(GpuVendor::Nvidia) {
        return;
    }
    let bytes = nv12_frame(0);
    let feed = g2g_plugins::appsrc::register_appsrc("m2104_in");
    assert!(feed.push(&bytes, 0));
    feed.end_of_stream();
    let pull = g2g_plugins::appsink::register_appsink_pull("m2104_out");
    let line = format!(
        "appsrc channel=m2104_in caps=video/x-raw,format=NV12,width={WIDTH},height={HEIGHT},framerate=30/1 \
         ! hipdownload ! appsink channel=m2104_out"
    );
    let registry = g2g_plugins::registry::default_registry();
    let graph = g2g_core::runtime::parse_launch(&registry, &line).expect("parses");
    let categories: Vec<&str> = (0..graph.node_count() as u32)
        .filter_map(|node| graph.element(g2g_core::graph::NodeId(node)))
        .map(|element| element.log_category())
        .collect();
    assert!(
        categories.contains(&"HipUpload"),
        "no hipupload spliced in: {categories:?}"
    );
    block_on(g2g_core::runtime::run_graph(graph, &ZeroClock, 4)).expect("runs");
    match pull.try_pull() {
        g2g_plugins::appsink::Pull::Frame(frame) => {
            assert!(
                system_bytes(&frame) == bytes,
                "the frame changed through HIP"
            )
        }
        other => panic!("no frame reached the appsink: {other:?}"),
    }
}
