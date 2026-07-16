//! `DmaBufToWgpu`: a dma-buf frame is imported into a GPU-resident `wgpu::Buffer`
//! with no CPU copy. Validated on this host (RTX 3060) by *exporting* GPU memory
//! as a dma-buf fd and re-importing it through the element: a discrete GPU can
//! bind a GPU-visible dma-buf (unlike a CPU/vmalloc-backed camera buffer, per
//! `libcamera_dmabuf`). Skips cleanly when no Vulkan adapter is present.
#![cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]

use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};

use ash::vk;

use g2g_core::element::{BoxFuture, PushOutcome};
use g2g_core::memory::{MemoryDomain, OwnedDmaBuf};
use g2g_core::runtime::block_on;
use g2g_core::{
    AsyncElement, Caps, Dim, Frame, FrameTiming, G2gError, OutputSink, PipelinePacket, Rate,
    RawVideoFormat,
};

use g2g_plugins::dmabufwgpu::{DmaBufToWgpu, DmaBufWgpuBuffer};

/// Collects packets the element pushes downstream.
#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}
impl OutputSink for Collect {
    fn push<'a>(&'a mut self, packet: PipelinePacket) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Export `size` bytes of device-local GPU memory as a dma-buf fd, so the dGPU
/// can re-import it. `None` when no Vulkan adapter (headless CI).
async fn export_gpu_dmabuf(size: u64) -> Option<OwnedFd> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .ok()?;
    // SAFETY: open a device carrying the external-memory export extensions.
    let open = unsafe {
        let hal = adapter.as_hal::<wgpu_hal::api::Vulkan>()?;
        hal.open_with_callback(
            wgpu::Features::empty(),
            &wgpu::Limits::default(),
            &wgpu::MemoryHints::default(),
            Some(Box::new(|args: wgpu_hal::vulkan::CreateDeviceCallbackArgs| {
                args.extensions.push(ash::khr::external_memory_fd::NAME);
                args.extensions.push(ash::ext::external_memory_dma_buf::NAME);
            })),
        )
        .ok()?
    };
    // SAFETY: `open` came from this adapter's hal.
    let (device, _q) =
        unsafe { adapter.create_device_from_hal(open, &wgpu::DeviceDescriptor::default()).ok()? };

    // SAFETY: raw Vulkan objects; the exported dma-buf holds its own reference to
    // the memory, so it stays valid after this one-shot export (buffer/memory
    // deliberately leaked for the test's lifetime).
    let fd = unsafe {
        let hal = device.as_hal::<wgpu_hal::api::Vulkan>()?;
        let raw = hal.raw_device();
        let instance = hal.shared_instance().raw_instance();
        let pdev = hal.raw_physical_device();

        let mut ext_buf = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let buf_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut ext_buf);
        let buffer = raw.create_buffer(&buf_info, None).ok()?;
        let req = raw.get_buffer_memory_requirements(buffer);

        let mem_props = instance.get_physical_device_memory_properties(pdev);
        let mut type_index = None;
        for i in 0..mem_props.memory_type_count {
            let ok = req.memory_type_bits & (1 << i) != 0;
            let dev_local = mem_props.memory_types[i as usize]
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
            if ok && dev_local {
                type_index = Some(i);
                break;
            }
        }
        let type_index = type_index?;

        let mut export = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(type_index)
            .push_next(&mut export);
        let memory = raw.allocate_memory(&alloc, None).ok()?;
        raw.bind_buffer_memory(buffer, memory, 0).ok()?;

        let loader = ash::khr::external_memory_fd::Device::new(instance, raw);
        let get = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        OwnedFd::from_raw_fd(loader.get_memory_fd(&get).ok()?)
    };
    Some(fd)
}

#[test]
fn imports_a_gpu_dmabuf_into_a_wgpu_buffer() {
    // 256-byte stride x 16 rows = 4096 bytes (an RGBA 64x16 frame).
    let (width, height, stride) = (64u32, 16u32, 256u32);
    let size = u64::from(stride) * u64::from(height);

    let Some(fd) = block_on(export_gpu_dmabuf(size)) else {
        eprintln!("SKIP: no Vulkan adapter");
        return;
    };
    // Transfer the fd to an OwnedDmaBuf (closes it on drop), as the bridge would
    // after dup-ing a GstBuffer's dma-buf memory.
    let raw = fd.into_raw_fd();
    // SAFETY: `raw` is a fresh dma-buf fd this test solely owns.
    let dmabuf = unsafe { OwnedDmaBuf::from_raw(raw, stride, 0) };

    let caps = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Fixed(30 << 16),
    };
    let mut elem = DmaBufToWgpu::new();
    elem.configure_pipeline(&caps).expect("configures for RGBA caps");

    let frame = Frame::new(MemoryDomain::DmaBuf(dmabuf), FrameTiming::default(), 0);
    let mut sink = Collect::default();
    block_on(elem.process(PipelinePacket::DataFrame(frame), &mut sink)).expect("import runs");

    assert_eq!(elem.imported(), 1, "one frame imported");
    let [PipelinePacket::DataFrame(out)] = sink.packets.as_slice() else {
        panic!("expected one output DataFrame, got {:?}", sink.packets);
    };
    let MemoryDomain::WgpuBuffer(buf) = &out.domain else {
        panic!("output should be GPU-resident WgpuBuffer, got {:?}", out.domain);
    };
    assert_eq!(buf.len, size as usize, "buffer sized to the frame");
    // The keep-alive owns a real wgpu::Buffer aliasing the imported dma-buf.
    let owner = buf.keep_alive().as_any().downcast_ref::<DmaBufWgpuBuffer>().expect("owner type");
    assert_eq!(owner.buffer().size(), size, "wgpu buffer wraps the imported memory");
}
