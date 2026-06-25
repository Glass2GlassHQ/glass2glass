//! M304: on-device probe for the Android MediaCodec -> wgpu/Vulkan bridge.
//!
//! The linchpin question for the zero-copy decode-to-GPU path: when the decoded
//! `AImage`'s `AHardwareBuffer` is imported into Vulkan, does
//! `vkGetAndroidHardwareBufferPropertiesANDROID` report a *known multi-planar
//! `VkFormat`* (e.g. `G8_B8R8_2PLANE_420_UNORM`, so we can `vkCmdCopyImage` per
//! plane and skip ycbcr, making the device-local copy viable) or an *opaque*
//! buffer (`external_format != 0`, `format == UNDEFINED`, forcing a
//! `VkSamplerYcbcrConversion` wgpu cannot express)? Everything downstream of
//! M304 hinges on that answer, so this decodes one real frame on the device,
//! captures its hardware buffer, and prints / asserts what Vulkan says.
//!
//! Runs only on `aarch64-linux-android` (et al.) with the `mediacodec-wgpu`
//! feature; compiles to nothing on the dev host. Build with cargo-ndk
//! `--platform 26` (the AHardwareBuffer NDK API and the Vulkan AHB extension are
//! API 26+), push, and run as a bare native binary. See the M304 notes in
//! `project_android_platform_track` for the full build / run recipe.

#![cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, ConfigureOutcome, Dim, G2gError, Rate, VideoCodec};
use g2g_plugins::mediacodec_wgpu::{ahb_format_info, create_android_interop_device};
use g2g_plugins::mediacodecdec::MediaCodecDec;

use ash::vk;

/// Start a binder threadpool so Codec2 can allocate the decoder's output graphic
/// buffers (it calls back over binder). A bare native binary has no threadpool,
/// so the allocate transaction stalls; the `ABinderProcess_*` symbols live in
/// the device's libbinder_ndk.so but not the NDK link stub, so dlsym them. Same
/// shim as `android_mediacodec_smoke`.
fn start_binder_threadpool() {
    use core::ffi::{c_char, c_int, c_void};
    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    const RTLD_NOW: c_int = 2;
    // SAFETY: libbinder_ndk.so is already loaded by MediaCodec; the dlsym'd
    // symbols have the C signatures from <android/binder_process.h>.
    unsafe {
        let lib = dlopen(b"libbinder_ndk.so\0".as_ptr() as *const c_char, RTLD_NOW);
        if lib.is_null() {
            return;
        }
        let set = dlsym(lib, b"ABinderProcess_setThreadPoolMaxThreadCount\0".as_ptr() as *const c_char);
        if !set.is_null() {
            let set: extern "C" fn(u32) -> bool = core::mem::transmute(set);
            set(1);
        }
        let start = dlsym(lib, b"ABinderProcess_startThreadPool\0".as_ptr() as *const c_char);
        if !start.is_null() {
            let start: extern "C" fn() = core::mem::transmute(start);
            start();
        }
    }
}

/// 640x480 H.264 Annex-B fixture (parameter sets + IDR + inter frames), embedded
/// so the binary is self-contained. HW decoders reject sub-minimum dimensions.
const H264: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

/// Sink that discards packets: this probe inspects the captured hardware buffer,
/// not the NV12 output.
#[derive(Default)]
struct Discard;

impl OutputSink for Discard {
    fn push<'a>(&'a mut self, _packet: PipelinePacket) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move { Ok(PushOutcome::Accepted) })
    }
}

/// Split an H.264 Annex-B stream into access units (the first AU carries the
/// parameter sets alongside the IDR, as `MediaCodecDec` requires).
fn access_units(s: &[u8]) -> Vec<&[u8]> {
    let nal_type = |hdr: u8| hdr & 0x1f;
    let is_vcl = |t: u8| (1..=5).contains(&t);

    let mut nals: Vec<(usize, u8)> = Vec::new();
    let mut i = 0;
    while i + 3 <= s.len() {
        let sc_len = if i + 4 <= s.len() && s[i] == 0 && s[i + 1] == 0 && s[i + 2] == 0 && s[i + 3] == 1
        {
            Some(4)
        } else if s[i] == 0 && s[i + 1] == 0 && s[i + 2] == 1 {
            Some(3)
        } else {
            None
        };
        match sc_len {
            Some(len) => {
                let hdr = i + len;
                if hdr < s.len() {
                    nals.push((i, nal_type(s[hdr])));
                }
                i = hdr + 1;
            }
            None => i += 1,
        }
    }

    let mut starts: Vec<usize> = Vec::new();
    let mut has_vcl = false;
    for &(off, t) in &nals {
        let vcl = is_vcl(t);
        if starts.is_empty() {
            starts.push(off);
            has_vcl = vcl;
        } else if vcl && has_vcl {
            starts.push(off);
        } else if vcl {
            has_vcl = true;
        }
    }

    starts
        .iter()
        .enumerate()
        .map(|(k, &start)| {
            let end = starts.get(k + 1).copied().unwrap_or(s.len());
            &s[start..end]
        })
        .collect()
}

#[tokio::test]
async fn probe_ahb_vulkan_format() {
    start_binder_threadpool();

    // Decode the fixture far enough to capture a real decoded hardware buffer.
    let mut dec = MediaCodecDec::h264();
    let upstream = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Any,
    };
    let narrowed = dec.intercept_caps(&upstream).expect("intercept caps");
    assert!(matches!(
        dec.configure_pipeline(&narrowed).expect("configure decoder"),
        ConfigureOutcome::Accepted
    ));

    let mut sink = Discard;
    let mut pts_ns = 0u64;
    for au in access_units(H264) {
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(au.to_vec().into_boxed_slice())),
            timing: FrameTiming { pts_ns, dts_ns: pts_ns, capture_ns: pts_ns, ..FrameTiming::default() },
            sequence: 0,
            meta: Default::default(),
        };
        dec.process(PipelinePacket::DataFrame(frame), &mut sink).await.expect("process access unit");
        pts_ns += 33_366_700;
    }
    dec.process(PipelinePacket::Eos, &mut sink).await.expect("Eos drains the codec");

    let ahb = dec
        .captured_hardware_buffer()
        .expect("decoder must have captured a hardware buffer from a decoded frame");

    // Open a Vulkan-backed wgpu device with the AHB external-memory extension and
    // ask it about the captured buffer.
    let dev = create_android_interop_device().await.expect("create android interop device");
    // SAFETY: `ahb` is an owned (acquired) reference held alive by `dec` for the
    // whole call; `dev` is a Vulkan interop device.
    let info = unsafe {
        ahb_format_info(&dev, ahb.as_ptr() as *const vk::AHardwareBuffer)
            .expect("query AHB Vulkan format properties")
    };

    eprintln!("=== M304 AHB Vulkan format probe ===");
    eprintln!("vk_format               = {:?}", info.vk_format);
    eprintln!("external_format         = {:#x}", info.external_format);
    eprintln!("allocation_size         = {}", info.allocation_size);
    eprintln!("memory_type_bits        = {:#x}", info.memory_type_bits);
    eprintln!("format_features         = {:?}", info.format_features);
    eprintln!("suggested_ycbcr_model   = {:?}", info.suggested_ycbcr_model);
    eprintln!("suggested_ycbcr_range   = {:?}", info.suggested_ycbcr_range);
    if info.is_importable_format() {
        eprintln!(
            ">>> CONCRETE VkFormat ({:?}): per-plane copy viable, device-local-copy path is GO.",
            info.vk_format
        );
    } else {
        eprintln!(
            ">>> OPAQUE buffer (external_format={:#x}): ycbcr conversion forced; \
             reassess the device-local-copy plan.",
            info.external_format
        );
    }

    // The probe is informational, but a useful invariant holds either way: the
    // import allocation is nonzero, and exactly one of {concrete format, opaque
    // external format} is reported.
    assert!(info.allocation_size > 0, "import allocation size must be nonzero");
    assert_ne!(info.memory_type_bits, 0, "import must report at least one memory type");
    assert!(
        (info.vk_format != vk::Format::UNDEFINED) ^ (info.external_format != 0),
        "buffer must be either a concrete VkFormat or opaque, not both/neither (format={:?}, external={:#x})",
        info.vk_format,
        info.external_format
    );
}
