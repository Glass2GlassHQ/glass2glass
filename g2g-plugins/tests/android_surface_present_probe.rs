//! M305: on-device probe for the Android decode -> GPU -> surface present path.
//!
//! M304 took the decoded frame onto the GPU as an RGBA `wgpu::Texture` (no CPU
//! readback). M305 closes the loop: present that texture on-screen through a
//! `wgpu::Surface` built over an Android `ANativeWindow`, on the *same* wgpu
//! device the decoder converted it on, so the whole glass-to-glass chain stays
//! GPU-resident. The on-screen target is normally a `SurfaceView` /
//! `NativeActivity` window an app owns; this probe stands one in headlessly with
//! an `ImageReader` whose Surface accepts GPU colour output, so the present path
//! is exercised on a device without an Activity.
//!
//! The hard invariant: `WgpuSink` actually acquires and presents swapchain
//! textures from the Android surface (`presented_count() > 0`), which is only
//! possible if `create_android_surface` built a usable surface on the interop
//! device and the decoder's textures bind to that same device. The pixel
//! read-back from the `ImageReader` is informational: whether a wgpu swapchain
//! over an `ImageReader`-backed window delivers its content back through the
//! reader's `BufferQueue` is itself something this probe discovers on real
//! hardware (a true `SurfaceView` is the production target).
//!
//! Runs only on `aarch64-linux-android` (et al.) with `mediacodec-wgpu`; compiles
//! to nothing on the dev host. Build with cargo-ndk `--platform 26`, push, and
//! run as a bare native binary (see `tools/android-surface-present-smoke.sh`).

#![cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, ConfigureOutcome, Dim, G2gError, Rate, RawVideoFormat, VideoCodec};
use g2g_plugins::mediacodec_wgpu::{create_android_interop_device, create_android_surface};
use g2g_plugins::mediacodecdec::MediaCodecDec;
use g2g_plugins::wgpusink::WgpuSink;

use ndk::hardware_buffer::HardwareBufferUsage;
use ndk::media::image_reader::{AcquireResult, ImageFormat, ImageReader};

/// Start a binder threadpool so Codec2 can allocate the decoder's output graphic
/// buffers (it calls back over binder). A bare native binary has no threadpool,
/// so the allocate transaction stalls; the `ABinderProcess_*` symbols live in
/// the device's libbinder_ndk.so but not the NDK link stub, so dlsym them. Same
/// shim as `android_mediacodec_smoke` / `android_mediacodec_wgpu_probe`.
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
        let set = dlsym(
            lib,
            b"ABinderProcess_setThreadPoolMaxThreadCount\0".as_ptr() as *const c_char,
        );
        if !set.is_null() {
            let set: extern "C" fn(u32) -> bool = core::mem::transmute(set);
            set(1);
        }
        let start = dlsym(
            lib,
            b"ABinderProcess_startThreadPool\0".as_ptr() as *const c_char,
        );
        if !start.is_null() {
            let start: extern "C" fn() = core::mem::transmute(start);
            start();
        }
    }
}

/// 640x480 H.264 Annex-B fixture (parameter sets + IDR + inter frames), embedded
/// so the binary is self-contained. HW decoders reject sub-minimum dimensions.
const H264: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

/// Discards packets: the `WgpuSink`'s own (empty) downstream.
#[derive(Default)]
struct Discard;

impl OutputSink for Discard {
    fn push<'a>(
        &'a mut self,
        _packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move { Ok(PushOutcome::Accepted) })
    }
}

/// Forwards the decoder's output packets straight into the present sink, so each
/// decoded frame is presented as it is produced (decode -> present, frame by
/// frame). `WgpuSink` consumes `DataFrame`s and ignores control packets.
struct PresentRelay<'s> {
    sink: &'s mut WgpuSink,
}

impl OutputSink for PresentRelay<'_> {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            let mut nil = Discard;
            self.sink.process(packet, &mut nil).await?;
            Ok(PushOutcome::Accepted)
        })
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
        let sc_len =
            if i + 4 <= s.len() && s[i] == 0 && s[i + 1] == 0 && s[i + 2] == 0 && s[i + 3] == 1 {
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

fn rgba_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    }
}

#[tokio::test]
async fn present_decoded_frames_to_android_surface() {
    start_binder_threadpool();
    let (w, h) = (640u32, 480u32);

    // Headless present target: an ImageReader whose Surface accepts GPU colour
    // output, standing in for an on-screen SurfaceView so the present path runs
    // without a window / Activity. GPU_FRAMEBUFFER lets wgpu render into the
    // swapchain buffers; CPU_READ_OFTEN lets us tap the result below.
    let reader = ImageReader::new_with_usage(
        w as i32,
        h as i32,
        ImageFormat::RGBA_8888,
        HardwareBufferUsage::GPU_FRAMEBUFFER | HardwareBufferUsage::CPU_READ_OFTEN,
        4,
    )
    .expect("create ImageReader present target");
    let window = reader.window().expect("ImageReader native window");

    // One interop device, shared by the decoder and the present sink: a wgpu
    // texture binds only to the device that made it, so decode and present must
    // be on the same device for the hand-off to be copy-free.
    let dev = create_android_interop_device()
        .await
        .expect("create interop device");
    let ctx = dev.gpu_context();
    let (surface, config) =
        create_android_surface(&dev, &window, w, h).expect("create android surface");
    eprintln!(
        "=== M305 surface configured: {:?} {}x{} ===",
        config.format, config.width, config.height
    );
    let mut sink = WgpuSink::with_surface(ctx, surface, config);
    sink.configure_pipeline(&rgba_caps(w, h))
        .expect("configure sink");

    let mut dec = MediaCodecDec::h264().with_gpu_device(dev);
    let upstream = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    };
    let narrowed = dec.intercept_caps(&upstream).expect("intercept caps");
    assert!(matches!(
        dec.configure_pipeline(&narrowed)
            .expect("configure decoder"),
        ConfigureOutcome::Accepted
    ));

    // decode -> present, frame by frame, through the relay.
    {
        let mut relay = PresentRelay { sink: &mut sink };
        let mut pts_ns = 0u64;
        for au in access_units(H264) {
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(
                    au.to_vec().into_boxed_slice(),
                )),
                timing: FrameTiming {
                    pts_ns,
                    dts_ns: pts_ns,
                    capture_ns: pts_ns,
                    ..FrameTiming::default()
                },
                sequence: 0,
                meta: Default::default(),
            };
            dec.process(PipelinePacket::DataFrame(frame), &mut relay)
                .await
                .expect("decode + present access unit");
            pts_ns += 33_366_700;
        }
        dec.process(PipelinePacket::Eos, &mut relay)
            .await
            .expect("Eos drains the codec");
    }

    let presented = sink.presented_count();
    eprintln!("=== M305 surface present: {presented} frame(s) presented ===");
    // The hard invariant: the Android surface produced acquirable swapchain
    // textures and we presented decoded frames onto them. A broken surface would
    // never return a current texture, so this would be zero.
    assert!(
        presented > 0,
        "expected at least one frame presented to the Android surface"
    );

    // Informational tap: try to read a presented frame back through the
    // ImageReader's BufferQueue. Whether a wgpu swapchain over an ImageReader
    // window routes content back here is device-dependent, so this never fails
    // the test; it just reports what the device did.
    let mut acquired = false;
    for _ in 0..8 {
        match reader.acquire_latest_image() {
            Ok(AcquireResult::Image(img)) => {
                let data = match img.plane_data(0) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let rs = img.plane_row_stride(0).unwrap_or((w * 4) as i32) as usize;
                let (mut min, mut max) = ([255u8; 3], [0u8; 3]);
                for row in 0..h as usize {
                    let base = row * rs;
                    for col in 0..w as usize {
                        let p = base + col * 4;
                        if p + 2 < data.len() {
                            for c in 0..3 {
                                min[c] = min[c].min(data[p + c]);
                                max[c] = max[c].max(data[p + c]);
                            }
                        }
                    }
                }
                let varies = (0..3).any(|c| max[c] > min[c]);
                eprintln!(
                    ">>> read-back from ImageReader: R[{}..{}] G[{}..{}] B[{}..{}] varies={varies}",
                    min[0], max[0], min[1], max[1], min[2], max[2]
                );
                acquired = true;
                break;
            }
            _ => continue,
        }
    }
    if !acquired {
        eprintln!(">>> no image acquired from the ImageReader (present path validated via presented_count)");
    }
    eprintln!(">>> M305 decode -> GPU -> Android surface present validated on device.");
}
