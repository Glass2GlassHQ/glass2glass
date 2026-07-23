//! M304 end-to-end: Android `MediaCodecDec` (GPU output) -> `WgpuPreprocess`,
//! entirely on the GPU. The decoder emits each frame as an RGBA
//! `MemoryDomain::WgpuTexture` (the decoded `AHardwareBuffer` converted through
//! an immutable ycbcr sampler); `WgpuPreprocess` adopts that texture's device and
//! samples it straight into an NCHW f32 tensor, no CPU round-trip of the frame.
//! This closes the decode -> preprocess loop on device.
//!
//! Runs only on `aarch64-linux-android` with the `mediacodec-wgpu` feature; build
//! with cargo-ndk `--platform 26`, push, and run as a bare native binary. The
//! tensor is read back here only to validate it.

#![cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, ConfigureOutcome, Dim, G2gError, Rate, RawVideoFormat, VideoCodec};
use g2g_ml::wgpupreprocess::WgpuPreprocess;
use g2g_plugins::mediacodecdec::MediaCodecDec;

/// Start a binder threadpool so Codec2 can allocate the decoder's output graphic
/// buffers (it calls back over binder). Same shim as the g2g-plugins tests; a
/// bare native binary has none, so the allocate transaction would stall.
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

/// 640x480 H.264 Annex-B fixture, shared with the g2g-plugins on-device tests.
const H264: &[u8] = include_bytes!("../../g2g-plugins/tests/fixtures/h264_640x480.h264");

/// Records every packet an element pushes.
#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Split an H.264 Annex-B stream into access units (first AU carries the
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

#[tokio::test]
async fn decode_gpu_to_preprocess_tensor() {
    start_binder_threadpool();

    let (w, h) = (640u32, 480u32);

    // Decode in GPU-output mode: frames come out as RGBA WgpuTextures.
    let mut dec = MediaCodecDec::h264().with_gpu_output();
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

    // Negotiation (#3): in GPU mode the decoder derives RGBA output.
    match dec.caps_constraint_as_transform() {
        g2g_core::CapsConstraint::DerivedOutput(derive) => {
            let derived = derive(&upstream);
            assert!(
                derived.alternatives().iter().any(|c| matches!(
                    c,
                    Caps::RawVideo {
                        format: RawVideoFormat::Rgba8,
                        ..
                    }
                )),
                "GPU-mode decoder must derive RGBA output, got {:?}",
                derived.alternatives()
            );
        }
        other => panic!("expected DerivedOutput, got {other:?}"),
    }

    let mut dec_sink = Collect::default();
    let mut pts_ns = 0u64;
    for au in access_units(H264) {
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(au.to_vec().into_boxed_slice())),
            timing: FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                capture_ns: pts_ns,
                ..FrameTiming::default()
            },
            sequence: 0,
            meta: Default::default(),
        };
        dec.process(PipelinePacket::DataFrame(frame), &mut dec_sink)
            .await
            .expect("decode AU");
        pts_ns += 33_366_700;
    }
    dec.process(PipelinePacket::Eos, &mut dec_sink)
        .await
        .expect("Eos drains the codec");

    // Configure WgpuPreprocess for RGBA input and feed it the decoded textures.
    let mut pre = WgpuPreprocess::new();
    let rgba_caps = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    };
    // Negotiation (#3): WgpuPreprocess accepts RGBA input.
    assert_eq!(
        pre.intercept_caps(&rgba_caps)
            .expect("preprocess must negotiate RGBA"),
        rgba_caps
    );
    assert!(matches!(
        pre.configure_pipeline(&rgba_caps)
            .expect("configure preprocess for RGBA"),
        ConfigureOutcome::Accepted
    ));

    let texture_frames: Vec<Frame> = dec_sink
        .packets
        .into_iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) if matches!(f.domain, MemoryDomain::WgpuTexture(_)) => {
                Some(f)
            }
            _ => None,
        })
        .collect();
    eprintln!(
        "=== M304 decode->preprocess: {} GPU texture frame(s) ===",
        texture_frames.len()
    );
    assert!(
        !texture_frames.is_empty(),
        "decoder must emit WgpuTexture frames"
    );

    let mut pre_sink = Collect::default();
    let first = texture_frames.into_iter().next().unwrap();
    pre.process(PipelinePacket::DataFrame(first), &mut pre_sink)
        .await
        .expect("preprocess GPU texture");

    // The preprocess output is an NCHW f32 tensor read back to system memory.
    let tensor_bytes = pre_sink.packets.iter().find_map(|p| match p {
        PipelinePacket::DataFrame(f) => f.domain.as_system_slice().map(<[u8]>::to_vec),
        _ => None,
    });
    let tensor_bytes = tensor_bytes.expect("preprocess must emit a System tensor frame");
    let expected_floats = 3 * (w as usize) * (h as usize);
    assert_eq!(
        tensor_bytes.len(),
        expected_floats * 4,
        "tensor byte length (NCHW f32)"
    );

    let floats: Vec<f32> = tensor_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let (mut min, mut max, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
    for &v in &floats {
        assert!(
            (0.0..=1.0).contains(&v),
            "normalized tensor value out of [0,1]: {v}"
        );
        min = min.min(v);
        max = max.max(v);
        sum += v as f64;
    }
    eprintln!(
        "tensor: {} floats, min={min:.3} max={max:.3} mean={:.3}",
        floats.len(),
        sum / floats.len() as f64
    );
    assert!(
        max > min,
        "preprocessed tensor must vary (got a flat tensor)"
    );
    eprintln!(">>> M304 decode -> preprocess (GPU, no CPU frame copy) validated on device.");
}
