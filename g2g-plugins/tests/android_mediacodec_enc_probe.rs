//! M306: on-device probe for the Android MediaCodec H.264 encoder.
//!
//! Synthesises NV12 frames (a moving gradient so the encoder has real content),
//! drives `MediaCodecEnc`, and asserts the emitted elementary stream is a valid
//! self-contained Annex-B H.264 stream: parameter sets (SPS + PPS) prepended to
//! the first key frame, an IDR slice, and several frames out for the frames fed
//! in. This is the encode mirror of `android_mediacodec_smoke` (which decodes).
//!
//! Runs only on `aarch64-linux-android` (et al.) with the `mediacodec` feature;
//! compiles to nothing on the dev host. Build with cargo-ndk `--platform 24`,
//! push, run as a bare native binary. See `tools/android-mediacodec-enc-smoke.sh`.

#![cfg(all(target_os = "android", feature = "mediacodec"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, ConfigureOutcome, Dim, G2gError, RawVideoFormat, Rate};
use g2g_plugins::mediacodecenc::MediaCodecEnc;

/// Start a binder threadpool so Codec2 can allocate the encoder's buffers (it
/// calls back over binder into this process). A bare native binary has no
/// threadpool, so the allocation transaction stalls; the `ABinderProcess_*`
/// symbols live in the device's libbinder_ndk.so but not the NDK link stub, so
/// dlsym them. Same shim as the decode probes.
fn start_binder_threadpool() {
    use core::ffi::{c_char, c_int, c_void};
    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    const RTLD_NOW: c_int = 2;
    // SAFETY: libbinder_ndk.so is loadable; the dlsym'd symbols have the C
    // signatures from <android/binder_process.h>.
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

/// Records every packet the encoder pushes, to inspect the encoded frames.
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

/// A `w x h` NV12 frame: a diagonal luma gradient shifted by `t` (so successive
/// frames differ and the encoder produces real inter frames), flat 128 chroma.
fn nv12_frame(w: usize, h: usize, t: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(w * h * 3 / 2);
    for y in 0..h {
        for x in 0..w {
            buf.push(((x + y + t * 8) & 0xff) as u8);
        }
    }
    buf.resize(w * h + w * h / 2, 128);
    buf
}

/// Split an H.264 Annex-B stream into (nal_type, len) pairs.
fn nal_types(s: &[u8]) -> Vec<u8> {
    let mut types = Vec::new();
    let mut i = 0;
    while i + 3 <= s.len() {
        let sc = if i + 4 <= s.len() && s[i] == 0 && s[i + 1] == 0 && s[i + 2] == 0 && s[i + 3] == 1 {
            Some(4)
        } else if s[i] == 0 && s[i + 1] == 0 && s[i + 2] == 1 {
            Some(3)
        } else {
            None
        };
        match sc {
            Some(len) => {
                let hdr = i + len;
                if hdr < s.len() {
                    types.push(s[hdr] & 0x1f);
                }
                i = hdr + 1;
            }
            None => i += 1,
        }
    }
    types
}

#[tokio::test]
async fn encode_nv12_to_annexb_h264() {
    start_binder_threadpool();
    let (w, h) = (640u32, 480u32);
    let frames_in = 30usize;

    let mut enc = MediaCodecEnc::h264().with_bitrate(2_000_000).with_framerate(30);
    let caps = Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    };
    let narrowed = enc.intercept_caps(&caps).expect("intercept caps");
    assert!(matches!(
        enc.configure_pipeline(&narrowed).expect("configure encoder"),
        ConfigureOutcome::Accepted
    ));

    let mut sink = Collect::default();
    let mut pts_ns = 0u64;
    for t in 0..frames_in {
        let nv12 = nv12_frame(w as usize, h as usize, t);
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(nv12.into_boxed_slice())),
            timing: FrameTiming { pts_ns, dts_ns: pts_ns, capture_ns: pts_ns, ..FrameTiming::default() },
            sequence: t as u64,
            meta: Default::default(),
        };
        enc.process(PipelinePacket::DataFrame(frame), &mut sink).await.expect("encode frame");
        pts_ns += 33_366_700;
    }
    enc.process(PipelinePacket::Eos, &mut sink).await.expect("Eos drains the encoder");

    // Collect the encoded access units, in order.
    let aus: Vec<&Frame> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();
    eprintln!("=== M306 encode: {} access unit(s) out for {frames_in} frame(s) in ===", aus.len());
    assert!(!aus.is_empty(), "encoder produced no access units");

    // The first key frame must carry SPS (7) + PPS (8) ahead of an IDR (5).
    let first_key = aus.iter().find(|f| f.timing.keyframe).expect("at least one key frame");
    let MemoryDomain::System(slice) = &first_key.domain else { panic!("system memory") };
    let types = nal_types(slice.as_slice());
    eprintln!(">>> first key-frame NAL types: {types:?}");
    assert!(types.contains(&7), "key frame must contain SPS (got {types:?})");
    assert!(types.contains(&8), "key frame must contain PPS (got {types:?})");
    assert!(types.contains(&5), "key frame must contain an IDR slice (got {types:?})");

    let total_bytes: usize = aus
        .iter()
        .map(|f| match &f.domain {
            MemoryDomain::System(s) => s.as_slice().len(),
            _ => 0,
        })
        .sum();
    eprintln!(">>> {} encoded bytes total; {} key frame(s)", total_bytes, aus.iter().filter(|f| f.timing.keyframe).count());
    assert!(total_bytes > 0, "encoded stream is empty");
    eprintln!(">>> M306 NV12 -> Annex-B H.264 encode validated on device.");
}
