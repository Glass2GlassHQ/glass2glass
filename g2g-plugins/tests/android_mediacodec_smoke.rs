//! M219: on-device smoke test for `MediaCodecDec` (Android NDK MediaCodec).
//!
//! Unlike the cross-compile `cargo check` CI runs, this actually decodes on a
//! real device: it feeds an embedded H.264 Annex-B clip through the NDK
//! MediaCodec H.264 decoder and asserts it emits NV12 frames of the right
//! geometry. The H.264 fixture is `include_bytes!`d (so the test binary is
//! self-contained: nothing extra to push to the device), and the decoder is
//! headless (configured with no Surface, ByteBuffer output), so this runs as a
//! plain native binary with no APK / Activity.
//!
//! It only builds for `aarch64-linux-android` (et al.) with the `mediacodec`
//! feature; on the dev host it compiles to nothing. Run it on a connected
//! device with `tools/android-mediacodec-smoke.sh` (cargo-ndk build + adb push +
//! run), or via cargo-dinghy: `cargo dinghy -d android test -p g2g-plugins
//! --features mediacodec`.

#![cfg(all(target_os = "android", feature = "mediacodec"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, ConfigureOutcome, Dim, G2gError, RawVideoFormat, Rate, VideoCodec};
use g2g_plugins::mediacodecdec::MediaCodecDec;

/// Start a binder threadpool so Codec2 can allocate the decoder's output graphic
/// buffers, which it does by calling back into this process over binder
/// (IGraphicBufferAllocator). An Android app gets a threadpool from the
/// framework; a bare native binary has none, so the allocate transaction has no
/// thread to service it and the codec stalls. The `ABinderProcess_*` functions
/// live in the device's libbinder_ndk.so but are not in the NDK link stub
/// (platform/LLNDK only), so resolve them at runtime with dlsym.
fn start_binder_threadpool() {
    use core::ffi::{c_char, c_int, c_void};
    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    const RTLD_NOW: c_int = 2;
    // SAFETY: libbinder_ndk.so is already loaded by MediaCodec; the dlsym'd
    // symbols have the C signatures declared in <android/binder_process.h>.
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

/// A short H.264 Annex-B clip (640x480, 10 frames, baseline: SPS/PPS + IDR +
/// P-frames), committed under `tests/fixtures/` and embedded so the test binary
/// needs no companion file on the device. 640x480 (not a tiny size) because
/// hardware decoders' graphic-buffer allocators reject sub-minimum dimensions.
const H264: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

/// `OutputSink` that records every packet the decoder pushes (CapsChanged +
/// DataFrames), the same collector the other decode smoke tests use.
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

/// Split an Annex-B byte stream into access units. A new AU begins at the first
/// VCL NAL (types 1..=5) once the current AU already holds one, so each AU
/// groups its leading SPS/PPS/SEI with the slice. This matters because
/// `MediaCodecDec` reads the parameter sets from the AU it is fed: the first AU
/// must carry SPS+PPS+IDR together for the codec to configure. Returns
/// contiguous sub-slices of `s` (NALs are already contiguous in the stream).
fn access_units(s: &[u8]) -> Vec<&[u8]> {
    // Offsets of each NAL's start code, paired with the NAL type.
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
                    nals.push((i, s[hdr] & 0x1f));
                }
                i = hdr + 1;
            }
            None => i += 1,
        }
    }

    // Byte offset where each access unit begins.
    let mut starts: Vec<usize> = Vec::new();
    let mut has_vcl = false;
    for &(off, t) in &nals {
        let is_vcl = (1..=5).contains(&t);
        if starts.is_empty() {
            starts.push(off);
            has_vcl = is_vcl;
        } else if is_vcl && has_vcl {
            starts.push(off); // new frame: start a new AU
        } else if is_vcl {
            has_vcl = true; // first VCL of the current AU
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
async fn mediacodec_decodes_h264_to_nv12() {
    start_binder_threadpool();

    let aus = access_units(H264);
    assert!(aus.len() > 1, "fixture must split into multiple access units, got {}", aus.len());

    let mut dec = MediaCodecDec::h264();

    // Negotiation surrogate: upstream H.264 with the fixture's geometry, as
    // `h264parse` would emit from the SPS. MediaCodec's `configure()` requires
    // width/height (it returns EINVAL without them), so the geometry is not
    // optional here, unlike the codec-private SPS/PPS which ride the stream.
    let upstream = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Any,
    };
    let narrowed = dec.intercept_caps(&upstream).expect("intercept H.264");
    let outcome = dec.configure_pipeline(&narrowed).expect("configure H.264 decoder");
    assert!(matches!(outcome, ConfigureOutcome::Accepted));

    let mut sink = Collect::default();
    let mut pts_ns = 0u64;
    for au in aus {
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(au.to_vec().into_boxed_slice())),
            timing: FrameTiming { pts_ns, dts_ns: pts_ns, capture_ns: pts_ns, ..FrameTiming::default() },
            sequence: 0,
            meta: Default::default(),
        };
        dec.process(PipelinePacket::DataFrame(frame), &mut sink).await.expect("process access unit");
        pts_ns += 33_366_700; // ~29.97 fps
    }
    // EOS drains any frames the codec is still holding.
    dec.process(PipelinePacket::Eos, &mut sink).await.expect("process Eos drains the codec");

    let caps_changes: Vec<_> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c.clone()),
            _ => None,
        })
        .collect();
    let data_frames: Vec<_> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();

    eprintln!("decoded {} frame(s); {} CapsChanged emitted", data_frames.len(), caps_changes.len());
    assert!(!caps_changes.is_empty(), "expected at least one NV12 CapsChanged");
    assert!(!data_frames.is_empty(), "expected at least one decoded frame");

    // The first CapsChanged advertises the decoded NV12 geometry; the first
    // frame's buffer must match it (Y + interleaved UV = w*h*3/2). On a device
    // whose decoder emits a vendor / flexible colour format the element does not
    // yet repack, this length check is exactly what would catch it.
    match caps_changes.first().unwrap() {
        Caps::RawVideo { format: RawVideoFormat::Nv12, width: Dim::Fixed(w), height: Dim::Fixed(h), .. } => {
            eprintln!("first NV12 caps: {}x{}", w, h);
            assert!(*w > 0 && *h > 0);
            let expected = (*w as usize) * (*h as usize) * 3 / 2;
            match &data_frames.first().unwrap().domain {
                MemoryDomain::System(slice) => {
                    assert_eq!(slice.as_slice().len(), expected, "NV12 byte length mismatch");
                }
                _ => panic!("decoder must emit System-domain NV12 frames"),
            }
        }
        other => panic!("expected NV12 fixed caps, got {:?}", other),
    }
}
