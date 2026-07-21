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
use g2g_core::{Caps, ConfigureOutcome, Dim, G2gError, Rate, RawVideoFormat, VideoCodec};
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

/// Short Annex-B clips (640x480, 10 frames: parameter sets + IDR + inter frames),
/// committed under `tests/fixtures/` and embedded so the test binary needs no
/// companion file on the device. 640x480 (not a tiny size) because hardware
/// decoders' graphic-buffer allocators reject sub-minimum dimensions.
const H264: &[u8] = include_bytes!("fixtures/h264_640x480.h264");
const H265: &[u8] = include_bytes!("fixtures/h265_640x480.h265");

/// `OutputSink` that records every packet the decoder pushes (CapsChanged +
/// DataFrames), the same collector the other decode smoke tests use.
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

/// Split an Annex-B byte stream into access units. A new AU begins at the first
/// VCL NAL once the current AU already holds one, so each AU groups its leading
/// parameter sets / SEI with the slice. This matters because `MediaCodecDec`
/// reads the parameter sets from the AU it is fed: the first AU must carry them
/// alongside the IDR for the codec to configure. Returns contiguous sub-slices
/// of `s` (NALs are already contiguous in the stream). `codec` selects the NAL
/// header parse: H.264 uses the low 5 bits (VCL = 1..=5); H.265 uses bits 1..=6
/// (VCL = 0..=31).
fn access_units(s: &[u8], codec: VideoCodec) -> Vec<&[u8]> {
    let is_h265 = codec == VideoCodec::H265;
    let nal_type = |hdr: u8| {
        if is_h265 {
            (hdr >> 1) & 0x3f
        } else {
            hdr & 0x1f
        }
    };
    let is_vcl = |t: u8| {
        if is_h265 {
            t <= 31
        } else {
            (1..=5).contains(&t)
        }
    };

    // Offsets of each NAL's start code, paired with the NAL type.
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

    // Byte offset where each access unit begins.
    let mut starts: Vec<usize> = Vec::new();
    let mut has_vcl = false;
    for &(off, t) in &nals {
        let vcl = is_vcl(t);
        if starts.is_empty() {
            starts.push(off);
            has_vcl = vcl;
        } else if vcl && has_vcl {
            starts.push(off); // new frame: start a new AU
        } else if vcl {
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

/// Drive `dec` with the access units of `stream` and assert it emits NV12 frames
/// of the right geometry. Shared by the H.264 and H.265 tests.
async fn decode_to_nv12(mut dec: MediaCodecDec, stream: &[u8], codec: VideoCodec, w: u32, h: u32) {
    start_binder_threadpool();

    let aus = access_units(stream, codec);
    assert!(
        aus.len() > 1,
        "fixture must split into multiple access units, got {}",
        aus.len()
    );

    // Negotiation surrogate: upstream caps carry the fixture's geometry, as a
    // parser would emit from the SPS. MediaCodec's `configure()` requires
    // width/height (it returns EINVAL without them), unlike the codec-private
    // parameter sets which ride the stream.
    let upstream = Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    };
    let narrowed = dec.intercept_caps(&upstream).expect("intercept caps");
    let outcome = dec
        .configure_pipeline(&narrowed)
        .expect("configure decoder");
    assert!(matches!(outcome, ConfigureOutcome::Accepted));

    let mut sink = Collect::default();

    // M756: the graph runner hands each element its own solved OUTPUT caps as an
    // incoming `CapsChanged` before the first frame (transform_arm in
    // graph_runner.rs sends `process(CapsChanged(forward_caps))`, where
    // forward_caps is the derived NV12 output). The element must accept and
    // forward it, not reject it as an unrecognised input shape. This drives that
    // packet exactly as the runner would; a decoder that only accepts its
    // compressed input shape returns a bare `CapsMismatch` here.
    let prefixed_out = Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    };
    dec.process(PipelinePacket::CapsChanged(prefixed_out.clone()), &mut sink)
        .await
        .expect("decoder must accept the runner's pre-fixed output CapsChanged");
    assert!(
        matches!(sink.packets.first(), Some(PipelinePacket::CapsChanged(c)) if *c == prefixed_out),
        "the pre-fixed output caps must be forwarded downstream before any frame, got {:?}",
        sink.packets.first()
    );

    let mut pts_ns = 0u64;
    for au in aus {
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
        dec.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .expect("process access unit");
        pts_ns += 33_366_700; // ~29.97 fps
    }
    // EOS drains any frames the codec is still holding.
    dec.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("process Eos drains the codec");

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

    eprintln!(
        "{:?}: decoded {} frame(s); {} CapsChanged emitted",
        codec,
        data_frames.len(),
        caps_changes.len()
    );
    assert!(
        !caps_changes.is_empty(),
        "expected at least one NV12 CapsChanged"
    );
    assert!(
        !data_frames.is_empty(),
        "expected at least one decoded frame"
    );

    // The first CapsChanged advertises the decoded NV12 geometry; the first
    // frame's buffer must match it (Y + interleaved UV = w*h*3/2).
    match caps_changes.first().unwrap() {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(fw),
            height: Dim::Fixed(fh),
            ..
        } => {
            eprintln!("first NV12 caps: {}x{}", fw, fh);
            assert!(*fw > 0 && *fh > 0);
            let expected = (*fw as usize) * (*fh as usize) * 3 / 2;
            match &data_frames.first().unwrap().domain {
                MemoryDomain::System(slice) => {
                    assert_eq!(
                        slice.as_slice().len(),
                        expected,
                        "NV12 byte length mismatch"
                    );
                }
                _ => panic!("decoder must emit System-domain NV12 frames"),
            }
        }
        other => panic!("expected NV12 fixed caps, got {:?}", other),
    }
}

#[tokio::test]
async fn mediacodec_decodes_h264_to_nv12() {
    decode_to_nv12(MediaCodecDec::h264(), H264, VideoCodec::H264, 640, 480).await;
}

#[tokio::test]
async fn mediacodec_decodes_h265_to_nv12() {
    decode_to_nv12(MediaCodecDec::h265(), H265, VideoCodec::H265, 640, 480).await;
}
