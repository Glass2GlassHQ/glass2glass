#![cfg(all(target_os = "linux", feature = "nvdec"))]
//! M1062: the `cuda-device-id` property on the CUDA producers. Each element that
//! creates its own CUDA context (`NvDec`, `CudaUpload`, `LocalCudaSrc`,
//! `FfmpegVideoDec` on the `NvdecCuda` backend, and the wgpu bridge
//! `WgpuToCuda`) opens the device this names instead of a hardcoded ordinal 0,
//! and stamps it onto every frame it emits.
//!
//! ```sh
//! cargo test -p g2g-plugins --features "nvdec cuda-wgpu local-ipc ffmpeg" --test m1062_cuda_device_id -- --nocapture
//! ```
//!
//! The property round-trips are pure CPU. The decode and bridge cases need an
//! NVIDIA GPU and skip (no panic) without one.
//!
//! VERIFY: this host has one GPU, so the decode case can only prove the emitted
//! frames carry ordinal 0 and that a nonexistent ordinal fails the open. Proving
//! a decode really lands on the *second* card (frames stamped 1, pointers valid
//! only in that device's context) needs a multi-GPU host.

use g2g_core::{AsyncElement, Caps, Dim, PropError, PropValue, PropertySpec, Rate, VideoCodec};
use g2g_plugins::cuda::CudaUpload;
use g2g_plugins::nvdec::NvDec;

/// Ordinal every producer opens when nothing sets the property.
const DEFAULT_DEVICE_ID: i32 = 0;
/// An ordinal this host does not have, so opening it must fail.
const MISSING_DEVICE_ID: i32 = 7;
/// `CUDA_ERROR_INVALID_DEVICE`, what `cuDeviceGet` returns for an ordinal the
/// driver does not have.
const CUDA_ERROR_INVALID_DEVICE: i32 = 101;

const PROP: &str = "cuda-device-id";

/// The fixture's geometry (`tests/fixtures/h264_640x480.h264`).
const W: u32 = 640;
const H: u32 = 480;

fn declares(specs: &[PropertySpec], name: &str) -> bool {
    specs.iter().any(|s| s.name == name)
}

fn spec_for(specs: &[PropertySpec], name: &str) -> PropertySpec {
    *specs
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("{name} is declared"))
}

#[test]
fn nvdec_cuda_device_id_round_trips() {
    let mut dec = NvDec::new();
    let spec = spec_for(dec.properties(), PROP);
    // The declared metadata is what `gst-inspect` and the launch parser read, so
    // it has to say what the element actually does.
    assert_eq!(spec.kind, g2g_core::PropKind::Int);
    assert_eq!(spec.default, Some("0"));
    let (min, max) = spec.range.expect("the ordinal range is declared");
    assert_eq!(min, DEFAULT_DEVICE_ID.to_string());
    // cuDeviceGet takes a C int, so that is the top of the range.
    assert_eq!(max, i32::MAX.to_string());
    assert_eq!(
        dec.get_property(PROP),
        Some(PropValue::Int(DEFAULT_DEVICE_ID as i64))
    );
    dec.set_property(PROP, PropValue::Int(3)).unwrap();
    assert_eq!(dec.get_property(PROP), Some(PropValue::Int(3)));
    // The builder is the same knob.
    assert_eq!(
        NvDec::new().with_cuda_device_id(3).get_property(PROP),
        Some(PropValue::Int(3))
    );
    // Beyond a CUDA ordinal's range: refused rather than truncated.
    assert_eq!(
        dec.set_property(PROP, PropValue::Int(i64::from(i32::MAX) + 1)),
        Err(PropError::Value)
    );
    assert_eq!(
        dec.set_property(PROP, PropValue::Uint(1)),
        Err(PropError::Type)
    );
    // No such thing as a negative ordinal, and the range says so.
    assert_eq!(
        dec.set_property(PROP, PropValue::Int(-1)),
        Err(PropError::Value)
    );
    // A refused set leaves the field alone.
    assert_eq!(dec.get_property(PROP), Some(PropValue::Int(3)));
}

#[cfg(feature = "ffmpeg")]
#[test]
fn ffmpegdec_cuda_device_id_round_trips() {
    use g2g_plugins::ffmpegdec::FfmpegVideoDec;

    let mut dec = FfmpegVideoDec::new();
    assert!(declares(dec.properties(), PROP));
    assert_eq!(
        dec.get_property(PROP),
        Some(PropValue::Int(DEFAULT_DEVICE_ID as i64))
    );
    dec.set_property(PROP, PropValue::Int(1)).unwrap();
    assert_eq!(dec.get_property(PROP), Some(PropValue::Int(1)));
    // The property and the builder write the same field, which the NvdecCuda
    // hwdevice is created on.
    assert_eq!(
        FfmpegVideoDec::new()
            .with_cuda_device_id(2)
            .cuda_device_id(),
        2
    );
    assert_eq!(
        dec.set_property(PROP, PropValue::Int(-1)),
        Err(PropError::Value)
    );
}

#[test]
fn cuda_upload_cuda_device_id_round_trips() {
    let mut upload = CudaUpload::new();
    assert!(declares(upload.properties(), PROP));
    assert_eq!(
        upload.get_property(PROP),
        Some(PropValue::Int(DEFAULT_DEVICE_ID as i64))
    );
    upload.set_property(PROP, PropValue::Int(2)).unwrap();
    assert_eq!(upload.get_property(PROP), Some(PropValue::Int(2)));
    assert_eq!(
        CudaUpload::new().with_cuda_device_id(1).get_property(PROP),
        Some(PropValue::Int(1))
    );
}

#[cfg(feature = "local-ipc")]
#[test]
fn local_cuda_src_cuda_device_id_round_trips() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::localcuda::LocalCudaSrc;

    let mut src = LocalCudaSrc::new("/tmp/g2g-m1062.sock");
    assert!(declares(src.properties(), PROP));
    assert_eq!(
        src.get_property(PROP),
        Some(PropValue::Int(DEFAULT_DEVICE_ID as i64))
    );
    src.set_property(PROP, PropValue::Int(1)).unwrap();
    assert_eq!(src.get_property(PROP), Some(PropValue::Int(1)));
    assert_eq!(
        LocalCudaSrc::new("/tmp/g2g-m1062.sock")
            .with_cuda_device_id(2)
            .get_property(PROP),
        Some(PropValue::Int(2))
    );
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// Annex-B access-unit splitter: a new AU starts at the first VCL slice once the
/// current one already has one.
fn split_access_units(bs: &[u8]) -> Vec<Vec<u8>> {
    let mut codes: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i + 3 <= bs.len() {
        if bs[i] == 0 && bs[i + 1] == 0 && bs[i + 2] == 1 {
            codes.push((i, i + 3));
            i += 3;
        } else if i + 4 <= bs.len()
            && bs[i] == 0
            && bs[i + 1] == 0
            && bs[i + 2] == 0
            && bs[i + 3] == 1
        {
            codes.push((i, i + 4));
            i += 4;
        } else {
            i += 1;
        }
    }
    let mut aus = Vec::new();
    let mut start: Option<usize> = None;
    let mut has_vcl = false;
    for &(sc, nal) in &codes {
        let is_vcl = (1..=5).contains(&(bs[nal] & 0x1f));
        if is_vcl && has_vcl {
            aus.push(bs[start.take().unwrap()..sc].to_vec());
            has_vcl = false;
        }
        if start.is_none() {
            start = Some(sc);
        }
        has_vcl |= is_vcl;
    }
    if let Some(s) = start {
        aus.push(bs[s..].to_vec());
    }
    aus
}

fn fixture_access_units() -> Vec<Vec<u8>> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/h264_640x480.h264"
    );
    let bs = std::fs::read(path).expect("read committed H.264 fixture");
    let aus = split_access_units(&bs);
    assert!(!aus.is_empty(), "no access units in fixture");
    aus
}

/// Records the device ordinal of every CUDA frame that reaches it.
#[derive(Default)]
struct OrdinalSink {
    ordinals: Vec<i32>,
}

impl g2g_core::OutputSink for OrdinalSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<g2g_core::PipelinePacket>,
    ) -> core::task::Poll<Result<g2g_core::PushOutcome, g2g_core::G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        if let g2g_core::PipelinePacket::DataFrame(f) = packet {
            if let g2g_core::MemoryDomain::Cuda(buf) = &f.domain {
                self.ordinals.push(buf.device_ordinal);
            }
        }
        core::task::Poll::Ready(Ok(g2g_core::PushOutcome::Accepted))
    }
}

#[tokio::test]
async fn nvdec_stamps_the_configured_device_on_every_frame() {
    use g2g_core::{Frame, FrameTiming, MemoryDomain, PipelinePacket};

    let mut dec = NvDec::new();
    dec.set_property(PROP, PropValue::Int(DEFAULT_DEVICE_ID as i64))
        .unwrap();
    if let Err(e) = dec.configure_pipeline(&h264_caps()) {
        eprintln!("skipping: NVDEC unavailable ({e:?})");
        return;
    }

    let mut sink = OrdinalSink::default();
    for (seq, au) in fixture_access_units().into_iter().enumerate() {
        let frame = Frame {
            domain: MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
                au.into_boxed_slice(),
            )),
            timing: FrameTiming {
                pts_ns: seq as u64 * 33_000_000,
                ..FrameTiming::default()
            },
            sequence: seq as u64,
            meta: Default::default(),
        };
        dec.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .expect("decode");
    }
    dec.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("flush");

    assert!(
        !sink.ordinals.is_empty(),
        "no CUDA frames decoded from the fixture"
    );
    assert!(
        sink.ordinals.iter().all(|&o| o == DEFAULT_DEVICE_ID),
        "frames carry the wrong device ordinal: {:?}",
        sink.ordinals
    );
}

#[test]
fn nvdec_fails_to_open_a_device_the_host_does_not_have() {
    // Only meaningful where CUDA itself works: on a host with no driver every
    // ordinal fails and the check proves nothing.
    if NvDec::new().configure_pipeline(&h264_caps()).is_err() {
        eprintln!("skipping: NVDEC unavailable on device {DEFAULT_DEVICE_ID}");
        return;
    }
    let mut dec = NvDec::new().with_cuda_device_id(MISSING_DEVICE_ID);
    let err = dec
        .configure_pipeline(&h264_caps())
        .expect_err("device 7 does not exist");
    assert_eq!(
        err,
        g2g_core::G2gError::Hardware(g2g_core::HardwareError::Cuda(CUDA_ERROR_INVALID_DEVICE))
    );
}

#[test]
fn nvdec_refuses_a_device_change_once_the_context_is_open() {
    let mut dec = NvDec::new();
    if dec.configure_pipeline(&h264_caps()).is_err() {
        eprintln!("skipping: NVDEC unavailable on device {DEFAULT_DEVICE_ID}");
        return;
    }
    assert_eq!(
        dec.set_property(PROP, PropValue::Int(1)),
        Err(PropError::ReadOnly)
    );
    assert_eq!(
        dec.get_property(PROP),
        Some(PropValue::Int(DEFAULT_DEVICE_ID as i64))
    );
}

/// The launch parser reads the kind out of `properties()` and then calls
/// `set_property`, so a line that parses proves both halves: a value the element
/// rejects has to come back as a parse error.
#[test]
fn nvdec_takes_cuda_device_id_from_a_launch_line() {
    use g2g_core::runtime::parse_launch;

    use g2g_core::runtime::GraphNodeRef;

    let registry = g2g_plugins::registry::default_registry();
    let graph = parse_launch(
        &registry,
        "filesrc location=/dev/null ! nvdec name=dec cuda-device-id=3 ! fakesink",
    )
    .expect("nvdec cuda-device-id=3 parses");
    let node = graph
        .node_by_name("dec")
        .expect("the line named the decoder");
    let Some(GraphNodeRef::Element(dec)) = graph.element(node) else {
        panic!("the decoder node carries an element");
    };
    assert_eq!(dec.get_property(PROP), Some(PropValue::Int(3)));

    parse_launch(
        &registry,
        "filesrc location=/dev/null ! nvdec cuda-device-id=99999999999 ! fakesink",
    )
    .expect_err("an ordinal beyond i32 is refused by set_property");
    parse_launch(
        &registry,
        "filesrc location=/dev/null ! nvdec cuda-device-id=-1 ! fakesink",
    )
    .expect_err("a negative ordinal is refused by set_property");
}

#[cfg(feature = "cuda-wgpu")]
#[tokio::test]
async fn wgpu_to_cuda_reports_its_device_and_refuses_a_change() {
    use g2g_plugins::cudawgpu::{create_interop_device, WgpuToCuda};

    const W: u32 = 320;
    const H: u32 = 240;

    let dev = match create_interop_device().await {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no Vulkan interop device ({e:?})");
            return;
        }
    };
    // SAFETY: `create_interop_device` opens a VK_KHR_external_memory_fd device;
    // the clones share it (wgpu handles are Arc-backed) and `dev` outlives both
    // bridges.
    let missing = unsafe {
        WgpuToCuda::new(
            dev.device.clone(),
            dev.queue.clone(),
            W,
            H,
            MISSING_DEVICE_ID,
        )
    };
    match missing {
        Ok(_) => panic!("device {MISSING_DEVICE_ID} does not exist"),
        Err(e) => assert_eq!(
            e,
            g2g_core::G2gError::Hardware(g2g_core::HardwareError::Cuda(CUDA_ERROR_INVALID_DEVICE))
        ),
    }

    // SAFETY: as above.
    let mut bridge = match unsafe {
        WgpuToCuda::new(
            dev.device.clone(),
            dev.queue.clone(),
            W,
            H,
            DEFAULT_DEVICE_ID,
        )
    } {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: WgpuToCuda unavailable (no CUDA? {e:?})");
            return;
        }
    };
    assert_eq!(
        bridge.get_property(PROP),
        Some(PropValue::Int(DEFAULT_DEVICE_ID as i64))
    );
    // The context is retained in `new`, so there is no later point a set could
    // take effect.
    assert_eq!(
        bridge.set_property(PROP, PropValue::Int(1)),
        Err(PropError::ReadOnly)
    );
}

/// The NVDEC-through-libavcodec producer: the hwdevice must open on the ordinal
/// the property names, and stamp it onto the frames it emits.
#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn ffmpegdec_nvdec_cuda_stamps_the_configured_device() {
    use g2g_core::{Frame, FrameTiming, MemoryDomain, PipelinePacket};
    use g2g_plugins::ffmpegdec::{Backend, FfmpegVideoDec};

    let mut dec = FfmpegVideoDec::new().with_backend(Backend::NvdecCuda);
    dec.set_property(PROP, PropValue::Int(DEFAULT_DEVICE_ID as i64))
        .unwrap();
    if let Err(e) = dec.configure_pipeline(&h264_caps()) {
        eprintln!("skipping: ffmpeg NvdecCuda unavailable ({e:?})");
        return;
    }

    let mut sink = OrdinalSink::default();
    let mut fed = 0usize;
    for (seq, au) in fixture_access_units().into_iter().enumerate() {
        let frame = Frame {
            domain: MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
                au.into_boxed_slice(),
            )),
            timing: FrameTiming {
                pts_ns: seq as u64 * 33_000_000,
                ..FrameTiming::default()
            },
            sequence: seq as u64,
            meta: Default::default(),
        };
        // The hwdevice is created lazily on the first access unit, so a bad
        // ordinal surfaces here rather than at configure.
        if let Err(e) = dec
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
        {
            eprintln!("skipping: ffmpeg NvdecCuda decode unavailable ({e:?})");
            return;
        }
        fed += 1;
    }
    assert!(fed > 0, "no access units in the fixture");
    dec.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("flush");

    assert!(
        !sink.ordinals.is_empty(),
        "no CUDA frames decoded from the fixture"
    );
    assert!(
        sink.ordinals.iter().all(|&o| o == DEFAULT_DEVICE_ID),
        "frames carry the wrong device ordinal: {:?}",
        sink.ordinals
    );
}

/// The ordinal really reaches libavcodec's hwdevice: one the host does not have
/// fails the decoder open instead of quietly landing on device 0.
#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn ffmpegdec_nvdec_cuda_fails_on_a_device_the_host_does_not_have() {
    use g2g_core::{Frame, FrameTiming, MemoryDomain, PipelinePacket};
    use g2g_plugins::ffmpegdec::{Backend, FfmpegVideoDec};

    let first_au = fixture_access_units().into_iter().next().expect("an AU");
    let au_frame = || Frame {
        domain: MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
            first_au.clone().into_boxed_slice(),
        )),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    };

    // Only meaningful where the CUDA hwdevice works at all on device 0.
    let mut ok = FfmpegVideoDec::new().with_backend(Backend::NvdecCuda);
    let mut sink = OrdinalSink::default();
    if ok.configure_pipeline(&h264_caps()).is_err()
        || ok
            .process(PipelinePacket::DataFrame(au_frame()), &mut sink)
            .await
            .is_err()
    {
        eprintln!("skipping: ffmpeg NvdecCuda unavailable on device {DEFAULT_DEVICE_ID}");
        return;
    }

    let mut dec = FfmpegVideoDec::new()
        .with_backend(Backend::NvdecCuda)
        .with_cuda_device_id(MISSING_DEVICE_ID);
    dec.configure_pipeline(&h264_caps()).expect("configure");
    dec.process(PipelinePacket::DataFrame(au_frame()), &mut sink)
        .await
        .expect_err("device 7 does not exist");
}
