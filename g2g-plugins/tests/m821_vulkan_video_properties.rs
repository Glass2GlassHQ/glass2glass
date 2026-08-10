//! M821: runtime properties on `VulkanVideoDec` (`low-latency`, `device-index`,
//! `num-dpb-slots`).
//!
//! Each must be declared (so `parse_launch` can look up its kind), round-trip
//! through `set_property` / `get_property`, and actually change what the element
//! does: the GPU it opens, the DPB it allocates, and whether decoded pictures are
//! held for display-order reordering. The behavioural halves run on the RTX 3060
//! and skip with no Vulkan decode support.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::block_on;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PropError, PropValue, PropertySpec, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, open_decode_device_at, open_h264_decode_device, Nv12Frame,
    VulkanVideoCodec, VulkanVideoDec, VulkanVideoError,
};

const BFRAMES: &[u8] = include_bytes!("fixtures/h264_640x480_bframes.h264");

fn declares(specs: &[PropertySpec], name: &str) -> bool {
    specs.iter().any(|s| s.name == name)
}

#[derive(Default)]
struct RecordingSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for RecordingSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Byte offsets of each NAL payload (just past its start code).
fn start_code_offsets(data: &[u8]) -> Vec<usize> {
    let mut offs = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                offs.push(i + 3);
                i += 3;
                continue;
            }
            if i + 4 <= data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                offs.push(i + 4);
                i += 4;
                continue;
            }
        }
        i += 1;
    }
    offs
}

/// Split an Annex-B stream into per-picture access units (the fixture is
/// single-slice, so each VCL NAL closes one).
fn split_access_units(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut units = Vec::new();
    let mut cur = Vec::new();
    let starts = start_code_offsets(stream);
    for (k, &begin) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(stream.len());
        let nal = &stream[begin..end];
        cur.extend_from_slice(&[0, 0, 0, 1]);
        cur.extend_from_slice(nal);
        if matches!(nal.first().map(|b| b & 0x1F), Some(1..=5)) {
            units.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        if let Some(last) = units.last_mut() {
            last.extend_from_slice(&cur);
        }
    }
    units
}

fn au_frame(bytes: Vec<u8>, seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: seq * 33_000_000,
            ..Default::default()
        },
        sequence: seq,
        meta: Default::default(),
    }
}

fn planes(f: &Nv12Frame) -> Vec<u8> {
    let mut v = Vec::with_capacity(f.luma.len() + f.chroma.len());
    v.extend_from_slice(&f.luma);
    v.extend_from_slice(&f.chroma);
    v
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Emitted NV12 buffers, one entry per `process` call (so a caller can see how
/// many frames each access unit released).
fn drive_per_au(dec: &mut VulkanVideoDec, aus: Vec<Vec<u8>>) -> Vec<Vec<Vec<u8>>> {
    let mut per_au = Vec::new();
    for (i, au) in aus.into_iter().enumerate() {
        let mut sink = RecordingSink::default();
        block_on(dec.process(PipelinePacket::DataFrame(au_frame(au, i as u64)), &mut sink))
            .expect("decode access unit");
        per_au.push(nv12_buffers(&sink));
    }
    let mut sink = RecordingSink::default();
    block_on(dec.process(PipelinePacket::Eos, &mut sink)).expect("flush at eos");
    per_au.push(nv12_buffers(&sink));
    per_au
}

fn nv12_buffers(sink: &RecordingSink) -> Vec<Vec<u8>> {
    sink.packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(Frame {
                domain: MemoryDomain::System(s),
                ..
            }) => Some(s.as_slice().to_vec()),
            _ => None,
        })
        .collect()
}

/// Skip (rather than fail) on a host with no Vulkan H.264 decode.
fn h264_device_or_skip(what: &str) -> Option<g2g_plugins::vulkanvideo::VulkanVideoDevice> {
    match block_on(open_h264_decode_device()) {
        Ok(d) => Some(d),
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m821 {what}: no Vulkan H.264 decode adapter");
            None
        }
        Err(e) => panic!("open h264 device: {e:?}"),
    }
}

#[test]
fn declares_and_round_trips_the_three_properties() {
    let mut e = VulkanVideoDec::new();
    for name in ["low-latency", "device-index", "num-dpb-slots"] {
        assert!(declares(e.properties(), name), "{name} is declared");
    }
    // Codec and output format come from negotiation, never from a property.
    assert!(!declares(e.properties(), "codec"));
    assert!(!declares(e.properties(), "format"));

    assert_eq!(e.get_property("low-latency"), Some(PropValue::Bool(false)));
    e.set_property("low-latency", PropValue::Bool(true))
        .unwrap();
    assert_eq!(e.get_property("low-latency"), Some(PropValue::Bool(true)));

    // Unset device-index means the default pick, which no index names.
    assert_eq!(e.get_property("device-index"), None);
    e.set_property("device-index", PropValue::Uint(1)).unwrap();
    assert_eq!(e.get_property("device-index"), Some(PropValue::Uint(1)));

    assert_eq!(e.get_property("num-dpb-slots"), Some(PropValue::Uint(0)));
    e.set_property("num-dpb-slots", PropValue::Uint(12))
        .unwrap();
    assert_eq!(e.get_property("num-dpb-slots"), Some(PropValue::Uint(12)));
    // 0 goes back to sizing the DPB from the stream.
    e.set_property("num-dpb-slots", PropValue::Uint(0)).unwrap();
    assert_eq!(e.get_property("num-dpb-slots"), Some(PropValue::Uint(0)));
}

#[test]
fn rejects_bad_property_values() {
    let mut e = VulkanVideoDec::new();
    assert_eq!(
        e.set_property("low-latency", PropValue::Uint(1)),
        Err(PropError::Type)
    );
    assert_eq!(
        e.set_property("device-index", PropValue::Uint(u64::from(u32::MAX) + 1)),
        Err(PropError::Value),
        "a device index is 32 bits"
    );
    assert_eq!(
        e.set_property("num-dpb-slots", PropValue::Uint(64)),
        Err(PropError::Value),
        "past any codec's DPB size"
    );
    assert_eq!(
        e.set_property("bogus", PropValue::Uint(1)),
        Err(PropError::Unknown)
    );
}

#[test]
fn parse_launch_sets_vulkanvideodec_properties() {
    use g2g_core::runtime::parse_launch;
    use g2g_plugins::registry::default_registry;
    let reg = default_registry();
    assert!(
        parse_launch(
            &reg,
            "filesrc location=in.h264 ! h264parse ! vulkanvideodec low-latency=true num-dpb-slots=8 device-index=0 ! fakesink"
        )
        .is_ok(),
        "a launch line setting the new properties parses"
    );
    assert!(
        parse_launch(
            &reg,
            "filesrc location=in.h264 ! h264parse ! vulkanvideodec bogus=1 ! fakesink"
        )
        .is_err(),
        "an undeclared property is rejected"
    );
}

/// `device-index` picks the GPU the decode device is opened on: index 0 is the
/// first decode-capable adapter, and an index past the last one fails rather
/// than silently opening a different GPU.
#[test]
fn device_index_selects_the_decode_gpu() {
    if h264_device_or_skip("device-index").is_none() {
        return;
    }
    let first = block_on(open_decode_device_at(VulkanVideoCodec::H264, Some(0)))
        .expect("index 0 is a decode-capable adapter");
    assert!(
        !first.device_name().is_empty(),
        "the opened device names itself"
    );
    eprintln!("m821 device-index=0: {}", first.device_name());
    drop(first);

    // No host has 64 decode-capable GPUs; the request must be refused.
    assert_eq!(
        block_on(open_decode_device_at(VulkanVideoCodec::H264, Some(63))).err(),
        Some(VulkanVideoError::NoSuchDevice),
        "an out-of-range device index is an error"
    );
}

/// `num-dpb-slots` grows the decoder's DPB image pool, and the enlarged pool
/// still decodes the clip bit-identically.
#[test]
fn num_dpb_slots_grows_the_dpb() {
    let Some(mut device) = h264_device_or_skip("num-dpb-slots") else {
        return;
    };
    let ps = extract_h264_parameter_sets(BFRAMES).expect("sps/pps");

    let session = device.create_h264_session(&ps, 640, 480).expect("session");
    let mut default_dec = device
        .create_h264_dpb_decoder(&session, &ps)
        .expect("decoder");
    let default_slots = default_dec.dpb_slots();
    let reference: Vec<Vec<u8>> = default_dec
        .decode_all(BFRAMES)
        .expect("decode_all")
        .iter()
        .map(planes)
        .collect();
    assert!(!reference.is_empty(), "clip decodes");

    let want = default_slots + 4;
    device.set_dpb_slots(Some(want as u32));
    let session2 = device.create_h264_session(&ps, 640, 480).expect("session");
    let mut big_dec = device
        .create_h264_dpb_decoder(&session2, &ps)
        .expect("decoder with the requested DPB");
    assert_eq!(
        big_dec.dpb_slots(),
        want,
        "the request sized the DPB (default was {default_slots})"
    );
    let with_request: Vec<Vec<u8>> = big_dec
        .decode_all(BFRAMES)
        .expect("decode_all")
        .iter()
        .map(planes)
        .collect();
    assert_eq!(
        with_request, reference,
        "a larger DPB decodes the clip identically"
    );

    // A request below what the stream needs cannot shrink the pool: fewer slots
    // than its references would evict a live one.
    device.set_dpb_slots(Some(1));
    let session3 = device.create_h264_session(&ps, 640, 480).expect("session");
    let small_dec = device
        .create_h264_dpb_decoder(&session3, &ps)
        .expect("decoder");
    assert_eq!(small_dec.dpb_slots(), default_slots);
    eprintln!("m821 num-dpb-slots: default {default_slots}, requested {want}");
}

/// `low-latency` turns off display-order reordering: a B-frame clip comes out in
/// coding order, one frame per access unit, instead of being held until its
/// display position is settled.
#[test]
fn low_latency_emits_in_coding_order_one_frame_per_au() {
    let Some(device) = h264_device_or_skip("low-latency") else {
        return;
    };
    let ps = extract_h264_parameter_sets(BFRAMES).expect("sps/pps");
    let session = device.create_h264_session(&ps, 640, 480).expect("session");
    let mut oracle_dec = device
        .create_h264_dpb_decoder(&session, &ps)
        .expect("decoder");

    // The fixture must genuinely reorder, else coding order == display order and
    // the property would be untestable here.
    let pocs: Vec<i32> = oracle_dec
        .index_pictures(BFRAMES)
        .expect("index")
        .iter()
        .map(|m| m.poc)
        .collect();
    assert!(
        pocs.windows(2).any(|w| w[1] < w[0]),
        "fixture has no B-frame reorder (POC monotonic)"
    );

    // Coding-order oracle: the pipelined decode as it retires, no reordering.
    let mut coding_order: Vec<Vec<u8>> = oracle_dec
        .decode_push(BFRAMES)
        .expect("decode_push")
        .1
        .iter()
        .map(planes)
        .collect();
    coding_order.extend(oracle_dec.decode_flush().expect("flush").iter().map(planes));

    let aus = split_access_units(BFRAMES);
    assert_eq!(aus.len(), coding_order.len(), "one AU per coded picture");

    let mut dec = VulkanVideoDec::new();
    dec.set_property("low-latency", PropValue::Bool(true))
        .unwrap();
    dec.configure_pipeline(&h264_caps())
        .expect("configure opens the decode device");
    let per_au = drive_per_au(&mut dec, aus.clone());

    for (i, frames) in per_au.iter().take(aus.len()).enumerate() {
        assert_eq!(
            frames.len(),
            1,
            "low-latency emits access unit {i}'s own frame from that process call"
        );
    }
    assert!(
        per_au[aus.len()].is_empty(),
        "nothing is left buffered at eos"
    );
    let got: Vec<Vec<u8>> = per_au.into_iter().flatten().collect();
    assert_eq!(got, coding_order, "low-latency output is in coding order");

    // The default (property off) element reorders the same clip, so the two
    // orders really do differ: low-latency is not a no-op.
    let mut ordered = VulkanVideoDec::new();
    ordered
        .configure_pipeline(&h264_caps())
        .expect("configure opens the decode device");
    let display_order: Vec<Vec<u8>> = drive_per_au(&mut ordered, aus)
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(
        display_order.len(),
        got.len(),
        "same frame count either way"
    );
    assert_ne!(display_order, got, "default reorders, low-latency does not");
    eprintln!("m821 low-latency: {} frames in coding order", got.len());
}

/// Both construction-time properties are refused once the state they size is
/// built, instead of being accepted and ignored.
#[test]
fn construction_properties_are_refused_once_built() {
    if h264_device_or_skip("late property set").is_none() {
        return;
    }
    let mut dec = VulkanVideoDec::new();
    dec.configure_pipeline(&h264_caps())
        .expect("configure opens the decode device");
    assert_eq!(
        dec.set_property("device-index", PropValue::Uint(0)),
        Err(PropError::ReadOnly),
        "the device is already open"
    );
    // num-dpb-slots still applies: the DPB is built at the first keyframe.
    dec.set_property("num-dpb-slots", PropValue::Uint(9))
        .expect("before the first access unit");

    let aus = split_access_units(BFRAMES);
    let mut sink = RecordingSink::default();
    block_on(dec.process(
        PipelinePacket::DataFrame(au_frame(aus[0].clone(), 0)),
        &mut sink,
    ))
    .expect("decode the keyframe access unit");
    assert_eq!(
        dec.set_property("num-dpb-slots", PropValue::Uint(10)),
        Err(PropError::ReadOnly),
        "the DPB is already allocated"
    );
    // low-latency stays settable mid-stream: it only changes how much is held.
    dec.set_property("low-latency", PropValue::Bool(true))
        .expect("low-latency applies at any time");
}
