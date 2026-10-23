#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use std::sync::{Arc, Mutex};

use collect_sink::CollectSink;
use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::log::{LogLevel, LogRecord, LogSink};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::block_on;
use g2g_core::{Caps, Dim, G2gError, Rate, VideoCodec};
use g2g_plugins::vulkanvideo::{
    open_h265_decode_device, parse_h265_sps, VulkanVideoDec, VulkanVideoError,
};

// tests in this file open the same device, one at a time
static GPU_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

// One access unit, 64x64 Format Range Extensions profile: a VPS, an SPS coding
// 12-bit 4:2:0, a PPS, and an IDR slice header followed by filler slice data
// the refusal never reaches.
const TWELVE_BIT_ACCESS_UNIT: &[u8] = &[
    0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0c, 0x01, 0xff, 0xff, 0x04, 0x08, 0x00, 0x00, 0x03, 0x00,
    0x90, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x5d, 0xac, 0x09, 0x00, 0x00, 0x00, 0x01, 0x42,
    0x01, 0x01, 0x04, 0x08, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00,
    0x5d, 0xa0, 0x20, 0x81, 0x04, 0x52, 0x96, 0xb9, 0x24, 0xc2, 0x08, 0x00, 0x00, 0x00, 0x01, 0x44,
    0x01, 0xc0, 0x71, 0x81, 0x12, 0x00, 0x00, 0x00, 0x01, 0x26, 0x01, 0xac, 0x02, 0x49, 0x24, 0x92,
];
const TWELVE_BIT_SIZE: u32 = 64;
const REFUSAL_REASON: &str = "bit depths above 10";
const H265_NAL_SPS: u8 = 33;

fn has_h265_decode() -> bool {
    match block_on(open_h265_decode_device()) {
        Ok(_) => true,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => false,
        Err(e) => panic!("open the H.265 decode device: {e:?}"),
    }
}

// keeps the error-level messages logged under the decoder's category
#[derive(Debug, Default)]
struct ErrorMessages(Arc<Mutex<Vec<String>>>);

impl LogSink for ErrorMessages {
    fn emit(&self, record: &LogRecord<'_>) {
        let decoder_category = g2g_core::log::short_type_name::<VulkanVideoDec>();
        if record.level == LogLevel::Error && record.category == decoder_category {
            self.0.lock().unwrap().push(record.message.to_string());
        }
    }
}

#[test]
fn twelve_bit_sps_is_refused_with_the_bit_depth_named() {
    let sps_start = TWELVE_BIT_ACCESS_UNIT
        .windows(5)
        .position(|w| w[..4] == [0, 0, 0, 1] && (w[4] >> 1) & 0x3F == H265_NAL_SPS)
        .expect("the access unit carries an SPS");
    let sps_end = TWELVE_BIT_ACCESS_UNIT[sps_start + 4..]
        .windows(4)
        .position(|w| w == [0, 0, 0, 1])
        .map(|offset| sps_start + 4 + offset)
        .expect("a PPS follows the SPS");
    let mut rbsp = Vec::new();
    let mut zeros = 0;
    // the RBSP starts after the two-byte NAL header, emulation prevention removed
    for &byte in &TWELVE_BIT_ACCESS_UNIT[sps_start + 6..sps_end] {
        if zeros >= 2 && byte == 3 {
            zeros = 0;
            continue;
        }
        zeros = if byte == 0 { zeros + 1 } else { 0 };
        rbsp.push(byte);
    }
    match parse_h265_sps(&rbsp) {
        Err(VulkanVideoError::StreamRefused(reason)) => assert!(
            reason.contains(REFUSAL_REASON),
            "the refusal names the bit depth: {reason}"
        ),
        other => panic!("a 12-bit SPS must be refused, got {other:?}"),
    }
}

#[test]
fn twelve_bit_h265_fails_with_a_named_error_and_no_pictures() {
    let _gpu = GPU_TEST_LOCK.blocking_lock();
    if !has_h265_decode() {
        eprintln!("skipping: no Vulkan H.265 decode adapter");
        return;
    }
    let messages = Arc::new(Mutex::new(Vec::new()));
    let sink_id = g2g_core::log::add_sink(Box::new(ErrorMessages(messages.clone())));

    let mut decoder = VulkanVideoDec::new();
    decoder
        .configure_pipeline(&Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Fixed(TWELVE_BIT_SIZE),
            height: Dim::Fixed(TWELVE_BIT_SIZE),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        })
        .expect("configure opens the decode device");
    let mut out = CollectSink::default();
    let access_unit = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(TWELVE_BIT_ACCESS_UNIT.into())),
        FrameTiming::default(),
        0,
    );
    let result = block_on(decoder.process(PipelinePacket::DataFrame(access_unit), &mut out));
    g2g_core::log::remove_sink(sink_id);

    assert_eq!(result.unwrap_err(), G2gError::CapsMismatch);
    let pictures = out
        .packets
        .iter()
        .filter(|packet| matches!(packet, PipelinePacket::DataFrame(_)))
        .count();
    assert_eq!(pictures, 0, "a refused stream emits no picture");
    let messages = messages.lock().unwrap();
    assert!(
        messages.iter().any(|m| m.contains(REFUSAL_REASON)),
        "the error names the refused coding tool: {messages:?}"
    );
}
