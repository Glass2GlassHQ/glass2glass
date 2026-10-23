//! M1717: `vulkanvideodec` refuses an H.264 stream it cannot decode with an
//! error that names the coding tool, and emits no picture, instead of ending
//! the stream as an empty success.
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
use g2g_plugins::vulkanvideo::{open_h264_decode_device, VulkanVideoDec, VulkanVideoError};

// tests in this file open the same device, one at a time
static GPU_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

// One access unit, 176x128 Main profile with frame_mbs_only_flag = 0: an SPS, a
// PPS, and an IDR slice whose header sets field_pic_flag = 1 (a top field),
// followed by filler slice data the refusal never reaches.
const FIELD_PICTURE_ACCESS_UNIT: &[u8] = &[
    0x00, 0x00, 0x00, 0x01, 0x67, 0x4d, 0x00, 0x1e, 0xf4, 0x16, 0x42, 0x40, 0x00, 0x00, 0x00, 0x01,
    0x68, 0xce, 0x3c, 0x80, 0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x85, 0x02, 0xa4, 0x92, 0x49, 0x24,
    0x92, 0x49, 0x24, 0x92, 0x60,
];
const FIELD_PICTURE_WIDTH: u32 = 176;
const FIELD_PICTURE_HEIGHT: u32 = 128;
const REFUSAL_REASON: &str = "interlaced field pictures";

fn has_h264_decode() -> bool {
    match block_on(open_h264_decode_device()) {
        Ok(_) => true,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => false,
        Err(e) => panic!("open the H.264 decode device: {e:?}"),
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
fn field_coded_h264_fails_with_a_named_error_and_no_pictures() {
    let _gpu = GPU_TEST_LOCK.blocking_lock();
    if !has_h264_decode() {
        eprintln!("skipping: no Vulkan H.264 decode adapter");
        return;
    }
    let messages = Arc::new(Mutex::new(Vec::new()));
    let sink_id = g2g_core::log::add_sink(Box::new(ErrorMessages(messages.clone())));

    let mut decoder = VulkanVideoDec::new();
    decoder
        .configure_pipeline(&Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(FIELD_PICTURE_WIDTH),
            height: Dim::Fixed(FIELD_PICTURE_HEIGHT),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        })
        .expect("configure opens the decode device");
    let mut out = CollectSink::default();
    let access_unit = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(FIELD_PICTURE_ACCESS_UNIT.into())),
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
