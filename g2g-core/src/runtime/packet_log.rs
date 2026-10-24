use core::fmt;

use crate::frame::{FrameTiming, PipelinePacket};
use crate::g2g_log_at;
use crate::log::{ClockTime, ElementLogName, LogLevel};
use crate::runtime::channel::frame_bytes;

// Each packet an arm hands to its element's `process`, under that element's name.
pub(crate) fn log_received(element: &ElementLogName, packet: &PipelinePacket) {
    g2g_log_at!(
        packet_level(packet),
        element,
        "have {}",
        PacketFields(packet)
    );
}

// Each packet a source pushes onto its output link.
pub(crate) fn log_pushed(source: &ElementLogName, packet: &PipelinePacket) {
    g2g_log_at!(
        packet_level(packet),
        source,
        "pushing {}",
        PacketFields(packet)
    );
}

// Buffers at LOG and everything else at DEBUG, the split gst's GST_SCHEDULING and GST_EVENT make.
fn packet_level(packet: &PipelinePacket) -> LogLevel {
    match packet {
        PipelinePacket::DataFrame(_) => LogLevel::Log,
        _ => LogLevel::Debug,
    }
}

struct PacketFields<'a>(&'a PipelinePacket);

impl fmt::Display for PacketFields<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            PipelinePacket::DataFrame(frame) => write!(
                f,
                "buffer pts {}, dur {}, size {}, domain {:?}",
                ClockTime(frame.timing.pts_ns),
                ClockTime(frame.timing.duration_ns),
                frame_bytes(frame),
                frame.domain.kind(),
            ),
            PipelinePacket::CapsChanged(caps) => write!(f, "event caps: {}", caps.to_gst_string()),
            PipelinePacket::Segment(segment) => write!(
                f,
                "event segment: rate {}, start {}, stop {}, base {}, time {}",
                segment.rate,
                ClockTime(segment.start),
                ClockTime(segment.stop.unwrap_or(FrameTiming::PTS_NONE)),
                ClockTime(segment.base),
                ClockTime(segment.time),
            ),
            PipelinePacket::Eos => f.write_str("event eos"),
            PipelinePacket::Flush => f.write_str("event flush"),
            PipelinePacket::Tick => f.write_str("event tick"),
        }
    }
}
