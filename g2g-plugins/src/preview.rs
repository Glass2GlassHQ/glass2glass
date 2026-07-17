//! Edge content preview (dev tooling, `observe` feature): convert a sampled
//! packet into a small JSON preview for the dashboard. Video (packed RGBA/BGRA in
//! system memory) becomes a downscaled thumbnail; PCM audio becomes min/max
//! waveform buckets; anything else (compressed, GPU, unhandled raw formats)
//! becomes a bounded hexdump. Pure and unit-testable; the rate-limited tap that
//! calls it lives in `dashboard`.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use serde_json::{json, Value};

use g2g_core::{AudioFormat, Caps, Dim, MemoryDomain, PipelinePacket, RawVideoFormat};

/// Longest thumbnail edge in pixels.
const MAX_THUMB: usize = 48;
/// Waveform buckets per audio preview.
const AUDIO_BUCKETS: usize = 64;
/// Bytes shown in a hexdump fallback.
const HEX_BYTES: usize = 128;

fn dim(d: &Dim) -> Option<u32> {
    match d {
        Dim::Fixed(v) => Some(*v),
        _ => None,
    }
}

/// A preview of one packet under `caps`, or `None` if the packet is not a
/// `DataFrame` (control packets have no content to show).
pub fn packet_preview(packet: &PipelinePacket, caps: &Caps) -> Option<Value> {
    let PipelinePacket::DataFrame(frame) = packet else {
        return None;
    };
    let MemoryDomain::System(slice) = &frame.domain else {
        // A GPU / foreign frame can't be sampled without a download.
        return Some(json!({ "kind": "opaque", "note": "non-system memory" }));
    };
    let bytes = slice.as_slice();
    Some(match caps {
        Caps::RawVideo { format, width, height, .. }
            if matches!(format, RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8) =>
        {
            match (dim(width), dim(height)) {
                (Some(w), Some(h)) => {
                    video_thumb(bytes, w as usize, h as usize, matches!(format, RawVideoFormat::Bgra8))
                }
                _ => hexdump(bytes),
            }
        }
        Caps::Audio { format: AudioFormat::PcmS16Le, .. } => audio_peaks(bytes),
        _ => hexdump(bytes),
    })
}

/// Nearest-neighbor downscale of a packed 8-bit RGBA/BGRA buffer to a thumbnail,
/// emitting RGBA bytes. Falls back to a hexdump if the buffer is too short.
fn video_thumb(bytes: &[u8], w: usize, h: usize, bgra: bool) -> Value {
    if w == 0 || h == 0 || bytes.len() < w.saturating_mul(h).saturating_mul(4) {
        return hexdump(bytes);
    }
    let tw = w.clamp(1, MAX_THUMB);
    let th = h.clamp(1, MAX_THUMB);
    let mut out = Vec::with_capacity(tw * th * 4);
    for ty in 0..th {
        let sy = ty * h / th;
        for tx in 0..tw {
            let sx = tx * w / tw;
            let o = (sy * w + sx) * 4;
            let (r, g, b, a) = if bgra {
                (bytes[o + 2], bytes[o + 1], bytes[o], bytes[o + 3])
            } else {
                (bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3])
            };
            out.extend_from_slice(&[r, g, b, a]);
        }
    }
    json!({ "kind": "video", "w": tw, "h": th, "rgba": out })
}

/// Min/max waveform buckets over interleaved PCM S16LE samples, normalized to
/// [-1, 1].
fn audio_peaks(bytes: &[u8]) -> Value {
    let n = bytes.len() / 2;
    if n == 0 {
        return json!({ "kind": "audio", "peaks": Vec::<Value>::new() });
    }
    let sample = |i: usize| i16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]) as f32 / 32768.0;
    let buckets = AUDIO_BUCKETS.min(n);
    let mut peaks: Vec<Value> = Vec::with_capacity(buckets);
    for b in 0..buckets {
        let start = b * n / buckets;
        let end = ((b + 1) * n / buckets).max(start + 1).min(n);
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for i in start..end {
            let s = sample(i);
            lo = lo.min(s);
            hi = hi.max(s);
        }
        peaks.push(json!([lo, hi]));
    }
    json!({ "kind": "audio", "peaks": peaks })
}

fn hexdump(bytes: &[u8]) -> Value {
    let shown = &bytes[..bytes.len().min(HEX_BYTES)];
    let mut hex = String::with_capacity(shown.len() * 2);
    for b in shown {
        hex.push_str(&format!("{b:02x}"));
    }
    json!({ "kind": "hex", "len": bytes.len(), "bytes": hex })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use g2g_core::{Dim, Frame, FrameTiming, Rate, SystemSlice};

    fn frame(bytes: Vec<u8>) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming::default(),
            0,
        ))
    }

    #[test]
    fn video_rgba_becomes_thumbnail() {
        // 4x2 RGBA, all red.
        let caps = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(4),
            height: Dim::Fixed(2),
            framerate: Rate::Fixed(30 << 16),
        };
        let buf = vec![255u8, 0, 0, 255].repeat(4 * 2);
        let v = packet_preview(&frame(buf), &caps).unwrap();
        assert_eq!(v["kind"], "video");
        assert_eq!(v["w"], 4);
        assert_eq!(v["h"], 2);
        let rgba = v["rgba"].as_array().unwrap();
        assert_eq!(rgba.len(), 4 * 2 * 4);
        assert_eq!(rgba[0], 255); // R
        assert_eq!(rgba[1], 0); // G
    }

    #[test]
    fn bgra_channels_are_swapped_to_rgba() {
        let caps = Caps::RawVideo {
            format: RawVideoFormat::Bgra8,
            width: Dim::Fixed(1),
            height: Dim::Fixed(1),
            framerate: Rate::Fixed(30 << 16),
        };
        // One BGRA pixel: B=10 G=20 R=30 A=40 -> preview RGBA 30,20,10,40.
        let v = packet_preview(&frame(vec![10, 20, 30, 40]), &caps).unwrap();
        let rgba = v["rgba"].as_array().unwrap();
        assert_eq!(rgba[0], 30);
        assert_eq!(rgba[1], 20);
        assert_eq!(rgba[2], 10);
        assert_eq!(rgba[3], 40);
    }

    #[test]
    fn short_video_buffer_falls_back_to_hex() {
        let caps = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(64),
            height: Dim::Fixed(64),
            framerate: Rate::Fixed(30 << 16),
        };
        let v = packet_preview(&frame(vec![1, 2, 3, 4]), &caps).unwrap();
        assert_eq!(v["kind"], "hex");
    }

    #[test]
    fn audio_pcm_becomes_peaks() {
        let caps = Caps::Audio { format: AudioFormat::PcmS16Le, channels: 1, sample_rate: 48_000 };
        // Full-scale +/- samples.
        let mut buf = Vec::new();
        for _ in 0..128 {
            buf.extend_from_slice(&i16::MAX.to_le_bytes());
            buf.extend_from_slice(&i16::MIN.to_le_bytes());
        }
        let v = packet_preview(&frame(buf), &caps).unwrap();
        assert_eq!(v["kind"], "audio");
        let peaks = v["peaks"].as_array().unwrap();
        assert!(!peaks.is_empty());
        // A bucket spanning +max and -max reads ~[-1, ~1].
        let hi = peaks[0][1].as_f64().unwrap();
        assert!(hi > 0.9, "peak hi {hi}");
    }

    #[test]
    fn compressed_falls_back_to_hex() {
        let caps = Caps::ByteStream { encoding: g2g_core::ByteStreamEncoding::MpegTs };
        let v = packet_preview(&frame(vec![0xde, 0xad, 0xbe, 0xef]), &caps).unwrap();
        assert_eq!(v["kind"], "hex");
        assert_eq!(v["len"], 4);
        assert_eq!(v["bytes"], "deadbeef");
    }

    #[test]
    fn control_packet_has_no_preview() {
        let caps = Caps::ByteStream { encoding: g2g_core::ByteStreamEncoding::MpegTs };
        assert!(packet_preview(&PipelinePacket::Eos, &caps).is_none());
    }
}
