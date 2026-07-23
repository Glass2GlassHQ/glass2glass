//! Edge content preview (dev tooling, `observe` feature): convert a sampled
//! packet into a small JSON preview for the dashboard. Raw video (packed
//! RGBA/BGRA and planar NV12/I420 in system memory) becomes a downscaled
//! thumbnail; PCM audio becomes min/max waveform buckets; a compressed edge
//! becomes a keyframe thumbnail where cheap (MJPEG, with the `mjpeg` feature) or
//! a labelled codec card otherwise; anything else (GPU, unhandled raw formats)
//! becomes a bounded hexdump. Pure and unit-testable; the rate-limited tap that
//! calls it lives in `dashboard`.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use serde_json::{json, Value};

use g2g_core::{AudioFormat, Caps, Dim, PipelinePacket, RawVideoFormat, VideoCodec};

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
    let Some(slice) = frame.domain.as_system_slice() else {
        // A GPU / foreign frame can't be sampled without a download.
        return Some(json!({ "kind": "opaque", "note": "non-system memory" }));
    };
    let bytes = slice;
    Some(match caps {
        Caps::RawVideo {
            format,
            width,
            height,
            ..
        } => match (dim(width), dim(height)) {
            (Some(w), Some(h)) => raw_video_thumb(bytes, *format, w as usize, h as usize),
            _ => hexdump(bytes),
        },
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            ..
        } => audio_peaks(bytes),
        Caps::CompressedVideo {
            codec,
            width,
            height,
            ..
        } => compressed_preview(bytes, *codec, dim(width), dim(height)),
        _ => hexdump(bytes),
    })
}

/// Thumbnail one raw-video frame. Packed RGBA/BGRA go straight to the downscaler;
/// planar NV12/I420 are converted to RGBA first via the shared `VideoConvert`
/// math. Unhandled formats or a short buffer fall back to a hexdump.
fn raw_video_thumb(bytes: &[u8], format: RawVideoFormat, w: usize, h: usize) -> Value {
    match format {
        RawVideoFormat::Rgba8 => video_thumb(bytes, w, h, false),
        RawVideoFormat::Bgra8 => video_thumb(bytes, w, h, true),
        RawVideoFormat::Nv12 | RawVideoFormat::I420 => {
            // 4:2:0 needs even dims and a luma + half-size chroma plane.
            let planar_len = w.saturating_mul(h).saturating_mul(3) / 2;
            if w == 0 || h == 0 || w % 2 != 0 || h % 2 != 0 || bytes.len() < planar_len {
                return hexdump(bytes);
            }
            let rgba = crate::videoconvert::convert(bytes, format, RawVideoFormat::Rgba8, w, h);
            video_thumb(&rgba, w, h, false)
        }
        _ => hexdump(bytes),
    }
}

/// Preview a compressed-video edge. With the `mjpeg` feature an MJPEG packet is a
/// self-contained baseline JPEG, so it decodes to a keyframe thumbnail; every
/// other codec (and a failed MJPEG decode) yields a labelled card carrying the
/// codec name and resolution.
#[cfg_attr(not(feature = "mjpeg"), allow(unused_variables))]
fn compressed_preview(bytes: &[u8], codec: VideoCodec, w: Option<u32>, h: Option<u32>) -> Value {
    #[cfg(feature = "mjpeg")]
    if codec == VideoCodec::Mjpeg {
        if let Ok((rgba, dw, dh)) = crate::mjpegdec::MjpegDec::new().decode(bytes) {
            return video_thumb(&rgba, dw as usize, dh as usize, false);
        }
    }
    // No decode (would need a heavy decoder, and a sampled packet is rarely a
    // self-contained keyframe): enrich the card from the packet header instead:
    // the codec, resolution, this packet's frame type, and its size.
    json!({
        "kind": "compressed",
        "codec": codec_name(codec),
        "w": w,
        "h": h,
        "frame": frame_kind(codec, bytes),
        "bytes": bytes.len(),
    })
}

/// Best-effort frame type from the packet header ("key" if it carries a keyframe
/// / IRAP unit, else "delta"), or `None` when the codec is not cheaply classified
/// here. Header inspection only, never a decode.
fn frame_kind(codec: VideoCodec, bytes: &[u8]) -> Option<&'static str> {
    match codec {
        // H.264 NAL type = header & 0x1F: 5 = IDR (key), 1 = non-IDR slice (delta).
        VideoCodec::H264 => annexb_kind(bytes, |h| match h & 0x1F {
            5 => Some(true),
            1 => Some(false),
            _ => None,
        }),
        // H.265 NAL type = (header >> 1) & 0x3F: 16..=23 are IRAP (key), 0..=9 the
        // trailing / leading VCL slices (delta).
        VideoCodec::H265 => annexb_kind(bytes, |h| {
            let t = (h >> 1) & 0x3F;
            if (16..=23).contains(&t) {
                Some(true)
            } else if t <= 9 {
                Some(false)
            } else {
                None
            }
        }),
        // VP8 frame tag: bit 0 of the first byte is the frame type (0 = key).
        VideoCodec::Vp8 => bytes
            .first()
            .map(|b| if b & 1 == 0 { "key" } else { "delta" }),
        _ => None,
    }
}

/// Scan Annex-B NAL units, classifying by the first unit `classify` recognizes
/// (`Some(true)` = keyframe, `Some(false)` = delta, `None` = skip, e.g. SPS / PPS
/// / SEI). A keyframe VCL unit anywhere wins; otherwise a delta unit does.
fn annexb_kind(bytes: &[u8], classify: impl Fn(u8) -> Option<bool>) -> Option<&'static str> {
    let mut i = 0;
    let mut saw_delta = false;
    while i + 3 < bytes.len() {
        if bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 1 {
            match classify(bytes[i + 3]) {
                Some(true) => return Some("key"),
                Some(false) => saw_delta = true,
                None => {}
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    if saw_delta {
        Some("delta")
    } else {
        None
    }
}

/// Stable lower-case name for a video codec, shown on the compressed preview card.
fn codec_name(codec: VideoCodec) -> &'static str {
    match codec {
        VideoCodec::H264 => "h264",
        VideoCodec::H265 => "h265",
        VideoCodec::Av1 => "av1",
        VideoCodec::Vp8 => "vp8",
        VideoCodec::Vp9 => "vp9",
        VideoCodec::Mjpeg => "mjpeg",
        VideoCodec::Mpeg4Part2 => "mpeg4",
        VideoCodec::JpegXs => "jpegxs",
        _ => "compressed",
    }
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
    use g2g_core::{Dim, Frame, FrameTiming, MemoryDomain, Rate, SystemSlice};

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
        let buf = [255u8, 0, 0, 255].repeat(4 * 2);
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
        let caps = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 1,
            sample_rate: 48_000,
        };
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
    fn i420_becomes_thumbnail() {
        let (w, h) = (4u32, 2u32);
        let caps = Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        };
        // luma + U + V planes = w*h*3/2 bytes, mid-gray chroma.
        let mut buf = vec![128u8; (w * h) as usize];
        buf.extend(vec![128u8; (w * h) as usize / 2]);
        let v = packet_preview(&frame(buf), &caps).unwrap();
        assert_eq!(v["kind"], "video");
        assert_eq!(v["w"], w);
        assert_eq!(v["h"], h);
        assert_eq!(v["rgba"].as_array().unwrap().len(), (w * h * 4) as usize);
    }

    #[test]
    fn nv12_becomes_thumbnail() {
        let (w, h) = (4u32, 2u32);
        let caps = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        };
        let mut buf = vec![128u8; (w * h) as usize];
        buf.extend(vec![128u8; (w * h) as usize / 2]);
        let v = packet_preview(&frame(buf), &caps).unwrap();
        assert_eq!(v["kind"], "video");
        assert_eq!(v["w"], w);
        assert_eq!(v["h"], h);
    }

    #[test]
    fn short_planar_buffer_falls_back_to_hex() {
        let caps = Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(64),
            height: Dim::Fixed(64),
            framerate: Rate::Fixed(30 << 16),
        };
        let v = packet_preview(&frame(vec![1, 2, 3, 4]), &caps).unwrap();
        assert_eq!(v["kind"], "hex");
    }

    fn h264_caps() -> Caps {
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    #[test]
    fn compressed_video_becomes_card() {
        // No recognizable NAL: card with codec + resolution + byte size, no frame.
        let v = packet_preview(&frame(vec![0xde, 0xad, 0xbe, 0xef]), &h264_caps()).unwrap();
        assert_eq!(v["kind"], "compressed");
        assert_eq!(v["codec"], "h264");
        assert_eq!(v["w"], 1920);
        assert_eq!(v["h"], 1080);
        assert_eq!(v["bytes"], 4);
        assert!(v["frame"].is_null());
    }

    #[test]
    fn h264_idr_reads_as_a_keyframe() {
        // Annex-B start code + NAL header 0x65 (nal_ref_idc=3, type=5 = IDR).
        let v = packet_preview(&frame(vec![0, 0, 1, 0x65, 0x88, 0x84]), &h264_caps()).unwrap();
        assert_eq!(v["frame"], "key");
    }

    #[test]
    fn h264_non_idr_reads_as_delta() {
        // NAL header 0x41 (type=1 = non-IDR slice), no IDR present.
        let v = packet_preview(&frame(vec![0, 0, 1, 0x41, 0x9a]), &h264_caps()).unwrap();
        assert_eq!(v["frame"], "delta");
    }

    #[test]
    fn bytestream_still_falls_back_to_hex() {
        let caps = Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::MpegTs,
        };
        let v = packet_preview(&frame(vec![0xde, 0xad, 0xbe, 0xef]), &caps).unwrap();
        assert_eq!(v["kind"], "hex");
        assert_eq!(v["len"], 4);
        assert_eq!(v["bytes"], "deadbeef");
    }

    #[cfg(feature = "mjpeg")]
    #[test]
    fn mjpeg_keyframe_becomes_thumbnail() {
        // A self-contained baseline JPEG decodes to a thumbnail, not a card.
        const RED16: &[u8] = include_bytes!("../tests/data/red16.jpg");
        let caps = Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::Mjpeg,
            width: Dim::Fixed(16),
            height: Dim::Fixed(16),
            framerate: Rate::Fixed(30 << 16),
        };
        let v = packet_preview(&frame(RED16.to_vec()), &caps).unwrap();
        assert_eq!(v["kind"], "video");
        assert_eq!(v["w"], 16);
        assert_eq!(v["h"], 16);
    }

    #[cfg(feature = "mjpeg")]
    #[test]
    fn undecodable_mjpeg_becomes_card() {
        let caps = Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::Mjpeg,
            width: Dim::Fixed(16),
            height: Dim::Fixed(16),
            framerate: Rate::Fixed(30 << 16),
        };
        let v = packet_preview(&frame(vec![0, 1, 2, 3]), &caps).unwrap();
        assert_eq!(v["kind"], "compressed");
        assert_eq!(v["codec"], "mjpeg");
    }

    #[test]
    fn control_packet_has_no_preview() {
        let caps = Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::MpegTs,
        };
        assert!(packet_preview(&PipelinePacket::Eos, &caps).is_none());
    }
}
