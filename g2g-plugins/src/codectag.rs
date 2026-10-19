use alloc::string::String;

use g2g_core::{AudioFormat, CodecTag, TagNamespace, VideoCodec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TaggedStream {
    Video(CodecTag),
    Audio(CodecTag),
}

const VIDEO_PREFIX: &str = "video-";
const AUDIO_PREFIX: &str = "audio-";
const RIFF_PREFIX: &str = "riff-";
const MOV_PREFIX: &str = "mov-";
const HEX_PREFIX: &str = "0x";

// a demuxer's `stream=` name, `video-riff-cvid`, with the tag in hex when its bytes are not all letters and digits
pub(crate) fn tagged_stream_name(stream: TaggedStream) -> String {
    let (kind, tag) = match stream {
        TaggedStream::Video(tag) => (VIDEO_PREFIX, tag),
        TaggedStream::Audio(tag) => (AUDIO_PREFIX, tag),
    };
    let namespace = match tag.namespace {
        TagNamespace::Riff => RIFF_PREFIX,
        TagNamespace::Mov => MOV_PREFIX,
    };
    let bytes = tag.tag.to_le_bytes();
    if bytes.iter().all(u8::is_ascii_alphanumeric) {
        let fourcc: String = bytes.iter().map(|&byte| char::from(byte)).collect();
        alloc::format!("{kind}{namespace}{fourcc}")
    } else {
        alloc::format!("{kind}{namespace}{HEX_PREFIX}{:08x}", tag.tag)
    }
}

#[cfg(feature = "ffmpeg-long-tail")]
pub(crate) fn tagged_stream_from_name(name: &str) -> Option<TaggedStream> {
    const FOURCC_LEN: usize = 4;
    let (wrap, rest): (fn(CodecTag) -> TaggedStream, &str) =
        if let Some(rest) = name.strip_prefix(VIDEO_PREFIX) {
            (TaggedStream::Video, rest)
        } else {
            (TaggedStream::Audio, name.strip_prefix(AUDIO_PREFIX)?)
        };
    let (namespace, tag) = if let Some(tag) = rest.strip_prefix(RIFF_PREFIX) {
        (TagNamespace::Riff, tag)
    } else {
        (TagNamespace::Mov, rest.strip_prefix(MOV_PREFIX)?)
    };
    let tag = if tag.len() == FOURCC_LEN {
        u32::from_le_bytes(tag.as_bytes().try_into().ok()?)
    } else {
        u32::from_str_radix(tag.strip_prefix(HEX_PREFIX)?, 16).ok()?
    };
    Some(wrap(CodecTag { namespace, tag }))
}

#[cfg(not(feature = "ffmpeg-long-tail"))]
pub(crate) fn tagged_stream_from_name(_name: &str) -> Option<TaggedStream> {
    None
}

// A demuxer's `stream=` name for a codec: its entry in the demuxer's table, else its tag.
pub(crate) fn video_stream_name(table: &[(&str, VideoCodec)], codec: VideoCodec) -> Option<String> {
    if let Some((name, _)) = table.iter().find(|(_, c)| *c == codec) {
        return Some(String::from(*name));
    }
    match codec {
        VideoCodec::Tagged(tag) => Some(tagged_stream_name(TaggedStream::Video(tag))),
        _ => None,
    }
}

pub(crate) fn audio_stream_name(
    table: &[(&str, AudioFormat)],
    format: AudioFormat,
) -> Option<String> {
    if let Some((name, _)) = table.iter().find(|(_, f)| *f == format) {
        return Some(String::from(*name));
    }
    match format {
        AudioFormat::Tagged(tag) => Some(tagged_stream_name(TaggedStream::Audio(tag))),
        _ => None,
    }
}

pub(crate) fn video_from_stream_name(
    table: &[(&str, VideoCodec)],
    name: &str,
) -> Option<VideoCodec> {
    if let Some((_, codec)) = table.iter().find(|(n, _)| *n == name) {
        return Some(*codec);
    }
    match tagged_stream_from_name(name)? {
        TaggedStream::Video(tag) => Some(VideoCodec::Tagged(tag)),
        TaggedStream::Audio(_) => None,
    }
}

pub(crate) fn audio_from_stream_name(
    table: &[(&str, AudioFormat)],
    name: &str,
) -> Option<AudioFormat> {
    if let Some((_, format)) = table.iter().find(|(n, _)| *n == name) {
        return Some(*format);
    }
    match tagged_stream_from_name(name)? {
        TaggedStream::Audio(tag) => Some(AudioFormat::Tagged(tag)),
        TaggedStream::Video(_) => None,
    }
}

// what the demuxer named the stream before it forwarded tagged codecs
#[cfg(all(target_os = "linux", feature = "ffmpeg-long-tail"))]
pub(crate) fn tagged_codec_name(caps: &g2g_core::Caps) -> Option<String> {
    let (tag, video) = match caps {
        g2g_core::Caps::CompressedVideo {
            codec: VideoCodec::Tagged(tag),
            ..
        } => (*tag, true),
        g2g_core::Caps::Audio {
            format: AudioFormat::Tagged(tag),
            ..
        } => (*tag, false),
        _ => return None,
    };
    let bytes = tag.tag.to_le_bytes();
    Some(match (tag.namespace, video) {
        (TagNamespace::Mov, _) => bytes.iter().map(|&byte| char::from(byte)).collect(),
        (TagNamespace::Riff, true) => {
            let printable: String = bytes
                .iter()
                .filter(|byte| byte.is_ascii_graphic())
                .map(|&byte| char::from(byte))
                .collect();
            if printable.is_empty() {
                alloc::format!("video compression {:#x}", tag.tag)
            } else {
                printable
            }
        }
        (TagNamespace::Riff, false) => {
            crate::waveformat::format_tag_name(u16::try_from(tag.tag).ok()?)
        }
    })
}

#[cfg(all(test, feature = "ffmpeg-long-tail"))]
mod tests {
    use super::*;

    #[test]
    fn selection_names_round_trip() {
        for stream in [
            TaggedStream::Video(CodecTag {
                namespace: TagNamespace::Riff,
                tag: u32::from_le_bytes(*b"cvid"),
            }),
            TaggedStream::Video(CodecTag {
                namespace: TagNamespace::Mov,
                tag: u32::from_le_bytes(*b"rle "),
            }),
            TaggedStream::Audio(CodecTag {
                namespace: TagNamespace::Riff,
                tag: 0x0022,
            }),
            TaggedStream::Audio(CodecTag {
                namespace: TagNamespace::Mov,
                tag: u32::from_le_bytes(*b"ms\0\x02"),
            }),
        ] {
            let name = tagged_stream_name(stream);
            assert_eq!(tagged_stream_from_name(&name), Some(stream), "{name}");
        }
        assert_eq!(
            tagged_stream_name(TaggedStream::Video(CodecTag {
                namespace: TagNamespace::Riff,
                tag: u32::from_le_bytes(*b"cvid"),
            })),
            "video-riff-cvid"
        );
    }
}
