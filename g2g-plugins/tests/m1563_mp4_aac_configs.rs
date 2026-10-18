//! M1563: `qtdemux stream=aac` reads the audio of a fragmented MP4 from its
//! fragments, and sends an AAC track with 960-sample frames, which ADTS cannot
//! signal, as raw access units behind its AudioSpecificConfig.
#![cfg(feature = "std")]

#[path = "common/audio_mp4.rs"]
mod audio_mp4;
#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/mp4_demux.rs"]
mod mp4_demux;

use g2g_core::{AudioFormat, Caps};

use audio_mp4::{
    access_units, fragmented_audio_mp4, mp4a_entry, progressive_audio_mp4, MPEG4_AUDIO,
};
use mp4_demux::demux_audio;

// AAC LC, 48 kHz, 960-sample frames, mono in a program config element, as FATE's aac/al04sf_48.mp4 carries it
const SHORT_FRAME_CONFIG: [u8; 9] = [0x11, 0x84, 0x04, 0xC4, 0x00, 0x20, 0x01, 0xE0, 0x00];
// AAC LC, 48 kHz, mono, then a sync extension declaring no SBR, as FATE's mpeg4/fragmented.mp4 carries it
const SYNC_EXTENSION_CONFIG: [u8; 5] = [0x11, 0x88, 0x56, 0xE5, 0x00];
// syncword and MPEG-4 / no-CRC bits of an ADTS header
const ADTS_SYNC: [u8; 2] = [0xFF, 0xF1];
const ADTS_HEADER_LEN: usize = 7;

#[tokio::test]
async fn short_frame_aac_leaves_raw_behind_its_config() {
    let units = access_units();
    let file = progressive_audio_mp4(&mp4a_entry(MPEG4_AUDIO, &SHORT_FRAME_CONFIG), &units);
    let demuxed = demux_audio(&file).await;
    let expected: Vec<Vec<u8>> = std::iter::once(SHORT_FRAME_CONFIG.to_vec())
        .chain(units)
        .collect();
    assert_eq!(
        demuxed.frames, expected,
        "the AudioSpecificConfig leads, then each access unit unframed"
    );
}

#[tokio::test]
async fn fragmented_aac_track_demuxes_from_its_fragments() {
    let units = access_units();
    let file = fragmented_audio_mp4(&mp4a_entry(MPEG4_AUDIO, &SYNC_EXTENSION_CONFIG), &units);
    let demuxed = demux_audio(&file).await;
    assert!(
        demuxed.caps.iter().any(|caps| matches!(
            caps,
            Caps::Audio {
                format: AudioFormat::Aac,
                ..
            }
        )),
        "AAC caps announced"
    );
    assert_eq!(
        demuxed.frames.len(),
        units.len(),
        "one frame per access unit"
    );
    for (frame, unit) in demuxed.frames.iter().zip(&units) {
        assert!(frame.starts_with(&ADTS_SYNC), "ADTS framed: {frame:02x?}");
        assert_eq!(
            &frame[ADTS_HEADER_LEN..],
            &unit[..],
            "the unit follows the header"
        );
    }
}
