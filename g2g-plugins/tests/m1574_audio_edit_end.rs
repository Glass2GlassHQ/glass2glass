#![cfg(feature = "std")]

#[path = "common/audio_mp4.rs"]
mod audio_mp4;
#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/mp4_demux.rs"]
mod mp4_demux;

use audio_mp4::{
    edited_quicktime_pcm_mov, sound_entry, AudioEdit, CHANNELS, MOVIE_TIMESCALE, SAMPLE_RATE,
};
use mp4_demux::{demux_audio, demux_first_port};

const SAMPLE_BITS: u16 = 16;
const FRAME_BYTES: usize = CHANNELS as usize * SAMPLE_BITS as usize / 8;
const FRAMES_PER_CHUNK: u16 = 400;
const CHUNK_COUNT: u16 = 3;
// starts inside the first chunk and ends inside the last
const EDIT: AudioEdit = AudioEdit {
    media_time: 100,
    duration: 10,
};

// each frame's samples hold its index, so a frame out of place shows
fn frame(index: u16) -> Vec<u8> {
    index.to_le_bytes().repeat(CHANNELS as usize)
}

fn edited_frame_count() -> u32 {
    EDIT.duration * SAMPLE_RATE / MOVIE_TIMESCALE
}

fn edited_mov() -> Vec<u8> {
    let chunks: Vec<Vec<u8>> = (0..CHUNK_COUNT)
        .map(|chunk| {
            (0..FRAMES_PER_CHUNK)
                .flat_map(|offset| frame(chunk * FRAMES_PER_CHUNK + offset))
                .collect()
        })
        .collect();
    edited_quicktime_pcm_mov(
        &sound_entry(b"sowt", SAMPLE_BITS),
        &chunks,
        FRAME_BYTES,
        Some(EDIT),
    )
}

fn presented_pcm() -> Vec<u8> {
    let first = EDIT.media_time as u16;
    let end = first + edited_frame_count() as u16;
    (first..end).flat_map(frame).collect()
}

#[tokio::test]
async fn mp4demux_emits_only_the_edit_of_a_pcm_track() {
    let demuxed = demux_audio(&edited_mov(), "pcm-s16le").await;
    assert_eq!(demuxed.frames.concat(), presented_pcm());
}

#[tokio::test]
async fn multi_track_demux_emits_only_the_edit_of_a_pcm_track() {
    let demuxed = demux_first_port(&edited_mov()).await;
    assert_eq!(demuxed.frames.concat(), presented_pcm());
}
