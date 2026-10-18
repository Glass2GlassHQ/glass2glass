#![cfg(feature = "std")]

#[path = "common/audio_mp4.rs"]
mod audio_mp4;
#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/mp4_demux.rs"]
mod mp4_demux;

use audio_mp4::{moov_first_quicktime_pcm_mov, sound_entry, CHANNELS};
use mp4_demux::{demux_audio, demux_first_port};

const SAMPLE_BITS: u16 = 8;
const FRAME_BYTES: usize = CHANNELS as usize * SAMPLE_BITS as usize / 8;
const FRAMES_PER_CHUNK: usize = 1000;
const CHUNK_COUNT: usize = 4;
const CHUNKS_LEFT: usize = 1;

fn chunks() -> Vec<Vec<u8>> {
    (0..CHUNK_COUNT)
        .map(|chunk| {
            (0..FRAMES_PER_CHUNK * FRAME_BYTES)
                .map(|byte| (chunk * FRAMES_PER_CHUNK + byte) as u8)
                .collect()
        })
        .collect()
}

// cut after the first chunk, so the stsz counts more frames than the file has bytes
fn truncated_mov() -> Vec<u8> {
    let mut file =
        moov_first_quicktime_pcm_mov(&sound_entry(b"raw ", SAMPLE_BITS), &chunks(), FRAME_BYTES);
    let lost_bytes = (CHUNK_COUNT - CHUNKS_LEFT) * FRAMES_PER_CHUNK * FRAME_BYTES;
    file.truncate(file.len() - lost_bytes);
    assert!(file.len() < CHUNK_COUNT * FRAMES_PER_CHUNK);
    file
}

#[tokio::test]
async fn mp4demux_reads_pcm_up_to_the_cut() {
    let demuxed = demux_audio(&truncated_mov(), "pcm-u8").await;
    assert_eq!(demuxed.frames, chunks()[..CHUNKS_LEFT]);
}

#[tokio::test]
async fn multi_track_demux_reads_pcm_up_to_the_cut() {
    let demuxed = demux_first_port(&truncated_mov()).await;
    assert_eq!(demuxed.frames, chunks()[..CHUNKS_LEFT]);
}
