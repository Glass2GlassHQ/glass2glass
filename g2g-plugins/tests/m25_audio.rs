//! M25: first audio elements. `AudioTestSrc -> WavSink` through the real
//! runner produces a structurally valid, byte-exact WAV file.

use g2g_core::runtime::run_simple_pipeline;
use g2g_core::{AudioFormat, Caps, G2gError, PipelineClock};
use g2g_plugins::audiotestsrc::{AudioTestSrc, Wave};
use g2g_plugins::wavsink::WavSink;

use std::path::PathBuf;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m25_{}_{}.wav", std::process::id(), name))
}

#[tokio::test]
async fn tone_records_to_a_valid_wav_file() {
    let path = temp_path("tone");
    // 100 buffers x 10 ms = 1 s of 1 kHz stereo sine at 48 kHz.
    let mut src = AudioTestSrc::new(48_000, 2, 1_000, 100);
    let mut sink = WavSink::new(&path);

    run_simple_pipeline(&mut src, &mut sink, &NullClock, 4)
        .await
        .expect("audio pipeline negotiates and flows");
    assert!(sink.eos_seen());

    let data = std::fs::read(&path).expect("wav exists");
    // canonical header
    assert_eq!(&data[..4], b"RIFF");
    assert_eq!(&data[8..12], b"WAVE");
    assert_eq!(&data[12..16], b"fmt ");
    assert_eq!(&data[36..40], b"data");
    let riff_size = u32::from_le_bytes(data[4..8].try_into().unwrap());
    assert_eq!(riff_size as usize, data.len() - 8, "riff size patched at Eos");
    let channels = u16::from_le_bytes(data[22..24].try_into().unwrap());
    let rate = u32::from_le_bytes(data[24..28].try_into().unwrap());
    let bits = u16::from_le_bytes(data[34..36].try_into().unwrap());
    assert_eq!((channels, rate, bits), (2, 48_000, 16));
    let data_size = u32::from_le_bytes(data[40..44].try_into().unwrap());
    // 1 s of stereo s16 at 48 kHz
    let expected = 48_000 * 2 * 2;
    assert_eq!(data_size, expected, "data size patched at Eos");
    assert_eq!(data.len(), 44 + expected as usize);
    assert_eq!(sink.bytes_written(), expected as u64);

    // payload sanity: a sine is zero at t=0 and not silent overall.
    let first = i16::from_le_bytes(data[44..46].try_into().unwrap());
    assert_eq!(first, 0, "sine starts at zero");
    let loud = data[44..]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]).unsigned_abs())
        .max()
        .unwrap();
    assert!(loud > 10_000, "tone has real amplitude, peak {loud}");
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn silence_is_all_zero_samples() {
    let path = temp_path("silence");
    let mut src = AudioTestSrc::new(8_000, 1, 440, 3).with_wave(Wave::Silence);
    let mut sink = WavSink::new(&path);
    run_simple_pipeline(&mut src, &mut sink, &NullClock, 4)
        .await
        .expect("pipeline runs");

    let data = std::fs::read(&path).expect("wav exists");
    // 3 buffers x 80 samples x 1 channel x 2 bytes
    assert_eq!(data.len(), 44 + 3 * 80 * 2);
    assert!(data[44..].iter().all(|b| *b == 0), "silence is zeroes");
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn wavsink_rejects_compressed_audio() {
    use g2g_core::element::AsyncElement;
    let mut sink = WavSink::new(temp_path("reject"));
    let aac = Caps::Audio {
        format: AudioFormat::Aac,
        channels: 2,
        sample_rate: 48_000,
    };
    let err = sink.configure_pipeline(&aac).expect_err("aac rejected");
    assert_eq!(err, G2gError::CapsMismatch);
}
