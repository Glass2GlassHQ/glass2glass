//! M34: `AudioConvert` in a real negotiated chain. An S16 stereo test tone
//! reaches a WAV sink as float32, and a stereo->mono downmix shrinks the track,
//! both negotiated through the converter (S16-producing source, PCM-accepting
//! sink).

use g2g_core::runtime::run_source_transform_sink;
use g2g_core::{AudioFormat, PipelineClock};
use g2g_plugins::audioconvert::AudioConvert;
use g2g_plugins::audiotestsrc::AudioTestSrc;
use g2g_plugins::wavsink::WavSink;

use std::path::PathBuf;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m34_{}_{}.wav", std::process::id(), name))
}

fn read_u16(data: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(data[at..at + 2].try_into().unwrap())
}

#[tokio::test]
async fn s16_source_records_as_float32_through_converter() {
    let path = temp_path("to_f32");
    // 10 buffers x 10 ms = 100 ms of 1 kHz stereo s16 at 48 kHz.
    let mut src = AudioTestSrc::new(48_000, 2, 1_000, 10);
    let mut conv = AudioConvert::new(AudioFormat::PcmF32Le, 2);
    let mut sink = WavSink::new(&path);

    run_source_transform_sink(&mut src, &mut conv, &mut sink, &NullClock, 4)
        .await
        .expect("s16 -> convert(f32) -> wav negotiates and flows");
    assert!(sink.eos_seen());

    let data = std::fs::read(&path).expect("wav exists");
    // WAVE_FORMAT_IEEE_FLOAT (tag 3), 32-bit, stereo, 48 kHz.
    assert_eq!(read_u16(&data, 20), 3, "format tag is float");
    assert_eq!(read_u16(&data, 22), 2, "stereo");
    assert_eq!(read_u16(&data, 34), 32, "32-bit samples");
    // 100 ms stereo f32 = 48000 * 0.1 * 2 ch * 4 bytes.
    let expected = (48_000f64 * 0.1) as usize * 2 * 4;
    assert_eq!(data.len(), 44 + expected, "float track length");
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn stereo_downmixes_to_mono_track() {
    let path = temp_path("to_mono");
    let mut src = AudioTestSrc::new(48_000, 2, 1_000, 10);
    let mut conv = AudioConvert::new(AudioFormat::PcmS16Le, 1);
    let mut sink = WavSink::new(&path);

    run_source_transform_sink(&mut src, &mut conv, &mut sink, &NullClock, 4)
        .await
        .expect("stereo -> convert(mono) -> wav negotiates and flows");

    let data = std::fs::read(&path).expect("wav exists");
    assert_eq!(read_u16(&data, 22), 1, "mono after downmix");
    assert_eq!(read_u16(&data, 34), 16, "still s16");
    // 100 ms mono s16 = 48000 * 0.1 * 1 ch * 2 bytes.
    let expected = (48_000f64 * 0.1) as usize * 2;
    assert_eq!(data.len(), 44 + expected, "mono track length");
    let _ = std::fs::remove_file(&path);
}
