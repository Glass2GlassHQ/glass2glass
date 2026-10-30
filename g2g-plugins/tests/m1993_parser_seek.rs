//! M1993: a bare `.mp3`, `.aac` or `.ac3` file seeks through its parser, which
//! rescans from the file start and hands the decoder the frames it needs before
//! the target, so the decode from the target matches ffmpeg's own decode.
#![cfg(all(feature = "std", target_os = "linux", feature = "ffmpeg"))]

use std::path::{Path, PathBuf};
use std::process::Command;

const CHANNELS: usize = 2;
const BYTES_PER_SAMPLE: usize = 2;
const SEEK_MILLISECONDS: usize = 750;
// three tones so the encoders spend uneven bits per frame
const SOURCE: &str = "sine=frequency=440:duration=2:sample_rate={rate}[a];\
                      sine=frequency=1250:duration=2:sample_rate={rate}[b];\
                      sine=frequency=3170:duration=2:sample_rate={rate}[c];\
                      [a][b][c]amix=inputs=3";

struct Format {
    extension: &'static str,
    encoder: &'static [&'static str],
    sample_rate: usize,
    // decoder state carried from frames before the preroll moves ffmpeg's own decode by up to this much
    tolerance: i32,
}

// a low bitrate makes the MP3 frames borrow from several frames back
const MP3: Format = Format {
    extension: "mp3",
    encoder: &["-c:a", "libmp3lame", "-b:a", "64k"],
    sample_rate: 44_100,
    tolerance: 0,
};

const AAC: Format = Format {
    extension: "aac",
    encoder: &["-c:a", "aac", "-b:a", "128k", "-f", "adts"],
    sample_rate: 44_100,
    tolerance: 40,
};

const AC3: Format = Format {
    extension: "ac3",
    encoder: &["-c:a", "ac3", "-b:a", "192k"],
    sample_rate: 48_000,
    tolerance: 2,
};

fn temp_path(tag: &str, extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "g2g_m1993_{tag}_{}.{extension}",
        std::process::id()
    ))
}

fn encode(format: &Format, path: &Path) -> bool {
    Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-filter_complex"])
        .arg(SOURCE.replace("{rate}", &format.sample_rate.to_string()))
        .args(["-ac", &CHANNELS.to_string()])
        .args(format.encoder)
        .arg(path)
        .status()
        .is_ok_and(|status| status.success())
}

fn ffmpeg_pcm(path: &Path) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-loglevel", "error", "-i"])
        .arg(path)
        .args(["-f", "s16le", "-c:a", "pcm_s16le", "pipe:1"])
        .output()
        .expect("ffmpeg runs");
    assert!(out.status.success(), "ffmpeg decodes {}", path.display());
    out.stdout
}

fn samples(pcm: &[u8]) -> Vec<i16> {
    pcm.as_chunks::<BYTES_PER_SAMPLE>()
        .0
        .iter()
        .map(|&sample| i16::from_le_bytes(sample))
        .collect()
}

fn seeks_to_the_sample(format: &Format) {
    let src = temp_path("seek", format.extension);
    if !encode(format, &src) {
        eprintln!("skipping: no ffmpeg {}", format.extension);
        return;
    }
    let reference = ffmpeg_pcm(&src);
    let out = temp_path(format.extension, "raw");
    let _ = std::fs::remove_file(&out);
    let line = format!(
        "filesrc location={} ! decodebin ! audioconvert ! \
         audio/x-raw,format=S16LE,rate={},channels={CHANNELS} ! filesink location={}",
        src.display(),
        format.sample_rate,
        out.display()
    );
    let launched = Command::new(env!("CARGO_BIN_EXE_g2g-launch"))
        .args([
            "-q",
            "--seek",
            &format!("{}", SEEK_MILLISECONDS as f64 / 1_000.0),
        ])
        .args(line.split_whitespace())
        .output()
        .expect("g2g-launch starts");
    let pcm = std::fs::read(&out);
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
    assert!(
        launched.status.success(),
        "{line}: {}",
        String::from_utf8_lossy(&launched.stderr)
    );
    let first_byte = format.sample_rate * SEEK_MILLISECONDS / 1_000 * CHANNELS * BYTES_PER_SAMPLE;
    let expected = samples(&reference[first_byte..]);
    let decoded = samples(&pcm.expect("pcm written"));
    assert_eq!(
        decoded.len(),
        expected.len(),
        "{}: the decode runs from the target sample to the end",
        format.extension
    );
    let worst = decoded
        .iter()
        .zip(&expected)
        .map(|(&got, &want)| (i32::from(got) - i32::from(want)).abs())
        .max()
        .unwrap_or(0);
    assert!(
        worst <= format.tolerance,
        "{}: the decode after the seek is off ffmpeg's by up to {worst}",
        format.extension
    );
}

#[test]
fn mp3_file_seeks_to_the_sample() {
    seeks_to_the_sample(&MP3);
}

#[test]
fn aac_file_seeks_to_the_sample() {
    seeks_to_the_sample(&AAC);
}

#[test]
fn ac3_file_seeks_to_the_sample() {
    seeks_to_the_sample(&AC3);
}
