//! Reading the `ffprobe -of json` output checked in beside a media fixture, so a
//! test asserts against probed values instead of numbers typed into it.
//!
//! The fixtures here are single-stream, so the first value of each key is the
//! one wanted and the reader needs no JSON object model.
#![allow(dead_code)] // no one test file reads every field here

use g2g_core::{AudioFormat, Caps};

/// A checked-in fixture's path, from its file name.
pub(crate) fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// What the checked-in `ffprobe` output says about a fixture.
pub(crate) struct Probe {
    json: String,
    pub(crate) sample_rate: u32,
    pub(crate) channels: u8,
    /// Frames ffmpeg's demuxer reads, the count a parser must match.
    pub(crate) frames: u64,
    pub(crate) duration_seconds: f64,
}

impl Probe {
    /// Read the probe output beside the fixture named `name`.
    pub(crate) fn read(name: &str) -> Probe {
        let path = fixture_path(name).with_extension("json");
        let json = std::fs::read_to_string(&path).expect("the probe output is checked in");
        let value =
            |key: &str| json_value(&json, key).unwrap_or_else(|| panic!("{key} in {path:?}"));
        Probe {
            sample_rate: value("sample_rate").parse().expect("a sample rate"),
            channels: value("channels").parse().expect("a channel count"),
            frames: value("nb_read_frames").parse().expect("a frame count"),
            duration_seconds: value("duration").parse().expect("a duration"),
            json,
        }
    }

    /// The probed layout as compressed-audio caps of `format`.
    pub(crate) fn audio_caps(&self, format: AudioFormat) -> Caps {
        Caps::Audio {
            format,
            channels: self.channels,
            sample_rate: self.sample_rate,
        }
    }

    /// Any other probed field, as text.
    pub(crate) fn text(&self, key: &str) -> String {
        json_value(&self.json, key).unwrap_or_else(|| panic!("{key} in the probe output"))
    }
}

/// The value of `"key"` in `json`, as text: a quoted string without its quotes,
/// or a bare number.
fn json_value(json: &str, key: &str) -> Option<String> {
    let at = json.find(&format!("\"{key}\""))? + key.len() + 2;
    let rest = json[at..].trim_start().strip_prefix(':')?.trim_start();
    match rest.strip_prefix('"') {
        Some(quoted) => Some(quoted[..quoted.find('"')?].to_string()),
        None => Some(
            rest[..rest.find([',', '\n', '}'])?]
                .trim()
                .trim_end_matches(',')
                .to_string(),
        ),
    }
}
