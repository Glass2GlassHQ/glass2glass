//! A bare `decodebin` decode checked frame by frame against ffmpeg's
//! `-f framemd5` checksums of the same fixture, kept beside it as `<fixture>.framemd5`.

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_plugins::registry::default_registry;
use md5::{Digest, Md5};

struct ZeroClock;
impl g2g_core::PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

pub(crate) struct FrameChecksums {
    pub(crate) frame_size: usize,
    pub(crate) md5s: Vec<String>,
}

// ffmpeg's framemd5 lines: `stream, dts, pts, duration, size, hash`
pub(crate) fn ffmpeg_frame_md5s(fixture: &str) -> FrameChecksums {
    let text = std::fs::read_to_string(format!("{fixture}.framemd5")).expect("framemd5 sidecar");
    let rows: Vec<Vec<&str>> = text
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split(',').map(str::trim).collect())
        .collect();
    FrameChecksums {
        frame_size: rows[0][4].parse().expect("frame size"),
        md5s: rows.iter().map(|row| row[5].to_string()).collect(),
    }
}

pub(crate) async fn decoded_i420_md5s(fixture: &str, frame_size: usize) -> Vec<String> {
    let name = std::path::Path::new(fixture)
        .file_name()
        .expect("fixture file name")
        .to_string_lossy();
    let yuv = std::env::temp_dir().join(format!("g2g-{name}-{}.yuv", std::process::id()));
    let line = format!(
        "filesrc location={fixture} ! decodebin ! video/x-raw,format=I420 ! filesink location={}",
        yuv.display()
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let decoded = std::fs::read(&yuv).expect("decoded yuv");
    std::fs::remove_file(&yuv).ok();
    decoded
        .chunks(frame_size)
        .map(|frame| {
            Md5::digest(frame)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect()
        })
        .collect()
}
