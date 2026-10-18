//! M1544: `decodebin` splices `mpegvideoparse` ahead of the MPEG-2 decoder, so a
//! transport stream whose PES packets split field pictures decodes bit-exact.

#![cfg(all(feature = "std", target_os = "linux", feature = "ffmpeg"))]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_plugins::registry::default_registry;
use md5::{Digest, Md5};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/mpeg2_field_pictures_unaligned_pes.ts"
);

struct ZeroClock;
impl g2g_core::PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

// ffmpeg's `-f framemd5` lines: `stream, dts, pts, duration, size, hash`
fn ffmpeg_frames() -> Vec<(usize, String)> {
    let text = std::fs::read_to_string(format!("{FIXTURE}.framemd5")).expect("framemd5 sidecar");
    text.lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            (
                fields[4].parse().expect("frame size"),
                fields[5].to_string(),
            )
        })
        .collect()
}

#[tokio::test]
async fn field_coded_mpeg2_in_ts_decodes_like_ffmpeg() {
    let yuv = std::env::temp_dir().join(format!("g2g-m1544-{}.yuv", std::process::id()));
    let line = format!(
        "filesrc location={FIXTURE} ! decodebin ! video/x-raw,format=I420 ! filesink location={}",
        yuv.display()
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let decoded = std::fs::read(&yuv).expect("decoded yuv");
    std::fs::remove_file(&yuv).ok();

    let expected = ffmpeg_frames();
    let frame_size = expected[0].0;
    assert_eq!(decoded.len(), frame_size * expected.len(), "frame count");
    let got: Vec<String> = decoded
        .chunks(frame_size)
        .map(|frame| {
            Md5::digest(frame)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect()
        })
        .collect();
    let want: Vec<String> = expected.into_iter().map(|(_, hash)| hash).collect();
    assert_eq!(got, want);
}
