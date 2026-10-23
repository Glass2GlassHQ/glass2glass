// A `decodebin` or `parsebin` fan-out splits every file the inline bin plays, so each
// branch carries the same stream the inline bin picks with `current-video` / `current-audio`.
#![cfg(all(feature = "std", feature = "ffmpeg", target_os = "linux"))]

use std::path::{Path, PathBuf};

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

// one file per container no multi-output demuxer parses, and a WAV whose samples
// the program-stream probe reads as a PES packet
const FIXTURES: [(&str, &[&str]); 6] = [
    ("sine_8k_mono_s16le.wav", &["audio"]),
    ("s16_stereo_pes_start_code.wav", &["audio"]),
    ("aac_stereo_44100.adts", &["audio"]),
    ("vp9_320x240.ivf", &["video"]),
    ("vp6f_112x80.flv", &["video"]),
    ("flv_h264_aac.flv", &["video", "audio"]),
];
const BINS: [&str; 2] = ["decodebin", "parsebin"];

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m1847-{}-{name}", std::process::id()))
}

async fn run(line: &str) {
    let graph =
        parse_launch(&default_registry(), line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
}

fn take(path: &Path) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    std::fs::remove_file(path).ok();
    bytes
}

#[tokio::test]
async fn a_fanout_branch_carries_the_stream_the_inline_bin_picks() {
    for (name, kinds) in FIXTURES {
        let source = fixture(name);
        for bin in BINS {
            let mut branches = String::new();
            let mut outputs = Vec::new();
            for kind in kinds {
                let out = temp_path(&format!("{name}-{bin}-fanout-{kind}"));
                branches.push_str(&format!(
                    "  b.{kind}_0 ! queue ! filesink location={}",
                    out.display()
                ));
                outputs.push((kind, out));
            }
            run(&format!(
                "filesrc location={} ! {bin} name=b{branches}",
                source.display()
            ))
            .await;
            for (kind, out) in outputs {
                let fanned_out = take(&out);
                let inline_out = temp_path(&format!("{name}-{bin}-inline-{kind}"));
                run(&format!(
                    "filesrc location={} ! {bin} current-{kind}=0 ! filesink location={}",
                    source.display(),
                    inline_out.display()
                ))
                .await;
                let inline = take(&inline_out);
                assert!(!inline.is_empty(), "{name} {bin} {kind} is empty inline");
                assert!(
                    fanned_out == inline,
                    "{name}: the {bin} fan-out {kind} branch differs from the inline {bin}"
                );
            }
        }
    }
}
