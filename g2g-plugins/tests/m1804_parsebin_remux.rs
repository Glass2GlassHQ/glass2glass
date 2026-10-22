// `parsebin` demuxes and parses without decoding, so its streams remux into another
// container, and the remuxed file decodes to the same pictures and samples as the original.
#![cfg(all(feature = "std", feature = "ffmpeg", target_os = "linux"))]

use std::path::{Path, PathBuf};

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

const AV_FIXTURE: &str = "av_h264_aac.mp4";
const MULTITRACK_FIXTURE: &str = "multitrack_h264_aac.mkv";
const MUXERS: [(&str, &str); 4] = [
    ("mp4mux", "mp4"),
    ("matroskamux", "mkv"),
    ("mpegtsmux", "ts"),
    ("flvmux", "flv"),
];

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
    std::env::temp_dir().join(format!("g2g-m1804-{}-{name}", std::process::id()))
}

async fn run(line: &str) {
    let graph =
        parse_launch(&default_registry(), line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
}

// the decoder's raw output for one stream, as the bytes a filesink writes
async fn decoded(path: &Path, property: &str, tag: &str) -> Vec<u8> {
    let out = temp_path(&format!("{tag}.raw"));
    run(&format!(
        "filesrc location={} ! decodebin {property} ! filesink location={}",
        path.display(),
        out.display()
    ))
    .await;
    let bytes = std::fs::read(&out).expect("the decode wrote its file");
    std::fs::remove_file(&out).ok();
    assert!(
        !bytes.is_empty(),
        "{} {property} decoded nothing",
        path.display()
    );
    bytes
}

fn remux_line(source: &Path, muxer: &str, remuxed: &Path) -> String {
    format!(
        "filesrc location={} ! parsebin name=p  \
         p.video_0 ! queue ! m.  p.audio_0 ! queue ! m.  \
         {muxer} name=m ! filesink location={}",
        source.display(),
        remuxed.display()
    )
}

#[test]
fn a_parsebin_plugs_parsers_and_no_decoder() {
    let line = remux_line(&fixture(AV_FIXTURE), "mp4mux", &temp_path("unused.mp4"));
    let graph = parse_launch(&default_registry(), &line)
        .unwrap_or_else(|e| panic!("parses `{line}`: {e}"))
        .finish()
        .expect("valid graph");
    let names: Vec<&str> = graph
        .topo()
        .iter()
        .filter_map(|&node| graph.element(node).map(|element| element.log_category()))
        .collect();
    for parser in ["NalParse", "AacParse"] {
        assert!(names.contains(&parser), "{parser} in {names:?}");
    }
    assert!(
        !names.iter().any(|name| name.ends_with("Dec")),
        "no decoder in {names:?}"
    );
}

#[tokio::test]
async fn a_parsebin_remux_decodes_like_the_original() {
    let source = fixture(AV_FIXTURE);
    let video = decoded(&source, "current-video=0", "source-video").await;
    let audio = decoded(&source, "current-audio=0", "source-audio").await;
    for (muxer, extension) in MUXERS {
        let remuxed = temp_path(&format!("remux.{extension}"));
        run(&remux_line(&source, muxer, &remuxed)).await;
        let remuxed_video = decoded(&remuxed, "current-video=0", "remux-video").await;
        let remuxed_audio = decoded(&remuxed, "current-audio=0", "remux-audio").await;
        std::fs::remove_file(&remuxed).ok();
        assert!(
            remuxed_video == video,
            "{muxer} remux changed the decoded video"
        );
        // a remux keeps AAC samples the source edit list cuts, as ffmpeg -c copy does
        assert!(
            remuxed_audio
                .windows(audio.len())
                .any(|samples| samples == audio.as_slice()),
            "{muxer} remux lost or changed decoded audio samples"
        );
    }
}

#[tokio::test]
async fn an_inline_parsebin_remuxes_the_stream_its_index_names() {
    let source = fixture(MULTITRACK_FIXTURE);
    let audio = decoded(&source, "current-audio=1", "multitrack-audio").await;
    let remuxed = temp_path("multitrack-audio.mkv");
    run(&format!(
        "filesrc location={} ! parsebin current-audio=1 ! matroskamux ! filesink location={}",
        source.display(),
        remuxed.display()
    ))
    .await;
    let remuxed_audio = decoded(&remuxed, "current-audio=0", "multitrack-remux-audio").await;
    std::fs::remove_file(&remuxed).ok();
    assert!(
        remuxed_audio == audio,
        "the remuxed second audio track decodes like the original's"
    );
}
