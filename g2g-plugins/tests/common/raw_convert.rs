#![allow(dead_code)] // no one test file uses every helper here

use std::path::{Path, PathBuf};

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{PipelineClock, RawVideoFormat};
use g2g_plugins::registry::default_registry;

pub(crate) struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

pub(crate) fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

pub(crate) fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-raw-convert-{tag}-{}.raw", std::process::id()))
}

pub(crate) fn read_fixture(name: &str) -> Vec<u8> {
    std::fs::read(fixture(name)).expect("the fixture is committed")
}

// what `rawvideoparse ! videoconvert ! <to>` writes for the frame in `input`
pub(crate) async fn convert(
    input: &Path,
    from: RawVideoFormat,
    to: RawVideoFormat,
    (width, height): (u32, u32),
) -> Vec<u8> {
    let out = temp_path(&format!(
        "{}-{width}-{}",
        input.file_name().unwrap().to_string_lossy(),
        to.gst_name()
    ));
    let line = format!(
        "filesrc location={} bytestream-format=raw ! rawvideoparse format={} width={width} height={height} ! videoconvert ! video/x-raw,format={} ! filesink location={}",
        input.display(),
        from.gst_name(),
        to.gst_name(),
        out.display()
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let bytes = std::fs::read(&out).expect("the sink wrote its file");
    std::fs::remove_file(&out).ok();
    bytes
}
