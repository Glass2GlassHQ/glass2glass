//! M1545: a container extension is checked against the file's own header, so a
//! transport stream saved as `.mpg` demuxes as one, and `mpegpsdemux` fails on
//! bytes that never show a pack header instead of ending early with success.

#![cfg(all(feature = "std", target_os = "linux", feature = "ffmpeg"))]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::G2gError;
use g2g_plugins::registry::default_registry;

const TS_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/mpeg2_field_pictures_unaligned_pes.ts"
);

struct ZeroClock;
impl g2g_core::PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp(tag: &str, ext: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("g2g-m1545-{tag}-{}.{ext}", std::process::id()))
}

async fn run(line: &str) -> Result<(), G2gError> {
    let graph =
        parse_launch(&default_registry(), line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4).await.map(|_| ())
}

async fn decode(source: &std::path::Path, tag: &str) -> Vec<u8> {
    let yuv = temp(tag, "yuv");
    let line = format!(
        "filesrc location={} ! decodebin ! video/x-raw,format=I420 ! filesink location={}",
        source.display(),
        yuv.display()
    );
    run(&line)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let bytes = std::fs::read(&yuv).expect("decoded yuv");
    std::fs::remove_file(&yuv).ok();
    bytes
}

#[tokio::test]
async fn a_transport_stream_named_mpg_decodes_like_the_ts() {
    let mpg = temp("ts", "mpg");
    std::fs::copy(TS_FIXTURE, &mpg).expect("copy fixture");
    let from_mpg = decode(&mpg, "mpg").await;
    std::fs::remove_file(&mpg).ok();
    let from_ts = decode(std::path::Path::new(TS_FIXTURE), "ts").await;
    assert!(!from_ts.is_empty());
    assert!(
        from_mpg == from_ts,
        "the `.mpg` copy decodes the same frames"
    );
}

#[tokio::test]
async fn mpegpsdemux_rejects_bytes_with_no_pack_header() {
    let line =
        format!("filesrc location={TS_FIXTURE} bytestream-format=mpegps ! mpegpsdemux ! fakesink");
    let result = run(&line).await;
    assert!(
        matches!(result, Err(G2gError::CapsMismatch)),
        "got {result:?}"
    );
}
