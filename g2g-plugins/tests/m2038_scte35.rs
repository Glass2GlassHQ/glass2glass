//! M2038: SCTE-35 splice sections ride `PipelinePacket::SpliceInfo` from
//! `tsdemux` to `mpegtsmux`, which writes them on `scte-35-pid`. The input is a
//! TSDuck splice_insert (`tools/scte35-fixture.sh`), and ffprobe reads both files:
//! the splice point keeps its distance from the first video PTS, and the rest of
//! the section is unchanged.
#![cfg(feature = "std")]

#[path = "common/ffprobe_tool.rs"]
mod ffprobe_tool;
use ffprobe_tool::have_ffprobe;

use std::path::{Path, PathBuf};
use std::process::Command;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_scte35_splice_insert.ts"
);
const SCTE35_PID: u16 = 500;
const SPLICE_INSERT: u8 = 0x05;
const PTS_MODULUS: u64 = 1 << 33;
// table_id through cw_index, the bytes before tier
const PTS_ADJUSTMENT_AT: usize = 4;
const TIER_AT: usize = 9;
const COMMAND_TYPE_AT: usize = 13;
// splice_event_id, flag bytes, then the splice_time byte holding time_specified_flag and pts bit 32
const SPLICE_TIME_AT: usize = 20;
const CRC_LEN: usize = 4;

struct ZeroClock;

impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

struct Probed {
    first_video_pts: u64,
    sections: Vec<Vec<u8>>,
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m2038-{}-{name}", std::process::id()))
}

fn ffprobe(path: &Path, args: &[&str]) -> String {
    let output = Command::new("ffprobe")
        .args(["-v", "error"])
        .args(args)
        .arg(path)
        .output()
        .expect("run ffprobe");
    assert!(output.status.success(), "ffprobe reads {}", path.display());
    String::from_utf8(output.stdout).expect("ffprobe prints text")
}

// ffprobe's `-show_data` hex dump lines, `00000000: fc30 2500 ...  ascii`
fn hex_dump_bytes(dump: &str) -> Vec<u8> {
    dump.split("\\n")
        .filter_map(|line| line.split_once(": "))
        .flat_map(|(_, rest)| {
            let hex = rest.split("  ").next().unwrap_or_default().replace(' ', "");
            (0..hex.len())
                .step_by(2)
                .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).expect("hex byte"))
                .collect::<Vec<u8>>()
        })
        .collect()
}

fn probe(path: &Path) -> Probed {
    let video = ffprobe(
        path,
        &[
            "-select_streams",
            "v",
            "-show_entries",
            "packet=pts",
            "-of",
            "csv=p=0",
        ],
    );
    let first_video_pts = video
        .lines()
        .next()
        .expect("a video packet")
        .trim_end_matches(',')
        .parse()
        .expect("a video pts");
    let data = ffprobe(
        path,
        &[
            "-select_streams",
            "d",
            "-show_packets",
            "-show_data",
            "-of",
            "json",
        ],
    );
    let sections = data
        .lines()
        .filter_map(|line| line.trim().strip_prefix("\"data\": \""))
        .map(|dump| hex_dump_bytes(dump.trim_end_matches('"')))
        .collect();
    Probed {
        first_video_pts,
        sections,
    }
}

fn pts_adjustment(section: &[u8]) -> u64 {
    let field = &section[PTS_ADJUSTMENT_AT..PTS_ADJUSTMENT_AT + 5];
    (u64::from(field[0] & 0x01) << 32)
        | u64::from(u32::from_be_bytes([field[1], field[2], field[3], field[4]]))
}

fn splice_pts(section: &[u8]) -> u64 {
    let field = &section[SPLICE_TIME_AT..SPLICE_TIME_AT + 5];
    assert!(field[0] & 0x80 != 0, "the splice_insert has a time");
    (u64::from(field[0] & 0x01) << 32)
        | u64::from(u32::from_be_bytes([field[1], field[2], field[3], field[4]]))
}

// the splice point less the first video PTS, on the 33-bit clock
fn splice_offset(probed: &Probed) -> Vec<u64> {
    splice_inserts(&probed.sections)
        .map(|section| {
            let point = (splice_pts(section) + pts_adjustment(section)) % PTS_MODULUS;
            (point + PTS_MODULUS - probed.first_video_pts % PTS_MODULUS) % PTS_MODULUS
        })
        .collect()
}

fn splice_inserts(sections: &[Vec<u8>]) -> impl Iterator<Item = &Vec<u8>> {
    sections
        .iter()
        .filter(|section| section.get(COMMAND_TYPE_AT) == Some(&SPLICE_INSERT))
}

// tier through the descriptors, the bytes a remux must keep
fn command_bytes(section: &[u8]) -> &[u8] {
    &section[TIER_AT..section.len() - CRC_LEN]
}

async fn remux(line: &str) {
    let graph = parse_launch(&default_registry(), line).unwrap_or_else(|e| panic!("`{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("`{line}` runs: {e:?}"));
}

async fn check_remux(chain: &str, name: &str) {
    let output = temp_path(name);
    remux(&format!(
        "filesrc location={FIXTURE} bytestream-format=mpegts ! tsdemux ! h264parse ! {chain} \
         ! mpegtsmux scte-35-pid={SCTE35_PID} ! filesink location={}",
        output.display()
    ))
    .await;
    let input = probe(Path::new(FIXTURE));
    let remuxed = probe(&output);
    let _ = std::fs::remove_file(&output);

    let input_offsets = splice_offset(&input);
    assert!(
        !input_offsets.is_empty(),
        "the fixture carries a splice_insert"
    );
    assert_eq!(
        splice_offset(&remuxed),
        input_offsets,
        "{name}: splice point against the video"
    );
    let input_commands: Vec<&[u8]> = splice_inserts(&input.sections)
        .map(|s| command_bytes(s))
        .collect();
    let remuxed_commands: Vec<&[u8]> = splice_inserts(&remuxed.sections)
        .map(|s| command_bytes(s))
        .collect();
    assert_eq!(
        remuxed_commands, input_commands,
        "{name}: splice command bytes"
    );
}

#[tokio::test]
async fn a_remux_keeps_the_splice_point_on_the_video() {
    if !have_ffprobe() {
        eprintln!("skipping m2038: no ffprobe on PATH");
        return;
    }
    check_remux("identity", "remux.ts").await;
}

#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn a_transcode_keeps_the_splice_point_on_the_video() {
    if !have_ffprobe() {
        eprintln!("skipping m2038 transcode: no ffprobe on PATH");
        return;
    }
    check_remux("avdec_h264 ! x264enc", "transcode.ts").await;
}

#[tokio::test]
async fn without_a_pid_the_muxer_drops_the_sections() {
    if !have_ffprobe() {
        eprintln!("skipping m2038: no ffprobe on PATH");
        return;
    }
    let output = temp_path("dropped.ts");
    remux(&format!(
        "filesrc location={FIXTURE} bytestream-format=mpegts ! tsdemux ! h264parse \
         ! mpegtsmux ! filesink location={}",
        output.display()
    ))
    .await;
    let data_streams = ffprobe(
        &output,
        &[
            "-select_streams",
            "d",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
        ],
    );
    let _ = std::fs::remove_file(&output);
    assert_eq!(
        data_streams.trim(),
        "",
        "no SCTE-35 stream without scte-35-pid"
    );
}
