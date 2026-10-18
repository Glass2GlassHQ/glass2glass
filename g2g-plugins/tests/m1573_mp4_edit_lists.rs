#![cfg(feature = "ffmpeg")]

#[path = "common/audio_mp4.rs"]
mod audio_mp4;

use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{PipelineClock, RawVideoFormat};
use g2g_plugins::registry::default_registry;

use audio_mp4::{full_box, mp4_box};

// 30 frames at 25 fps with B-frames, its moov after its mdat
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_64x48_bframes_elst.mp4"
);
const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const FRAME_COUNT: usize = 30;
// the fixture's mvhd and mdhd timescales
const MOVIE_TIMESCALE: u32 = 1000;
const MEDIA_TIMESCALE: u32 = 12_800;
const FRAMES_PER_SECOND: u32 = 25;
const FRAME_TICKS: u32 = MEDIA_TIMESCALE / FRAMES_PER_SECOND;
// the first frame presents two frames late, behind the B-frame reorder
const FIRST_PTS_TICKS: u32 = 2 * FRAME_TICKS;
// an elst media_time of -1
const EMPTY_EDIT: i32 = -1;
// an elst media_rate of 1.0 in 16.16 fixed point
const NORMAL_RATE: u32 = 0x0001_0000;

// a span of the fixture's frames in presentation order
#[derive(Clone, Copy)]
struct Frames {
    first: usize,
    count: usize,
}

const ALL_FRAMES: Frames = Frames {
    first: 0,
    count: FRAME_COUNT,
};

// one elst entry: the frames it presents, or an empty edit of that many frames
enum Edit {
    Media(Frames),
    Empty { count: usize },
}

fn movie_ticks(frames: usize) -> u32 {
    frames as u32 * FRAME_TICKS * MOVIE_TIMESCALE / MEDIA_TIMESCALE
}

fn elst(edits: &[Edit]) -> Vec<u8> {
    let mut payload = (edits.len() as u32).to_be_bytes().to_vec();
    for edit in edits {
        let (duration, media_time) = match edit {
            Edit::Media(frames) => (
                movie_ticks(frames.count),
                (FIRST_PTS_TICKS + frames.first as u32 * FRAME_TICKS) as i32,
            ),
            Edit::Empty { count } => (movie_ticks(*count), EMPTY_EDIT),
        };
        payload.extend(duration.to_be_bytes());
        payload.extend(media_time.to_be_bytes());
        payload.extend(NORMAL_RATE.to_be_bytes());
    }
    full_box(b"elst", &payload)
}

// the boxes with every elst replaced, no chunk offset moving because the moov comes last
fn with_elst(boxes: &[u8], elst: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < boxes.len() {
        let size = u32::from_be_bytes(boxes[at..at + 4].try_into().unwrap()) as usize;
        let kind: &[u8; 4] = boxes[at + 4..at + 8].try_into().unwrap();
        match kind {
            b"moov" | b"trak" | b"edts" => {
                out.extend(mp4_box(kind, &with_elst(&boxes[at + 8..at + size], elst)))
            }
            b"elst" => out.extend_from_slice(elst),
            _ => out.extend_from_slice(&boxes[at..at + size]),
        }
        at += size;
    }
    out
}

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m1573-{}-{name}", std::process::id()))
}

async fn decoded_frames(edits: &[Edit], decode_chain: &str, name: &str) -> Vec<Vec<u8>> {
    let fixture = std::fs::read(FIXTURE).expect("fixture reads");
    let input = temp_path(&format!("{name}.mp4"));
    std::fs::write(&input, with_elst(&fixture, &elst(edits))).expect("edited file writes");
    let output = temp_path(&format!("{name}.yuv"));
    let line = format!(
        "filesrc location={} ! {decode_chain} ! video/x-raw,format=I420 ! filesink location={}",
        input.display(),
        output.display()
    );
    let graph = parse_launch(&default_registry(), &line).expect("launch line parses");
    run_graph(graph, &ZeroClock, 4).await.expect("decode runs");
    let raw = std::fs::read(&output).expect("decode output reads");
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
    let frame_len = RawVideoFormat::I420
        .unpadded_frame_bytes(WIDTH, HEIGHT)
        .expect("I420 frame size") as usize;
    assert_eq!(raw.len() % frame_len, 0, "whole I420 frames");
    raw.chunks(frame_len).map(<[u8]>::to_vec).collect()
}

fn presented(all: &[Vec<u8>], spans: &[Frames]) -> Vec<Vec<u8>> {
    spans
        .iter()
        .flat_map(|span| all[span.first..span.first + span.count].iter().cloned())
        .collect()
}

// the first edit ends on a B-frame whose later-decoded reference it cuts, the
// second starts on a B-frame past the keyframe it needs
const FIRST_EDIT: Frames = Frames { first: 0, count: 8 };
const SECOND_EDIT: Frames = Frames {
    first: 14,
    count: 10,
};

async fn presents_both_edits(decode_chain: &str, name: &str) {
    let all = decoded_frames(
        &[Edit::Media(ALL_FRAMES)],
        decode_chain,
        &format!("{name}-all"),
    )
    .await;
    assert_eq!(all.len(), FRAME_COUNT);
    let edited = decoded_frames(
        &[Edit::Media(FIRST_EDIT), Edit::Media(SECOND_EDIT)],
        decode_chain,
        name,
    )
    .await;
    assert_eq!(edited.len(), FIRST_EDIT.count + SECOND_EDIT.count);
    assert!(edited == presented(&all, &[FIRST_EDIT, SECOND_EDIT]));
}

#[tokio::test]
async fn decodebin_presents_two_edits_back_to_back() {
    presents_both_edits("decodebin", "decodebin").await;
}

#[tokio::test]
async fn multi_track_demux_presents_two_edits_back_to_back() {
    presents_both_edits("qtdemux name=d d.video_0 ! ffmpegdec", "qtdemux").await;
}

#[tokio::test]
async fn an_empty_first_edit_keeps_the_media_edit_after_it() {
    const MEDIA_EDIT: Frames = Frames {
        first: 3,
        count: 12,
    };
    let all = decoded_frames(&[Edit::Media(ALL_FRAMES)], "decodebin", "empty-first-all").await;
    let edited = decoded_frames(
        &[Edit::Empty { count: 5 }, Edit::Media(MEDIA_EDIT)],
        "decodebin",
        "empty-first",
    )
    .await;
    assert_eq!(edited.len(), MEDIA_EDIT.count);
    assert!(edited == presented(&all, &[MEDIA_EDIT]));
}
