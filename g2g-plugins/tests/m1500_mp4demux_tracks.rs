#![cfg(feature = "std")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/mp4_demux.rs"]
mod mp4_demux;
use g2g_core::Caps;
use mp4_demux::demux;

const FRAGMENTED_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_64x48_bframes_frag.mp4"
);
// one "pts,dts" line per sample of the fragmented fixture
const FRAGMENTED_FIXTURE_TIMES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_64x48_bframes_frag.mp4.pts_dts"
);
// moov at the end, video trak first, then audio
const AV_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av_h264_aac.mp4"
);
const BOX_HEADER_LEN: usize = 8;

// (offset, size, type) of each box laid out back to back in `data`
fn boxes(data: &[u8]) -> Vec<(usize, usize, [u8; 4])> {
    let mut out = Vec::new();
    let mut at = 0;
    while at + BOX_HEADER_LEN <= data.len() {
        let size = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as usize;
        out.push((at, size, data[at + 4..at + 8].try_into().unwrap()));
        at += size;
    }
    out
}

#[tokio::test]
async fn mp4demux_reads_a_fragmented_file() {
    let file = std::fs::read(FRAGMENTED_FIXTURE).expect("fixture reads");
    let sample_count = std::fs::read_to_string(FRAGMENTED_FIXTURE_TIMES)
        .expect("timing list reads")
        .lines()
        .count();
    let out = demux(&file).await;
    assert!(matches!(
        out.caps.as_slice(),
        [Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            ..
        }]
    ));
    assert_eq!(out.frames.len(), sample_count);
}

#[tokio::test]
async fn mp4demux_finds_the_video_trak_behind_an_audio_trak() {
    let file = std::fs::read(AV_FIXTURE).expect("fixture reads");
    let expected = demux(&file).await.frames;
    assert!(!expected.is_empty());

    let (moov_at, moov_size, _) = *boxes(&file)
        .iter()
        .find(|(_, _, kind)| kind == b"moov")
        .expect("moov");
    let moov = &file[moov_at + BOX_HEADER_LEN..moov_at + moov_size];
    let children = boxes(moov);
    let traks: Vec<&[u8]> = children
        .iter()
        .filter(|(_, _, kind)| kind == b"trak")
        .map(|&(at, size, _)| &moov[at..at + size])
        .collect();
    let trak_boxes: Vec<_> = children
        .iter()
        .filter(|(_, _, kind)| kind == b"trak")
        .collect();
    assert_eq!(trak_boxes.len(), 2, "video and audio trak");
    let first_trak_at = trak_boxes[0].0;
    assert_eq!(
        first_trak_at + trak_boxes[0].1,
        trak_boxes[1].0,
        "adjacent traks"
    );
    // same total length, so the absolute chunk offsets into the earlier mdat still hold
    let mut swapped = file.clone();
    let swapped_traks = [traks[1], traks[0]].concat();
    let traks_at = moov_at + BOX_HEADER_LEN + first_trak_at;
    swapped[traks_at..traks_at + swapped_traks.len()].copy_from_slice(&swapped_traks);

    assert_eq!(demux(&swapped).await.frames, expected);
}
