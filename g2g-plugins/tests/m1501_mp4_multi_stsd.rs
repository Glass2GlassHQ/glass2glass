#![cfg(feature = "std")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/mp4_demux.rs"]
mod mp4_demux;
use mp4_demux::demux;

// FFmpeg FATE extradata-reload-multi-stsd.mov: 4 samples, the last two under a second sample entry
const H264_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_multi_stsd.mov"
);
const H265_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h265_multi_stsd.mov"
);
const START_CODE: [u8; 4] = [0, 0, 0, 1];
const SAMPLES_PER_ENTRY: usize = 2;
const H264_SPS: u8 = 7;
const H265_VPS: u8 = 32;

fn first_nal(frame: &[u8]) -> &[u8] {
    let body = frame.strip_prefix(&START_CODE).expect("annex-b frame");
    let end = body
        .windows(START_CODE.len())
        .position(|window| window == START_CODE)
        .unwrap_or(body.len());
    &body[..end]
}

async fn assert_second_entry_sets_go_in_band(path: &str, is_parameter_set: fn(&[u8]) -> bool) {
    let frames = demux(&std::fs::read(path).expect("fixture reads"))
        .await
        .frames;
    assert_eq!(frames.len(), 2 * SAMPLES_PER_ENTRY);
    let (first_entry, second_entry) = frames.split_at(SAMPLES_PER_ENTRY);
    assert!(is_parameter_set(first_nal(&first_entry[0])));
    assert!(is_parameter_set(first_nal(&second_entry[0])));
    assert_ne!(
        first_nal(&first_entry[0]),
        first_nal(&second_entry[0]),
        "the second entry's own parameter sets lead its first sample"
    );
    assert!(!is_parameter_set(first_nal(&second_entry[1])));
}

#[tokio::test]
async fn h264_sample_entry_switch_carries_its_parameter_sets() {
    assert_second_entry_sets_go_in_band(H264_FIXTURE, |nal| nal[0] & 0x1F == H264_SPS).await;
}

#[tokio::test]
async fn h265_sample_entry_switch_carries_its_parameter_sets() {
    assert_second_entry_sets_go_in_band(H265_FIXTURE, |nal| (nal[0] >> 1) & 0x3F == H265_VPS).await;
}
