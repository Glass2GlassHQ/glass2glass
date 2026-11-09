#![cfg(all(target_os = "linux", feature = "decklink"))]

use std::sync::Arc;

use g2g_core::{
    AsyncElement, ClockSync, Interlace, PipelinePacket, PropError, PropValue, RawVideoFormat,
};
use g2g_plugins::st291::{AncPacket, AncillaryMeta, CAPTION_DID, CEA608_SDID, CEA708_CDP_SDID};

#[path = "common/decklink_board.rs"]
mod decklink_board;
use decklink_board::{
    arrival_ns, buffer, manual_board_time, mode, picture_bytes, sink_on_mock, video_caps,
    ManualClock, NullOutput, SharedLog, BOARD_START_NS, ELECTED_START_NS,
};

const MODE: &str = "1080p25";
const CAPTION_LINE: u16 = 9;
const UNPOSITIONED_LINE: u16 = 0x7FF;

fn caption_packets() -> Vec<AncPacket> {
    let mut cdp = AncPacket::from_bytes(CAPTION_DID, CEA708_CDP_SDID, &[0x96, 0x69, 0x10]);
    cdp.line_number = CAPTION_LINE;
    let mut cea608 = AncPacket::from_bytes(CAPTION_DID, CEA608_SDID, &[0x80, 0x94, 0x2C]);
    cea608.line_number = UNPOSITIONED_LINE;
    Vec::from([cdp, cea608])
}

fn frame_with_packets(pts_ns: u64, packets: Vec<AncPacket>) -> PipelinePacket {
    let PipelinePacket::DataFrame(mut frame) =
        buffer(picture_bytes(RawVideoFormat::V210, mode(MODE)), pts_ns)
    else {
        unreachable!("buffer builds a data frame");
    };
    frame.meta.attach(AncillaryMeta { packets });
    PipelinePacket::DataFrame(frame)
}

// plays one frame carrying the caption packets, then one carrying none
async fn play_two_frames(device_number: u32, output_vanc: bool) -> SharedLog {
    let board = Arc::new(ManualClock::default());
    let elected = Arc::new(ManualClock::default());
    let (mut sink, log) = sink_on_mock(device_number, manual_board_time(&board));
    sink.set_property("output-vanc", PropValue::Bool(output_vanc))
        .expect("output-vanc");
    sink.configure_pipeline(&video_caps(
        RawVideoFormat::V210,
        mode(MODE),
        Interlace::Progressive,
    ))
    .expect("configure");
    sink.set_clock_sync(ClockSync::new(elected.clone(), ELECTED_START_NS));

    let frame_ns = mode(MODE).frame_duration_ns();
    let frames = [caption_packets(), Vec::new()];
    for (index, packets) in (0u64..).zip(frames) {
        let pts_ns = index * frame_ns;
        board.set(BOARD_START_NS + arrival_ns(index, pts_ns));
        elected.set(ELECTED_START_NS + arrival_ns(index, pts_ns));
        sink.process(frame_with_packets(pts_ns, packets), &mut NullOutput)
            .await
            .expect("frame");
    }
    log
}

#[tokio::test]
async fn output_vanc_sends_each_frames_packets_with_its_picture() {
    let log = play_two_frames(214301, true).await;
    let log = log.lock().expect("log");
    assert!(log.video_vanc);
    let ancillary: Vec<&[AncPacket]> = log
        .pictures
        .iter()
        .map(|picture| picture.ancillary.as_slice())
        .collect();
    assert_eq!(ancillary, [caption_packets().as_slice(), &[]]);
}

#[tokio::test]
async fn without_output_vanc_no_packets_reach_the_board() {
    let log = play_two_frames(214302, false).await;
    let log = log.lock().expect("log");
    assert!(!log.video_vanc);
    assert_eq!(log.pictures.len(), 2);
    assert!(log
        .pictures
        .iter()
        .all(|picture| picture.ancillary.is_empty()));
}

#[test]
fn output_vanc_cannot_change_once_video_is_enabled() {
    let (mut sink, _) = sink_on_mock(214303, manual_board_time(&Arc::default()));
    sink.configure_pipeline(&video_caps(
        RawVideoFormat::V210,
        mode(MODE),
        Interlace::Progressive,
    ))
    .expect("configure");
    assert_eq!(
        sink.set_property("output-vanc", PropValue::Bool(true)),
        Err(PropError::Value)
    );
    assert_eq!(
        sink.get_property("output-vanc"),
        Some(PropValue::Bool(false))
    );
}
