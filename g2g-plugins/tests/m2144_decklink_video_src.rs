// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "decklink"))]

use std::sync::{Arc, Mutex};

use g2g_core::meta::{CaptionMeta, CaptionTriple};
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AspectRatio, Caps, G2gError, PipelinePacket, PixelAspect, PropValue, RawVideoFormat,
};
use g2g_plugins::afd::{AfdBar, AfdBarMeta, BarData, ST2016_DID, ST2016_SDID};
use g2g_plugins::cea::{build_cdp, write_s334_1a, CcTriple, CDP_FRAME_RATE_29_97};
use g2g_plugins::decklink::{DisplayMode, InputEvent};
use g2g_plugins::decklinkvideosrc::DeckLinkVideoSrc;
use g2g_plugins::st291::{AncPacket, AncillaryMeta, CAPTION_DID, CEA608_SDID, CEA708_CDP_SDID};

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

#[path = "common/decklink_board.rs"]
mod decklink_board;
use decklink_board::{
    captured, manual_board_time, mock_input_opener, mode, send_from_board, EnabledInput, InputLog,
    SharedInputLog, BOARD_START_NS,
};

const CAPTION_LINE: u16 = 9;

fn src_on_mock(device_number: u32) -> (DeckLinkVideoSrc, SharedInputLog) {
    let log: SharedInputLog = Arc::new(Mutex::new(InputLog::default()));
    let time = manual_board_time(&Arc::default());
    let src =
        DeckLinkVideoSrc::with_opener(device_number, mock_input_opener(Arc::clone(&log), time));
    (src, log)
}

fn set(src: &mut DeckLinkVideoSrc, name: &str, value: PropValue) {
    src.set_property(name, value).expect(name);
}

// the board's time for the frame `index` periods after the first
fn board_time(mode: &DisplayMode, index: u64) -> u64 {
    BOARD_START_NS + index * mode.frame_duration_ns()
}

fn frame_pts(packets: &[PipelinePacket]) -> Vec<u64> {
    packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame.timing.pts_ns),
            _ => None,
        })
        .collect()
}

fn caption_packet() -> AncPacket {
    let mut packet = AncPacket::from_bytes(CAPTION_DID, CEA708_CDP_SDID, &[0x96, 0x69, 0x10]);
    packet.line_number = CAPTION_LINE;
    packet
}

fn afd_packet(line: u16, widescreen: bool) -> AncPacket {
    let mut packet = AncPacket::from_bytes(
        ST2016_DID,
        ST2016_SDID,
        &[
            0x20 | (u8::from(widescreen) << 2),
            0,
            0,
            0xc0,
            0xc0,
            0x64,
            0xc1,
            0xc0,
        ],
    );
    packet.line_number = line;
    packet
}

fn with_ancillary(mut event: InputEvent, packets: Vec<AncPacket>) -> InputEvent {
    let InputEvent::Frame(frame) = &mut event else {
        panic!("a captured frame");
    };
    frame.ancillary = packets;
    event
}

#[tokio::test]
async fn auto_sd_aspect_refines_caps_before_the_first_affected_frame() {
    let (mut src, log) = src_on_mock(214405);
    let ntsc = mode("ntsc");
    let pal = mode("pal");
    let format = RawVideoFormat::Uyvy;
    let make_ntsc = |index, no_signal| captured(ntsc, format, board_time(ntsc, index), no_signal);
    set(&mut src, "num-buffers", PropValue::Int(6));
    set(&mut src, "buffer-size", PropValue::Uint(8));

    let first = with_ancillary(
        make_ntsc(0, false),
        vec![afd_packet(274, false), afd_packet(12, true)],
    );
    let (caps, ()) = tokio::join!(src.intercept_caps(), send_from_board(&log, vec![first]));
    let caps = caps.expect("detected SD aspect");
    assert_eq!(caps, mode("ntsc-widescreen").caps(format));
    src.configure_pipeline(&caps).expect("configure");

    let mut invalid = afd_packet(12, false);
    invalid.user_data.pop();
    let events = vec![
        make_ntsc(1, false),
        with_ancillary(make_ntsc(2, true), vec![afd_packet(12, false)]),
        with_ancillary(make_ntsc(3, false), vec![invalid]),
        with_ancillary(make_ntsc(4, false), vec![afd_packet(12, false)]),
        InputEvent::FormatChanged {
            mode: Some(pal),
            format: Some(format),
        },
        captured(pal, format, board_time(ntsc, 5), false),
    ];
    let mut sink = CollectSink::default();
    let (captured_frames, ()) = tokio::join!(src.run(&mut sink), send_from_board(&log, events));
    assert_eq!(captured_frames, Ok(6));
    let [PipelinePacket::DataFrame(first), PipelinePacket::DataFrame(_), PipelinePacket::DataFrame(_), PipelinePacket::DataFrame(_), PipelinePacket::CapsChanged(standard), PipelinePacket::DataFrame(_), PipelinePacket::CapsChanged(changed), PipelinePacket::DataFrame(_), PipelinePacket::Eos] =
        sink.packets.as_slice()
    else {
        panic!("unexpected packet order");
    };
    assert!(first.meta.get::<AfdBarMeta>().is_none());
    let Caps::RawVideo { pixel_aspect, .. } = standard else {
        panic!("expected raw video");
    };
    assert_eq!(
        *pixel_aspect,
        PixelAspect::Exact(AspectRatio::new(10, 11).unwrap())
    );
    assert_eq!(*changed, pal.caps(format));
    assert_eq!(log.lock().unwrap().restarts, [(pal.name, format)]);
}

#[tokio::test]
async fn typed_vanc_metadata_keeps_field_and_caption_data_in_a_fixed_mode() {
    let (mut src, log) = src_on_mock(214406);
    let ntsc = mode("ntsc");
    let format = RawVideoFormat::Uyvy;
    set(&mut src, "mode", PropValue::Str(ntsc.name.into()));
    set(&mut src, "num-buffers", PropValue::Int(1));
    set(&mut src, "output-cc", PropValue::Bool(true));
    set(&mut src, "output-afd-bar", PropValue::Bool(true));
    let caps = src.intercept_caps().await.expect("fixed capture caps");
    assert_eq!(caps, ntsc.caps(format));
    src.configure_pipeline(&caps).expect("configure");

    let digital = CcTriple {
        cc_type: 3,
        b0: 0x21,
        b1: 0x33,
    };
    let line21 = CcTriple {
        cc_type: 1,
        b0: 0xc1,
        b1: 0xc2,
    };
    let packets = vec![
        afd_packet(12, true),
        afd_packet(274, true),
        AncPacket::from_bytes(
            CAPTION_DID,
            CEA708_CDP_SDID,
            &build_cdp(&[digital], CDP_FRAME_RATE_29_97, 0),
        ),
        AncPacket::from_bytes(CAPTION_DID, CEA608_SDID, &write_s334_1a(&[line21])),
    ];
    let first = with_ancillary(captured(ntsc, format, board_time(ntsc, 0), false), packets);
    let mut sink = CollectSink::default();
    let (captured_frames, ()) =
        tokio::join!(src.run(&mut sink), send_from_board(&log, vec![first]));
    assert_eq!(captured_frames, Ok(1));
    let [PipelinePacket::DataFrame(first), PipelinePacket::Eos] = sink.packets.as_slice() else {
        panic!("the fixed mode emits a frame and then EOS");
    };
    assert!(first.meta.get::<AncillaryMeta>().is_none());
    assert_eq!(
        first.meta.get::<CaptionMeta>().unwrap().triples,
        [CaptionTriple::from(digital), CaptionTriple::from(line21)],
    );
    let entries = &first.meta.get::<AfdBarMeta>().unwrap().entries;
    let expected = AfdBar {
        line_number: Some(12),
        active_format: Some(4),
        widescreen: Some(true),
        bars: Some(BarData::TopBottom {
            top_end: 100,
            bottom_start: 448,
        }),
    };
    assert_eq!(
        entries,
        &[
            expected,
            AfdBar {
                line_number: Some(274),
                ..expected
            }
        ]
    );
    assert!(log.lock().unwrap().restarts.is_empty());
}

#[tokio::test]
async fn a_set_mode_captures_frames_timed_by_the_board() {
    let (mut src, log) = src_on_mock(214401);
    let mode = mode("1080p25");
    set(&mut src, "mode", PropValue::Str(mode.name.into()));
    set(&mut src, "video-format", PropValue::Str("10bit-yuv".into()));
    set(&mut src, "connection", PropValue::Str("hdmi".into()));
    set(&mut src, "output-vanc", PropValue::Bool(true));
    set(&mut src, "num-buffers", PropValue::Int(3));

    let caps = src.intercept_caps().await.expect("caps");
    assert_eq!(caps, mode.caps(RawVideoFormat::V210));
    assert!(log.lock().unwrap().enabled.is_empty());
    src.configure_pipeline(&caps).expect("configure");

    let mut first = captured(mode, RawVideoFormat::V210, board_time(mode, 0), false);
    if let InputEvent::Frame(frame) = &mut first {
        frame.ancillary.push(caption_packet());
    }
    let events = Vec::from([
        first,
        captured(mode, RawVideoFormat::V210, board_time(mode, 1), false),
        captured(mode, RawVideoFormat::V210, board_time(mode, 2), false),
    ]);
    let mut sink = CollectSink::default();
    let (captured_frames, ()) = tokio::join!(src.run(&mut sink), send_from_board(&log, events));
    assert_eq!(captured_frames, Ok(3));

    let log = log.lock().unwrap();
    assert_eq!(log.connection, Some("hdmi"));
    assert_eq!(
        log.enabled,
        [EnabledInput {
            mode: mode.name,
            format: RawVideoFormat::V210,
            format_detection: false,
        }]
    );
    let period = mode.frame_duration_ns();
    assert_eq!(frame_pts(&sink.packets), [0, period, 2 * period]);
    let PipelinePacket::DataFrame(first) = &sink.packets[0] else {
        panic!("the first packet is a frame");
    };
    assert_eq!(
        first
            .meta
            .get::<AncillaryMeta>()
            .map(|meta| meta.packets.clone()),
        Some(Vec::from([caption_packet()]))
    );
    assert!(matches!(sink.packets.last(), Some(PipelinePacket::Eos)));
}

#[tokio::test]
async fn auto_mode_follows_the_input_and_changes_caps_mid_stream() {
    let (mut src, log) = src_on_mock(214402);
    set(&mut src, "num-buffers", PropValue::Int(2));
    let detected = mode("1080i5994");
    let stale = mode("ntsc");
    let events = Vec::from([
        InputEvent::FormatChanged {
            mode: Some(detected),
            format: Some(RawVideoFormat::V210),
        },
        captured(stale, RawVideoFormat::Uyvy, board_time(stale, 0), true),
        captured(
            detected,
            RawVideoFormat::V210,
            board_time(detected, 1),
            false,
        ),
    ]);
    let (caps, ()) = tokio::join!(src.intercept_caps(), send_from_board(&log, events));
    assert_eq!(caps, Ok(detected.caps(RawVideoFormat::V210)));
    assert_eq!(src.get_property("signal"), Some(PropValue::Bool(true)));
    {
        let log = log.lock().unwrap();
        assert_eq!(
            log.enabled,
            [EnabledInput {
                mode: stale.name,
                format: RawVideoFormat::Uyvy,
                format_detection: true,
            }]
        );
        assert_eq!(log.restarts, [(detected.name, RawVideoFormat::V210)]);
    }
    src.configure_pipeline(&detected.caps(RawVideoFormat::V210))
        .expect("configure");

    let next = mode("720p50");
    let events = Vec::from([
        InputEvent::FormatChanged {
            mode: Some(next),
            format: Some(RawVideoFormat::Uyvy),
        },
        captured(next, RawVideoFormat::Uyvy, board_time(detected, 2), false),
    ]);
    let mut sink = CollectSink::default();
    let (captured_frames, ()) = tokio::join!(src.run(&mut sink), send_from_board(&log, events));
    assert_eq!(captured_frames, Ok(2));
    let shapes: Vec<Option<_>> = sink
        .packets
        .iter()
        .map(|packet| match packet {
            PipelinePacket::CapsChanged(caps) => Some(caps.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        shapes,
        [None, Some(next.caps(RawVideoFormat::Uyvy)), None, None]
    );
}

#[tokio::test]
async fn frames_without_signal_are_dropped_when_asked() {
    let (mut src, log) = src_on_mock(214403);
    let mode = mode("1080p25");
    set(&mut src, "mode", PropValue::Str(mode.name.into()));
    set(&mut src, "drop-no-signal-frames", PropValue::Bool(true));
    set(&mut src, "num-buffers", PropValue::Int(2));
    let caps = src.intercept_caps().await.expect("caps");
    src.configure_pipeline(&caps).expect("configure");

    // the signal arrives five frames after the board started
    let signal_index = 5;
    let events = Vec::from([
        captured(mode, RawVideoFormat::Uyvy, board_time(mode, 0), true),
        captured(
            mode,
            RawVideoFormat::Uyvy,
            board_time(mode, signal_index),
            false,
        ),
        captured(
            mode,
            RawVideoFormat::Uyvy,
            board_time(mode, signal_index + 1),
            false,
        ),
    ]);
    let mut sink = CollectSink::default();
    let (captured_frames, ()) = tokio::join!(src.run(&mut sink), send_from_board(&log, events));
    assert_eq!(captured_frames, Ok(2));
    assert_eq!(frame_pts(&sink.packets), [0, mode.frame_duration_ns()]);
}

#[tokio::test]
async fn a_set_video_format_refuses_an_input_in_another_format() {
    let (mut src, log) = src_on_mock(214404);
    set(&mut src, "video-format", PropValue::Str("8bit-yuv".into()));
    let events = Vec::from([InputEvent::FormatChanged {
        mode: Some(mode("1080p25")),
        format: Some(RawVideoFormat::V210),
    }]);
    let (caps, ()) = tokio::join!(src.intercept_caps(), send_from_board(&log, events));
    assert_eq!(caps, Err(G2gError::CapsMismatch));
    assert!(log.lock().unwrap().restarts.is_empty());
}
