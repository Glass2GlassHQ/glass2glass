// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "decklink"))]

use std::sync::Arc;

use g2g_core::{
    AspectRatio, AsyncElement, Caps, CapsConstraint, G2gError, Interlace, PixelAspect, PropValue,
    RawVideoFormat,
};

#[path = "common/decklink_board.rs"]
mod decklink_board;
use decklink_board::{manual_board_time, mode, sink_on_mock, video_caps};

fn tagged_caps(name: &str, ratio: (u32, u32)) -> Caps {
    let mode = mode(name);
    let scan = if mode.interlaced {
        Interlace::Interleaved
    } else {
        Interlace::Progressive
    };
    let mut caps = video_caps(RawVideoFormat::Uyvy, mode, scan);
    if let Caps::RawVideo { pixel_aspect, .. } = &mut caps {
        *pixel_aspect = PixelAspect::Exact(AspectRatio::new(ratio.0, ratio.1).unwrap());
    }
    caps
}

#[test]
fn auto_selects_the_widescreen_mode_for_all_five_sd_timings() {
    let timings = [
        ("ntsc", "ntsc-widescreen", (40, 33)),
        ("ntsc2398", "ntsc2398-widescreen", (40, 33)),
        ("pal", "pal-widescreen", (16, 11)),
        ("ntsc-p", "ntsc-p-widescreen", (40, 33)),
        ("pal-p", "pal-p-widescreen", (16, 11)),
    ];
    for (index, (name, widescreen, ratio)) in timings.into_iter().enumerate() {
        let (mut sink, log) =
            sink_on_mock(261800 + index as u32, manual_board_time(&Arc::default()));
        let caps = tagged_caps(name, ratio);
        sink.configure_pipeline(&caps)
            .expect("configure widescreen");
        assert_eq!(log.lock().unwrap().video_mode, Some(widescreen));
    }
}

#[test]
fn explicit_widescreen_mode_pins_the_ratio() {
    let (mut sink, log) = sink_on_mock(261810, manual_board_time(&Arc::default()));
    sink.set_property("mode", PropValue::Str("pal-widescreen".into()))
        .unwrap();
    let widescreen = tagged_caps("pal", (16, 11));
    let standard = tagged_caps("pal", (12, 11));
    let CapsConstraint::Accepts(advertised) = sink.caps_constraint_as_sink() else {
        panic!("expected sink caps");
    };
    assert!(advertised.accepts(&widescreen));
    assert!(!advertised.accepts(&standard));
    assert_eq!(
        sink.configure_pipeline(&standard).err(),
        Some(G2gError::CapsMismatch)
    );
    assert_eq!(log.lock().unwrap().video_mode, None);
    sink.configure_pipeline(&widescreen).unwrap();
    assert_eq!(log.lock().unwrap().video_mode, Some("pal-widescreen"));
}

#[test]
fn changing_only_the_aspect_keeps_the_sd_output_enabled() {
    let (mut sink, log) = sink_on_mock(261811, manual_board_time(&Arc::default()));
    sink.configure_pipeline(&tagged_caps("ntsc", (10, 11)))
        .unwrap();
    assert_eq!(log.lock().unwrap().video_mode, Some("ntsc"));
    sink.configure_pipeline(&tagged_caps("ntsc", (40, 33)))
        .unwrap();
    assert_eq!(log.lock().unwrap().video_mode, Some("ntsc"));
}
