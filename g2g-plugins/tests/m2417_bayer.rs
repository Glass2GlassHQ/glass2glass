// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

// tools/bayer-fixtures.sh writes the expected outputs with GStreamer's elements
#![cfg(feature = "std")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

use std::path::PathBuf;

use g2g_core::frame::Frame;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{AsyncElement, Caps, CapsSet, FrameTiming, PipelineClock, PipelinePacket};
use g2g_plugins::bayer::{Bayer2Rgb, Rgb2Bayer};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[derive(Clone, Copy)]
struct Case {
    input: &'static str,
    input_caps: &'static str,
    output_caps: &'static str,
    expected: &'static str,
}

const DEMOSAIC_CASES: &[Case] = &[
    Case {
        input: "bayer_12x8_8bit.raw",
        input_caps: "video/x-bayer,format=bggr,width=12,height=8",
        output_caps: "video/x-raw,format=BGRA,width=12,height=8",
        expected: "bayer_12x8_bggr_to_bgra.raw",
    },
    Case {
        input: "bayer_12x8_8bit.raw",
        input_caps: "video/x-bayer,format=rggb,width=12,height=8",
        output_caps: "video/x-raw,format=BGRA,width=12,height=8",
        expected: "bayer_12x8_rggb_to_bgra.raw",
    },
    Case {
        input: "bayer_12x8_8bit.raw",
        input_caps: "video/x-bayer,format=grbg,width=12,height=8",
        output_caps: "video/x-raw,format=BGRA,width=12,height=8",
        expected: "bayer_12x8_grbg_to_bgra.raw",
    },
    Case {
        input: "bayer_12x8_8bit.raw",
        input_caps: "video/x-bayer,format=gbrg,width=12,height=8",
        output_caps: "video/x-raw,format=BGRA,width=12,height=8",
        expected: "bayer_12x8_gbrg_to_bgra.raw",
    },
    Case {
        input: "bayer_12x8_8bit.raw",
        input_caps: "video/x-bayer,format=rggb,width=12,height=8",
        output_caps: "video/x-raw,format=RGBA,width=12,height=8",
        expected: "bayer_12x8_rggb_to_rgba.raw",
    },
    Case {
        input: "bayer_12x8_8bit.raw",
        input_caps: "video/x-bayer,format=rggb,width=12,height=8",
        output_caps: "video/x-raw,format=ARGB,width=12,height=8",
        expected: "bayer_12x8_rggb_to_argb.raw",
    },
    Case {
        input: "bayer_12x8_8bit.raw",
        input_caps: "video/x-bayer,format=rggb,width=12,height=8",
        output_caps: "video/x-raw,format=BGRx,width=12,height=8",
        expected: "bayer_12x8_rggb_to_bgrx.raw",
    },
    Case {
        input: "bayer_12x8_grbg10le.raw",
        input_caps: "video/x-bayer,format=grbg10le,width=12,height=8",
        output_caps: "video/x-raw,format=BGRA,width=12,height=8",
        expected: "bayer_12x8_grbg10le_to_bgra.raw",
    },
    Case {
        input: "bayer_12x8_grbg12le.raw",
        input_caps: "video/x-bayer,format=grbg12le,width=12,height=8",
        output_caps: "video/x-raw,format=BGRA,width=12,height=8",
        expected: "bayer_12x8_grbg12le_to_bgra.raw",
    },
    Case {
        input: "bayer_12x8_grbg12be.raw",
        input_caps: "video/x-bayer,format=grbg12be,width=12,height=8",
        output_caps: "video/x-raw,format=BGRA,width=12,height=8",
        expected: "bayer_12x8_grbg12be_to_bgra.raw",
    },
    Case {
        input: "bayer_12x8_grbg16be.raw",
        input_caps: "video/x-bayer,format=grbg16be,width=12,height=8",
        output_caps: "video/x-raw,format=BGRA,width=12,height=8",
        expected: "bayer_12x8_grbg16be_to_bgra.raw",
    },
    Case {
        input: "bayer_12x8_grbg12le.raw",
        input_caps: "video/x-bayer,format=grbg12le,width=12,height=8",
        output_caps: "video/x-raw,format=RGBA64_LE,width=12,height=8",
        expected: "bayer_12x8_grbg12le_to_rgba64le.raw",
    },
    Case {
        input: "bayer_12x8_grbg16be.raw",
        input_caps: "video/x-bayer,format=grbg16be,width=12,height=8",
        output_caps: "video/x-raw,format=RGBA64_LE,width=12,height=8",
        expected: "bayer_12x8_grbg16be_to_rgba64le.raw",
    },
    Case {
        input: "bayer_12x3_8bit.raw",
        input_caps: "video/x-bayer,format=rggb,width=12,height=3",
        output_caps: "video/x-raw,format=BGRA,width=12,height=3",
        expected: "bayer_12x3_rggb_to_bgra.raw",
    },
    Case {
        input: "bayer_6x4_8bit.raw",
        input_caps: "video/x-bayer,format=bggr,width=6,height=4",
        output_caps: "video/x-raw,format=BGRA,width=6,height=4",
        expected: "bayer_6x4_bggr_to_bgra.raw",
    },
    Case {
        input: "bayer_12x5_8bit.raw",
        input_caps: "video/x-bayer,format=gbrg,width=12,height=5",
        output_caps: "video/x-raw,format=BGRA,width=12,height=5",
        expected: "bayer_12x5_gbrg_to_bgra.raw",
    },
];

const MOSAIC_CASES: &[Case] = &[
    Case {
        input: "bayer_12x8_argb.raw",
        input_caps: "video/x-raw,format=ARGB,width=12,height=8",
        output_caps: "video/x-bayer,format=bggr,width=12,height=8",
        expected: "bayer_12x8_argb_to_bggr.raw",
    },
    Case {
        input: "bayer_12x8_argb.raw",
        input_caps: "video/x-raw,format=ARGB,width=12,height=8",
        output_caps: "video/x-bayer,format=rggb,width=12,height=8",
        expected: "bayer_12x8_argb_to_rggb.raw",
    },
    Case {
        input: "bayer_12x8_argb.raw",
        input_caps: "video/x-raw,format=ARGB,width=12,height=8",
        output_caps: "video/x-bayer,format=grbg,width=12,height=8",
        expected: "bayer_12x8_argb_to_grbg.raw",
    },
    Case {
        input: "bayer_12x8_argb.raw",
        input_caps: "video/x-raw,format=ARGB,width=12,height=8",
        output_caps: "video/x-bayer,format=gbrg,width=12,height=8",
        expected: "bayer_12x8_argb_to_gbrg.raw",
    },
    Case {
        input: "bayer_12x8_argb.raw",
        input_caps: "video/x-raw,format=ARGB,width=12,height=8",
        output_caps: "video/x-bayer,format=rggb10le,width=12,height=8",
        expected: "bayer_12x8_argb_to_rggb10le.raw",
    },
    Case {
        input: "bayer_12x8_argb.raw",
        input_caps: "video/x-raw,format=ARGB,width=12,height=8",
        output_caps: "video/x-bayer,format=rggb16be,width=12,height=8",
        expected: "bayer_12x8_argb_to_rggb16be.raw",
    },
];

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn read_fixture(name: &str) -> Vec<u8> {
    std::fs::read(fixture(name)).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

fn caps(text: &str) -> Caps {
    let set = CapsSet::from_gst_string(text).unwrap_or_else(|| panic!("caps {text}"));
    let [caps] = set.alternatives() else {
        panic!("{text} is not fixed");
    };
    caps.clone()
}

async fn convert<E: AsyncElement>(mut element: E, case: &Case) -> Vec<u8> {
    element
        .configure_pipeline(&caps(case.input_caps))
        .unwrap_or_else(|e| panic!("{} input: {e:?}", case.expected));
    element
        .configure_output(&caps(case.output_caps))
        .unwrap_or_else(|e| panic!("{} output: {e:?}", case.expected));
    let frame = Frame::from_system_bytes(read_fixture(case.input), FrameTiming::default(), 0);
    let mut out = CollectSink::default();
    element
        .process(PipelinePacket::DataFrame(frame), &mut out)
        .await
        .unwrap_or_else(|e| panic!("{}: {e:?}", case.expected));
    let frames: Vec<Vec<u8>> = out
        .packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(
                frame
                    .domain
                    .as_system_slice()
                    .expect("system frame")
                    .to_vec(),
            ),
            _ => None,
        })
        .collect();
    let [frame] = &frames[..] else {
        panic!("{}: one frame, got {}", case.expected, frames.len());
    };
    frame.clone()
}

#[tokio::test]
async fn bayer2rgb_matches_the_reference_demosaic() {
    for case in DEMOSAIC_CASES {
        assert!(
            convert(Bayer2Rgb::new(), case).await == read_fixture(case.expected),
            "{}",
            case.expected
        );
    }
}

#[tokio::test]
async fn rgb2bayer_matches_the_reference_mosaic() {
    for case in MOSAIC_CASES {
        assert!(
            convert(Rgb2Bayer::new(), case).await == read_fixture(case.expected),
            "{}",
            case.expected
        );
    }
}

#[tokio::test]
async fn bayer2rgb_widens_8_bit_samples_to_rgba64_by_257() {
    let eight_bit = &DEMOSAIC_CASES[4];
    assert_eq!(eight_bit.expected, "bayer_12x8_rggb_to_rgba.raw");
    let wide = Case {
        output_caps: "video/x-raw,format=RGBA64_LE,width=12,height=8",
        ..*eight_bit
    };
    let expected: Vec<u8> = read_fixture(eight_bit.expected)
        .into_iter()
        .flat_map(|channel| (u16::from(channel) * 257).to_le_bytes())
        .collect();
    assert!(convert(Bayer2Rgb::new(), &wide).await == expected);
}

fn bayer_frame_with_one_value_per_colour(
    pattern: &str,
    width: usize,
    height: usize,
    rgb: [u8; 3],
) -> Vec<u8> {
    let stride = width.div_ceil(4) * 4;
    let colours: Vec<u8> = pattern
        .bytes()
        .map(|colour| match colour {
            b'r' => rgb[0],
            b'g' => rgb[1],
            _ => rgb[2],
        })
        .collect();
    let mut frame = vec![0; stride * height];
    for y in 0..height {
        for x in 0..width {
            frame[y * stride + x] = colours[(y % 2) * 2 + x % 2];
        }
    }
    frame
}

#[tokio::test]
async fn bayer2rgb_keeps_a_flat_colour_at_any_size() {
    let rgb = [200, 100, 30];
    for pattern in ["bggr", "rggb", "grbg", "gbrg"] {
        for (width, height) in [(2, 2), (3, 3), (5, 2), (7, 5), (2, 7), (9, 4)] {
            let mut element = Bayer2Rgb::new();
            element
                .configure_pipeline(&caps(&format!(
                    "video/x-bayer,format={pattern},width={width},height={height}"
                )))
                .unwrap_or_else(|e| panic!("{pattern} {width}x{height}: {e:?}"));
            element
                .configure_output(&caps(&format!(
                    "video/x-raw,format=RGBA,width={width},height={height}"
                )))
                .expect("RGBA output");
            let frame = Frame::from_system_bytes(
                bayer_frame_with_one_value_per_colour(pattern, width, height, rgb),
                FrameTiming::default(),
                0,
            );
            let mut out = CollectSink::default();
            element
                .process(PipelinePacket::DataFrame(frame), &mut out)
                .await
                .expect("demosaics");
            let pixels: Vec<u8> = out
                .packets
                .iter()
                .find_map(|packet| match packet {
                    PipelinePacket::DataFrame(frame) => Some(
                        frame
                            .domain
                            .as_system_slice()
                            .expect("system frame")
                            .to_vec(),
                    ),
                    _ => None,
                })
                .expect("a frame");
            let expected = [rgb[0], rgb[1], rgb[2], 255].repeat(width * height);
            assert!(pixels == expected, "{pattern} {width}x{height}");
        }
    }
}

#[tokio::test]
async fn bayer2rgb_rejects_a_single_row_or_column() {
    for too_small in [
        "video/x-bayer,format=rggb,width=1,height=8",
        "video/x-bayer,format=rggb,width=12,height=1",
    ] {
        assert!(
            Bayer2Rgb::new()
                .configure_pipeline(&caps(too_small))
                .is_err(),
            "{too_small}"
        );
    }
}

#[tokio::test]
async fn rgb2bayer_pads_rows_to_four_pixels() {
    let width = 6;
    let height = 2;
    let argb: Vec<u8> = (0..width * height)
        .flat_map(|pixel| [255, 10 + pixel as u8, 100 + pixel as u8, 200 + pixel as u8])
        .collect();
    let mut element = Rgb2Bayer::new();
    element
        .configure_pipeline(&caps("video/x-raw,format=ARGB,width=6,height=2"))
        .expect("ARGB input");
    element
        .configure_output(&caps("video/x-bayer,format=rggb,width=6,height=2"))
        .expect("Bayer output");
    let frame = Frame::from_system_bytes(argb, FrameTiming::default(), 0);
    let mut out = CollectSink::default();
    element
        .process(PipelinePacket::DataFrame(frame), &mut out)
        .await
        .expect("mosaics");
    let PipelinePacket::DataFrame(bayer) = out.packets.last().expect("a frame") else {
        panic!("the last packet is the frame");
    };
    let bayer = bayer
        .domain
        .as_system_slice()
        .expect("system frame")
        .to_vec();
    let red = |pixel: u8| 10 + pixel;
    let green = |pixel: u8| 100 + pixel;
    let blue = |pixel: u8| 200 + pixel;
    assert_eq!(
        bayer,
        [
            red(0),
            green(1),
            red(2),
            green(3),
            red(4),
            green(5),
            0,
            0,
            green(6),
            blue(7),
            green(8),
            blue(9),
            green(10),
            blue(11),
            0,
            0,
        ]
    );
}

#[tokio::test]
async fn a_bayer_dump_demosaics_through_a_launch_line() {
    let case = &DEMOSAIC_CASES[1];
    let output = std::env::temp_dir().join(format!("g2g-m2417-{}.bgra", std::process::id()));
    let line = format!(
        "filesrc location={} ! rawvideoparse format=rggb width=12 height=8 ! bayer2rgb ! video/x-raw,format=BGRA ! filesink location={}",
        fixture(case.input).display(),
        output.display()
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");
    let written = std::fs::read(&output).expect("the sink wrote a file");
    std::fs::remove_file(&output).ok();
    assert!(written == read_fixture(case.expected));
}
