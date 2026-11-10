// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M2173: AV1 metadata OBUs. SVT-AV1 codes the HDR10 mastering display and
//! content light level as metadata OBUs, `av1parse` attaches them as
//! `HdrStaticMeta`, and the values must equal what ffprobe reads from the same
//! stream. Skips when ffmpeg has no libsvtav1.
#![cfg(all(feature = "std", feature = "metadata"))]

use std::collections::BTreeMap;
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::meta::HdrStaticMeta;
use g2g_core::{AsyncElement, Caps, Dim, G2gError, OutputSink, PushOutcome, Rate, VideoCodec};
use g2g_plugins::av1parse::Av1Parse;

const SVT_HDR_PARAMS: &str = "mastering-display=G(0.17,0.797)B(0.131,0.046)R(0.708,0.292)WP(0.3127,0.329)L(1000,0.005):content-light=1200,300";
const FRAMES: usize = 10;

#[derive(Default)]
struct CaptureSink {
    frames: Vec<Frame>,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
            self.frames.push(frame);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn ffmpeg_has_libsvtav1() -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .is_ok_and(|output| String::from_utf8_lossy(&output.stdout).contains("libsvtav1"))
}

fn encode_hdr_av1(path: &std::path::Path) {
    let output = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=30",
        ])
        .args(["-frames:v", &FRAMES.to_string()])
        .args(["-pix_fmt", "yuv420p10le", "-c:v", "libsvtav1"])
        .args(["-svtav1-params", SVT_HDR_PARAMS, "-f", "obu"])
        .arg(path)
        .output()
        .expect("run ffmpeg");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ffprobe prints each side data field as `name=numerator/denominator` or `name=value`
fn ffprobe_side_data(path: &std::path::Path) -> BTreeMap<String, f64> {
    let output = Command::new("ffprobe")
        .args(["-v", "error", "-show_frames", "-read_intervals", "%+#1"])
        .arg(path)
        .output()
        .expect("run ffprobe");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let (name, value) = line.split_once('=')?;
            let number = match value.split_once('/') {
                Some((numerator, denominator)) => {
                    numerator.parse::<f64>().ok()? / denominator.parse::<f64>().ok()?
                }
                None => value.parse().ok()?,
            };
            Some((name.to_string(), number))
        })
        .collect()
}

#[tokio::test]
async fn svt_av1_hdr_metadata_matches_ffprobe() {
    let path = std::env::temp_dir().join(format!("g2g-m2173-{}.obu", std::process::id()));
    if !ffmpeg_has_libsvtav1() {
        eprintln!("skipping: ffmpeg has no libsvtav1");
        return;
    }
    encode_hdr_av1(&path);
    let reference = ffprobe_side_data(&path);
    let stream = std::fs::read(&path).expect("read the encode");
    std::fs::remove_file(&path).expect("remove the encode");

    let mut parse = Av1Parse::reframing();
    parse
        .configure_pipeline(&Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        })
        .expect("configure");
    let mut sink = CaptureSink::default();
    let chunk = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(stream.into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    parse
        .process(PipelinePacket::DataFrame(chunk), &mut sink)
        .await
        .expect("parse");
    parse
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");

    assert_eq!(sink.frames.len(), FRAMES);
    let field = |name: &str| reference[name] as f32;
    for (index, frame) in sink.frames.iter().enumerate() {
        let hdr = frame
            .meta
            .get::<HdrStaticMeta>()
            .unwrap_or_else(|| panic!("frame {index} carries HDR metadata"));
        let mastering = hdr.mastering.expect("mastering display");
        let [red, green, blue] = mastering.display_primaries;
        let coded = [
            (red.x, "red_x"),
            (red.y, "red_y"),
            (green.x, "green_x"),
            (green.y, "green_y"),
            (blue.x, "blue_x"),
            (blue.y, "blue_y"),
            (mastering.white_point.x, "white_point_x"),
            (mastering.white_point.y, "white_point_y"),
            (mastering.max_luminance, "max_luminance"),
            (mastering.min_luminance, "min_luminance"),
        ];
        for (value, name) in coded {
            assert_eq!(value, field(name), "{name}");
        }
        assert_eq!(
            hdr.max_content_light_level.map(f32::from),
            Some(field("max_content"))
        );
        assert_eq!(
            hdr.max_frame_average_light_level.map(f32::from),
            Some(field("max_average"))
        );
    }
}
