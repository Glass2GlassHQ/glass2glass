// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use std::process::Command;

#[path = "common/ffmpeg_tool.rs"]
mod ffmpeg_tool;
#[path = "common/g2g_launch.rs"]
mod g2g_launch;

const NTSC_RATE: &str = "30000/1001";

#[test]
fn d10_audio_matches_the_pcm_ffmpeg_extracts() {
    if !ffmpeg_tool::have_ffmpeg() {
        eprintln!("skipping the D-10 reference comparison without ffmpeg");
        return;
    }
    let sound =
        "aevalsrc=0.25*sin(2*PI*440*t)|0.125*cos(2*PI*660*t):sample_rate=48000:duration=0.168";
    for (packing, stream, height, fps, vbv) in [
        ("pcm_s24le", "S24LE", "576", "25", "1200000"),
        ("pcm_s16le", "S16LE", "512", NTSC_RATE, "1001000"),
    ] {
        let picture = format!("color=c=blue:size=720x{height}:rate={fps}:duration=0.168");
        let input = format!("{}/m2505_{packing}.mxf", env!("CARGO_TARGET_TMPDIR"));
        let mux = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i", &picture])
            .args(["-f", "lavfi", "-i", sound, "-c:v", "mpeg2video"])
            .args(["-pix_fmt", "yuv422p", "-g", "1", "-bf", "0", "-flags", "+ildct+ilme", "-top", "1"])
            // D-10 requires one constant-size CBR picture in each edit unit.
            .args(["-b:v", "30M", "-minrate", "30M", "-maxrate", "30M"])
            .args(["-bufsize", vbv, "-rc_init_occupancy", vbv, "-c:a", packing, "-f", "mxf_d10", &input])
            .output()
            .expect("ffmpeg starts");
        assert!(
            mux.status.success(),
            "ffmpeg D-10 mux: {}",
            String::from_utf8_lossy(&mux.stderr)
        );
        let reference = Command::new("ffmpeg")
            .args([
                "-v", "error", "-i", &input, "-map", "0:a", "-c", "copy", "-f", "data", "-",
            ])
            .output()
            .expect("ffmpeg extracts the audio");
        assert!(
            reference.status.success(),
            "ffmpeg D-10 demux: {}",
            String::from_utf8_lossy(&reference.stderr)
        );
        assert!(reference.stdout.iter().any(|&byte| byte != 0));
        let output_name = format!("m2505_{packing}.pcm");
        let actual = g2g_launch::written_bytes(
            &format!("filesrc location={input} ! mxfdemux stream={stream}"),
            &output_name,
        );
        assert_eq!(actual.len(), reference.stdout.len(), "{stream} PCM length");
        assert!(actual == reference.stdout, "{stream} PCM samples differ");
        if fps == NTSC_RATE {
            let pipeline = format!(
                "filesrc location={input} ! mxfdemux name=d \
                 d.audio_0 ! checksumsink d.video_0 ! fakesink"
            );
            let whole = g2g_launch::checksum_lines(&g2g_launch::launch(&[], &pipeline));
            // 34 ms lies in the second edit unit, which starts after 1602 samples.
            let sought =
                g2g_launch::checksum_lines(&g2g_launch::launch(&["--seek", "0.034"], &pipeline));
            assert!(whole[1].starts_with("0:00:00.033375000 "));
            assert_eq!(
                sought,
                whole[1..],
                "indexed seek preserves audio PCM and timestamps"
            );
        }
        std::fs::remove_file(input).expect("remove the generated MXF");
        let output = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(output_name);
        std::fs::remove_file(output).expect("remove the generated PCM");
    }
}
