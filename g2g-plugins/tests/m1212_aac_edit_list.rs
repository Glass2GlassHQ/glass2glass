//! M1212: an AAC track's encoder priming survives an MP4 remux. The source's
//! `elst` says presentation starts 1024 samples into the media; `qtdemux` turns
//! that into a `Segment` starting past the priming, and `mp4mux` writes it back
//! as an edit list, so ffmpeg reads the remux back to the source's samples from
//! the first one (not 21 ms of priming late).

#![cfg(feature = "std")]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
        && Command::new("ffprobe").arg("-version").output().is_ok()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m1212-{}-{name}", std::process::id()))
}

/// A 1 s H.264 + AAC file from ffmpeg, whose AAC track carries ffmpeg's usual
/// 1024-sample priming as an edit list.
fn author(path: &PathBuf) {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args([
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=30:duration=1",
        ])
        .args([
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
        ])
        .args([
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2",
        ])
        .arg(path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the H.264 + AAC source");
}

/// ffprobe's `initial_padding` for the first audio stream: the priming an
/// `elst` trims, as ffmpeg reads it.
fn initial_padding(path: &PathBuf) -> u32 {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0"])
        .args(["-show_entries", "stream=initial_padding", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .expect("run ffprobe");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .expect("ffprobe reports initial_padding")
}

/// ffmpeg's decode of the first audio stream as interleaved f32 bytes.
fn decode_audio(path: &PathBuf) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-map", "0:a", "-f", "f32le", "-c:a", "pcm_f32le", "-"])
        .output()
        .expect("run ffmpeg");
    assert!(
        out.status.success() && out.stderr.is_empty(),
        "ffmpeg decoded {} cleanly: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

#[tokio::test]
async fn aac_priming_survives_a_remux() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("src.mp4");
    let out = temp_path("out.mp4");
    author(&src);
    let priming = initial_padding(&src);
    assert!(priming > 0, "ffmpeg's AAC source declares its priming");

    let line = format!(
        "filesrc location={} ! qtdemux name=d  \
         d.video_0 ! queue ! mux.  d.audio_0 ! queue ! mux.  \
         mp4mux name=mux ! filesink location={}",
        src.display(),
        out.display()
    );
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).expect("remux line parses");
    run_graph(graph, &ZeroClock, 4).await.expect("remux runs");

    assert_eq!(
        initial_padding(&out),
        priming,
        "the remux trims the source's priming"
    );
    // The presentation starts on the same sample: the source's decode is a
    // prefix of the remux's (a fragmented file keeps the last frame's tail).
    let from_src = decode_audio(&src);
    let from_out = decode_audio(&out);
    assert!(
        from_out.len() >= from_src.len(),
        "the remux decodes at least the source's {} bytes, got {}",
        from_src.len(),
        from_out.len()
    );
    assert_eq!(
        &from_out[..from_src.len()],
        &from_src[..],
        "the remux decodes to the source's samples from the first one"
    );

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
}
