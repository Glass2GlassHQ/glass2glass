//! Plays a real MP4 with its sidecar subtitle file in a Wayland window, the Rust
//! form of:
//!
//!   mp4src ! avdec_h264(NV12) ! videoconvert(RGBA8) ! textoverlay(file)
//!         ! videoconvert(NV12) ! waylandsink
//!
//! The decoder emits NV12; `TextOverlay` paints in RGBA8, so a `videoconvert`
//! sits on each side of it, and `WaylandSink` consumes NV12. Cues are loaded from
//! the subtitle file (`.srt` / `.vtt`) and rendered by PTS.
//!
//!   cargo run -p g2g-plugins --features "ffmpeg wayland-sink" --example subtitle_player -- \
//!       /path/to/clip.mp4 /path/to/clip.vtt [WxH]
//!
//! The optional third argument scales the video (and so the window) to `WxH`,
//! e.g. `1280x720`, so a 1080p clip fits a laptop screen. A `videoscale` is
//! inserted right after the decoder (scaling the cheaper NV12 before the RGBA8
//! conversion). Omit it to play at the clip's native size.
//!
//! Heads-up: the built-in overlay font is an 8x8 all-caps ASCII bitmap, so a
//! Latin clip renders (uppercased, no diacritics) but CJK paints nothing (the
//! Unicode font backend is a separate, deferred piece).

#[cfg(all(feature = "ffmpeg", feature = "wayland-sink"))]
fn main() {
    use std::path::Path;

    use g2g_core::graph::Graph;
    use g2g_core::runtime::{run_graph_with_bus, GraphNodeRef};
    use g2g_core::{Bus, BusMessage, RawVideoFormat};
    use g2g_plugins::clock::WallClock;
    use g2g_plugins::ffmpegdec::{FfmpegH264Dec, OutputFormat};
    use g2g_plugins::mp4src::Mp4Src;
    use g2g_plugins::textoverlay::TextOverlay;
    use g2g_plugins::videoconvert::VideoConvert;
    use g2g_plugins::videoscale::VideoScale;
    use g2g_plugins::waylandsink::WaylandSink;

    let mut args = std::env::args().skip(1);
    let video = args
        .next()
        .expect("usage: subtitle_player <video.mp4> <subs.srt|.vtt> [WxH]");
    let subs = args.next().expect("need a subtitle file (.srt / .vtt)");
    // Optional WxH: scale the video (and window) to fit, e.g. 1280x720.
    let scale_to: Option<(u32, u32)> = args.next().and_then(|s| {
        let (w, h) = s.split_once('x')?;
        Some((w.parse().ok()?, h.parse().ok()?))
    });

    let sub_text = std::fs::read_to_string(&subs).expect("read subtitle file");
    let is_srt = Path::new(&subs)
        .extension()
        .map(|e| e.eq_ignore_ascii_case("srt"))
        .unwrap_or(false);
    let overlay = if is_srt {
        TextOverlay::from_srt(&sub_text)
    } else {
        TextOverlay::from_webvtt(&sub_text)
    };
    println!("loaded {} cues from {subs}", overlay.cue_count());

    // mp4src -> avdec_h264(NV12) -> [videoscale(NV12)] -> videoconvert(RGBA8)
    //        -> textoverlay -> videoconvert(NV12) -> waylandsink
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(Mp4Src::new(&video)));
    let dec = g.add_transform(GraphNodeRef::element(
        FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12),
    ));
    let to_rgba = g.add_transform(GraphNodeRef::element(VideoConvert::new(RawVideoFormat::Rgba8)));
    let ov = g.add_transform(GraphNodeRef::element(overlay));
    let to_nv12 = g.add_transform(GraphNodeRef::element(VideoConvert::new(RawVideoFormat::Nv12)));
    let sink =
        g.add_sink(GraphNodeRef::element(WaylandSink::new().with_title("g2g subtitle player")));
    g.link(src, dec).expect("link mp4src->dec");
    // Insert the scaler on the NV12 path when a target size is given.
    match scale_to {
        Some((w, h)) => {
            println!("scaling to {w}x{h}");
            let scale = g.add_transform(GraphNodeRef::element(VideoScale::new(w, h)));
            g.link(dec, scale).expect("link dec->scale");
            g.link(scale, to_rgba).expect("link scale->rgba");
        }
        None => {
            g.link(dec, to_rgba).expect("link dec->rgba");
        }
    }
    g.link(to_rgba, ov).expect("link rgba->overlay");
    g.link(ov, to_nv12).expect("link overlay->nv12");
    g.link(to_nv12, sink).expect("link nv12->sink");

    println!("playing {video} with subtitles...");
    let clock = WallClock::new();
    let (bus, handle) = Bus::new(64);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio rt");
    match rt.block_on(run_graph_with_bus(g, &clock, 4, &handle)) {
        Ok(stats) => println!("done: {} frames presented", stats.frames_consumed),
        Err(e) => eprintln!("pipeline error: {e:?}"),
    }
    // Surface a startup negotiation failure with its detail, if any.
    while let Some(msg) = bus.try_recv() {
        if let BusMessage::NegotiationFailed(f) = msg {
            eprintln!("negotiation failed: {f:?}");
        }
    }
}

#[cfg(not(all(feature = "ffmpeg", feature = "wayland-sink")))]
fn main() {
    eprintln!("build with --features \"ffmpeg wayland-sink\"");
}
