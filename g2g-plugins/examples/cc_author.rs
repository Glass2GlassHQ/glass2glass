//! Author embedded CEA-608 closed captions into a video from a subtitle file.
//!
//!   cargo run -p g2g-plugins --example cc_author -- in.mp4 subs.srt out.ts
//!
//! Builds the turnkey authoring pipeline (M431-M433):
//!
//!   mp4src(in.mp4) -> h264parse -> ccinsert -> tsmux -> filesink(out.ts)
//!   subtitlesrc(subs.srt) -> subparse ----^ (cue pad)
//!
//! `SubtitleSrc` reads the `.srt` / `.vtt`, `SubParse` turns it into timed cues,
//! `CcInsert` encodes them to CEA-608 and writes a `GA94` SEI into each access
//! unit, and `TsMux` muxes the result. Play it back (captions render via the
//! M430 `playbin` caption path) with:
//!
//!   g2g-launch 'playbin uri=file:///abs/out.ts#closed-captions=cc1'
//!
//! IMPORTANT: the video source must carry timestamps (an MP4 / MKV container,
//! not a raw `.264` elementary stream). `CcInsert` paces the caption bytes against
//! the video frame PTS; an untimed source makes the merge deliver every cue after
//! the last frame, so the captions are dropped (the element logs a warning). End
//! in a muxer (`tsmux` / `mp4mux`), not a raw `.264` file: a raw-stream round trip
//! through an external remuxer can duplicate parameter sets and fail to re-play.

#[cfg(feature = "std")]
fn main() {
    use g2g_core::graph::Graph;
    use g2g_core::runtime::{run_graph, GraphNode};
    use g2g_core::PipelineClock;
    use g2g_plugins::ccinsert::CcInsert;
    use g2g_plugins::filesink::FileSink;
    use g2g_plugins::h264parse::H264Parse;
    use g2g_plugins::mp4src::Mp4Src;
    use g2g_plugins::subparse::SubParse;
    use g2g_plugins::subtitlesrc::SubtitleSrc;
    use g2g_plugins::tsmux::TsMux;

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: cc_author <video.mp4> <subtitles.srt|.vtt> <out.ts>");
        std::process::exit(2);
    }
    let (video, subs, out) = (&args[1], &args[2], &args[3]);

    // A trivial clock: file-to-file authoring is non-live, so no pacing is needed.
    struct Immediate;
    impl PipelineClock for Immediate {
        fn now_ns(&self) -> u64 {
            0
        }
    }

    let mut g: Graph<GraphNode> = Graph::new();
    // Video branch: timed H.264 -> reframe to access units -> CcInsert video pad.
    let vsrc = g.add_source(GraphNode::source(Mp4Src::new(video)));
    let parse = g.add_transform(GraphNode::element(H264Parse::reframing()));
    let cc = g.add_muxer(GraphNode::muxer(CcInsert::new()), 2);
    let tsmux = g.add_transform(GraphNode::element(TsMux::new()));
    let sink = g.add_sink(GraphNode::element(FileSink::new(out)));
    // Caption branch: subtitle file -> timed cues -> CcInsert cue pad.
    let ssrc = g.add_source(GraphNode::source(SubtitleSrc::from_location(subs)));
    let sp = g.add_transform(GraphNode::element(SubParse::new()));

    g.link(vsrc, parse).expect("link mp4src -> h264parse");
    g.link(parse, cc.input(0)).expect("link h264parse -> ccinsert.video");
    g.link(ssrc, sp).expect("link subtitlesrc -> subparse");
    g.link(sp, cc.input(1)).expect("link subparse -> ccinsert.cue");
    g.link(cc.output(), tsmux).expect("link ccinsert -> tsmux");
    g.link(tsmux, sink).expect("link tsmux -> filesink");

    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("runtime");
    match rt.block_on(run_graph(g, &Immediate, 4)) {
        Ok(stats) => {
            println!("authored {} frames -> {out}", stats.frames_consumed);
            println!("play it:  g2g-launch 'playbin uri=file://{out}#closed-captions=cc1'");
        }
        Err(e) => {
            eprintln!("authoring failed: {e:?}");
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "std"))]
fn main() {}
