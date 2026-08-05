//! M926: the VobSub `.idx` / `.sub` sidecar source. The fixture is the
//! hand-authored pair of the `vobsub_fixture` module, the same one ffmpeg's
//! `vobsub` demuxer reads in M899, so the bytes `VobSubSrc` indexes into are a
//! real program stream rather than a shape invented here.
//!
//! What is asserted: the emitted stream opens on the `.idx` text (the in-band
//! config `vobsubdec` needs), each cue lands at its indexed time with the
//! subpicture unit's own duration, the cues decode to the authored rectangles in
//! the authored palette colours, and the times match ffmpeg's own read of the
//! same pair.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;
use std::process::Command;

use g2g_core::frame::{Frame, PipelinePacket};
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, Caps, G2gError, OutputSink, PropValue, PropertySpec, PushOutcome,
    SubPictureFormat,
};
use g2g_plugins::registry::default_registry;
use g2g_plugins::vobsub::parse_idx;
use g2g_plugins::vobsubdec::VobSubDec;
use g2g_plugins::vobsubsrc::VobSubSrc;

mod vobsub_fixture;
use vobsub_fixture::{
    author_spanning_vobsub, author_vobsub, cues, have_ffmpeg, spanning_sample, CUE_DURATION_NS, H,
    PALETTE, SPANNING_CUE, W,
};

#[derive(Default)]
struct CaptureSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        self.packets.push(packet);
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m926-{}-{name}", std::process::id()))
}

fn vobsub_caps() -> Caps {
    Caps::SubPicture {
        format: SubPictureFormat::VobSub,
    }
}

/// Run the source over a sidecar pair, returning the data frames it emitted.
async fn read_sidecar(src: &mut VobSubSrc) -> Vec<Frame> {
    src.configure_pipeline(&vobsub_caps())
        .expect("vobsubsrc accepts its own caps");
    let mut sink = CaptureSink::default();
    let pushed = src.run(&mut sink).await.expect("read the sidecar pair");
    assert!(
        matches!(sink.packets.last(), Some(PipelinePacket::Eos)),
        "the sidecar ends with Eos"
    );
    let frames: Vec<Frame> = sink
        .packets
        .into_iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();
    assert_eq!(pushed as usize, frames.len(), "run reports what it pushed");
    frames
}

fn bytes(frame: &Frame) -> Vec<u8> {
    frame
        .domain
        .as_system_slice()
        .expect("system frame")
        .to_vec()
}

/// Bounding box of the non-transparent pixels of a decoded canvas, and how many
/// there are. `None` for a fully transparent one.
fn opaque_bbox(rgba: &[u8]) -> Option<(u32, u32, u32, u32, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (W, H, 0u32, 0u32);
    let mut count = 0usize;
    for y in 0..H {
        for x in 0..W {
            if rgba[((y * W + x) * 4 + 3) as usize] != 0 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
                count += 1;
            }
        }
    }
    (count > 0).then_some((x0, y0, x1, y1, count))
}

fn rgba_at(canvas: &[u8], x: u32, y: u32) -> [u8; 4] {
    let at = ((y * W + x) * 4) as usize;
    canvas[at..at + 4].try_into().expect("four channels")
}

/// The RGBA a palette entry paints opaque.
fn opaque(entry: u32) -> [u8; 4] {
    [(entry >> 16) as u8, (entry >> 8) as u8, entry as u8, 255]
}

#[test]
fn vobsubsrc_builds_from_a_launch_line() {
    let reg = default_registry();
    assert!(reg.element_names().contains(&"vobsubsrc"));
    let graph = g2g_core::runtime::parse_launch(
        &reg,
        "vobsubsrc location=movie.idx sub-location=other.sub language=fr ! vobsubdec ! fakesink",
    );
    assert!(graph.is_ok(), "launch line parses: {:?}", graph.err());
}

#[test]
fn properties_round_trip() {
    let declares = |specs: &[PropertySpec], name: &str| specs.iter().any(|s| s.name == name);
    let mut src = VobSubSrc::new("/x/movie.idx");
    for name in ["location", "sub-location", "language"] {
        assert!(declares(src.properties(), name), "{name} is declared");
    }
    // the derived `.sub` path is what the element reports until one is set
    assert_eq!(
        src.get_property("sub-location"),
        Some(PropValue::Str("/x/movie.sub".into()))
    );
    for (name, value) in [
        ("location", "/y/other.idx"),
        ("sub-location", "/y/elsewhere.sub"),
        ("language", "fr"),
    ] {
        src.set_property(name, PropValue::Str(value.into()))
            .unwrap();
        assert_eq!(src.get_property(name), Some(PropValue::Str(value.into())));
    }
}

#[tokio::test]
async fn the_sidecar_pair_becomes_the_idx_config_then_timed_cues() {
    let (idx, sub) = (temp_path("read.idx"), temp_path("read.sub"));
    author_vobsub(&idx, &sub);

    let mut src = VobSubSrc::new(&idx);
    let frames = read_sidecar(&mut src).await;
    let expected = cues();
    assert_eq!(
        frames.len(),
        1 + expected.len(),
        "the config, then the cues"
    );

    // The stream opens on the `.idx` text, which is what the decoder reads its
    // palette and geometry from.
    let config = parse_idx(&bytes(&frames[0])).expect("the first frame is .idx text");
    assert_eq!(config.size, Some((W, H)));
    assert_eq!(config.palette, Some(PALETTE));

    for (i, cue) in expected.iter().enumerate() {
        let frame = &frames[1 + i];
        let spu = bytes(frame);
        assert_eq!(
            frame.timing.pts_ns,
            (cue.pts_s * 1_000_000_000.0) as u64,
            "cue {i} carries its indexed timestamp"
        );
        assert_eq!(
            frame.timing.duration_ns, CUE_DURATION_NS,
            "cue {i} carries the unit's own hide time as its duration"
        );
        // The packet is the subpicture unit exactly: its own size field, with
        // neither the PES headers around it nor the padding after it.
        assert_eq!(
            u16::from_be_bytes([spu[0], spu[1]]) as usize,
            spu.len(),
            "cue {i} is delimited by its declared packet size"
        );
    }
    for p in [idx, sub] {
        let _ = std::fs::remove_file(p);
    }
}

/// The end-to-end case the element exists for: `vobsubsrc ! vobsubdec` renders
/// the sidecar's cues, palette and all.
#[tokio::test]
async fn the_cues_decode_to_the_authored_rectangles_and_colours() {
    let (idx, sub) = (temp_path("decode.idx"), temp_path("decode.sub"));
    author_vobsub(&idx, &sub);

    let mut src = VobSubSrc::new(&idx);
    let frames = read_sidecar(&mut src).await;

    let mut dec = VobSubDec::new();
    dec.configure_pipeline(&vobsub_caps())
        .expect("vobsubdec accepts a VobSub stream");
    let mut decoded = CaptureSink::default();
    for frame in frames {
        dec.process(PipelinePacket::DataFrame(frame), &mut decoded)
            .await
            .expect("decode");
    }
    let canvases: Vec<(u64, Vec<u8>)> = decoded
        .packets
        .into_iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some((f.timing.pts_ns, bytes(&f))),
            _ => None,
        })
        .collect();

    let expected = cues();
    // the opening empty canvas, then a shown / cleared pair per cue
    assert_eq!(canvases.len(), 1 + 2 * expected.len(), "canvas count");
    assert!(
        opaque_bbox(&canvases[0].1).is_none(),
        "the stream opens on an empty canvas"
    );

    for (i, cue) in expected.iter().enumerate() {
        let (pts, shown) = &canvases[1 + i * 2];
        let (cleared_pts, cleared) = &canvases[2 + i * 2];
        let start = (cue.pts_s * 1_000_000_000.0) as u64;
        assert_eq!(*pts, start, "cue {i} shows at its indexed time");
        assert_eq!(
            *cleared_pts,
            start + CUE_DURATION_NS,
            "cue {i} clears at its hide time"
        );
        assert!(
            opaque_bbox(cleared).is_none(),
            "cue {i}'s clear canvas is fully transparent"
        );
        assert_eq!(
            opaque_bbox(shown),
            Some((
                cue.x,
                cue.y,
                cue.x + cue.w - 1,
                cue.y + cue.h - 1,
                (cue.w * cue.h) as usize
            )),
            "cue {i} lands on its authored display rectangle"
        );
        // The fixture's cues are a one-pixel border of sample 3 around sample 1,
        // both through this cue's own colormap into the `.idx` palette: the
        // exact colours prove the config frame reached the decoder.
        assert_eq!(
            rgba_at(shown, cue.x + cue.w / 2, cue.y + cue.h / 2),
            opaque(PALETTE[cue.colormap[1] as usize]),
            "cue {i}'s interior takes its palette colour"
        );
        assert_eq!(
            rgba_at(shown, cue.x, cue.y),
            opaque(PALETTE[cue.colormap[3] as usize]),
            "cue {i}'s border takes its palette colour"
        );
    }
    for p in [idx, sub] {
        let _ = std::fs::remove_file(p);
    }
}

/// The reassembly path: a cue whose subpicture unit is longer than one DVD PES
/// packet arrives as several, each in its own pack and only the first stamped
/// with a PTS. Decoding it end to end is what proves the join: the bitmap is one
/// run-length code per pixel, so a dropped, doubled or reordered fragment shifts
/// every pixel after it.
#[tokio::test]
async fn a_cue_spanning_several_pes_packets_reassembles_and_decodes() {
    let (idx, sub) = (temp_path("span.idx"), temp_path("span.sub"));
    let packets = author_spanning_vobsub(&idx, &sub);
    assert!(
        packets >= 3,
        "the fixture cue must really span packets, got {packets}"
    );

    let mut src = VobSubSrc::new(&idx);
    let frames = read_sidecar(&mut src).await;
    assert_eq!(frames.len(), 2, "the config, then the one spanning cue");

    let cue = SPANNING_CUE;
    let spu = bytes(&frames[1]);
    assert_eq!(
        u16::from_be_bytes([spu[0], spu[1]]) as usize,
        spu.len(),
        "the rejoined unit is exactly its declared size"
    );
    assert!(
        spu.len() > 2048,
        "the unit outgrew a sector, {} bytes",
        spu.len()
    );
    assert_eq!(
        frames[1].timing.pts_ns,
        (cue.pts_s * 1_000_000_000.0) as u64,
        "the cue keeps the PTS of the packet that opened it"
    );
    assert_eq!(frames[1].timing.duration_ns, CUE_DURATION_NS);

    let mut dec = VobSubDec::new();
    dec.configure_pipeline(&vobsub_caps())
        .expect("vobsubdec accepts a VobSub stream");
    let mut decoded = CaptureSink::default();
    for frame in frames {
        dec.process(PipelinePacket::DataFrame(frame), &mut decoded)
            .await
            .expect("decode");
    }
    let canvases: Vec<Vec<u8>> = decoded
        .packets
        .into_iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(bytes(&f)),
            _ => None,
        })
        .collect();
    assert_eq!(canvases.len(), 3, "the empty canvas, the cue, its clear");

    let shown = &canvases[1];
    assert_eq!(
        opaque_bbox(shown),
        Some((
            cue.x,
            cue.y,
            cue.x + cue.w - 1,
            cue.y + cue.h - 1,
            (cue.w * cue.h) as usize
        )),
        "the rejoined cue covers its whole authored rectangle"
    );
    // Every pixel, not a sample of them: the reassembly is only right if the
    // run-length stream reads back in the order it was written.
    let wrong = (0..cue.h)
        .flat_map(|row| (0..cue.w).map(move |col| (row, col)))
        .find(|&(row, col)| {
            let sample = spanning_sample(col, row);
            rgba_at(shown, cue.x + col, cue.y + row)
                != opaque(PALETTE[cue.colormap[sample as usize] as usize])
        });
    assert_eq!(wrong, None, "first pixel the reassembled cue got wrong");
    assert!(
        opaque_bbox(&canvases[2]).is_none(),
        "the cue clears at its hide time"
    );

    for p in [idx, sub] {
        let _ = std::fs::remove_file(p);
    }
}

/// A second `.idx` over the same `.sub`, indexing one cue under each of two
/// languages, so `language=` selects between real entry lists.
fn two_language_idx(idx: &PathBuf, entries: &[String]) -> String {
    let text = std::fs::read_to_string(idx).expect("read the authored .idx");
    let head: Vec<&str> = text.lines().take_while(|l| !l.starts_with("id:")).collect();
    format!(
        "{}\nid: en, index: 0\n{}\nid: fr, index: 1\n{}\n",
        head.join("\n"),
        entries[0],
        entries[1]
    )
}

#[tokio::test]
async fn language_selects_between_the_indexed_streams() {
    let (idx, sub) = (temp_path("lang.idx"), temp_path("lang.sub"));
    author_vobsub(&idx, &sub);
    let text = std::fs::read_to_string(&idx).expect("read the authored .idx");
    let entries: Vec<String> = text
        .lines()
        .filter(|l| l.starts_with("timestamp:"))
        .map(String::from)
        .collect();
    let multi = temp_path("lang-multi.idx");
    std::fs::write(&multi, two_language_idx(&idx, &entries)).expect("write the two-language .idx");

    let expected = cues();
    for (language, cue) in [(None, &expected[0]), (Some("fr"), &expected[1])] {
        let mut src = VobSubSrc::new(&multi).with_sub_location(&sub);
        if let Some(language) = language {
            src = src.with_language(language);
        }
        let frames = read_sidecar(&mut src).await;
        assert_eq!(frames.len(), 2, "the config and the one indexed cue");
        assert_eq!(
            frames[1].timing.pts_ns,
            (cue.pts_s * 1_000_000_000.0) as u64,
            "language {language:?} emits its own stream's cue"
        );
    }
    for p in [idx, sub, multi] {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn a_corrupt_index_drops_only_the_cue_it_names() {
    let (idx, sub) = (temp_path("bad.idx"), temp_path("bad.sub"));
    author_vobsub(&idx, &sub);
    // Point the first cue's entry past the end of the `.sub`, and truncate the
    // file under the second so its packet no longer completes.
    let text = std::fs::read_to_string(&idx).expect("read the authored .idx");
    let mut lines: Vec<String> = text.lines().map(String::from).collect();
    let first = lines
        .iter()
        .position(|l| l.starts_with("timestamp:"))
        .expect("an indexed cue");
    lines[first] = "timestamp: 00:00:01:500, filepos: 0fffffff".into();
    std::fs::write(&idx, lines.join("\n")).expect("rewrite the .idx");

    let mut src = VobSubSrc::new(&idx);
    let frames = read_sidecar(&mut src).await;
    assert_eq!(
        frames.len(),
        2,
        "the config plus the one cue that is still readable"
    );

    // A `.sub` cut short mid-packet drops that cue too, rather than emitting a
    // half unit or failing the whole stream.
    let data = std::fs::read(&sub).expect("read the .sub");
    std::fs::write(&sub, &data[..64]).expect("truncate the .sub");
    let mut src = VobSubSrc::new(&idx);
    let frames = read_sidecar(&mut src).await;
    assert_eq!(frames.len(), 1, "only the config survives a truncated .sub");

    for p in [idx, sub] {
        let _ = std::fs::remove_file(p);
    }
}

/// Reference peer: ffmpeg's own `vobsub` demuxer over the same pair. Its
/// subtitle packet times are what g2g's index read must agree with.
#[tokio::test]
async fn the_cue_times_match_ffmpegs_read_of_the_same_pair() {
    if !have_ffmpeg() {
        eprintln!("skipping m926 ffmpeg cross-check: no ffmpeg on PATH");
        return;
    }
    let (idx, sub) = (temp_path("probe.idx"), temp_path("probe.sub"));
    author_vobsub(&idx, &sub);

    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "s:0"])
        .args(["-show_entries", "packet=pts_time", "-of", "csv=p=0"])
        .arg(&idx)
        .output()
        .expect("run ffprobe");
    assert!(out.status.success(), "ffprobe read the sidecar pair");
    let theirs: Vec<f64> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().trim_end_matches(',').parse().ok())
        .collect();

    let mut src = VobSubSrc::new(&idx);
    let frames = read_sidecar(&mut src).await;
    let ours: Vec<f64> = frames[1..]
        .iter()
        .map(|f| f.timing.pts_ns as f64 / 1e9)
        .collect();
    assert_eq!(theirs.len(), ours.len(), "same cue count as ffmpeg");
    for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "cue {i}: g2g reads {a}s where ffmpeg reads {b}s"
        );
    }

    for p in [idx, sub] {
        let _ = std::fs::remove_file(p);
    }
}
