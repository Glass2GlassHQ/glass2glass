//! M924: EBU teletext subtitles in MPEG-TS, validated against ffmpeg's libzvbi
//! teletext decoder.
//!
//! ffmpeg has no teletext *encoder*, so the fixture is authored here: teletext
//! lines built per EN 300 706, wrapped as EN 300 472 data units, carried in a
//! private PES by the repo's own `TsMuxer` with a `teletext_descriptor` in the
//! PMT. ffmpeg's `libzvbi_teletextdec` reads that transport stream, which is what
//! pins the wire details a loopback cannot check: the data unit's reversed bit
//! order, the hamming 8/4 address, the odd-parity rows, and the page header's
//! page number. g2g's own `tsdemux stream=teletext ! teletextdec` decodes the
//! same bytes and must agree.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph,
    OutputSink, PipelineClock, PushOutcome, Rate, RawVideoFormat, TextFormat,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::mpegts::{TeletextService, TsMuxer, STREAM_TYPE_PRIVATE_PES};
use g2g_plugins::registry::default_registry;
use g2g_plugins::teletext::{encode_payload, DataUnit};
use g2g_plugins::teletextdec::TeletextDec;
use g2g_plugins::textoverlay::TextOverlayN;
use g2g_plugins::tsdemux::{TsDemux, TsStream};

/// The subtitle page the fixture's `teletext_descriptor` names, and therefore the
/// page both decoders follow without being told.
const PAGE: u16 = 888;
/// Magazine 8, page 0x88: page 888 as the descriptor and the page header spell it.
const MAGAZINE: u8 = 8;
const PAGE_BCD: u8 = 0x88;
/// `teletext_type` 0x02: an initial teletext subtitle page.
const TELETEXT_TYPE_SUBTITLE: u8 = 0x02;

/// The cues the fixture carries: when the page goes up, and its rows.
fn cues() -> Vec<(u64, &'static [&'static str])> {
    Vec::from([
        (1_000_000_000u64, &["FIRST SUBTITLE"][..]),
        (3_000_000_000, &["SECOND SUBTITLE", "ON TWO ROWS"][..]),
    ])
}

/// When the last cue is erased, and therefore the end of the fixture.
const END_NS: u64 = 5_000_000_000;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m924-{}-{name}", std::process::id()))
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

/// Whether this ffmpeg was built with libzvbi, without which it has no teletext
/// decoder at all.
fn have_teletext_decoder() -> bool {
    Command::new("ffmpeg")
        .arg("-decoders")
        .output()
        .ok()
        .is_some_and(|o| String::from_utf8_lossy(&o.stdout).contains("libzvbi_teletextdec"))
}

/// Author the fixture transport stream: one private-PES elementary stream marked
/// as teletext by its PMT descriptor, carrying a page header + rows per cue and a
/// bare header to erase the last one.
fn author_ts() -> Vec<u8> {
    let mut mux = TsMuxer::with_streams(&[STREAM_TYPE_PRIVATE_PES]);
    mux.set_stream_teletext(
        0,
        TeletextService {
            language: *b"eng",
            teletext_type: TELETEXT_TYPE_SUBTITLE,
            magazine: MAGAZINE,
            page: PAGE_BCD,
        },
    );
    // A receiver that joins mid-stream needs the tables again; a broadcast
    // repeats them, and it also exercises the demuxer's re-parse.
    mux.set_table_interval_90khz(90_000);

    let mut out = Vec::new();
    for (pts_ns, rows) in cues() {
        let mut units = Vec::from([DataUnit::page_header(PAGE, 0, true)]);
        for (i, row) in rows.iter().enumerate() {
            units.push(DataUnit::text_row(MAGAZINE, 20 + i as u8, row));
        }
        out.extend_from_slice(&mux.push_au(&encode_payload(&units), Some(ns_to_90k(pts_ns)), None));
    }
    // A page header with no rows behind it erases the subtitle on screen.
    out.extend_from_slice(&mux.push_au(
        &encode_payload(&[DataUnit::page_header(PAGE, 0, true)]),
        Some(ns_to_90k(END_NS)),
        None,
    ));
    out
}

fn ns_to_90k(ns: u64) -> u64 {
    ns * 90_000 / 1_000_000_000
}

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

fn data(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

/// Demux and decode the fixture with the real elements:
/// `tsdemux stream=teletext ! teletextdec`. Returns `(pts, duration, text)`.
async fn demux_and_decode(ts: &[u8]) -> Vec<(u64, u64, String)> {
    let mut demux = TsDemux::new().with_stream(TsStream::Teletext);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        })
        .expect("tsdemux accepts an MPEG-TS byte stream");
    let mut demuxed = CaptureSink::default();
    for chunk in ts.chunks(4096) {
        demux
            .process(data(chunk.to_vec()), &mut demuxed)
            .await
            .expect("demux");
    }
    demux
        .process(PipelinePacket::Eos, &mut demuxed)
        .await
        .expect("demux eos");

    // The decoder is told nothing: the page comes from the PMT descriptor the
    // demuxer forwards in band.
    let mut dec = TeletextDec::new();
    dec.configure_pipeline(&Caps::Text {
        format: TextFormat::Teletext,
    })
    .expect("teletextdec accepts a teletext stream");
    let mut decoded = CaptureSink::default();
    for packet in demuxed.packets {
        dec.process(packet, &mut decoded).await.expect("decode");
    }
    dec.process(PipelinePacket::Eos, &mut decoded)
        .await
        .expect("decode eos");

    decoded
        .packets
        .into_iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some((
                f.timing.pts_ns,
                f.timing.duration_ns,
                String::from_utf8(f.domain.as_system_slice().expect("system frame").to_vec())
                    .expect("cue text is UTF-8"),
            )),
            _ => None,
        })
        .collect()
}

/// ffmpeg's own decode of the same transport stream, as SubRip cue texts.
fn reference_decode(ts: &PathBuf, srt: &PathBuf) -> Option<Vec<String>> {
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-txt_page", &PAGE.to_string(), "-txt_format", "text"])
        .arg("-i")
        .arg(ts)
        .args(["-map", "0:s:0", "-c:s", "text", "-f", "srt"])
        .arg(srt)
        .output()
        .expect("run ffmpeg");
    if !out.status.success() {
        eprintln!(
            "ffmpeg teletext decode failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    let text = std::fs::read_to_string(srt).expect("read reference cues");
    // SubRip: blank-line separated blocks of index, time range, then the lines.
    Some(
        text.split("\n\n")
            .filter(|b| !b.trim().is_empty())
            .map(|block| {
                block
                    .lines()
                    .skip(2)
                    .map(str::trim)
                    .collect::<Vec<_>>()
                    .join("\n")
                    .trim()
                    .to_string()
            })
            .filter(|s| !s.is_empty())
            .collect(),
    )
}

#[test]
fn teletextdec_builds_from_a_launch_line() {
    let reg = default_registry();
    assert!(reg.make_element("teletextdec").is_some());
    assert!(reg.element_names().contains(&"teletextdec"));
    let graph = g2g_core::runtime::parse_launch(
        &reg,
        "filesrc location=x.ts bytestream-format=mpegts ! tsdemux stream=teletext ! teletextdec page=888 ! fakesink",
    );
    assert!(graph.is_ok(), "launch line parses: {:?}", graph.err());
}

#[tokio::test]
async fn the_pmt_descriptor_selects_the_page_and_cues_carry_their_display_span() {
    let cues = demux_and_decode(&author_ts()).await;
    let expected: Vec<(u64, u64, String)> = cues_expected();
    assert_eq!(cues, expected);
}

fn cues_expected() -> Vec<(u64, u64, String)> {
    let authored = cues();
    let mut out = Vec::new();
    for (i, (pts, rows)) in authored.iter().enumerate() {
        let end = authored.get(i + 1).map(|c| c.0).unwrap_or(END_NS);
        out.push((*pts, end - pts, rows.join("\n")));
    }
    out
}

const W: u32 = 64;
const H: u32 = 64;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Emits black RGBA8 frames at the given PTS values, then Eos.
struct BlackVideoSrc {
    pts: Vec<u64>,
}

impl SourceLoop for BlackVideoSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(W),
            height: Dim::Fixed(H),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
        }))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let n = self.pts.len() as u64;
            for &pts in &self.pts {
                let buf = [0u8, 0, 0, 255].repeat((W * H) as usize).into_boxed_slice();
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(buf)),
                    FrameTiming {
                        pts_ns: pts,
                        ..FrameTiming::default()
                    },
                    0,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(n)
        })
    }
}

/// Emits the teletext PES payloads a subtitle service sends: the page at 1s,
/// then the blank page that erases it at 3s.
struct TeletextPesSrc;

impl SourceLoop for TeletextPesSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(Caps::Text {
            format: TextFormat::Teletext,
        }))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let payloads = [
                (
                    1_000_000_000u64,
                    encode_payload(&[
                        DataUnit::page_header(PAGE, 0, true),
                        DataUnit::text_row(MAGAZINE, 20, "HELLO"),
                    ]),
                ),
                (
                    3_000_000_000,
                    encode_payload(&[DataUnit::page_header(PAGE, 0, true)]),
                ),
            ];
            for (pts_ns, payload) in payloads {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
                    FrameTiming {
                        pts_ns,
                        dts_ns: pts_ns,
                        ..FrameTiming::default()
                    },
                    0,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(2)
        })
    }
}

/// Records each received frame's `(pts, painted?)`.
struct RecSink {
    log: Arc<Mutex<Vec<(u64, bool)>>>,
}

impl AsyncElement for RecSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                if let Some(buf) = frame.domain.as_system_slice() {
                    let painted = (0..(W * H) as usize)
                        .any(|i| buf[i * 4] != 0 || buf[i * 4 + 1] != 0 || buf[i * 4 + 2] != 0);
                    self.log
                        .lock()
                        .unwrap()
                        .push((frame.timing.pts_ns, painted));
                }
            }
            Ok(())
        })
    }
}

/// The milestone's design claim: a decoded teletext page is an ordinary
/// `Caps::Text{Utf8}` cue stream, so `TextOverlayN` paints it onto video exactly
/// as it paints a `subparse`d SRT track (M403). Frames inside the page's display
/// window come out painted, frames outside it untouched.
#[tokio::test]
async fn a_decoded_page_paints_video_through_textoverlayn() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();

    let video = g.add_source(GraphNode::source(BlackVideoSrc {
        pts: vec![0, 1_500_000_000, 2_500_000_000, 4_000_000_000],
    }));
    let teletext = g.add_source(GraphNode::source(TeletextPesSrc));
    let dec = g.add_transform(GraphNode::element(TeletextDec::new().with_page(PAGE)));
    let mux = g.add_muxer(GraphNode::muxer(TextOverlayN::new()), 2);
    let sink = g.add_sink(GraphNode::element(RecSink { log: log.clone() }));

    g.link(video, mux.input(0)).unwrap();
    g.link(teletext, dec).unwrap();
    g.link(dec, mux.input(1)).unwrap();
    g.link(mux.output(), sink).unwrap();

    let stats = run_graph(g, &NullClock, 4)
        .await
        .expect("teletext overlay graph runs");
    assert_eq!(stats.frames_consumed, 4);

    let log = log.lock().unwrap();
    assert_eq!(log.len(), 4);
    assert!(
        log.iter().any(|&(_, painted)| painted),
        "the decoded page painted at least one frame"
    );
    for &(pts, painted) in log.iter() {
        let in_window = (1_000_000_000..3_000_000_000).contains(&pts);
        assert_eq!(
            painted, in_window,
            "frame at {pts} ns: painted={painted}, expected in_window={in_window}"
        );
    }
}

#[tokio::test]
async fn ffmpegs_teletext_decoder_reads_the_same_cues_from_the_authored_stream() {
    if !have_ffmpeg() {
        eprintln!("skipping m924 teletext oracle: no ffmpeg on PATH");
        return;
    }
    if !have_teletext_decoder() {
        eprintln!("skipping m924 teletext oracle: this ffmpeg has no libzvbi teletext decoder");
        return;
    }
    let (ts_path, srt) = (temp_path("fixture.ts"), temp_path("reference.srt"));
    let ts = author_ts();
    std::fs::write(&ts_path, &ts).expect("write fixture");

    let Some(reference) = reference_decode(&ts_path, &srt) else {
        panic!("ffmpeg could not decode the authored teletext stream");
    };
    let ours: Vec<String> = demux_and_decode(&ts)
        .await
        .into_iter()
        .map(|c| c.2)
        .collect();
    assert_eq!(
        reference, ours,
        "ffmpeg's libzvbi teletext decoder reads the same page text g2g does"
    );
    assert_eq!(ours.len(), cues().len(), "both cues were decoded");

    persist::record_evidence(
        "teletextdec",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("dvb_teletext")
            .detail(
                "ffmpeg's libzvbi teletext decoder reads the same subtitle page text from a g2g-authored MPEG-TS as tsdemux ! teletextdec",
            ),
    )
    .expect("record oracle evidence");

    for p in [ts_path, srt] {
        let _ = std::fs::remove_file(p);
    }
}
