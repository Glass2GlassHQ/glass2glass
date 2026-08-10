//! M927: the muxer write paths for a bitmap-subtitle (`Caps::SubPicture`) pad.
//! Matroska writes the pad as an `S_VOBSUB` or `S_DVBSUB` track carrying the
//! out-of-band blob each format needs as its `CodecPrivate`; MPEG-TS carries a
//! DVB subtitle stream on a private PES whose PMT entry declares it with a
//! `subtitling_descriptor`.
//!
//! The VobSub cues are the hand-authored `.idx` / `.sub` pair of the
//! `vobsub_fixture` module, read through `vobsubsrc`. The DVB display sets are
//! ffmpeg's own: ffmpeg transcodes that same pair to `dvbsub` in a transport
//! stream, and the segments it wrote are what g2g re-muxes, so the write path is
//! checked against a reference muxer's bytes rather than against itself.
#![cfg(feature = "std")]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::frame::{Frame, PipelinePacket};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, FrameTiming, G2gError, MemoryDomain, MultiInputElement,
    OutputSink, PropValue, PropertySpec, PushOutcome, SubPictureFormat,
};

use g2g_plugins::dvbsub::{page_id_blob, parse_page_ids, segment_span, PageIds};
use g2g_plugins::matroska::{MatroskaDemuxer, MkvCodec};
use g2g_plugins::mkvdemux::{MkvDemux, MkvStream};
use g2g_plugins::mkvmuxn::MkvMuxN;
use g2g_plugins::registry::default_registry;
use g2g_plugins::tsdemux::{TsDemux, TsStream};
use g2g_plugins::tsmux::TsMux;
use g2g_plugins::tsmuxn::TsMux as TsMuxN;
use g2g_plugins::vobsub::{idx_config_text, parse_idx};
use g2g_plugins::vobsubsrc::VobSubSrc;

mod vobsub_fixture;
use vobsub_fixture::{author_vobsub, cues, have_ffmpeg, CUE_DURATION_NS, H, PALETTE, W};

#[derive(Default)]
struct CaptureSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");

        self.packets.push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl CaptureSink {
    /// Every data frame's bytes and timing, in push order.
    fn frames(&self) -> Vec<(Vec<u8>, FrameTiming)> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some((
                    f.domain.as_system_slice().expect("system frame").to_vec(),
                    f.timing,
                )),
                _ => None,
            })
            .collect()
    }

    /// Every data frame's bytes concatenated: the muxed byte stream.
    fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (data, _) in self.frames() {
            out.extend_from_slice(&data);
        }
        out
    }
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m927-{}-{name}", std::process::id()))
}

fn caps(format: SubPictureFormat) -> Caps {
    Caps::SubPicture { format }
}

fn frame(data: Vec<u8>, timing: FrameTiming) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        timing,
        0,
    ))
}

fn at(pts_ns: u64, duration_ns: u64) -> FrameTiming {
    FrameTiming {
        pts_ns,
        dts_ns: pts_ns,
        duration_ns,
        ..FrameTiming::default()
    }
}

/// Mux one bitmap subtitle stream into a Matroska byte stream.
async fn mux_mkv(format: SubPictureFormat, stream: &[(Vec<u8>, FrameTiming)]) -> Vec<u8> {
    let mut mux = MkvMuxN::new(1);
    mux.configure_pipeline(0, &caps(format))
        .expect("matroskamux accepts a subpicture pad");
    let mut sink = CaptureSink::default();
    for (data, timing) in stream {
        mux.process(0, frame(data.clone(), *timing), &mut sink)
            .await
            .expect("mux a cue");
    }
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux EOS");
    sink.bytes()
}

/// Mux one DVB subtitle stream into an MPEG-TS byte stream.
async fn mux_ts(stream: &[(Vec<u8>, FrameTiming)]) -> Vec<u8> {
    let mut mux = TsMux::new();
    mux.configure_pipeline(&caps(SubPictureFormat::DvbSub))
        .expect("mpegtsmux accepts a DVB subtitle pad");
    let mut sink = CaptureSink::default();
    for (data, timing) in stream {
        mux.process(frame(data.clone(), *timing), &mut sink)
            .await
            .expect("mux a display set");
    }
    sink.bytes()
}

/// Run a demuxer element over a whole byte stream, returning what it emitted.
async fn demux<E: AsyncElement>(
    mut el: E,
    input: Caps,
    bytes: Vec<u8>,
) -> Vec<(Vec<u8>, FrameTiming)> {
    el.configure_pipeline(&input).expect("demuxer configures");
    let mut sink = CaptureSink::default();
    el.process(frame(bytes, FrameTiming::default()), &mut sink)
        .await
        .expect("demux");
    el.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("demux EOS");
    sink.frames()
}

/// The VobSub sidecar as a pad stream: the `.idx` config frame then the cues,
/// exactly what `vobsubsrc` puts on a `Caps::SubPicture{VobSub}` pad.
async fn vobsub_stream(idx: &PathBuf) -> Vec<(Vec<u8>, FrameTiming)> {
    let mut src = VobSubSrc::new(idx);
    src.configure_pipeline(&caps(SubPictureFormat::VobSub))
        .expect("vobsubsrc configures");
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.expect("read the sidecar pair");
    sink.frames()
}

/// The `CodecPrivate` of the first track of a muxed Matroska stream.
fn mkv_codec_private(bytes: &[u8]) -> Vec<u8> {
    let mut d = MatroskaDemuxer::new();
    d.push_data(bytes);
    d.codec_private(1)
        .expect("track 1 has a CodecPrivate")
        .to_vec()
}

/// The one DVB `subtitling_descriptor` in a transport stream, as ffprobe would
/// read it: the language, the subtitling type, and the pages it declares.
/// Scanning for the tag is enough here, the streams under test carry one.
fn subtitling_descriptor(ts: &[u8]) -> Option<(String, u8, PageIds)> {
    ts.windows(10)
        .find(|w| w[0] == 0x59 && w[1] == 8 && w[2..5].iter().all(u8::is_ascii_alphabetic))
        .map(|w| {
            (
                String::from_utf8_lossy(&w[2..5]).into_owned(),
                w[5],
                PageIds {
                    composition: u16::from_be_bytes([w[6], w[7]]),
                    ancillary: u16::from_be_bytes([w[8], w[9]]),
                },
            )
        })
}

fn mkv_codec(bytes: &[u8]) -> MkvCodec {
    let mut d = MatroskaDemuxer::new();
    d.push_data(bytes);
    d.tracks().first().expect("one track").codec
}

// ---- Matroska ----

/// The `.idx` a VobSub track carries as its `CodecPrivate` is the size and
/// palette lines, byte for byte the text ffmpeg writes, and the cues come back
/// out of our own demuxer at their authored times.
#[tokio::test]
async fn a_vobsub_pad_writes_an_s_vobsub_track_with_the_idx_as_codec_private() {
    let (idx, sub) = (temp_path("mkv.idx"), temp_path("mkv.sub"));
    author_vobsub(&idx, &sub);
    let stream = vobsub_stream(&idx).await;
    assert_eq!(stream.len(), 1 + cues().len(), "the config, then the cues");

    let muxed = mux_mkv(SubPictureFormat::VobSub, &stream).await;
    assert!(
        muxed.windows(8).any(|w| w == b"S_VOBSUB"),
        "the track declares the S_VOBSUB CodecID"
    );
    assert_eq!(mkv_codec(&muxed), MkvCodec::VobSub);

    // The `.idx` cue index is a sidecar's file offset table, so what the track
    // carries is the configuration alone, in ffmpeg's exact wording.
    let palette = PALETTE
        .iter()
        .map(|c| format!("{c:06x}"))
        .collect::<Vec<_>>()
        .join(", ");
    assert_eq!(
        String::from_utf8(mkv_codec_private(&muxed)).expect("the .idx is text"),
        format!("size: {W}x{H}\npalette: {palette}\n"),
    );

    // Round trip: our own demuxer hands the config back in band, then the cues
    // at their authored times with their own durations.
    let back = demux(
        MkvDemux::new().with_stream(MkvStream::VobSub),
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        },
        muxed,
    )
    .await;
    assert_eq!(back.len(), stream.len(), "config plus every cue survives");
    let config = parse_idx(&back[0].0).expect("the stream opens on the .idx text");
    assert_eq!(config.size, Some((W, H)));
    assert_eq!(config.palette, Some(PALETTE));
    for (i, cue) in cues().iter().enumerate() {
        let (data, timing) = &back[1 + i];
        assert_eq!(
            *data,
            stream[1 + i].0,
            "cue {i} is the same subpicture unit"
        );
        assert_eq!(
            timing.pts_ns,
            (cue.pts_s * 1_000_000_000.0) as u64,
            "cue {i} keeps its time"
        );
        assert_eq!(
            timing.duration_ns, CUE_DURATION_NS,
            "cue {i} keeps its display window"
        );
    }

    for p in [idx, sub] {
        let _ = std::fs::remove_file(p);
    }
}

/// A hand-built DVB display set: a page composition listing no region, the way a
/// stream ends a cue, behind the segment framing EN 300 743 defines.
fn dvb_display_set(page_id: u16) -> Vec<u8> {
    let mut out = Vec::from([0x0Fu8, 0x10]);
    out.extend_from_slice(&page_id.to_be_bytes());
    // page_time_out 30 s, page_version 0 / page_state 2 (mode change), no region
    out.extend_from_slice(&2u16.to_be_bytes());
    out.extend_from_slice(&[30, 0x80]);
    // end of display set
    out.extend_from_slice(&[0x0F, 0x80]);
    out.extend_from_slice(&page_id.to_be_bytes());
    out.extend_from_slice(&0u16.to_be_bytes());
    out
}

/// The DVB pad's page-id blob becomes the `S_DVBSUB` `CodecPrivate`, and the
/// blocks are the bare segments (no PES data-field header).
#[tokio::test]
async fn a_dvbsub_pad_writes_an_s_dvbsub_track_with_the_page_ids_as_codec_private() {
    let ids = PageIds {
        composition: 7,
        ancillary: 9,
    };
    let blob = page_id_blob(ids, 0x10);
    let set = dvb_display_set(7);
    let stream = vec![
        (blob.to_vec(), FrameTiming::default()),
        (set.clone(), at(1_500_000_000, 0)),
    ];

    let muxed = mux_mkv(SubPictureFormat::DvbSub, &stream).await;
    assert!(
        muxed.windows(8).any(|w| w == b"S_DVBSUB"),
        "the track declares the S_DVBSUB CodecID"
    );
    assert_eq!(mkv_codec(&muxed), MkvCodec::DvbSub);
    assert_eq!(
        mkv_codec_private(&muxed),
        blob.to_vec(),
        "the CodecPrivate is the five-byte page-id blob"
    );

    let back = demux(
        MkvDemux::new().with_stream(MkvStream::DvbSub),
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        },
        muxed,
    )
    .await;
    assert_eq!(back.len(), 2, "the page ids, then the display set");
    assert_eq!(
        parse_page_ids(&back[0].0),
        Some(ids),
        "the demuxer hands the page ids back in band"
    );
    assert_eq!(back[1].0, set, "the display set is the bare segments");
    assert_eq!(back[1].1.pts_ns, 1_500_000_000);
}

/// A display set that reached the muxer in its transport-stream carriage (the
/// data_identifier header and end marker around it) is unwrapped for Matroska,
/// which carries the bare segments.
#[tokio::test]
async fn a_matroska_block_drops_the_transport_stream_data_field_header() {
    let set = dvb_display_set(1);
    let mut wrapped = Vec::from([0x20u8, 0x00]);
    wrapped.extend_from_slice(&set);
    wrapped.push(0xFF);
    let stream = vec![
        (
            page_id_blob(
                PageIds {
                    composition: 1,
                    ancillary: 1,
                },
                0x10,
            )
            .to_vec(),
            FrameTiming::default(),
        ),
        (wrapped, at(0, 0)),
    ];
    let muxed = mux_mkv(SubPictureFormat::DvbSub, &stream).await;
    let mut d = MatroskaDemuxer::new();
    d.push_data(&muxed);
    let blocks = d.take_frames();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].data, set, "the block holds the segments alone");
}

// ---- MPEG-TS ----

/// The PMT entry of a DVB subtitle stream, and the PES payload framing: the
/// descriptor names the pages the config blob declared, and each display set
/// goes out in its data field.
#[tokio::test]
async fn a_dvbsub_pad_writes_a_private_pes_with_a_subtitling_descriptor() {
    let ids = PageIds {
        composition: 3,
        ancillary: 4,
    };
    let set = dvb_display_set(3);
    let stream = vec![
        (page_id_blob(ids, 0x10).to_vec(), FrameTiming::default()),
        (set.clone(), at(2_000_000_000, 0)),
    ];
    let muxed = mux_ts(&stream).await;

    assert_eq!(
        subtitling_descriptor(&muxed),
        Some((String::from("und"), 0x10, ids)),
        "the PMT declares the stream with a subtitling_descriptor"
    );

    let back = demux(
        TsDemux::new().with_stream(TsStream::DvbSub),
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
        muxed,
    )
    .await;
    assert_eq!(back.len(), 2, "the page ids, then the display set");
    assert_eq!(
        parse_page_ids(&back[0].0),
        Some(ids),
        "the demuxer rebuilds the page ids from the descriptor"
    );
    // The PES payload is the data field: the data_identifier and subtitle stream
    // id ahead of the segments, the end marker behind.
    assert_eq!(&back[1].0[..2], &[0x20, 0x00]);
    assert_eq!(back[1].0.last(), Some(&0xFF));
    assert_eq!(segment_span(&back[1].0), set, "the segments are unchanged");
    assert_eq!(back[1].1.pts_ns, 2_000_000_000);
}

/// The fan-in muxer carries the same subtitle pad next to video, and the
/// stream's language reaches the descriptor.
#[tokio::test]
async fn the_fan_in_ts_muxer_declares_a_subtitle_pad_in_its_language() {
    use g2g_core::{Dim, Rate, Tag, TagList, VideoCodec};

    let mut tags = TagList::new();
    tags.push(Tag::Language("deu".into()));
    let mut mux = TsMuxN::new(2).with_track_tags(1, tags);
    mux.configure_pipeline(
        0,
        &Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Any,
        },
    )
    .unwrap();
    mux.configure_pipeline(1, &caps(SubPictureFormat::DvbSub))
        .unwrap();

    let ids = PageIds {
        composition: 1,
        ancillary: 1,
    };
    let mut sink = CaptureSink::default();
    mux.process(
        1,
        frame(page_id_blob(ids, 0x10).to_vec(), FrameTiming::default()),
        &mut sink,
    )
    .await
    .unwrap();
    let idr = vec![0u8, 0, 0, 1, 0x65, 0xAA];
    mux.process(0, frame(idr, at(0, 0)), &mut sink)
        .await
        .unwrap();
    mux.process(1, frame(dvb_display_set(1), at(40_000_000, 0)), &mut sink)
        .await
        .unwrap();
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();

    assert_eq!(
        subtitling_descriptor(&sink.bytes()),
        Some((String::from("deu"), 0x10, ids)),
        "the subtitle stream's PMT entry names its language and pages"
    );
}

// ---- properties ----

#[test]
fn the_page_id_property_round_trips_on_every_muxer() {
    let declares = |specs: &[PropertySpec]| specs.iter().any(|s| s.name == "dvbsub-page-id");

    let mut mkv = MkvMuxN::new(1);
    assert!(declares(MultiInputElement::properties(&mkv)));
    MultiInputElement::set_property(&mut mkv, "dvbsub-page-id", PropValue::Uint(42)).unwrap();
    assert_eq!(
        MultiInputElement::get_property(&mkv, "dvbsub-page-id"),
        Some(PropValue::Uint(42))
    );

    let mut ts = TsMux::new();
    assert!(declares(AsyncElement::properties(&ts)));
    AsyncElement::set_property(&mut ts, "dvbsub-page-id", PropValue::Uint(42)).unwrap();
    assert_eq!(
        AsyncElement::get_property(&ts, "dvbsub-page-id"),
        Some(PropValue::Uint(42))
    );

    let mut tsn = TsMuxN::new(1);
    assert!(declares(MultiInputElement::properties(&tsn)));
    MultiInputElement::set_property(&mut tsn, "dvbsub-page-id", PropValue::Uint(42)).unwrap();
    assert_eq!(
        MultiInputElement::get_property(&tsn, "dvbsub-page-id"),
        Some(PropValue::Uint(42))
    );

    // A page id past the 16-bit field is refused rather than truncated.
    assert!(
        AsyncElement::set_property(&mut ts, "dvbsub-page-id", PropValue::Uint(70_000)).is_err()
    );
}

/// The property is what a stream carrying no page-id config is declared on.
#[tokio::test]
async fn the_page_id_property_declares_a_stream_that_sends_no_config() {
    let mut mux = TsMux::new();
    AsyncElement::set_property(&mut mux, "dvbsub-page-id", PropValue::Uint(5)).unwrap();
    mux.configure_pipeline(&caps(SubPictureFormat::DvbSub))
        .unwrap();
    let mut sink = CaptureSink::default();
    mux.process(frame(dvb_display_set(5), at(0, 0)), &mut sink)
        .await
        .unwrap();
    assert_eq!(
        subtitling_descriptor(&sink.bytes()),
        Some((
            String::from("und"),
            0x10,
            PageIds {
                composition: 5,
                ancillary: 5
            }
        )),
        "the property's page reaches the descriptor"
    );

    let mut mkv = MkvMuxN::new(1);
    MultiInputElement::set_property(&mut mkv, "dvbsub-page-id", PropValue::Uint(5)).unwrap();
    mkv.configure_pipeline(0, &caps(SubPictureFormat::DvbSub))
        .unwrap();
    let mut sink = CaptureSink::default();
    mkv.process(0, frame(dvb_display_set(5), at(0, 0)), &mut sink)
        .await
        .unwrap();
    mkv.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    assert_eq!(
        mkv_codec_private(&sink.bytes()),
        page_id_blob(
            PageIds {
                composition: 5,
                ancillary: 5
            },
            0x10
        )
        .to_vec()
    );
}

#[test]
fn the_muxers_accept_a_subpicture_pad_from_a_launch_line() {
    let reg = default_registry();
    let graph = g2g_core::runtime::parse_launch(
        &reg,
        "filesrc location=movie.h264 ! h264parse ! m.   vobsubsrc location=movie.idx ! m.   matroskamux name=m ! fakesink",
    );
    assert!(graph.is_ok(), "launch line parses: {:?}", graph.err());
}

// ---- reference peer: ffmpeg ----

fn ffmpeg(args: &[&str]) -> String {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "warning", "-y"])
        .args(args)
        .output()
        .expect("run ffmpeg");
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "ffmpeg {args:?} failed: {err}");
    err
}

fn ffprobe(args: &[&str]) -> String {
    let out = Command::new("ffprobe")
        .args(["-hide_banner", "-v", "error"])
        .args(args)
        .output()
        .expect("run ffprobe");
    assert!(
        out.status.success(),
        "ffprobe {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// ffprobe reads the muxed VobSub track as a DVD subtitle stream, and ffmpeg
/// decodes the cues out of it: a bitmap-to-bitmap transcode back to a `.idx` /
/// `.sub` pair recovers the same cue times, which it can only do by reading the
/// blocks and the `CodecPrivate` this muxer wrote.
#[tokio::test]
async fn ffmpeg_reads_the_muxed_vobsub_track() {
    if !have_ffmpeg() {
        eprintln!("skipping m927 ffmpeg cross-check: no ffmpeg on PATH");
        return;
    }
    let (idx, sub) = (temp_path("ff.idx"), temp_path("ff.sub"));
    author_vobsub(&idx, &sub);
    let stream = vobsub_stream(&idx).await;
    let out = temp_path("vobsub.mkv");
    std::fs::write(&out, mux_mkv(SubPictureFormat::VobSub, &stream).await).expect("write the mkv");

    let probe = ffprobe(&[
        "-select_streams",
        "s:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "csv=p=0",
        out.to_str().unwrap(),
    ]);
    assert_eq!(probe.trim(), "dvd_subtitle", "ffprobe reads a VobSub track");

    let times = ffprobe(&[
        "-select_streams",
        "s:0",
        "-show_entries",
        "packet=pts_time",
        "-of",
        "csv=p=0",
        out.to_str().unwrap(),
    ]);
    let theirs: Vec<f64> = times
        .lines()
        .filter_map(|l| l.trim().trim_end_matches(',').parse().ok())
        .collect();
    let expected: Vec<f64> = cues().iter().map(|c| c.pts_s).collect();
    assert_eq!(theirs.len(), expected.len(), "every cue is a packet");
    for (i, (a, b)) in theirs.iter().zip(&expected).enumerate() {
        assert!((a - b).abs() < 1e-3, "cue {i}: ffmpeg reads {a}s, not {b}s");
    }

    // Decode: transcoding the track to DVB subtitles is bitmap to bitmap, so it
    // only succeeds if ffmpeg parsed the subpicture units out of our blocks.
    let decoded = temp_path("vobsub-decoded.ts");
    ffmpeg(&[
        "-i",
        out.to_str().unwrap(),
        "-c:s",
        "dvbsub",
        "-f",
        "mpegts",
        decoded.to_str().unwrap(),
    ]);
    let count = ffprobe(&[
        "-select_streams",
        "s:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "csv=p=0",
        decoded.to_str().unwrap(),
    ]);
    let packets: u32 = count
        .lines()
        .find_map(|l| l.trim().parse().ok())
        .unwrap_or(0);
    assert!(
        packets >= cues().len() as u32,
        "ffmpeg decoded every cue (it re-encoded {packets} packets)"
    );

    for p in [idx, sub, out, decoded] {
        let _ = std::fs::remove_file(p);
    }
}

/// The reference peer authors the DVB subtitles: ffmpeg transcodes the fixture
/// to `dvbsub` in a transport stream, g2g reads the display sets back out and
/// re-muxes them, and the result is compared with what ffmpeg wrote (the PMT
/// descriptor and every PES payload) and handed back to ffmpeg to decode.
#[tokio::test]
async fn the_dvb_write_paths_match_ffmpegs_own_transport_stream() {
    if !have_ffmpeg() {
        eprintln!("skipping m927 ffmpeg cross-check: no ffmpeg on PATH");
        return;
    }
    let (idx, sub) = (temp_path("dvb.idx"), temp_path("dvb.sub"));
    author_vobsub(&idx, &sub);
    let reference = temp_path("dvb-ref.ts");
    ffmpeg(&[
        "-i",
        idx.to_str().unwrap(),
        "-c:s",
        "dvbsub",
        "-f",
        "mpegts",
        reference.to_str().unwrap(),
    ]);

    // Read ffmpeg's display sets back: the page ids its descriptor declared,
    // then one data field per display set.
    let theirs = demux(
        TsDemux::new().with_stream(TsStream::DvbSub),
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
        std::fs::read(&reference).expect("read the reference TS"),
    )
    .await;
    assert!(theirs.len() > 1, "ffmpeg wrote display sets to read back");
    let ids = parse_page_ids(&theirs[0].0).expect("ffmpeg declared its pages");

    // Re-mux them and compare with the reference wire bytes: the same PMT
    // descriptor, and the same PES payload for every display set.
    let ours = mux_ts(&theirs).await;
    let reference_bytes = std::fs::read(&reference).expect("read the reference TS");
    assert_eq!(
        subtitling_descriptor(&ours),
        subtitling_descriptor(&reference_bytes),
        "g2g writes the same subtitling_descriptor ffmpeg does"
    );
    assert_eq!(
        subtitling_descriptor(&ours).map(|(_, _, ids)| ids),
        Some(ids),
        "the descriptor declares the pages the stream carried"
    );
    let ours_back = demux(
        TsDemux::new().with_stream(TsStream::DvbSub),
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
        ours,
    )
    .await;
    assert_eq!(
        ours_back.iter().map(|(d, _)| d).collect::<Vec<_>>(),
        theirs.iter().map(|(d, _)| d).collect::<Vec<_>>(),
        "every data field is byte for byte what ffmpeg wrote"
    );
    for (i, ((_, ours), (_, theirs))) in ours_back.iter().zip(&theirs).enumerate() {
        assert_eq!(ours.pts_ns, theirs.pts_ns, "display set {i} keeps its time");
    }

    // Decode: transcoding each transport stream to DVD subpictures is bitmap to
    // bitmap, so ffmpeg has to compose every display set out of the carriage. It
    // gets the same cues out of ours as out of its own.
    let ours_ts = temp_path("dvb-ours.ts");
    std::fs::write(&ours_ts, mux_ts(&theirs).await).expect("write our TS");
    let decode = |source: &PathBuf, into: &PathBuf| {
        ffmpeg(&[
            "-i",
            source.to_str().unwrap(),
            "-c:s",
            "dvdsub",
            "-f",
            "matroska",
            into.to_str().unwrap(),
        ]);
        ffprobe(&[
            "-select_streams",
            "s:0",
            "-show_entries",
            "packet=pts_time,duration_time",
            "-of",
            "csv=p=0",
            into.to_str().unwrap(),
        ])
    };
    let (ours_dec, theirs_dec) = (temp_path("dec-ours.mkv"), temp_path("dec-ref.mkv"));
    let decoded = decode(&ours_ts, &ours_dec);
    assert!(
        !decoded.trim().is_empty(),
        "ffmpeg decoded cues out of ours"
    );
    assert_eq!(
        decoded,
        decode(&reference, &theirs_dec),
        "the same cues at the same times as out of ffmpeg's own stream"
    );

    // The same display sets into Matroska: ffmpeg must read the track back as
    // DVB subtitles carrying the page ids it declared itself.
    let mkv = temp_path("dvb.mkv");
    std::fs::write(&mkv, mux_mkv(SubPictureFormat::DvbSub, &theirs).await).expect("write the mkv");
    let probe = ffprobe(&[
        "-select_streams",
        "s:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "csv=p=0",
        mkv.to_str().unwrap(),
    ]);
    assert_eq!(probe.trim(), "dvb_subtitle", "ffprobe reads a DVB track");

    // ffmpeg's own remux of our Matroska back to a transport stream rebuilds the
    // descriptor from the CodecPrivate we wrote, so the pages survive the trip
    // through the other implementation.
    let remux = temp_path("dvb-remux.ts");
    ffmpeg(&[
        "-i",
        mkv.to_str().unwrap(),
        "-c:s",
        "copy",
        "-f",
        "mpegts",
        remux.to_str().unwrap(),
    ]);
    // The language is Matroska's own default (`eng` for a track that declares
    // none), so only the pages and the type carry across; they are what the
    // `CodecPrivate` holds.
    assert_eq!(
        subtitling_descriptor(&std::fs::read(&remux).expect("read the remuxed TS"))
            .map(|(_, kind, ids)| (kind, ids)),
        Some((0x10, ids)),
        "ffmpeg rebuilt the descriptor from our CodecPrivate"
    );

    for p in [
        idx, sub, reference, mkv, remux, ours_ts, ours_dec, theirs_dec,
    ] {
        let _ = std::fs::remove_file(p);
    }
}

/// The `.idx` this muxer writes as a `CodecPrivate` is the text ffmpeg writes
/// for the same pair, byte for byte.
#[test]
fn the_idx_codec_private_is_byte_identical_to_ffmpegs() {
    if !have_ffmpeg() {
        eprintln!("skipping m927 ffmpeg cross-check: no ffmpeg on PATH");
        return;
    }
    let (idx, sub) = (temp_path("cp.idx"), temp_path("cp.sub"));
    author_vobsub(&idx, &sub);
    let theirs = temp_path("cp-ref.mkv");
    ffmpeg(&[
        "-i",
        idx.to_str().unwrap(),
        "-c:s",
        "copy",
        theirs.to_str().unwrap(),
    ]);
    let reference = std::fs::read(&theirs).expect("read ffmpeg's mkv");
    let mut d = MatroskaDemuxer::new();
    d.push_data(&reference);
    let their_private = d
        .codec_private(1)
        .expect("ffmpeg wrote a CodecPrivate")
        .to_vec();

    let ours = idx_config_text(&parse_idx(&std::fs::read(&idx).unwrap()).expect("parse the .idx"));
    assert_eq!(
        ours.into_bytes(),
        their_private,
        "the .idx text matches ffmpeg's byte for byte"
    );

    for p in [idx, sub, theirs] {
        let _ = std::fs::remove_file(p);
    }
}
