//! M928: the single-track Matroska muxer takes a subtitle pad. `matroskamux`
//! with one link writes a `Caps::Text` pad as an `S_TEXT/*` track and a
//! `Caps::SubPicture` pad as `S_VOBSUB` / `S_DVBSUB`, so a sidecar subtitle file
//! muxes without the `name=m` fan-in shape.
//!
//! The VobSub cues are the hand-authored `.idx` / `.sub` pair of the
//! `vobsub_fixture` module, read through `vobsubsrc`; ffprobe is the reference
//! peer over the files this muxer writes.
#![cfg(feature = "std")]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::frame::{Frame, PipelinePacket};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph, SourceLoop};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PropValue, PropertySpec, PushOutcome, SubPictureFormat, TextFormat,
};

use g2g_plugins::dvbsub::{page_id_blob, parse_page_ids, PageIds};
use g2g_plugins::matroska::{MatroskaDemuxer, MkvCodec};
use g2g_plugins::mkvdemux::{MkvDemux, MkvStream};
use g2g_plugins::mkvmux::MkvMux;
use g2g_plugins::registry::default_registry;
use g2g_plugins::subparse::ASS_SCRIPT_HEADER;
use g2g_plugins::vobsubsrc::VobSubSrc;

mod vobsub_fixture;
use vobsub_fixture::{author_vobsub, cues, have_ffmpeg, CUE_DURATION_NS, H, PALETTE, W};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

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

    fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (data, _) in self.frames() {
            out.extend_from_slice(&data);
        }
        out
    }
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m928-{}-{name}", std::process::id()))
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

fn matroska_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::Matroska,
    }
}

/// Mux one subtitle stream through the single-track muxer, the `! matroskamux !`
/// shape this milestone is about.
async fn mux(mut el: MkvMux, caps: &Caps, stream: &[(Vec<u8>, FrameTiming)]) -> Vec<u8> {
    el.configure_pipeline(caps)
        .expect("matroskamux accepts the subtitle pad");
    let mut sink = CaptureSink::default();
    for (data, timing) in stream {
        el.process(frame(data.clone(), *timing), &mut sink)
            .await
            .expect("mux a cue");
    }
    el.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux EOS");
    sink.bytes()
}

/// Run a demuxer element over a whole byte stream, returning what it emitted.
async fn demux(mut el: MkvDemux, bytes: Vec<u8>) -> Vec<(Vec<u8>, FrameTiming)> {
    el.configure_pipeline(&matroska_caps())
        .expect("demuxer configures");
    let mut sink = CaptureSink::default();
    el.process(frame(bytes, FrameTiming::default()), &mut sink)
        .await
        .expect("demux");
    el.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("demux EOS");
    sink.frames()
}

fn first_track(bytes: &[u8]) -> MkvCodec {
    let mut d = MatroskaDemuxer::new();
    d.push_data(bytes);
    d.tracks().first().expect("one track").codec
}

fn codec_private(bytes: &[u8]) -> Vec<u8> {
    let mut d = MatroskaDemuxer::new();
    d.push_data(bytes);
    d.codec_private(1)
        .expect("track 1 has a CodecPrivate")
        .to_vec()
}

/// The VobSub sidecar as a pad stream: the `.idx` config frame then the cues,
/// exactly what `vobsubsrc` puts on a `Caps::SubPicture{VobSub}` pad.
async fn vobsub_stream(idx: &PathBuf) -> Vec<(Vec<u8>, FrameTiming)> {
    let mut src = VobSubSrc::new(idx);
    src.configure_pipeline(&Caps::SubPicture {
        format: SubPictureFormat::VobSub,
    })
    .expect("vobsubsrc configures");
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.expect("read the sidecar pair");
    sink.frames()
}

/// The `.idx` text a VobSub track carries: the size and palette lines alone.
fn expected_idx() -> String {
    let palette = PALETTE
        .iter()
        .map(|c| format!("{c:06x}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("size: {W}x{H}\npalette: {palette}\n")
}

/// A hand-built DVB display set: a page composition listing no region, behind the
/// segment framing EN 300 743 defines.
fn dvb_display_set(page_id: u16) -> Vec<u8> {
    let mut out = Vec::from([0x0Fu8, 0x10]);
    out.extend_from_slice(&page_id.to_be_bytes());
    out.extend_from_slice(&2u16.to_be_bytes());
    out.extend_from_slice(&[30, 0x80]);
    out.extend_from_slice(&[0x0F, 0x80]);
    out.extend_from_slice(&page_id.to_be_bytes());
    out.extend_from_slice(&0u16.to_be_bytes());
    out
}

// ---- text pads ----

/// The cues of a `Caps::Text` pad become an `S_TEXT/UTF8` track, and each keeps
/// its display window: a `SimpleBlock` has nowhere to put one, so this fails if
/// the muxer writes a cue as one.
#[tokio::test]
async fn a_text_pad_writes_an_s_text_track_with_cue_durations() {
    let caps = Caps::Text {
        format: TextFormat::Utf8,
    };
    let stream = vec![
        (
            Vec::from(b"first cue".as_slice()),
            at(1_000_000_000, 2_000_000_000),
        ),
        (
            Vec::from("second cue\nover two lines".as_bytes()),
            at(4_000_000_000, 1_500_000_000),
        ),
    ];

    let muxed = mux(MkvMux::new(), &caps, &stream).await;
    assert!(
        muxed.windows(11).any(|w| w == b"S_TEXT/UTF8"),
        "the track declares the S_TEXT/UTF8 CodecID"
    );
    assert_eq!(first_track(&muxed), MkvCodec::Subtitle(TextFormat::Utf8));

    let back = demux(
        MkvDemux::new().with_stream(MkvStream::Subtitle(TextFormat::Utf8)),
        muxed,
    )
    .await;
    assert_eq!(back.len(), stream.len(), "every cue survives");
    for (i, (data, timing)) in back.iter().enumerate() {
        assert_eq!(data, &stream[i].0, "cue {i} keeps its text");
        assert_eq!(timing.pts_ns, stream[i].1.pts_ns, "cue {i} keeps its time");
        assert_eq!(
            timing.duration_ns, stream[i].1.duration_ns,
            "cue {i} keeps its display window"
        );
    }
}

/// The storage syntax is the muxer's, not the pad's: `ass` writes `S_TEXT/ASS`
/// behind the script header `CodecPrivate` a reader needs, and the blocks
/// de-frame back to the same plain text.
#[tokio::test]
async fn the_ass_storage_syntax_writes_the_script_header_as_codec_private() {
    let caps = Caps::Text {
        format: TextFormat::Utf8,
    };
    let stream = vec![
        (Vec::from(b"one".as_slice()), at(0, 1_000_000_000)),
        (
            Vec::from(b"two".as_slice()),
            at(2_000_000_000, 1_000_000_000),
        ),
    ];

    let muxed = mux(
        MkvMux::new().with_subtitle_format(TextFormat::Ssa),
        &caps,
        &stream,
    )
    .await;
    assert_eq!(first_track(&muxed), MkvCodec::Subtitle(TextFormat::Ssa));
    assert_eq!(
        codec_private(&muxed),
        ASS_SCRIPT_HEADER.as_bytes(),
        "the ASS script header is the CodecPrivate"
    );
    // Each block leads with its own ReadOrder, which a reader orders events by.
    let mut d = MatroskaDemuxer::new();
    d.push_data(&muxed);
    let blocks = d.take_frames();
    assert_eq!(blocks.len(), 2);
    assert!(blocks[0].data.starts_with(b"0,0,Default,,0,0,0,,"));
    assert!(blocks[1].data.starts_with(b"1,0,Default,,0,0,0,,"));

    let back = demux(
        MkvDemux::new().with_stream(MkvStream::Subtitle(TextFormat::Ssa)),
        muxed,
    )
    .await;
    assert_eq!(
        back.iter().map(|(d, _)| d.clone()).collect::<Vec<_>>(),
        stream.iter().map(|(d, _)| d.clone()).collect::<Vec<_>>(),
        "the cues de-frame back to the text that went in"
    );
}

// ---- bitmap pads ----

/// A `vobsubsrc` stream muxes with one link: the `.idx` config frame it opens
/// with becomes the track's `CodecPrivate` rather than a cue, and the subpicture
/// units come back at their authored times with their display windows.
#[tokio::test]
async fn a_vobsub_pad_writes_an_s_vobsub_track_with_the_idx_as_codec_private() {
    let (idx, sub) = (temp_path("mkv.idx"), temp_path("mkv.sub"));
    author_vobsub(&idx, &sub);
    let stream = vobsub_stream(&idx).await;
    assert_eq!(stream.len(), 1 + cues().len(), "the config, then the cues");

    let caps = Caps::SubPicture {
        format: SubPictureFormat::VobSub,
    };
    let muxed = mux(MkvMux::new(), &caps, &stream).await;
    assert!(
        muxed.windows(8).any(|w| w == b"S_VOBSUB"),
        "the track declares the S_VOBSUB CodecID"
    );
    assert_eq!(first_track(&muxed), MkvCodec::VobSub);
    assert_eq!(
        String::from_utf8(codec_private(&muxed)).expect("the .idx is text"),
        expected_idx(),
        "the config frame became the CodecPrivate, not a cue"
    );

    let back = demux(MkvDemux::new().with_stream(MkvStream::VobSub), muxed).await;
    assert_eq!(back.len(), stream.len(), "config plus every cue survives");
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

/// The DVB pad's page-id blob becomes the `S_DVBSUB` `CodecPrivate`, and a
/// display set that arrived in its transport-stream carriage is unwrapped: a
/// Matroska block holds the bare segments.
#[tokio::test]
async fn a_dvbsub_pad_writes_the_page_ids_and_drops_the_data_field_header() {
    let ids = PageIds {
        composition: 7,
        ancillary: 9,
    };
    let blob = page_id_blob(ids, 0x10);
    let set = dvb_display_set(7);
    let mut wrapped = Vec::from([0x20u8, 0x00]);
    wrapped.extend_from_slice(&set);
    wrapped.push(0xFF);
    let stream = vec![
        (blob.to_vec(), FrameTiming::default()),
        (wrapped, at(1_500_000_000, 3_000_000_000)),
    ];

    let caps = Caps::SubPicture {
        format: SubPictureFormat::DvbSub,
    };
    let muxed = mux(MkvMux::new(), &caps, &stream).await;
    assert!(
        muxed.windows(8).any(|w| w == b"S_DVBSUB"),
        "the track declares the S_DVBSUB CodecID"
    );
    assert_eq!(first_track(&muxed), MkvCodec::DvbSub);
    assert_eq!(
        codec_private(&muxed),
        blob.to_vec(),
        "the CodecPrivate is the five-byte page-id blob"
    );

    let back = demux(MkvDemux::new().with_stream(MkvStream::DvbSub), muxed).await;
    assert_eq!(back.len(), 2, "the page ids, then the display set");
    assert_eq!(
        parse_page_ids(&back[0].0),
        Some(ids),
        "the demuxer hands the page ids back in band"
    );
    assert_eq!(back[1].0, set, "the block holds the segments alone");
    assert_eq!(back[1].1.pts_ns, 1_500_000_000);
    assert_eq!(back[1].1.duration_ns, 3_000_000_000);
}

// ---- properties ----

/// Both subtitle knobs are settable at runtime, the path `parse_launch` takes.
#[test]
fn the_subtitle_properties_round_trip_on_the_single_track_muxer() {
    let mux = MkvMux::new();
    let props: &[PropertySpec] = mux.properties();
    for name in ["subtitle-format", "dvbsub-page-id"] {
        assert!(
            props.iter().any(|s| s.name == name),
            "{name} is declared, so parse_launch can set it"
        );
    }

    let mut mux = MkvMux::new();
    assert_eq!(
        mux.get_property("subtitle-format"),
        Some(PropValue::Str("utf8".into()))
    );
    mux.set_property("subtitle-format", PropValue::Str("ass".into()))
        .expect("ass is a storage syntax");
    assert_eq!(
        mux.get_property("subtitle-format"),
        Some(PropValue::Str("ass".into()))
    );
    assert!(
        mux.set_property("subtitle-format", PropValue::Str("webvtt".into()))
            .is_err(),
        "a syntax the muxer cannot write is refused, not ignored"
    );

    mux.set_property("dvbsub-page-id", PropValue::Uint(42))
        .unwrap();
    assert_eq!(
        mux.get_property("dvbsub-page-id"),
        Some(PropValue::Uint(42))
    );
    assert!(
        mux.set_property("dvbsub-page-id", PropValue::Uint(70_000))
            .is_err(),
        "a page id past the 16-bit field is refused rather than truncated"
    );

    // A document-format text pad carries whole-file bytes, not cues: refused.
    assert!(mux
        .intercept_caps(&Caps::Text {
            format: TextFormat::Srt
        })
        .is_err());
    assert!(mux
        .intercept_caps(&Caps::Text {
            format: TextFormat::Utf8
        })
        .is_ok());
    // PGS has no Matroska carriage here, so its pad is refused too.
    assert!(mux
        .intercept_caps(&Caps::SubPicture {
            format: SubPictureFormat::Pgs
        })
        .is_err());
}

/// The property is what a stream carrying no page-id config is declared on.
#[tokio::test]
async fn the_page_id_property_declares_a_stream_that_sends_no_config() {
    let mut el = MkvMux::new();
    el.set_property("dvbsub-page-id", PropValue::Uint(5))
        .unwrap();
    let muxed = mux(
        el,
        &Caps::SubPicture {
            format: SubPictureFormat::DvbSub,
        },
        &[(dvb_display_set(5), at(0, 0))],
    )
    .await;
    assert_eq!(
        codec_private(&muxed),
        page_id_blob(
            PageIds {
                composition: 5,
                ancillary: 5
            },
            0x10
        )
        .to_vec(),
        "the property's pages reach the CodecPrivate"
    );
}

// ---- the launch line ----

/// The milestone's shape: one link, no `name=m`. The graph parses, negotiates and
/// runs, and the file on disk is the `S_VOBSUB` track with its cues.
#[tokio::test]
async fn a_sidecar_subtitle_file_muxes_over_a_single_link() {
    let (idx, sub, out) = (
        temp_path("launch.idx"),
        temp_path("launch.sub"),
        temp_path("launch.mkv"),
    );
    author_vobsub(&idx, &sub);

    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        &format!(
            "vobsubsrc location={} ! matroskamux ! filesink location={}",
            idx.display(),
            out.display()
        ),
    )
    .expect("the single-link launch line parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");

    let file = std::fs::read(&out).expect("the muxer wrote a file");
    assert_eq!(first_track(&file), MkvCodec::VobSub);
    assert_eq!(
        String::from_utf8(codec_private(&file)).expect("the .idx is text"),
        expected_idx()
    );
    let mut d = MatroskaDemuxer::new();
    d.push_data(&file);
    assert_eq!(
        d.take_frames().len(),
        cues().len(),
        "every cue reached the file, and the config blob was not one"
    );

    for p in [idx, sub, out] {
        let _ = std::fs::remove_file(p);
    }
}

// ---- reference peer: ffprobe ----

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

/// ffprobe reads the subtitle track out of what the single-track muxer wrote:
/// a text pad as `subrip`, a VobSub pad as `dvd_subtitle` with its cues at the
/// times they were authored at.
#[tokio::test]
async fn ffprobe_reads_the_tracks_the_single_track_muxer_wrote() {
    if !have_ffmpeg() {
        eprintln!("skipping m928 ffprobe cross-check: no ffmpeg on PATH");
        return;
    }
    let codec_name = |path: &PathBuf| {
        ffprobe(&[
            "-select_streams",
            "s:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            path.to_str().unwrap(),
        ])
        .trim()
        .to_string()
    };

    // A text pad.
    let text = temp_path("text.mkv");
    let cues_in = vec![
        (
            Vec::from(b"hello".as_slice()),
            at(1_000_000_000, 2_000_000_000),
        ),
        (
            Vec::from(b"goodbye".as_slice()),
            at(4_000_000_000, 2_000_000_000),
        ),
    ];
    std::fs::write(
        &text,
        mux(
            MkvMux::new(),
            &Caps::Text {
                format: TextFormat::Utf8,
            },
            &cues_in,
        )
        .await,
    )
    .expect("write the text mkv");
    assert_eq!(codec_name(&text), "subrip", "ffprobe reads a text track");

    // A VobSub pad, through the launch line's own file.
    let (idx, sub, bitmap) = (
        temp_path("probe.idx"),
        temp_path("probe.sub"),
        temp_path("probe.mkv"),
    );
    author_vobsub(&idx, &sub);
    let stream = vobsub_stream(&idx).await;
    std::fs::write(
        &bitmap,
        mux(
            MkvMux::new(),
            &Caps::SubPicture {
                format: SubPictureFormat::VobSub,
            },
            &stream,
        )
        .await,
    )
    .expect("write the vobsub mkv");
    assert_eq!(
        codec_name(&bitmap),
        "dvd_subtitle",
        "ffprobe reads a VobSub track"
    );

    let times = ffprobe(&[
        "-select_streams",
        "s:0",
        "-show_entries",
        "packet=pts_time",
        "-of",
        "csv=p=0",
        bitmap.to_str().unwrap(),
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

    for p in [text, idx, sub, bitmap] {
        let _ = std::fs::remove_file(p);
    }
}
