//! M1049 - AV1 carriage in MPEG-TS, and DVB EIT present/following.
//!
//! AV1 has no `stream_type` of its own: the AOM carriage puts it on a private PES
//! (0x06) told apart by a `registration_descriptor`. The spec assigns 'AV01', but
//! GStreamer, the only implementation that identifies the carriage at all, reads
//! and writes 'AV1G' (ffmpeg's muxer writes a bare 0x06 with no descriptor, which
//! not even its own demuxer identifies). So the demux accepts both and the mux
//! writes 'AV1G', and the reference peer for both directions is GStreamer, with
//! ffmpeg's `obu` demuxer as a second, independent check that the elementary
//! stream we recover is a valid AV1 bitstream.
//!
//! The EIT legs need no external tool: sections are hand-built in `eit_common`
//! with a real MPEG-2 CRC-32 (computed there, so a broken CRC in the parser
//! cannot pass by agreeing with itself).
#![cfg(feature = "std")]

use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, Bus, BusMessage, ByteStreamEncoding, Caps, Dim, MultiInputElement,
    PropValue, Rate, Tag, VideoCodec,
};
use g2g_plugins::mpegts::{
    EitSlot, TsDemuxer, TsMuxer, STREAM_TYPE_PRIVATE_PES, TAG_KEY_EVENT_NAME, TAG_KEY_EVENT_TEXT,
    TAG_KEY_NEXT_EVENT_NAME, TAG_KEY_NEXT_EVENT_TEXT, TS_PACKET_LEN,
};
use g2g_plugins::tsdemux::{TsDemux, TsStream};

mod eit_common;
use eit_common::{
    crc32_mpeg, data_frame, eit_event, eit_section, parse_sections, psi_packets,
    short_event_descriptor, CaptureSink, ANNEX_C_START_TIME, EVENT_DURATION, PID_EIT,
    TABLE_ID_EIT_PF, TABLE_ID_EIT_SCHEDULE_OTHER_TS,
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const FRAMES: usize = 25;
/// The `registration_descriptor` GStreamer reads and writes for AV1 in TS.
const AV1G_DESCRIPTOR: &[u8] = &[0x05, 0x04, b'A', b'V', b'1', b'G'];

// ---------------------------------------------------------------- test helpers

fn have(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
        || Command::new(cmd)
            .arg("-version")
            .output()
            .is_ok_and(|o| o.status.success())
}

fn temp_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("g2g-m1049-{tag}-{}.bin", std::process::id()))
}

/// The AV1 elements a GStreamer reference run needs. `mpegtsmux` refuses AV1
/// without `enable-custom-mappings`, and only recent builds carry it.
fn have_gstreamer_av1() -> bool {
    if !have("gst-launch-1.0") || !have("gst-inspect-1.0") {
        return false;
    }
    ["svtav1enc", "av1parse", "tsdemux", "mpegtsmux", "dav1ddec"]
        .iter()
        .all(|e| {
            Command::new("gst-inspect-1.0")
                .arg(e)
                .output()
                .is_ok_and(|o| o.status.success())
        })
}

/// An AV1 transport stream authored end to end by GStreamer: the third-party
/// stream the demux side is validated against. `None` when the pipeline fails.
/// `tag` keeps concurrent tests off each other's scratch file.
fn gstreamer_av1_ts(tag: &str) -> Option<Vec<u8>> {
    let path = temp_path(tag);
    let status = Command::new("gst-launch-1.0")
        .args([
            "videotestsrc",
            &format!("num-buffers={FRAMES}"),
            "!",
            &format!("video/x-raw,width={WIDTH},height={HEIGHT},framerate=25/1"),
            "!",
            "svtav1enc",
            "preset=10",
            "!",
            "av1parse",
            "!",
            "mpegtsmux",
            "enable-custom-mappings=true",
            "!",
            "filesink",
        ])
        .arg(format!("location={}", path.display()))
        .output()
        .ok()?;
    let ts = std::fs::read(&path).ok().filter(|b| !b.is_empty());
    let _ = std::fs::remove_file(&path);
    status.status.success().then_some(ts).flatten()
}

fn timed_frame(bytes: &[u8], pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}

/// One ADTS AAC frame around `payload`, so the fan-in multiplex has a second
/// elementary stream that the muxer treats as ordinary audio.
fn adts_au(payload: &[u8]) -> Vec<u8> {
    let frame_len = payload.len() + 7;
    let mut au = Vec::from([
        0xFF,
        0xF1,
        (1 << 6) | (3 << 2),
        (2 << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F,
        0xFC,
    ]);
    au.extend_from_slice(payload);
    au
}

/// Run a whole transport stream through `TsDemux` selecting `stream`.
async fn demux_stream(ts: &[u8], stream: TsStream) -> CaptureSink {
    let mut demux = TsDemux::new().with_stream(stream);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        })
        .expect("tsdemux accepts an MPEG-TS byte stream");
    let mut sink = CaptureSink::default();
    demux.process(data_frame(ts), &mut sink).await.unwrap();
    demux.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    sink
}

/// Run a whole transport stream through `TsDemux` on the AV1 port.
async fn demux_av1(ts: &[u8]) -> CaptureSink {
    demux_stream(ts, TsStream::Av1).await
}

// ------------------------------------------------------------------ AV1 legs

/// The demux side against a third-party author: a GStreamer-written AV1 TS must
/// yield one temporal unit per frame, and ffmpeg's `obu` demuxer must decode the
/// concatenated units back to full-size frames. Two independent peers, so neither
/// a g2g mux nor a g2g parse is in the loop on the reference side.
#[tokio::test]
async fn g2g_demuxes_a_gstreamer_authored_av1_ts() {
    if !have_gstreamer_av1() {
        eprintln!("skipping: no GStreamer AV1 elements");
        return;
    }
    let Some(ts) = gstreamer_av1_ts("gst-ts-demux") else {
        eprintln!("skipping: the GStreamer AV1 mux pipeline did not run");
        return;
    };
    let sink = demux_av1(&ts).await;
    // At least one temporal unit per encoded frame: a random-access encode also
    // emits show-existing-frame units, which are units of their own.
    assert!(
        sink.aus.len() >= FRAMES,
        "every encoded frame's temporal unit is forwarded: {}",
        sink.aus.len()
    );

    if !have("ffmpeg") {
        eprintln!("note: no ffmpeg, skipping the elementary-stream decode check");
        return;
    }
    let obu_path = temp_path("demuxed-obu");
    let raw_path = temp_path("demuxed-raw");
    std::fs::write(&obu_path, sink.aus.concat()).unwrap();
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "fatal", "-f", "obu", "-i"])
        .arg(&obu_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&raw_path)
        .status()
        .unwrap();
    assert!(
        status.success(),
        "ffmpeg reads the demuxed bytes as an AV1 OBU stream"
    );
    let frame_bytes = (WIDTH * HEIGHT) as usize * 3 / 2;
    assert_eq!(
        std::fs::read(&raw_path).unwrap().len(),
        frame_bytes * FRAMES,
        "every frame decodes at {WIDTH}x{HEIGHT}"
    );
    let _ = std::fs::remove_file(&obu_path);
    let _ = std::fs::remove_file(&raw_path);
}

/// The mux side against a real receiver: GStreamer's `tsdemux ! av1parse !
/// dav1ddec` must decode the g2g-muxed transport stream to full-size frames,
/// which only happens if the PMT entry carries the AV1 registration GStreamer
/// keys on and the PES framing holds.
#[tokio::test]
async fn gstreamer_decodes_a_g2g_muxed_av1_ts() {
    if !have_gstreamer_av1() {
        eprintln!("skipping: no GStreamer AV1 elements");
        return;
    }
    let Some(source) = gstreamer_av1_ts("gst-ts-mux") else {
        eprintln!("skipping: the GStreamer AV1 mux pipeline did not run");
        return;
    };
    // Access units authored by GStreamer's encoder, recovered as the demux leg
    // proves they can be, so this leg exercises only the muxer.
    let aus = demux_av1(&source).await.aus;
    assert!(aus.len() >= FRAMES, "the source units to re-mux");

    let mut mux = TsMuxer::new(STREAM_TYPE_PRIVATE_PES);
    mux.set_stream_av1(0);
    let mut ts = Vec::new();
    for (i, au) in aus.iter().enumerate() {
        ts.extend_from_slice(&mux.push_au_on(0, au, Some(i as u64 * 3600), None));
    }
    assert!(
        ts.windows(AV1G_DESCRIPTOR.len())
            .any(|w| w == AV1G_DESCRIPTOR),
        "the PMT carries the AV1 registration descriptor"
    );

    let ts_path = temp_path("g2g-av1-ts");
    let raw_path = temp_path("gst-decoded");
    std::fs::write(&ts_path, &ts).unwrap();
    let run = Command::new("gst-launch-1.0")
        .arg("filesrc")
        .arg(format!("location={}", ts_path.display()))
        .args([
            "!",
            "tsdemux",
            "!",
            "av1parse",
            "!",
            "dav1ddec",
            "!",
            "videoconvert",
            "!",
            "video/x-raw,format=I420",
            "!",
            "filesink",
        ])
        .arg(format!("location={}", raw_path.display()))
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "GStreamer decodes the g2g-muxed AV1 TS: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let frame_bytes = (WIDTH * HEIGHT) as usize * 3 / 2;
    assert_eq!(
        std::fs::read(&raw_path).unwrap().len(),
        frame_bytes * FRAMES,
        "GStreamer decodes every frame at {WIDTH}x{HEIGHT}"
    );
    let _ = std::fs::remove_file(&ts_path);
    let _ = std::fs::remove_file(&raw_path);
}

/// The recovered stream is what `av1parse` reads: it must recover the encoded
/// geometry from the sequence header of the demuxed temporal units.
#[tokio::test]
async fn av1parse_reads_the_geometry_of_the_demuxed_stream() {
    if !have_gstreamer_av1() {
        eprintln!("skipping: no GStreamer AV1 elements");
        return;
    }
    let Some(ts) = gstreamer_av1_ts("gst-ts-parse") else {
        eprintln!("skipping: the GStreamer AV1 mux pipeline did not run");
        return;
    };
    let aus = demux_av1(&ts).await.aus;
    assert!(!aus.is_empty(), "the demuxer forwarded AV1 units");

    let mut parse = g2g_plugins::av1parse::Av1Parse::new();
    parse
        .configure_pipeline(&Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Any,
        })
        .expect("av1parse accepts the demuxed caps");
    let mut sink = CaptureSink::default();
    for au in &aus {
        parse.process(data_frame(au), &mut sink).await.unwrap();
    }
    let sized = sink.caps.iter().find_map(|c| match c {
        Caps::CompressedVideo { width, height, .. } => Some((width.clone(), height.clone())),
        _ => None,
    });
    assert_eq!(
        sized,
        Some((Dim::Fixed(WIDTH), Dim::Fixed(HEIGHT))),
        "the sequence header of the demuxed units carries the encoded geometry"
    );
}

/// A g2g-muxed AV1 stream round-trips through the g2g demuxer, and the PMT entry
/// reads back as AV1 rather than the private PES default of KLV. No external
/// tool, so this leg runs everywhere and pins the descriptor mapping itself.
#[tokio::test]
async fn av1_survives_a_g2g_mux_demux_round_trip() {
    // A minimal but well-formed temporal unit: a temporal delimiter, then a frame
    // OBU. The bytes only have to survive the container.
    let aus: Vec<Vec<u8>> = (0..4u8)
        .map(|i| Vec::from([0x12, 0x00, 0x32, 0x03, 0x10, 0x20, i]))
        .collect();
    let mut mux = TsMuxer::new(STREAM_TYPE_PRIVATE_PES);
    mux.set_stream_av1(0);
    let mut ts = Vec::new();
    for (i, au) in aus.iter().enumerate() {
        ts.extend_from_slice(&mux.push_au_on(0, au, Some(i as u64 * 3600), None));
    }

    let mut parser = TsDemuxer::new();
    for pkt in ts.chunks(TS_PACKET_LEN) {
        parser.push_packet(pkt);
    }
    let es = parser
        .streams()
        .iter()
        .find(|s| s.stream_type == STREAM_TYPE_PRIVATE_PES)
        .expect("the PMT names the private stream");
    assert!(es.av1, "the registration marks the stream as AV1");
    assert!(!es.klv, "an AV1 stream is not read as KLV");

    let sink = demux_av1(&ts).await;
    assert_eq!(sink.aus, aus, "every access unit survives the round trip");
}

/// The fan-in muxer declares AV1 per input pad, so an A/V multiplex whose AV1
/// rides a non-zero pad still gets its registration: AAC on input 0, AV1 on
/// input 1. Both streams come back off their own PID.
#[tokio::test]
async fn a_fan_in_multiplex_carries_av1_on_a_non_zero_pad() {
    let video: Vec<Vec<u8>> = (0..3u8)
        .map(|i| Vec::from([0x12, 0x00, 0x32, 0x03, 0x10, 0x20, i]))
        .collect();
    let audio: Vec<Vec<u8>> = (0..3u8).map(|i| adts_au(&[0xA0 + i, 0xB0])).collect();

    let mut mux = g2g_plugins::tsmuxn::TsMux::new(2);
    mux.configure_pipeline(
        0,
        &Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
        },
    )
    .expect("the fan-in muxer accepts AAC on input 0");
    mux.configure_pipeline(
        1,
        &Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Fixed(25 << 16),
        },
    )
    .expect("the fan-in muxer accepts AV1 on input 1");

    let mut sink = CaptureSink::default();
    for i in 0..video.len() {
        let pts = i as u64 * 40_000_000;
        mux.process(1, timed_frame(&video[i], pts), &mut sink)
            .await
            .unwrap();
        mux.process(0, timed_frame(&audio[i], pts), &mut sink)
            .await
            .unwrap();
    }
    for input in 0..2 {
        mux.process(input, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
    }
    let ts = sink.aus.concat();
    assert!(
        ts.windows(AV1G_DESCRIPTOR.len())
            .any(|w| w == AV1G_DESCRIPTOR),
        "the PMT carries the AV1 registration for the non-zero pad"
    );

    assert_eq!(
        demux_av1(&ts).await.aus,
        video,
        "the AV1 units come back off their own PID"
    );
    let recovered_audio = demux_stream(&ts, TsStream::Aac).await.aus;
    assert_eq!(recovered_audio, audio, "the AAC AUs are untouched");
}

/// A private PES with no AV1 registration must not reach an AV1 port: the marker
/// is what tells AV1 apart from every other 0x06 use.
#[tokio::test]
async fn an_unmarked_private_stream_is_not_av1() {
    let mut mux = TsMuxer::new(STREAM_TYPE_PRIVATE_PES); // the KLVA default
    let mut ts = Vec::new();
    for i in 0..4u64 {
        ts.extend_from_slice(&mux.push_au_on(0, &[0x12, 0x00, i as u8], Some(i * 3600), None));
    }
    let sink = demux_av1(&ts).await;
    assert!(
        sink.aus.is_empty(),
        "a KLV private stream is not forwarded as AV1"
    );
}

/// A launch line reaches the AV1 port through the `stream` property, which needs
/// the name in the spec table as well as in `set_property`.
#[test]
fn the_stream_property_selects_av1() {
    let mut demux = TsDemux::new();
    assert!(
        demux
            .properties()
            .iter()
            .any(|s| s.name == "stream" && s.blurb.contains("av1")),
        "the spec table names av1 among the selectable streams"
    );
    demux
        .set_property("stream", PropValue::Str("av1".into()))
        .expect("stream=av1 is accepted");
    assert_eq!(
        demux.get_property("stream"),
        Some(PropValue::Str("av1".into()))
    );
}

// ------------------------------------------------------------------ EIT legs

/// The present and following events of a service, with their names and short
/// descriptions, read off a valid pair of sections.
#[test]
fn eit_present_following_parses_both_events() {
    let present = eit_section(
        TABLE_ID_EIT_PF,
        1,
        3,
        0,
        &eit_event(
            0x1000,
            ANNEX_C_START_TIME,
            EVENT_DURATION,
            &short_event_descriptor(b"News at Ten", b"The headlines"),
        ),
    );
    let following = eit_section(
        TABLE_ID_EIT_PF,
        1,
        3,
        1,
        &eit_event(
            0x1001,
            ANNEX_C_START_TIME,
            EVENT_DURATION,
            &short_event_descriptor(b"Late Film", b"A thriller"),
        ),
    );
    let demux = parse_sections(&[present, following]);

    let events = demux.eit_events();
    assert_eq!(events.len(), 2, "present and following: {events:?}");
    let now = events
        .iter()
        .find(|e| e.slot == EitSlot::Present)
        .expect("present event");
    assert_eq!(now.service_id, 1);
    assert_eq!(now.event_id, 0x1000);
    assert_eq!(now.name, "News at Ten");
    assert_eq!(now.text, "The headlines");
    let next = events
        .iter()
        .find(|e| e.slot == EitSlot::Following)
        .expect("following event");
    assert_eq!(next.event_id, 0x1001);
    assert_eq!(next.name, "Late Film");
    assert_eq!(next.text, "A thriller");
}

/// A section repeating its `version_number` costs nothing, and a new version
/// replaces the event in place rather than piling up.
#[test]
fn eit_suppresses_a_repeated_version_and_adopts_a_new_one() {
    let v3 = eit_section(
        TABLE_ID_EIT_PF,
        1,
        3,
        0,
        &eit_event(
            0x1000,
            ANNEX_C_START_TIME,
            EVENT_DURATION,
            &short_event_descriptor(b"First", b""),
        ),
    );
    let mut demux = parse_sections(std::slice::from_ref(&v3));
    let after_first = demux.eit_generation();
    assert_eq!(after_first, 1, "the first section reports one event");

    for pkt in psi_packets(PID_EIT, &v3).chunks(TS_PACKET_LEN) {
        demux.push_packet(pkt);
    }
    assert_eq!(
        demux.eit_generation(),
        after_first,
        "a section repeating its version does not re-report"
    );

    let v4 = eit_section(
        TABLE_ID_EIT_PF,
        1,
        4,
        0,
        &eit_event(
            0x2000,
            ANNEX_C_START_TIME,
            EVENT_DURATION,
            &short_event_descriptor(b"Second", b""),
        ),
    );
    for pkt in psi_packets(PID_EIT, &v4).chunks(TS_PACKET_LEN) {
        demux.push_packet(pkt);
    }
    assert!(
        demux.eit_generation() > after_first,
        "a new version re-reports"
    );
    assert_eq!(
        demux.eit_events().len(),
        1,
        "the slot is replaced, not added"
    );
    assert_eq!(demux.eit_events()[0].name, "Second");
}

/// A section whose CRC does not check out is ignored: nothing else cross-checks
/// the text it carries.
#[test]
fn eit_rejects_a_corrupt_section() {
    let mut section = eit_section(
        TABLE_ID_EIT_PF,
        1,
        1,
        0,
        &eit_event(
            0x1000,
            ANNEX_C_START_TIME,
            EVENT_DURATION,
            &short_event_descriptor(b"Bad", b""),
        ),
    );
    let last = section.len() - 1;
    section[last] ^= 0xFF;
    assert!(
        parse_sections(&[section]).eit_events().is_empty(),
        "a bad CRC yields no events"
    );
}

/// The other-TS tables sharing PID 0x12 describe services carried elsewhere, so
/// neither the present/following variant (0x4F) nor a schedule one (0x60..=0x6F)
/// is read.
#[test]
fn eit_ignores_other_tables_on_its_pid() {
    for table_id in [0x4Fu8, TABLE_ID_EIT_SCHEDULE_OTHER_TS] {
        let mut section = eit_section(
            TABLE_ID_EIT_PF,
            1,
            1,
            0,
            &eit_event(
                0x1000,
                ANNEX_C_START_TIME,
                EVENT_DURATION,
                &short_event_descriptor(b"Elsewhere", b""),
            ),
        );
        section[0] = table_id;
        // Re-checksum, so the section is rejected on its table id, not its CRC.
        let body_len = section.len() - 4;
        let crc = crc32_mpeg(&section[..body_len]);
        section[body_len..].copy_from_slice(&crc.to_be_bytes());
        let mut demux = parse_sections(&[section]);
        assert!(
            demux.eit_events().is_empty() && demux.take_eit_schedule().is_empty(),
            "table_id {table_id:#04x} describes another transport stream"
        );
    }
}

/// A section too long for one TS packet reassembles across them: EIT event text
/// routinely overflows the 184-byte payload the other tables fit in.
#[test]
fn eit_reassembles_a_section_spanning_packets() {
    // A descriptor's length field is one byte, so the fields stay under it; the
    // section still outgrows a packet.
    let name = vec![b'A'; 100];
    let text = vec![b'B'; 100];
    let section = eit_section(
        TABLE_ID_EIT_PF,
        1,
        1,
        0,
        &eit_event(
            0x1000,
            ANNEX_C_START_TIME,
            EVENT_DURATION,
            &short_event_descriptor(&name, &text),
        ),
    );
    assert!(
        section.len() > TS_PACKET_LEN,
        "the fixture really does span packets ({} bytes)",
        section.len()
    );
    let demux = parse_sections(&[section]);
    let events = demux.eit_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].name, String::from_utf8(name).unwrap());
    assert_eq!(events[0].text, String::from_utf8(text).unwrap());
}

/// An event name in UTF-8 (character table 0x15) decodes; one in a table this
/// parser will not decode comes back empty rather than garbled.
#[test]
fn eit_decodes_utf8_text_and_declines_unknown_tables() {
    let mut utf8_name = Vec::from([0x15u8]);
    utf8_name.extend_from_slice("Tagesschau um 20\u{00A0}Uhr".as_bytes());
    let demux = parse_sections(&[eit_section(
        TABLE_ID_EIT_PF,
        1,
        1,
        0,
        &eit_event(
            0x1000,
            ANNEX_C_START_TIME,
            EVENT_DURATION,
            &short_event_descriptor(&utf8_name, &[0x05, b'x']),
        ),
    )]);
    let events = demux.eit_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].name, "Tagesschau um 20\u{00A0}Uhr");
    assert_eq!(
        events[0].text, "",
        "a character table this parser will not decode reports nothing"
    );
}

/// The demux element posts each service's events on the bus scoped to its
/// program, the way the SDT service name posts, and only when the table changes.
#[tokio::test]
async fn tsdemux_posts_eit_events_on_the_bus() {
    let (bus, handle) = Bus::new(32);
    let mut demux = TsDemux::new()
        .with_stream(TsStream::Av1)
        .with_bus(handle.clone());
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        })
        .unwrap();

    let mut ts = Vec::new();
    for section in [
        eit_section(
            TABLE_ID_EIT_PF,
            7,
            1,
            0,
            &eit_event(
                0x1000,
                ANNEX_C_START_TIME,
                EVENT_DURATION,
                &short_event_descriptor(b"Now", b"On air"),
            ),
        ),
        eit_section(
            TABLE_ID_EIT_PF,
            7,
            1,
            1,
            &eit_event(
                0x1001,
                ANNEX_C_START_TIME,
                EVENT_DURATION,
                &short_event_descriptor(b"Next", b"After this"),
            ),
        ),
    ] {
        ts.extend_from_slice(&psi_packets(PID_EIT, &section));
    }
    let mut sink = CaptureSink::default();
    demux.process(data_frame(&ts), &mut sink).await.unwrap();

    let mut posted = Vec::new();
    while let Some(msg) = bus.try_recv() {
        if let BusMessage::Tag { tags, program } = msg {
            assert_eq!(program, Some(7), "the events post on their own program");
            for tag in tags.tags() {
                if let Tag::Other { key, value } = tag {
                    posted.push((key.clone(), value.clone()));
                }
            }
        }
    }
    let has = |key: &str, value: &str| posted.iter().any(|(k, v)| k == key && v == value);
    assert!(has(TAG_KEY_EVENT_NAME, "Now"), "posted: {posted:?}");
    assert!(has(TAG_KEY_EVENT_TEXT, "On air"), "posted: {posted:?}");
    assert!(has(TAG_KEY_NEXT_EVENT_NAME, "Next"), "posted: {posted:?}");
    assert!(
        has(TAG_KEY_NEXT_EVENT_TEXT, "After this"),
        "posted: {posted:?}"
    );

    // Feeding the same tables again must not re-post.
    demux.process(data_frame(&ts), &mut sink).await.unwrap();
    assert!(
        bus.try_recv().is_none(),
        "an unchanged table version does not re-post"
    );
}
