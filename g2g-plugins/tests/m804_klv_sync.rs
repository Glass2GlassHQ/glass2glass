//! M804 - synchronous KLV in MPEG-TS. With `klv-sync`, a `Caps::Klv` input is
//! muxed as metadata-in-PES (`stream_type` 0x15, PES `stream_id` 0xFC, one
//! ISO 13818-1 metadata AU cell per access unit) instead of the asynchronous
//! private PES; the demux recovers it bit-exact, and forwards a payload that is
//! not cell-wrapped unchanged. ffmpeg is the reference peer for the wire format
//! (skipped, with a note, when it is not installed).

use std::pin::Pin;
use std::process::Command;

use core::future::Future;

use g2g_core::element::{AsyncElement, BoxFuture, PushOutcome};
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_muxer_sink, DynSourceLoop, SourceLoop};
use g2g_core::{
    ByteStreamEncoding, Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain,
    MultiInputElement, OutputSink, PipelineClock, PipelinePacket, PropValue, Rate, VideoCodec,
};

use g2g_plugins::klv::UasDatalink;
use g2g_plugins::mpegts::{
    TsDemuxer, TsMuxer, STREAM_TYPE_H264, STREAM_TYPE_METADATA_PES, TS_PACKET_LEN,
};
use g2g_plugins::tsdemux::{TsDemux, TsStream};
use g2g_plugins::tsmuxn::TsMux;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn have(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn temp_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("g2g-klvsync-{tag}-{}.bin", std::process::id()))
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn ts_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    }
}

fn telemetry(i: u64) -> UasDatalink {
    UasDatalink {
        timestamp_us: Some(1_700_000_000_000_000 + i * 40_000),
        sensor_lat_deg: Some(60.1768 + i as f64 * 0.0001),
        sensor_lon_deg: Some(24.8288),
        sensor_alt_m: Some(145.0),
        heading_deg: Some(87.3),
        version: Some(19),
        ..Default::default()
    }
}

/// Wrap `payload` in one ISO 13818-1 metadata AU cell: metadata_service_id,
/// sequence_number, flags, then a big-endian 16-bit AU_cell_data_length.
fn au_cell(seq: u8, payload: &[u8]) -> Vec<u8> {
    let mut cell = vec![0x01, seq, 0x00];
    cell.push((payload.len() >> 8) as u8);
    cell.push(payload.len() as u8);
    cell.extend_from_slice(payload);
    cell
}

/// Emits a script of (access-unit, pts_ns) for one elementary stream, then EOS.
struct AuSrc {
    caps: Caps,
    aus: Vec<(Vec<u8>, u64)>,
    configured: bool,
}

impl SourceLoop for AuSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps.clone()))
    }
    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        let aus = self.aus.clone();
        let configured = self.configured;
        Box::pin(async move {
            assert!(configured, "runner configures before run");
            for (i, (au, pts)) in aus.iter().enumerate() {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(au.clone().into_boxed_slice())),
                    FrameTiming {
                        pts_ns: *pts,
                        ..FrameTiming::default()
                    },
                    i as u64,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
                tokio::task::yield_now().await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(aus.len() as u64)
        })
    }
}

/// Collects the muxed TS byte frames.
#[derive(Default)]
struct CaptureTs {
    bytes: Vec<u8>,
}
impl AsyncElement for CaptureTs {
    type ProcessFuture<'a> = BoxFuture<'a, Result<(), G2gError>>;
    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let PipelinePacket::DataFrame(f) = packet {
            if let Some(s) = f.domain.as_system_slice() {
                self.bytes.extend_from_slice(s);
            }
        }
        Box::pin(async { Ok(()) })
    }
}

/// An `OutputSink` recording each recovered frame's bytes and PTS.
#[derive(Default)]
struct CaptureAus {
    aus: Vec<(Vec<u8>, u64)>,
}
impl OutputSink for CaptureAus {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.aus.push((s.to_vec(), f.timing.pts_ns));
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Drive a whole TS byte buffer through a single-output `TsDemux` selecting
/// `stream`, returning the recovered (bytes, pts_ns) pairs.
async fn demux_stream(ts: &[u8], stream: TsStream) -> Vec<(Vec<u8>, u64)> {
    let mut demux = TsDemux::new().with_stream(stream);
    demux.configure_pipeline(&ts_caps()).unwrap();
    let mut sink = CaptureAus::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(ts.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    demux
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .unwrap();
    demux.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    sink.aus
}

/// The `stream_type`s the PMT of `ts` names, read back with the pure demuxer.
fn pmt_stream_types(ts: &[u8]) -> Vec<u8> {
    let mut d = TsDemuxer::new();
    for pkt in ts.chunks(TS_PACKET_LEN) {
        d.push_packet(pkt);
    }
    d.streams().iter().map(|s| s.stream_type).collect()
}

/// A TS whose single 0x15 stream carries `payload` as one PES payload verbatim:
/// the muxer's own PAT/PMT (the first two packets it writes) followed by a
/// hand-built PES, so the test controls the exact bytes a foreign writer put on
/// the wire.
fn raw_metadata_pes_ts(payload: &[u8]) -> Vec<u8> {
    let mut mux = TsMuxer::with_streams(&[STREAM_TYPE_METADATA_PES]);
    let tables = mux.push_au(&[0xAA], Some(0), None);
    let mut ts = tables[..2 * TS_PACKET_LEN].to_vec();
    let mut probe = TsDemuxer::new();
    for pkt in ts.chunks(TS_PACKET_LEN) {
        probe.push_packet(pkt);
    }
    let pid = probe.streams()[0].pid;

    let mut pes = vec![0x00, 0x00, 0x01, 0xFC];
    let pes_len = payload.len() + 3;
    pes.push((pes_len >> 8) as u8);
    pes.push(pes_len as u8);
    pes.extend_from_slice(&[0x80, 0x00, 0x00]); // marker, no PTS, no header data
    pes.extend_from_slice(payload);
    assert!(pes.len() < 183, "these payloads fit one TS packet");

    let af_len = 183 - pes.len();
    let mut pkt = vec![
        0x47,
        0x40 | (pid >> 8) as u8,
        pid as u8,
        0x30, // adaptation field + payload, continuity counter 0
        af_len as u8,
        0x00, // adaptation flags
    ];
    pkt.resize(6 + af_len - 1, 0xFF); // stuffing
    pkt.extend_from_slice(&pes);
    assert_eq!(pkt.len(), TS_PACKET_LEN);
    ts.extend_from_slice(&pkt);
    ts
}

/// `klv-sync` on the fan-in muxer puts the KLV on a metadata-in-PES stream: the
/// PMT names 0x15 with a metadata descriptor and no 'KLVA' registration, and both
/// the video and the KLV come back out of the demuxer bit-exact with their
/// timestamps.
#[tokio::test]
async fn klv_sync_mux_writes_metadata_pes_and_round_trips() {
    let video_aus = vec![
        (vec![0u8, 0, 0, 1, 0x65, 0x11], 0),
        (vec![0u8, 0, 0, 1, 0x41, 0x22], 40_000_000),
        (vec![0u8, 0, 0, 1, 0x41, 0x33], 80_000_000),
    ];
    let klv_packets: Vec<Vec<u8>> = (0..2).map(|i| telemetry(i).encode()).collect();
    let klv_aus: Vec<(Vec<u8>, u64)> = klv_packets
        .iter()
        .enumerate()
        .map(|(i, p)| (p.clone(), 20_000_000 + i as u64 * 40_000_000))
        .collect();

    let mut video = AuSrc {
        caps: h264_caps(),
        aus: video_aus,
        configured: false,
    };
    let mut klv = AuSrc {
        caps: Caps::Klv,
        aus: klv_aus,
        configured: false,
    };
    let mut mux = TsMux::new(2); // input 0 = video, input 1 = klv
                                 // Set through the property (the `parse_launch` path), not the builder.
    mux.set_property("klv-sync", PropValue::Bool(true)).unwrap();
    assert_eq!(mux.get_property("klv-sync"), Some(PropValue::Bool(true)));
    let mut sink = CaptureTs::default();
    {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut video, &mut klv];
        run_muxer_sink(sources, &mut mux, &mut sink, &ZeroClock, 4)
            .await
            .expect("video + sync klv mux pipeline completes");
    }
    let ts = sink.bytes;

    assert_eq!(
        pmt_stream_types(&ts),
        vec![STREAM_TYPE_H264, STREAM_TYPE_METADATA_PES],
        "the KLV stream is announced as metadata-in-PES"
    );
    assert!(
        !ts.windows(6)
            .any(|w| w == [0x05, 4, b'K', b'L', b'V', b'A']),
        "no KLVA registration descriptor on the synchronous path"
    );
    // Instead the PMT entry carries a metadata_descriptor (tag 0x26) naming
    // 'KLVA', which is what a third-party demuxer keys on for 0x15.
    assert!(
        ts.windows(9)
            .any(|w| w == [0x26, 13, 0xFF, 0xFF, b'K', b'L', b'V', b'A', 0xFF]),
        "PMT carries the KLV metadata_descriptor"
    );

    let got_video = demux_stream(&ts, TsStream::H264).await;
    assert_eq!(got_video.len(), 3, "three video AUs recovered");
    assert_eq!(got_video[1].1, 40_000_000, "video PTS preserved");

    let got_klv = demux_stream(&ts, TsStream::Klv).await;
    let want: Vec<&[u8]> = klv_packets.iter().map(|p| &p[..]).collect();
    let got: Vec<&[u8]> = got_klv.iter().map(|(p, _)| &p[..]).collect();
    assert_eq!(got, want, "KLV packets recovered bit-exact");
    assert_eq!(got_klv[0].1, 20_000_000, "sync KLV keeps its PTS");
    assert_eq!(got_klv[1].1, 60_000_000);
    for (i, (pkt, _)) in got_klv.iter().enumerate() {
        let ls = UasDatalink::parse(pkt).expect("valid ST 0601 set");
        assert_eq!(ls.timestamp_us, telemetry(i as u64).timestamp_us);
    }

    // The builder is the same knob as the property.
    assert_eq!(
        TsMux::new(2).with_klv_sync(true).get_property("klv-sync"),
        Some(PropValue::Bool(true))
    );
}

/// ffmpeg is the reference peer for the wire format: ffprobe must call the
/// metadata-in-PES stream `klv`, and ffmpeg must extract its payload bytes
/// bit-exact (which is also what pins the PES `stream_id` choice).
#[tokio::test]
async fn ffmpeg_reads_the_sync_klv_stream() {
    if !have("ffprobe") || !have("ffmpeg") {
        eprintln!("skipping: no ffmpeg/ffprobe");
        return;
    }
    // A stream long enough for ffprobe's probing: 30 video AUs + 30 local sets.
    let mut mux = TsMuxer::with_streams(&[STREAM_TYPE_H264, STREAM_TYPE_METADATA_PES]);
    let mut ts = Vec::new();
    let mut packets = Vec::new();
    for i in 0..30u64 {
        let pts = i * 3600; // 40 ms in 90 kHz ticks
        let au = [0u8, 0, 0, 1, if i == 0 { 0x65 } else { 0x41 }, i as u8];
        ts.extend_from_slice(&mux.push_au_on(0, &au, Some(pts), None));
        let klv = telemetry(i).encode();
        ts.extend_from_slice(&mux.push_au_on(1, &klv, Some(pts), None));
        packets.push(klv);
    }

    let ts_path = temp_path("mux");
    std::fs::write(&ts_path, &ts).unwrap();

    let probe = Command::new("ffprobe")
        .args(["-v", "fatal", "-show_streams", "-of", "json"])
        .arg(&ts_path)
        .output()
        .unwrap();
    let json = String::from_utf8_lossy(&probe.stdout).into_owned();
    assert!(
        json.contains("\"codec_name\": \"klv\""),
        "ffprobe identifies the metadata-in-PES stream as KLV: {json}"
    );

    let out_path = temp_path("extract");
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "fatal", "-i"])
        .arg(&ts_path)
        .args(["-map", "0:d:0", "-c", "copy", "-f", "data"])
        .arg(&out_path)
        .status()
        .unwrap();
    assert!(status.success(), "ffmpeg extracts the data stream");
    let extracted = std::fs::read(&out_path).unwrap();
    assert_eq!(
        extracted,
        packets.concat(),
        "KLV bytes bit-exact through ffmpeg"
    );

    let _ = std::fs::remove_file(&ts_path);
    let _ = std::fs::remove_file(&out_path);
}

/// A spec-strict writer wraps each KLV packet in a metadata AU cell; the demux
/// unwraps them and emits the KLV the cells carry.
#[tokio::test]
async fn au_cell_wrapped_payload_unwraps_to_the_klv_packets() {
    let a = telemetry(0).encode();
    let b = telemetry(1).encode();
    let mut wrapped = au_cell(0, &a);
    wrapped.extend_from_slice(&au_cell(1, &b));
    let ts = raw_metadata_pes_ts(&wrapped);

    let got = demux_stream(&ts, TsStream::Klv).await;
    assert_eq!(got.len(), 1, "one PES in, one frame out");
    let mut want = a.clone();
    want.extend_from_slice(&b);
    assert_eq!(got[0].0, want, "the cell headers are stripped");
    assert_eq!(
        UasDatalink::parse(&got[0].0[..a.len()])
            .expect("first set intact")
            .timestamp_us,
        telemetry(0).timestamp_us
    );
}

/// A 0x15 payload that is neither bare KLV nor a valid cell run is forwarded
/// unchanged: a wrong guess about the cell layout can never eat the payload.
#[tokio::test]
async fn garbage_metadata_payload_is_forwarded_raw() {
    let garbage = vec![0xDEu8, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03];
    let ts = raw_metadata_pes_ts(&garbage);
    let got = demux_stream(&ts, TsStream::Klv).await;
    assert_eq!(
        got.iter().map(|(d, _)| d.clone()).collect::<Vec<_>>(),
        vec![garbage],
        "unrecognized payload passes through byte for byte"
    );
}
