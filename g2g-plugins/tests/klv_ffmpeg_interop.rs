//! KLV-in-TS interop against ffmpeg as the reference peer (M799/M800): ffmpeg
//! must identify the g2g-muxed metadata stream as KLV (proving the PMT's 'KLVA'
//! registration parses), extract the packets bit-exact, and a TS re-authored by
//! ffmpeg's own muxer must demux back through g2g bit-exact. Skips (with a note)
//! when ffmpeg / ffprobe are not installed.

use std::pin::Pin;
use std::process::Command;

use core::future::Future;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    ByteStreamEncoding, Caps, ConfigureOutcome, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome,
};

use g2g_core::element::AsyncElement;
use g2g_plugins::klv::UasDatalink;
use g2g_plugins::mpegts::{TsMuxer, STREAM_TYPE_H264, STREAM_TYPE_PRIVATE_PES};
use g2g_plugins::tsdemux::{TsDemux, TsStream};

fn have(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn temp_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("g2g-klv-{tag}-{}.bin", std::process::id()))
}

fn telemetry(i: u64) -> UasDatalink {
    UasDatalink {
        timestamp_us: Some(1_700_000_000_000_000 + i * 40_000),
        sensor_lat_deg: Some(60.1768),
        sensor_lon_deg: Some(24.8288 + i as f64 * 0.0001),
        sensor_alt_m: Some(145.0),
        version: Some(19),
        ..Default::default()
    }
}

/// Mux 30 video AUs + 30 KLV local sets into one TS with the pure muxer (a
/// stream long enough for ffprobe's probing) and return (ts, klv_packets).
fn muxed_ts() -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut mux = TsMuxer::with_streams(&[STREAM_TYPE_H264, STREAM_TYPE_PRIVATE_PES]);
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
    (ts, packets)
}

/// An `OutputSink` recording each recovered frame's bytes.
#[derive(Default)]
struct CaptureSink {
    aus: Vec<Vec<u8>>,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.aus.push(s.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

async fn g2g_demux_klv(ts: &[u8]) -> Vec<Vec<u8>> {
    let mut demux = TsDemux::new().with_stream(TsStream::Klv);
    assert!(matches!(
        demux.configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs
        }),
        Ok(ConfigureOutcome::Accepted)
    ));
    let mut sink = CaptureSink::default();
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

/// ffprobe sees the g2g mux's metadata stream as codec `klv`, and ffmpeg
/// extracts the packet bytes g2g wrote, bit-exact.
#[tokio::test]
async fn ffmpeg_reads_g2g_klv_stream() {
    if !have("ffprobe") || !have("ffmpeg") {
        eprintln!("skipping: no ffmpeg/ffprobe");
        return;
    }
    let (ts, packets) = muxed_ts();
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
        "ffprobe identifies the KLV data stream (KLVA registration): {json}"
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
    let want: Vec<u8> = packets.concat();
    assert_eq!(extracted, want, "KLV bytes bit-exact through ffmpeg");

    let _ = std::fs::remove_file(&ts_path);
    let _ = std::fs::remove_file(&out_path);
}

/// A TS re-authored by ffmpeg's muxer (from the g2g mux, `-c copy`) demuxes
/// back through g2g bit-exact and parses to the original telemetry: the
/// receive side is validated against a stream a third-party muxer wrote.
#[tokio::test]
async fn g2g_demuxes_ffmpeg_authored_klv_ts() {
    if !have("ffmpeg") {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let (ts, packets) = muxed_ts();
    let in_path = temp_path("in");
    let remux_path = temp_path("remux");
    std::fs::write(&in_path, &ts).unwrap();
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "fatal", "-i"])
        .arg(&in_path)
        .args(["-map", "0", "-c", "copy", "-f", "mpegts"])
        .arg(&remux_path)
        .status()
        .unwrap();
    assert!(status.success(), "ffmpeg remuxes the TS");
    let remuxed = std::fs::read(&remux_path).unwrap();

    let got = g2g_demux_klv(&remuxed).await;
    assert_eq!(
        got.len(),
        packets.len(),
        "every KLV packet survives the ffmpeg remux"
    );
    assert_eq!(
        got, packets,
        "KLV bytes bit-exact from the ffmpeg-written TS"
    );
    for (i, pkt) in got.iter().enumerate() {
        let ls = UasDatalink::parse(pkt).expect("valid ST 0601 set");
        assert_eq!(ls.timestamp_us, telemetry(i as u64).timestamp_us);
    }

    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&remux_path);
}
