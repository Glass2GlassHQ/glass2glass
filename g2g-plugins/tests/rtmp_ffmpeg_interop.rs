//! Real-peer RTMP interop (M527): ffmpeg publishes an RTMP stream
//! (`-f flv rtmp://...`, ffmpeg's native RTMP client) into the g2g `RtmpSrc`
//! *listener*, which demuxes it to an FLV byte stream. This proves the g2g RTMP
//! server handshake (auto-detecting ffmpeg's simple or digest C1), chunk-stream
//! reassembly, and AMF0 publish flow match a reference RTMP client on the wire,
//! which the in-tree `RtmpPublisher <-> RtmpSession` loopback cannot (both sides
//! are g2g).
//!
//! Ignored by default (needs ffmpeg + ffprobe, opens a local TCP port). Run:
//!
//! ```sh
//! cargo test -p g2g-plugins --features rtmp --test rtmp_ffmpeg_interop \
//!     -- --ignored --nocapture
//! ```
#![cfg(feature = "rtmp")]

use core::future::Future;
use core::pin::Pin;
use std::net::TcpListener as StdTcpListener;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, G2gError, OutputSink, PipelinePacket, PushOutcome};

use g2g_plugins::rtmpsrc::RtmpSrc;

/// Concatenates every FLV byte-stream `DataFrame` the source emits.
#[derive(Default)]
struct FlvCollect {
    bytes: Vec<u8>,
}

impl OutputSink for FlvCollect {
    fn push<'a>(
        &'a mut self,
        p: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        if let PipelinePacket::DataFrame(frame) = &p {
            if let Some(slice) = frame.domain.as_system_slice() {
                self.bytes.extend_from_slice(slice);
            }
        }
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

fn ffprobe_video_frame_count(path: &Path) -> u64 {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            path.to_str().unwrap(),
        ])
        .output()
        .expect("spawn ffprobe");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

#[tokio::test]
#[ignore = "needs ffmpeg+ffprobe; opens a local TCP port + a temp file"]
async fn ffmpeg_rtmp_publisher_into_rtmpsrc() {
    // Bind the RTMP listener up front so ffmpeg's TCP connect (queued in the
    // listen backlog) succeeds before RtmpSrc calls accept().
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtmp listener");
    let port = listener.local_addr().unwrap().port();
    let mut src = RtmpSrc::from_listener(listener).expect("adopt listener");
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Flv,
    })
    .expect("configure RtmpSrc");

    // ffmpeg publishes ~2 s of synthetic H.264 over RTMP to our listener.
    let url = format!("rtmp://127.0.0.1:{port}/live/stream");
    let ffmpeg = tokio::task::spawn_blocking(move || {
        Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-re",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=15:duration=2",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-f",
                "flv",
                &url,
            ])
            .status()
    });

    // RtmpSrc accepts the publisher, drives the handshake + publish, and streams
    // the demuxed FLV until ffmpeg closes the connection (then EOS).
    let mut sink = FlvCollect::default();
    let emitted = tokio::time::timeout(Duration::from_secs(20), src.run(&mut sink))
        .await
        .expect("RtmpSrc completes within 20s")
        .expect("RtmpSrc runs");
    let _ = ffmpeg.await;

    assert!(
        emitted > 0,
        "RtmpSrc emitted FLV frames from the ffmpeg publisher"
    );
    assert!(sink.bytes.len() > 3, "collected an FLV byte stream");
    // The demuxed stream is a proper FLV file: the "FLV" signature + version.
    assert_eq!(
        &sink.bytes[0..3],
        b"FLV",
        "output starts with the FLV signature"
    );

    // ffprobe confirms the demuxed FLV decodes back to video frames intact.
    let out_flv = std::env::temp_dir().join(format!("g2g_rtmp_{}_{port}.flv", std::process::id()));
    std::fs::write(&out_flv, &sink.bytes).expect("write collected FLV");
    let frames = ffprobe_video_frame_count(&out_flv);
    let _ = std::fs::remove_file(&out_flv);
    assert!(
        frames > 0,
        "ffprobe decoded {frames} video frames from the RTMP-ingested FLV stream"
    );

    // g2g's RTMP server handshake (ffmpeg's native client) + chunk-stream reassembly
    // produced a valid FLV a reference demuxer decoded: persist `Oracle` evidence.
    use g2g_core::conformance::{ConformanceDimension, Evidence};
    g2g_plugins::conformance::persist::record_evidence(
        "rtmpsrc",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("h264")
            .detail(
                "ffmpeg RTMP publisher: handshake + chunk-stream demuxed to FLV, ffprobe-decoded",
            ),
    )
    .expect("record oracle evidence");
}
