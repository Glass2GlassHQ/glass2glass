//! Real-peer SRT interop (M522): an `ffmpeg` SRT *caller* (libsrt under the
//! hood) publishes a short MPEG-TS stream into the g2g `SrtSrc` *listener*. This
//! is the interop the g2g<->g2g loopback (`srt_loopback`) cannot prove: that the
//! g2g HSv5 induction/conclusion handshake and the SRT data framing actually
//! match the reference libsrt implementation on the wire.
//!
//! Ignored by default (needs an ffmpeg built with libsrt, and a local UDP
//! socket). Run it explicitly:
//!
//! ```sh
//! cargo test -p g2g-plugins --features srt --test srt_ffmpeg_interop \
//!     -- --ignored --nocapture
//! ```
#![cfg(feature = "srt")]

use core::future::Future;
use core::pin::Pin;
use std::net::UdpSocket as StdUdpSocket;
use std::time::Duration;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome,
};

use g2g_plugins::srtsink::SrtSink;
use g2g_plugins::srtsrc::SrtSrc;

use std::path::Path;
use std::process::Command;

/// Records the first byte of each received SRT payload (MPEG-TS packets start
/// with the 0x47 sync byte) and a running total of payload bytes.
#[derive(Default)]
struct Capture {
    first_bytes: Vec<u8>,
    total_bytes: usize,
}

impl OutputSink for Capture {
    fn push<'a>(
        &'a mut self,
        p: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        if let PipelinePacket::DataFrame(frame) = &p {
            if let Some(slice) = frame.domain.as_system_slice() {
                let s = slice;
                self.first_bytes.push(s.first().copied().unwrap_or(0));
                self.total_bytes += s.len();
            }
        }
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

/// Number of SRT payloads to collect before declaring success.
const N: u64 = 20;

/// Drive an `ffmpeg` SRT caller into a g2g `SrtSrc` listener and return the
/// first byte of each received payload. `crypto`, when set, is
/// `(passphrase, pbkeylen)` and enables AES on both sides (the ffmpeg URL query
/// + `SrtSrc::with_passphrase`).
async fn run_interop(crypto: Option<(&str, u32)>) -> Vec<u8> {
    // Bind the listener socket up front so we know the port before ffmpeg calls;
    // the kernel buffers ffmpeg's induction datagrams until SrtSrc reads them.
    let sock = StdUdpSocket::bind("127.0.0.1:0").expect("bind srt listener");
    let port = sock.local_addr().unwrap().port();
    let mut src = SrtSrc::from_socket(sock)
        .expect("adopt socket")
        .with_frame_limit(N);
    if let Some((passphrase, _)) = crypto {
        src = src.with_passphrase(passphrase);
    }
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .expect("configure");

    // ffmpeg pushes ~3 s of synthetic MPEG-TS as an SRT caller to our listener.
    let mut url = format!("srt://127.0.0.1:{port}");
    if let Some((passphrase, pbkeylen)) = crypto {
        url.push_str(&format!("?passphrase={passphrase}&pbkeylen={pbkeylen}"));
    }
    let ffmpeg = tokio::task::spawn_blocking(move || {
        std::process::Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-re",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=15:duration=3",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-f",
                "mpegts",
                &url,
            ])
            .status()
    });

    let mut cap = Capture::default();
    let received = tokio::time::timeout(Duration::from_secs(20), src.run(&mut cap))
        .await
        .expect("SrtSrc receives N payloads within 20s")
        .expect("SrtSrc runs");
    let _ = ffmpeg.await;

    assert_eq!(received, N, "received the requested number of SRT payloads");
    assert!(cap.total_bytes > 0, "payloads carried data");

    // A reference libsrt caller (ffmpeg) completed the HSv5 handshake and its data
    // framing was received intact: persist peer-tagged `Oracle` evidence so
    // `--maturity` derives srtsrc as InteropTested. The encrypted variants also
    // prove the KM exchange / AES cipher match libsrt.
    let detail: String = match crypto {
        None => "ffmpeg (libsrt) caller: HSv5 handshake + data framing".into(),
        Some((_, pbkeylen)) => {
            format!(
                "ffmpeg (libsrt) caller: AES-{}-encrypted KM exchange + payload",
                pbkeylen * 8
            )
        }
    };
    g2g_plugins::conformance::persist::record_evidence(
        "srtsrc",
        &g2g_core::conformance::Evidence::new(g2g_core::conformance::ConformanceDimension::Oracle)
            .peer("ffmpeg (libsrt)")
            .codec("mpegts")
            .detail(detail),
    )
    .expect("record oracle evidence");

    cap.first_bytes
}

/// Assert every received payload begins on an MPEG-TS packet boundary (0x47
/// sync). A handshake or (for the encrypted case) a key-material / cipher
/// mismatch with libsrt would corrupt this.
fn assert_all_ts_sync(first_bytes: &[u8]) {
    let sync = first_bytes.iter().filter(|&&b| b == 0x47).count();
    assert_eq!(
        sync,
        first_bytes.len(),
        "every payload starts with the MPEG-TS 0x47 sync byte (got firsts {first_bytes:?})"
    );
}

#[tokio::test]
#[ignore = "needs ffmpeg built with libsrt; opens a local UDP socket"]
async fn ffmpeg_srt_caller_streams_mpegts_into_srtsrc() {
    assert_all_ts_sync(&run_interop(None).await);
}

/// Encrypted interop: an ffmpeg caller with an AES-128 passphrase into a g2g
/// listener with the matching passphrase. Proves the SRT KM exchange + AES-CTR
/// payload cipher match libsrt on the wire (the handshake `encryption` field,
/// KM message layout, and counter-block construction), not just g2g<->g2g.
#[tokio::test]
#[ignore = "needs ffmpeg built with libsrt; opens a local UDP socket"]
async fn ffmpeg_srt_encrypted_caller_streams_mpegts_into_srtsrc() {
    assert_all_ts_sync(&run_interop(Some(("g2g_srt_interop_passphrase", 16))).await);
}

/// AES-256 variant (pbkeylen=32), exercising the larger wrapped key.
#[tokio::test]
#[ignore = "needs ffmpeg built with libsrt; opens a local UDP socket"]
async fn ffmpeg_srt_aes256_caller_streams_mpegts_into_srtsrc() {
    assert_all_ts_sync(&run_interop(Some(("g2g_srt_interop_passphrase", 32))).await);
}

/// A do-nothing sink so `SrtSink` (a terminal egress element) can be driven.
struct NullOut;
impl OutputSink for NullOut {
    fn push<'a>(
        &'a mut self,
        _p: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

/// A grab an ephemeral UDP port, then release it so ffmpeg's listener can bind it.
fn free_udp_port() -> u16 {
    StdUdpSocket::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn ffmpeg_generate_ts(path: &Path) {
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=15:duration=2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-f",
            "mpegts",
            path.to_str().unwrap(),
        ])
        .status()
        .expect("spawn ffmpeg to generate the input TS");
    assert!(
        status.success(),
        "ffmpeg generated the input MPEG-TS fixture"
    );
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
    // ffprobe may print a line per stream; take the first that parses as a count.
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// The reverse interop direction: g2g `SrtSink` (caller) publishes an MPEG-TS
/// stream to an ffmpeg SRT *listener* (libsrt), which remuxes it to a file. This
/// exercises the g2g caller-side HSv5 handshake + data framing against the
/// reference implementation (the mirror of the caller->g2g-listener tests above),
/// and asserts ffmpeg decoded the same number of video frames the fixture had, so
/// the whole stream survived g2g's SRT sender -> libsrt intact.
/// Drive the reverse direction: g2g `SrtSink` (caller) streams a generated
/// MPEG-TS to an ffmpeg SRT *listener* (libsrt), which remuxes it to a file.
/// `crypto`, when set, is `(passphrase, pbkeylen)` enabling AES on both sides.
/// Asserts ffmpeg decoded ~the same video frame count the fixture had, so the
/// stream survived g2g's SRT sender -> libsrt intact.
async fn run_reverse_interop(crypto: Option<(&str, u32)>) {
    let port = free_udp_port();
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let in_ts = dir.join(format!("g2g_srt_in_{pid}_{port}.ts"));
    let out_ts = dir.join(format!("g2g_srt_out_{pid}_{port}.ts"));

    ffmpeg_generate_ts(&in_ts);
    let ts_bytes = std::fs::read(&in_ts).expect("read the input TS fixture");
    assert!(!ts_bytes.is_empty(), "fixture has data");
    let want_frames = ffprobe_video_frame_count(&in_ts);
    assert!(want_frames > 0, "fixture has decodable video frames");

    // ffmpeg SRT listener remuxes whatever the caller sends to out_ts, finishing
    // when the caller's SRT Shutdown (sent on EOS) closes the connection.
    let mut url = format!("srt://127.0.0.1:{port}?mode=listener");
    if let Some((passphrase, pbkeylen)) = crypto {
        url.push_str(&format!("&passphrase={passphrase}&pbkeylen={pbkeylen}"));
    }
    let out_path = out_ts.clone();
    let ffmpeg = tokio::task::spawn_blocking(move || {
        Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                &url,
                "-c",
                "copy",
                "-f",
                "mpegts",
                out_path.to_str().unwrap(),
            ])
            .status()
    });
    // Let the listener bind its UDP port before the caller (no connect retry).
    tokio::time::sleep(Duration::from_millis(1500)).await;

    // Drive the g2g SRT caller: feed the TS in SRT-payload-sized chunks, then EOS.
    let mut sink = SrtSink::new(([127, 0, 0, 1], port).into());
    if let Some((passphrase, pbkeylen)) = crypto {
        sink = sink.with_passphrase(passphrase);
        if pbkeylen == 32 {
            sink = sink.with_aes256();
        }
    }
    sink.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .expect("configure SrtSink");
    let mut null = NullOut;
    for (i, chunk) in ts_bytes.chunks(1316).enumerate() {
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(
                chunk.to_vec().into_boxed_slice(),
            )),
            timing: FrameTiming {
                pts_ns: i as u64 * 1_000_000,
                ..FrameTiming::default()
            },
            sequence: i as u64,
            meta: Default::default(),
        };
        sink.process(PipelinePacket::DataFrame(frame), &mut null)
            .await
            .expect("send TS chunk");
        // Pace lightly so the SRT send buffer does not overrun in live mode.
        if i % 8 == 0 {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }
    sink.process(PipelinePacket::Eos, &mut null)
        .await
        .expect("EOS closes the SRT session");

    let status = tokio::time::timeout(Duration::from_secs(20), ffmpeg)
        .await
        .expect("ffmpeg listener finishes within 20s")
        .expect("join ffmpeg task")
        .expect("ffmpeg status");
    assert!(
        status.success(),
        "ffmpeg listener received + finalized the stream"
    );

    let got_frames = ffprobe_video_frame_count(&out_ts);
    let _ = std::fs::remove_file(&in_ts);
    let _ = std::fs::remove_file(&out_ts);
    // Allow a small tail loss (last partial payload); the bulk must arrive intact.
    assert!(
        got_frames + 2 >= want_frames && got_frames > 0,
        "ffmpeg decoded {got_frames} frames from the g2g SRT caller (fixture had {want_frames})"
    );

    // g2g's caller-side handshake + framing (and, when encrypted, its KMREQ) were
    // accepted by a reference libsrt listener: persist `Oracle` evidence for srtsink.
    let detail: String = match crypto {
        None => "g2g caller -> ffmpeg (libsrt) listener: HSv5 handshake + data framing".into(),
        Some((_, pbkeylen)) => {
            format!(
                "g2g caller -> ffmpeg (libsrt): AES-{}-encrypted KMREQ + payload",
                pbkeylen * 8
            )
        }
    };
    g2g_plugins::conformance::persist::record_evidence(
        "srtsink",
        &g2g_core::conformance::Evidence::new(g2g_core::conformance::ConformanceDimension::Oracle)
            .peer("ffmpeg (libsrt)")
            .codec("mpegts")
            .detail(detail),
    )
    .expect("record oracle evidence");
}

/// Reverse direction, plaintext.
#[tokio::test]
#[ignore = "needs ffmpeg+ffprobe with libsrt; opens a local UDP socket + temp files"]
async fn srtsink_caller_streams_mpegts_to_ffmpeg_listener() {
    run_reverse_interop(None).await;
}

/// Reverse direction, AES-128: the g2g caller offers a KMREQ to the libsrt
/// listener (the mirror of the forward encrypted test, exercising the caller-side
/// key-material path against the reference implementation).
#[tokio::test]
#[ignore = "needs ffmpeg+ffprobe with libsrt; opens a local UDP socket + temp files"]
async fn srtsink_encrypted_caller_streams_mpegts_to_ffmpeg_listener() {
    run_reverse_interop(Some(("g2g_srt_interop_passphrase", 16))).await;
}

/// Reverse direction, AES-256 (the larger wrapped key over the caller KM path).
#[tokio::test]
#[ignore = "needs ffmpeg+ffprobe with libsrt; opens a local UDP socket + temp files"]
async fn srtsink_aes256_caller_streams_mpegts_to_ffmpeg_listener() {
    run_reverse_interop(Some(("g2g_srt_interop_passphrase", 32))).await;
}
