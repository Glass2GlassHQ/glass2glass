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

use std::net::UdpSocket as StdUdpSocket;
use std::sync::Arc;
use std::time::Duration;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome,
};

use g2g_plugins::srt;
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
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let p = packet_slot.take().expect("poll_push without a packet");

        if let PipelinePacket::DataFrame(frame) = &p {
            if let Some(slice) = frame.domain.as_system_slice() {
                let s = slice;
                self.first_bytes.push(s.first().copied().unwrap_or(0));
                self.total_bytes += s.len();
            }
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
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
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
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

/// The `streamid` property reaches libsrt as the caller wrote it: libsrt reads
/// each 32-bit word of the id byte-reversed, the way mediamtx routes on it.
#[tokio::test]
#[ignore = "needs ffmpeg with libsrt; opens a local UDP socket"]
async fn srtsink_streamid_reaches_the_ffmpeg_listener() {
    const STREAM_ID: &str = "publish:egress-srt";
    let port = free_udp_port();
    let url = format!("srt://127.0.0.1:{port}?mode=listener");
    let ffmpeg = tokio::task::spawn_blocking(move || {
        Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "debug",
                "-i",
                &url,
                "-c",
                "copy",
                "-f",
                "null",
                "-",
            ])
            .output()
    });
    tokio::time::sleep(Duration::from_millis(1500)).await;

    let mut sink = SrtSink::new(([127, 0, 0, 1], port).into());
    sink.set_property("streamid", g2g_core::PropValue::Str(STREAM_ID.into()))
        .expect("set streamid");
    sink.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .expect("configure SrtSink");
    let path = std::env::temp_dir().join(format!("g2g_srt_sid_{}_{port}.ts", std::process::id()));
    ffmpeg_generate_ts(&path);
    let ts_bytes = std::fs::read(&path).expect("read the input TS fixture");
    let _ = std::fs::remove_file(&path);
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
    }
    sink.process(PipelinePacket::Eos, &mut null)
        .await
        .expect("EOS closes the SRT session");

    let output = tokio::time::timeout(Duration::from_secs(20), ffmpeg)
        .await
        .expect("ffmpeg listener finishes within 20s")
        .expect("join ffmpeg task")
        .expect("ffmpeg output");
    let log = String::from_utf8_lossy(&output.stderr);
    assert!(
        log.contains(&format!("accept streamid [{STREAM_ID}]")),
        "libsrt read the stream id the caller set"
    );
}

const MPEGTS_SYNC_BYTE: u8 = 0x47;

fn spawn_ffmpeg_caller(url: &str) -> std::process::Child {
    Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-re",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=15:duration=60",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-f",
            "mpegts",
            url,
        ])
        .stdin(std::process::Stdio::null())
        .spawn()
        .expect("spawn the ffmpeg SRT caller")
}

#[derive(Debug, PartialEq)]
enum Received {
    Data(u8),
    Flush,
    Eos,
}

// signals `first_payloads` once that many payloads arrived and `flushed` on a Flush
struct ReceivedLog {
    events: Vec<Received>,
    first_payload_count: usize,
    first_payloads: Arc<tokio::sync::Notify>,
    flushed: Arc<tokio::sync::Notify>,
}

impl OutputSink for ReceivedLog {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        match packet_slot.take().expect("poll_push without a packet") {
            PipelinePacket::DataFrame(frame) => {
                let slice = frame.domain.as_system_slice().expect("system frame");
                self.events.push(Received::Data(slice[0]));
                if self.events.len() == self.first_payload_count {
                    self.first_payloads.notify_one();
                }
            }
            PipelinePacket::Flush => {
                self.events.push(Received::Flush);
                self.flushed.notify_one();
            }
            PipelinePacket::Eos => self.events.push(Received::Eos),
            _ => {}
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// The canary's lost-caller case: libsrt's caller dies without a SHUTDOWN, the
/// listener times the silent peer out and accepts the next ffmpeg caller.
#[tokio::test]
#[ignore = "needs ffmpeg built with libsrt; opens a local UDP socket"]
async fn keep_listening_srtsrc_takes_a_new_ffmpeg_caller_after_one_dies() {
    const FIRST_CALLER_PAYLOADS: usize = 20;
    const SECOND_CALLER_PAYLOADS: u64 = 40;

    let sock = StdUdpSocket::bind("127.0.0.1:0").expect("bind srt listener");
    let url = format!("srt://127.0.0.1:{}", sock.local_addr().unwrap().port());
    let mut src = SrtSrc::from_socket(sock)
        .expect("adopt socket")
        .with_frame_limit(FIRST_CALLER_PAYLOADS as u64 + SECOND_CALLER_PAYLOADS);
    src.set_property("keep-listening", g2g_core::PropValue::Bool(true))
        .expect("keep-listening");
    let first_payloads = Arc::new(tokio::sync::Notify::new());
    let flushed = Arc::new(tokio::sync::Notify::new());
    let mut log = ReceivedLog {
        events: Vec::new(),
        first_payload_count: FIRST_CALLER_PAYLOADS,
        first_payloads: first_payloads.clone(),
        flushed: flushed.clone(),
    };

    let callers = async {
        let mut first = spawn_ffmpeg_caller(&url);
        first_payloads.notified().await;
        first.kill().expect("kill the first caller");
        first.wait().expect("reap the first caller");
        flushed.notified().await;
        spawn_ffmpeg_caller(&url)
    };
    // the second caller waits for the Flush, so the timeout covers both
    let (result, mut second) = tokio::time::timeout(Duration::from_secs(40), async {
        tokio::join!(src.run(&mut log), callers)
    })
    .await
    .expect("the second caller streams within the timeout");
    second.kill().expect("kill the second caller");
    second.wait().expect("reap the second caller");

    result.expect("run ok");
    let flush_at = log
        .events
        .iter()
        .position(|event| *event == Received::Flush)
        .expect("a Flush between the callers");
    let (before, after) = log.events.split_at(flush_at);
    assert!(before.len() >= FIRST_CALLER_PAYLOADS);
    assert_eq!(after.last(), Some(&Received::Eos));
    let payloads = before.iter().chain(&after[1..after.len() - 1]);
    assert!(
        payloads
            .clone()
            .all(|event| *event == Received::Data(MPEGTS_SYNC_BYTE)),
        "both callers' payloads start on a TS packet: {:?}",
        log.events
    );
    assert_eq!(
        payloads.count() as u64,
        FIRST_CALLER_PAYLOADS as u64 + SECOND_CALLER_PAYLOADS
    );
}

// drops every copy of the data packet after `skip_packets`, and every drop request
async fn proxy_losing_one_packet_for_good(
    proxy: tokio::net::UdpSocket,
    listener: std::net::SocketAddr,
    skip_packets: usize,
) {
    let mut caller = None;
    let mut data_packets = 0;
    let mut lost_seq = None;
    let mut buf = [0u8; 2048];
    loop {
        let Ok((n, from)) = proxy.recv_from(&mut buf).await else {
            return;
        };
        let packet = &buf[..n];
        if from == listener {
            if let Some(caller) = caller {
                let _ = proxy.send_to(packet, caller).await;
            }
            continue;
        }
        caller = Some(from);
        if let Some(data) = srt::parse_data_packet(packet) {
            if lost_seq.is_none() && !data.retransmit {
                data_packets += 1;
                if data_packets > skip_packets {
                    lost_seq = Some(data.seq);
                }
            }
            if lost_seq == Some(data.seq) {
                continue;
            }
        }
        if matches!(
            srt::parse_control(packet),
            Some(srt::Control::DropRequest { .. })
        ) {
            continue;
        }
        let _ = proxy.send_to(packet, listener).await;
    }
}

/// The canary's capped-link case: libsrt gives up on a packet, and the listener
/// releases the packets after it once they are due instead of waiting forever.
#[tokio::test]
#[ignore = "needs ffmpeg built with libsrt; opens a local UDP socket"]
async fn srtsrc_carries_on_past_a_packet_ffmpeg_never_delivers() {
    const PAYLOADS_BEFORE_LOSS: usize = 10;
    const PAYLOADS: u64 = 60;

    let sock = StdUdpSocket::bind("127.0.0.1:0").expect("bind srt listener");
    let listener = sock.local_addr().unwrap();
    let proxy = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let url = format!("srt://{}", proxy.local_addr().unwrap());
    let proxy_task = tokio::spawn(proxy_losing_one_packet_for_good(
        proxy,
        listener,
        PAYLOADS_BEFORE_LOSS,
    ));
    let mut src = SrtSrc::from_socket(sock)
        .expect("adopt socket")
        .with_frame_limit(PAYLOADS);
    let mut cap = Capture::default();

    let mut caller = spawn_ffmpeg_caller(&url);
    let result = tokio::time::timeout(Duration::from_secs(30), src.run(&mut cap)).await;
    caller.kill().expect("kill the caller");
    caller.wait().expect("reap the caller");
    proxy_task.abort();

    let received = result
        .expect("delivery carries on past the lost packet")
        .expect("run ok");
    assert_eq!(received, PAYLOADS);
    assert_all_ts_sync(&cap.first_bytes);
}
