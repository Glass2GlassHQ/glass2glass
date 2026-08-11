//! M901: the distributed-graph primitive over a WebTransport boundary, the third
//! carrier of the same `g2g-core` wire codec after TCP (M551) and WebSocket
//! (M554). A WebTransport session's bidirectional stream is a QUIC byte stream,
//! so the framing is the TCP pair's `u32` length prefix, not the WebSocket pair's
//! one-message-per-packet.
//!
//! The legs:
//!   1. loopback: our `RemoteWtSink` -> our `RemoteWtSrc` over real sockets, every
//!      frame's bytes / timing / sequence intact and the stream ending on `Eos`,
//!      plus the M558 reconnect behaviour against a server that binds late.
//!   2. remote transform: `RemoteWtTransform` against a peer that inverts each
//!      frame, proving the FIFO frame-out / processed-frame-back round trip.
//!   3. independent peers, both directions: our client against an **aioquic**
//!      (Python) WebTransport server, and our source against an aioquic client.
//!      Neither shares code with us, so the QUIC / HTTP-3 CONNECT handshake and
//!      the length framing are validated by the protocol rather than by agreement
//!      with ourselves. Each skips with a printed reason when `uv` or the network
//!      is unavailable.
//!
//! The wire codec's per-variant fidelity is unit-tested in `g2g_core::wire`; this
//! file asserts the WebTransport carriage of it.
#![cfg(feature = "webtransport")]

use core::future::Future;
use core::pin::Pin;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use tokio::io::AsyncReadExt;

use web_transport_quinn::quinn::rustls::pki_types::CertificateDer;
use web_transport_quinn::{Server, ServerBuilder};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{
    run_simple_pipeline, run_source_transform_sink, LatencyProfile, SourceLoop,
};
use g2g_core::wire::{decode_packet, encode_packet};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, PropValue, PushOutcome, Rate,
    RawVideoFormat,
};

use g2g_plugins::remotewtsink::RemoteWtSink;
use g2g_plugins::remotewtsrc::RemoteWtSrc;
use g2g_plugins::remotewttransform::RemoteWtTransform;

// ---------------------------------------------------------------- test scaffold

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

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

/// Collects the caps and each frame's (sequence, timing, bytes) so a test can
/// assert the whole stream crossed the boundary intact.
#[derive(Default)]
struct CollectSink {
    caps: Vec<Caps>,
    frames: Vec<(u64, FrameTiming, Vec<u8>)>,
    eos: bool,
}

impl AsyncElement for CollectSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn configure_pipeline(&mut self, c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // The initial caps reach a sink via configure_pipeline, not process.
        self.caps.push(c.clone());
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if let Some(slice) = frame.domain.as_system_slice() {
                        self.frames
                            .push((frame.sequence, frame.timing, slice.to_vec()));
                    }
                }
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Ok(())
        })
    }
}

/// Emits `n` RGBA frames (each byte = frame index) then EOS.
struct CountSrc {
    n: u8,
}

impl SourceLoop for CountSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(test_caps()))
    }
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(test_caps()))))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(test_caps())).await?;
            for i in 0..self.n {
                out.push(PipelinePacket::DataFrame(test_frame(i, FRAME_LEN)))
                    .await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.n as u64)
        })
    }
}

const FRAME_LEN: usize = 2 * 2 * 4; // 2x2 RGBA

fn test_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

/// A frame tagged twice: byte 0 is the index and byte 1 a fixed marker, so a
/// mis-framed read (the length prefix is the only message boundary) is obvious.
fn test_frame(i: u8, len: usize) -> Frame {
    let mut bytes = vec![i; len];
    bytes[1] = 0xAB;
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: i as u64 * 1_000_000,
            dts_ns: i as u64 * 1_000_000,
            duration_ns: 33_000,
            keyframe: i == 0,
            ..FrameTiming::default()
        },
        sequence: i as u64,
        meta: Default::default(),
    }
}

// ------------------------------------------------------------------------ TLS

/// A self-signed certificate for the loopback server, written out as the PEM file
/// pair the `certificate` / `private-key` properties take, plus the hex SHA-256
/// digest a client passes as `server-certificate-hashes` (no system root covers a
/// throwaway certificate, exactly as for a browser's `serverCertificateHashes`).
struct TestCert {
    cert_path: PathBuf,
    key_path: PathBuf,
    hash_hex: String,
}

impl TestCert {
    fn generate(tag: &str) -> Self {
        let issued = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
            .expect("self-signed certificate");
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let cert_path = dir.join(format!("g2g-m901-{pid}-{tag}.crt"));
        let key_path = dir.join(format!("g2g-m901-{pid}-{tag}.key"));
        write_file(&cert_path, issued.cert.pem().as_bytes());
        write_file(&key_path, issued.signing_key.serialize_pem().as_bytes());

        let der = CertificateDer::from(issued.cert.der().to_vec());
        let provider = web_transport_quinn::crypto::default_provider();
        let digest = web_transport_quinn::crypto::sha256(&provider, &der);
        let hash_hex = digest.as_ref().iter().map(|b| format!("{b:02x}")).collect();

        Self {
            cert_path,
            key_path,
            hash_hex,
        }
    }

    fn cert(&self) -> String {
        self.cert_path.display().to_string()
    }

    fn key(&self) -> String {
        self.key_path.display().to_string()
    }
}

impl Drop for TestCert {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.cert_path);
        let _ = std::fs::remove_file(&self.key_path);
    }
}

fn write_file(path: &Path, bytes: &[u8]) {
    let mut f = std::fs::File::create(path).expect("create pem");
    f.write_all(bytes).expect("write pem");
}

// ------------------------------------------------------------------- leg 1

#[tokio::test]
async fn webtransport_carries_a_split_graph_edge() {
    const N: u8 = 8;

    let tls = TestCert::generate("loopback");

    // Far side: bind the QUIC endpoint up front (so the near side can connect
    // before accept() runs) and read the actual ephemeral port.
    let mut src = RemoteWtSrc::new("127.0.0.1:0".parse().unwrap())
        .with_certificate(tls.cert(), tls.key())
        .with_frame_limit(N as u64);
    let port = src.listen().await.expect("bind quic endpoint").port();
    let mut sink = CollectSink::default();
    let clock = ZeroClock;

    // Near side: a RemoteWtSink dialing the far-side port over https://, sending
    // the negotiated caps then N tagged frames then Eos. The QUIC + CONNECT
    // handshake is async, so the connect happens inside the first `process`.
    let sender = async {
        let mut remote = RemoteWtSink::new(format!("https://127.0.0.1:{port}"))
            .with_server_certificate_hashes(tls.hash_hex.clone());
        remote.configure_pipeline(&test_caps()).expect("configure");
        let mut null = NullOut;
        for i in 0u8..N {
            if remote
                .process(
                    PipelinePacket::DataFrame(test_frame(i, FRAME_LEN)),
                    &mut null,
                )
                .await
                .is_err()
            {
                break;
            }
        }
        // The far side stops at its frame limit; a late Eos send may fail once it
        // has closed, which is fine.
        let _ = remote.process(PipelinePacket::Eos, &mut null).await;
        remote.sent()
    };

    let recv = tokio::time::timeout(
        Duration::from_secs(20),
        run_simple_pipeline(
            &mut src,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );

    let (recv_res, sent) = tokio::join!(recv, sender);
    let stats = recv_res
        .expect("receiver finishes within 20s")
        .expect("receive pipeline ok");

    assert_eq!(
        stats.frames_emitted, N as u64,
        "all frames crossed the boundary"
    );
    assert!(
        sent >= (N as u64 + 1),
        "sender emitted caps + {N} frames: {sent}"
    );

    // The far side discovered the sender's caps from the wire.
    assert_eq!(
        sink.caps.first(),
        Some(&test_caps()),
        "discovered caps match the sender's"
    );

    // Each frame's sequence, timing, and bytes survived byte-for-byte.
    assert_eq!(sink.frames.len(), N as usize);
    for (i, (seq, timing, bytes)) in sink.frames.iter().enumerate() {
        assert_eq!(*seq, i as u64, "sequence preserved");
        assert_eq!(timing.pts_ns, i as u64 * 1_000_000, "pts preserved");
        assert_eq!(timing.keyframe, i == 0, "keyframe flag preserved");
        assert_eq!(bytes.len(), FRAME_LEN, "frame length preserved");
        assert_eq!(bytes[0], i as u8, "payload tag preserved (correct framing)");
        assert_eq!(bytes[1], 0xAB, "second marker preserved (no mis-framing)");
    }
    assert!(sink.eos, "the stream ended on Eos");
}

/// Reconnection matches the other carriers (M558): `with_reconnect` defers the
/// connect and retries it, so a server that binds late is tolerated.
#[tokio::test]
async fn webtransport_sink_retries_connect_until_the_server_is_up() {
    const N: u8 = 4;

    let tls = TestCert::generate("reconnect");
    // A port the server will only bind later: take one from the OS and drop it.
    let port = {
        let probe = std::net::UdpSocket::bind("127.0.0.1:0").expect("probe bind");
        probe.local_addr().expect("probe addr").port()
    };

    let hashes = tls.hash_hex.clone();
    let sender = async move {
        let mut sink = RemoteWtSink::new(format!("https://127.0.0.1:{port}"))
            .with_server_certificate_hashes(hashes)
            .with_reconnect(200);
        sink.configure_pipeline(&test_caps()).expect("configure");
        let mut null = NullOut;
        for i in 0..N {
            sink.process(
                PipelinePacket::DataFrame(test_frame(i, FRAME_LEN)),
                &mut null,
            )
            .await
            .expect("frame delivered after reconnect");
        }
        let _ = sink.process(PipelinePacket::Eos, &mut null).await;
    };

    let receiver = async {
        tokio::time::sleep(Duration::from_millis(300)).await;
        let mut src = RemoteWtSrc::new(format!("127.0.0.1:{port}").parse().unwrap())
            .with_certificate(tls.cert(), tls.key())
            .with_frame_limit(N as u64);
        let mut sink = CollectSink::default();
        let clock = ZeroClock;
        let stats = run_simple_pipeline(
            &mut src,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        )
        .await
        .expect("receive ok");
        (stats.frames_emitted, sink)
    };

    let recv = tokio::time::timeout(Duration::from_secs(30), receiver);
    let (recv_res, ()) = tokio::join!(recv, sender);
    let (emitted, sink) = recv_res.expect("finishes within 30s");

    assert_eq!(
        emitted, N as u64,
        "all frames crossed after the sink retried the handshake"
    );
    assert_eq!(
        sink.caps.first(),
        Some(&test_caps()),
        "caps re-sent on the connection that finally came up"
    );
    assert_eq!(
        sink.frames.iter().map(|f| f.0).collect::<Vec<_>>(),
        (0..N as u64).collect::<Vec<_>>(),
        "every frame in order"
    );
}

// ------------------------------------------------- leg 1b: datagram carrier (M911)

/// The drop-tolerant carrier: with `datagrams=true` each data frame is one QUIC
/// datagram and the control packets stay on the session's stream, so the receiver
/// still discovers the caps and still sees the end. The last frame is deliberately
/// larger than any path MTU, which is the documented fallback: it goes on the
/// stream rather than being truncated or dropped, and `datagrams-sent` says so.
///
/// Both ends also run `congestion-control=low-latency`, so the nick is proven to
/// reach a builder that still produces a working endpoint, not just to round-trip
/// through `set_property`.
#[tokio::test]
async fn webtransport_datagram_carrier_delivers_frames_and_falls_back_when_too_large() {
    const N: u8 = 6;
    const BIG_LEN: usize = 8 * 1024;

    let tls = TestCert::generate("datagram");

    let mut src = RemoteWtSrc::new("127.0.0.1:0".parse().unwrap())
        .with_certificate(tls.cert(), tls.key())
        .with_frame_limit(N as u64 + 1);
    SourceLoop::set_property(
        &mut src,
        "congestion-control",
        PropValue::Str("low-latency".into()),
    )
    .expect("congestion-control on the server");
    let port = src.listen().await.expect("bind quic endpoint").port();
    let mut sink = CollectSink::default();
    let clock = ZeroClock;

    let sender = async {
        let mut remote = RemoteWtSink::new(format!("https://127.0.0.1:{port}"))
            .with_server_certificate_hashes(tls.hash_hex.clone())
            .with_datagrams(true);
        AsyncElement::set_property(
            &mut remote,
            "congestion-control",
            PropValue::Str("low-latency".into()),
        )
        .expect("congestion-control on the client");
        remote.configure_pipeline(&test_caps()).expect("configure");
        let mut null = NullOut;
        for i in 0u8..N {
            if remote
                .process(
                    PipelinePacket::DataFrame(test_frame(i, FRAME_LEN)),
                    &mut null,
                )
                .await
                .is_err()
            {
                break;
            }
        }
        // One frame no datagram can carry: the carrier must fall back.
        let _ = remote
            .process(PipelinePacket::DataFrame(test_frame(N, BIG_LEN)), &mut null)
            .await;
        let _ = remote.process(PipelinePacket::Eos, &mut null).await;
        AsyncElement::get_property(&remote, "datagrams-sent")
    };

    let recv = tokio::time::timeout(
        Duration::from_secs(20),
        run_simple_pipeline(
            &mut src,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );

    let (recv_res, datagrams_sent) = tokio::join!(recv, sender);
    let stats = recv_res
        .expect("receiver finishes within 20s")
        .expect("receive pipeline ok");

    assert_eq!(
        datagrams_sent,
        Some(PropValue::Uint(N as u64)),
        "every frame that fits went out as a datagram, the oversized one did not"
    );
    assert_eq!(
        stats.frames_emitted,
        N as u64 + 1,
        "the datagram frames and the fallback frame all reached the graph"
    );
    assert_eq!(
        sink.caps.first(),
        Some(&test_caps()),
        "caps still discovered from the stream half of the carrier"
    );

    // Datagrams are unordered against the stream, so match by sequence.
    for i in 0..=N {
        let (_, timing, bytes) = sink
            .frames
            .iter()
            .find(|(seq, _, _)| *seq == i as u64)
            .unwrap_or_else(|| panic!("frame {i} arrived"));
        let want = if i == N { BIG_LEN } else { FRAME_LEN };
        assert_eq!(bytes.len(), want, "frame {i} length preserved");
        assert_eq!(bytes[0], i, "frame {i} payload tag preserved");
        assert_eq!(bytes[1], 0xAB, "frame {i} second marker preserved");
        assert_eq!(
            timing.pts_ns,
            i as u64 * 1_000_000,
            "frame {i} pts preserved"
        );
    }
}

// ------------------------------------------------------------------- leg 2

/// The remote stage: read the length-framed wire stream off the session's
/// bidirectional stream, invert each frame's bytes (a stand-in for real work, e.g.
/// inference), and reply one processed frame per frame. Ignores caps (config only)
/// and ends on Eos; never echoes control, so the client's per-frame read pairs
/// with its frame.
async fn invert_server(server: &mut Server) -> Result<u64, Box<dyn std::error::Error>> {
    let request = server.accept().await.ok_or("no session offered")?;
    let session = request.ok().await?;
    let (mut tx, mut rx) = session.accept_bi().await?;
    let mut processed = 0u64;
    while let Some(body) = read_framed(&mut rx).await? {
        match decode_packet(&body).map_err(|e| format!("decode: {e:?}"))? {
            PipelinePacket::DataFrame(mut frame) => {
                if let MemoryDomain::System(s) = &mut frame.domain {
                    for b in s.as_mut_slice() {
                        *b = !*b; // the "processing": invert every byte
                    }
                }
                let out = encode_packet(&PipelinePacket::DataFrame(frame))
                    .map_err(|e| format!("encode: {e:?}"))?;
                tx.write_all(&(out.len() as u32).to_le_bytes()).await?;
                tx.write_all(&out).await?;
                processed += 1;
            }
            PipelinePacket::Eos => break,
            _ => {} // caps / segment: no reply
        }
    }
    Ok(processed)
}

/// Read one `u32`-length-framed body; `None` at a clean stream end.
async fn read_framed<R: tokio::io::AsyncRead + Unpin>(
    rx: &mut R,
) -> Result<Option<Vec<u8>>, std::io::Error> {
    let mut len = [0u8; 4];
    match rx.read_exact(&mut len).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let mut body = vec![0u8; u32::from_le_bytes(len) as usize];
    rx.read_exact(&mut body).await?;
    Ok(Some(body))
}

#[tokio::test]
async fn webtransport_transform_offloads_and_returns_processed_frames() {
    const N: u8 = 6;

    let tls = TestCert::generate("transform");
    let (chain, key) = load_pem(&tls);
    let mut server = ServerBuilder::new()
        .with_addr("127.0.0.1:0".parse().unwrap())
        .with_certificate(chain, key)
        .expect("quic server");
    let port = server.local_addr().expect("bound port").port();

    let mut src = CountSrc { n: N };
    let mut xform = RemoteWtTransform::new(format!("https://127.0.0.1:{port}"))
        .with_server_certificate_hashes(tls.hash_hex.clone());
    let mut sink = CollectSink::default();
    let clock = ZeroClock;

    // Drive the pipeline and the remote stage concurrently on this task (a
    // Box<dyn Error> server result is not Send, so join! not spawn).
    let run = tokio::time::timeout(
        Duration::from_secs(20),
        run_source_transform_sink(
            &mut src,
            &mut xform,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );
    let (run_res, server_res) = tokio::join!(run, invert_server(&mut server));
    let stats = run_res.expect("finishes within 20s").expect("pipeline ok");

    assert_eq!(
        stats.frames_emitted, N as u64,
        "all frames crossed and returned"
    );
    assert_eq!(xform.emitted(), N as u64, "transform emitted one per frame");
    assert_eq!(sink.frames.len(), N as usize);
    for (i, (seq, _timing, bytes)) in sink.frames.iter().enumerate() {
        assert_eq!(*seq, i as u64, "order preserved (FIFO reply pairing)");
        // The source sent byte == i; the remote stage inverted it.
        assert_eq!(bytes[0], !(i as u8), "the remote stage's work was applied");
        assert_eq!(
            bytes[1], !0xABu8,
            "whole payload processed, not just byte 0"
        );
    }
    assert!(sink.eos, "stream ended on Eos");

    let processed = server_res.expect("server ok");
    assert_eq!(processed, N as u64, "server processed every frame");
}

fn load_pem(
    tls: &TestCert,
) -> (
    Vec<CertificateDer<'static>>,
    web_transport_quinn::quinn::rustls::pki_types::PrivateKeyDer<'static>,
) {
    use web_transport_quinn::quinn::rustls::pki_types::pem::PemObject;
    use web_transport_quinn::quinn::rustls::pki_types::PrivateKeyDer;
    let chain: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(&tls.cert_path)
        .expect("open cert")
        .collect::<Result<_, _>>()
        .expect("parse cert");
    let key = PrivateKeyDer::from_pem_file(&tls.key_path).expect("parse key");
    (chain, key)
}

// ------------------------------------------------------------------- leg 3

/// The aioquic peer: a Python WebTransport server that reads our length-framed
/// wire stream, drops the leading caps message, and echoes every later message
/// back verbatim. Independent of our stack top to bottom (its own QUIC, HTTP/3,
/// WebTransport and length-prefix parser), so it validates the handshake *and*
/// that our framing puts message boundaries where the protocol says.
const AIOQUIC_PEER: &str = include_str!("m901_aioquic_peer.py");

/// The aioquic client peer: pushes a prepared wire stream at our `RemoteWtSrc`,
/// so the *server* half is validated against a foreign implementation too.
const AIOQUIC_CLIENT: &str = include_str!("m901_aioquic_client.py");

/// Run one of the peer scripts under `uv`, or `None` with a printed reason (no
/// `uv`, no network, no Python) so the leg skips rather than fails on a host
/// without them.
#[cfg(unix)]
fn spawn_python(tag: &str, source: &str, args: &[String]) -> Option<(Child, PathBuf)> {
    use std::os::unix::process::CommandExt;

    // Resolve the environment first (`uv` fetches aioquic on a cold cache): doing
    // it here turns "no uv / no network" into an immediate skip instead of a leg
    // that waits out its whole budget for a peer that will never start.
    match Command::new("uv")
        .args(["run", "--with", "aioquic", "python", "-c", "import aioquic"])
        .output()
    {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            eprintln!(
                "SKIP: `uv run --with aioquic` failed, no independent peer: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            return None;
        }
        Err(e) => {
            eprintln!("SKIP: `uv` not usable, cannot run the aioquic independent peer: {e}");
            return None;
        }
    }
    let script = std::env::temp_dir().join(format!("g2g-m901-{}-{tag}.py", std::process::id()));
    write_file(&script, source.as_bytes());

    let child = Command::new("uv")
        .args(["run", "--with", "aioquic", "python"])
        .arg(&script)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // `uv` runs Python as a child of its own, so put the pair in one process
        // group: killing only `uv` would leave Python holding the port and the
        // stdout pipe, and reading that pipe would then never reach EOF.
        .process_group(0)
        .spawn();
    match child {
        Ok(child) => Some((child, script)),
        Err(e) => {
            eprintln!("SKIP: could not start the aioquic peer: {e}");
            None
        }
    }
}

#[cfg(not(unix))]
fn spawn_python(_tag: &str, _source: &str, _args: &[String]) -> Option<(Child, PathBuf)> {
    eprintln!("SKIP: the aioquic peer harness reaps `uv`'s python child by POSIX process group");
    None
}

/// Let the peer exit on its own (it does when the session closes), then kill its
/// whole process group, drop its script, and collect what it logged.
fn reap((mut peer, script): (Child, PathBuf)) -> std::process::Output {
    for _ in 0..100 {
        if matches!(peer.try_wait(), Ok(Some(_))) {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    #[cfg(unix)]
    {
        let _ = Command::new("kill")
            .arg("-KILL")
            .arg(format!("-{}", peer.id()))
            .stderr(Stdio::null())
            .status();
    }
    let _ = peer.kill();
    let _ = std::fs::remove_file(&script);
    peer.wait_with_output().expect("peer output")
}

#[tokio::test]
async fn webtransport_client_round_trips_through_an_aioquic_peer() {
    const N: u8 = 4;

    let tls = TestCert::generate("aioquic");
    // aioquic binds the port itself, so pick a free one by binding and releasing
    // a UDP socket (a race window no other test on this host is likely to hit).
    let port = {
        let probe = std::net::UdpSocket::bind("127.0.0.1:0").expect("probe bind");
        probe.local_addr().expect("probe addr").port()
    };
    let ready_path =
        std::env::temp_dir().join(format!("g2g-m901-{}-peer.ready", std::process::id()));
    let _ = std::fs::remove_file(&ready_path);
    let args = [
        tls.cert(),
        tls.key(),
        port.to_string(),
        ready_path.display().to_string(),
    ];
    let Some(peer) = spawn_python("peer", AIOQUIC_PEER, &args) else {
        return;
    };

    let mut src = CountSrc { n: N };
    let mut xform = RemoteWtTransform::new(format!("https://127.0.0.1:{port}"))
        .with_server_certificate_hashes(tls.hash_hex.clone());
    let mut sink = CollectSink::default();
    let clock = ZeroClock;

    // Wait for the peer to report its socket is bound: the pipeline has no
    // reconnect, so this is the readiness gate.
    if !wait_for_ready(&ready_path, Duration::from_secs(30)).await {
        let out = reap(peer);
        let _ = std::fs::remove_file(&ready_path);
        eprintln!(
            "SKIP: the aioquic peer never bound its socket (no network for `uv`?): {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return;
    }
    let _ = std::fs::remove_file(&ready_path);

    let run = tokio::time::timeout(
        Duration::from_secs(30),
        run_source_transform_sink(
            &mut src,
            &mut xform,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await;
    let peer_out = reap(peer);
    let stats = run
        .expect("finishes within 30s")
        .unwrap_or_else(|e| panic!("pipeline failed against the aioquic peer: {e:?}"));

    assert_eq!(
        stats.frames_emitted, N as u64,
        "every frame round-tripped through the independent peer"
    );
    assert_eq!(sink.frames.len(), N as usize);
    for (i, (seq, timing, bytes)) in sink.frames.iter().enumerate() {
        // The peer echoes verbatim, so the packet that comes back must decode to
        // exactly the frame we sent: byte-exact framing across a foreign parser.
        assert_eq!(*seq, i as u64, "sequence survived the foreign peer");
        assert_eq!(timing.pts_ns, i as u64 * 1_000_000, "timing survived");
        assert_eq!(bytes.len(), FRAME_LEN, "payload length survived");
        assert_eq!(bytes[0], i as u8, "payload tag survived");
        assert_eq!(bytes[1], 0xAB, "second marker survived (no mis-framing)");
    }
    // The peer reports what it saw, so a silently-empty echo cannot pass.
    let log = String::from_utf8_lossy(&peer_out.stdout);
    assert!(
        log.contains(&format!("echoed {N}")),
        "aioquic peer echoed {N} messages, log: {log}{}",
        String::from_utf8_lossy(&peer_out.stderr)
    );
}

/// The mirror of the leg above: a foreign *client* pushes a wire stream at our
/// `RemoteWtSrc`, so the server half (session accept, HTTP-3 CONNECT response,
/// bidirectional-stream accept, framing) is validated against an implementation
/// that shares no code with it. The wire bytes are built here (the codec is ours,
/// and it is unit-tested in `g2g_core::wire`); what the peer supplies is
/// everything under it.
#[tokio::test]
async fn webtransport_source_accepts_an_aioquic_client() {
    const N: u8 = 4;

    let tls = TestCert::generate("aioclient");
    let mut src =
        RemoteWtSrc::new("127.0.0.1:0".parse().unwrap()).with_certificate(tls.cert(), tls.key());
    let port = src.listen().await.expect("bind quic endpoint").port();

    // Caps, N frames, Eos: each `encode_packet` body behind its u32 LE length.
    let mut blob: Vec<u8> = Vec::new();
    let mut push = |packet: &PipelinePacket| {
        let body = encode_packet(packet).expect("encode");
        blob.extend_from_slice(&(body.len() as u32).to_le_bytes());
        blob.extend_from_slice(&body);
    };
    push(&PipelinePacket::CapsChanged(test_caps()));
    for i in 0..N {
        push(&PipelinePacket::DataFrame(test_frame(i, FRAME_LEN)));
    }
    push(&PipelinePacket::Eos);
    let blob_path =
        std::env::temp_dir().join(format!("g2g-m901-{}-stream.bin", std::process::id()));
    write_file(&blob_path, &blob);

    let args = [port.to_string(), blob_path.display().to_string()];
    let Some(peer) = spawn_python("client", AIOQUIC_CLIENT, &args) else {
        let _ = std::fs::remove_file(&blob_path);
        return;
    };

    let mut sink = CollectSink::default();
    let clock = ZeroClock;
    // The source is already listening, so the peer connects whenever it is ready.
    let run = tokio::time::timeout(
        Duration::from_secs(30),
        run_simple_pipeline(
            &mut src,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await;
    let peer_out = reap(peer);
    let _ = std::fs::remove_file(&blob_path);
    let log = String::from_utf8_lossy(&peer_out.stdout);
    let errs = String::from_utf8_lossy(&peer_out.stderr);
    if !log.contains("connect status 200") {
        eprintln!("SKIP: the aioquic client never connected (no network for `uv`?): {errs}");
        return;
    }

    let stats = run
        .expect("finishes within 30s")
        .unwrap_or_else(|e| panic!("source failed against the aioquic client: {e:?} {errs}"));
    assert_eq!(
        stats.frames_emitted, N as u64,
        "every frame the foreign client pushed reached the graph"
    );
    assert_eq!(
        sink.caps.first(),
        Some(&test_caps()),
        "caps discovered from the foreign client's leading wire packet"
    );
    for (i, (seq, timing, bytes)) in sink.frames.iter().enumerate() {
        assert_eq!(*seq, i as u64, "sequence intact");
        assert_eq!(timing.pts_ns, i as u64 * 1_000_000, "timing intact");
        assert_eq!(bytes[0], i as u8, "payload tag intact");
        assert_eq!(bytes[1], 0xAB, "second marker intact (no mis-framing)");
    }
    assert!(sink.eos, "the foreign client's Eos ended the stream");
}

/// Poll for the peer's ready file (it writes one once its socket is bound).
async fn wait_for_ready(path: &Path, budget: Duration) -> bool {
    let deadline = std::time::Instant::now() + budget;
    while std::time::Instant::now() < deadline {
        if path.exists() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    false
}

// ------------------------------------------------------------------ properties

#[test]
fn webtransport_elements_expose_their_knobs() {
    fn declares(specs: &[g2g_core::PropertySpec], name: &str) -> bool {
        specs.iter().any(|s| s.name == name)
    }

    let mut sink = RemoteWtSink::new("https://127.0.0.1:9603");
    for name in [
        "location",
        "server-certificate-hashes",
        "reconnect-attempts",
    ] {
        assert!(declares(AsyncElement::properties(&sink), name), "{name}");
    }
    AsyncElement::set_property(
        &mut sink,
        "location",
        PropValue::Str("https://example.test:4433".into()),
    )
    .unwrap();
    assert_eq!(
        AsyncElement::get_property(&sink, "location"),
        Some(PropValue::Str("https://example.test:4433".into()))
    );
    AsyncElement::set_property(
        &mut sink,
        "server-certificate-hashes",
        PropValue::Str("ab".repeat(32)),
    )
    .unwrap();
    assert_eq!(
        AsyncElement::get_property(&sink, "server-certificate-hashes"),
        Some(PropValue::Str("ab".repeat(32)))
    );
    AsyncElement::set_property(&mut sink, "reconnect-attempts", PropValue::Uint(3)).unwrap();
    assert_eq!(
        AsyncElement::get_property(&sink, "reconnect-attempts"),
        Some(PropValue::Uint(3))
    );

    let mut src = RemoteWtSrc::new("0.0.0.0:9603".parse().unwrap());
    for name in [
        "address",
        "port",
        "keep-listening",
        "certificate",
        "private-key",
    ] {
        assert!(declares(SourceLoop::properties(&src), name), "{name}");
    }
    SourceLoop::set_property(&mut src, "certificate", PropValue::Str("/tmp/a.crt".into())).unwrap();
    SourceLoop::set_property(&mut src, "private-key", PropValue::Str("/tmp/a.key".into())).unwrap();
    assert_eq!(
        SourceLoop::get_property(&src, "certificate"),
        Some(PropValue::Str("/tmp/a.crt".into()))
    );
    assert_eq!(
        SourceLoop::get_property(&src, "private-key"),
        Some(PropValue::Str("/tmp/a.key".into()))
    );
    SourceLoop::set_property(&mut src, "port", PropValue::Uint(9700)).unwrap();
    assert_eq!(
        SourceLoop::get_property(&src, "port"),
        Some(PropValue::Uint(9700))
    );

    let mut xform = RemoteWtTransform::new("https://127.0.0.1:9604");
    for name in [
        "location",
        "server-certificate-hashes",
        "congestion-control",
    ] {
        assert!(declares(AsyncElement::properties(&xform), name), "{name}");
    }
    AsyncElement::set_property(
        &mut xform,
        "location",
        PropValue::Str("https://peer.test:443".into()),
    )
    .unwrap();
    assert_eq!(
        AsyncElement::get_property(&xform, "location"),
        Some(PropValue::Str("https://peer.test:443".into()))
    );
}

/// M911's knobs: the datagram carrier on the sink (the transform deliberately has
/// none: a dropped request would strand its reply), and the congestion controller
/// on every element that builds a QUIC endpoint.
#[test]
fn webtransport_elements_expose_the_datagram_and_congestion_knobs() {
    fn spec<'a>(
        specs: &'a [g2g_core::PropertySpec],
        name: &str,
    ) -> Option<&'a g2g_core::PropertySpec> {
        specs.iter().find(|s| s.name == name)
    }

    let mut sink = RemoteWtSink::new("https://127.0.0.1:9603");
    assert!(spec(AsyncElement::properties(&sink), "datagrams").is_some());
    assert_eq!(
        AsyncElement::get_property(&sink, "datagrams"),
        Some(PropValue::Bool(false)),
        "the reliable stream stays the default carrier"
    );
    AsyncElement::set_property(&mut sink, "datagrams", PropValue::Bool(true)).unwrap();
    assert_eq!(
        AsyncElement::get_property(&sink, "datagrams"),
        Some(PropValue::Bool(true))
    );
    // Nothing sent yet, and the counter is readable but not settable.
    assert_eq!(
        AsyncElement::get_property(&sink, "datagrams-sent"),
        Some(PropValue::Uint(0))
    );
    assert_eq!(
        AsyncElement::set_property(&mut sink, "datagrams-sent", PropValue::Uint(7)),
        Err(g2g_core::PropError::ReadOnly)
    );
    assert!(
        !spec(AsyncElement::properties(&sink), "datagrams-sent")
            .expect("declared")
            .flags
            .writable,
        "datagrams-sent is a status readout"
    );
    // The FIFO round trip has no drop-tolerant mode to offer.
    let xform = RemoteWtTransform::new("https://127.0.0.1:9604");
    assert!(spec(AsyncElement::properties(&xform), "datagrams").is_none());

    // congestion-control: the same closed nick set on client, server, transform.
    let cc = spec(AsyncElement::properties(&sink), "congestion-control").expect("declared");
    assert_eq!(cc.default, Some("default"));
    assert_eq!(
        cc.enum_values,
        Some("default | throughput | low-latency"),
        "the nicks web-transport-quinn's builders accept"
    );
    for nick in ["default", "throughput", "low-latency"] {
        AsyncElement::set_property(&mut sink, "congestion-control", PropValue::Str(nick.into()))
            .unwrap_or_else(|e| panic!("{nick}: {e:?}"));
        assert_eq!(
            AsyncElement::get_property(&sink, "congestion-control"),
            Some(PropValue::Str(nick.into()))
        );
    }
    assert_eq!(
        AsyncElement::set_property(
            &mut sink,
            "congestion-control",
            PropValue::Str("reno".into())
        ),
        Err(g2g_core::PropError::Value),
        "an unknown controller is rejected, not silently ignored"
    );
    assert_eq!(
        AsyncElement::get_property(&sink, "congestion-control"),
        Some(PropValue::Str("low-latency".into())),
        "the rejected value did not overwrite the last good one"
    );

    let mut src = RemoteWtSrc::new("0.0.0.0:9603".parse().unwrap());
    assert!(spec(SourceLoop::properties(&src), "congestion-control").is_some());
    SourceLoop::set_property(
        &mut src,
        "congestion-control",
        PropValue::Str("low-latency".into()),
    )
    .unwrap();
    assert_eq!(
        SourceLoop::get_property(&src, "congestion-control"),
        Some(PropValue::Str("low-latency".into()))
    );

    let mut xform = xform;
    AsyncElement::set_property(
        &mut xform,
        "congestion-control",
        PropValue::Str("throughput".into()),
    )
    .unwrap();
    assert_eq!(
        AsyncElement::get_property(&xform, "congestion-control"),
        Some(PropValue::Str("throughput".into()))
    );
}

/// The launch names are how a `gst-launch` line reaches these elements, so the
/// registry must carry all three.
#[test]
fn webtransport_elements_are_registered_for_launch() {
    let reg = g2g_plugins::registry::default_registry();
    assert!(reg.make_element("remotewtsink").is_some());
    assert!(reg.make_element("remotewttransform").is_some());
    assert!(reg.make_source("remotewtsrc").is_some());
}
