//! Helpers shared by the MoQ Transport test binaries (`m902_moqt`,
//! `m903_moqt_subscribe`, `m905_moqt_datagram`, `m906_moqt_control_pump`): the
//! throwaway certificate and QUIC endpoint, the fMP4 the publisher is fed, the
//! output sinks, the reference-peer lookup, and the fragment comparisons. One
//! definition, included per test binary via `mod moqt_common;`.
#![allow(dead_code)] // no one test binary uses every helper here

use core::future::Future;
use core::pin::Pin;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use web_transport_quinn::quinn::rustls::pki_types::pem::PemObject;
use web_transport_quinn::quinn::rustls::pki_types::{CertificateDer, PrivateKeyDer};
use web_transport_quinn::{Server, ServerBuilder};

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, Dim, G2gError, OutputSink, PushOutcome, Rate, VideoCodec};

// ------------------------------------------------------------------- the peer

pub(crate) struct TestCert {
    pub(crate) cert_path: PathBuf,
    pub(crate) key_path: PathBuf,
    pub(crate) hash_hex: String,
}

impl TestCert {
    pub(crate) fn generate() -> Self {
        let issued = rcgen::generate_simple_self_signed(vec![
            "localhost".to_string(),
            "127.0.0.1".to_string(),
        ])
        .expect("self-signed certificate");
        let dir = std::env::temp_dir();
        let unique = format!("{}-{:?}", std::process::id(), std::thread::current().id());
        let cert_path = dir.join(format!("g2g-moqt-{unique}.crt"));
        let key_path = dir.join(format!("g2g-moqt-{unique}.key"));
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
}

impl Drop for TestCert {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.cert_path);
        let _ = std::fs::remove_file(&self.key_path);
    }
}

fn write_file(path: &Path, bytes: &[u8]) {
    let mut f = std::fs::File::create(path).expect("create file");
    f.write_all(bytes).expect("write file");
}

/// A WebTransport server on an ephemeral port with `tls`, and the port it took.
pub(crate) fn bind_server(tls: &TestCert) -> (Server, u16) {
    let chain: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(&tls.cert_path)
        .expect("read certificate")
        .collect::<Result<_, _>>()
        .expect("parse certificate");
    let key = PrivateKeyDer::from_pem_file(&tls.key_path).expect("read key");
    let server = ServerBuilder::new()
        .with_addr("127.0.0.1:0".parse().expect("addr"))
        .with_certificate(chain, key)
        .expect("bind quic endpoint");
    let port = server.local_addr().expect("local addr").port();
    (server, port)
}

pub(crate) fn free_udp_port() -> u16 {
    let probe = std::net::UdpSocket::bind("127.0.0.1:0").expect("probe bind");
    probe.local_addr().expect("probe addr").port()
}

// -------------------------------------------------------------- the reference

/// Kills its child on drop, so a failing assertion never leaves a relay, a
/// publisher or a subscriber behind.
pub(crate) struct Reaped(pub(crate) Child);

impl Drop for Reaped {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Where the reference peers live: `$MOQ_RS_BIN`, else the release build of a
/// `moq-rs` checkout beside this one, else `PATH`.
pub(crate) fn reference_binary(name: &str) -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("MOQ_RS_BIN") {
        let path = PathBuf::from(dir).join(name);
        return path.is_file().then_some(path);
    }
    if let Some(home) = std::env::var_os("HOME") {
        let path = PathBuf::from(home)
            .join("src/moq-rs/target/release")
            .join(name);
        if path.is_file() {
            return Some(path);
        }
    }
    let probe = Command::new(name).arg("--help").output().ok()?;
    probe.status.success().then(|| PathBuf::from(name))
}

pub(crate) fn relay_missing_reason() -> String {
    String::from(
        "SKIP: moq-relay-ietf not found. Build it with `cargo +stable build --release \
         -p moq-relay-ietf` in a cloudflare/moq-rs checkout, or point $MOQ_RS_BIN at \
         its directory.",
    )
}

/// Start `moq-relay-ietf` on `port` with a freshly issued certificate. `None`
/// (with a printed reason) when the binary is absent.
pub(crate) fn spawn_relay(tls: &TestCert, port: u16) -> Option<Reaped> {
    let relay_bin = match reference_binary("moq-relay-ietf") {
        Some(bin) => bin,
        None => {
            eprintln!("{}", relay_missing_reason());
            return None;
        }
    };
    match Command::new(&relay_bin)
        .arg("--bind")
        .arg(format!("127.0.0.1:{port}"))
        .arg("--tls-cert")
        .arg(&tls.cert_path)
        .arg("--tls-key")
        .arg(&tls.key_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => Some(Reaped(child)),
        Err(e) => {
            eprintln!("SKIP: could not start moq-relay-ietf: {e}");
            None
        }
    }
}

// -------------------------------------------------------------- the publisher

pub(crate) fn h264_caps(w: u32, h: u32) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

#[derive(Default)]
pub(crate) struct CaptureSink {
    pub(crate) frames: Vec<Vec<u8>>,
}

impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames.push(s.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

pub(crate) struct NullOut;
impl OutputSink for NullOut {
    fn push<'a>(
        &'a mut self,
        _packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

pub(crate) fn frame(bytes: Vec<u8>, pts_ns: u64, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns: 33_333_333,
            ..FrameTiming::default()
        },
        sequence,
        meta: Default::default(),
    }
}

/// The SPS every IDR access unit carries. A fragment holding it is a keyframe
/// fragment, which is where the publisher starts a new group.
pub(crate) const SPS: [u8; 6] = [0x67, 0x42, 0xC0, 0x1E, 0x11, 0x22];

/// An IDR access unit with the parameter sets every tenth frame, then P slices,
/// so the muxer can build a `moov` and marks a sync sample once per GOP.
/// `big_every` (0 = never) pads that many-th unit past the path MTU, so datagram
/// mode has to fall back to a stream for it.
pub(crate) fn access_unit(index: u64, big_every: u64) -> Vec<u8> {
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let mut unit = if index % 10 == 0 {
        [
            &[0, 0, 0, 1][..],
            &SPS,
            &[0, 0, 0, 1],
            &pps,
            &[0, 0, 0, 1],
            &[0x65, index as u8, 0xAA],
        ]
        .concat()
    } else {
        [&[0, 0, 0, 1][..], &[0x41, index as u8, 0xBB]].concat()
    };
    if big_every != 0 && index % big_every == 0 {
        unit.extend(core::iter::repeat_n(index as u8, 2500));
    }
    unit
}

// ------------------------------------------------------------ the comparisons

/// The top-level box offsets of an fMP4 stream, so a comparison can insist a
/// match starts and ends on a box boundary rather than anywhere.
pub(crate) fn box_boundaries(stream: &[u8]) -> Vec<usize> {
    let mut out = vec![0usize];
    let mut at = 0usize;
    while at + 8 <= stream.len() {
        let size = u32::from_be_bytes(stream[at..at + 4].try_into().expect("4 bytes")) as usize;
        if size < 8 || at + size > stream.len() {
            break;
        }
        at += size;
        out.push(at);
    }
    out
}

pub(crate) fn moof_count(stream: &[u8]) -> usize {
    let mut count = 0;
    let mut at = 0usize;
    while at + 8 <= stream.len() {
        let size = u32::from_be_bytes(stream[at..at + 4].try_into().expect("4 bytes")) as usize;
        if size < 8 || at + size > stream.len() {
            break;
        }
        if &stream[at + 4..at + 8] == b"moof" {
            count += 1;
        }
        at += size;
    }
    count
}

/// Keyframe fragments in a stream, counted by the SPS each IDR carries: the
/// number of groups the bytes cover.
pub(crate) fn group_starts(stream: &[u8]) -> usize {
    stream.windows(SPS.len()).filter(|w| *w == SPS).count()
}

/// The length of the `ftyp`+`moov` prefix: the init segment a publisher makes
/// out of it.
pub(crate) fn init_len(stream: &[u8]) -> usize {
    box_boundaries(stream)
        .into_iter()
        .find(|at| {
            at + 8 <= stream.len()
                && &stream[at + 4..at + 8] != b"ftyp"
                && &stream[at + 4..at + 8] != b"moov"
        })
        .expect("a fragment follows the init segment")
}

/// Frames a subscription is asked for: one init segment plus enough fragments
/// to span more than one group (a GOP is ten fragments here).
pub(crate) const FRAMES_WANTED: u64 = 14;

/// Assert `received` is the init segment followed by an unbroken run of
/// `published`'s fragments.
pub(crate) fn assert_matches_published(
    received: &[Vec<u8>],
    published: &[u8],
    min_fragments: usize,
) {
    assert!(!received.is_empty(), "the subscriber emitted nothing");
    let init = init_len(published);
    assert_eq!(
        received[0],
        published[..init],
        "the init segment came back byte for byte"
    );
    let tail: Vec<u8> = received[1..].concat();
    assert!(
        moof_count(&tail) >= min_fragments,
        "expected at least {min_fragments} fragments, got {}",
        moof_count(&tail)
    );
    let boundaries = box_boundaries(published);
    let start = published
        .windows(tail.len())
        .position(|w| w == tail)
        .expect("the received fragments are a contiguous run of the published ones");
    assert!(
        boundaries.contains(&start),
        "the run starts at a box boundary"
    );
    assert!(
        boundaries.contains(&(start + tail.len())),
        "the run ends at a box boundary"
    );
    assert!(
        start >= init,
        "the fragments come from the fragment region, not the init segment"
    );
}

/// Assert `received` is the init segment followed by whole fragments of
/// `published` in publish order, with gaps allowed: what a path that loses
/// objects delivers.
pub(crate) fn assert_ordered_fragments(received: &[Vec<u8>], published: &[u8]) {
    assert!(!received.is_empty(), "the subscriber emitted nothing");
    let init = init_len(published);
    assert_eq!(
        received[0],
        published[..init],
        "the init segment came first"
    );
    let mut last = init;
    for fragment in &received[1..] {
        assert_eq!(moof_count(fragment), 1, "each frame is one whole fragment");
        let at = published
            .windows(fragment.len())
            .position(|w| w == fragment.as_slice())
            .expect("every frame is a fragment that was published");
        assert!(at >= last, "fragments arrive in publish order");
        last = at;
    }
}
