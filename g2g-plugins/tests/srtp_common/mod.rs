//! Helpers shared by the SRTP element batteries (`m1098_srtp`,
//! `m1099_srtp_session`, `m1100_dtls_srtp`, `m1101_srtp_legacy`): the
//! one-packet-per-frame source, the collecting sink, the packet builders, and
//! the `gst-launch-1.0` peer harness the interop legs drive. One definition,
//! included per test binary via `mod srtp_common;`.
#![allow(dead_code)] // no one battery uses every helper here

use core::future::Future;
use core::pin::Pin;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::rtp::RtpHeader;
use g2g_core::runtime::{run_graph, run_source_transform_sink, GraphNodeRef, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, Graph, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket,
};

use g2g_plugins::srtp::{SrtpFlow, SrtpMasterKey, SrtpPolicy};
use g2g_plugins::srtpdec::SrtpDec;
use g2g_plugins::srtpenc::SrtpEnc;

// The pieces a `key=` is built from, each spelled once. They are macros rather
// than constants because `concat!` takes only literals.
macro_rules! aes_128_master_key {
    () => {
        "000102030405060708090a0b0c0d0e0f"
    };
}
macro_rules! aes_256_master_key {
    () => {
        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
    };
}
/// The 12-byte master salt the AES-GCM profiles take, and the 14-byte one the
/// counter-mode profiles take.
macro_rules! aead_master_salt {
    () => {
        "517569642070726f2071756f"
    };
}
macro_rules! counter_mode_master_salt {
    () => {
        concat!(aead_master_salt!(), "2121")
    };
}

/// The key every leg uses unless it is testing a key change or another cipher:
/// the master key then the master salt, the 56 hexadecimal digits an
/// AES-128-GCM `key=` carries.
pub(crate) const MASTER_KEY_HEX: &str = concat!(aes_128_master_key!(), aead_master_salt!());
/// A second key, for a leg that needs two streams or two MKI-tagged keys.
pub(crate) const SECOND_KEY_HEX: &str =
    concat!("ffeeddccbbaa99887766554433221100", aead_master_salt!());
/// 60 digits: gst's default key length, an `aes-128-icm` key and its 14-byte
/// master salt.
pub(crate) const COUNTER_MODE_KEY_HEX: &str =
    concat!(aes_128_master_key!(), counter_mode_master_salt!());
/// 92 digits, the `aes-256-icm` length.
pub(crate) const COUNTER_MODE_256_KEY_HEX: &str =
    concat!(aes_256_master_key!(), counter_mode_master_salt!());
/// 88 digits, the AES-256-GCM length.
pub(crate) const AES_256_GCM_KEY_HEX: &str = concat!(aes_256_master_key!(), aead_master_salt!());

pub(crate) const SYNCHRONIZATION_SOURCE: u32 = 0x1020_3040;
pub(crate) const SECOND_SYNCHRONIZATION_SOURCE: u32 = 0x5060_7080;
pub(crate) const PAYLOAD_TYPE: u8 = 96;
/// 90 kHz, the RTP video clock, so successive packets carry distinct stamps.
pub(crate) const TIMESTAMP_STEP: u32 = 3000;
pub(crate) const PACKET_COUNT: usize = 8;
/// Sequence an RTP leg starts at to cross the wrap: three short of it, so a run
/// of [`PACKET_COUNT`] packets carries the rollover counter to 1.
pub(crate) const WRAP_START_SEQUENCE: u16 = u16::MAX - 2;

/// Deep enough that the arms interleave, shallow enough that a stall shows up
/// as a hang in one test rather than a whole suite.
pub(crate) const LINK_CAPACITY: usize = 2;

/// How long a `gst-launch-1.0` peer gets to finish writing its files.
pub(crate) const PEER_DEADLINE_SECONDS: u64 = 60;
/// Buffers the interop legs ask `audiotestsrc` for.
pub(crate) const PEER_BUFFERS: &str = "20";

pub(crate) struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

pub(crate) fn plain_caps(flow: SrtpFlow) -> Caps {
    Caps::ByteStream {
        encoding: flow.plain_encoding(),
    }
}

pub(crate) fn protected_caps(flow: SrtpFlow) -> Caps {
    Caps::ByteStream {
        encoding: flow.protected_encoding(),
    }
}

/// Pushes one prepared packet per `DataFrame`, then EOS: the shape an SRTP
/// element sees on a real link, where one frame is one datagram.
#[derive(Debug)]
pub(crate) struct PacketSource {
    packets: VecDeque<Vec<u8>>,
    caps: Caps,
    pacing: Option<std::time::Duration>,
}

impl PacketSource {
    pub(crate) fn new(packets: Vec<Vec<u8>>, caps: Caps) -> Self {
        Self {
            packets: packets.into(),
            caps,
            pacing: None,
        }
    }

    /// Wait this long between packets, so a leg whose element needs wall time to
    /// reach a working state (a DTLS handshake) still gets its stream. Needs a
    /// tokio runtime, which every `#[tokio::test]` leg has.
    pub(crate) fn with_pacing(mut self, gap: std::time::Duration) -> Self {
        self.pacing = Some(gap);
        self
    }
}

impl SourceLoop for PacketSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(self.caps.clone()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut pushed = 0;
            while let Some(packet) = self.packets.pop_front() {
                if let Some(gap) = self.pacing {
                    tokio::time::sleep(gap).await;
                }
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(packet.into_boxed_slice())),
                    FrameTiming::default(),
                    pushed,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
                pushed += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(pushed)
        })
    }
}

/// Keeps the bytes of every frame that reached it, so a leg can compare them
/// with what it sent.
#[derive(Debug, Default)]
pub(crate) struct CollectingSink {
    pub(crate) packets: Vec<Vec<u8>>,
}

impl AsyncElement for CollectingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                if let Some(slice) = frame.domain.as_system_slice() {
                    self.packets.push(slice.to_vec());
                }
            }
            Ok(())
        })
    }
}

pub(crate) fn rtp_packet(sequence: u16, synchronization_source: u32, payload: &[u8]) -> Vec<u8> {
    let header = RtpHeader {
        payload_type: PAYLOAD_TYPE,
        marker: false,
        sequence,
        timestamp: u32::from(sequence).wrapping_mul(TIMESTAMP_STEP),
        ssrc: synchronization_source,
    };
    let mut packet = header.to_bytes().to_vec();
    packet.extend_from_slice(payload);
    packet
}

/// `count` RTP packets starting at `start`, each carrying its own index so a
/// recovered packet identifies itself.
pub(crate) fn rtp_packets(start: u16, count: usize, synchronization_source: u32) -> Vec<Vec<u8>> {
    (0..count)
        .map(|index| {
            let payload = format!("packet {index:04} of this stream");
            rtp_packet(
                start.wrapping_add(index as u16),
                synchronization_source,
                payload.as_bytes(),
            )
        })
        .collect()
}

/// RTCP sender reports with no report block: version 2, the SR payload type,
/// and the six 32-bit words the fixed part occupies.
pub(crate) fn rtcp_packets(count: usize, synchronization_source: u32) -> Vec<Vec<u8>> {
    /// `RTCP SR`, and the length in 32-bit words after the first.
    const SENDER_REPORT_TYPE: u8 = 200;
    const SENDER_REPORT_WORDS: u16 = 6;

    (0..count)
        .map(|index| {
            let mut packet = Vec::new();
            packet.push(0x80);
            packet.push(SENDER_REPORT_TYPE);
            packet.extend_from_slice(&SENDER_REPORT_WORDS.to_be_bytes());
            packet.extend_from_slice(&synchronization_source.to_be_bytes());
            // NTP seconds / fraction, RTP timestamp, packet and octet counts.
            for word in 0..5_u32 {
                packet.extend_from_slice(&(index as u32 * 5 + word).to_be_bytes());
            }
            packet
        })
        .collect()
}

pub(crate) fn master_key(hexadecimal: &str) -> SrtpMasterKey {
    SrtpMasterKey::from_hexadecimal(hexadecimal).expect("the constant is valid key material")
}

pub(crate) fn encoder(hexadecimal: &str) -> SrtpEnc {
    let key = master_key(hexadecimal);
    SrtpEnc::new(key.policy(), key.master_key(), key.master_salt()).expect("valid encoder key")
}

pub(crate) fn decoder(hexadecimal: &str) -> SrtpDec {
    let key = master_key(hexadecimal);
    SrtpDec::new(key.policy(), key.master_key(), key.master_salt()).expect("valid decoder key")
}

pub(crate) fn rekey(element_key: &str) -> (SrtpPolicy, Vec<u8>, Vec<u8>) {
    let key = master_key(element_key);
    (
        key.policy(),
        key.master_key().to_vec(),
        key.master_salt().to_vec(),
    )
}

/// `source ! element ! sink` on the real runner, returning what reached the
/// sink. The element is borrowed, so a caller can rekey it and run again.
pub(crate) async fn run_one<E: AsyncElement>(
    element: &mut E,
    input_caps: Caps,
    packets: Vec<Vec<u8>>,
) -> Vec<Vec<u8>> {
    let mut source = PacketSource::new(packets, input_caps);
    let mut sink = CollectingSink::default();
    run_source_transform_sink(&mut source, element, &mut sink, &ZeroClock, LINK_CAPACITY)
        .await
        .expect("the pipeline runs");
    sink.packets
}

/// `source ! srtpenc ! srtpdec ! sink`, the whole protected link in one graph.
pub(crate) async fn run_protected_link(
    flow: SrtpFlow,
    encoder: &mut SrtpEnc,
    decoder: &mut SrtpDec,
    packets: Vec<Vec<u8>>,
) -> Vec<Vec<u8>> {
    let mut source = PacketSource::new(packets, plain_caps(flow));
    let mut sink = CollectingSink::default();

    let mut graph: Graph<GraphNodeRef<'_>> = Graph::new();
    let source_id = graph.add_source(GraphNodeRef::source_ref(&mut source));
    let encoder_id = graph.add_transform(GraphNodeRef::element_ref(encoder));
    let decoder_id = graph.add_transform(GraphNodeRef::element_ref(decoder));
    let sink_id = graph.add_sink(GraphNodeRef::element_ref(&mut sink));
    graph
        .link(source_id, encoder_id)
        .expect("source -> srtpenc");
    graph
        .link(encoder_id, decoder_id)
        .expect("srtpenc -> srtpdec");
    graph.link(decoder_id, sink_id).expect("srtpdec -> sink");
    run_graph(graph, &ZeroClock, LINK_CAPACITY)
        .await
        .expect("the protected link runs");

    sink.packets
}

pub(crate) fn scratch_path(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(name);
    let _ = std::fs::remove_file(&path);
    path
}

// GStreamer interop. A loopback between two g2g ends cannot catch a wire-format
// bug, so each leg puts `gst-launch-1.0`'s libsrtp on the other side.

/// Directory the interop fixtures are written to, emptied first.
pub(crate) fn peer_directory(name: &str) -> Option<PathBuf> {
    if Command::new("gst-launch-1.0")
        .arg("--version")
        .output()
        .is_err()
    {
        println!("skipping: gst-launch-1.0 is not installed");
        return None;
    }
    let path = std::env::temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("create the interop directory");
    Some(path)
}

pub(crate) fn run_peer(directory: &Path, description: &str) {
    let mut child = Command::new("gst-launch-1.0")
        .arg("-q")
        .args(description.split_whitespace())
        .current_dir(directory)
        .spawn()
        .expect("start gst-launch-1.0");
    let deadline =
        std::time::Instant::now() + std::time::Duration::from_secs(PEER_DEADLINE_SECONDS);
    loop {
        match child.try_wait().expect("poll gst-launch-1.0") {
            Some(status) => {
                assert!(status.success(), "gst-launch-1.0 failed: {status}");
                return;
            }
            None if std::time::Instant::now() > deadline => {
                let _ = child.kill();
                panic!("gst-launch-1.0 did not finish within {PEER_DEADLINE_SECONDS}s");
            }
            None => std::thread::sleep(std::time::Duration::from_millis(50)),
        }
    }
}

/// Every `prefix%05d.bin` file the peer wrote, in index order.
pub(crate) fn numbered_files(directory: &Path, prefix: &str) -> Vec<Vec<u8>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(directory)
        .expect("read the interop directory")
        .map(|entry| entry.expect("directory entry").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(prefix))
        })
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|path| std::fs::read(path).expect("read a packet file"))
        .collect()
}

pub(crate) fn write_numbered_files(directory: &Path, prefix: &str, packets: &[Vec<u8>]) {
    for (index, packet) in packets.iter().enumerate() {
        std::fs::write(directory.join(format!("{prefix}{index:05}.bin")), packet)
            .expect("write a packet file");
    }
}

/// The counter-mode cipher a key of this length keys.
fn counter_mode_cipher(key: &str) -> &'static str {
    if key.len() == COUNTER_MODE_256_KEY_HEX.len() {
        GST_AES_256_COUNTER_MODE
    } else {
        GST_AES_128_COUNTER_MODE
    }
}

/// The cipher and authentication gst runs on each flow. Both carry the leg's
/// pair, except that a NULL cipher on both flows makes gst size its `key`
/// buffer at zero bytes, which a launch line cannot spell: the flow the leg
/// does not exercise then keeps the counter-mode cipher the key length keys.
fn peer_policies<'a>(
    key: &str,
    flow: SrtpFlow,
    cipher: &'a str,
    authentication: &'a str,
) -> ((&'a str, &'a str), (&'a str, &'a str)) {
    let leg = (cipher, authentication);
    if cipher != GST_NULL {
        return (leg, leg);
    }
    let other = (counter_mode_cipher(key), GST_HMAC_SHA1_80);
    match flow {
        SrtpFlow::Rtp => (leg, other),
        SrtpFlow::Rtcp => (other, leg),
    }
}

/// The `srtpenc` arguments a leg keys gst with, for the flow it exercises.
pub(crate) fn peer_arguments(
    key: &str,
    flow: SrtpFlow,
    cipher: &str,
    authentication: &str,
) -> String {
    let ((rtp_cipher, rtp_auth), (rtcp_cipher, rtcp_auth)) =
        peer_policies(key, flow, cipher, authentication);
    format!(
        "key={key} rtp-cipher={rtp_cipher} rtp-auth={rtp_auth} \
         rtcp-cipher={rtcp_cipher} rtcp-auth={rtcp_auth}"
    )
}

/// The `srtpenc` arguments the AES-GCM interop legs key, with the ciphers g2g
/// takes from the key length spelled out for gst.
pub(crate) fn peer_cipher_arguments() -> String {
    format!(
        "key={MASTER_KEY_HEX} rtp-cipher={GST_AES_128_GCM} rtp-auth={GST_NULL} \
         rtcp-cipher={GST_AES_128_GCM} rtcp-auth={GST_NULL}"
    )
}

/// The sink-pad caps gst's `srtpdec` reads its key from, without the `caps=`
/// a `multifilesrc` needs in front of them, since a `filesrc` takes them as a
/// caps filter instead. They must also name the stream's `ssrc`: without it the
/// element asks its `request-key` signal and drops every packet.
pub(crate) fn peer_decoder_caps(
    flow: SrtpFlow,
    synchronization_source: u32,
    key: &str,
    cipher: &str,
    authentication: &str,
) -> String {
    let encoding = match flow {
        SrtpFlow::Rtp => GST_SRTP_CAPS,
        SrtpFlow::Rtcp => GST_SRTCP_CAPS,
    };
    let ((rtp_cipher, rtp_auth), (rtcp_cipher, rtcp_auth)) =
        peer_policies(key, flow, cipher, authentication);
    format!(
        "{encoding},ssrc=(uint){synchronization_source},srtp-key=(buffer){key},\
srtp-cipher=(string){rtp_cipher},srtp-auth=(string){rtp_auth},\
srtcp-cipher=(string){rtcp_cipher},srtcp-auth=(string){rtcp_auth},roc=(uint)0"
    )
}

/// The `GstSrtpCipherType` and `GstSrtpAuthType` nicks a gst leg spells.
pub(crate) const GST_NULL: &str = "null";
pub(crate) const GST_AES_128_GCM: &str = "aes-128-gcm";
pub(crate) const GST_AES_128_COUNTER_MODE: &str = "aes-128-icm";
pub(crate) const GST_AES_256_COUNTER_MODE: &str = "aes-256-icm";
pub(crate) const GST_HMAC_SHA1_80: &str = "hmac-sha1-80";
pub(crate) const GST_HMAC_SHA1_32: &str = "hmac-sha1-32";
/// The caps a gst leg names for each protected flow.
const GST_SRTP_CAPS: &str = "application/x-srtp";
const GST_SRTCP_CAPS: &str = "application/x-srtcp";
