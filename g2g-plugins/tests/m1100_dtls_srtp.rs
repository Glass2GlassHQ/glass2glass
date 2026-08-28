//! M1100: DTLS-SRTP key delivery, the `dtlssrtpenc` / `dtlssrtpdec` pair.
//!
//! A handshake runs over the media socket and the RFC 7714 layer is keyed from
//! the RFC 5764 export, so no `key=` is set anywhere. The loopback legs put two
//! g2g graphs on `127.0.0.1`, one as the DTLS client and one as the server.
//!
//! The `#[ignore]`d legs put `gst-launch-1.0`'s `dtlssrtpenc` / `dtlssrtpdec` on
//! the other side, which a g2g <-> g2g loopback cannot check: both ends would
//! share a bug. Run:
//!
//! ```sh
//! cargo test -p g2g-plugins --features dtls-srtp,udp-ingress,udp-egress \
//!     --test m1100_dtls_srtp
//! cargo test -p g2g-plugins --features dtls-srtp,udp-ingress,udp-egress \
//!     --test m1100_dtls_srtp -- --ignored --nocapture
//! ```
#![cfg(all(feature = "dtls-srtp", feature = "udp-ingress", feature = "udp-egress"))]

mod srtp_common;

use std::time::{Duration, Instant};

use g2g_plugins::dtlssrtp::{
    connection_for_id, lock_connection, DtlsSrtpConnection, DtlsSrtpHandle, DtlsSrtpState,
};

/// How long the in-process handshake legs get before they are called stalled.
const HANDSHAKE_DEADLINE: Duration = Duration::from_secs(5);
/// How often the in-process exchange runs each end's retransmission timer.
const HANDSHAKE_STEP: Duration = Duration::from_millis(5);

/// Move every DTLS record one end produced to the other, running both timers,
/// until both are connected or the deadline passes. Returns whether they
/// connected, so a leg that expects a failure can assert on the state instead.
fn exchange_until_connected(client: &DtlsSrtpHandle, server: &DtlsSrtpHandle) -> bool {
    let deadline = Instant::now() + HANDSHAKE_DEADLINE;
    while Instant::now() < deadline {
        let now = Instant::now();
        let mut moved = false;
        for (from, to) in [(client, server), (server, client)] {
            let records = {
                let mut end = lock_connection(from).expect("lock the sending end");
                if end.drive(now).is_err() {
                    return false;
                }
                let mut records = Vec::new();
                while let Some(record) = end.take_outbound() {
                    records.push(record);
                }
                records
            };
            moved |= !records.is_empty();
            let mut peer = lock_connection(to).expect("lock the receiving end");
            for record in records {
                if peer.handle_datagram(&record, now).is_err() {
                    return false;
                }
            }
        }
        let states = [client, server].map(|end| {
            lock_connection(end)
                .expect("lock an end to read its state")
                .state()
        });
        if states
            .iter()
            .all(|state| *state == DtlsSrtpState::Connected)
            && [client, server].iter().all(|end| {
                lock_connection(end)
                    .expect("lock an end to read its keys")
                    .keys()
                    .is_some()
            })
        {
            return true;
        }
        if states.contains(&DtlsSrtpState::Failed) {
            return false;
        }
        if !moved {
            std::thread::sleep(HANDSHAKE_STEP);
        }
    }
    false
}

fn client_and_server() -> (DtlsSrtpHandle, DtlsSrtpHandle) {
    let client = DtlsSrtpConnection::shared();
    let server = DtlsSrtpConnection::shared();
    lock_connection(&client)
        .expect("lock the client")
        .set_client(true);
    (client, server)
}

/// The RFC 5764 split: each end protects with its own half of the exported block
/// and recovers the peer's packets with the other half.
#[test]
fn a_handshake_keys_both_directions() {
    let (client, server) = client_and_server();
    assert!(
        exchange_until_connected(&client, &server),
        "the in-process handshake has to complete"
    );

    let client_end = lock_connection(&client).expect("lock the client");
    let server_end = lock_connection(&server).expect("lock the server");
    let client_keys = client_end.keys().expect("the client exported keys");
    let server_keys = server_end.keys().expect("the server exported keys");

    assert_eq!(
        client_keys.send.master_key(),
        server_keys.receive.master_key(),
        "the client sends under the key the server receives with"
    );
    assert_eq!(
        client_keys.send.master_salt(),
        server_keys.receive.master_salt()
    );
    assert_eq!(
        server_keys.send.master_key(),
        client_keys.receive.master_key(),
        "the server sends under the key the client receives with"
    );
    assert_eq!(
        server_keys.send.master_salt(),
        client_keys.receive.master_salt()
    );
    assert_ne!(
        client_keys.send.master_key(),
        client_keys.receive.master_key(),
        "the two directions are keyed independently"
    );
    assert_eq!(
        client_keys.send.policy(),
        server_keys.send.policy(),
        "both ends took the same protection profile from the handshake"
    );
    assert!(
        client_end.peer_certificate_pem().is_some(),
        "the client saw the server's certificate"
    );
    assert!(
        server_end.peer_certificate_pem().is_some(),
        "the server saw the client's certificate"
    );
}

/// The fingerprint signalling carried names the peer's own certificate, so the
/// handshake completes and keys the media.
#[test]
fn a_matching_fingerprint_keys_the_connection() {
    let (client, server) = client_and_server();
    let peer_fingerprint = lock_connection(&server)
        .expect("lock the server")
        .local_fingerprint()
        .expect("the server has a certificate");
    lock_connection(&client)
        .expect("lock the client")
        .set_expected_peer_fingerprint(
            peer_fingerprint
                .as_slice()
                .try_into()
                .expect("a SHA-256 fingerprint"),
        );

    assert!(
        exchange_until_connected(&client, &server),
        "the peer's own fingerprint has to be accepted"
    );
    assert!(lock_connection(&client)
        .expect("lock the client")
        .keys()
        .is_some());
}

/// A peer whose certificate does not hash to the expected fingerprint is
/// refused, and the connection stays failed rather than keying the media.
#[test]
fn a_fingerprint_mismatch_fails_the_connection() {
    let (client, server) = client_and_server();
    let mut wrong_fingerprint = lock_connection(&server)
        .expect("lock the server")
        .local_fingerprint()
        .expect("the server has a certificate");
    wrong_fingerprint[0] ^= 0xff;
    lock_connection(&client)
        .expect("lock the client")
        .set_expected_peer_fingerprint(
            wrong_fingerprint
                .as_slice()
                .try_into()
                .expect("a SHA-256 fingerprint"),
        );

    assert!(
        !exchange_until_connected(&client, &server),
        "a mismatched fingerprint must not produce a connected session"
    );
    let client_end = lock_connection(&client).expect("lock the client");
    assert_eq!(client_end.state(), DtlsSrtpState::Failed);
    assert!(
        client_end.keys().is_none(),
        "a failed connection hands out no key material"
    );
}

/// The `connection-id` pairing: two elements naming one id drive one handshake.
#[test]
fn a_connection_id_pairs_two_elements() {
    let encoder_side = connection_for_id("m1100-pairing").expect("a connection");
    let decoder_side = connection_for_id("m1100-pairing").expect("the same connection");
    lock_connection(&encoder_side)
        .expect("lock the connection")
        .set_client(true);
    assert!(
        lock_connection(&decoder_side)
            .expect("lock through the other handle")
            .is_client(),
        "both elements see one connection, so the role set on either applies"
    );
}

// The loopback legs: two g2g graphs per side on 127.0.0.1, one side the DTLS
// client and one the server, the protected media riding the handshake's socket.

use std::net::UdpSocket as StdUdpSocket;

use g2g_core::runtime::{parse_launch, run_graph, GraphNodeRef};
use g2g_core::{ByteStreamEncoding, Graph};
use g2g_plugins::clock::WallClock;
use g2g_plugins::dtlssrtp::DtlsSrtpHandle as Handle;
use g2g_plugins::dtlssrtpdec::{DtlsSrtpDec, RTCP_PORT, RTP_PORT};
use g2g_plugins::dtlssrtpenc::DtlsSrtpEnc;
use g2g_plugins::registry::default_registry;
use g2g_plugins::srtp::SrtpFlow;
use g2g_plugins::udpsink::UdpSink;
use g2g_plugins::udpsrc::UdpSrc;

use srtp_common::{
    plain_caps, rtcp_packets, rtp_packets, CollectingSink, PacketSource, ZeroClock, LINK_CAPACITY,
    PACKET_COUNT, SECOND_SYNCHRONIZATION_SOURCE, SYNCHRONIZATION_SOURCE,
};

/// The encoder's input pads, and the decoder's output ports, in the order the
/// two elements declare them.
const RTP_PAD: u8 = 0;
const RTCP_PAD: u8 = 1;

/// The RTP sequence a loopback leg starts at.
const START_SEQUENCE: u16 = 1000;
/// Gap between packets, so the first ones are held while the handshake runs and
/// the rest are protected once it keyed the stream.
const PACKET_GAP: Duration = Duration::from_millis(25);
/// How long each receiving graph runs before it is stopped and read. Longer than
/// the paced send plus a handshake on the loopback interface.
const RECEIVE_DEADLINE: Duration = Duration::from_secs(8);

/// One side's sending half: two paced flows into the encoder, out over UDP.
struct SendHalf {
    rtp_source: PacketSource,
    rtcp_source: PacketSource,
    encoder: DtlsSrtpEnc,
    sink: UdpSink,
    sent_rtp: Vec<Vec<u8>>,
    sent_rtcp: Vec<Vec<u8>>,
}

impl SendHalf {
    fn new(
        connection: Handle,
        is_client: bool,
        synchronization_source: u32,
        peer_port: u16,
        pacing: Duration,
        peer_fingerprint: Option<[u8; FINGERPRINT_LENGTH]>,
    ) -> Self {
        let sent_rtp = rtp_packets(START_SEQUENCE, PACKET_COUNT, synchronization_source);
        let sent_rtcp = rtcp_packets(PACKET_COUNT, synchronization_source);
        let mut encoder = DtlsSrtpEnc::new(2)
            .with_connection(connection)
            .with_client_role(is_client);
        if let Some(fingerprint) = peer_fingerprint {
            encoder = encoder.with_peer_fingerprint(fingerprint);
        }
        Self {
            rtp_source: PacketSource::new(sent_rtp.clone(), plain_caps(SrtpFlow::Rtp))
                .with_pacing(pacing),
            rtcp_source: PacketSource::new(sent_rtcp.clone(), plain_caps(SrtpFlow::Rtcp))
                .with_pacing(pacing),
            encoder,
            sink: UdpSink::new(
                format!("127.0.0.1:{peer_port}")
                    .parse()
                    .expect("a loopback address"),
            ),
            sent_rtp,
            sent_rtcp,
        }
    }

    /// `rtp ! e.  rtcp ! e.  dtlssrtpenc name=e ! udpsink`.
    fn graph(&mut self) -> Graph<GraphNodeRef<'_>> {
        let mut graph: Graph<GraphNodeRef<'_>> = Graph::new();
        let rtp = graph.add_source(GraphNodeRef::source_ref(&mut self.rtp_source));
        let rtcp = graph.add_source(GraphNodeRef::source_ref(&mut self.rtcp_source));
        let encoder = graph.add_muxer(GraphNodeRef::muxer_ref(&mut self.encoder), 2);
        let sink = graph.add_sink(GraphNodeRef::element_ref(&mut self.sink));
        graph
            .link(rtp, encoder.input(RTP_PAD))
            .expect("rtp -> dtlssrtpenc");
        graph
            .link(rtcp, encoder.input(RTCP_PAD))
            .expect("rtcp -> dtlssrtpenc");
        graph
            .link(encoder.output(), sink)
            .expect("dtlssrtpenc -> udpsink");
        graph
    }
}

/// One side's receiving half: the shared socket in, recovered RTP and RTCP out.
struct ReceiveHalf {
    source: UdpSrc,
    decoder: DtlsSrtpDec,
    rtp: CollectingSink,
    rtcp: CollectingSink,
}

impl ReceiveHalf {
    fn new(connection: Handle, socket: StdUdpSocket) -> Self {
        Self {
            source: UdpSrc::from_socket(socket)
                .expect("adopt the socket")
                .with_bytestream(ByteStreamEncoding::Dtls),
            decoder: DtlsSrtpDec::new(2).with_connection(connection),
            rtp: CollectingSink::default(),
            rtcp: CollectingSink::default(),
        }
    }

    /// `udpsrc bytestream-format=dtls ! dtlssrtpdec name=d  d. ! sink  d. ! sink`.
    fn graph(&mut self) -> Graph<GraphNodeRef<'_>> {
        let mut graph: Graph<GraphNodeRef<'_>> = Graph::new();
        let udp = graph.add_source(GraphNodeRef::source_ref(&mut self.source));
        let decoder = graph.add_demux(GraphNodeRef::demux_ref(&mut self.decoder), 2);
        let rtp = graph.add_sink(GraphNodeRef::element_ref(&mut self.rtp));
        let rtcp = graph.add_sink(GraphNodeRef::element_ref(&mut self.rtcp));
        graph
            .link(udp, decoder.input())
            .expect("udpsrc -> dtlssrtpdec");
        graph
            .link(decoder.out(RTP_PAD), rtp)
            .expect("the recovered rtp port");
        graph
            .link(decoder.out(RTCP_PAD), rtcp)
            .expect("the recovered rtcp port");
        graph
    }
}

/// Every packet that arrived has to be one that was sent, in order. The first
/// packets are pushed before the handshake keys the stream, so a run may start
/// late; what arrives has to be a contiguous tail of what was sent.
fn assert_recovered(sent: &[Vec<u8>], received: &[Vec<u8>], flow: &str) {
    assert!(
        !received.is_empty(),
        "no {flow} packet crossed the protected link"
    );
    assert!(
        received.len() <= sent.len(),
        "{flow}: {} packets received for {} sent",
        received.len(),
        sent.len()
    );
    assert_eq!(
        &sent[sent.len() - received.len()..],
        received,
        "{flow}: the recovered packets differ from the ones sent"
    );
}

/// Two g2g sides exchange RTP and RTCP protected by keys a DTLS handshake
/// delivered over the same UDP socket, with no `key=` set anywhere.
#[tokio::test]
async fn dtls_srtp_crosses_a_udp_loopback() {
    let socket_client = StdUdpSocket::bind("127.0.0.1:0").expect("bind the client socket");
    let socket_server = StdUdpSocket::bind("127.0.0.1:0").expect("bind the server socket");
    let port_client = socket_client.local_addr().expect("the client port").port();
    let port_server = socket_server.local_addr().expect("the server port").port();

    let client_connection =
        connection_for_id("m1100-loopback-client").expect("the client connection");
    let server_connection =
        connection_for_id("m1100-loopback-server").expect("the server connection");

    let mut client_send = SendHalf::new(
        client_connection.clone(),
        true,
        SYNCHRONIZATION_SOURCE,
        port_server,
        PACKET_GAP,
        None,
    );
    let mut server_send = SendHalf::new(
        server_connection.clone(),
        false,
        SECOND_SYNCHRONIZATION_SOURCE,
        port_client,
        PACKET_GAP,
        None,
    );
    let mut client_receive = ReceiveHalf::new(client_connection, socket_client);
    let mut server_receive = ReceiveHalf::new(server_connection, socket_server);

    // The sending graphs need a clock that can sleep: the handshake advances on
    // the encoder's tick, not only when a media packet arrives.
    let clock = WallClock::new();
    let receive_client = tokio::time::timeout(
        RECEIVE_DEADLINE,
        run_graph(client_receive.graph(), &ZeroClock, LINK_CAPACITY),
    );
    let receive_server = tokio::time::timeout(
        RECEIVE_DEADLINE,
        run_graph(server_receive.graph(), &ZeroClock, LINK_CAPACITY),
    );
    let send_client = run_graph(client_send.graph(), &clock, LINK_CAPACITY);
    let send_server = run_graph(server_send.graph(), &clock, LINK_CAPACITY);

    let (_, _, client_sent, server_sent) =
        tokio::join!(receive_client, receive_server, send_client, send_server);
    client_sent.expect("the client sending pipeline runs");
    server_sent.expect("the server sending pipeline runs");
    for (side, stats) in [
        ("client", client_receive.decoder.stats()),
        ("server", server_receive.decoder.stats()),
    ] {
        assert!(
            stats.records_handled > 0,
            "the {side} answered no handshake record"
        );
        assert_eq!(stats.packets_dropped, 0, "the {side} dropped a datagram");
    }

    // What the client sent arrived at the server, and the other way round.
    assert_recovered(
        &client_send.sent_rtp,
        &server_receive.rtp.packets,
        "client rtp",
    );
    assert_recovered(
        &client_send.sent_rtcp,
        &server_receive.rtcp.packets,
        "client rtcp",
    );
    assert_recovered(
        &server_send.sent_rtp,
        &client_receive.rtp.packets,
        "server rtp",
    );
    assert_recovered(
        &server_send.sent_rtcp,
        &client_receive.rtcp.packets,
        "server rtcp",
    );
}

/// A launch line names the pair by `connection-id`, and both halves parse,
/// negotiate and run: the encoder's two typed pads, the decoder's two ports.
#[tokio::test]
async fn a_launch_line_builds_both_halves() {
    let registry = default_registry();
    let connection_id = "m1100-launch";
    let peer_port = free_port();

    let sender = parse_launch(
        &registry,
        &format!(
            "udpsrc port=0 bytestream-format=rtp num-buffers=0 ! e. \
             udpsrc port=0 bytestream-format=rtcp num-buffers=0 ! e. \
             dtlssrtpenc name=e connection-id={connection_id} is-client=true \
             ! udpsink host=127.0.0.1 port={peer_port}"
        ),
    )
    .expect("the sending line parses");
    let receiver = parse_launch(
        &registry,
        &format!(
            "udpsrc port=0 bytestream-format=dtls num-buffers=0 \
             ! dtlssrtpdec name=d connection-id={connection_id} \
             d.src_0 ! fakesink  d.src_1 ! fakesink"
        ),
    )
    .expect("the receiving line parses");

    run_graph(sender, &ZeroClock, LINK_CAPACITY)
        .await
        .expect("the sending line runs");
    run_graph(receiver, &ZeroClock, LINK_CAPACITY)
        .await
        .expect("the receiving line runs");
}

/// A line that links one branch gets one port, so the flow the other would have
/// carried has nowhere to go and the element must not reach for it.
#[test]
fn a_decoder_declares_only_the_ports_its_graph_linked() {
    use g2g_core::MultiOutputElement;

    let both = DtlsSrtpDec::new(2);
    assert!(both.port_output_caps(RTP_PORT).is_some());
    assert!(both.port_output_caps(RTCP_PORT).is_some());

    let rtp_only = DtlsSrtpDec::new(1);
    assert!(rtp_only.port_output_caps(RTP_PORT).is_some());
    assert!(
        rtp_only.port_output_caps(RTCP_PORT).is_none(),
        "a one-branch line has no port for the RTCP flow"
    );
}

/// A port nothing listens on, for a line that has to name a destination.
fn free_port() -> u16 {
    StdUdpSocket::bind("127.0.0.1:0")
        .expect("bind an ephemeral port")
        .local_addr()
        .expect("the bound address")
        .port()
}

// GStreamer interop. A g2g <-> g2g loopback cannot catch a negotiation or
// wire-format bug, so each leg puts GStreamer's own DTLS stack on the other
// side, once with g2g as the client and once as the server. gst's
// `libgstdtls.so` offers only `SRTP_AES128_CM_SHA1_80`, so these legs are what
// prove the RFC 3711 packet layer keys from an RFC 5764 export.
//
// The peer runs through GStreamer's Python bindings rather than
// `gst-launch-1.0`: the certificate has to be an ECDSA one (gst's default `pem`
// is RSA and `dimpl` offers only ECDHE-ECDSA suites), and `gst-launch-1.0`
// cannot carry a PEM. It escapes an argument containing spaces but not one
// containing newlines, so a multi-line value is split across tokens and the
// pipeline fails to parse. Setting the property through `Gst.parse_launch` plus
// `set_property` builds the same `dtlssrtpenc` / `dtlssrtpdec` bins.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use g2g_plugins::dtlssrtp::{parse_fingerprint, DtlsSrtpError, FINGERPRINT_LENGTH};
use g2g_plugins::srtp::{SrtpAuthentication, SrtpCipher, SrtpPolicy};

/// How long a leg gives the two peers to exchange their paced media.
const INTEROP_DEADLINE: Duration = Duration::from_secs(20);
/// Gap between the interop leg's packets. The g2g side only sends DTLS records
/// while its sending graph runs, so the stream is paced to keep that graph alive
/// across the whole handshake.
const INTEROP_PACKET_GAP: Duration = Duration::from_millis(500);
/// How long the peer gets to bind its socket before g2g starts sending. UDP
/// drops what nobody is listening for.
const PEER_LISTEN_WAIT: Duration = Duration::from_millis(1500);
/// How long the peer's own main loop runs before it stops itself, so a leg that
/// panics does not leave a process behind.
const PEER_LIFETIME_SECONDS: &str = "25";
/// The curve the peer certificate is generated on: `dimpl` offers only
/// ECDHE-ECDSA cipher suites, so an RSA peer certificate shares none with it.
const PEER_CURVE: &str = "prime256v1";
/// Days the generated peer certificate is valid for.
const PEER_CERTIFICATE_DAYS: &str = "1";
/// The one DTLS-SRTP protection profile GStreamer's `dtls` plugin offers.
const GST_PROTECTION_PROFILE: &str = "SRTP_AES128_CM_SHA1_80";
/// What that profile keys, RFC 5764 section 4.1.2.
const GST_PROTECTION_POLICY: SrtpPolicy = SrtpPolicy {
    cipher: SrtpCipher::Aes128CounterMode,
    authentication: SrtpAuthentication::HmacSha1Tag80,
};
/// Where the peer writes the RTP and RTCP it recovered, and the RTP it sent in
/// the clear before protecting it.
const PEER_RECOVERED_RTP: &str = "peer-rtp";
const PEER_RECOVERED_RTCP: &str = "peer-rtcp";
const PEER_PLAIN_RTP: &str = "peer-plain";

/// The peer: GStreamer's `dtlssrtpdec` / `dtlssrtpenc` pair around a UDP socket,
/// with the certificate set as a property because a PEM cannot travel through a
/// launch line. Every packet it recovers and every one it protects is written
/// out, so the leg can check the media byte for byte in both directions.
const PEER_SCRIPT: &str = r#"
import sys
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

pem_path, peer_port, g2g_port, is_client, seconds = sys.argv[1:6]
Gst.init(None)
pipeline = Gst.parse_launch(
    f"udpsrc address=127.0.0.1 port={peer_port} ! dtlssrtpdec name=d connection-id=peer "
    "d.rtp_src ! multifilesink location=peer-rtp%05d.bin async=false "
    "d.rtcp_src ! multifilesink location=peer-rtcp%05d.bin async=false "
    "audiotestsrc is-live=true ! alawenc ! rtppcmapay ! tee name=t "
    "t. ! queue ! multifilesink location=peer-plain%05d.bin async=false "
    "t. ! queue ! e.rtp_sink_0 "
    f"dtlssrtpenc name=e connection-id=peer is-client={is_client} "
    f"! udpsink host=127.0.0.1 port={g2g_port}"
)
pipeline.get_by_name("d").set_property("pem", open(pem_path).read())
pipeline.set_state(Gst.State.PLAYING)
loop = GLib.MainLoop()
GLib.timeout_add_seconds(int(seconds), lambda: (loop.quit(), False)[1])
def on_message(_bus, message):
    if message.type == Gst.MessageType.ERROR:
        print("peer error:", message.parse_error()[0].message, flush=True)
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_message)
loop.run()
pipeline.set_state(Gst.State.NULL)
"#;

/// Whether this host can run the peer at all.
fn peer_is_runnable() -> bool {
    let bindings = Command::new("python3")
        .args([
            "-c",
            "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    if !matches!(&bindings, Ok(status) if status.success()) {
        println!("skipping: GStreamer's Python bindings are not installed");
        return false;
    }
    true
}

/// An ECDSA P-256 certificate and its key in one PEM, generated under
/// `directory`. `None` when `openssl` is missing. Never checked in: a private
/// key in the repository is a private key everyone has.
fn peer_certificate_pem(directory: &Path) -> Option<(PathBuf, PathBuf)> {
    let key = directory.join("peer-key.pem");
    let certificate = directory.join("peer-cert.pem");
    let generated = Command::new("openssl")
        .args(["ecparam", "-name", PEER_CURVE, "-genkey", "-noout", "-out"])
        .arg(&key)
        .status();
    if !matches!(&generated, Ok(status) if status.success()) {
        println!("skipping: openssl is not installed");
        return None;
    }
    let signed = Command::new("openssl")
        .args([
            "req",
            "-new",
            "-x509",
            "-days",
            PEER_CERTIFICATE_DAYS,
            "-subj",
            "/CN=g2g-peer",
        ])
        .arg("-key")
        .arg(&key)
        .arg("-out")
        .arg(&certificate)
        .status()
        .expect("run openssl req");
    assert!(signed.success(), "openssl could not sign the certificate");

    let pem = directory.join("peer.pem");
    let mut bytes = std::fs::read(&certificate).expect("read the certificate");
    bytes.extend_from_slice(&std::fs::read(&key).expect("read the private key"));
    std::fs::write(&pem, bytes).expect("write the peer PEM");
    Some((pem, certificate))
}

/// The peer certificate's SHA-256 fingerprint, read from `openssl` rather than
/// from the same code the element uses.
fn certificate_fingerprint(certificate: &Path) -> [u8; FINGERPRINT_LENGTH] {
    let printed = Command::new("openssl")
        .args(["x509", "-noout", "-fingerprint", "-sha256", "-in"])
        .arg(certificate)
        .output()
        .expect("run openssl x509");
    assert!(
        printed.status.success(),
        "openssl could not read the certificate"
    );
    let text = String::from_utf8(printed.stdout).expect("openssl prints text");
    let digest = text
        .split_once('=')
        .expect("openssl prints `sha256 Fingerprint=...`")
        .1;
    parse_fingerprint(digest.trim()).expect("openssl prints colon-separated octets")
}

fn spawn_peer(
    directory: &Path,
    pem: &Path,
    peer_port: u16,
    g2g_port: u16,
    g2g_is_client: bool,
) -> Child {
    let script = directory.join("peer.py");
    std::fs::write(&script, PEER_SCRIPT).expect("write the peer script");
    Command::new("python3")
        // The peer's `multifilesink` locations are relative, so it has to run
        // in the leg's own directory.
        .current_dir(directory)
        .arg(&script)
        .arg(pem)
        .arg(peer_port.to_string())
        .arg(g2g_port.to_string())
        .arg(if g2g_is_client { "false" } else { "true" })
        .arg(PEER_LIFETIME_SECONDS)
        .spawn()
        .expect("start the GStreamer peer")
}

/// What the g2g side pins the peer's certificate to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PinnedFingerprint {
    /// Accept whatever certificate the handshake presents.
    Any,
    /// The peer's own, read from its certificate with `openssl`.
    Peer,
    /// One byte of the peer's, flipped: the peer signalling did not name.
    Wrong,
}

/// Where one interop leg ended, and the media that crossed it.
struct InteropLeg {
    state: DtlsSrtpState,
    reason: String,
    policy: Option<SrtpPolicy>,
    g2g_sent_rtp: Vec<Vec<u8>>,
    g2g_sent_rtcp: Vec<Vec<u8>>,
    g2g_received_rtp: Vec<Vec<u8>>,
    peer_plain_rtp: Vec<Vec<u8>>,
    peer_recovered_rtp: Vec<Vec<u8>>,
    peer_recovered_rtcp: Vec<Vec<u8>>,
}

/// Run one leg against the GStreamer peer and report what crossed it.
async fn run_interop_leg(
    name: &str,
    g2g_is_client: bool,
    pinned: PinnedFingerprint,
) -> Option<InteropLeg> {
    if !peer_is_runnable() {
        return None;
    }
    let directory = srtp_common::peer_directory(name)?;
    let (pem, certificate) = peer_certificate_pem(&directory)?;
    let fingerprint = match pinned {
        PinnedFingerprint::Any => None,
        PinnedFingerprint::Peer => Some(certificate_fingerprint(&certificate)),
        PinnedFingerprint::Wrong => {
            let mut wrong = certificate_fingerprint(&certificate);
            wrong[0] ^= 0xff;
            Some(wrong)
        }
    };

    let g2g_socket = StdUdpSocket::bind("127.0.0.1:0").expect("bind the g2g socket");
    let g2g_port = g2g_socket.local_addr().expect("the g2g port").port();
    let peer_port = free_port();
    let mut peer = spawn_peer(&directory, &pem, peer_port, g2g_port, g2g_is_client);
    tokio::time::sleep(PEER_LISTEN_WAIT).await;

    let connection = connection_for_id(name).expect("the leg's connection");
    let mut send = SendHalf::new(
        connection.clone(),
        g2g_is_client,
        SYNCHRONIZATION_SOURCE,
        peer_port,
        INTEROP_PACKET_GAP,
        fingerprint,
    );
    let mut receive = ReceiveHalf::new(connection.clone(), g2g_socket);

    let clock = WallClock::new();
    let _ = tokio::time::timeout(INTEROP_DEADLINE, async {
        tokio::join!(
            run_graph(receive.graph(), &ZeroClock, LINK_CAPACITY),
            run_graph(send.graph(), &clock, LINK_CAPACITY),
        )
    })
    .await;
    let _ = peer.kill();
    let _ = peer.wait();

    let end = lock_connection(&connection).expect("lock the connection");
    let reason = end
        .failure()
        .map(|failure| failure.to_string())
        .unwrap_or_else(|| String::from("no failure"));
    println!(
        "g2g as {}, peer fingerprint {pinned:?}: state {}, profile {:?}, {reason}",
        if g2g_is_client { "client" } else { "server" },
        end.state().as_str(),
        end.policy()
    );
    Some(InteropLeg {
        state: end.state(),
        reason,
        policy: end.policy(),
        g2g_sent_rtp: send.sent_rtp.clone(),
        g2g_sent_rtcp: send.sent_rtcp.clone(),
        g2g_received_rtp: receive.rtp.packets.clone(),
        peer_plain_rtp: srtp_common::numbered_files(&directory, PEER_PLAIN_RTP),
        peer_recovered_rtp: srtp_common::numbered_files(&directory, PEER_RECOVERED_RTP),
        peer_recovered_rtcp: srtp_common::numbered_files(&directory, PEER_RECOVERED_RTCP),
    })
}

/// Every packet that arrived has to be one that was sent, in order and with
/// nothing in between. Neither side starts protecting until the handshake keys
/// it and both are stopped on a deadline, so what arrives is a run out of the
/// middle of what was sent rather than the whole list.
fn assert_contiguous_run(sent: &[Vec<u8>], received: &[Vec<u8>], flow: &str) {
    assert!(!received.is_empty(), "no {flow} packet crossed the link");
    assert!(
        sent.windows(received.len())
            .any(|window| window == received),
        "{flow}: the {} packets received are not a run of the {} sent",
        received.len(),
        sent.len()
    );
}

/// Both directions with GStreamer's own DTLS stack on the other side: the
/// handshake settles on `SRTP_AES128_CM_SHA1_80`, and the media crosses
/// byte-exact each way, RTCP included.
#[tokio::test]
#[ignore = "needs GStreamer's Python bindings and openssl"]
async fn gst_interop_carries_media_under_the_negotiated_profile() {
    for (name, g2g_is_client) in [
        ("m1100-interop-g2g-client", true),
        ("m1100-interop-g2g-server", false),
    ] {
        let Some(leg) = run_interop_leg(name, g2g_is_client, PinnedFingerprint::Any).await else {
            return;
        };
        assert_eq!(
            leg.state,
            DtlsSrtpState::Connected,
            "{name}: {}",
            leg.reason
        );
        assert_eq!(
            leg.policy,
            Some(GST_PROTECTION_POLICY),
            "{name}: the handshake has to settle on {GST_PROTECTION_PROFILE}"
        );

        assert_contiguous_run(
            &leg.g2g_sent_rtp,
            &leg.peer_recovered_rtp,
            &format!("{name} g2g -> gst rtp"),
        );
        assert_contiguous_run(
            &leg.g2g_sent_rtcp,
            &leg.peer_recovered_rtcp,
            &format!("{name} g2g -> gst rtcp"),
        );
        assert_contiguous_run(
            &leg.peer_plain_rtp,
            &leg.g2g_received_rtp,
            &format!("{name} gst -> g2g rtp"),
        );
        println!(
            "{name}: {GST_PROTECTION_PROFILE}, {} rtp and {} rtcp packets to gst, {} rtp back",
            leg.peer_recovered_rtp.len(),
            leg.peer_recovered_rtcp.len(),
            leg.g2g_received_rtp.len()
        );
    }
}

/// The `peer-fingerprint` pin against a real peer: the value signalling carried
/// lets the handshake through, and any other value stops the run.
#[tokio::test]
#[ignore = "needs GStreamer's Python bindings and openssl"]
async fn a_pinned_peer_fingerprint_gates_the_gst_handshake() {
    let Some(matching) = run_interop_leg(
        "m1100-interop-fingerprint-match",
        true,
        PinnedFingerprint::Peer,
    )
    .await
    else {
        return;
    };
    assert_eq!(
        matching.state,
        DtlsSrtpState::Connected,
        "the peer's own fingerprint has to be accepted: {}",
        matching.reason
    );
    assert_contiguous_run(
        &matching.g2g_sent_rtp,
        &matching.peer_recovered_rtp,
        "pinned rtp",
    );

    let Some(mismatched) = run_interop_leg(
        "m1100-interop-fingerprint-mismatch",
        true,
        PinnedFingerprint::Wrong,
    )
    .await
    else {
        return;
    };
    assert_eq!(
        mismatched.state,
        DtlsSrtpState::Failed,
        "a fingerprint the peer does not hash to must stop the run"
    );
    assert_eq!(
        mismatched.reason,
        DtlsSrtpError::FingerprintMismatch.to_string()
    );
    assert!(
        mismatched.peer_recovered_rtp.is_empty(),
        "a refused peer must receive no protected media"
    );
}
