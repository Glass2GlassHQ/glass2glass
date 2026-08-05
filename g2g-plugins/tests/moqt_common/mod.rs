//! Helpers shared by the MoQ Transport test binaries (`m902_moqt`,
//! `m903_moqt_subscribe`, `m905_moqt_datagram`, `m906_moqt_control_pump`,
//! `m907_moqt_v18`, `m912_moqt_fetch`): the throwaway certificate and QUIC
//! endpoint, the fMP4 the publisher is fed, the output sinks, the
//! reference-peer lookup, the fragment comparisons, and the scripted peers of
//! each draft. One definition, included per test binary via `mod moqt_common;`.
//!
//! The peers keep no track state and make no routing decision: what they record
//! is exactly what the element wrote.
#![allow(dead_code)] // no one test binary uses every helper here

use core::future::Future;
use core::pin::Pin;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use web_transport_quinn::quinn::rustls::pki_types::pem::PemObject;
use web_transport_quinn::quinn::rustls::pki_types::{CertificateDer, PrivateKeyDer};
use web_transport_quinn::{Server, ServerBuilder};

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, Dim, G2gError, OutputSink, PushOutcome, Rate, VideoCodec};

use g2g_plugins::moqtsink::MoqtSink;
use g2g_plugins::mp4mux::Mp4Mux;

/// How long the dial from `configure_pipeline` gets to reach the peer.
pub(crate) const DIAL_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a message the element owes us gets to arrive.
pub(crate) const CONTROL_TIMEOUT: Duration = Duration::from_secs(5);
/// How long the data plane gets to deliver the objects that were written.
pub(crate) const OBJECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Every object that arrived on a data stream: track alias (or fetch request
/// id), group, payload.
pub(crate) type Objects = Arc<Mutex<Vec<(u64, u64, Vec<u8>)>>>;

/// Wait until the recorded objects satisfy `done`, so an assertion never races
/// the peer's read tasks, and return them.
pub(crate) async fn objects_when(
    objects: &Objects,
    done: impl Fn(&[(u64, u64, Vec<u8>)]) -> bool,
) -> Vec<(u64, u64, Vec<u8>)> {
    let deadline = Instant::now() + OBJECT_TIMEOUT;
    loop {
        {
            let log = objects.lock().expect("object log");
            if done(&log) {
                return log.clone();
            }
        }
        assert!(
            Instant::now() < deadline,
            "the objects the element wrote never arrived"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// Mux `count` access units and publish every fragment they produce, returning
/// the fMP4 byte stream that went into the sink.
pub(crate) async fn publish_fragments(sink: &mut MoqtSink, count: u64) -> Vec<u8> {
    publish_padded_fragments(sink, count, 0).await
}

/// The same, with `pad` bytes added to every access unit, so the objects are as
/// large as the test needs (a fetch response big enough that the peer's flow
/// control holds the writer, for instance).
pub(crate) async fn publish_padded_fragments(
    sink: &mut MoqtSink,
    count: u64,
    pad: usize,
) -> Vec<u8> {
    let mut mux = Mp4Mux::new();
    mux.configure_pipeline(&h264_caps(64, 48))
        .expect("mp4mux caps");
    let mut published = Vec::new();
    for index in 0..count {
        let mut captured = CaptureSink::default();
        mux.process(
            PipelinePacket::DataFrame(frame(
                padded_access_unit(index, pad),
                index * 33_333_333,
                index,
            )),
            &mut captured,
        )
        .await
        .expect("mux access unit");
        for chunk in captured.frames {
            published.extend_from_slice(&chunk);
            sink.process(
                PipelinePacket::DataFrame(frame(chunk, index * 33_333_333, index)),
                &mut NullOut,
            )
            .await
            .expect("publish fragment");
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    published
}

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

// ------------------------------------------------------------ the draft-16 peer

/// A scripted draft-16 peer: a WebTransport server that completes SETUP and
/// then leaves the control stream to the test, while recording the objects the
/// element writes on its data streams.
pub(crate) mod peer16 {
    use super::*;

    use tokio::time::timeout;
    use web_transport_quinn::{RecvStream, SendStream, Server, Session};

    use g2g_plugins::moqt::coding::{setup_param, MoqtError, Params, TrackName, TrackNamespace};
    use g2g_plugins::moqt::fetch::FETCH_HEADER_TYPE;
    use g2g_plugins::moqt::message::ControlMessage;
    use g2g_plugins::moqt::reassembly::{StreamItem, SubgroupStreamDecoder};
    use g2g_plugins::moqt::MoqtVersion;

    pub(crate) struct Relay {
        /// Held because dropping the session closes the QUIC connection.
        pub(crate) session: Session,
        pub(crate) tx: SendStream,
        pub(crate) rx: RecvStream,
        pub(crate) buf: Vec<u8>,
        /// Objects off subgroup streams, keyed by track alias.
        pub(crate) objects: Objects,
        /// Objects off FETCH response streams, keyed by request id.
        pub(crate) fetched: Objects,
    }

    impl Relay {
        pub(crate) async fn send(&mut self, msg: ControlMessage) {
            let mut out = Vec::new();
            msg.encode(&mut out).expect("encode control message");
            self.tx
                .write_all(&out)
                .await
                .expect("write control message");
        }

        pub(crate) async fn recv(&mut self) -> ControlMessage {
            loop {
                match ControlMessage::decode(&self.buf) {
                    Ok((msg, used)) => {
                        self.buf.drain(..used);
                        return msg;
                    }
                    Err(MoqtError::Incomplete) => {
                        let mut chunk = vec![0u8; 8192];
                        match self.rx.read(&mut chunk).await {
                            Ok(Some(n)) if n > 0 => self.buf.extend_from_slice(&chunk[..n]),
                            _ => panic!("the control stream ended"),
                        }
                    }
                    Err(e) => panic!("malformed control message: {e:?}"),
                }
            }
        }

        /// The next control message, or `None` when none arrives within `within`.
        pub(crate) async fn recv_within(&mut self, within: Duration) -> Option<ControlMessage> {
            timeout(within, self.recv()).await.ok()
        }

        pub(crate) fn subscribe(&self, id: u64, namespace: &str, track: &str) -> ControlMessage {
            ControlMessage::Subscribe {
                id,
                namespace: TrackNamespace::from_path(namespace),
                track_name: TrackName::new(track),
                params: Params::new(),
            }
        }
    }

    /// Accept the publisher's session, answer its CLIENT_SETUP, and start
    /// reading whatever data streams it opens.
    pub(crate) async fn accept_publisher(server: &mut Server) -> Relay {
        let request = server.accept().await.expect("a session");
        let session = request.ok().await.expect("CONNECT");
        let (tx, rx) = session.accept_bi().await.expect("the control stream");
        let objects: Objects = Arc::new(Mutex::new(Vec::new()));
        let fetched: Objects = Arc::new(Mutex::new(Vec::new()));
        let mut relay = Relay {
            session: session.clone(),
            tx,
            rx,
            buf: Vec::new(),
            objects: Arc::clone(&objects),
            fetched: Arc::clone(&fetched),
        };
        match relay.recv().await {
            ControlMessage::ClientSetup { .. } => {}
            other => panic!("expected CLIENT_SETUP, got {}", other.name()),
        }
        let mut params = Params::new();
        params.set_int(setup_param::MAX_REQUEST_ID, 100);
        relay.send(ControlMessage::ServerSetup { params }).await;
        tokio::spawn(read_data_streams(session, objects, fetched));
        relay
    }

    /// Accept the publisher's data streams, telling a subgroup from a fetch
    /// response by the type varint that opens it.
    async fn read_data_streams(session: Session, objects: Objects, fetched: Objects) {
        while let Ok(mut stream) = session.accept_uni().await {
            let Ok((code, prefix)) =
                g2g_plugins::moqt::read_stream_type(MoqtVersion::V16, &mut stream).await
            else {
                continue;
            };
            if code == FETCH_HEADER_TYPE {
                tokio::spawn(read_fetch(stream, prefix, Arc::clone(&fetched)));
            } else {
                tokio::spawn(read_subgroup(stream, prefix, Arc::clone(&objects)));
            }
        }
    }

    /// Read one subgroup stream to its end, recording each whole object under
    /// the alias and group its header named.
    async fn read_subgroup(mut stream: RecvStream, prefix: Vec<u8>, objects: Objects) {
        let mut decoder = SubgroupStreamDecoder::new(8 * 1024 * 1024);
        if decoder.push(&prefix).is_err() {
            return;
        }
        let mut chunk = vec![0u8; 16 * 1024];
        let mut route: Option<(u64, u64)> = None;
        loop {
            while let Ok(Some(item)) = decoder.next_item() {
                match item {
                    StreamItem::Header(header) => {
                        route = Some((header.track_alias, header.group_id))
                    }
                    StreamItem::Object(object) => {
                        let Some((alias, group)) = route else {
                            return; // an object before the header is impossible
                        };
                        objects
                            .lock()
                            .expect("object log")
                            .push((alias, group, object.payload));
                    }
                }
            }
            match stream.read(&mut chunk).await {
                Ok(Some(n)) if n > 0 => {
                    if decoder.push(&chunk[..n]).is_err() {
                        return;
                    }
                }
                _ => return,
            }
        }
    }

    /// Read one FETCH response stream, recording each object under the request
    /// id its header named.
    async fn read_fetch(mut stream: RecvStream, prefix: Vec<u8>, fetched: Objects) {
        read_fetch_stream(MoqtVersion::V16, &mut stream, prefix, fetched).await
    }

    /// Serve one group's payloads on one draft-16 subgroup stream, consecutive
    /// object ids from 0.
    pub(crate) async fn serve_group(
        session: &Session,
        alias: u64,
        group: u64,
        payloads: &[Vec<u8>],
    ) {
        use g2g_plugins::moqt::data::{StreamHeaderType, SubgroupHeader, SubgroupObjectHeader};

        let header_type = StreamHeaderType::SubgroupIdExt;
        let mut bytes = Vec::new();
        SubgroupHeader {
            header_type,
            track_alias: alias,
            group_id: group,
            subgroup_id: Some(0),
            publisher_priority: 127,
        }
        .encode(&mut bytes)
        .expect("encode subgroup header");
        for payload in payloads {
            SubgroupObjectHeader::normal(0, payload.len())
                .encode(header_type, &mut bytes)
                .expect("encode object header");
            bytes.extend_from_slice(payload);
        }
        let mut stream = session.open_uni().await.expect("a subgroup stream");
        stream.write_all(&bytes).await.expect("write subgroup");
        let _ = stream.finish();
    }
}

/// A FETCH response a scripted publisher writes by hand, so a test can
/// interleave the response with the live objects behind it.
pub(crate) struct FetchServer {
    stream: web_transport_quinn::SendStream,
    writer: g2g_plugins::moqt::fetch::FetchWriter,
}

impl FetchServer {
    pub(crate) async fn open(
        version: g2g_plugins::moqt::MoqtVersion,
        session: &web_transport_quinn::Session,
        request_id: u64,
    ) -> Self {
        let writer = g2g_plugins::moqt::fetch::FetchWriter::new(version);
        let mut bytes = Vec::new();
        writer.header(request_id, &mut bytes);
        let mut stream = session.open_uni().await.expect("a fetch stream");
        stream.write_all(&bytes).await.expect("write fetch header");
        Self { stream, writer }
    }

    pub(crate) async fn object(&mut self, group_id: u64, object_id: u64, payload: &[u8]) {
        let mut bytes = Vec::new();
        self.writer
            .object(group_id, object_id, 127, payload, &mut bytes)
            .expect("encode fetch object");
        self.stream
            .write_all(&bytes)
            .await
            .expect("write fetch object");
    }

    pub(crate) fn finish(mut self) {
        let _ = self.stream.finish();
    }
}

/// Read one FETCH response stream to its end, recording every object under the
/// request id its header named. Shared by both drafts' peers.
pub(crate) async fn read_fetch_stream(
    version: g2g_plugins::moqt::MoqtVersion,
    stream: &mut web_transport_quinn::RecvStream,
    prefix: Vec<u8>,
    fetched: Objects,
) {
    use g2g_plugins::moqt::fetch::{FetchItem, FetchStreamDecoder};

    let mut decoder = FetchStreamDecoder::new(version, 8 * 1024 * 1024);
    if decoder.push(&prefix).is_err() {
        return;
    }
    let mut chunk = vec![0u8; 16 * 1024];
    let mut request_id = None;
    loop {
        while let Ok(Some(item)) = decoder.next_item() {
            match item {
                FetchItem::Header { request_id: id } => request_id = Some(id),
                FetchItem::Object(object) => {
                    let Some(id) = request_id else {
                        return;
                    };
                    fetched
                        .lock()
                        .expect("fetch log")
                        .push((id, object.group_id, object.payload));
                }
                FetchItem::Gap { .. } => {}
            }
        }
        match stream.read(&mut chunk).await {
            Ok(Some(n)) if n > 0 => {
                if decoder.push(&chunk[..n]).is_err() {
                    return;
                }
            }
            _ => return,
        }
    }
}

// ------------------------------------------------------------ the draft-18 peer

/// A scripted draft-18 peer: paired control streams, one bidirectional stream
/// per request, and the same object recording as the draft-16 one.
pub(crate) mod peer18 {
    use super::*;

    use tokio::time::timeout;
    use web_transport_quinn::proto::ConnectResponse;
    use web_transport_quinn::{RecvStream, SendStream, Server, Session};

    use g2g_plugins::moqt::coding::{Params, TrackName, TrackNamespace};
    use g2g_plugins::moqt::fetch::FETCH_HEADER_TYPE;
    use g2g_plugins::moqt::v18::coding::MessageParams;
    use g2g_plugins::moqt::v18::data::{StreamItem, SubgroupHeaderType, SubgroupStreamDecoder};
    use g2g_plugins::moqt::v18::message::ControlMessage;
    use g2g_plugins::moqt::v18::session::{write_message, MessageReader, MOQT_PROTOCOL};
    use g2g_plugins::moqt::MoqtVersion;

    /// One request stream this peer opened toward the element, or accepted from
    /// it: the send half, the read half, and its buffered reader.
    pub(crate) struct RequestStream {
        pub(crate) tx: SendStream,
        pub(crate) rx: RecvStream,
        pub(crate) reader: MessageReader,
    }

    impl RequestStream {
        pub(crate) fn new(tx: SendStream, rx: RecvStream) -> Self {
            Self {
                tx,
                rx,
                reader: MessageReader::new(),
            }
        }

        pub(crate) async fn send(&mut self, msg: ControlMessage) {
            write_message(&mut self.tx, &msg)
                .await
                .expect("write request message");
        }

        pub(crate) async fn recv(&mut self) -> ControlMessage {
            self.reader
                .next(&mut self.rx)
                .await
                .expect("read request message")
                .expect("the request stream ended")
        }

        pub(crate) async fn recv_within(&mut self, within: Duration) -> Option<ControlMessage> {
            timeout(within, self.recv()).await.ok()
        }

        /// `Ok(None)` from the reader: the element finished the stream.
        pub(crate) async fn ended(&mut self) -> bool {
            matches!(self.reader.next(&mut self.rx).await, Ok(None))
        }
    }

    /// Accept the element's session as a draft-18 server: select the `moqt-18`
    /// subprotocol and complete the CONNECT.
    pub(crate) async fn accept_v18(server: &mut Server) -> Session {
        let request = server.accept().await.expect("a session");
        assert!(
            request
                .connect()
                .protocols
                .iter()
                .any(|p| p == MOQT_PROTOCOL),
            "the element offered moqt-18: {:?}",
            request.connect().protocols
        );
        request
            .respond(ConnectResponse::OK.with_protocol(MOQT_PROTOCOL))
            .await
            .expect("CONNECT")
    }

    /// Exchange SETUPs: write ours on a fresh unidirectional stream, read the
    /// element's from the first one it opened. Both halves are returned so the
    /// caller keeps them alive: dropping a control stream mid-session is a
    /// violation the element acts on.
    pub(crate) async fn exchange_setup(session: &Session) -> (RecvStream, SendStream) {
        let mut ours = session.open_uni().await.expect("our control stream");
        write_message(
            &mut ours,
            &ControlMessage::Setup {
                options: Params::new(),
            },
        )
        .await
        .expect("write SETUP");

        let mut theirs = session.accept_uni().await.expect("their control stream");
        let mut reader = MessageReader::new();
        match reader.next(&mut theirs).await {
            Ok(Some(ControlMessage::Setup { .. })) => (theirs, ours),
            other => panic!("expected SETUP first, got {other:?}"),
        }
    }

    /// Accept the publisher's data streams and record their objects: subgroups
    /// by track alias, fetch responses by request id.
    pub(crate) fn record_data(session: Session) -> (Objects, Objects) {
        let objects: Objects = Arc::new(Mutex::new(Vec::new()));
        let fetched: Objects = Arc::new(Mutex::new(Vec::new()));
        let (log, fetch_log) = (Arc::clone(&objects), Arc::clone(&fetched));
        tokio::spawn(async move {
            while let Ok(mut stream) = session.accept_uni().await {
                let Ok((code, prefix)) =
                    g2g_plugins::moqt::read_stream_type(MoqtVersion::V18, &mut stream).await
                else {
                    continue;
                };
                if code == FETCH_HEADER_TYPE {
                    let fetch_log = Arc::clone(&fetch_log);
                    tokio::spawn(async move {
                        read_fetch_stream(MoqtVersion::V18, &mut stream, prefix, fetch_log).await
                    });
                } else {
                    tokio::spawn(read_subgroup(stream, prefix, Arc::clone(&log)));
                }
            }
        });
        (objects, fetched)
    }

    /// Read one draft-18 subgroup stream to its end.
    pub(crate) async fn read_subgroup(mut stream: RecvStream, prefix: Vec<u8>, objects: Objects) {
        let mut decoder = SubgroupStreamDecoder::new(8 * 1024 * 1024);
        if decoder.push(&prefix).is_err() {
            return;
        }
        let mut route: Option<(u64, u64)> = None;
        loop {
            while let Ok(Some(item)) = decoder.next_item() {
                match item {
                    StreamItem::Header(header) => {
                        route = Some((header.track_alias, header.group_id))
                    }
                    StreamItem::Object(object) => {
                        let Some((alias, group)) = route else {
                            return;
                        };
                        objects
                            .lock()
                            .expect("object log")
                            .push((alias, group, object.payload));
                    }
                }
            }
            let mut chunk = vec![0u8; 16 * 1024];
            match stream.read(&mut chunk).await {
                Ok(Some(n)) if n > 0 => {
                    if decoder.push(&chunk[..n]).is_err() {
                        return;
                    }
                }
                _ => return,
            }
        }
    }

    /// Open a request stream toward the publisher with one SUBSCRIBE on it.
    pub(crate) async fn subscribe(
        session: &Session,
        id: u64,
        namespace: &str,
        track: &str,
    ) -> RequestStream {
        let (mut tx, rx) = session.open_bi().await.expect("a request stream");
        let msg = ControlMessage::Subscribe {
            id,
            namespace: TrackNamespace::from_path(namespace),
            track_name: TrackName::new(track),
            params: MessageParams::new(),
        };
        write_message(&mut tx, &msg).await.expect("write SUBSCRIBE");
        RequestStream::new(tx, rx)
    }

    /// Serve one group's payloads on one draft-18 subgroup stream, consecutive
    /// object ids from 0.
    pub(crate) async fn serve_group(
        session: &Session,
        alias: u64,
        group: u64,
        payloads: &[Vec<u8>],
    ) {
        use g2g_plugins::moqt::v18::data::{SubgroupHeader, SubgroupObjectHeader};

        let header_type = SubgroupHeaderType::explicit();
        let mut bytes = Vec::new();
        SubgroupHeader {
            header_type,
            track_alias: alias,
            group_id: group,
            subgroup_id: Some(0),
            publisher_priority: Some(127),
        }
        .encode(&mut bytes)
        .expect("encode subgroup header");
        for payload in payloads {
            // Consecutive ids: the first object's delta is its id (0), then
            // zero deltas mean "previous plus one".
            SubgroupObjectHeader::normal(0, payload.len())
                .encode(header_type, &mut bytes)
                .expect("encode object header");
            bytes.extend_from_slice(payload);
        }
        let mut stream = session.open_uni().await.expect("a subgroup stream");
        stream.write_all(&bytes).await.expect("write subgroup");
        let _ = stream.finish();
    }

    /// Accept the request stream the element opened and read its first message.
    pub(crate) async fn accept_request(session: &Session) -> (RequestStream, ControlMessage) {
        let (tx, rx) = session.accept_bi().await.expect("a request stream");
        let mut request = RequestStream::new(tx, rx);
        let first = request.recv().await;
        (request, first)
    }
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
    let pad = if big_every != 0 && index % big_every == 0 {
        2500
    } else {
        0
    };
    padded_access_unit(index, pad)
}

/// The same access unit with `pad` bytes appended.
pub(crate) fn padded_access_unit(index: u64, pad: usize) -> Vec<u8> {
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
    unit.extend(core::iter::repeat_n(index as u8, pad));
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
