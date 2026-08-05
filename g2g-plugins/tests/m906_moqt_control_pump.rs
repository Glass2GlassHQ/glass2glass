//! M906: `moqtsink` answers control messages without a frame to ride on.
//!
//! The publisher used to decode inbound control messages only while it had a
//! fragment to push, and only dialled the relay on its first frame: a SUBSCRIBE
//! that landed during startup, or between two fragments seconds apart, went
//! unanswered until the next one, and the relay gave up establishing the
//! subscription. This drives the sink against an in-process peer that completes
//! SETUP and then sends SUBSCRIBEs by hand, so the two things the fix has to
//! deliver are asserted directly:
//!
//!   - the namespace is published and a SUBSCRIBE is answered with no frame ever
//!     having been pushed into the element.
//!   - a media SUBSCRIBE that arrives before the `moov` names any track is held,
//!     then answered (SUBSCRIBE_OK, or DOES_NOT_EXIST for a name the `moov` does
//!     not name) once the tracks exist, with the catalog object following on a
//!     subgroup stream.
//!
//! The peer keeps no track state and makes no routing decision: what it records
//! is exactly what `moqtsink` wrote.
#![cfg(feature = "moqt")]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::time::timeout;
use web_transport_quinn::{RecvStream, SendStream, Server, Session};

use g2g_core::element::AsyncElement;
use g2g_core::frame::PipelinePacket;
use g2g_core::{ByteStreamEncoding, Caps, G2gError};

use g2g_plugins::moqt::coding::{setup_param, MoqtError, Params, TrackName, TrackNamespace};
use g2g_plugins::moqt::message::{request_error_code, ControlMessage};
use g2g_plugins::moqt::reassembly::{StreamItem, SubgroupStreamDecoder};
use g2g_plugins::moqtsink::MoqtSink;
use g2g_plugins::mp4mux::Mp4Mux;

mod moqt_common;
use moqt_common::{access_unit, bind_server, frame, h264_caps, CaptureSink, NullOut, TestCert};

/// How long the dial from `configure_pipeline` gets to reach the peer.
const DIAL_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a control message the sink owes us gets to arrive.
const CONTROL_TIMEOUT: Duration = Duration::from_secs(5);
/// How long the data plane gets to deliver the objects that were written.
const OBJECT_TIMEOUT: Duration = Duration::from_secs(10);

// ------------------------------------------------------------------- the peer

/// Every object that arrived on a subgroup stream: track alias, group, payload.
type Objects = Arc<Mutex<Vec<(u64, u64, Vec<u8>)>>>;

/// The publisher's peer: a WebTransport server that completes SETUP and then
/// leaves the control stream to the test, while recording the subgroup objects
/// the sink writes.
struct Relay {
    /// Held because dropping the session closes the QUIC connection under it.
    _session: Session,
    tx: SendStream,
    rx: RecvStream,
    buf: Vec<u8>,
    objects: Objects,
}

impl Relay {
    async fn send(&mut self, msg: ControlMessage) {
        let mut out = Vec::new();
        msg.encode(&mut out).expect("encode control message");
        self.tx
            .write_all(&out)
            .await
            .expect("write control message");
    }

    async fn recv(&mut self) -> ControlMessage {
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
    async fn recv_within(&mut self, within: Duration) -> Option<ControlMessage> {
        timeout(within, self.recv()).await.ok()
    }
}

/// Accept the publisher's session, answer its CLIENT_SETUP, and start reading
/// whatever subgroup streams it opens.
async fn accept_publisher(server: &mut Server) -> Relay {
    let request = server.accept().await.expect("a session");
    let session = request.ok().await.expect("CONNECT");
    let (tx, rx) = session.accept_bi().await.expect("the control stream");
    let objects: Objects = Arc::new(Mutex::new(Vec::new()));
    let mut relay = Relay {
        _session: session.clone(),
        tx,
        rx,
        buf: Vec::new(),
        objects: Arc::clone(&objects),
    };
    match relay.recv().await {
        ControlMessage::ClientSetup { .. } => {}
        other => panic!("expected CLIENT_SETUP, got {}", other.name()),
    }
    let mut params = Params::new();
    params.set_int(setup_param::MAX_REQUEST_ID, 100);
    relay.send(ControlMessage::ServerSetup { params }).await;

    tokio::spawn(async move {
        while let Ok(stream) = session.accept_uni().await {
            tokio::spawn(read_subgroup(stream, Arc::clone(&objects)));
        }
    });
    relay
}

/// Read one subgroup stream to its end, recording each whole object under the
/// alias and group its header named.
async fn read_subgroup(mut stream: RecvStream, objects: Objects) {
    let mut decoder = SubgroupStreamDecoder::new(4 * 1024 * 1024);
    let mut chunk = vec![0u8; 16 * 1024];
    let mut route: Option<(u64, u64)> = None;
    while let Ok(Some(n)) = stream.read(&mut chunk).await {
        if n == 0 || decoder.push(&chunk[..n]).is_err() {
            return;
        }
        while let Ok(Some(item)) = decoder.next_item() {
            match item {
                StreamItem::Header(header) => route = Some((header.track_alias, header.group_id)),
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
    }
}

/// Wait until the recorded objects satisfy `done`, so an assertion never races
/// the peer's read tasks, and return them.
async fn objects_when(
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
            "the objects the sink wrote never arrived"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

fn subscribe(id: u64, namespace: &TrackNamespace, track: &str) -> ControlMessage {
    ControlMessage::Subscribe {
        id,
        namespace: namespace.clone(),
        track_name: TrackName::new(track),
        params: Params::new(),
    }
}

// -------------------------------------------------------------- the publisher

/// Mux `count` access units and publish every fragment they produce, returning
/// the fMP4 byte stream that went into the sink.
async fn publish_fragments(sink: &mut MoqtSink, count: u64) -> Vec<u8> {
    let mut mux = Mp4Mux::new();
    mux.configure_pipeline(&h264_caps(64, 48))
        .expect("mp4mux caps");
    let mut published = Vec::new();
    for index in 0..count {
        let mut captured = CaptureSink::default();
        mux.process(
            PipelinePacket::DataFrame(frame(access_unit(index, 0), index * 33_333_333, index)),
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

// ------------------------------------------------------------------ the tests

/// The defect and its fix: configuring the pipeline publishes the namespace, a
/// SUBSCRIBE arriving with no frame in sight is answered anyway, and a media
/// SUBSCRIBE that beat the `moov` is resolved once the tracks exist.
#[tokio::test]
async fn a_subscribe_is_answered_before_the_first_frame() {
    let tls = TestCert::generate();
    let (mut server, port) = bind_server(&tls);
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2gpump";

    let mut sink = MoqtSink::new(&url, namespace).with_server_certificate_hashes(&tls.hash_hex);
    // Nothing is muxed here, and nothing will be until the subscriptions below
    // have been answered.
    sink.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .expect("moqtsink caps");

    let mut relay = timeout(DIAL_TIMEOUT, accept_publisher(&mut server))
        .await
        .expect("the sink dialled when the pipeline was configured");

    let ns = TrackNamespace::from_path(namespace);
    match timeout(CONTROL_TIMEOUT, relay.recv())
        .await
        .expect("PUBLISH_NAMESPACE without a frame")
    {
        ControlMessage::PublishNamespace {
            id, namespace: got, ..
        } => {
            assert_eq!(got, ns, "the namespace the sink published");
            relay
                .send(ControlMessage::RequestOk {
                    id,
                    params: Params::new(),
                })
                .await;
        }
        other => panic!("expected PUBLISH_NAMESPACE, got {}", other.name()),
    }

    // Three subscriptions, none of which the sink has any media for yet: the
    // catalog track it can answer, the media track it cannot name until the
    // `moov`, and a track that will never exist.
    relay.send(subscribe(1, &ns, ".catalog")).await;
    relay.send(subscribe(3, &ns, "1.m4s")).await;
    relay.send(subscribe(5, &ns, "9.m4s")).await;

    match timeout(CONTROL_TIMEOUT, relay.recv())
        .await
        .expect("SUBSCRIBE_OK with no frame ever pushed")
    {
        ControlMessage::SubscribeOk {
            id, track_alias, ..
        } => assert_eq!((id, track_alias), (1, 1), "the catalog subscription"),
        other => panic!("expected SUBSCRIBE_OK, got {}", other.name()),
    }
    assert_eq!(
        sink.objects_published(),
        0,
        "the catalog object cannot exist before the moov, so the subscription pends"
    );
    // The two media requests are held rather than refused, so nothing else is
    // sent until a `moov` names the tracks.
    assert!(
        relay
            .recv_within(Duration::from_millis(300))
            .await
            .is_none(),
        "a media SUBSCRIBE before the moov is held, not answered"
    );

    // Now publish: the `moov` resolves both held requests and the catalog goes
    // out on the subscription accepted minutes of pipeline time earlier.
    let published = publish_fragments(&mut sink, 12).await;
    assert_eq!(sink.track_names(), vec![String::from("1.m4s")]);

    let mut answers = Vec::new();
    while answers.len() < 2 {
        let msg = relay
            .recv_within(CONTROL_TIMEOUT)
            .await
            .expect("the held subscriptions were answered once the moov arrived");
        answers.push(msg);
    }
    assert!(
        answers.iter().any(|msg| matches!(
            msg,
            ControlMessage::SubscribeOk {
                id: 3,
                track_alias: 3,
                ..
            }
        )),
        "the media track the moov named was accepted: {answers:?}"
    );
    assert!(
        answers.iter().any(|msg| matches!(
            msg,
            ControlMessage::RequestError {
                id: 5,
                error_code,
                ..
            } if *error_code == request_error_code::DOES_NOT_EXIST
        )),
        "the track the moov never named was refused: {answers:?}"
    );

    let log = objects_when(&relay.objects, |log| {
        log.iter().any(|(alias, ..)| *alias == 1)
            && log.iter().filter(|(alias, ..)| *alias == 3).count() >= 2
    })
    .await;

    let catalog: Vec<&(u64, u64, Vec<u8>)> = log.iter().filter(|(alias, ..)| *alias == 1).collect();
    assert_eq!(catalog.len(), 1, "the catalog track carries one object");
    assert_eq!(catalog[0].1, 0, "in group 0");
    assert_eq!(
        catalog[0].2.as_slice(),
        sink.catalog(),
        "the object is the catalog document as published"
    );
    assert!(
        String::from_utf8_lossy(&catalog[0].2).contains("\"1.m4s\""),
        "the catalog names the media track: {}",
        String::from_utf8_lossy(&catalog[0].2)
    );

    for (_, _, payload) in log.iter().filter(|(alias, ..)| *alias == 3) {
        assert!(
            published.windows(payload.len()).any(|w| w == payload),
            "every media object is a fragment that was published"
        );
    }

    sink.process(PipelinePacket::Eos, &mut NullOut)
        .await
        .expect("clean end of stream");
}

/// A background dial that cannot succeed must not panic or wedge the element:
/// the failure surfaces as an error from the frame that needed the session.
#[tokio::test]
async fn a_failed_dial_surfaces_on_the_first_frame() {
    let mut sink = MoqtSink::new("not a url", "g2gpump");
    sink.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .expect("moqtsink caps");

    let ftyp = Vec::from(*b"\0\0\0\x08ftyp");
    let outcome = timeout(
        DIAL_TIMEOUT,
        sink.process(PipelinePacket::DataFrame(frame(ftyp, 0, 0)), &mut NullOut),
    )
    .await
    .expect("the frame was not left hanging on the dial");
    let err = outcome.expect_err("a dial that cannot succeed fails the frame");
    assert!(matches!(err, G2gError::Hardware(_)), "{err:?}");
}
