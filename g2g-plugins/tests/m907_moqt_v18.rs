//! M907: `moqtsink` and `moqtsrc` speak MoQ Transport draft-18 when the server
//! selects it.
//!
//! The version is negotiated per session: the elements offer `moqt-18` and
//! `moqt-16` as WebTransport subprotocols (the `versions` property) and the
//! server's pick decides which handshake runs. These tests script the draft-18
//! side of the wire in-process, the way `m906` scripts draft-16: the peer keeps
//! no track state, so what it records is exactly what the element wrote.
//!
//!   - the publisher completes the paired-control-stream SETUP, publishes its
//!     namespace on a request stream, answers SUBSCRIBEs on the streams they
//!     arrive on (held ones included), and carries objects on draft-18
//!     subgroup streams, or in draft-18 datagrams when `datagrams=true`.
//!   - the subscriber drives SUBSCRIBE request streams, resolves the track
//!     alias from SUBSCRIBE_OK, reorders the subgroup objects, and ends the
//!     stream on PUBLISH_DONE.
//!   - the draft-16 path is untouched: `m902`/`m903`/`m905`/`m906` run against
//!     the same elements with their new default `versions=18,16`, and the
//!     draft-16 relay selects `moqt-16`.
#![cfg(feature = "moqt")]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::time::timeout;
use web_transport_quinn::Session;

use g2g_core::element::AsyncElement;
use g2g_core::frame::PipelinePacket;
use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, PropValue};

use g2g_plugins::moqt::catalog;
use g2g_plugins::moqt::coding::{Params, TrackNamespace};
use g2g_plugins::moqt::v18::coding::MessageParams;
use g2g_plugins::moqt::v18::data::{SubgroupHeader, SubgroupHeaderType, SubgroupObjectHeader};
use g2g_plugins::moqt::v18::datagram::DatagramObject;
use g2g_plugins::moqt::v18::message::{msg_type, request_error_code, ControlMessage};
use g2g_plugins::moqtsink::MoqtSink;
use g2g_plugins::moqtsrc::MoqtSrc;

mod moqt_common;
use moqt_common::peer18::{accept_request, accept_v18, exchange_setup, record_data, subscribe};
use moqt_common::{
    bind_server, objects_when, publish_fragments, CaptureSink, NullOut, Objects, TestCert,
    CONTROL_TIMEOUT, DIAL_TIMEOUT, OBJECT_TIMEOUT,
};

fn bmff_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    }
}

// ------------------------------------------------------------------ the tests

/// The publisher end to end on draft-18: SETUP over paired control streams,
/// PUBLISH_NAMESPACE answered on its own stream, SUBSCRIBEs answered on theirs
/// (a media one held until the `moov`), objects on draft-18 subgroup streams,
/// and PUBLISH_DONE closing each request at EOS.
#[tokio::test]
async fn the_sink_publishes_draft_18_when_the_server_selects_it() {
    let tls = TestCert::generate();
    let (mut server, port) = bind_server(&tls);
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2g18";

    let mut sink = MoqtSink::new(&url, namespace).with_server_certificate_hashes(&tls.hash_hex);
    sink.configure_pipeline(&bmff_caps())
        .expect("moqtsink caps");

    let session = timeout(DIAL_TIMEOUT, accept_v18(&mut server))
        .await
        .expect("the sink dialled when the pipeline was configured");
    let _their_control = exchange_setup(&session).await;

    // The namespace publish arrives on its own request stream and is answered
    // there.
    let (mut ns, first) = timeout(CONTROL_TIMEOUT, accept_request(&session))
        .await
        .expect("PUBLISH_NAMESPACE without a frame");
    match first {
        ControlMessage::PublishNamespace {
            id, namespace: got, ..
        } => {
            assert_eq!(got, TrackNamespace::from_path(namespace));
            assert_eq!(id % 2, 0, "a client request id is even");
            ns.send(ControlMessage::RequestOk {
                params: MessageParams::new(),
                properties: Params::new(),
            })
            .await;
        }
        other => panic!("expected PUBLISH_NAMESPACE, got {}", other.name()),
    }

    let (objects, _fetched) = record_data(session.clone());

    // Three subscriptions before any frame: the catalog (answerable), a media
    // track the `moov` has not named yet (held), and one that never exists.
    let mut catalog_req = subscribe(&session, 1, namespace, ".catalog").await;
    let mut media_req = subscribe(&session, 3, namespace, "1.m4s").await;
    let mut missing_req = subscribe(&session, 5, namespace, "9.m4s").await;

    match timeout(CONTROL_TIMEOUT, catalog_req.recv())
        .await
        .expect("SUBSCRIBE_OK with no frame ever pushed")
    {
        ControlMessage::SubscribeOk { track_alias, .. } => {
            assert_eq!(track_alias, 1, "the alias is the request id");
        }
        other => panic!("expected SUBSCRIBE_OK, got {}", other.name()),
    }
    assert!(
        media_req
            .recv_within(Duration::from_millis(300))
            .await
            .is_none(),
        "a media SUBSCRIBE before the moov is held, not answered"
    );

    let published = publish_fragments(&mut sink, 12).await;
    assert_eq!(sink.track_names(), vec![String::from("1.m4s")]);

    match timeout(CONTROL_TIMEOUT, media_req.recv())
        .await
        .expect("the held media SUBSCRIBE was answered after the moov")
    {
        ControlMessage::SubscribeOk { track_alias, .. } => assert_eq!(track_alias, 3),
        other => panic!("expected SUBSCRIBE_OK, got {}", other.name()),
    }
    match timeout(CONTROL_TIMEOUT, missing_req.recv())
        .await
        .expect("the track the moov never named was refused")
    {
        ControlMessage::RequestError { error_code, .. } => {
            assert_eq!(error_code, request_error_code::DOES_NOT_EXIST);
        }
        other => panic!("expected REQUEST_ERROR, got {}", other.name()),
    }
    assert!(
        missing_req.ended().await,
        "a refused request stream is finished"
    );

    let log = objects_when(&objects, |log| {
        log.iter().any(|(alias, ..)| *alias == 1)
            && log.iter().filter(|(alias, ..)| *alias == 3).count() >= 2
    })
    .await;

    let catalog_objects: Vec<_> = log.iter().filter(|(alias, ..)| *alias == 1).collect();
    assert_eq!(catalog_objects.len(), 1, "one catalog object, group 0");
    assert_eq!(catalog_objects[0].1, 0);
    assert_eq!(catalog_objects[0].2.as_slice(), sink.catalog());
    for (_, _, payload) in log.iter().filter(|(alias, ..)| *alias == 3) {
        assert!(
            published.windows(payload.len()).any(|w| w == payload),
            "every media object is a fragment that was published"
        );
    }

    sink.process(PipelinePacket::Eos, &mut NullOut)
        .await
        .expect("clean end of stream");
    match timeout(CONTROL_TIMEOUT, media_req.recv())
        .await
        .expect("EOS tells each subscription the track ended")
    {
        ControlMessage::PublishDone { stream_count, .. } => {
            assert!(stream_count > 0, "the subscription opened data streams");
        }
        other => panic!("expected PUBLISH_DONE, got {}", other.name()),
    }
    assert!(media_req.ended().await, "PUBLISH_DONE finishes the request");
}

/// `datagrams=true` under draft-18: media objects arrive as draft-18 datagrams,
/// and the group is closed by an end-of-group status datagram.
#[tokio::test]
async fn the_sink_carries_draft_18_objects_in_datagrams() {
    let tls = TestCert::generate();
    let (mut server, port) = bind_server(&tls);
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2g18dg";

    let mut sink = MoqtSink::new(&url, namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_datagrams(true);
    sink.configure_pipeline(&bmff_caps())
        .expect("moqtsink caps");

    let session = timeout(DIAL_TIMEOUT, accept_v18(&mut server))
        .await
        .expect("dial");
    let _their_control = exchange_setup(&session).await;

    let (mut ns, first) = timeout(CONTROL_TIMEOUT, accept_request(&session))
        .await
        .expect("namespace");
    match first {
        ControlMessage::PublishNamespace { .. } => {
            ns.send(ControlMessage::RequestOk {
                params: MessageParams::new(),
                properties: Params::new(),
            })
            .await;
        }
        other => panic!("expected PUBLISH_NAMESPACE, got {}", other.name()),
    }

    // Record datagrams: each is one whole draft-18 object.
    let datagrams: Objects = Arc::new(Mutex::new(Vec::new()));
    let markers = Arc::new(Mutex::new(Vec::<(u64, u64, u64)>::new()));
    {
        let session = session.clone();
        let datagrams = Arc::clone(&datagrams);
        let markers = Arc::clone(&markers);
        tokio::spawn(async move {
            while let Ok(bytes) = session.read_datagram().await {
                let object =
                    DatagramObject::decode(&bytes, 4 * 1024 * 1024).expect("a valid datagram");
                if object.datagram_type.status {
                    markers.lock().expect("markers").push((
                        object.track_alias,
                        object.group_id,
                        object.object_id,
                    ));
                } else {
                    datagrams.lock().expect("datagrams").push((
                        object.track_alias,
                        object.group_id,
                        object.payload,
                    ));
                }
            }
        });
    }

    // The catalog SUBSCRIBE is answerable before any frame, so its SUBSCRIBE_OK
    // is the barrier proving the sink processed our requests: the media
    // SUBSCRIBE behind it is then held until the moov, and serves from the
    // first group.
    let mut catalog_req = subscribe(&session, 1, namespace, ".catalog").await;
    let mut media_req = subscribe(&session, 3, namespace, "1.m4s").await;
    match timeout(CONTROL_TIMEOUT, catalog_req.recv())
        .await
        .expect("SUBSCRIBE_OK for the catalog before any frame")
    {
        ControlMessage::SubscribeOk { track_alias, .. } => assert_eq!(track_alias, 1),
        other => panic!("expected SUBSCRIBE_OK, got {}", other.name()),
    }

    let published = publish_fragments(&mut sink, 15).await;
    match timeout(CONTROL_TIMEOUT, media_req.recv())
        .await
        .expect("SUBSCRIBE_OK after the moov")
    {
        ControlMessage::SubscribeOk { track_alias, .. } => assert_eq!(track_alias, 3),
        other => panic!("expected SUBSCRIBE_OK, got {}", other.name()),
    }

    let got = objects_when(&datagrams, |log| {
        log.iter().filter(|(alias, ..)| *alias == 3).count() >= 2
    })
    .await;
    for (_, _, payload) in got.iter().filter(|(alias, ..)| *alias == 1) {
        assert!(
            published.windows(payload.len()).any(|w| w == payload),
            "every datagram object is a fragment that was published"
        );
    }
    assert!(sink.datagram_objects() > 0, "objects rode datagrams");

    // Groups are GOPs: with more than one group published, the earlier ones
    // were closed by an end-of-group marker datagram.
    let deadline = Instant::now() + OBJECT_TIMEOUT;
    while markers.lock().expect("markers").is_empty() {
        assert!(
            Instant::now() < deadline,
            "an end-of-group marker datagram arrived"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    sink.process(PipelinePacket::Eos, &mut NullOut)
        .await
        .expect("clean end of stream");
}

/// The subscriber end to end on draft-18: a scripted publisher answers its
/// SUBSCRIBE request streams, serves catalog, init and media objects on
/// draft-18 subgroup streams, and ends the media track with PUBLISH_DONE.
#[tokio::test]
async fn the_source_plays_a_draft_18_broadcast() {
    let tls = TestCert::generate();
    let (mut server, port) = bind_server(&tls);
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2g18src";

    let init = b"init-segment-bytes".to_vec();
    let media = vec![
        (0u64, b"frag-0".to_vec()),
        (0u64, b"frag-1".to_vec()),
        (1u64, b"frag-2".to_vec()),
        (1u64, b"frag-3".to_vec()),
    ];

    let mut src = MoqtSrc::new(&url, namespace).with_server_certificate_hashes(&tls.hash_hex);
    src.configure_pipeline(&bmff_caps()).expect("moqtsrc caps");

    let publisher = {
        let init = init.clone();
        let media = media.clone();
        async move {
            let session = accept_v18(&mut server).await;
            // Held to the end of the block: dropping a control stream resets
            // it, which the subscriber rightly treats as the session ending.
            let control = exchange_setup(&session).await;

            // The source opens one request stream per SUBSCRIBE: catalog, then
            // init, then media. Serve each on a draft-18 subgroup stream.
            let catalog_json = catalog::build(
                &format!("/{namespace}"),
                "0.mp4",
                &[(String::from("1.m4s"), String::new())],
            );
            let mut expected = vec![
                (
                    ".catalog".to_string(),
                    11u64,
                    vec![catalog_json.into_bytes()],
                ),
                ("0.mp4".to_string(), 12, vec![init.clone()]),
            ];
            let media_by_group: Vec<Vec<Vec<u8>>> = vec![
                media
                    .iter()
                    .filter(|(g, _)| *g == 0)
                    .map(|(_, p)| p.clone())
                    .collect(),
                media
                    .iter()
                    .filter(|(g, _)| *g == 1)
                    .map(|(_, p)| p.clone())
                    .collect(),
            ];

            // A real publisher holds each request stream open for the
            // subscription's lifetime; dropping one resets it, which cancels
            // the subscription under the subscriber.
            let mut kept = Vec::new();
            for _ in 0..3 {
                let (mut request, first) = accept_request(&session).await;
                let (id, name) = match first {
                    ControlMessage::Subscribe { id, track_name, .. } => {
                        (id, track_name.as_str_lossy())
                    }
                    other => panic!("expected SUBSCRIBE, got {}", other.name()),
                };
                assert_eq!(id % 2, 0, "a client request id is even");
                let alias = match expected.iter().position(|(n, ..)| *n == name) {
                    Some(at) => expected[at].1,
                    None => {
                        assert_eq!(name, "1.m4s", "the media track from the catalog");
                        13
                    }
                };
                request
                    .send(ControlMessage::SubscribeOk {
                        track_alias: alias,
                        params: MessageParams::new(),
                        properties: Params::new(),
                    })
                    .await;
                if name == "1.m4s" {
                    for (group, payloads) in media_by_group.iter().enumerate() {
                        serve_group(&session, alias, group as u64, payloads).await;
                    }
                    request
                        .send(ControlMessage::PublishDone {
                            status_code:
                                g2g_plugins::moqt::v18::message::publish_done_code::TRACK_ENDED,
                            stream_count: media_by_group.len() as u64,
                            reason: String::from("end of stream"),
                        })
                        .await;
                    let _ = request.tx.finish();
                } else {
                    let at = expected.iter().position(|(n, ..)| *n == name).unwrap();
                    let (_, alias, payloads) = expected.remove(at);
                    serve_group(&session, alias, 0, &payloads).await;
                }
                kept.push(request);
            }
            // Hold the session, the control stream and every request stream
            // open until the subscriber is done with them.
            (session, control, kept)
        }
    };

    let run = async {
        let mut captured = CaptureSink::default();
        src.run(&mut captured).await.expect("the run completes");
        captured.frames
    };

    let (frames, _keep) = tokio::join!(run, publisher);
    let mut want = vec![init];
    want.extend(media.into_iter().map(|(_, p)| p));
    assert_eq!(frames, want, "init then the media objects in order");
    assert_eq!(src.selected_track(), "1.m4s");
}

/// Serve one group's payloads on one draft-18 subgroup stream.
async fn serve_group(session: &Session, alias: u64, group: u64, payloads: &[Vec<u8>]) {
    let mut stream = session.open_uni().await.expect("a subgroup stream");
    let mut bytes = Vec::new();
    SubgroupHeader {
        header_type: SubgroupHeaderType::explicit(),
        track_alias: alias,
        group_id: group,
        subgroup_id: Some(0),
        publisher_priority: Some(127),
    }
    .encode(&mut bytes)
    .expect("encode subgroup header");
    for payload in payloads {
        // Consecutive ids: the first object's delta is its id (0), then zero
        // deltas mean "previous plus one".
        SubgroupObjectHeader::normal(0, payload.len())
            .encode(SubgroupHeaderType::explicit(), &mut bytes)
            .expect("encode object header");
        bytes.extend_from_slice(payload);
    }
    stream.write_all(&bytes).await.expect("write subgroup");
    let _ = stream.finish();
}

/// The `versions` property: exposed, round-trips, and refuses a version this
/// build does not speak.
#[test]
fn the_versions_property_round_trips_and_validates() {
    let mut sink = MoqtSink::default();
    assert_eq!(
        sink.get_property("versions"),
        Some(PropValue::Str(String::from("18,16")))
    );
    sink.set_property("versions", PropValue::Str(String::from("16")))
        .expect("a draft this build speaks");
    assert_eq!(
        sink.get_property("versions"),
        Some(PropValue::Str(String::from("16")))
    );
    assert!(
        sink.set_property("versions", PropValue::Str(String::from("17")))
            .is_err(),
        "a draft this build does not speak is refused"
    );

    let mut src = MoqtSrc::default();
    assert_eq!(
        src.get_property("versions"),
        Some(PropValue::Str(String::from("18,16")))
    );
    src.set_property("versions", PropValue::Str(String::from("18")))
        .expect("a draft this build speaks");
    assert!(
        src.set_property("versions", PropValue::Str(String::from("")))
            .is_err(),
        "an empty list is refused"
    );
}

/// The draft-18 SETUP stream type is what a scripted server must parse, pinned
/// here so the tests and the elements agree on the two-byte prefix.
#[test]
fn the_setup_type_is_the_expected_code_point() {
    assert_eq!(msg_type::SETUP, 0x2f00);
}
