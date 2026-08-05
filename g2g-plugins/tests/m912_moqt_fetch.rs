//! M912: FETCH on both sides and both drafts.
//!
//! The publisher keeps the last `cache-groups` groups of every track and serves
//! a FETCH from them; the subscriber can ask for the groups before the live edge
//! with a joining FETCH and plays them ahead of the live objects.
//!
//! The legs, each against a scripted in-process peer of the draft under test
//! (the peer keeps no track state, so what it records is exactly what the
//! element wrote):
//!
//!   1. draft-16 publisher: a standalone FETCH is answered with FETCH_OK and
//!      the cached objects in order on a fetch stream; a range that was never
//!      published, one that has fallen out of the cache, an unknown track and a
//!      joining FETCH naming no subscription are each refused, never left
//!      hanging.
//!   2. draft-16 publisher: FETCH_CANCEL stops a response part way.
//!   3. draft-18 publisher: a joining FETCH after SUBSCRIBE is answered on the
//!      request stream and served on its own response stream, contiguous with
//!      the subscription.
//!   4. subscriber, both drafts: `catchup-groups` issues the joining FETCH and
//!      the fetched objects are emitted before the live ones, even when the
//!      live objects arrive first.
#![cfg(feature = "moqt")]

use std::time::Duration;

use tokio::time::timeout;

use g2g_core::element::AsyncElement;
use g2g_core::frame::PipelinePacket;
use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, PropValue};

use g2g_plugins::moqt::coding::{param, Params, TrackName, TrackNamespace};
use g2g_plugins::moqt::message::{
    request_error_code, ControlMessage, FetchType, JoiningFetch, Location, StandaloneFetch,
};
use g2g_plugins::moqt::v18::coding::MessageParams;
use g2g_plugins::moqt::v18::message::ControlMessage as Msg18;
use g2g_plugins::moqt::MoqtVersion;
use g2g_plugins::moqtsink::MoqtSink;
use g2g_plugins::moqtsrc::MoqtSrc;

mod moqt_common;
use moqt_common::{
    bind_server, objects_when, peer16, peer18, publish_fragments, publish_padded_fragments,
    CaptureSink, FetchServer, NullOut, TestCert, CONTROL_TIMEOUT, DIAL_TIMEOUT,
};

fn bmff_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    }
}

/// A standalone FETCH of one group range on the media track.
fn fetch(
    id: u64,
    namespace: &str,
    track: &str,
    start: (u64, u64),
    end: (u64, u64),
) -> ControlMessage {
    ControlMessage::Fetch {
        id,
        fetch_type: FetchType::Standalone,
        standalone: Some(StandaloneFetch {
            namespace: TrackNamespace::from_path(namespace),
            track_name: TrackName::new(track),
            start: Location {
                group_id: start.0,
                object_id: start.1,
            },
            end: Location {
                group_id: end.0,
                object_id: end.1,
            },
        }),
        joining: None,
        params: Params::new(),
    }
}

// ------------------------------------------------------------------ leg 1

/// The publisher answers a standalone FETCH out of its cache, in order, and
/// refuses everything it cannot serve rather than leaving the request hanging.
#[tokio::test]
async fn the_sink_serves_a_standalone_fetch_from_its_cache() {
    let tls = TestCert::generate();
    let (mut server, port) = bind_server(&tls);
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2gfetch";

    let mut sink = MoqtSink::new(&url, namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_cache_groups(2);
    sink.configure_pipeline(&bmff_caps())
        .expect("moqtsink caps");

    let mut relay = timeout(DIAL_TIMEOUT, peer16::accept_publisher(&mut server))
        .await
        .expect("the sink dialled when the pipeline was configured");
    match timeout(CONTROL_TIMEOUT, relay.recv())
        .await
        .expect("PUBLISH_NAMESPACE")
    {
        ControlMessage::PublishNamespace { id, .. } => {
            relay
                .send(ControlMessage::RequestOk {
                    id,
                    params: Params::new(),
                })
                .await;
        }
        other => panic!("expected PUBLISH_NAMESPACE, got {}", other.name()),
    }

    // Three GOPs of ten fragments each: groups 0, 1 and 2, of which the cache
    // keeps the last two.
    let published = publish_fragments(&mut sink, 30).await;
    assert_eq!(sink.track_names(), vec![String::from("1.m4s")]);
    assert_eq!(sink.fetches_served(), 0, "nothing has asked yet");

    // Group 1 whole: end object 0 means the entire group.
    relay
        .send(fetch(2, namespace, "1.m4s", (1, 0), (1, 0)))
        .await;
    match timeout(CONTROL_TIMEOUT, relay.recv())
        .await
        .expect("FETCH_OK")
    {
        ControlMessage::FetchOk { id, end, .. } => {
            assert_eq!(id, 2);
            assert_eq!(end.group_id, 1, "the response covers group 1: {end:?}");
        }
        other => panic!("expected FETCH_OK, got {}", other.name()),
    }
    let log = objects_when(&relay.fetched, |log| log.len() >= 10).await;
    assert!(
        log.iter().all(|(id, group, _)| *id == 2 && *group == 1),
        "every object is group 1 of request 2"
    );
    for pair in log.windows(2) {
        assert!(
            published
                .windows(pair[0].2.len())
                .position(|w| w == pair[0].2)
                .unwrap()
                < published
                    .windows(pair[1].2.len())
                    .position(|w| w == pair[1].2)
                    .unwrap(),
            "the fetched objects arrive in publish order"
        );
    }
    assert_eq!(sink.fetches_served(), 1);

    // Group 0 has fallen out of a two-group cache; group 9 was never published;
    // the track does not exist; and the joining fetch names no subscription.
    for (id, request, code) in [
        (
            4,
            fetch(4, namespace, "1.m4s", (0, 0), (0, 0)),
            request_error_code::INVALID_RANGE,
        ),
        (
            6,
            fetch(6, namespace, "1.m4s", (9, 0), (9, 0)),
            request_error_code::INVALID_RANGE,
        ),
        (
            8,
            fetch(8, namespace, "9.m4s", (1, 0), (1, 0)),
            request_error_code::DOES_NOT_EXIST,
        ),
        (
            10,
            // An end before its start asks for nothing.
            fetch(10, namespace, "1.m4s", (2, 5), (2, 1)),
            request_error_code::INVALID_RANGE,
        ),
        (
            12,
            ControlMessage::Fetch {
                id: 12,
                fetch_type: FetchType::RelativeJoining,
                standalone: None,
                joining: Some(JoiningFetch {
                    joining_request_id: 99,
                    joining_start: 1,
                }),
                params: Params::new(),
            },
            request_error_code::INVALID_JOINING_REQUEST_ID,
        ),
        (
            14,
            // A range whose arithmetic would leave u64 is refused, not folded.
            fetch(14, namespace, "1.m4s", (u64::MAX, u64::MAX), (u64::MAX, 0)),
            request_error_code::INVALID_RANGE,
        ),
    ] {
        relay.send(request).await;
        match timeout(CONTROL_TIMEOUT, relay.recv())
            .await
            .unwrap_or_else(|_| panic!("request {id} was answered"))
        {
            ControlMessage::RequestError {
                id: got,
                error_code,
                ..
            } => assert_eq!((got, error_code), (id, code), "request {id}"),
            other => panic!("request {id}: expected REQUEST_ERROR, got {}", other.name()),
        }
    }
    assert_eq!(sink.fetches_served(), 1, "a refused fetch serves nothing");
    assert_eq!(sink.fetches_cancelled(), 0);

    // A publisher that caches nothing refuses every FETCH.
    let mut off = MoqtSink::default();
    off.set_property("cache-groups", PropValue::Uint(0))
        .expect("cache-groups");
    assert_eq!(off.get_property("cache-groups"), Some(PropValue::Uint(0)));

    sink.process(PipelinePacket::Eos, &mut NullOut)
        .await
        .expect("clean end of stream");
}

// ------------------------------------------------------------------ leg 2

/// FETCH_CANCEL stops a response part way. The peer never reads the response
/// stream, so flow control holds the writer between objects and the cancel
/// reaches it before the range is done.
#[tokio::test]
async fn a_fetch_cancel_stops_the_response() {
    let tls = TestCert::generate();
    let (mut server, port) = bind_server(&tls);
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2gcancel";

    let mut sink = MoqtSink::new(&url, namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_cache_groups(4);
    sink.configure_pipeline(&bmff_caps())
        .expect("moqtsink caps");

    let mut relay = timeout(DIAL_TIMEOUT, peer16::accept_publisher(&mut server))
        .await
        .expect("dial");
    match timeout(CONTROL_TIMEOUT, relay.recv())
        .await
        .expect("PUBLISH_NAMESPACE")
    {
        ControlMessage::PublishNamespace { id, .. } => {
            relay
                .send(ControlMessage::RequestOk {
                    id,
                    params: Params::new(),
                })
                .await;
        }
        other => panic!("expected PUBLISH_NAMESPACE, got {}", other.name()),
    }

    // Twenty fragments of a quarter megabyte each: more than the peer's flow
    // control will take while nothing reads the response stream.
    publish_padded_fragments(&mut sink, 20, 256 * 1024).await;

    relay
        .send(fetch(2, namespace, "1.m4s", (0, 0), (2, 0)))
        .await;
    match timeout(CONTROL_TIMEOUT, relay.recv())
        .await
        .expect("FETCH_OK")
    {
        ControlMessage::FetchOk { id, .. } => assert_eq!(id, 2),
        other => panic!("expected FETCH_OK, got {}", other.name()),
    }
    relay.send(ControlMessage::FetchCancel { id: 2 }).await;

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while sink.fetches_cancelled() == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the cancelled response was abandoned"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(sink.fetches_served(), 1);
    // The response was reset part way, so the peer never saw the whole range.
    let delivered = relay.fetched.lock().expect("fetch log").len();
    assert!(
        delivered < 20,
        "the cancel stopped the response, got {delivered} of 20 objects"
    );
}

// ------------------------------------------------------------------ leg 3

/// Draft-18: a joining FETCH after a SUBSCRIBE is answered on the request
/// stream, and its objects end where the subscription starts.
#[tokio::test]
async fn the_sink_serves_a_draft_18_joining_fetch() {
    let tls = TestCert::generate();
    let (mut server, port) = bind_server(&tls);
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2gfetch18";

    let mut sink = MoqtSink::new(&url, namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_cache_groups(4);
    sink.configure_pipeline(&bmff_caps())
        .expect("moqtsink caps");

    let session = timeout(DIAL_TIMEOUT, peer18::accept_v18(&mut server))
        .await
        .expect("dial");
    let _control = peer18::exchange_setup(&session).await;
    let (mut ns, first) = timeout(CONTROL_TIMEOUT, peer18::accept_request(&session))
        .await
        .expect("PUBLISH_NAMESPACE");
    match first {
        Msg18::PublishNamespace { .. } => {
            ns.send(Msg18::RequestOk {
                params: MessageParams::new(),
                properties: Params::new(),
            })
            .await;
        }
        other => panic!("expected PUBLISH_NAMESPACE, got {}", other.name()),
    }
    let (_objects, fetched) = peer18::record_data(session.clone());

    // Three groups published before anyone subscribes.
    publish_fragments(&mut sink, 30).await;

    let mut media = peer18::subscribe(&session, 1, namespace, "1.m4s").await;
    match timeout(CONTROL_TIMEOUT, media.recv())
        .await
        .expect("SUBSCRIBE_OK")
    {
        g2g_plugins::moqt::v18::message::ControlMessage::SubscribeOk { track_alias, .. } => {
            assert_eq!(track_alias, 1)
        }
        other => panic!("expected SUBSCRIBE_OK, got {}", other.name()),
    }

    // One group back from the subscription's live edge.
    let (mut tx, rx) = session.open_bi().await.expect("a request stream");
    g2g_plugins::moqt::v18::session::write_message(
        &mut tx,
        &g2g_plugins::moqt::v18::message::ControlMessage::Fetch {
            id: 3,
            fetch_type: g2g_plugins::moqt::v18::message::FetchType::RelativeJoining,
            standalone: None,
            joining: Some(g2g_plugins::moqt::v18::message::JoiningFetch {
                joining_request_id: 1,
                joining_start: 1,
            }),
            params: MessageParams::new(),
        },
    )
    .await
    .expect("write FETCH");
    let mut fetch_request = peer18::RequestStream::new(tx, rx);
    match timeout(CONTROL_TIMEOUT, fetch_request.recv())
        .await
        .expect("FETCH_OK on the request stream")
    {
        g2g_plugins::moqt::v18::message::ControlMessage::FetchOk { end, .. } => {
            // The subscription was accepted at the end of group 2, so the fetch
            // ends there and the two are contiguous.
            assert_eq!(end.group_id, 2, "{end:?}");
        }
        other => panic!("expected FETCH_OK, got {}", other.name()),
    }

    let log = objects_when(&fetched, |log| log.iter().any(|(_, group, _)| *group == 2)).await;
    assert!(
        log.iter().all(|(id, ..)| *id == 3),
        "every object answers request 3"
    );
    let groups: Vec<u64> = log.iter().map(|(_, group, _)| *group).collect();
    assert!(
        groups.windows(2).all(|w| w[0] <= w[1]),
        "groups come out ascending: {groups:?}"
    );
    assert!(
        groups.contains(&1) && groups.contains(&2),
        "one group back from the live edge, and the current one: {groups:?}"
    );
    assert!(
        !groups.contains(&0),
        "a relative joining fetch of one group does not reach further back"
    );
    assert_eq!(sink.fetches_served(), 1);

    sink.process(PipelinePacket::Eos, &mut NullOut)
        .await
        .expect("clean end of stream");
}

// ------------------------------------------------------------------ leg 4

/// The subscriber's catch-up: `catchup-groups` issues a joining FETCH and its
/// objects are emitted before the live ones, even though the live objects
/// arrive while the fetch is still being written.
async fn catchup_plays_before_the_live_edge(version: MoqtVersion) {
    let tls = TestCert::generate();
    let (mut server, port) = bind_server(&tls);
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2gcatchup";

    let init = b"init-segment-bytes".to_vec();
    let fetched: Vec<Vec<u8>> = (0..4).map(|i| format!("old-{i}").into_bytes()).collect();
    let live: Vec<Vec<u8>> = (0..3).map(|i| format!("live-{i}").into_bytes()).collect();

    let mut src = MoqtSrc::new(&url, namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_catchup_groups(2);
    SourceLoop::set_property(&mut src, "catalog", PropValue::Bool(false)).expect("catalog");
    SourceLoop::set_property(&mut src, "versions", PropValue::Str(version_list(version)))
        .expect("versions");
    src.configure_pipeline(&bmff_caps()).expect("moqtsrc caps");

    let publisher = {
        let init = init.clone();
        let fetched = fetched.clone();
        let live = live.clone();
        async move {
            match version {
                MoqtVersion::V16 => publish16(&mut server, &init, &fetched, &live).await,
                MoqtVersion::V18 => publish18(&mut server, &init, &fetched, &live).await,
            }
        }
    };
    let run = async {
        let mut captured = CaptureSink::default();
        src.run(&mut captured).await.expect("the run completes");
        captured.frames
    };
    let (frames, _keep) = tokio::join!(run, publisher);

    let mut want = vec![init];
    want.extend(fetched.clone());
    want.extend(live);
    assert_eq!(
        frames, want,
        "{version:?}: the init segment, then the fetched groups, then the live ones"
    );
    assert_eq!(src.catchup_objects(), fetched.len() as u64);
}

fn version_list(version: MoqtVersion) -> String {
    match version {
        MoqtVersion::V16 => String::from("16"),
        MoqtVersion::V18 => String::from("18"),
    }
}

/// The draft-16 half of leg 4: answer the two SUBSCRIBEs and the joining FETCH,
/// serving the live group before the fetch response is finished.
/// The value returned exists only to keep the peer, its session and its streams
/// alive until the subscriber is done with them: dropping a stream resets it.
async fn publish16(
    server: &mut web_transport_quinn::Server,
    init: &[u8],
    fetched: &[Vec<u8>],
    live: &[Vec<u8>],
) -> Box<dyn core::any::Any> {
    let mut relay = peer16::accept_publisher(server).await;
    // The init track first, then the media track with its catch-up.
    let init_request = match relay.recv().await {
        ControlMessage::Subscribe { id, track_name, .. } => {
            assert_eq!(track_name.as_str_lossy(), "0.mp4");
            id
        }
        other => panic!("expected SUBSCRIBE, got {}", other.name()),
    };
    relay
        .send(ControlMessage::SubscribeOk {
            id: init_request,
            track_alias: 11,
            params: Params::new(),
            extensions: Params::new(),
        })
        .await;
    peer16::serve_group(&relay.session, 11, 0, &[init.to_vec()]).await;

    let media_request = match relay.recv().await {
        ControlMessage::Subscribe {
            id,
            track_name,
            params,
            ..
        } => {
            assert_eq!(track_name.as_str_lossy(), "1.m4s");
            assert!(
                params
                    .0
                    .iter()
                    .any(|(k, _)| *k == param::SUBSCRIPTION_FILTER),
                "a catch-up subscription asks for the Largest Object filter"
            );
            id
        }
        other => panic!("expected SUBSCRIBE, got {}", other.name()),
    };
    relay
        .send(ControlMessage::SubscribeOk {
            id: media_request,
            track_alias: 12,
            params: Params::new(),
            extensions: Params::new(),
        })
        .await;

    let fetch_request = match relay.recv().await {
        ControlMessage::Fetch {
            id,
            fetch_type,
            joining: Some(body),
            ..
        } => {
            assert_eq!(fetch_type, FetchType::RelativeJoining);
            assert_eq!(body.joining_request_id, media_request);
            assert_eq!(body.joining_start, 2);
            id
        }
        other => panic!("expected FETCH, got {}", other.name()),
    };
    relay
        .send(ControlMessage::FetchOk {
            id: fetch_request,
            end_of_track: false,
            end: Location {
                group_id: 4,
                object_id: 2,
            },
            params: Params::new(),
            extensions: Params::new(),
        })
        .await;

    // Half the fetch response, then the whole live group, then the rest: the
    // subscriber must still emit the older objects first.
    let mut response = FetchServer::open(MoqtVersion::V16, &relay.session, fetch_request).await;
    response.object(3, 0, &fetched[0]).await;
    response.object(3, 1, &fetched[1]).await;
    peer16::serve_group(&relay.session, 12, 5, live).await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    response.object(4, 0, &fetched[2]).await;
    response.object(4, 1, &fetched[3]).await;
    response.finish();

    relay
        .send(ControlMessage::PublishDone {
            id: media_request,
            status_code: g2g_plugins::moqt::message::publish_done_code::TRACK_ENDED,
            stream_count: 1,
            reason: String::from("end of stream"),
        })
        .await;
    Box::new(relay)
}

/// The draft-18 half of leg 4, with each request on its own stream.
/// See [`publish16`] on the return value.
async fn publish18(
    server: &mut web_transport_quinn::Server,
    init: &[u8],
    fetched: &[Vec<u8>],
    live: &[Vec<u8>],
) -> Box<dyn core::any::Any> {
    use g2g_plugins::moqt::v18::message::ControlMessage as Msg;

    let session = peer18::accept_v18(server).await;
    let control = peer18::exchange_setup(&session).await;
    let mut kept = Vec::new();

    let (mut init_request, first) = peer18::accept_request(&session).await;
    match first {
        Msg::Subscribe { track_name, .. } => assert_eq!(track_name.as_str_lossy(), "0.mp4"),
        other => panic!("expected SUBSCRIBE, got {}", other.name()),
    }
    init_request
        .send(Msg::SubscribeOk {
            track_alias: 11,
            params: MessageParams::new(),
            properties: Params::new(),
        })
        .await;
    peer18::serve_group(&session, 11, 0, &[init.to_vec()]).await;
    kept.push(init_request);

    let (mut media_request, first) = peer18::accept_request(&session).await;
    match first {
        Msg::Subscribe {
            track_name, params, ..
        } => {
            assert_eq!(track_name.as_str_lossy(), "1.m4s");
            assert!(
                params
                    .get(g2g_plugins::moqt::v18::coding::param::SUBSCRIPTION_FILTER)
                    .is_some(),
                "a catch-up subscription asks for the Largest Object filter"
            );
        }
        other => panic!("expected SUBSCRIBE, got {}", other.name()),
    }
    media_request
        .send(Msg::SubscribeOk {
            track_alias: 12,
            params: MessageParams::new(),
            properties: Params::new(),
        })
        .await;

    let (mut fetch_request, first) = peer18::accept_request(&session).await;
    let request_id = match first {
        Msg::Fetch {
            id,
            fetch_type,
            joining: Some(body),
            ..
        } => {
            assert_eq!(
                fetch_type,
                g2g_plugins::moqt::v18::message::FetchType::RelativeJoining
            );
            assert_eq!(body.joining_start, 2);
            id
        }
        other => panic!("expected FETCH, got {}", other.name()),
    };
    fetch_request
        .send(Msg::FetchOk {
            end_of_track: false,
            end: Location {
                group_id: 4,
                object_id: 2,
            },
            params: MessageParams::new(),
            properties: Params::new(),
        })
        .await;

    let mut response = FetchServer::open(MoqtVersion::V18, &session, request_id).await;
    response.object(3, 0, &fetched[0]).await;
    response.object(3, 1, &fetched[1]).await;
    peer18::serve_group(&session, 12, 5, live).await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    response.object(4, 0, &fetched[2]).await;
    response.object(4, 1, &fetched[3]).await;
    response.finish();

    media_request
        .send(Msg::PublishDone {
            status_code: g2g_plugins::moqt::v18::message::publish_done_code::TRACK_ENDED,
            stream_count: 1,
            reason: String::from("end of stream"),
        })
        .await;
    kept.push(media_request);
    kept.push(fetch_request);
    Box::new((session, control, kept))
}

#[tokio::test]
async fn the_source_plays_a_draft_16_catch_up_before_the_live_edge() {
    catchup_plays_before_the_live_edge(MoqtVersion::V16).await;
}

#[tokio::test]
async fn the_source_plays_a_draft_18_catch_up_before_the_live_edge() {
    catchup_plays_before_the_live_edge(MoqtVersion::V18).await;
}

// ------------------------------------------------------------------ properties

#[test]
fn the_fetch_properties_round_trip_and_are_declared() {
    let mut sink = MoqtSink::default();
    assert_eq!(
        AsyncElement::get_property(&sink, "cache-groups"),
        Some(PropValue::Uint(4))
    );
    AsyncElement::set_property(&mut sink, "cache-groups", PropValue::Uint(12)).expect("set");
    assert_eq!(
        AsyncElement::get_property(&sink, "cache-groups"),
        Some(PropValue::Uint(12))
    );
    assert!(AsyncElement::properties(&sink)
        .iter()
        .any(|p| p.name == "cache-groups"));

    let mut src = MoqtSrc::default();
    assert_eq!(
        SourceLoop::get_property(&src, "catchup-groups"),
        Some(PropValue::Uint(0))
    );
    SourceLoop::set_property(&mut src, "catchup-groups", PropValue::Uint(3)).expect("set");
    assert_eq!(
        SourceLoop::get_property(&src, "catchup-groups"),
        Some(PropValue::Uint(3))
    );
    assert!(SourceLoop::properties(&src)
        .iter()
        .any(|p| p.name == "catchup-groups"));
}
