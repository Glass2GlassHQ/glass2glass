//! M913: PUBLISH-initiated subscriptions, both sides and both drafts.
//!
//! A publisher normally waits to be asked. With `publish=true` `moqtsink`
//! offers each track with PUBLISH as soon as the `moov` names it, and `moqtsrc`
//! accepts an incoming PUBLISH for a track it wants as the other way a
//! subscription is established.
//!
//! The legs, each against a scripted in-process peer:
//!
//!   1. publisher, draft-16: PUBLISH per track once the `moov` exists, the
//!      accepted ones serve objects and the refused one serves nothing.
//!   2. publisher, draft-18: the same over one request stream per PUBLISH, with
//!      the answer and PUBLISH_DONE riding that stream.
//!   3. subscriber, both drafts: a PUBLISH for the tracks this run wants is
//!      answered with PUBLISH_OK and plays without any SUBSCRIBE being needed,
//!      while a PUBLISH for another track is refused.
#![cfg(feature = "moqt")]

use std::time::Duration;

use tokio::time::timeout;

use g2g_core::element::AsyncElement;
use g2g_core::frame::PipelinePacket;
use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, PropValue};

use g2g_plugins::moqt::coding::{Params, TrackName, TrackNamespace};
use g2g_plugins::moqt::message::{request_error_code, ControlMessage};
use g2g_plugins::moqt::v18::coding::MessageParams;
use g2g_plugins::moqt::v18::message::ControlMessage as Msg18;
use g2g_plugins::moqt::MoqtVersion;
use g2g_plugins::moqtsink::MoqtSink;
use g2g_plugins::moqtsrc::MoqtSrc;

mod moqt_common;
use moqt_common::{
    bind_server, objects_when, peer16, peer18, publish_fragments, CaptureSink, NullOut, TestCert,
    CONTROL_TIMEOUT, DIAL_TIMEOUT,
};

fn bmff_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    }
}

// ------------------------------------------------------------------ leg 1

/// Draft-16: the sink offers each track with PUBLISH, serves the ones the peer
/// accepts, and stays quiet on the one it refuses.
#[tokio::test]
async fn the_sink_initiates_with_publish_on_draft_16() {
    let tls = TestCert::generate();
    let (mut server, port) = bind_server(&tls);
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2gpublish";

    let mut sink = MoqtSink::new(&url, namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_publish(true);
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

    // Nothing is offered before the `moov` names the tracks.
    assert!(
        relay
            .recv_within(Duration::from_millis(300))
            .await
            .is_none(),
        "a PUBLISH cannot name a track the moov has not declared"
    );

    let published = publish_fragments(&mut sink, 12).await;

    // One PUBLISH per track: the catalog, the init segment and the media track.
    let mut offered = Vec::new();
    while offered.len() < 3 {
        match timeout(CONTROL_TIMEOUT, relay.recv())
            .await
            .expect("a PUBLISH per track")
        {
            ControlMessage::Publish {
                id,
                namespace: got,
                track_name,
                track_alias,
                ..
            } => {
                assert_eq!(got, TrackNamespace::from_path(namespace));
                assert_eq!(track_alias, id, "the alias is the request id");
                offered.push((id, track_name.as_str_lossy()));
            }
            other => panic!("expected PUBLISH, got {}", other.name()),
        }
    }
    let names: Vec<&str> = offered.iter().map(|(_, name)| name.as_str()).collect();
    assert!(
        names.contains(&".catalog") && names.contains(&"0.mp4") && names.contains(&"1.m4s"),
        "every track was offered: {names:?}"
    );

    // Accept the init and media tracks, refuse the catalog.
    let mut media_alias = 0;
    let mut catalog_alias = 0;
    for (id, name) in &offered {
        if name == ".catalog" {
            catalog_alias = *id;
            relay
                .send(ControlMessage::RequestError {
                    id: *id,
                    error_code: request_error_code::UNINTERESTED,
                    retry_interval: 0,
                    reason: String::from("no catalog wanted"),
                })
                .await;
            continue;
        }
        if name == "1.m4s" {
            media_alias = *id;
        }
        relay
            .send(ControlMessage::PublishOk {
                id: *id,
                params: Params::new(),
            })
            .await;
    }

    let more = publish_fragments(&mut sink, 20).await;
    let log = objects_when(&relay.objects, |log| {
        log.iter()
            .filter(|(alias, ..)| *alias == media_alias)
            .count()
            >= 2
    })
    .await;
    for (_, _, payload) in log.iter().filter(|(alias, ..)| *alias == media_alias) {
        let whole: Vec<u8> = [published.clone(), more.clone()].concat();
        assert!(
            whole.windows(payload.len()).any(|w| w == payload),
            "every media object is a fragment that was published"
        );
    }
    assert!(
        !log.iter().any(|(alias, ..)| *alias == catalog_alias),
        "the refused track serves nothing"
    );

    sink.process(PipelinePacket::Eos, &mut NullOut)
        .await
        .expect("clean end of stream");
}

// ------------------------------------------------------------------ leg 2

/// Draft-18: each PUBLISH opens its own request stream, is answered there, and
/// PUBLISH_DONE closes it at end of stream.
#[tokio::test]
async fn the_sink_initiates_with_publish_on_draft_18() {
    let tls = TestCert::generate();
    let (mut server, port) = bind_server(&tls);
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2gpublish18";

    let mut sink = MoqtSink::new(&url, namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_publish(true);
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
    let (objects, _fetched) = peer18::record_data(session.clone());

    let published = publish_fragments(&mut sink, 12).await;

    let mut streams = Vec::new();
    let mut media_alias = None;
    for _ in 0..3 {
        let (mut request, first) = timeout(CONTROL_TIMEOUT, peer18::accept_request(&session))
            .await
            .expect("a PUBLISH per track");
        let (id, name) = match first {
            Msg18::Publish {
                id,
                track_name,
                track_alias,
                ..
            } => {
                assert_eq!(track_alias, id, "the alias is the request id");
                (id, track_name.as_str_lossy())
            }
            other => panic!("expected PUBLISH, got {}", other.name()),
        };
        if name == "1.m4s" {
            media_alias = Some(id);
        }
        request
            .send(Msg18::PublishOk {
                params: MessageParams::new(),
                properties: Params::new(),
            })
            .await;
        streams.push((name, request));
    }
    let media_alias = media_alias.expect("the media track was offered");

    let more = publish_fragments(&mut sink, 20).await;
    let log = objects_when(&objects, |log| {
        log.iter()
            .filter(|(alias, ..)| *alias == media_alias)
            .count()
            >= 2
    })
    .await;
    let whole: Vec<u8> = [published, more].concat();
    for (_, _, payload) in log.iter().filter(|(alias, ..)| *alias == media_alias) {
        assert!(
            whole.windows(payload.len()).any(|w| w == payload),
            "every media object is a fragment that was published"
        );
    }

    sink.process(PipelinePacket::Eos, &mut NullOut)
        .await
        .expect("clean end of stream");
    // PUBLISH_DONE rides each accepted track's request stream.
    let mut done = 0;
    for (_, request) in &mut streams {
        if let Some(Msg18::PublishDone { .. }) = request.recv_within(CONTROL_TIMEOUT).await {
            done += 1;
        }
    }
    assert!(
        done > 0,
        "end of stream told the subscriber the track ended"
    );
}

// ------------------------------------------------------------------ leg 3

/// The subscriber plays a broadcast it never subscribed to: the publisher's
/// PUBLISH establishes each track, and a track this run does not want is
/// refused.
async fn the_source_plays_a_publish_initiated_broadcast(version: MoqtVersion) {
    let tls = TestCert::generate();
    let (mut server, port) = bind_server(&tls);
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2gpubsrc";

    let init = b"init-segment-bytes".to_vec();
    let media: Vec<Vec<u8>> = (0..3).map(|i| format!("frag-{i}").into_bytes()).collect();

    let mut src = MoqtSrc::new(&url, namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_track_name("1.m4s");
    SourceLoop::set_property(&mut src, "catalog", PropValue::Bool(false)).expect("catalog");
    SourceLoop::set_property(
        &mut src,
        "versions",
        PropValue::Str(match version {
            MoqtVersion::V16 => String::from("16"),
            MoqtVersion::V18 => String::from("18"),
        }),
    )
    .expect("versions");
    src.configure_pipeline(&bmff_caps()).expect("moqtsrc caps");

    let publisher = {
        let init = init.clone();
        let media = media.clone();
        async move {
            match version {
                MoqtVersion::V16 => publish16(&mut server, &init, &media).await,
                MoqtVersion::V18 => publish18(&mut server, &init, &media).await,
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
    want.extend(media);
    assert_eq!(
        frames, want,
        "{version:?}: the init segment and the media objects, with no SUBSCRIBE needed"
    );
}

/// See [`publish16`] in `m912`: the return value keeps the peer alive.
async fn publish16(
    server: &mut web_transport_quinn::Server,
    init: &[u8],
    media: &[Vec<u8>],
) -> Box<dyn core::any::Any> {
    let mut relay = peer16::accept_publisher(server).await;
    let ns = TrackNamespace::from_path("g2gpubsrc");
    for (id, name, alias) in [(1u64, "0.mp4", 11u64), (3, "1.m4s", 12)] {
        relay
            .send(ControlMessage::Publish {
                id,
                namespace: ns.clone(),
                track_name: TrackName::new(name),
                track_alias: alias,
                params: Params::new(),
                extensions: Params::new(),
            })
            .await;
    }
    // A track this run did not ask for.
    relay
        .send(ControlMessage::Publish {
            id: 5,
            namespace: ns.clone(),
            track_name: TrackName::new("9.m4s"),
            track_alias: 13,
            params: Params::new(),
            extensions: Params::new(),
        })
        .await;

    let mut accepted = Vec::new();
    let mut refused = Vec::new();
    while accepted.len() < 2 || refused.is_empty() {
        match relay.recv().await {
            ControlMessage::PublishOk { id, .. } => accepted.push(id),
            ControlMessage::RequestError { id, error_code, .. } => {
                assert_eq!(error_code, request_error_code::UNINTERESTED);
                refused.push(id);
            }
            // The subscriber's own SUBSCRIBE for a track it is about to be
            // offered: this peer only publishes, so it goes unanswered.
            ControlMessage::Subscribe { .. } => {}
            other => panic!("unexpected {}", other.name()),
        }
    }
    assert_eq!(refused, vec![5], "the track this run does not want");
    assert!(accepted.contains(&1) && accepted.contains(&3));

    peer16::serve_group(&relay.session, 11, 0, &[init.to_vec()]).await;
    peer16::serve_group(&relay.session, 12, 0, media).await;
    relay
        .send(ControlMessage::PublishDone {
            id: 3,
            status_code: g2g_plugins::moqt::message::publish_done_code::TRACK_ENDED,
            stream_count: 1,
            reason: String::from("end of stream"),
        })
        .await;
    Box::new(relay)
}

/// The draft-18 half: one request stream per PUBLISH.
async fn publish18(
    server: &mut web_transport_quinn::Server,
    init: &[u8],
    media: &[Vec<u8>],
) -> Box<dyn core::any::Any> {
    let session = peer18::accept_v18(server).await;
    let control = peer18::exchange_setup(&session).await;
    let ns = TrackNamespace::from_path("g2gpubsrc");
    let mut kept = Vec::new();

    for (id, name, alias, wanted) in [
        (1u64, "0.mp4", 11u64, true),
        (3, "1.m4s", 12, true),
        (5, "9.m4s", 13, false),
    ] {
        let (mut tx, rx) = session.open_bi().await.expect("a request stream");
        g2g_plugins::moqt::v18::session::write_message(
            &mut tx,
            &Msg18::Publish {
                id,
                namespace: ns.clone(),
                track_name: TrackName::new(name),
                track_alias: alias,
                params: MessageParams::new(),
                properties: Params::new(),
            },
        )
        .await
        .expect("write PUBLISH");
        let mut request = peer18::RequestStream::new(tx, rx);
        match request.recv().await {
            Msg18::PublishOk { .. } => assert!(wanted, "{name} was accepted"),
            Msg18::RequestError { error_code, .. } => {
                assert!(!wanted, "{name} was refused");
                assert_eq!(
                    error_code,
                    g2g_plugins::moqt::v18::message::request_error_code::UNINTERESTED
                );
            }
            other => panic!("unexpected {}", other.name()),
        }
        kept.push(request);
    }

    peer18::serve_group(&session, 11, 0, &[init.to_vec()]).await;
    peer18::serve_group(&session, 12, 0, media).await;
    kept[1]
        .send(Msg18::PublishDone {
            status_code: g2g_plugins::moqt::v18::message::publish_done_code::TRACK_ENDED,
            stream_count: 1,
            reason: String::from("end of stream"),
        })
        .await;
    Box::new((session, control, kept))
}

#[tokio::test]
async fn the_source_plays_a_draft_16_publish_initiated_broadcast() {
    the_source_plays_a_publish_initiated_broadcast(MoqtVersion::V16).await;
}

#[tokio::test]
async fn the_source_plays_a_draft_18_publish_initiated_broadcast() {
    the_source_plays_a_publish_initiated_broadcast(MoqtVersion::V18).await;
}

#[test]
fn the_publish_property_round_trips_and_is_declared() {
    let mut sink = MoqtSink::default();
    assert_eq!(
        AsyncElement::get_property(&sink, "publish"),
        Some(PropValue::Bool(false))
    );
    AsyncElement::set_property(&mut sink, "publish", PropValue::Bool(true)).expect("set");
    assert_eq!(
        AsyncElement::get_property(&sink, "publish"),
        Some(PropValue::Bool(true))
    );
    assert!(AsyncElement::properties(&sink)
        .iter()
        .any(|p| p.name == "publish"));
}
