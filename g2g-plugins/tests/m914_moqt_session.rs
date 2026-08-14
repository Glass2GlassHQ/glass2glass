//! M914: multi-track playback over one MoQ Transport session.
//!
//! `moqtsessionsrc` subscribes to N tracks of one broadcast and emits each on
//! its own pad, sharing the session driver with the single-track `moqtsrc`.
//!
//! The legs:
//!
//!   1. the publisher's side of it: a two-track `moov` (video + audio) names
//!      both tracks in the catalog and both are served, each on its own alias.
//!   2. the new element against a scripted two-track publisher: pad 0 gets the
//!      first track's fragments and pad 1 the second's, both after the shared
//!      init segment.
//!   3. the element surface: properties round-trip and `moqtsessionsrc`
//!      resolves for `parse_launch` as a fan-out source.
//!   4. interop: `mp4muxn ! moqtsink` -> `moq-relay-ietf` -> `moqtsessionsrc`,
//!      both tracks end to end. Skipped with a printed reason when the
//!      reference relay is absent.
#![cfg(feature = "moqt")]

use std::time::Duration;

use tokio::time::timeout;

use g2g_core::element::AsyncElement;
use g2g_core::frame::PipelinePacket;
use g2g_core::runtime::parse_launch;
use g2g_core::{ByteStreamEncoding, Caps, MultiOutputSource, PropValue};

use g2g_plugins::moqt::catalog;
use g2g_plugins::moqt::coding::Params;
use g2g_plugins::moqt::message::ControlMessage;
use g2g_plugins::moqtsessionsrc::MoqtSessionSrc;
use g2g_plugins::moqtsink::MoqtSink;
use g2g_plugins::registry::default_registry;

mod moqt_common;
use moqt_common::{
    bind_server, free_udp_port, objects_when, peer16, publish_av_fragments, spawn_relay, AvMuxer,
    CaptureMultiSink, NullOut, TestCert, CONTROL_TIMEOUT, DIAL_TIMEOUT,
};

fn bmff_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    }
}

// ------------------------------------------------------------------ leg 1

/// A two-track `moov` is published as two tracks: the catalog names both and
/// each serves its own fragments under its own alias.
#[tokio::test]
async fn the_sink_publishes_both_tracks_of_a_two_track_moov() {
    let tls = TestCert::generate();
    let (mut server, port) = bind_server(&tls);
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2gav";

    let mut sink = MoqtSink::new(&url, namespace).with_server_certificate_hashes(&tls.hash_hex);
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

    // Both media tracks, subscribed before the `moov` names them.
    for (id, track) in [(1u64, "1.m4s"), (3, "2.m4s")] {
        let msg = relay.subscribe(id, namespace, track);
        relay.send(msg).await;
    }

    let published = publish_av_fragments(&mut sink, 24).await;
    assert_eq!(
        sink.track_names(),
        vec![String::from("1.m4s"), String::from("2.m4s")],
        "the moov named a video and an audio track"
    );
    let catalog = String::from_utf8_lossy(sink.catalog()).into_owned();
    assert!(
        catalog.contains("\"1.m4s\"") && catalog.contains("\"2.m4s\""),
        "{catalog}"
    );
    assert!(
        catalog.contains("avc1.") && catalog.contains("mp4a.40.2"),
        "the catalog describes both codecs: {catalog}"
    );

    for id in [1u64, 3] {
        match timeout(CONTROL_TIMEOUT, relay.recv())
            .await
            .expect("both held subscriptions were answered")
        {
            ControlMessage::SubscribeOk { .. } => {}
            other => panic!("expected SUBSCRIBE_OK, got {}", other.name()),
        }
        let _ = id;
    }

    let log = objects_when(&relay.objects, |log| {
        log.iter().any(|(alias, ..)| *alias == 1) && log.iter().any(|(alias, ..)| *alias == 3)
    })
    .await;
    for (_, _, payload) in &log {
        assert!(
            published.windows(payload.len()).any(|w| w == payload),
            "every object is a fragment that was published"
        );
    }

    sink.process(PipelinePacket::Eos, &mut NullOut)
        .await
        .expect("clean end of stream");
}

// ------------------------------------------------------------------ leg 2

/// The session source plays two tracks at once: each pad gets the shared init
/// segment and then only its own track's fragments.
#[tokio::test]
async fn the_session_source_plays_two_tracks_on_two_pads() {
    let tls = TestCert::generate();
    let (mut server, port) = bind_server(&tls);
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2gsession";

    let init = b"init-segment-bytes".to_vec();
    let video: Vec<Vec<u8>> = (0..3).map(|i| format!("video-{i}").into_bytes()).collect();
    let audio: Vec<Vec<u8>> = (0..2).map(|i| format!("audio-{i}").into_bytes()).collect();

    let mut src = MoqtSessionSrc::new(&url, namespace)
        .with_outputs(2)
        .with_server_certificate_hashes(&tls.hash_hex);
    // No `tracks`: the pads take the catalog's tracks in order.
    let publisher = {
        let init = init.clone();
        let video = video.clone();
        let audio = audio.clone();
        async move {
            let mut relay = peer16::accept_publisher(&mut server).await;
            let catalog_json = catalog::build(
                &format!("/{namespace}"),
                "0.mp4",
                &[
                    (String::from("1.m4s"), String::new()),
                    (String::from("2.m4s"), String::new()),
                ],
            );
            let mut served: Vec<(String, u64)> = Vec::new();
            // Answer each SUBSCRIBE as it arrives and serve that track's group.
            while served.len() < 4 {
                let (id, name) = match relay.recv().await {
                    ControlMessage::Subscribe { id, track_name, .. } => {
                        (id, track_name.as_str_lossy())
                    }
                    other => panic!("expected SUBSCRIBE, got {}", other.name()),
                };
                let alias = 20 + served.len() as u64;
                relay
                    .send(ControlMessage::SubscribeOk {
                        id,
                        track_alias: alias,
                        params: Params::new(),
                        extensions: Params::new(),
                    })
                    .await;
                let payloads: Vec<Vec<u8>> = match name.as_str() {
                    ".catalog" => vec![catalog_json.clone().into_bytes()],
                    "0.mp4" => vec![init.clone()],
                    "1.m4s" => video.clone(),
                    "2.m4s" => audio.clone(),
                    other => panic!("unexpected track {other}"),
                };
                peer16::serve_group(&relay.session, alias, 0, &payloads).await;
                if name == "1.m4s" || name == "2.m4s" {
                    relay
                        .send(ControlMessage::PublishDone {
                            id,
                            status_code: g2g_plugins::moqt::message::publish_done_code::TRACK_ENDED,
                            stream_count: 1,
                            reason: String::from("end of stream"),
                        })
                        .await;
                }
                served.push((name, alias));
            }
            relay
        }
    };

    let mut captured = CaptureMultiSink::new(2);
    let run = async {
        MultiOutputSource::run(&mut src, &mut captured)
            .await
            .expect("the run completes")
    };
    let (emitted, _keep) = tokio::join!(run, publisher);

    assert_eq!(
        src.selected_tracks(),
        ["1.m4s".to_string(), "2.m4s".to_string()],
        "the pads took the catalog's track order"
    );
    let mut want_video = vec![init.clone()];
    want_video.extend(video);
    let mut want_audio = vec![init];
    want_audio.extend(audio);
    assert_eq!(
        captured.ports[0], want_video,
        "pad 0 played the video track"
    );
    assert_eq!(
        captured.ports[1], want_audio,
        "pad 1 played the audio track"
    );
    assert_eq!(emitted, 7, "two init segments and five fragments");
}

// ------------------------------------------------------------------ leg 3

#[test]
fn properties_round_trip_and_moqtsessionsrc_resolves_for_launch() {
    let mut src = MoqtSessionSrc::default();
    assert_eq!(src.output_count(), 1, "one pad until more are asked for");
    src.set_property("tracks", PropValue::Str("a.m4s,b.m4s".into()))
        .expect("tracks");
    assert_eq!(src.output_count(), 2, "naming two tracks asks for two pads");
    assert_eq!(
        src.get_property("tracks"),
        Some(PropValue::Str("a.m4s,b.m4s".into()))
    );
    assert!(src.output_caps(1).is_ok());
    assert!(src.output_caps(2).is_err(), "no third pad");

    for (name, value) in [
        ("location", PropValue::Str("https://relay:4443/".into())),
        ("namespace", PropValue::Str("live/cam".into())),
        ("catalog", PropValue::Bool(false)),
        ("catchup-groups", PropValue::Uint(2)),
        ("num-buffers", PropValue::Int(9)),
        ("timeout", PropValue::Uint(2500)),
        ("max-groups", PropValue::Uint(4)),
    ] {
        src.set_property(name, value.clone())
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(src.get_property(name), Some(value), "{name} reads back");
    }
    assert!(src
        .set_property("versions", PropValue::Str("17".into()))
        .is_err());
    assert!(src.set_property("nope", PropValue::Uint(1)).is_err());

    // parse_launch resolves it as a fan-out source with one branch per pad.
    let reg = default_registry();
    let line = "moqtsessionsrc name=s location=https://relay:4443/ namespace=live/cam \
                tracks=1.m4s,2.m4s  s. ! fmp4demux ! fakesink  s. ! fmp4demux ! fakesink";
    let err = parse_launch(&reg, line)
        .err()
        .map(|e| format!("{e}"))
        .unwrap_or_default();
    assert!(
        !err.contains("unknown element") && !err.contains("unknown property"),
        "moqtsessionsrc and its properties resolve: {err}"
    );
}

// ------------------------------------------------------------------ leg 4

/// `mp4muxn ! moqtsink` -> `moq-relay-ietf` -> `moqtsessionsrc`: two tracks
/// through the reference relay, each arriving on its own pad.
#[tokio::test]
async fn two_tracks_round_trip_through_the_reference_relay() {
    let tls = TestCert::generate();
    let port = free_udp_port();
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2gavrelay";
    let Some(_relay) = spawn_relay(&tls, port) else {
        return;
    };
    tokio::time::sleep(Duration::from_millis(750)).await;

    let mut sink = MoqtSink::new(&url, namespace).with_server_certificate_hashes(&tls.hash_hex);
    sink.configure_pipeline(&bmff_caps())
        .expect("moqtsink caps");

    let mut src = MoqtSessionSrc::new(&url, namespace)
        .with_outputs(2)
        .with_tracks("1.m4s,2.m4s")
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_num_buffers(10);

    let done = std::cell::Cell::new(false);
    let publish = async {
        let mut mux = AvMuxer::new();
        let mut published = Vec::new();
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while !done.get() && std::time::Instant::now() < deadline {
            published.extend(mux.step(&mut sink).await);
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        sink.process(PipelinePacket::Eos, &mut NullOut)
            .await
            .expect("clean end of stream");
        published
    };

    let mut captured = CaptureMultiSink::new(2);
    let subscribe = async {
        tokio::time::sleep(Duration::from_millis(1500)).await;
        let emitted = MultiOutputSource::run(&mut src, &mut captured).await;
        done.set(true);
        emitted
    };
    let (published, emitted) = tokio::join!(publish, subscribe);
    let emitted = emitted.expect("subscribe and play");

    assert!(emitted >= 2, "both pads got at least their init segment");
    for port in 0..2 {
        assert!(
            !captured.ports[port].is_empty(),
            "pad {port} received the broadcast"
        );
        for payload in &captured.ports[port] {
            assert!(
                published.windows(payload.len()).any(|w| w == payload),
                "pad {port} emitted bytes that were published"
            );
        }
    }
    assert!(
        captured.ports[1].len() > 1,
        "the second pad played fragments of its own track, not just the init segment"
    );
}
