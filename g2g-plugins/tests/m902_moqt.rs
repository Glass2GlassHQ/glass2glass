//! M902: native IETF MoQ Transport (draft-16, `0xff000010`) over the M901
//! WebTransport carrier, and the `moqtsink` publisher on top of it.
//!
//! The legs:
//!   1. the wire codec: every control message and data header round-trips, and
//!      the byte layouts match the ones the reference implementation
//!      (`cloudflare/moq-rs`, `moq-transport/src/{message,setup,data}`) asserts
//!      for itself. A round trip alone cannot catch two fields swapped with each
//!      other, which is exactly the mistake that breaks interop.
//!   2. malformed peer input: truncated varints, lengths that overrun the
//!      buffer, absurd tuple counts, reserved enum values. Each must fail the
//!      parse, not panic and not allocate on the peer's number.
//!   3. the element surface: properties round-trip and `moqtsink` resolves for
//!      `parse_launch`.
//!   4. interop: `mp4mux ! moqtsink` publishing through a locally spawned
//!      `moq-relay-ietf`, consumed by `moq-sub`, asserting the fMP4 that comes
//!      out the far side is byte-for-byte the fMP4 that went in. Skipped with a
//!      printed reason when the reference binaries are absent.
#![cfg(feature = "moqt")]

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use g2g_core::element::AsyncElement;
use g2g_core::frame::PipelinePacket;
use g2g_core::{ByteStreamEncoding, Caps, PropValue};

use g2g_core::runtime::parse_launch;
use g2g_plugins::moqt::coding::{
    put_varint, MoqtError, ParamValue, Params, Reader, TrackName, TrackNamespace,
    TrackNamespacePrefix, MAX_FULL_TRACK_NAME_LEN, VARINT_MAX,
};
use g2g_plugins::moqt::data::{
    ObjectStatus, StreamHeaderType, SubgroupHeader, SubgroupObjectHeader,
};
use g2g_plugins::moqt::message::{
    msg_type, ControlMessage, FetchType, JoiningFetch, Location, StandaloneFetch,
    SubscribeNamespaceOptions,
};
use g2g_plugins::moqt::session::{MOQT_PROTOCOL, MOQT_VERSION};
use g2g_plugins::moqtsink::MoqtSink;
use g2g_plugins::mp4mux::Mp4Mux;
use g2g_plugins::registry::default_registry;

mod moqt_common;
use moqt_common::{
    access_unit, box_boundaries, frame, free_udp_port, h264_caps, init_len, moof_count,
    reference_binary, spawn_relay, CaptureSink, NullOut, Reaped, TestCert,
};

// ------------------------------------------------------------------ leg 1

fn encoded(msg: &ControlMessage) -> Vec<u8> {
    let mut out = Vec::new();
    msg.encode(&mut out).expect("encode");
    out
}

fn params() -> Params {
    let mut p = Params::new();
    p.set_int(2, 100);
    p.set_bytes(3, b"token".to_vec());
    p
}

/// Every control message the session speaks survives a round trip through the
/// public codec, consuming exactly its frame.
#[test]
fn every_control_message_round_trips() {
    let ns = TrackNamespace::from_path("live/cam");
    let messages = vec![
        ControlMessage::ClientSetup { params: params() },
        ControlMessage::ServerSetup { params: params() },
        ControlMessage::RequestUpdate {
            id: 2,
            existing_request_id: 0,
            params: params(),
        },
        ControlMessage::Subscribe {
            id: 4,
            namespace: ns.clone(),
            track_name: TrackName::new("1.m4s"),
            params: params(),
        },
        ControlMessage::SubscribeOk {
            id: 4,
            track_alias: 4,
            params: params(),
            extensions: Params(vec![(0, ParamValue::Int(7))]),
        },
        ControlMessage::RequestError {
            id: 4,
            error_code: 0x10,
            retry_interval: 0,
            reason: "no such track".into(),
        },
        ControlMessage::PublishNamespace {
            id: 0,
            namespace: ns.clone(),
            params: params(),
        },
        ControlMessage::RequestOk {
            id: 0,
            params: params(),
        },
        ControlMessage::Namespace {
            suffix: TrackNamespacePrefix(vec![b"cam".to_vec()]),
        },
        ControlMessage::NamespaceDone {
            suffix: TrackNamespacePrefix(vec![b"cam".to_vec()]),
        },
        ControlMessage::PublishNamespaceDone { id: 0 },
        ControlMessage::PublishNamespaceCancel {
            id: 0,
            error_code: 1,
            reason: "expired".into(),
        },
        ControlMessage::Unsubscribe { id: 4 },
        ControlMessage::PublishDone {
            id: 4,
            status_code: 2,
            stream_count: 12,
            reason: "eos".into(),
        },
        ControlMessage::TrackStatus {
            id: 6,
            namespace: ns.clone(),
            track_name: TrackName::new("1.m4s"),
            params: params(),
        },
        ControlMessage::GoAway {
            uri: "https://relay.example/2".into(),
        },
        ControlMessage::SubscribeNamespace {
            id: 8,
            prefix: TrackNamespacePrefix(vec![b"live".to_vec()]),
            options: SubscribeNamespaceOptions::Namespace,
            params: params(),
        },
        ControlMessage::MaxRequestId { request_id: 100 },
        ControlMessage::RequestsBlocked {
            max_request_id: 100,
        },
        ControlMessage::Fetch {
            id: 10,
            fetch_type: FetchType::Standalone,
            standalone: Some(StandaloneFetch {
                namespace: ns.clone(),
                track_name: TrackName::new("1.m4s"),
                start: Location::default(),
                end: Location {
                    group_id: 3,
                    object_id: 9,
                },
            }),
            joining: None,
            params: params(),
        },
        ControlMessage::Fetch {
            id: 12,
            fetch_type: FetchType::AbsoluteJoining,
            standalone: None,
            joining: Some(JoiningFetch {
                joining_request_id: 4,
                joining_start: 2,
            }),
            params: params(),
        },
        ControlMessage::FetchCancel { id: 10 },
        ControlMessage::FetchOk {
            id: 10,
            end_of_track: true,
            end: Location {
                group_id: 3,
                object_id: 9,
            },
            params: params(),
            extensions: Params::new(),
        },
        ControlMessage::Publish {
            id: 14,
            namespace: ns,
            track_name: TrackName::new("1.m4s"),
            track_alias: 14,
            params: params(),
            extensions: Params::new(),
        },
        ControlMessage::PublishOk {
            id: 14,
            params: params(),
        },
    ];
    // Everything in draft-16 Table 1 that a session over this carrier can see.
    assert_eq!(messages.len(), 25);

    for msg in messages {
        let bytes = encoded(&msg);
        let (decoded, used) = ControlMessage::decode(&bytes).expect("decode");
        assert_eq!(used, bytes.len(), "{} consumed its frame", msg.name());
        assert_eq!(decoded, msg, "{} round trip", msg.name());
    }
}

/// Byte layouts copied from the reference's own assertions. These are what
/// interop actually rests on.
#[test]
fn wire_layouts_match_the_reference_implementation() {
    assert_eq!(MOQT_VERSION, 0xff00_0010, "draft-16");
    assert_eq!(MOQT_PROTOCOL, "moqt-16");

    let ns = TrackNamespace::from_path("ns");

    // moq-transport/src/message/mod.rs: draft16_wire_layouts_for_changed_control_messages
    assert_eq!(
        encoded(&ControlMessage::Subscribe {
            id: 0,
            namespace: ns.clone(),
            track_name: TrackName::new("t"),
            params: Params::new(),
        }),
        vec![0x03, 0x00, 0x08, 0x00, 0x01, 0x02, b'n', b's', 0x01, b't', 0x00]
    );
    assert_eq!(
        encoded(&ControlMessage::SubscribeOk {
            id: 0,
            track_alias: 1,
            params: Params::new(),
            extensions: Params::new(),
        }),
        vec![0x04, 0x00, 0x03, 0x00, 0x01, 0x00]
    );
    assert_eq!(
        encoded(&ControlMessage::SubscribeNamespace {
            id: 0,
            prefix: TrackNamespacePrefix::default(),
            options: SubscribeNamespaceOptions::Both,
            params: Params::new(),
        }),
        vec![0x11, 0x00, 0x04, 0x00, 0x00, 0x02, 0x00]
    );

    // moq-transport/src/setup/{client,server}.rs: draft-16 SETUP is parameters
    // only, since the version rides the WebTransport subprotocol.
    assert_eq!(
        encoded(&ControlMessage::ClientSetup {
            params: Params::new()
        }),
        vec![0x20, 0x00, 0x01, 0x00]
    );
    assert_eq!(
        encoded(&ControlMessage::ServerSetup {
            params: Params::new()
        }),
        vec![0x21, 0x00, 0x01, 0x00]
    );

    // moq-transport/src/coding/kvp.rs: a KVP's key is a delta from the previous
    // one, odd keys carry bytes and even keys a varint.
    let mut p = Params::new();
    p.set_bytes(1, b"testpath".to_vec());
    p.set_int(2, 100);
    let mut kvp = Vec::new();
    p.encode(&mut kvp).expect("encode");
    assert_eq!(
        kvp,
        vec![0x02, 0x01, 0x08, b't', b'e', b's', b't', b'p', b'a', b't', b'h', 0x01, 0x40, 0x64]
    );

    // moq-transport/src/session/subscribed.rs: the reference publisher opens
    // every subgroup with 0x15 (explicit subgroup id, extension headers) and
    // writes each object with a zero delta and an empty extension block.
    let mut stream = Vec::new();
    SubgroupHeader {
        header_type: StreamHeaderType::SubgroupIdExt,
        track_alias: 4,
        group_id: 7,
        subgroup_id: Some(0),
        publisher_priority: 127,
    }
    .encode(&mut stream)
    .expect("header");
    SubgroupObjectHeader::normal(0, 3)
        .encode(StreamHeaderType::SubgroupIdExt, &mut stream)
        .expect("object");
    stream.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
    assert_eq!(
        stream,
        vec![0x15, 0x04, 0x07, 0x00, 0x7f, 0x00, 0x00, 0x03, 0xAA, 0xBB, 0xCC]
    );
}

// ------------------------------------------------------------------ leg 2

/// Everything a peer controls (varints, lengths, counts, enum values) is
/// bounded before use: a malformed message fails the parse rather than
/// panicking or allocating on the number the peer sent.
#[test]
fn malformed_peer_input_fails_the_parse() {
    // A truncated varint needs more bytes; it does not read past the buffer.
    assert_eq!(
        Reader::new(&[0x9d, 0x7f]).varint(),
        Err(MoqtError::Incomplete)
    );

    // A frame cut at every possible point is incomplete, never a panic.
    let frame = encoded(&ControlMessage::Subscribe {
        id: 0,
        namespace: TrackNamespace::from_path("ns"),
        track_name: TrackName::new("t"),
        params: Params::new(),
    });
    for cut in 0..frame.len() {
        assert_eq!(
            ControlMessage::decode(&frame[..cut]).map(|(_, n)| n),
            Err(MoqtError::Incomplete),
            "a {cut}-byte prefix is a partial frame"
        );
    }

    // A declared length longer than the message needs leaves bytes over: §9
    // calls that a protocol violation, so it must not decode.
    let mut padded = frame.clone();
    padded[2] += 1;
    padded.push(0x00);
    assert_eq!(
        ControlMessage::decode(&padded).map(|(_, n)| n),
        Err(MoqtError::Malformed)
    );

    // An unassigned message type closes the session rather than being skipped.
    assert_eq!(
        ControlMessage::decode(&[0x40, 0x99, 0x00, 0x00]).map(|(_, n)| n),
        Err(MoqtError::Malformed)
    );

    // A namespace tuple count past the draft's 32-field limit.
    assert_eq!(
        ControlMessage::decode_payload(msg_type::SUBSCRIBE, &[0x00, 0x40, 0xff, 0x01, b'a']),
        Err(MoqtError::Malformed)
    );
    // ...and one within the limit whose fields overrun the payload.
    assert_eq!(
        ControlMessage::decode_payload(msg_type::SUBSCRIBE, &[0x00, 0x20, 0x01, b'a']),
        Err(MoqtError::Incomplete)
    );
    // A namespace field length that overruns the buffer.
    assert_eq!(
        TrackNamespace::decode(&mut Reader::new(&[0x01, 0x40, 0xff, b'a'])),
        Err(MoqtError::Incomplete)
    );
    // An empty namespace field is a protocol violation.
    assert_eq!(
        TrackNamespace::decode(&mut Reader::new(&[0x01, 0x00])),
        Err(MoqtError::Malformed)
    );
    // A full track name past 4096 bytes.
    let mut oversized = Vec::new();
    put_varint(&mut oversized, 1);
    put_varint(&mut oversized, MAX_FULL_TRACK_NAME_LEN as u64 + 1);
    oversized.extend_from_slice(&[b'a'; 16]);
    assert_eq!(
        TrackNamespace::decode(&mut Reader::new(&oversized)),
        Err(MoqtError::Malformed)
    );

    // A parameter count of 2^62-1 must run out of buffer, not allocate on it.
    let mut absurd_count = Vec::new();
    put_varint(&mut absurd_count, VARINT_MAX);
    assert_eq!(
        Params::decode(&mut Reader::new(&absurd_count)),
        Err(MoqtError::Incomplete)
    );

    // Reserved / unassigned enum values in each position that carries one.
    assert_eq!(
        ControlMessage::decode(&[0x11, 0x00, 0x04, 0x00, 0x00, 0x09, 0x00]).map(|(_, n)| n),
        Err(MoqtError::Malformed),
        "SUBSCRIBE_NAMESPACE option"
    );
    assert_eq!(
        ControlMessage::decode(&[0x16, 0x00, 0x02, 0x00, 0x09]).map(|(_, n)| n),
        Err(MoqtError::Malformed),
        "FETCH type"
    );
    assert_eq!(
        StreamHeaderType::from_code(0x16),
        Err(MoqtError::Malformed),
        "stream header type"
    );
    assert_eq!(
        SubgroupObjectHeader::decode(
            StreamHeaderType::SubgroupIdExt,
            &mut Reader::new(&[0x00, 0x00, 0x00, 0x01])
        ),
        Err(MoqtError::Malformed),
        "object status 0x1 was removed in draft-16"
    );

    // A subgroup header cut anywhere is incomplete.
    let header = [0x15u8, 0x04, 0x07, 0x00, 0x7f];
    for cut in 0..header.len() {
        assert_eq!(
            SubgroupHeader::decode(&mut Reader::new(&header[..cut])),
            Err(MoqtError::Incomplete)
        );
    }

    // A zero-length object carries a status, and a non-normal status with
    // extension headers is refused.
    let mut zero = SubgroupObjectHeader::normal(0, 0);
    assert_eq!(
        zero.encode(StreamHeaderType::SubgroupIdExt, &mut Vec::new()),
        Err(MoqtError::Malformed)
    );
    zero.status = Some(ObjectStatus::EndOfGroup);
    zero.encode(StreamHeaderType::SubgroupIdExt, &mut Vec::new())
        .expect("a status makes it encodable");
}

// ------------------------------------------------------------------ leg 3

#[test]
fn properties_round_trip_and_moqtsink_resolves_for_launch() {
    let mut sink = MoqtSink::new("https://127.0.0.1:4443/", "g2g");

    let names: Vec<&str> = AsyncElement::properties(&sink)
        .iter()
        .map(|p| p.name)
        .collect();
    for expected in [
        "location",
        "namespace",
        "track-name",
        "init-track-name",
        "catalog-track-name",
        "catalog",
        "priority",
        "max-request-id",
        "server-certificate-hashes",
    ] {
        assert!(names.contains(&expected), "{expected} is declared");
    }

    let set = |sink: &mut MoqtSink, name: &str, value: PropValue| {
        AsyncElement::set_property(sink, name, value).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    };
    set(
        &mut sink,
        "location",
        PropValue::Str("https://relay.example:4443/live".into()),
    );
    set(&mut sink, "namespace", PropValue::Str("live/cam1".into()));
    set(&mut sink, "track-name", PropValue::Str("video.m4s".into()));
    set(
        &mut sink,
        "init-track-name",
        PropValue::Str("init.mp4".into()),
    );
    set(
        &mut sink,
        "catalog-track-name",
        PropValue::Str("cat".into()),
    );
    set(&mut sink, "catalog", PropValue::Bool(false));
    set(&mut sink, "priority", PropValue::Uint(3));
    set(&mut sink, "max-request-id", PropValue::Uint(64));
    set(
        &mut sink,
        "server-certificate-hashes",
        PropValue::Str("aa".repeat(32)),
    );

    assert_eq!(
        AsyncElement::get_property(&sink, "location"),
        Some(PropValue::Str("https://relay.example:4443/live".into()))
    );
    assert_eq!(
        AsyncElement::get_property(&sink, "namespace"),
        Some(PropValue::Str("live/cam1".into()))
    );
    assert_eq!(
        AsyncElement::get_property(&sink, "track-name"),
        Some(PropValue::Str("video.m4s".into()))
    );
    assert_eq!(
        AsyncElement::get_property(&sink, "catalog"),
        Some(PropValue::Bool(false))
    );
    assert_eq!(
        AsyncElement::get_property(&sink, "priority"),
        Some(PropValue::Uint(3))
    );
    assert_eq!(
        AsyncElement::get_property(&sink, "max-request-id"),
        Some(PropValue::Uint(64))
    );

    // A wrong-typed value and an unknown name are both refused.
    assert!(
        AsyncElement::set_property(&mut sink, "priority", PropValue::Str("high".into())).is_err()
    );
    assert!(AsyncElement::set_property(&mut sink, "nope", PropValue::Uint(1)).is_err());

    // The sink accepts an fMP4 byte stream and nothing else.
    assert!(sink
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff
        })
        .is_ok());
    assert!(sink
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs
        })
        .is_err());

    // parse_launch resolves the element through the registry and reads each
    // property's kind from the same declaration it just round-tripped.
    let reg = default_registry();
    let line =
        "fakesrc ! moqtsink location=https://relay.example:4443/ namespace=live/cam priority=5";
    let err = parse_launch(&reg, line)
        .err()
        .map(|e| format!("{e}"))
        .unwrap_or_default();
    assert!(
        !err.contains("unknown element") && !err.contains("unknown property"),
        "moqtsink and its properties resolve: {err}"
    );
}

// ------------------------------------------------------------------ leg 4

/// Fragments the subscriber must receive before the comparison runs. One GOP is
/// ten fragments here, so this spans a group boundary.
const FRAGMENTS_WANTED: usize = 12;

/// `mp4mux ! moqtsink` -> `moq-relay-ietf` -> `moq-sub`: the bytes `moq-sub`
/// writes must be the init segment we published followed by an unbroken run of
/// the fragments we published, in order.
#[tokio::test]
async fn moqtsink_publishes_through_the_reference_relay_to_moq_sub() {
    let Some(sub_bin) = reference_binary("moq-sub") else {
        eprintln!(
            "SKIP: moq-sub not found. Build it with `cargo +stable build --release \
             -p moq-relay-ietf -p moq-sub` in a cloudflare/moq-rs checkout, or point \
             $MOQ_RS_BIN at its directory."
        );
        return;
    };

    let tls = TestCert::generate();
    let port = free_udp_port();
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2gtest";
    let Some(_relay) = spawn_relay(&tls, port) else {
        return;
    };

    // The relay binds asynchronously; the sink's first frame dials it, so give
    // it a moment rather than racing the handshake.
    tokio::time::sleep(Duration::from_millis(750)).await;

    let mut mux = Mp4Mux::new();
    mux.configure_pipeline(&h264_caps(64, 48))
        .expect("mp4mux caps");
    let mut sink = MoqtSink::new(&url, namespace).with_server_certificate_hashes(&tls.hash_hex);
    sink.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .expect("moqtsink caps");

    let out_path = std::env::temp_dir().join(format!("g2g-m902-{}.mp4", std::process::id()));
    let mut subscriber: Option<Reaped> = None;
    // Everything mp4mux produced, in order: the sink's objects are exactly this
    // stream cut at fragment boundaries.
    let mut published = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut index = 0u64;

    while Instant::now() < deadline {
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
        index += 1;

        // Attach the subscriber once the namespace is live at the relay.
        if index == 10 && subscriber.is_none() {
            let file = std::fs::File::create(&out_path).expect("create output");
            match Command::new(&sub_bin)
                .arg(&url)
                .arg("--name")
                .arg(namespace)
                .arg("--tls-disable-verify")
                .stdout(Stdio::from(file))
                .stderr(Stdio::null())
                .spawn()
            {
                Ok(child) => subscriber = Some(Reaped(child)),
                Err(e) => {
                    eprintln!("SKIP: could not start moq-sub: {e}");
                    return;
                }
            }
        }

        // Stop once the subscriber has written more than one whole group, so
        // the run that is compared spans a group boundary: a finished stream,
        // a new one, and a bumped group id.
        if subscriber.is_some() {
            if let Ok(received) = std::fs::read(&out_path) {
                if moof_count(&received) >= FRAGMENTS_WANTED {
                    break;
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(33)).await;
    }

    // End of stream: PUBLISH_DONE to every subscriber, then close the session.
    sink.process(PipelinePacket::Eos, &mut NullOut)
        .await
        .expect("clean end of stream");

    // Let the last write land, then stop the subscriber and read what it wrote.
    tokio::time::sleep(Duration::from_millis(500)).await;
    drop(subscriber);
    let received = std::fs::read(&out_path).unwrap_or_default();
    let _ = std::fs::remove_file(&out_path);

    assert!(
        !received.is_empty(),
        "moq-sub wrote nothing: the relay never delivered a subscription"
    );
    assert_eq!(
        sink.track_names(),
        vec![String::from("1.m4s")],
        "the moov's one video track is named the way moq-sub expects"
    );
    assert!(
        sink.objects_published() > 0,
        "the sink counted the objects it wrote"
    );

    // The init segment is the ftyp+moov prefix of what mp4mux produced.
    let boundaries = box_boundaries(&published);
    let init_len = init_len(&published);
    assert_eq!(
        &received[..init_len],
        &published[..init_len],
        "the init segment came back byte for byte"
    );

    let tail = &received[init_len..];
    assert!(
        moof_count(tail) >= FRAGMENTS_WANTED - 1,
        "the run spans more than one group, got {} fragments",
        moof_count(tail)
    );

    // The fragments are an unbroken run of the published stream, starting and
    // ending on a top-level box boundary.
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
        start >= init_len,
        "the fragments come from the fragment region, not the init segment"
    );
}
