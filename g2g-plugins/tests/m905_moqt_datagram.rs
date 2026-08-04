//! M905: MoQ Transport datagram objects, and a live proof that one group
//! spread across concurrent subgroup streams reassembles in order.
//!
//! The legs:
//!   1. the datagram wire layout: every type's bytes match what the reference
//!      implementation's own encoder (`moq-transport/src/data/datagram.rs`)
//!      produces, not just a round trip of ours.
//!   2. malformed datagrams from a peer: a truncated header, an extension block
//!      that overruns, an empty block the type forbids, a payload past the
//!      bound. Each fails the decode rather than panicking or allocating on the
//!      peer's number.
//!   3. loss policy, driven offline: a lost datagram object leaves a hole, and
//!      the end-of-group marker that closes a datagram-carried group lets the
//!      subscriber step over it instead of waiting.
//!   4. the element surface: `datagrams` and `subgroups` round-trip and resolve
//!      for `parse_launch`.
//!   5. live, `moqtsink` -> `moqtsrc` direct over a real QUIC connection with no
//!      relay in the path: datagram mode end to end, media intact across more
//!      than one group, with the oversize objects falling back to a stream.
//!   6. live loss: the same path with datagrams dropped on purpose. The
//!      subscriber must keep playing and account for what it lost.
//!   7. live multi-subgroup: a group spread across concurrent subgroup streams,
//!      merged back into (group, object) order. Direct, because the reference
//!      relay renumbers each subgroup's objects from zero and so cannot carry
//!      more than one subgroup of a group; the relay leg asserts only that the
//!      subscriber keeps playing through that.
//!
//! There is no relay leg for datagrams: `moq-relay-ietf` has no datagram code at
//! all (`moq-transport` has `send_datagram` / `recv_datagram`, the relay crate
//! references neither), so a relay-mediated datagram test cannot be run against
//! the reference stack. The direct leg below is the honest substitute: the peer
//! between the two elements copies control messages, streams and datagrams
//! byte for byte and keeps no track state, so what `moqtsrc` decodes is exactly
//! what `moqtsink` encoded.
#![cfg(feature = "moqt")]

use core::future::Future;
use core::pin::Pin;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use web_transport_quinn::quinn::rustls::pki_types::pem::PemObject;
use web_transport_quinn::quinn::rustls::pki_types::{CertificateDer, PrivateKeyDer};
use web_transport_quinn::{RecvStream, SendStream, Server, ServerBuilder, Session};

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, SourceLoop};
use g2g_core::{
    ByteStreamEncoding, Caps, Dim, G2gError, OutputSink, PropValue, PushOutcome, Rate, VideoCodec,
};

use g2g_plugins::moqt::coding::{setup_param, MoqtError, Params, Reader};
use g2g_plugins::moqt::data::{ObjectStatus, SubgroupHeader};
use g2g_plugins::moqt::datagram::{DatagramObject, DatagramType};
use g2g_plugins::moqt::message::ControlMessage;
use g2g_plugins::moqt::reassembly::Reassembler;
use g2g_plugins::moqtsink::MoqtSink;
use g2g_plugins::moqtsrc::MoqtSrc;
use g2g_plugins::mp4mux::Mp4Mux;
use g2g_plugins::registry::default_registry;

// ------------------------------------------------------------------ leg 1

fn encoded(object: &DatagramObject) -> Vec<u8> {
    let mut out = Vec::new();
    object.encode(&mut out).expect("encode");
    out
}

/// The byte vectors here were printed by the reference implementation's own
/// encoder (`moq_transport::data::Datagram::encode`), not transcribed from the
/// draft: a round trip alone cannot catch two fields swapped with each other,
/// which is exactly the mistake that breaks interop.
#[test]
fn datagram_layouts_match_the_reference_implementation() {
    // DatagramType::ObjectIdPayload, alias 12, group 10, object 1234,
    // priority 127, payload "payload".
    assert_eq!(
        encoded(&DatagramObject::media(
            12,
            10,
            1234,
            127,
            b"payload".to_vec()
        )),
        vec![0x00, 0x0c, 0x0a, 0x44, 0xd2, 0x7f, b'p', b'a', b'y', b'l', b'o', b'a', b'd']
    );

    // DatagramType::ObjectIdPayloadExt: the same, with a byte-length-prefixed
    // extension block between the priority and the payload.
    let mut ext = Params::new();
    ext.set_int(0, 42);
    assert_eq!(
        encoded(&DatagramObject {
            datagram_type: DatagramType::ObjectIdPayloadExt,
            extension_headers: ext,
            ..DatagramObject::media(12, 10, 1234, 127, b"payload".to_vec())
        }),
        vec![
            0x01, 0x0c, 0x0a, 0x44, 0xd2, 0x7f, 0x02, 0x00, 0x2a, b'p', b'a', b'y', b'l', b'o',
            b'a', b'd'
        ]
    );

    // DatagramType::ObjectIdStatus with EndOfGroup (0x03): no payload follows.
    assert_eq!(
        encoded(&DatagramObject::end_of_group(4, 7, 3, 127)),
        vec![0x20, 0x04, 0x07, 0x03, 0x7f, 0x03]
    );

    // DatagramType::PayloadEndOfGroup: the types at 0x04 and above carry no
    // object id at all, so the priority follows the group id directly.
    assert_eq!(
        encoded(&DatagramObject {
            datagram_type: DatagramType::PayloadEndOfGroup,
            ..DatagramObject::media(12, 10, 0, 127, b"payload".to_vec())
        }),
        vec![0x06, 0x0c, 0x0a, 0x7f, b'p', b'a', b'y', b'l', b'o', b'a', b'd']
    );

    // The header type table itself (`data/datagram.rs`).
    for (code, ty) in [
        (0x00, DatagramType::ObjectIdPayload),
        (0x01, DatagramType::ObjectIdPayloadExt),
        (0x02, DatagramType::ObjectIdPayloadEndOfGroup),
        (0x03, DatagramType::ObjectIdPayloadExtEndOfGroup),
        (0x04, DatagramType::Payload),
        (0x05, DatagramType::PayloadExt),
        (0x06, DatagramType::PayloadEndOfGroup),
        (0x07, DatagramType::PayloadExtEndOfGroup),
        (0x20, DatagramType::ObjectIdStatus),
        (0x21, DatagramType::ObjectIdStatusExt),
    ] {
        assert_eq!(DatagramType::from_code(code), Ok(ty));
        assert_eq!(ty.code(), code);
    }
    // 0x08 through 0x1f are not datagram types.
    assert_eq!(DatagramType::from_code(0x08), Err(MoqtError::Malformed));
    assert_eq!(DatagramType::from_code(0x22), Err(MoqtError::Malformed));
}

// ------------------------------------------------------------------ leg 2

/// A datagram is a whole message a peer chose the contents of. Everything in it
/// is bounded before use, and a short one is a violation rather than something
/// to wait for, because nothing follows it.
#[test]
fn malformed_datagram_input_fails_the_decode() {
    let full = encoded(&DatagramObject::media(
        12,
        10,
        1234,
        127,
        b"payload".to_vec(),
    ));
    for cut in 0..6 {
        assert_eq!(
            DatagramObject::decode(&full[..cut], 1 << 20),
            Err(MoqtError::Malformed),
            "a {cut}-byte datagram is truncated, and nothing more is coming"
        );
    }

    // A payload larger than the subscriber's per-object bound is refused
    // without allocating on it.
    assert_eq!(
        DatagramObject::decode(&full, 6),
        Err(MoqtError::Malformed),
        "seven payload bytes against a six byte bound"
    );
    assert!(DatagramObject::decode(&full, 7).is_ok());

    // An extension block whose declared length overruns the datagram.
    assert_eq!(
        DatagramObject::decode(&[0x01, 0x0c, 0x0a, 0x01, 0x7f, 0x40, 0xc8, 0x01], 1 << 20),
        Err(MoqtError::Malformed)
    );
    // ...one that is present but empty, which the Ext types forbid.
    assert_eq!(
        DatagramObject::decode(&[0x01, 0x0c, 0x0a, 0x01, 0x7f, 0x00], 1 << 20),
        Err(MoqtError::Malformed)
    );
    // ...and one whose pairs do not fill their declared length.
    assert_eq!(
        DatagramObject::decode(&[0x01, 0x0c, 0x0a, 0x01, 0x7f, 0x04, 0x00, 0x2a], 1 << 20),
        Err(MoqtError::Malformed)
    );

    // A reserved status value, and a non-normal status carrying extension
    // headers, are both protocol violations.
    assert_eq!(
        DatagramObject::decode(&[0x20, 0x04, 0x07, 0x03, 0x7f, 0x01], 1 << 20),
        Err(MoqtError::Malformed)
    );
    assert_eq!(
        DatagramObject::decode(
            &[0x21, 0x01, 0x01, 0x01, 0x7f, 0x02, 0x00, 0x01, 0x04],
            1 << 20
        ),
        Err(MoqtError::Malformed)
    );

    // A reserved header type.
    assert_eq!(
        DatagramObject::decode(&[0x08, 0x0c, 0x0a, 0x01, 0x7f], 1 << 20),
        Err(MoqtError::Malformed)
    );
}

// ------------------------------------------------------------------ leg 3

/// One object's worth of the receive path: the bytes a datagram carried,
/// decoded and handed to the same reassembler the subgroup streams feed.
fn deliver_datagram(r: &mut Reassembler, object: &DatagramObject) {
    let decoded = DatagramObject::decode(&encoded(object), 1 << 20).expect("decode");
    r.push(decoded.into_received());
}

/// A datagram that never arrives is normal, not an error. The group it belonged
/// to still completes (its end-of-group marker says so), so the subscriber steps
/// over the hole and keeps playing, and the loss is counted.
#[test]
fn a_lost_datagram_does_not_stall_the_subscriber() {
    let mut r = Reassembler::new(8, 1 << 20);
    let alias = 4;
    for object_id in [0u64, 1, 3] {
        // Object 2 was lost on the way.
        deliver_datagram(
            &mut r,
            &DatagramObject::media(alias, 0, object_id, 127, vec![object_id as u8; 4]),
        );
    }
    assert_eq!(
        r.drain(),
        vec![vec![0u8; 4], vec![1u8; 4]],
        "the run before the hole plays while the group could still fill it"
    );

    // The publisher closes the group: nothing can fill the hole now.
    deliver_datagram(&mut r, &DatagramObject::end_of_group(alias, 0, 4, 127));
    assert_eq!(
        r.drain(),
        vec![vec![3u8; 4]],
        "the stream steps over the hole"
    );
    assert_eq!(r.stats().objects_dropped, 1, "the lost object is counted");

    // The next group plays straight through, so the loss cost one object and
    // not the stream.
    for object_id in [0u64, 1] {
        deliver_datagram(
            &mut r,
            &DatagramObject::media(alias, 1, object_id, 127, vec![0xB0 + object_id as u8; 4]),
        );
    }
    assert_eq!(r.drain(), vec![vec![0xB0u8; 4], vec![0xB1u8; 4]]);
    assert_eq!(r.stats().objects_emitted, 5);

    // A lost end-of-group marker cannot stall it either: the buffering bound
    // moves the cursor on rather than waiting for a group that never ends.
    let mut bounded = Reassembler::new(2, 1 << 20);
    deliver_datagram(
        &mut bounded,
        &DatagramObject::media(alias, 0, 0, 127, vec![1; 4]),
    );
    deliver_datagram(
        &mut bounded,
        &DatagramObject::media(alias, 0, 2, 127, vec![2; 4]),
    );
    assert_eq!(bounded.drain(), vec![vec![1u8; 4]]);
    for group in 1..4u64 {
        deliver_datagram(
            &mut bounded,
            &DatagramObject::media(alias, group, 0, 127, vec![group as u8; 4]),
        );
        bounded.drain();
    }
    assert!(
        bounded.stats().groups_dropped > 0,
        "the group whose marker was lost was abandoned at the bound"
    );
    assert!(bounded.buffered_groups() <= 2, "buffering stayed bounded");

    // An end-of-group datagram that carries media both delivers it and closes
    // the group.
    let mut ending = Reassembler::new(8, 1 << 20);
    deliver_datagram(
        &mut ending,
        &DatagramObject::media(alias, 0, 0, 127, vec![7; 4]),
    );
    deliver_datagram(
        &mut ending,
        &DatagramObject {
            datagram_type: DatagramType::ObjectIdPayloadEndOfGroup,
            ..DatagramObject::media(alias, 0, 1, 127, vec![8; 4])
        },
    );
    deliver_datagram(
        &mut ending,
        &DatagramObject::media(alias, 1, 0, 127, vec![9; 4]),
    );
    assert_eq!(
        ending.drain(),
        vec![vec![7u8; 4], vec![8u8; 4], vec![9u8; 4]],
        "the group ended without the next one waiting on it"
    );
    assert_eq!(
        DatagramObject {
            datagram_type: DatagramType::ObjectIdPayloadEndOfGroup,
            ..DatagramObject::media(alias, 0, 1, 127, vec![8; 4])
        }
        .object_status(),
        ObjectStatus::EndOfGroup
    );
}

// ------------------------------------------------------------------ leg 4

#[test]
fn datagram_properties_round_trip_and_resolve_for_launch() {
    let mut sink = MoqtSink::new("https://127.0.0.1:4443/", "g2g");
    let names: Vec<&str> = AsyncElement::properties(&sink)
        .iter()
        .map(|p| p.name)
        .collect();
    for expected in ["datagrams", "subgroups"] {
        assert!(names.contains(&expected), "{expected} is declared");
    }
    assert_eq!(
        AsyncElement::get_property(&sink, "datagrams"),
        Some(PropValue::Bool(false)),
        "datagram delivery is off by default: it changes the delivery guarantee"
    );
    assert_eq!(
        AsyncElement::get_property(&sink, "subgroups"),
        Some(PropValue::Uint(1))
    );

    AsyncElement::set_property(&mut sink, "datagrams", PropValue::Bool(true)).expect("datagrams");
    AsyncElement::set_property(&mut sink, "subgroups", PropValue::Uint(3)).expect("subgroups");
    assert_eq!(
        AsyncElement::get_property(&sink, "datagrams"),
        Some(PropValue::Bool(true))
    );
    assert_eq!(
        AsyncElement::get_property(&sink, "subgroups"),
        Some(PropValue::Uint(3))
    );
    assert!(
        AsyncElement::set_property(&mut sink, "datagrams", PropValue::Uint(1)).is_err(),
        "a wrong-typed value is refused"
    );

    // The builders set the same state the properties do.
    let built = MoqtSink::new("https://127.0.0.1:4443/", "g2g")
        .with_datagrams(true)
        .with_subgroups(3);
    assert_eq!(
        AsyncElement::get_property(&built, "datagrams"),
        Some(PropValue::Bool(true))
    );
    assert_eq!(
        AsyncElement::get_property(&built, "subgroups"),
        Some(PropValue::Uint(3))
    );

    let reg = default_registry();
    let line = "fakesrc ! mp4mux ! moqtsink location=https://relay.example:4443/ \
                namespace=live/cam datagrams=true subgroups=2";
    let err = parse_launch(&reg, line)
        .err()
        .map(|e| format!("{e}"))
        .unwrap_or_default();
    assert!(
        !err.contains("unknown element") && !err.contains("unknown property"),
        "moqtsink's datagram properties resolve: {err}"
    );
}

// ------------------------------------------------------------- live harness

fn h264_caps(w: u32, h: u32) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

#[derive(Default)]
struct CaptureSink {
    frames: Vec<Vec<u8>>,
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

struct NullOut;
impl OutputSink for NullOut {
    fn push<'a>(
        &'a mut self,
        _packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

fn frame(bytes: Vec<u8>, pts_ns: u64, sequence: u64) -> Frame {
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
const SPS: [u8; 6] = [0x67, 0x42, 0xC0, 0x1E, 0x11, 0x22];

/// An IDR access unit with the parameter sets every tenth frame, then P slices.
/// `big_every` (0 = never) pads that many-th unit past the path MTU, so datagram
/// mode has to fall back to a stream for it.
fn access_unit(index: u64, big_every: u64) -> Vec<u8> {
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

struct TestCert {
    cert_path: PathBuf,
    key_path: PathBuf,
    hash_hex: String,
}

impl TestCert {
    fn generate() -> Self {
        let issued = rcgen::generate_simple_self_signed(vec![
            "localhost".to_string(),
            "127.0.0.1".to_string(),
        ])
        .expect("self-signed certificate");
        let dir = std::env::temp_dir();
        let unique = format!("{}-{:?}", std::process::id(), std::thread::current().id());
        let cert_path = dir.join(format!("g2g-m905-{unique}.crt"));
        let key_path = dir.join(format!("g2g-m905-{unique}.key"));
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

/// Kills its child on drop, so a failing assertion never leaves a relay behind.
struct Reaped(Child);

impl Drop for Reaped {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Where the reference peers live: `$MOQ_RS_BIN`, else the release build of a
/// `moq-rs` checkout beside this one, else `PATH`.
fn reference_binary(name: &str) -> Option<PathBuf> {
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

fn relay_missing_reason() -> String {
    String::from(
        "SKIP: moq-relay-ietf not found. Build it with `cargo +stable build --release \
         -p moq-relay-ietf` in a cloudflare/moq-rs checkout, or point $MOQ_RS_BIN at \
         its directory.",
    )
}

fn free_udp_port() -> u16 {
    let probe = std::net::UdpSocket::bind("127.0.0.1:0").expect("probe bind");
    probe.local_addr().expect("probe addr").port()
}

/// Start `moq-relay-ietf` on `port`. `None` (with a printed reason) when the
/// binary is absent.
fn spawn_relay(tls: &TestCert, port: u16) -> Option<Reaped> {
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

/// The top-level box offsets of an fMP4 stream, so a comparison can insist a
/// match starts and ends on a box boundary rather than anywhere.
fn box_boundaries(stream: &[u8]) -> Vec<usize> {
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

fn moof_count(stream: &[u8]) -> usize {
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
fn group_starts(stream: &[u8]) -> usize {
    stream.windows(SPS.len()).filter(|w| *w == SPS).count()
}

/// The length of the `ftyp`+`moov` prefix.
fn init_len(stream: &[u8]) -> usize {
    box_boundaries(stream)
        .into_iter()
        .find(|at| {
            at + 8 <= stream.len()
                && &stream[at + 4..at + 8] != b"ftyp"
                && &stream[at + 4..at + 8] != b"moov"
        })
        .expect("a fragment follows the init segment")
}

/// Frames a subscription is asked for: one init segment plus enough fragments to
/// span more than one group (a GOP is ten fragments here).
const FRAMES_WANTED: u64 = 14;

/// Assert `received` is the init segment followed by an unbroken run of
/// `published`'s fragments.
fn assert_matches_published(received: &[Vec<u8>], published: &[u8], min_fragments: usize) {
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
fn assert_ordered_fragments(received: &[Vec<u8>], published: &[u8]) {
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

/// What one live publish / subscribe run produced.
struct RoundTrip {
    published: Vec<u8>,
    frames: Vec<Vec<u8>>,
    emitted: u64,
    sink: MoqtSink,
}

/// Drive `mp4mux ! moqtsink` and `moqtsrc` against each other over whatever peer
/// `sink` and `src` were pointed at, publishing until the subscriber has what it
/// asked for. The sink applies control messages when a frame arrives, so it has
/// to keep publishing while the subscriber runs.
async fn round_trip(
    mut sink: MoqtSink,
    src: &mut MoqtSrc,
    big_every: u64,
    subscribe_after: Duration,
) -> RoundTrip {
    let mut mux = Mp4Mux::new();
    mux.configure_pipeline(&h264_caps(64, 48))
        .expect("mp4mux caps");
    sink.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .expect("moqtsink caps");
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .expect("moqtsrc caps");

    let done = std::cell::Cell::new(false);
    let publish = async {
        let mut published = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(40);
        let mut index = 0u64;
        while !done.get() && Instant::now() < deadline {
            let mut captured = CaptureSink::default();
            mux.process(
                PipelinePacket::DataFrame(frame(
                    access_unit(index, big_every),
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
            index += 1;
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        sink.process(PipelinePacket::Eos, &mut NullOut)
            .await
            .expect("clean end of stream");
        (published, sink)
    };

    let mut captured = CaptureSink::default();
    let subscribe = async {
        // Let the publisher reach the peer before subscribing.
        tokio::time::sleep(subscribe_after).await;
        let emitted = src.run(&mut captured).await;
        done.set(true);
        emitted
    };

    let ((published, sink), emitted) = tokio::join!(publish, subscribe);
    RoundTrip {
        published,
        frames: captured.frames,
        emitted: emitted.expect("subscribe and play"),
        sink,
    }
}

// ------------------------------------------------------------ the pipe peer

/// The (group, subgroup) of every subgroup stream the pipe carried.
type SubgroupLog = Arc<Mutex<Vec<(u64, u64)>>>;

/// A WebTransport server that carries one MoQT publisher session to one
/// subscriber session and nothing else: it answers each side's CLIENT_SETUP,
/// then copies control messages, subgroup streams and datagrams across byte for
/// byte. It keeps no track state and makes no routing decision, so what
/// `moqtsrc` decodes is exactly what `moqtsink` encoded. Not a relay: it exists
/// because a relay-mediated datagram test is not available (`moq-relay-ietf` has
/// no datagram code).
struct Pipe {
    port: u16,
    datagrams_forwarded: Arc<AtomicU64>,
    datagrams_dropped: Arc<AtomicU64>,
    /// The (group, subgroup) of every subgroup stream that crossed, so a test
    /// can insist one group really arrived on more than one stream.
    subgroups: SubgroupLog,
    task: tokio::task::JoinHandle<()>,
}

impl Pipe {
    fn url(&self) -> String {
        format!("https://127.0.0.1:{}/", self.port)
    }

    /// The largest number of concurrent subgroup streams any one group used.
    fn widest_group(&self) -> usize {
        let seen = self.subgroups.lock().expect("subgroup log");
        seen.iter()
            .map(|(group, _)| {
                seen.iter()
                    .filter(|(g, _)| g == group)
                    .map(|(_, sub)| *sub)
                    .collect::<std::collections::BTreeSet<u64>>()
                    .len()
            })
            .max()
            .unwrap_or(0)
    }
}

impl Drop for Pipe {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// One side of the pipe: its session, the control stream, and whatever it had
/// already sent past the SETUP.
struct Side {
    session: Session,
    tx: SendStream,
    rx: RecvStream,
    pending: Vec<u8>,
    publisher: bool,
}

/// Start the pipe on an ephemeral port. `drop_every` (0 = none) throws away that
/// many-th datagram, which is what an unreliable path does on its own.
fn start_pipe(tls: &TestCert, drop_every: u64) -> Pipe {
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
    let forwarded = Arc::new(AtomicU64::new(0));
    let dropped = Arc::new(AtomicU64::new(0));
    let subgroups = Arc::new(Mutex::new(Vec::new()));
    let task = tokio::spawn(run_pipe(
        server,
        drop_every,
        forwarded.clone(),
        dropped.clone(),
        subgroups.clone(),
    ));
    Pipe {
        port,
        datagrams_forwarded: forwarded,
        datagrams_dropped: dropped,
        subgroups,
        task,
    }
}

async fn run_pipe(
    mut server: Server,
    drop_every: u64,
    forwarded: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
    subgroups: SubgroupLog,
) {
    let Some(first) = accept_side(&mut server).await else {
        return;
    };
    let Some(second) = accept_side(&mut server).await else {
        return;
    };
    // The publisher opens with PUBLISH_NAMESPACE and the subscriber with
    // SUBSCRIBE, so the roles are read off the wire rather than assumed from the
    // order the two sessions happened to arrive in.
    let (publisher, subscriber) = if first.publisher {
        (first, second)
    } else {
        (second, first)
    };

    tokio::spawn(pipe_control(publisher.rx, subscriber.tx, publisher.pending));
    tokio::spawn(pipe_control(
        subscriber.rx,
        publisher.tx,
        subscriber.pending,
    ));

    let from = publisher.session.clone();
    let to = subscriber.session.clone();
    tokio::spawn(async move {
        while let Ok(rx) = from.accept_uni().await {
            let Ok(tx) = to.open_uni().await else {
                return;
            };
            tokio::spawn(copy_stream(rx, tx, Some(subgroups.clone())));
        }
    });

    let mut seen = 0u64;
    while let Ok(bytes) = publisher.session.read_datagram().await {
        seen += 1;
        if drop_every != 0 && seen % drop_every == 0 {
            dropped.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        if subscriber.session.send_datagram(bytes).is_ok() {
            forwarded.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Accept one session, answer its CLIENT_SETUP, and read far enough into its
/// control stream to tell which side it is.
async fn accept_side(server: &mut Server) -> Option<Side> {
    let request = server.accept().await?;
    let session = request.ok().await.ok()?;
    let (mut tx, mut rx) = session.accept_bi().await.ok()?;
    let mut buf = Vec::new();
    match read_control(&mut rx, &mut buf).await? {
        ControlMessage::ClientSetup { .. } => {}
        other => panic!("expected CLIENT_SETUP, got {}", other.name()),
    }
    let mut params = Params::new();
    params.set_int(setup_param::MAX_REQUEST_ID, 100);
    let mut out = Vec::new();
    ControlMessage::ServerSetup { params }
        .encode(&mut out)
        .expect("encode SERVER_SETUP");
    tx.write_all(&out).await.ok()?;
    let publisher = matches!(
        peek_control(&mut rx, &mut buf).await?,
        ControlMessage::PublishNamespace { .. }
    );
    Some(Side {
        session,
        tx,
        rx,
        pending: buf,
        publisher,
    })
}

/// Read until one whole control message is buffered, and consume it.
async fn read_control(rx: &mut RecvStream, buf: &mut Vec<u8>) -> Option<ControlMessage> {
    loop {
        match ControlMessage::decode(buf) {
            Ok((msg, used)) => {
                buf.drain(..used);
                return Some(msg);
            }
            Err(MoqtError::Incomplete) => fill(rx, buf).await?,
            Err(_) => return None,
        }
    }
}

/// The same, leaving the message in the buffer so it is forwarded like the rest.
async fn peek_control(rx: &mut RecvStream, buf: &mut Vec<u8>) -> Option<ControlMessage> {
    loop {
        match ControlMessage::decode(buf) {
            Ok((msg, _)) => return Some(msg),
            Err(MoqtError::Incomplete) => fill(rx, buf).await?,
            Err(_) => return None,
        }
    }
}

async fn fill(rx: &mut RecvStream, buf: &mut Vec<u8>) -> Option<()> {
    let mut chunk = vec![0u8; 8192];
    match rx.read(&mut chunk).await {
        Ok(Some(n)) if n > 0 => {
            buf.extend_from_slice(&chunk[..n]);
            Some(())
        }
        _ => None,
    }
}

async fn pipe_control(rx: RecvStream, mut tx: SendStream, pending: Vec<u8>) {
    if !pending.is_empty() && tx.write_all(&pending).await.is_err() {
        return;
    }
    copy_stream(rx, tx, None).await;
}

/// Copy one stream across verbatim. With a log, the subgroup header at its head
/// is decoded on the way past and recorded, so a test can see which (group,
/// subgroup) each stream carried; the bytes themselves are untouched.
async fn copy_stream(mut rx: RecvStream, mut tx: SendStream, subgroups: Option<SubgroupLog>) {
    let mut chunk = vec![0u8; 16 * 1024];
    let mut head = Vec::new();
    let mut log = subgroups;
    while let Ok(Some(n)) = rx.read(&mut chunk).await {
        if n == 0 || tx.write_all(&chunk[..n]).await.is_err() {
            break;
        }
        let Some(subgroups) = log.as_ref() else {
            continue;
        };
        head.extend_from_slice(&chunk[..n]);
        if let Ok(header) = SubgroupHeader::decode(&mut Reader::new(&head)) {
            subgroups
                .lock()
                .expect("subgroup log")
                .push((header.group_id, header.subgroup_id.unwrap_or(0)));
            log = None;
        } else if head.len() > 64 {
            log = None;
        }
    }
    let _ = tx.finish();
}

// ------------------------------------------------------------------ leg 5

/// `mp4mux ! moqtsink datagrams=true` -> `moqtsrc`, direct over a real QUIC
/// connection with no relay: the media comes back intact across more than one
/// group, and the objects that will not fit the path MTU ride a stream instead.
#[tokio::test]
async fn moqtsink_datagrams_play_through_moqtsrc_directly() {
    let tls = TestCert::generate();
    let pipe = start_pipe(&tls, 0);
    let namespace = "g2gdatagram";

    let sink = MoqtSink::new(pipe.url(), namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_datagrams(true);
    let mut src = MoqtSrc::new(pipe.url(), namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_num_buffers(FRAMES_WANTED);

    // Every seventh access unit is padded past the MTU, so both carriages run.
    let run = round_trip(sink, &mut src, 7, Duration::from_millis(1500)).await;

    assert_eq!(run.emitted, FRAMES_WANTED);
    assert_eq!(src.selected_track(), "1.m4s");
    assert!(
        run.sink.datagram_objects() > 0,
        "the publisher sent objects as datagrams"
    );
    assert!(
        run.sink.datagram_fallbacks() > 0,
        "the objects past the path MTU fell back to a stream"
    );
    assert!(
        pipe.datagrams_forwarded.load(Ordering::Relaxed) > 0,
        "datagrams actually crossed the connection"
    );
    assert_matches_published(&run.frames, &run.published, FRAMES_WANTED as usize - 1);
    let tail: Vec<u8> = run.frames[1..].concat();
    assert!(
        group_starts(&tail) > 1,
        "the byte-identical run spans more than one group, got {}",
        group_starts(&tail)
    );
}

// ------------------------------------------------------------------ leg 6

/// The same path with every third datagram thrown away: a lost object is normal
/// on this carriage, so the subscriber must keep playing and count what it lost
/// rather than waiting for it.
#[tokio::test]
async fn a_dropped_datagram_does_not_stall_moqtsrc() {
    let tls = TestCert::generate();
    let pipe = start_pipe(&tls, 3);
    let namespace = "g2gdatagramloss";

    let sink = MoqtSink::new(pipe.url(), namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_datagrams(true);
    let mut src = MoqtSrc::new(pipe.url(), namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_num_buffers(FRAMES_WANTED);

    let run = round_trip(sink, &mut src, 0, Duration::from_millis(1500)).await;

    assert_eq!(
        run.emitted, FRAMES_WANTED,
        "the subscriber kept playing through the loss"
    );
    assert!(
        pipe.datagrams_dropped.load(Ordering::Relaxed) > 0,
        "the peer really dropped datagrams"
    );
    assert!(
        src.objects_dropped() > 0,
        "the reassembly stats account for the lost objects"
    );
    // What did arrive is real media in publish order, holes and all.
    assert_eq!(run.frames.len(), FRAMES_WANTED as usize);
    assert_ordered_fragments(&run.frames, &run.published);
}

// ------------------------------------------------------------------ leg 7

/// One group spread across three concurrent subgroup streams, merged back into
/// (group, object) order live. Neither `moq-pub` nor `moqtsink` used to open
/// more than one subgroup stream per group, so this path had only ever run in
/// unit tests.
#[tokio::test]
async fn moqtsrc_merges_concurrent_subgroup_streams_of_one_group() {
    let tls = TestCert::generate();
    let pipe = start_pipe(&tls, 0);
    let namespace = "g2gsubgroups";

    let sink = MoqtSink::new(pipe.url(), namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_subgroups(3);
    let mut src = MoqtSrc::new(pipe.url(), namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_num_buffers(FRAMES_WANTED);

    let run = round_trip(sink, &mut src, 0, Duration::from_millis(1500)).await;

    assert_eq!(run.emitted, FRAMES_WANTED);
    assert_eq!(
        pipe.datagrams_forwarded.load(Ordering::Relaxed),
        0,
        "this leg is streams only"
    );
    assert!(
        pipe.widest_group() > 1,
        "one group arrived on more than one concurrent subgroup stream, got {}",
        pipe.widest_group()
    );
    assert_matches_published(&run.frames, &run.published, FRAMES_WANTED as usize - 1);
    let tail: Vec<u8> = run.frames[1..].concat();
    assert!(
        group_starts(&tail) > 1,
        "the run spans more than one group, so more than one group was merged"
    );
}

/// The same publisher through `moq-relay-ietf`. The reference relay cannot carry
/// a group on more than one subgroup stream: it renumbers every subgroup's
/// objects from zero (`session/subscriber.rs`, "TODO SLG - object_id_delta and
/// object status are still being ignored", feeding `serve::SubgroupWriter`
/// whose ids run from zero), so the three subgroups collide on one set of ids
/// and the duplicates are dropped. That is the reference's limit, not something
/// to assert as correct: what this leg asserts is that `moqtsrc` keeps playing
/// through it, in publish order and a whole fragment at a time, instead of
/// stalling on the ids that never arrive. Skipped with a printed reason when the
/// binary is absent.
#[tokio::test]
async fn moqtsrc_keeps_playing_when_the_reference_relay_drops_subgroups() {
    let tls = TestCert::generate();
    let port = free_udp_port();
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2gsubgrouprelay";
    let Some(_relay) = spawn_relay(&tls, port) else {
        return;
    };
    // The relay binds asynchronously; give it a moment rather than racing it.
    tokio::time::sleep(Duration::from_millis(750)).await;

    let sink = MoqtSink::new(&url, namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_subgroups(3);
    let mut src = MoqtSrc::new(&url, namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_num_buffers(FRAMES_WANTED);

    let run = round_trip(sink, &mut src, 0, Duration::from_millis(1500)).await;

    assert_eq!(run.emitted, FRAMES_WANTED, "the subscriber never stalled");
    assert_eq!(src.selected_track(), "1.m4s");
    assert_ordered_fragments(&run.frames, &run.published);
    let tail: Vec<u8> = run.frames[1..].concat();
    assert!(
        group_starts(&tail) > 1,
        "what did arrive spans more than one group"
    );
}

// ------------------------------------------------------------------ leg 8

/// The skip path itself: a binary that is not there is reported, not silently
/// passed over.
#[test]
fn a_missing_reference_binary_prints_its_skip_reason() {
    assert!(
        reference_binary("moq-relay-ietf-that-does-not-exist").is_none(),
        "a binary that is not installed is not found"
    );
    let reason = relay_missing_reason();
    eprintln!("{reason}");
    assert!(
        reason.starts_with("SKIP: "),
        "the reason names itself a skip"
    );
    assert!(
        reason.contains("moq-relay-ietf") && reason.contains("MOQ_RS_BIN"),
        "the reason says what is missing and how to supply it"
    );
}
