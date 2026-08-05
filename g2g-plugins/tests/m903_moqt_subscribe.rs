//! M903: the MoQ Transport draft-16 subscriber, `moqtsrc`, on the M902 session
//! and codec.
//!
//! The legs:
//!   1. reassembly: subgroups are separate QUIC streams, so a track's objects
//!      arrive concurrently and out of order. Ordering by (group, object) is
//!      driven directly here (no network): out-of-order groups, interleaved
//!      subgroup streams, a duplicate, a group that never completes, and the
//!      bound holding instead of the buffer growing.
//!   2. malformed data-plane input from the relay: a bad stream header type, a
//!      truncated object header, an extension block that overruns.
//!   3. the element surface: properties round-trip and `moqtsrc` resolves for
//!      `parse_launch`.
//!   4. interop, both directions: `moq-pub` publishing through a locally spawned
//!      `moq-relay-ietf` into `moqtsrc`, and the g2g round trip
//!      `mp4mux ! moqtsink` -> relay -> `moqtsrc`, byte-identical across more
//!      than one group. Skipped with a printed reason when the reference
//!      binaries are absent.
#![cfg(feature = "moqt")]

use core::future::Future;
use core::pin::Pin;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use web_transport_quinn::quinn::rustls::pki_types::CertificateDer;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, SourceLoop};
use g2g_core::{
    ByteStreamEncoding, Caps, Dim, G2gError, OutputSink, PropValue, PushOutcome, Rate, VideoCodec,
};

use g2g_plugins::moqt::catalog;
use g2g_plugins::moqt::coding::MoqtError;
use g2g_plugins::moqt::data::{
    ObjectStatus, StreamHeaderType, SubgroupHeader, SubgroupObjectHeader,
};
use g2g_plugins::moqt::reassembly::{
    Reassembler, ReceivedObject, StreamItem, SubgroupStreamDecoder,
};
use g2g_plugins::moqtsink::MoqtSink;
use g2g_plugins::moqtsrc::MoqtSrc;
use g2g_plugins::mp4mux::Mp4Mux;
use g2g_plugins::registry::default_registry;

// ------------------------------------------------------------------ leg 1

fn object(group_id: u64, object_id: u64, payload: &[u8]) -> ReceivedObject {
    ReceivedObject {
        group_id,
        object_id,
        status: ObjectStatus::Normal,
        payload: payload.to_vec(),
    }
}

fn subgroup_header(group_id: u64) -> SubgroupHeader {
    SubgroupHeader {
        header_type: StreamHeaderType::SubgroupIdExt,
        track_alias: 4,
        group_id,
        subgroup_id: Some(0),
        publisher_priority: 127,
    }
}

/// The objects of a subgroup stream, without its header: consecutive ids, so
/// every delta is zero.
fn encoded_objects(payloads: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for payload in payloads {
        SubgroupObjectHeader::normal(0, payload.len())
            .encode(StreamHeaderType::SubgroupIdExt, &mut out)
            .expect("object header");
        out.extend_from_slice(payload);
    }
    out
}

/// One whole subgroup stream on the wire: the header, then each object.
fn encoded_stream(group_id: u64, payloads: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    subgroup_header(group_id).encode(&mut out).expect("header");
    out.extend_from_slice(&encoded_objects(payloads));
    out
}

/// One live subgroup stream, decoded into the reassembler exactly the way the
/// session's per-stream reader task does it.
struct WireStream {
    decoder: SubgroupStreamDecoder,
    group: Option<u64>,
}

impl WireStream {
    fn open(r: &mut Reassembler, bytes: &[u8]) -> Self {
        let mut stream = Self {
            decoder: SubgroupStreamDecoder::new(1 << 20),
            group: None,
        };
        stream.push(r, bytes);
        stream
    }

    fn push(&mut self, r: &mut Reassembler, bytes: &[u8]) {
        self.decoder.push(bytes).expect("push");
        while let Some(item) = self.decoder.next_item().expect("decode") {
            match item {
                StreamItem::Header(header) => {
                    self.group = Some(header.group_id);
                    r.stream_opened(header.group_id);
                }
                StreamItem::Object(object) => r.push(object),
            }
        }
    }

    fn close(self, r: &mut Reassembler) {
        if let Some(group) = self.group {
            r.stream_closed(group);
        }
    }
}

/// A whole stream delivered at once.
fn feed(r: &mut Reassembler, bytes: &[u8]) {
    WireStream::open(r, bytes).close(r);
}

/// The whole path a real object takes: encoded on the wire, decoded off one
/// stream, reordered against the others.
#[test]
fn decoded_streams_reassemble_into_group_and_object_order() {
    let mut r = Reassembler::new(8, 1 << 20);
    // Group 0's stream is still arriving when group 1's lands whole, which is
    // what concurrent QUIC streams do.
    let mut first = WireStream::open(&mut r, &encoded_stream(0, &[b"a0"]));
    feed(&mut r, &encoded_stream(1, &[b"b0", b"b1"]));
    assert_eq!(
        r.drain(),
        vec![b"a0".to_vec()],
        "group 1 waits behind the group still open"
    );

    first.push(&mut r, &encoded_objects(&[b"a1"]));
    first.close(&mut r);
    assert_eq!(
        r.drain(),
        vec![b"a1".to_vec(), b"b0".to_vec(), b"b1".to_vec()]
    );
    assert_eq!(r.stats().objects_emitted, 4);
    assert_eq!(r.stats().objects_dropped, 0);
}

#[test]
fn interleaved_subgroup_streams_and_duplicates_are_handled() {
    let mut r = Reassembler::new(8, 1 << 20);
    // Two streams carry one group, and their objects interleave.
    r.stream_opened(0);
    r.stream_opened(0);
    r.push(object(0, 1, b"o1"));
    r.push(object(0, 3, b"o3"));
    assert!(r.drain().is_empty(), "object 0 has not arrived");
    r.push(object(0, 0, b"o0"));
    r.push(object(0, 2, b"o2"));
    // A relay that duplicates an object must not duplicate the media.
    r.push(object(0, 2, b"o2 again"));
    assert_eq!(
        r.drain(),
        vec![
            b"o0".to_vec(),
            b"o1".to_vec(),
            b"o2".to_vec(),
            b"o3".to_vec()
        ]
    );
    // ...and a copy that arrives after the cursor passed it is dropped too.
    r.push(object(0, 1, b"o1 again"));
    assert!(r.drain().is_empty());
    assert_eq!(r.stats().objects_dropped, 2);
}

#[test]
fn a_group_that_never_completes_is_bounded_not_buffered_forever() {
    let mut r = Reassembler::new(2, 4096);
    // Group 0 stalls with a hole at object 1 and a stream that never ends.
    r.stream_opened(0);
    r.push(object(0, 0, b"a0"));
    r.push(object(0, 2, b"a2"));
    assert_eq!(r.drain(), vec![b"a0".to_vec()]);

    // The publisher keeps going. Buffering must stay under both bounds and the
    // stream must continue at the next group boundary.
    let payload = vec![0xAAu8; 1024];
    let mut emitted = 0usize;
    for group in 1..20u64 {
        feed(&mut r, &encoded_stream(group, &[&payload, &payload]));
        emitted += r.drain().len();
        assert!(r.buffered_groups() <= 2, "group bound held");
        assert!(r.buffered_bytes() <= 4096, "byte bound held");
    }
    assert_eq!(emitted, 38, "every later group played whole");
    assert_eq!(r.stats().groups_dropped, 1, "only the stalled group went");

    // Late objects for the abandoned group are refused rather than reordered
    // backwards into the stream.
    r.push(object(0, 1, b"a1"));
    assert!(r.drain().is_empty());
}

#[test]
fn joining_mid_group_starts_at_the_next_group() {
    let mut r = Reassembler::new(8, 1 << 20);
    // The relay drops us into the middle of a group: its stream starts at
    // object 3, so there is no complete GOP to play.
    let mut mid = Vec::new();
    let header = subgroup_header(4);
    header.encode(&mut mid).expect("header");
    SubgroupObjectHeader::normal(3, 7)
        .encode(header.header_type, &mut mid)
        .expect("object");
    mid.extend_from_slice(b"partial");
    feed(&mut r, &mid);
    assert!(r.drain().is_empty(), "a partial group is not played");

    feed(&mut r, &encoded_stream(5, &[b"c0", b"c1"]));
    assert_eq!(r.drain(), vec![b"c0".to_vec(), b"c1".to_vec()]);
    assert_eq!(r.stats().groups_dropped, 1);
}

// ------------------------------------------------------------------ leg 2

/// Everything the relay controls on a data stream is bounded before use: a
/// malformed stream fails its decode instead of panicking or allocating on the
/// number the relay sent.
#[test]
fn malformed_data_plane_input_fails_the_decode() {
    // A stream header type that is not a subgroup at all.
    let mut decoder = SubgroupStreamDecoder::new(1 << 20);
    decoder.push(&[0x16, 0x04, 0x07, 0x7f]).expect("push");
    assert_eq!(decoder.next_item(), Err(MoqtError::Malformed));

    // A stream header cut at every point needs more bytes, never a panic.
    let full = encoded_stream(7, &[]);
    for cut in 0..full.len() {
        let mut decoder = SubgroupStreamDecoder::new(1 << 20);
        decoder.push(&full[..cut]).expect("push");
        assert_eq!(
            decoder.next_item(),
            Ok(None),
            "a {cut}-byte prefix is partial"
        );
    }

    // A truncated object header, and a payload that has not all arrived.
    let stream = encoded_stream(7, &[b"payload"]);
    for cut in full.len()..stream.len() {
        let mut decoder = SubgroupStreamDecoder::new(1 << 20);
        decoder.push(&stream[..cut]).expect("push");
        assert_eq!(
            decoder.next_item(),
            Ok(Some(StreamItem::Header(subgroup_header(7))))
        );
        assert_eq!(
            decoder.next_item(),
            Ok(None),
            "a {cut}-byte prefix is partial"
        );
    }

    // An extension block whose declared length is past the 64 KiB the codec
    // allows is a protocol violation, not a wait for more bytes.
    let mut decoder = SubgroupStreamDecoder::new(1 << 20);
    let mut bytes = Vec::new();
    subgroup_header(1).encode(&mut bytes).expect("header");
    bytes.extend_from_slice(&[0x00, 0x80, 0x01, 0x11, 0x70]);
    decoder.push(&bytes).expect("push");
    assert_eq!(
        decoder.next_item(),
        Ok(Some(StreamItem::Header(subgroup_header(1))))
    );
    assert_eq!(decoder.next_item(), Err(MoqtError::Malformed));

    // An object length past the per-object bound is refused without allocating
    // on it, and a stream that never completes an object stops growing.
    let mut decoder = SubgroupStreamDecoder::new(16);
    let mut bytes = Vec::new();
    subgroup_header(1).encode(&mut bytes).expect("header");
    decoder.push(&bytes).expect("push");
    assert_eq!(
        decoder.next_item(),
        Ok(Some(StreamItem::Header(subgroup_header(1))))
    );
    let mut oversized = decoder;
    oversized
        .push(&[0x00, 0x00, 0x80, 0x10, 0x00, 0x00])
        .expect("push");
    assert_eq!(oversized.next_item(), Err(MoqtError::Malformed));

    let mut decoder = SubgroupStreamDecoder::new(16);
    decoder.push(&bytes).expect("push");
    assert_eq!(
        decoder.next_item(),
        Ok(Some(StreamItem::Header(subgroup_header(1))))
    );
    assert_eq!(
        decoder.push(&vec![0xffu8; 200 * 1024]),
        Err(MoqtError::Malformed)
    );

    // A catalog from a hostile relay yields no tracks rather than a panic.
    assert!(catalog::parse(b"{\"tracks\":[{\"name\":\"a").is_empty());
    assert!(catalog::parse(&vec![b'{'; catalog::MAX_CATALOG_BYTES + 1]).is_empty());
}

// ------------------------------------------------------------------ leg 3

#[test]
fn properties_round_trip_and_moqtsrc_resolves_for_launch() {
    let mut src = MoqtSrc::new("https://127.0.0.1:4443/", "g2g");

    let names: Vec<&str> = SourceLoop::properties(&src)
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
        "max-request-id",
        "max-groups",
        "max-buffer-bytes",
        "max-object-size",
        "num-buffers",
        "timeout",
        "server-certificate-hashes",
    ] {
        assert!(names.contains(&expected), "{expected} is declared");
    }

    let set = |src: &mut MoqtSrc, name: &str, value: PropValue| {
        SourceLoop::set_property(src, name, value).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    };
    set(
        &mut src,
        "location",
        PropValue::Str("https://relay.example:4443/live".into()),
    );
    set(&mut src, "namespace", PropValue::Str("live/cam1".into()));
    set(&mut src, "track-name", PropValue::Str("video.m4s".into()));
    set(
        &mut src,
        "init-track-name",
        PropValue::Str("init.mp4".into()),
    );
    set(&mut src, "catalog-track-name", PropValue::Str("cat".into()));
    set(&mut src, "catalog", PropValue::Bool(false));
    set(&mut src, "max-request-id", PropValue::Uint(64));
    set(&mut src, "max-groups", PropValue::Uint(4));
    set(&mut src, "max-buffer-bytes", PropValue::Uint(1 << 20));
    set(&mut src, "max-object-size", PropValue::Uint(1 << 19));
    set(&mut src, "num-buffers", PropValue::Uint(12));
    set(&mut src, "timeout", PropValue::Uint(2500));
    set(
        &mut src,
        "server-certificate-hashes",
        PropValue::Str("aa".repeat(32)),
    );

    for (name, expected) in [
        (
            "location",
            PropValue::Str("https://relay.example:4443/live".into()),
        ),
        ("namespace", PropValue::Str("live/cam1".into())),
        ("track-name", PropValue::Str("video.m4s".into())),
        ("init-track-name", PropValue::Str("init.mp4".into())),
        ("catalog-track-name", PropValue::Str("cat".into())),
        ("catalog", PropValue::Bool(false)),
        ("max-request-id", PropValue::Uint(64)),
        ("max-groups", PropValue::Uint(4)),
        ("max-buffer-bytes", PropValue::Uint(1 << 20)),
        ("max-object-size", PropValue::Uint(1 << 19)),
        ("num-buffers", PropValue::Uint(12)),
        ("timeout", PropValue::Uint(2500)),
    ] {
        assert_eq!(
            SourceLoop::get_property(&src, name),
            Some(expected),
            "{name} reads back"
        );
    }

    // A wrong-typed value and an unknown name are both refused.
    assert!(
        SourceLoop::set_property(&mut src, "max-groups", PropValue::Str("many".into())).is_err()
    );
    assert!(SourceLoop::set_property(&mut src, "nope", PropValue::Uint(1)).is_err());

    // The source produces an fMP4 byte stream and nothing else.
    assert!(src
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff
        })
        .is_ok());
    assert!(src
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs
        })
        .is_err());

    // parse_launch resolves the element through the registry and reads each
    // property's kind from the same declaration it just round-tripped.
    let reg = default_registry();
    let line = "moqtsrc location=https://relay.example:4443/ namespace=live/cam num-buffers=4 \
                max-groups=3 ! fmp4demux ! fakesink";
    let err = parse_launch(&reg, line)
        .err()
        .map(|e| format!("{e}"))
        .unwrap_or_default();
    assert!(
        !err.contains("unknown element") && !err.contains("unknown property"),
        "moqtsrc and its properties resolve: {err}"
    );
}

// ------------------------------------------------------------------ leg 4

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

/// An IDR access unit carrying the parameter sets every tenth frame, then P
/// slices, so the muxer marks a sync sample once per GOP and the publisher
/// starts a new group there.
fn access_unit(index: u64) -> Vec<u8> {
    let sps = SPS;
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    if index % 10 == 0 {
        [
            &[0, 0, 0, 1][..],
            &sps,
            &[0, 0, 0, 1],
            &pps,
            &[0, 0, 0, 1],
            &[0x65, index as u8, 0xAA],
        ]
        .concat()
    } else {
        [&[0, 0, 0, 1][..], &[0x41, index as u8, 0xBB]].concat()
    }
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
        let cert_path = dir.join(format!("g2g-m903-{unique}.crt"));
        let key_path = dir.join(format!("g2g-m903-{unique}.key"));
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

/// Kills its child on drop, so a failing assertion never leaves a relay or a
/// publisher behind.
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

fn free_udp_port() -> u16 {
    let probe = std::net::UdpSocket::bind("127.0.0.1:0").expect("probe bind");
    probe.local_addr().expect("probe addr").port()
}

/// Start `moq-relay-ietf` on an ephemeral port with a freshly issued
/// certificate. `None` (with a printed reason) when the binary is absent.
fn spawn_relay(tls: &TestCert, port: u16) -> Option<Reaped> {
    let relay_bin = match reference_binary("moq-relay-ietf") {
        Some(bin) => bin,
        None => {
            eprintln!(
                "SKIP: moq-relay-ietf not found. Build it with `cargo +stable build \
                 --release -p moq-relay-ietf -p moq-pub` in a cloudflare/moq-rs \
                 checkout, or point $MOQ_RS_BIN at its directory."
            );
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

/// Everything `mp4mux` produces for `count` access units, as one byte stream.
async fn muxed_stream(count: u64) -> Vec<u8> {
    let mut mux = Mp4Mux::new();
    mux.configure_pipeline(&h264_caps(64, 48))
        .expect("mp4mux caps");
    let mut captured = CaptureSink::default();
    for index in 0..count {
        mux.process(
            PipelinePacket::DataFrame(frame(access_unit(index), index * 33_333_333, index)),
            &mut captured,
        )
        .await
        .expect("mux access unit");
    }
    captured.frames.concat()
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

/// Keyframe fragments in a stream, counted by the SPS each IDR carries. The
/// publisher starts a group at each one, so this is the number of groups the
/// bytes cover.
fn group_starts(stream: &[u8]) -> usize {
    stream.windows(SPS.len()).filter(|w| *w == SPS).count()
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

/// The length of the `ftyp`+`moov` prefix: the init segment a publisher makes
/// out of it.
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

/// Frames a subscription is asked for: one init segment plus enough fragments
/// to span more than one group (a GOP is ten fragments here).
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

/// `moq-pub` -> `moq-relay-ietf` -> `moqtsrc`: the reference publisher's
/// broadcast plays through our subscriber, and the fMP4 that comes out is the
/// fMP4 that went in.
#[tokio::test]
async fn moq_pub_publishes_through_the_reference_relay_to_moqtsrc() {
    let Some(pub_bin) = reference_binary("moq-pub") else {
        eprintln!(
            "SKIP: moq-pub not found. Build it with `cargo +stable build --release \
             -p moq-relay-ietf -p moq-pub` in a cloudflare/moq-rs checkout, or point \
             $MOQ_RS_BIN at its directory."
        );
        return;
    };
    let tls = TestCert::generate();
    let port = free_udp_port();
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2gpub";
    let Some(_relay) = spawn_relay(&tls, port) else {
        return;
    };
    // The relay binds asynchronously; give it a moment rather than racing it.
    tokio::time::sleep(Duration::from_millis(750)).await;

    let published = muxed_stream(140).await;

    let mut publisher = match Command::new(&pub_bin)
        .arg(&url)
        .arg("--name")
        .arg(namespace)
        .arg("--tls-disable-verify")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => Reaped(child),
        Err(e) => {
            eprintln!("SKIP: could not start moq-pub: {e}");
            return;
        }
    };
    let mut stdin = publisher.0.stdin.take().expect("moq-pub stdin");

    // Feed the publisher a fragment at a time on a blocking thread, so the
    // broadcast is live while the subscriber attaches.
    let feed = std::thread::spawn(move || {
        let boundaries = box_boundaries(&published);
        for pair in boundaries.windows(2) {
            if stdin.write_all(&published[pair[0]..pair[1]]).is_err() {
                break;
            }
            let _ = stdin.flush();
            std::thread::sleep(Duration::from_millis(25));
        }
        published
    });

    tokio::time::sleep(Duration::from_millis(750)).await;

    let mut src = MoqtSrc::new(&url, namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_num_buffers(FRAMES_WANTED);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .expect("moqtsrc caps");
    let mut captured = CaptureSink::default();
    let emitted = src.run(&mut captured).await.expect("subscribe and play");

    let published = feed.join().expect("feed thread");
    drop(publisher);

    assert_eq!(emitted, FRAMES_WANTED);
    assert_eq!(
        src.selected_track(),
        "1.m4s",
        "the catalog named the reference publisher's video track"
    );
    assert_matches_published(&captured.frames, &published, FRAMES_WANTED as usize - 1);
}

/// `mp4mux ! moqtsink` -> `moq-relay-ietf` -> `moqtsrc`: our own round trip
/// through the reference relay, byte-identical across more than one group.
#[tokio::test]
async fn moqtsink_round_trips_through_the_reference_relay_to_moqtsrc() {
    let tls = TestCert::generate();
    let port = free_udp_port();
    let url = format!("https://127.0.0.1:{port}/");
    let namespace = "g2groundtrip";
    let Some(_relay) = spawn_relay(&tls, port) else {
        return;
    };
    tokio::time::sleep(Duration::from_millis(750)).await;

    let mut mux = Mp4Mux::new();
    mux.configure_pipeline(&h264_caps(64, 48))
        .expect("mp4mux caps");
    let mut sink = MoqtSink::new(&url, namespace).with_server_certificate_hashes(&tls.hash_hex);
    sink.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .expect("moqtsink caps");

    let mut src = MoqtSrc::new(&url, namespace)
        .with_server_certificate_hashes(&tls.hash_hex)
        .with_num_buffers(FRAMES_WANTED);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .expect("moqtsrc caps");

    // The sink applies control messages when a frame arrives, so it has to keep
    // publishing until the subscriber is done; the subscriber stops itself at
    // num-buffers.
    let done = std::cell::Cell::new(false);
    let publish = async {
        let mut published = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut index = 0u64;
        while !done.get() && Instant::now() < deadline {
            let mut captured = CaptureSink::default();
            mux.process(
                PipelinePacket::DataFrame(frame(access_unit(index), index * 33_333_333, index)),
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
        published
    };

    let mut captured = CaptureSink::default();
    let subscribe = async {
        // Let the publisher reach the relay before subscribing.
        tokio::time::sleep(Duration::from_millis(1500)).await;
        let emitted = src.run(&mut captured).await;
        done.set(true);
        emitted
    };

    let (published, emitted) = tokio::join!(publish, subscribe);
    let emitted = emitted.expect("subscribe and play");

    assert_eq!(emitted, FRAMES_WANTED);
    assert_eq!(src.selected_track(), "1.m4s");
    assert!(
        sink.objects_published() > 0,
        "the sink counted the objects it wrote"
    );
    assert_matches_published(&captured.frames, &published, FRAMES_WANTED as usize - 1);
    // The publisher starts a group at each keyframe fragment, so more than one
    // of them in the run means the run crossed a group boundary: a finished
    // subgroup stream, a new one, and a bumped group id.
    let tail: Vec<u8> = captured.frames[1..].concat();
    assert!(
        group_starts(&tail) > 1,
        "the byte-identical run spans more than one group, got {}",
        group_starts(&tail)
    );
}
