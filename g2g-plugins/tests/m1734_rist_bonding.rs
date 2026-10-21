// M1734 / M1735: RIST bonding, multicast and the TR-06-2 RTP header extension
// on `ristsink` / `ristsrc`. The extension is checked on byte vectors, then g2g
// runs against itself, GStreamer (`gst-launch-1.0` with the `rist` plugin) and
// librist (`ristsender` / `ristreceiver` under `~/opt/librist`) over loopback.
// A peer leg prints why it skipped when its peer is missing, and the multicast
// legs skip when multicast loopback does not reach this host.
#![cfg(feature = "rist")]

use std::future::Future;
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::path::PathBuf;
use std::pin::Pin;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use g2g_core::rtp::{RtpHeader, RTP_HEADER_LEN};
use g2g_core::runtime::{run_simple_pipeline, LatencyProfile, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, OutputSink,
    PipelinePacket, PropValue,
};

use g2g_plugins::bytestream::{mpegts_caps, TS_DATAGRAM_PAYLOAD, TS_PACKET_SIZE};
use g2g_plugins::filesink::FileSink;
use g2g_plugins::rist::{
    self, RistExtension, RistExtensionRemover, RistExtensionSettings, MP2T_PAYLOAD_TYPE,
    RESTORED_NULL_PACKET_HEADER, RIST_EXTENSION_PROFILE,
};
use g2g_plugins::ristsink::RistSink;
use g2g_plugins::ristsrc::RistSrc;

mod rist_common;
use rist_common::{
    assert_recovered_through, bound_even_port_pair, collect_from_gst, free_even_port, loopback,
    output_path, spawn_gst, stop_gst, synthetic_ts, LossyRelay, PacedTsSource, PeerProcess,
    RelayLog, ZeroClock, GST_BIND_WAIT, GST_DEADLINE, RELAY_POLL,
};

const MEDIA_SSRC: u32 = 0x1234_5670;
// RTP packets the paced g2g sender emits.
const SENT_RTP_PACKETS: usize = 60;
// RTP packets in the file a GStreamer sender plays, long enough to run past
// its first RTCP packet, before which the receiver cannot NACK.
const GST_SENT_RTP_PACKETS: usize = 1000;
// How long a GStreamer sender waits between two 1316-byte reads, in µs.
const GST_SEND_INTERVAL_US: u32 = 2_000;
// Every this many TS packets one is a null packet.
const NULL_PACKET_SPACING: usize = 5;
// Sequence offsets, from the first packet the lossy link carried, that its
// relay drops. Even, so a round-robin link (every other packet) has them.
const DROPPED_ON_LOSSY_LINK: &[u16] = &[4, 6, 8, 10, 20, 30, 40];
// The same past a GStreamer sender's first RTCP packet.
const DROPPED_FROM_GST_ON_LOSSY_LINK: &[u16] = &[700, 702, 704, 706, 800, 900];
const FIRST_LINK: usize = 0;
const SECOND_LINK: usize = 1;
// Offsets around the 16-bit sequence wrap of the extension legs.
const DROPPED_ACROSS_THE_WRAP: &[u16] = &[
    PACKETS_BEFORE_WRAP - 1,
    PACKETS_BEFORE_WRAP,
    PACKETS_BEFORE_WRAP + 1,
];
const NO_DROPS: &[u16] = &[];
// The extension legs start this many packets before the sequence wraps.
const PACKETS_BEFORE_WRAP: u16 = (SENT_RTP_PACKETS / 2) as u16;
const GST_PACKETS_BEFORE_WRAP: u16 = (GST_SENT_RTP_PACKETS / 2) as u16;
const LINK_COUNT: usize = 2;
const MULTICAST_GROUP: Ipv4Addr = Ipv4Addr::new(239, 255, 73, 34);
const MULTICAST_PROBE_WAIT: Duration = Duration::from_millis(500);
const LOOPBACK_DEADLINE: Duration = Duration::from_secs(20);
// RFC 3550: the X bit of the first header byte.
const RTP_EXTENSION_BIT: u8 = 0x10;
const RTP_EXTENSION_HEADER_LEN: usize = 4;
const RIST_EXTENSION_LEN: usize = RTP_EXTENSION_HEADER_LEN + 4;
// Bits of the first RIST extension byte: N (null packets deleted), E
// (sequence number extension), then the three-bit original packet count.
const NULL_PACKETS_DELETED_FLAG: u8 = 0x80;
const SEQUENCE_EXTENSION_FLAG: u8 = 0x40;
const ORIGINAL_COUNT_SHIFT: u8 = 3;
// TS packets the deletion bitmap covers, the first in the high bit.
const NULL_DELETION_SLOTS: usize = 7;
const LIBRIST_DIR: &str = "opt/librist";
const SIMPLE_PROFILE: &str = "0";

fn first_sequence_before_wrap(packets_before_wrap: u16) -> u16 {
    (u32::from(u16::MAX) + 1 - u32::from(packets_before_wrap)) as u16
}

fn null_ts_packet() -> Vec<u8> {
    let mut packet = RESTORED_NULL_PACKET_HEADER.to_vec();
    packet.resize(TS_PACKET_SIZE, 0);
    packet
}

// [`synthetic_ts`] with every [`NULL_PACKET_SPACING`]th packet a null packet
// in the form `ristrtpdeext` restores it, so a deleted one comes back equal.
fn ts_with_null_packets(rtp_packets: usize) -> Vec<u8> {
    let mut stream = synthetic_ts(rtp_packets);
    for packet in stream
        .chunks_exact_mut(TS_PACKET_SIZE)
        .step_by(NULL_PACKET_SPACING)
    {
        packet.copy_from_slice(&null_ts_packet());
    }
    stream
}

fn media_header(sequence: u16) -> RtpHeader {
    RtpHeader {
        payload_type: MP2T_PAYLOAD_TYPE,
        marker: false,
        sequence,
        timestamp: 0,
        ssrc: MEDIA_SSRC,
    }
}

fn deletion_bits(positions: &[usize]) -> u8 {
    positions.iter().fold(0, |bits, position| {
        bits | (1 << (NULL_DELETION_SLOTS - 1 - position))
    })
}

fn null_positions(payload: &[u8]) -> Vec<usize> {
    payload
        .chunks_exact(TS_PACKET_SIZE)
        .enumerate()
        .filter(|(_, packet)| *packet == null_ts_packet().as_slice())
        .map(|(index, _)| index)
        .collect()
}

const BOTH_EXTENSIONS: RistExtensionSettings = RistExtensionSettings {
    drop_null_packets: true,
    sequence_number_extension: true,
};

// ---- the header extension on byte vectors ----

#[test]
fn the_extension_has_the_ristrtpext_layout() {
    let payload = ts_with_null_packets(1);
    let deleted = null_positions(&payload);
    assert!(!deleted.is_empty(), "the payload holds null packets");
    let sequence_high_bits = 1;
    let header = media_header(0x1234);
    let packet =
        rist::build_media_packet(header, sequence_high_bits, &payload, BOTH_EXTENSIONS).unwrap();

    assert_eq!(packet[0] & RTP_EXTENSION_BIT, RTP_EXTENSION_BIT);
    let extension = &packet[RTP_HEADER_LEN..RTP_HEADER_LEN + RIST_EXTENSION_LEN];
    assert_eq!(&extension[..2], b"RI");
    assert_eq!(
        u16::from_be_bytes([extension[0], extension[1]]),
        RIST_EXTENSION_PROFILE
    );
    assert_eq!(
        u16::from_be_bytes([extension[2], extension[3]]),
        1,
        "one 32-bit word"
    );
    let packet_count = payload.len() / TS_PACKET_SIZE;
    let flags = NULL_PACKETS_DELETED_FLAG
        | SEQUENCE_EXTENSION_FLAG
        | ((packet_count as u8) << ORIGINAL_COUNT_SHIFT);
    assert_eq!(extension[4], flags);
    assert_eq!(
        extension[5],
        deletion_bits(&deleted),
        "188-byte packets, deleted bits"
    );
    assert_eq!(&extension[6..8], &sequence_high_bits.to_be_bytes());

    let kept: Vec<u8> = payload
        .chunks_exact(TS_PACKET_SIZE)
        .enumerate()
        .filter(|(index, _)| !deleted.contains(index))
        .flat_map(|(_, packet)| packet.iter().copied())
        .collect();
    assert_eq!(
        &packet[RTP_HEADER_LEN + RIST_EXTENSION_LEN..],
        kept.as_slice()
    );
    assert_eq!(
        rist::parse_rist_extension(&packet),
        Some(RistExtension {
            null_packets_deleted: true,
            original_packet_count: packet_count as u8,
            dvb_packet_size: false,
            deleted_packets: deletion_bits(&deleted),
            sequence_high_bits: Some(sequence_high_bits),
        })
    );
}

#[test]
fn stripping_the_extension_restores_the_deleted_null_packets() {
    let payload = ts_with_null_packets(1);
    let header = media_header(7);
    let packet = rist::build_media_packet(header, 0, &payload, BOTH_EXTENSIONS).unwrap();
    let stripped = RistExtensionRemover::default()
        .strip(&packet)
        .expect("a valid packet is kept")
        .into_owned();
    let mut expected = header.to_bytes().to_vec();
    expected.extend_from_slice(&payload);
    assert_eq!(stripped, expected);
}

#[test]
fn the_sequence_extension_alone_leaves_the_payload_whole() {
    let payload = ts_with_null_packets(1);
    let settings = RistExtensionSettings {
        drop_null_packets: false,
        sequence_number_extension: true,
    };
    let packet = rist::build_media_packet(media_header(1), 3, &payload, settings).unwrap();
    let extension = rist::parse_rist_extension(&packet).unwrap();
    assert_eq!(
        extension,
        RistExtension {
            null_packets_deleted: false,
            original_packet_count: 0,
            dvb_packet_size: false,
            deleted_packets: 0,
            sequence_high_bits: Some(3),
        },
        "GStreamer leaves the deletion fields zero"
    );
    assert_eq!(
        &packet[RTP_HEADER_LEN + RIST_EXTENSION_LEN..],
        payload.as_slice()
    );

    let plain = rist::build_media_packet(
        media_header(1),
        3,
        &payload,
        RistExtensionSettings::default(),
    )
    .unwrap();
    assert_eq!(
        plain[0] & RTP_EXTENSION_BIT,
        0,
        "no extension when both are off"
    );
    assert_eq!(&plain[RTP_HEADER_LEN..], payload.as_slice());
}

#[test]
fn null_packets_go_only_from_whole_ts_packets_with_sync_bytes() {
    let payload = ts_with_null_packets(1);
    let partial = &payload[..payload.len() - 1];
    let packet = rist::build_media_packet(media_header(1), 0, partial, BOTH_EXTENSIONS).unwrap();
    let extension = rist::parse_rist_extension(&packet).unwrap();
    assert!(
        !extension.null_packets_deleted,
        "a partial payload keeps its nulls"
    );
    assert_eq!(&packet[RTP_HEADER_LEN + RIST_EXTENSION_LEN..], partial);

    let mut not_ts = payload.clone();
    not_ts[TS_PACKET_SIZE] = 0;
    assert!(
        rist::build_media_packet(media_header(1), 0, &not_ts, BOTH_EXTENSIONS).is_err(),
        "a TS packet without its sync byte fails the send"
    );
}

fn extended_packet(sequence_high_bits: u16, sequence: u16) -> Vec<u8> {
    let settings = RistExtensionSettings {
        drop_null_packets: false,
        sequence_number_extension: true,
    };
    rist::build_media_packet(
        media_header(sequence),
        sequence_high_bits,
        &synthetic_ts(1),
        settings,
    )
    .unwrap()
}

#[test]
fn a_packet_far_behind_the_highest_extended_sequence_is_dropped() {
    let mut remover = RistExtensionRemover::default();
    let half_space = u32::from(u16::MAX) / 2 + 1;
    assert!(remover.strip(&extended_packet(1, 5)).is_some());
    let highest = (1u32 << u16::BITS) + 5;
    let too_old = highest - half_space - 1;
    assert!(
        remover
            .strip(&extended_packet(
                (too_old >> u16::BITS) as u16,
                too_old as u16
            ))
            .is_none(),
        "more than half the 16-bit space behind"
    );
    let recent = highest - 3;
    assert!(remover
        .strip(&extended_packet(
            (recent >> u16::BITS) as u16,
            recent as u16
        ))
        .is_some());
}

#[test]
fn malformed_extensions_fail_without_panicking() {
    let packet = rist::build_media_packet(
        media_header(1),
        0,
        &ts_with_null_packets(1),
        BOTH_EXTENSIONS,
    )
    .unwrap();
    for cut in 0..packet.len() {
        let _ = RistExtensionRemover::default().strip(&packet[..cut]);
    }

    // one deleted bit on the last slot of a payload that keeps every packet
    let kept_packets = 5;
    let mut overclaimed = media_header(1).to_bytes().to_vec();
    overclaimed[0] |= RTP_EXTENSION_BIT;
    overclaimed.extend_from_slice(&RIST_EXTENSION_PROFILE.to_be_bytes());
    overclaimed.extend_from_slice(&1u16.to_be_bytes());
    overclaimed.extend_from_slice(&[
        NULL_PACKETS_DELETED_FLAG,
        deletion_bits(&[NULL_DELETION_SLOTS - 1]),
        0,
        0,
    ]);
    overclaimed.extend_from_slice(&synthetic_ts(1)[..kept_packets * TS_PACKET_SIZE]);
    assert!(
        RistExtensionRemover::default()
            .strip(&overclaimed)
            .is_none(),
        "the bits name more kept packets than the payload holds"
    );

    // an RFC 8285 extension is not the RIST one and passes untouched
    let mut foreign = media_header(1).to_bytes().to_vec();
    foreign[0] |= RTP_EXTENSION_BIT;
    foreign.extend_from_slice(&g2g_core::rtp::ONE_BYTE_EXTENSION_PROFILE.to_be_bytes());
    foreign.extend_from_slice(&1u16.to_be_bytes());
    foreign.extend_from_slice(&[0; 4]);
    foreign.extend_from_slice(&synthetic_ts(1));
    let passed = RistExtensionRemover::default().strip(&foreign).unwrap();
    assert_eq!(passed.as_ref(), foreign.as_slice());
}

// ---- peers and the environment ----

fn missing_gst_elements(elements: &[&str]) -> Vec<String> {
    elements
        .iter()
        .filter(|element| {
            !Command::new("gst-inspect-1.0")
                .arg(element)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|status| status.success())
        })
        .map(|element| element.to_string())
        .collect()
}

// `true` when every element is installed, else prints why the leg skips.
fn gst_peer_ready(leg: &str, elements: &[&str]) -> bool {
    let missing = missing_gst_elements(elements);
    if !missing.is_empty() {
        eprintln!(
            "skipping {leg}: gst-launch-1.0 lacks {}",
            missing.join(", ")
        );
    }
    missing.is_empty()
}

fn librist_tool(leg: &str, name: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let tool = PathBuf::from(home).join(LIBRIST_DIR).join("bin").join(name);
    if !tool.is_file() {
        eprintln!("skipping {leg}: no {}", tool.display());
        return None;
    }
    Some(tool)
}

fn spawn_librist(tool: &PathBuf, args: &[String]) -> PeerProcess {
    let lib = tool.parent().unwrap().parent().unwrap().join("lib");
    Command::new(tool)
        .args(args)
        .env("LD_LIBRARY_PATH", lib)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(PeerProcess)
        .expect("start the librist tool")
}

// `true` when a multicast datagram sent with loopback on comes back to this
// host, else prints why the leg skips.
fn multicast_loopback_works(leg: &str) -> bool {
    let probe = || -> std::io::Result<()> {
        let receiver = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))?;
        receiver.join_multicast_v4(&MULTICAST_GROUP, &Ipv4Addr::UNSPECIFIED)?;
        receiver.set_read_timeout(Some(MULTICAST_PROBE_WAIT))?;
        let port = receiver.local_addr()?.port();
        let sender = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))?;
        sender.set_multicast_loop_v4(true)?;
        sender.send_to(b"probe", (MULTICAST_GROUP, port))?;
        receiver.recv_from(&mut [0u8; 16]).map(|_| ())
    };
    match probe() {
        Ok(()) => true,
        Err(error) => {
            eprintln!("skipping {leg}: multicast loopback does not work here: {error}");
            false
        }
    }
}

fn distinct_even_ports(count: usize) -> Vec<u16> {
    let pairs: Vec<_> = (0..count).map(|_| bound_even_port_pair()).collect();
    pairs
        .iter()
        .map(|(media, _)| media.local_addr().expect("bound address").port())
        .collect()
}

fn bonding_addresses(ports: &[u16]) -> String {
    let entries: Vec<String> = ports
        .iter()
        .map(|port| format!("127.0.0.1:{port}"))
        .collect();
    entries.join(",")
}

fn ristsink_with(properties: &[(&str, PropValue)]) -> RistSink {
    let mut sink = RistSink::default();
    for (name, value) in properties {
        sink.set_property(name, value.clone())
            .unwrap_or_else(|error| panic!("ristsink `{name}`: {error:?}"));
    }
    sink
}

fn ristsrc_with(properties: &[(&str, PropValue)]) -> RistSrc {
    let mut source = RistSrc::default();
    for (name, value) in properties {
        source
            .set_property(name, value.clone())
            .unwrap_or_else(|error| panic!("ristsrc `{name}`: {error:?}"));
    }
    source
}

fn bonded_to(ports: &[u16]) -> (&'static str, PropValue) {
    (
        "bonding-addresses",
        PropValue::Str(bonding_addresses(ports)),
    )
}

fn bonding_method(method: &str) -> (&'static str, PropValue) {
    ("bonding-method", PropValue::Str(method.into()))
}

fn relays_to(
    receiver_ports: &[u16],
    lossy_link: usize,
    dropped: &'static [u16],
) -> Vec<LossyRelay> {
    receiver_ports
        .iter()
        .enumerate()
        .map(|(link, &port)| {
            let dropped = if link == lossy_link {
                dropped
            } else {
                NO_DROPS
            };
            LossyRelay::start(loopback(port), dropped)
        })
        .collect()
}

fn relay_ports(relays: &[LossyRelay]) -> Vec<u16> {
    relays.iter().map(|relay| relay.media_port).collect()
}

fn finish(relays: Vec<LossyRelay>) -> Vec<RelayLog> {
    relays.into_iter().map(LossyRelay::finish).collect()
}

// Run a paced g2g sender and a g2g receiver to the end, returning what the
// receiver wrote.
async fn g2g_to_g2g(mut sink: RistSink, mut source: RistSrc, sent: &[u8], name: &str) -> Vec<u8> {
    let received_path = output_path(name);
    let mut recorder = FileSink::new(&received_path);
    let mut player = PacedTsSource {
        stream: sent.to_vec(),
    };
    let receive = run_simple_pipeline(
        &mut source,
        &mut recorder,
        &ZeroClock,
        LatencyProfile::Live.link_capacity(),
    );
    let send = run_simple_pipeline(
        &mut player,
        &mut sink,
        &ZeroClock,
        LatencyProfile::Live.link_capacity(),
    );
    let (received, sent_run) =
        tokio::time::timeout(LOOPBACK_DEADLINE, async { tokio::join!(receive, send) })
            .await
            .expect("both ends finish");
    sent_run.expect("the send pipeline runs");
    received.expect("the receive pipeline runs");
    assert!(sink.eos_seen(), "the sender reached Eos");
    assert!(
        recorder.eos_seen(),
        "the receiver ended on the sender's BYE"
    );
    std::fs::read(&received_path).expect("read the received stream")
}

// Run a paced g2g sender to the end.
async fn send_from_g2g(mut sink: RistSink, sent: &[u8]) {
    let mut player = PacedTsSource {
        stream: sent.to_vec(),
    };
    tokio::time::timeout(
        GST_DEADLINE,
        run_simple_pipeline(
            &mut player,
            &mut sink,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("the sender finishes")
    .expect("the send pipeline runs");
}

// Run a g2g receiver until the sender's BYE, returning what it wrote.
async fn receive_until_goodbye(mut source: RistSrc, name: &str) -> Vec<u8> {
    let received_path = output_path(name);
    let mut recorder = FileSink::new(&received_path);
    tokio::time::timeout(
        GST_DEADLINE,
        run_simple_pipeline(
            &mut source,
            &mut recorder,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("ristsrc ends on the sender's BYE")
    .expect("the receive pipeline runs");
    std::fs::read(&received_path).expect("read the received stream")
}

// Every original packet a relay saw carries the RIST extension, and its
// 32-bit sequence number counts up from `first_sequence` across the wrap.
fn assert_extended_sequences(log: &RelayLog, first_sequence: u16) {
    let mut deleted_any = false;
    for (index, packet) in log.originals.iter().enumerate() {
        let sequence = RtpHeader::parse(packet).unwrap().header.sequence;
        let extension = rist::parse_rist_extension(packet).expect("the RIST extension");
        let high_bits = extension
            .sequence_high_bits
            .expect("the sequence extension");
        let extended = (u32::from(high_bits) << u16::BITS) | u32::from(sequence);
        assert_eq!(extended, u32::from(first_sequence) + index as u32);
        deleted_any |= extension.deleted_packets != 0;
    }
    let last = log.originals.len() as u32 + u32::from(first_sequence);
    assert!(
        last > u32::from(u16::MAX),
        "the stream crossed the 16-bit wrap"
    );
    assert!(deleted_any, "null packets were deleted on the wire");
}

// A gst sender playing `ts` paced like a live source into `sink_tail`.
fn gst_ts_sender(ts: &[u8], name: &str, payloader: &str, sink_tail: &str) -> PeerProcess {
    let path = output_path(name);
    std::fs::write(&path, ts).expect("write the TS the gst sender plays");
    spawn_gst(&format!(
        "filesrc location={} blocksize={TS_DATAGRAM_PAYLOAD} ! \
         identity sleep-time={GST_SEND_INTERVAL_US} ! \
         video/mpegts,systemstream=true,packetsize={TS_PACKET_SIZE} ! {payloader} ! {sink_tail}",
        path.display()
    ))
}

// `rtpmp2tpay` never sends the group of TS packets it holds at EOS, so a gst
// sender delivers all of `muxed` but that tail.
fn assert_gst_sent_all_but_its_tail(received: &[u8], muxed: &[u8]) {
    let unsent_tail = muxed.len() - received.len();
    assert!(
        unsent_tail <= TS_DATAGRAM_PAYLOAD && unsent_tail.is_multiple_of(TS_PACKET_SIZE),
        "gst sent all but its last {unsent_tail} bytes"
    );
    assert!(
        received == &muxed[..received.len()],
        "the received TS is what gst played"
    );
}

// ---- g2g to g2g ----

#[tokio::test]
async fn broadcast_bonding_delivers_every_packet_past_a_lossy_link() {
    let sent = synthetic_ts(SENT_RTP_PACKETS);
    let receiver_ports = distinct_even_ports(LINK_COUNT);
    let relays = relays_to(&receiver_ports, FIRST_LINK, DROPPED_ON_LOSSY_LINK);
    let sink = ristsink_with(&[
        bonded_to(&relay_ports(&relays)),
        bonding_method("broadcast"),
    ]);
    let source = ristsrc_with(&[bonded_to(&receiver_ports)]);
    let received = g2g_to_g2g(sink, source, &sent, "g2g_m1734_broadcast.ts").await;
    let logs = finish(relays);
    assert!(received == sent, "the received bytes equal the sent ones");
    assert_eq!(logs[0].dropped.len(), DROPPED_ON_LOSSY_LINK.len());
    for log in &logs {
        assert!(
            log.sent_payloads() == sent,
            "every link carried the whole stream"
        );
    }
}

#[tokio::test]
async fn round_robin_bonding_splits_the_stream_and_recovers_on_the_lossy_link() {
    let sent = synthetic_ts(SENT_RTP_PACKETS);
    let receiver_ports = distinct_even_ports(LINK_COUNT);
    let relays = relays_to(&receiver_ports, FIRST_LINK, DROPPED_ON_LOSSY_LINK);
    let sink = ristsink_with(&[
        bonded_to(&relay_ports(&relays)),
        bonding_method("round-robin"),
    ]);
    let source = ristsrc_with(&[bonded_to(&receiver_ports)]);
    let received = g2g_to_g2g(sink, source, &sent, "g2g_m1734_round_robin.ts").await;
    let logs = finish(relays);
    assert!(received == sent, "the received bytes equal the sent ones");
    for log in &logs {
        assert_eq!(
            log.originals.len(),
            SENT_RTP_PACKETS / LINK_COUNT,
            "each link carried half"
        );
    }
    let first_sequences: Vec<u16> = logs
        .iter()
        .map(|log| RtpHeader::parse(&log.originals[0]).unwrap().header.sequence)
        .collect();
    assert_eq!(
        first_sequences[1],
        first_sequences[0].wrapping_add(1),
        "the links alternate"
    );
    assert_recovered_through(&logs[0], DROPPED_ON_LOSSY_LINK);
    assert_eq!(
        logs[1].retransmissions, 0,
        "a link resends only the packets it carried"
    );
}

#[tokio::test]
async fn multicast_g2g_to_g2g() {
    if !multicast_loopback_works("multicast_g2g_to_g2g") {
        return;
    }
    let sent = synthetic_ts(SENT_RTP_PACKETS);
    let port = free_even_port();
    let group = multicast_group_properties(port);
    let received = g2g_to_g2g(
        ristsink_with(&group),
        ristsrc_with(&group),
        &sent,
        "g2g_m1734_multicast.ts",
    )
    .await;
    assert!(received == sent, "the received bytes equal the sent ones");
}

fn extension_sink_properties(first_sequence: u16, port: u16) -> Vec<(&'static str, PropValue)> {
    vec![
        ("address", PropValue::Str("127.0.0.1".into())),
        ("port", PropValue::Uint(port.into())),
        ("sequence-number-extension", PropValue::Bool(true)),
        ("drop-null-ts-packets", PropValue::Bool(true)),
        ("seqnum-offset", PropValue::Int(first_sequence.into())),
    ]
}

#[tokio::test]
async fn the_extension_carries_g2g_to_g2g_across_the_sequence_wrap() {
    let sent = ts_with_null_packets(SENT_RTP_PACKETS);
    let receiver_port = free_even_port();
    let relay = LossyRelay::start(loopback(receiver_port), DROPPED_ACROSS_THE_WRAP);
    let first_sequence = first_sequence_before_wrap(PACKETS_BEFORE_WRAP);
    let sink = ristsink_with(&extension_sink_properties(first_sequence, relay.media_port));
    let source = ristsrc_with(&[
        ("address", PropValue::Str("127.0.0.1".into())),
        ("port", PropValue::Uint(receiver_port.into())),
    ]);
    let received = g2g_to_g2g(sink, source, &sent, "g2g_m1734_extension.ts").await;
    let log = relay.finish();
    assert!(
        received == sent,
        "the null packets came back and nothing was lost"
    );
    assert!(
        log.sent_payloads().len() < sent.len(),
        "the wire carried fewer bytes"
    );
    assert_extended_sequences(&log, first_sequence);
    assert_recovered_through(&log, DROPPED_ACROSS_THE_WRAP);
}

// ---- against GStreamer ----

async fn g2g_bonded_to_gst(method: &str, dropped_on_first_link: &'static [u16], name: &str) {
    if !gst_peer_ready(name, &["ristsrc", "rtpmp2tdepay"]) {
        return;
    }
    let sent = synthetic_ts(SENT_RTP_PACKETS);
    let received_path = output_path(name);
    let receiver_ports = distinct_even_ports(LINK_COUNT);
    let peer = spawn_gst(&format!(
        "ristsrc bonding-addresses={} ! rtpmp2tdepay ! filesink buffer-mode=unbuffered location={}",
        bonding_addresses(&receiver_ports),
        received_path.display()
    ));
    tokio::time::sleep(GST_BIND_WAIT).await;
    let relays = relays_to(&receiver_ports, FIRST_LINK, dropped_on_first_link);
    let sink = ristsink_with(&[bonded_to(&relay_ports(&relays)), bonding_method(method)]);
    send_from_g2g(sink, &sent).await;
    let received = collect_from_gst(peer, &received_path, sent.len());
    let logs = finish(relays);
    assert_eq!(logs[0].dropped.len(), dropped_on_first_link.len());
    assert_eq!(received.len(), sent.len(), "every TS byte arrived");
    assert!(received == sent, "gst received the TS g2g sent");
}

#[tokio::test]
async fn g2g_broadcast_bonding_feeds_gst_ristsrc_past_a_lossy_link() {
    g2g_bonded_to_gst(
        "broadcast",
        DROPPED_ON_LOSSY_LINK,
        "g2g_m1734_broadcast_to_gst.ts",
    )
    .await;
}

#[tokio::test]
async fn g2g_round_robin_bonding_feeds_gst_ristsrc() {
    g2g_bonded_to_gst("round-robin", NO_DROPS, "g2g_m1734_round_robin_to_gst.ts").await;
}

async fn gst_bonded_to_g2g(method: &str, lossy_link: usize, name: &str) {
    if !gst_peer_ready(name, &["ristsink", "rtpmp2tpay", "identity"]) {
        return;
    }
    let played = synthetic_ts(GST_SENT_RTP_PACKETS);
    let receiver_ports = distinct_even_ports(LINK_COUNT);
    let relays = relays_to(&receiver_ports, lossy_link, DROPPED_FROM_GST_ON_LOSSY_LINK);
    let source = ristsrc_with(&[bonded_to(&receiver_ports)]);
    let peer = gst_ts_sender(
        &played,
        &format!("{name}.sent"),
        "rtpmp2tpay",
        &format!(
            "ristsink bonding-addresses={} bonding-method={method}",
            bonding_addresses(&relay_ports(&relays))
        ),
    );
    let received = receive_until_goodbye(source, name).await;
    stop_gst(peer);
    let logs = finish(relays);
    assert_eq!(
        logs[lossy_link].dropped.len(),
        DROPPED_FROM_GST_ON_LOSSY_LINK.len()
    );
    assert_gst_sent_all_but_its_tail(&received, &played);
}

#[tokio::test]
async fn gst_broadcast_bonding_feeds_ristsrc_past_a_lossy_link() {
    gst_bonded_to_g2g("broadcast", FIRST_LINK, "g2g_m1734_broadcast_from_gst.ts").await;
}

#[tokio::test]
async fn gst_round_robin_bonding_feeds_ristsrc_past_a_lossy_link() {
    // GStreamer 1.28's round-robin `ristsink` sends no RTCP on its first bond
    // until the stream ends, so the receiver can only NACK on the second
    gst_bonded_to_g2g(
        "round-robin",
        SECOND_LINK,
        "g2g_m1734_round_robin_from_gst.ts",
    )
    .await;
}

#[tokio::test]
async fn multicast_g2g_to_gst_ristsrc() {
    let leg = "multicast_g2g_to_gst_ristsrc";
    if !gst_peer_ready(leg, &["ristsrc", "rtpmp2tdepay"]) || !multicast_loopback_works(leg) {
        return;
    }
    let sent = synthetic_ts(SENT_RTP_PACKETS);
    let received_path = output_path("g2g_m1734_multicast_to_gst.ts");
    let port = free_even_port();
    let peer = spawn_gst(&format!(
        "ristsrc address={MULTICAST_GROUP} port={port} multicast-loopback=true ! rtpmp2tdepay ! \
         filesink buffer-mode=unbuffered location={}",
        received_path.display()
    ));
    tokio::time::sleep(GST_BIND_WAIT).await;
    let sink = ristsink_with(&multicast_group_properties(port));
    send_from_g2g(sink, &sent).await;
    let received = collect_from_gst(peer, &received_path, sent.len());
    assert!(
        received == sent,
        "gst received the TS g2g sent to the group"
    );
}

// g2g adds the extension, gst `ristrtpdeext` removes it: a wrong high half
// past the wrap makes `ristrtpdeext` drop every later packet.
#[tokio::test]
async fn g2g_extension_feeds_gst_ristrtpdeext_across_the_wrap() {
    let leg = "g2g_extension_feeds_gst_ristrtpdeext_across_the_wrap";
    if !gst_peer_ready(leg, &["ristrtpdeext", "rtpmp2tdepay", "udpsrc"]) {
        return;
    }
    let sent = ts_with_null_packets(SENT_RTP_PACKETS);
    let received_path = output_path("g2g_m1734_extension_to_gst.ts");
    let port = free_even_port();
    let peer = spawn_gst(&format!(
        "udpsrc address=127.0.0.1 port={port} \
         caps=application/x-rtp,media=video,clock-rate=90000,encoding-name=MP2T,payload={MP2T_PAYLOAD_TYPE} ! \
         ristrtpdeext ! rtpmp2tdepay ! filesink buffer-mode=unbuffered location={}",
        received_path.display()
    ));
    tokio::time::sleep(GST_BIND_WAIT).await;
    let first_sequence = first_sequence_before_wrap(PACKETS_BEFORE_WRAP);
    send_from_g2g(
        ristsink_with(&extension_sink_properties(first_sequence, port)),
        &sent,
    )
    .await;
    let received = collect_from_gst(peer, &received_path, sent.len());
    assert_eq!(
        received.len(),
        sent.len(),
        "every packet past the wrap arrived"
    );
    assert!(
        received == sent,
        "gst restored the null packets g2g deleted"
    );
}

// gst `ristrtpext` adds the extension, `ristsrc` removes it.
#[tokio::test]
async fn gst_ristrtpext_feeds_ristsrc_across_the_wrap() {
    let leg = "gst_ristrtpext_feeds_ristsrc_across_the_wrap";
    if !gst_peer_ready(leg, &["ristrtpext", "ristsink", "rtpmp2tpay", "identity"]) {
        return;
    }
    let played = ts_with_null_packets(GST_SENT_RTP_PACKETS);
    let receiver_port = free_even_port();
    let relay = LossyRelay::start(loopback(receiver_port), NO_DROPS);
    let first_sequence = first_sequence_before_wrap(GST_PACKETS_BEFORE_WRAP);
    let source = ristsrc_with(&[
        ("address", PropValue::Str("127.0.0.1".into())),
        ("port", PropValue::Uint(receiver_port.into())),
    ]);
    let peer = gst_ts_sender(
        &played,
        "g2g_m1734_extension_from_gst.sent",
        &format!(
            "rtpmp2tpay seqnum-offset={first_sequence} ! \
             ristrtpext sequence-number-extension=true drop-null-ts-packets=true"
        ),
        &format!("ristsink address=127.0.0.1 port={}", relay.media_port),
    );
    let received = receive_until_goodbye(source, "g2g_m1734_extension_from_gst.ts").await;
    stop_gst(peer);
    let log = relay.finish();
    assert_extended_sequences(&log, first_sequence);
    assert_gst_sent_all_but_its_tail(&received, &played);
}

// ---- against librist ----

#[derive(Debug)]
struct CollectingSink {
    received: Arc<Mutex<Vec<u8>>>,
}

impl AsyncElement for CollectingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&mpegts_caps())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(mpegts_caps()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                let bytes = frame
                    .domain
                    .as_system_slice()
                    .ok_or(G2gError::UnsupportedDomain)?;
                self.received.lock().unwrap().extend_from_slice(bytes);
            }
            Ok(())
        })
    }
}

// Run `source` until it has emitted `expected` bytes: librist ends a stream
// without the BYE `ristsrc` waits for.
async fn receive_bytes(mut source: RistSrc, expected: usize) -> Vec<u8> {
    let received = Arc::new(Mutex::new(Vec::new()));
    let mut sink = CollectingSink {
        received: received.clone(),
    };
    let run = run_simple_pipeline(
        &mut source,
        &mut sink,
        &ZeroClock,
        LatencyProfile::Live.link_capacity(),
    );
    let filled = async {
        while received.lock().unwrap().len() < expected {
            tokio::time::sleep(RELAY_POLL).await;
        }
    };
    tokio::time::timeout(GST_DEADLINE, async {
        tokio::select! {
            result = run => {
                result.expect("the receive pipeline runs");
            }
            () = filled => {}
        }
    })
    .await
    .expect("the stream arrives");
    let bytes = received.lock().unwrap().clone();
    bytes
}

// Start `ristreceiver` on `input_url` and collect what it puts out over UDP
// until `expected` bytes have arrived.
fn ristreceiver_output(
    tool: &PathBuf,
    input_url: String,
    expected: usize,
) -> (PeerProcess, std::thread::JoinHandle<Vec<u8>>) {
    let output = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind the output socket");
    output
        .set_read_timeout(Some(RELAY_POLL))
        .expect("read timeout");
    let output_port = output.local_addr().expect("bound address").port();
    let peer = spawn_librist(
        tool,
        &[
            "-p".into(),
            SIMPLE_PROFILE.into(),
            "-i".into(),
            input_url,
            "-o".into(),
            format!("udp://127.0.0.1:{output_port}"),
        ],
    );
    let collector = std::thread::spawn(move || {
        let mut received = Vec::new();
        let mut datagram = vec![0u8; rist::MAX_DATAGRAM];
        let deadline = Instant::now() + GST_DEADLINE;
        while received.len() < expected && Instant::now() < deadline {
            if let Ok(length) = output.recv(&mut datagram) {
                received.extend_from_slice(&datagram[..length]);
            }
        }
        received
    });
    (peer, collector)
}

// Start `ristsender` towards `output_urls` and feed it `stream` over UDP, paced
// like a live source, once it had time to bind.
fn ristsender_input(
    tool: &PathBuf,
    output_urls: String,
    stream: Vec<u8>,
) -> (PeerProcess, std::thread::JoinHandle<()>) {
    let input_port = free_even_port();
    let peer = spawn_librist(
        tool,
        &[
            "-p".into(),
            SIMPLE_PROFILE.into(),
            "-i".into(),
            format!("udp://@127.0.0.1:{input_port}"),
            "-o".into(),
            output_urls,
        ],
    );
    let player = std::thread::spawn(move || {
        std::thread::sleep(GST_BIND_WAIT);
        let socket = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind the input socket");
        for datagram in stream.chunks(TS_DATAGRAM_PAYLOAD) {
            socket
                .send_to(
                    datagram,
                    SocketAddr::from((Ipv4Addr::LOCALHOST, input_port)),
                )
                .expect("feed ristsender");
            std::thread::sleep(rist_common::SEND_INTERVAL);
        }
    });
    (peer, player)
}

fn multicast_group_properties(port: u16) -> [(&'static str, PropValue); 3] {
    [
        ("address", PropValue::Str(MULTICAST_GROUP.to_string())),
        ("port", PropValue::Uint(port.into())),
        ("multicast-loopback", PropValue::Bool(true)),
    ]
}

// librist's `ristreceiver` passes on the copies of every bonded input, so the
// leg into it runs one link, through loss.
#[tokio::test]
async fn g2g_ristsink_feeds_ristreceiver_through_loss() {
    let leg = "g2g_ristsink_feeds_ristreceiver_through_loss";
    let Some(tool) = librist_tool(leg, "ristreceiver") else {
        return;
    };
    let sent = synthetic_ts(SENT_RTP_PACKETS);
    let receiver_port = free_even_port();
    let (peer, collector) = ristreceiver_output(
        &tool,
        format!("rist://@127.0.0.1:{receiver_port}"),
        sent.len(),
    );
    tokio::time::sleep(GST_BIND_WAIT).await;
    let relay = LossyRelay::start(loopback(receiver_port), DROPPED_ON_LOSSY_LINK);
    let sink = ristsink_with(&[
        ("address", PropValue::Str("127.0.0.1".into())),
        ("port", PropValue::Uint(relay.media_port.into())),
    ]);
    send_from_g2g(sink, &sent).await;
    let received = collector.join().expect("the output collector");
    stop_gst(peer);
    let log = relay.finish();
    assert!(received == sent, "ristreceiver put out the TS g2g sent");
    assert_recovered_through(&log, DROPPED_ON_LOSSY_LINK);
}

#[tokio::test]
async fn ristsender_broadcast_bonding_feeds_ristsrc() {
    let leg = "ristsender_broadcast_bonding_feeds_ristsrc";
    let Some(tool) = librist_tool(leg, "ristsender") else {
        return;
    };
    let sent = synthetic_ts(SENT_RTP_PACKETS);
    let receiver_ports = distinct_even_ports(LINK_COUNT);
    let source = ristsrc_with(&[bonded_to(&receiver_ports)]);
    // weight 0 makes every librist peer a duplicate path
    let outputs: Vec<String> = receiver_ports
        .iter()
        .map(|port| format!("rist://127.0.0.1:{port}?weight=0"))
        .collect();
    let (peer, player) = ristsender_input(&tool, outputs.join(","), sent.clone());
    let received = receive_bytes(source, sent.len()).await;
    player.join().expect("the input player");
    stop_gst(peer);
    assert!(
        received == sent,
        "ristsrc merged the TS ristsender sent on both links"
    );
}

#[tokio::test]
async fn multicast_g2g_to_ristreceiver() {
    let leg = "multicast_g2g_to_ristreceiver";
    let Some(tool) = librist_tool(leg, "ristreceiver") else {
        return;
    };
    if !multicast_loopback_works(leg) {
        return;
    }
    let sent = synthetic_ts(SENT_RTP_PACKETS);
    let port = free_even_port();
    let (peer, collector) = ristreceiver_output(
        &tool,
        format!("rist://@{MULTICAST_GROUP}:{port}"),
        sent.len(),
    );
    tokio::time::sleep(GST_BIND_WAIT).await;
    send_from_g2g(ristsink_with(&multicast_group_properties(port)), &sent).await;
    let received = collector.join().expect("the output collector");
    stop_gst(peer);
    assert!(
        received == sent,
        "ristreceiver received the TS g2g sent to the group"
    );
}

#[tokio::test]
async fn multicast_ristsender_to_ristsrc() {
    let leg = "multicast_ristsender_to_ristsrc";
    let Some(tool) = librist_tool(leg, "ristsender") else {
        return;
    };
    if !multicast_loopback_works(leg) {
        return;
    }
    let sent = synthetic_ts(SENT_RTP_PACKETS);
    let port = free_even_port();
    let source = ristsrc_with(&multicast_group_properties(port));
    let (peer, player) = ristsender_input(
        &tool,
        format!("rist://{MULTICAST_GROUP}:{port}"),
        sent.clone(),
    );
    let received = receive_bytes(source, sent.len()).await;
    player.join().expect("the input player");
    stop_gst(peer);
    assert!(
        received == sent,
        "ristsrc received the TS ristsender sent to the group"
    );
}
