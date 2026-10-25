//! M1217: RIST (VSF TR-06-1 Simple Profile) `ristsink` / `ristsrc`. The wire
//! pieces are checked on byte vectors, then a g2g sender and receiver run over
//! loopback UDP through a relay that drops a fixed set of media packets, so
//! NACK and retransmission carry the stream back to byte equality.
//!
//! The `#[ignore]`d interop tests put GStreamer's `ristsink` / `ristsrc` on the
//! other end of the same relay. They need `gst-launch-1.0` with the `rist`,
//! `openh264` and `mpegtsmux` plugins. Run:
//!
//! ```sh
//! cargo test -p g2g-plugins --features rist --test m1217_rist -- --ignored --nocapture
//! ```
#![cfg(feature = "rist")]

use std::time::Duration;

use g2g_core::rtp::{RtpHeader, RTP_HEADER_LEN};
use g2g_core::runtime::{parse_launch, run_simple_pipeline, LatencyProfile, SourceLoop};
use g2g_core::{AsyncElement, PropValue, PropertySpec};

use g2g_plugins::bytestream::{TS_DATAGRAM_PAYLOAD, TS_PACKET_SIZE};
use g2g_plugins::filesink::FileSink;
use g2g_plugins::gst_compat::{gst_equivalent, GstEquivalent};
use g2g_plugins::registry::default_registry;
use g2g_plugins::rist::{
    self, NackScheduler, RetransmissionHistory, SentOn, SequenceRange, MAX_RANGES_PER_NACK,
    MP2T_PAYLOAD_TYPE, RIST_APP_NAME, SUBTYPE_RANGE_NACK, SUBTYPE_RTT_ECHO_REQUEST,
    SUBTYPE_RTT_ECHO_RESPONSE,
};
use g2g_plugins::ristsink::RistSink;
use g2g_plugins::ristsrc::RistSrc;
use g2g_plugins::rtcp::{self, RtcpPacket};
use g2g_plugins::rtpjitter::{JitterConfig, RtpJitterBuffer};

mod rist_common;
use rist_common::{
    assert_recovered_through, collect_from_gst, free_even_port, loopback, output_path, spawn_gst,
    stop_gst, synthetic_ts, FirstMediaOrder, LossyRelay, PacedTsSource, RelayLog, ZeroClock,
    GST_BIND_WAIT, GST_DEADLINE,
};

const MEDIA_SSRC: u32 = 0x1234_5670;
const RECEIVER_SSRC: u32 = 0x0BAD_CAFE;
/// RTCP fixed header: V=2 byte, packet type, 16-bit length.
const RTCP_HEADER_LEN: usize = 4;
const RTCP_WORD: usize = 4;
const RTCP_VERSION_BITS: u8 = 0x80;
const SDES_CNAME_ITEM: u8 = 1;

/// RTP packets the paced g2g sender emits.
const SENT_RTP_PACKETS: usize = 60;
/// Packet indices (from the first original) the relay drops on the g2g sender
/// leg: a burst the receiver sees at once, so a range NACK is the shorter
/// form, then scattered losses a generic NACK covers more cheaply.
const DROPPED_FROM_G2G: &[u16] = &[
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    40, 42, 44,
];
/// The same pattern on the GStreamer sender leg, past its first RTCP packet.
const DROPPED_FROM_GST: &[u16] = &[
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
    119, 120, 121, 122, 123, 124, 140, 142, 144,
];

/// Long enough for both ends to drain their buffers, short enough that a hang
/// fails instead of stalling CI.
const LOOPBACK_DEADLINE: Duration = Duration::from_secs(20);
/// Video frames the GStreamer sender encodes, and its encoder bitrate, sized
/// so the stream runs well past [`DROPPED_FROM_GST`].
const GST_VIDEO_FRAMES: u32 = 150;
const GST_BITRATE: u32 = 2_000_000;

/// The g2g receiver sees the burst of losses at once and the scattered ones
/// apart, so it must have used both NACK forms.
fn assert_both_nack_forms(log: &RelayLog) {
    assert!(
        log.range_nacks > 0,
        "the receiver sent a range NACK: {log:?}"
    );
    assert!(
        log.generic_nacks > 0,
        "the receiver sent a generic NACK: {log:?}"
    );
}

fn ristsrc_on(port: u16) -> RistSrc {
    let mut source = RistSrc::default();
    source
        .set_property("address", PropValue::Str("127.0.0.1".into()))
        .unwrap();
    source
        .set_property("port", PropValue::Uint(port.into()))
        .unwrap();
    source
}

fn ristsink_to(port: u16) -> RistSink {
    let mut sink = RistSink::default();
    sink.set_property("address", PropValue::Str("127.0.0.1".into()))
        .unwrap();
    sink.set_property("port", PropValue::Uint(port.into()))
        .unwrap();
    sink
}

/// Run a paced g2g sender into `sink_port` and a g2g receiver on
/// `receiver_port`, returning what the receiver wrote.
async fn g2g_to_g2g(sink_port: u16, receiver_port: u16, sent: &[u8], name: &str) -> Vec<u8> {
    let received_path = output_path(name);
    let mut source = ristsrc_on(receiver_port);
    let mut recorder = FileSink::new(&received_path);
    let mut player = PacedTsSource {
        stream: sent.to_vec(),
    };
    let mut sink = ristsink_to(sink_port);
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

// ---- wire pieces ----

#[test]
fn the_retransmission_ssrc_differs_only_in_the_low_bit() {
    let retransmission = rist::retransmission_ssrc(MEDIA_SSRC);
    assert_eq!(retransmission ^ MEDIA_SSRC, 1);
    assert!(rist::is_retransmission(retransmission));
    assert!(!rist::is_retransmission(MEDIA_SSRC));
    assert_eq!(rist::media_ssrc(retransmission), MEDIA_SSRC);
    assert_eq!(rist::media_ssrc(MEDIA_SSRC), MEDIA_SSRC);
}

#[test]
fn a_range_nack_has_the_tr_06_1_layout() {
    let missing = [10u16, 11, 12, 20];
    let bytes = rist::build_range_nack(MEDIA_SSRC, &missing);
    let entries = [(10u16, 2u16), (20, 0)];
    let words = RTCP_HEADER_LEN / RTCP_WORD + 2 + entries.len();
    let mut expected = vec![
        RTCP_VERSION_BITS | SUBTYPE_RANGE_NACK,
        rtcp::PT_APP,
        0,
        (words - 1) as u8,
    ];
    expected.extend(MEDIA_SSRC.to_be_bytes());
    expected.extend(RIST_APP_NAME);
    for (start, additional) in entries {
        expected.extend(start.to_be_bytes());
        expected.extend(additional.to_be_bytes());
    }
    assert_eq!(bytes, expected);

    let parsed = rtcp::parse_compound(&bytes);
    assert_eq!(parsed.len(), 1);
    let ranges = rist::requested_ranges(&parsed[0], MEDIA_SSRC).expect("a NACK for the stream");
    assert_eq!(
        ranges,
        entries.map(|(start, additional)| SequenceRange { start, additional })
    );
}

#[test]
fn a_range_nack_splits_at_sixteen_ranges() {
    let scattered: Vec<u16> = (0..=MAX_RANGES_PER_NACK as u16)
        .map(|index| index * 2)
        .collect();
    let packets = rtcp::parse_compound(&rist::build_range_nack(MEDIA_SSRC, &scattered));
    let counts: Vec<usize> = packets
        .iter()
        .map(|packet| rist::requested_ranges(packet, MEDIA_SSRC).unwrap().len())
        .collect();
    assert_eq!(counts, vec![MAX_RANGES_PER_NACK, 1]);
}

#[test]
fn a_generic_nack_is_honored_for_either_ssrc_of_the_stream() {
    for media in [MEDIA_SSRC, rist::retransmission_ssrc(MEDIA_SSRC)] {
        let parsed = rtcp::parse_compound(&rtcp::build_nack(RECEIVER_SSRC, media, &[5, 7]));
        let ranges = rist::requested_ranges(&parsed[0], MEDIA_SSRC).expect("a NACK");
        let starts: Vec<u16> = ranges.iter().map(|range| range.start).collect();
        assert_eq!(starts, vec![5, 7]);
        assert!(ranges.iter().all(|range| range.additional == 0));
    }
    let other_stream = rtcp::parse_compound(&rtcp::build_nack(RECEIVER_SSRC, 0x4444, &[5]));
    assert!(rist::requested_ranges(&other_stream[0], MEDIA_SSRC).is_none());
}

#[test]
fn the_shorter_nack_form_is_sent() {
    let packet_type = |bytes: &[u8]| bytes[1];
    let burst: Vec<u16> = (100..140).collect();
    assert_eq!(
        packet_type(&rist::build_nack_feedback(
            RECEIVER_SSRC,
            MEDIA_SSRC,
            &burst
        )),
        rtcp::PT_APP,
        "one range beats three bitmask words"
    );
    assert_eq!(
        packet_type(&rist::build_nack_feedback(
            RECEIVER_SSRC,
            MEDIA_SSRC,
            &[1, 3, 5]
        )),
        rtcp::PT_RTPFB,
        "one bitmask word beats three ranges"
    );
    assert_eq!(
        packet_type(&rist::build_nack_feedback(RECEIVER_SSRC, MEDIA_SSRC, &[1])),
        rtcp::PT_RTPFB,
        "a tie goes to the generic form"
    );
}

#[test]
fn sdes_carries_one_cname_and_ends_on_a_word() {
    for cname in ["ab", "abcde", "receiver@example"] {
        let bytes = rtcp::build_sdes_cname(RECEIVER_SSRC, cname);
        assert_eq!(bytes.len() % RTCP_WORD, 0, "{cname}: whole words");
        assert_eq!(bytes[0], RTCP_VERSION_BITS | 1, "{cname}: one chunk");
        assert_eq!(bytes[1], rtcp::PT_SDES);
        let words = u16::from_be_bytes([bytes[2], bytes[3]]) as usize + 1;
        assert_eq!(words * RTCP_WORD, bytes.len(), "{cname}: the length field");
        assert_eq!(&bytes[4..8], &RECEIVER_SSRC.to_be_bytes());
        assert_eq!(bytes[8], SDES_CNAME_ITEM);
        assert_eq!(bytes[9] as usize, cname.len());
        let name_end = 10 + cname.len();
        assert_eq!(&bytes[10..name_end], cname.as_bytes());
        let terminator = &bytes[name_end..];
        assert!(
            (1..=RTCP_WORD).contains(&terminator.len()) && terminator.iter().all(|&b| b == 0),
            "{cname}: one to four zero bytes end the chunk"
        );
    }
}

#[test]
fn an_rtt_echo_request_is_echoed_with_its_padding() {
    let timestamp = 0x0102_0304_0506_0708u64;
    let padding = [0xAA_u8; 8];
    let mut body = timestamp.to_be_bytes().to_vec();
    body.extend(0xFFFF_FFFFu32.to_be_bytes());
    body.extend(padding);
    let request = rtcp::build_app(SUBTYPE_RTT_ECHO_REQUEST, MEDIA_SSRC, RIST_APP_NAME, &body);
    let parsed = rtcp::parse_compound(&request);
    let response = rist::rtt_echo_response(&parsed[0]).expect("a request gets a response");
    let RtcpPacket::App {
        subtype,
        ssrc,
        name,
        data,
    } = &rtcp::parse_compound(&response)[0]
    else {
        panic!("the response is an APP packet");
    };
    assert_eq!(*subtype, SUBTYPE_RTT_ECHO_RESPONSE);
    assert_eq!((*ssrc, *name), (MEDIA_SSRC, RIST_APP_NAME));
    assert_eq!(
        &data[..8],
        &timestamp.to_be_bytes(),
        "the timestamp comes back"
    );
    assert_eq!(&data[8..12], &[0; 4], "answered at once");
    assert_eq!(&data[12..], &padding, "the padding comes back");

    let truncated = rtcp::build_app(SUBTYPE_RTT_ECHO_REQUEST, MEDIA_SSRC, RIST_APP_NAME, &[1, 2]);
    assert!(rist::rtt_echo_response(&rtcp::parse_compound(&truncated)[0]).is_none());
}

#[test]
fn malformed_rtcp_is_dropped_without_panicking() {
    let range_nack = rist::build_range_nack(MEDIA_SSRC, &[1, 2, 3]);
    for cut in 0..range_nack.len() {
        let packets = rtcp::parse_compound(&range_nack[..cut]);
        assert!(packets.is_empty(), "a packet cut at {cut} bytes is dropped");
    }
    // An APP whose length field stops before the name.
    let short_app = [RTCP_VERSION_BITS, rtcp::PT_APP, 0, 1, 0, 0, 0, 1];
    assert!(rtcp::parse_compound(&short_app).is_empty());
    // A length field pointing past the datagram.
    let overlong = [RTCP_VERSION_BITS, rtcp::PT_APP, 0xFF, 0xFF, 0, 0, 0, 0];
    assert!(rtcp::parse_compound(&overlong).is_empty());

    let foreign_name = rtcp::build_app(SUBTYPE_RANGE_NACK, MEDIA_SSRC, *b"XXXX", &[0, 1, 0, 0]);
    let packet = &rtcp::parse_compound(&foreign_name)[0];
    assert!(rist::requested_ranges(packet, MEDIA_SSRC).is_none());
    assert!(rist::rtt_echo_response(packet).is_none());
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

// a sender with one link sends everything on it
const ONLY_LINK: usize = 0;

fn packet_with_payload(sequence: u16, payload: u8) -> Vec<u8> {
    let mut packet = media_header(sequence).to_bytes().to_vec();
    packet.push(payload);
    packet
}

#[test]
fn the_history_resends_what_a_range_covers_across_the_sequence_wrap() {
    let mut history = RetransmissionHistory::new(u64::MAX);
    let first = u16::MAX - 1;
    let payloads: Vec<u8> = (0..4).collect();
    for (step, payload) in payloads.iter().enumerate() {
        let sequence = first.wrapping_add(step as u16);
        history.record(
            sequence,
            packet_with_payload(sequence, *payload),
            SentOn::EveryLink,
            0,
        );
    }
    let resent = history.retransmissions(
        &[SequenceRange {
            start: first,
            additional: payloads.len() as u16 - 1,
        }],
        ONLY_LINK,
    );
    let resent_payloads: Vec<u8> = resent.iter().map(|packet| packet[RTP_HEADER_LEN]).collect();
    assert_eq!(resent_payloads, payloads);
    for packet in &resent {
        let header = RtpHeader::parse(packet).unwrap().header;
        assert_eq!(header.ssrc, rist::retransmission_ssrc(MEDIA_SSRC));
    }

    let reaches_into_history = SequenceRange {
        start: first.wrapping_sub(2),
        additional: 3,
    };
    assert_eq!(
        history
            .retransmissions(&[reaches_into_history], ONLY_LINK)
            .len(),
        2
    );
    let everything = SequenceRange {
        start: 0,
        additional: u16::MAX,
    };
    assert_eq!(
        history.retransmissions(&[everything], ONLY_LINK).len(),
        payloads.len()
    );
}

#[test]
fn the_history_forgets_packets_older_than_the_sender_buffer() {
    let keep_ns = 1_000;
    let mut history = RetransmissionHistory::new(keep_ns);
    history.record(0, packet_with_payload(0, 0), SentOn::EveryLink, 0);
    history.record(1, packet_with_payload(1, 1), SentOn::EveryLink, keep_ns + 1);
    let all = SequenceRange {
        start: 0,
        additional: 1,
    };
    let resent = history.retransmissions(&[all], ONLY_LINK);
    assert_eq!(resent.len(), 1, "the first packet aged out");
    assert_eq!(resent[0][RTP_HEADER_LEN], 1);
}

#[test]
fn a_loss_is_requested_after_the_reorder_section_then_retried_across_the_buffer() {
    let reorder_ns = 70;
    let buffer_ns = 1_000;
    let retries = 3;
    let mut scheduler = NackScheduler::new(reorder_ns, buffer_ns, retries);
    let limit = usize::MAX;
    assert!(
        scheduler.due(&[9], 0, limit).is_empty(),
        "still in the reorder section"
    );
    assert_eq!(scheduler.due(&[9], reorder_ns, limit), vec![9]);
    let spacing = (buffer_ns - reorder_ns) / u64::from(retries);
    assert!(scheduler
        .due(&[9], reorder_ns + spacing - 1, limit)
        .is_empty());
    assert_eq!(scheduler.due(&[9], reorder_ns + spacing, limit), vec![9]);
    assert_eq!(
        scheduler.due(&[9], reorder_ns + 2 * spacing, limit),
        vec![9]
    );
    assert!(
        scheduler
            .due(&[9], reorder_ns + 3 * spacing, limit)
            .is_empty(),
        "no more than max-rtx-retries requests"
    );

    let mut scheduler = NackScheduler::new(reorder_ns, buffer_ns, retries);
    scheduler.due(&[4], 0, limit);
    scheduler.due(&[], 1, limit);
    assert!(
        scheduler.due(&[4], reorder_ns, limit).is_empty(),
        "a filled hole that opens again waits a fresh reorder section"
    );
}

fn rtp_packet(sequence: u16) -> Vec<u8> {
    packet_with_payload(sequence, sequence as u8)
}

#[test]
fn constant_delay_holds_every_packet_and_a_late_fill_keeps_its_slot() {
    let delay_ns = 100;
    let mut jitter = RtpJitterBuffer::new(JitterConfig {
        max_hold_ns: delay_ns,
        max_depth: usize::MAX,
    })
    .with_constant_delay();
    jitter.push(&rtp_packet(0), 0);
    jitter.push(&rtp_packet(2), 10);
    assert!(
        jitter.pop(delay_ns - 1).is_none(),
        "held for the whole delay"
    );
    assert_eq!(jitter.pop(delay_ns), Some(rtp_packet(0)));
    assert_eq!(jitter.missing_seqs(), vec![1]);
    // The retransmission of 1 arrives late but goes out with 2.
    jitter.push(&rtp_packet(1), 90);
    assert!(jitter.pop(10 + delay_ns - 1).is_none());
    assert_eq!(jitter.pop(10 + delay_ns), Some(rtp_packet(1)));
    assert_eq!(jitter.pop(10 + delay_ns), Some(rtp_packet(2)));
    assert_eq!(jitter.stats().lost, 0);
}

// ---- properties and registration ----

fn declared_default(specs: &[PropertySpec], name: &str) -> PropValue {
    let spec = specs.iter().find(|s| s.name == name).unwrap();
    spec.parse_value(spec.default.unwrap()).unwrap()
}

fn assert_defaults_match(element: &str, specs: &[PropertySpec], get: impl Fn(&str) -> PropValue) {
    for spec in specs.iter().filter(|spec| spec.default.is_some()) {
        assert_eq!(
            get(spec.name),
            declared_default(specs, spec.name),
            "{element} reports its declared `{}` default",
            spec.name
        );
    }
}

#[test]
fn ristsink_properties_round_trip() {
    let mut sink = RistSink::default();
    let specs = sink.properties();
    assert_defaults_match("ristsink", specs, |name| sink.get_property(name).unwrap());
    for (name, value) in [
        ("address", PropValue::Str("10.0.0.1".into())),
        ("port", PropValue::Uint(6000)),
        ("sender-buffer", PropValue::Uint(2500)),
        ("min-rtcp-interval", PropValue::Uint(50)),
        ("cname", PropValue::Str("sender@example".into())),
    ] {
        sink.set_property(name, value.clone()).unwrap();
        assert_eq!(sink.get_property(name), Some(value), "`{name}` round-trips");
    }
    assert_odd_and_out_of_range_rejected(|name, value| sink.set_property(name, value).is_err());
}

#[test]
fn ristsrc_properties_round_trip() {
    let mut source = RistSrc::default();
    let specs = source.properties();
    assert_defaults_match("ristsrc", specs, |name| source.get_property(name).unwrap());
    for (name, value) in [
        ("address", PropValue::Str("127.0.0.1".into())),
        ("port", PropValue::Uint(6002)),
        ("receiver-buffer", PropValue::Uint(500)),
        ("reorder-section", PropValue::Uint(25)),
        ("max-rtx-retries", PropValue::Uint(3)),
        ("min-rtcp-interval", PropValue::Uint(20)),
        ("cname", PropValue::Str("receiver@example".into())),
    ] {
        source.set_property(name, value.clone()).unwrap();
        assert_eq!(
            source.get_property(name),
            Some(value),
            "`{name}` round-trips"
        );
    }
    assert_odd_and_out_of_range_rejected(|name, value| source.set_property(name, value).is_err());
    assert!(source.latency().live, "a network receiver is live");
    let buffer_ns = 500 * 1_000_000;
    assert_eq!(
        source.latency().min_ns,
        buffer_ns,
        "it adds receiver-buffer"
    );
}

fn assert_odd_and_out_of_range_rejected(mut rejects: impl FnMut(&str, PropValue) -> bool) {
    assert!(
        rejects("port", PropValue::Uint(5005)),
        "an odd port is refused"
    );
    assert!(rejects("port", PropValue::Uint(0)), "port 0 is refused");
    assert!(
        rejects("port", PropValue::Uint(65_536)),
        "a port past 16 bits is refused"
    );
    assert!(
        rejects("min-rtcp-interval", PropValue::Uint(101)),
        "TR-06-1 needs RTCP at least every 100 ms"
    );
    let too_long = "x".repeat(usize::from(u8::MAX) + 1);
    assert!(
        rejects("cname", PropValue::Str(too_long)),
        "a CNAME fits one SDES item"
    );
    assert!(rejects("address", PropValue::Str(String::new())));
}

#[test]
fn both_elements_build_from_a_launch_line_and_the_gst_helpers_answer() {
    let registry = default_registry();
    for line in [
        "ristsrc address=127.0.0.1 port=6004 receiver-buffer=500 ! fakesink",
        "audiotestsrc num-buffers=1 ! ristsink address=127.0.0.1 port=6006 sender-buffer=500",
    ] {
        assert!(
            parse_launch(&registry, line).is_ok(),
            "`{line}` builds a graph"
        );
    }
    assert!(
        parse_launch(&registry, "ristsrc port=6005 ! fakesink").is_err(),
        "an odd port fails the launch line"
    );
    for name in ["ristsink", "ristsrc"] {
        assert_eq!(gst_equivalent(&registry, name), GstEquivalent::Available);
    }
    for (name, element) in [
        ("ristrtxsend", "ristsink"),
        ("ristrtxreceive", "ristsrc"),
        ("ristrtpext", "ristsink"),
        ("ristrtpdeext", "ristsrc"),
        ("rtpmp2tpay", "ristsink"),
        ("rtpmp2tdepay", "ristsrc"),
    ] {
        let GstEquivalent::Unsupported(hint) = gst_equivalent(&registry, name) else {
            panic!("{name} has a hint");
        };
        assert!(hint.contains(element), "{name} points at {element}: {hint}");
    }
}

// ---- g2g to g2g over loopback ----

#[tokio::test]
async fn g2g_to_g2g_without_loss() {
    let sent = synthetic_ts(SENT_RTP_PACKETS);
    let port = free_even_port();
    let received = g2g_to_g2g(port, port, &sent, "g2g_m1217_direct.ts").await;
    assert!(received == sent, "the received bytes equal the sent ones");
}

#[tokio::test]
async fn g2g_to_g2g_recovers_dropped_packets() {
    let sent = synthetic_ts(SENT_RTP_PACKETS);
    let receiver_port = free_even_port();
    let relay = LossyRelay::start(
        loopback(receiver_port),
        DROPPED_FROM_G2G,
        FirstMediaOrder::Unordered,
    );
    let received = g2g_to_g2g(
        relay.media_port,
        receiver_port,
        &sent,
        "g2g_m1217_relayed.ts",
    )
    .await;
    let log = relay.finish();
    assert!(
        log.sent_payloads() == sent,
        "ristsink put the whole stream on the wire"
    );
    assert!(received == sent, "the received bytes equal the sent ones");
    assert_recovered_through(&log, DROPPED_FROM_G2G);
    assert_both_nack_forms(&log);
}

// ---- real-peer interop against gst-launch-1.0 (ignored: needs GStreamer) ----

/// gst sends, g2g receives: `videotestsrc ! openh264enc ! mpegtsmux ! rtpmp2tpay
/// ! ristsink` through the relay into `ristsrc`, against the same TS written by a
/// `tee` branch on the gst side.
#[tokio::test]
#[ignore = "needs gst-launch-1.0"]
async fn gst_ristsink_feeds_ristsrc_through_loss() {
    let reference = output_path("g2g_m1217_gst_sent.ts");
    let received_path = output_path("g2g_m1217_from_gst.ts");
    let receiver_port = free_even_port();
    let relay = LossyRelay::start(
        loopback(receiver_port),
        DROPPED_FROM_GST,
        FirstMediaOrder::Unordered,
    );

    let mut source = ristsrc_on(receiver_port);
    let mut recorder = FileSink::new(&received_path);
    let peer = spawn_gst(&format!(
        "videotestsrc num-buffers={GST_VIDEO_FRAMES} ! openh264enc bitrate={GST_BITRATE} ! \
         h264parse ! mpegtsmux ! tee name=t t. ! queue ! filesink buffer-mode=unbuffered location={} \
         t. ! queue ! rtpmp2tpay ! ristsink address=127.0.0.1 port={}",
        reference.display(),
        relay.media_port
    ));
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
    .expect("ristsrc ends on the gst sender's BYE")
    .expect("the receive pipeline runs");
    stop_gst(peer);
    let log = relay.finish();

    let muxed = std::fs::read(&reference).expect("read the TS gst muxed");
    let received = std::fs::read(&received_path).expect("read the TS g2g received");
    let sent_payloads = log.sent_payloads();
    assert_eq!(
        received.len(),
        sent_payloads.len(),
        "every byte gst sent arrived"
    );
    assert!(
        received == sent_payloads,
        "the received TS equals the sent one"
    );
    // `rtpmp2tpay` never sends the group of TS packets it holds at EOS.
    let unsent_tail = muxed.len() - received.len();
    assert!(
        unsent_tail <= TS_DATAGRAM_PAYLOAD && unsent_tail.is_multiple_of(TS_PACKET_SIZE),
        "gst sent all but its last {unsent_tail} bytes"
    );
    assert!(
        received == muxed[..received.len()],
        "the received TS is what gst muxed"
    );
    assert_recovered_through(&log, DROPPED_FROM_GST);
    assert_both_nack_forms(&log);
}

/// g2g sends, gst receives: `ristsink` through the relay into `gst ristsrc !
/// rtpmp2tdepay ! filesink`.
#[tokio::test]
#[ignore = "needs gst-launch-1.0"]
async fn ristsink_feeds_gst_ristsrc_through_loss() {
    let sent = synthetic_ts(SENT_RTP_PACKETS);
    let received_path = output_path("g2g_m1217_to_gst.ts");
    let receiver_port = free_even_port();
    let peer = spawn_gst(&format!(
        "ristsrc address=127.0.0.1 port={receiver_port} ! rtpmp2tdepay ! \
         filesink buffer-mode=unbuffered location={}",
        received_path.display()
    ));
    tokio::time::sleep(GST_BIND_WAIT).await;
    let relay = LossyRelay::start(
        loopback(receiver_port),
        DROPPED_FROM_G2G,
        FirstMediaOrder::Unordered,
    );

    let mut player = PacedTsSource {
        stream: sent.clone(),
    };
    let mut sink = ristsink_to(relay.media_port);
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

    let received = collect_from_gst(peer, &received_path, sent.len());
    let log = relay.finish();
    assert_eq!(received.len(), sent.len(), "every TS byte arrived");
    assert!(received == sent, "gst received the TS g2g sent");
    assert_recovered_through(&log, DROPPED_FROM_G2G);
}
