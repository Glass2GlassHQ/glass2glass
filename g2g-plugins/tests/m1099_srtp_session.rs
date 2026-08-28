//! M1099: the SRTP session controls a long-running stream needs, on the real
//! runner: a sized replay window, repeated transmission, MKI-selected keys, and
//! the counters each element keeps.
//!
//! ```sh
//! cargo test -p g2g-plugins --features srtp,std --test m1099_srtp_session
//! ```
//!
//! The `#[ignore]`d legs put `gst-launch-1.0`'s libsrtp on the other side of an
//! MKI exchange, which a g2g to g2g loopback cannot check. Run:
//!
//! ```sh
//! cargo test -p g2g-plugins --features srtp,std --test m1099_srtp_session -- --ignored --nocapture
//! ```
#![cfg(all(feature = "srtp", feature = "std"))]

use g2g_core::rtp::RtpHeader;
use g2g_core::{AsyncElement, PropValue};

use g2g_plugins::srtp::{
    SrtpFlow, SrtpKeyProvider, SrtpKeyingMaterial, SrtpStreamStats, AUTHENTICATION_TAG_LENGTH,
    DEFAULT_REPLAY_WINDOW, MINIMUM_REPLAY_WINDOW, SRTCP_INDEX_LENGTH,
};
use g2g_plugins::srtpdec::{SrtpDec, SrtpDecStats};
use g2g_plugins::srtpenc::SrtpEncStats;

mod srtp_common;
use srtp_common::{
    decoder, encoder, master_key, numbered_files, peer_cipher_arguments, peer_directory,
    plain_caps, protected_caps, rtcp_packets, rtp_packet, rtp_packets, run_one, run_peer,
    write_numbered_files, MASTER_KEY_HEX, PACKET_COUNT, PEER_BUFFERS, SECOND_KEY_HEX,
    SYNCHRONIZATION_SOURCE, WRAP_START_SEQUENCE,
};

/// Sequence the reordering legs start at: far enough above zero that a packet
/// arriving late is a reorder, not a wrap back past the rollover counter.
const REORDER_START_SEQUENCE: u16 = 1_000;
/// Positions the delayed packet falls behind the newest one: past a 64-packet
/// window, inside the 128-packet default.
const REORDER_DISTANCE: usize = 100;

/// The MKIs the two decoder keys are selected by. Four bytes, the width gst's
/// `mki=deadbeef` writes.
const FIRST_MKI: &[u8] = &[0xde, 0xad, 0xbe, 0xef];
const SECOND_MKI: &[u8] = &[0x0b, 0xad, 0xca, 0xfe];
/// An MKI no key claims.
const UNKNOWN_MKI: &[u8] = &[0xff, 0xff, 0xff, 0xff];

const FIRST_MKI_HEX: &str = "deadbeef";
const SECOND_MKI_HEX: &str = "0badcafe";

/// The packet a leg damages or re-tags, somewhere in the middle of the run.
const TAMPERED_PACKET: usize = 3;

fn window_property(packets: usize) -> PropValue {
    PropValue::Uint(packets as u64)
}

/// `count` packets from [`REORDER_START_SEQUENCE`], protected, then reordered so
/// the first one arrives last.
async fn reordered_packets(count: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let packets = rtp_packets(REORDER_START_SEQUENCE, count, SYNCHRONIZATION_SOURCE);
    let protected = run_one(
        &mut encoder(MASTER_KEY_HEX),
        plain_caps(SrtpFlow::Rtp),
        packets.clone(),
    )
    .await;

    let delay_to_the_end = |mut ordered: Vec<Vec<u8>>| {
        let first = ordered.remove(0);
        ordered.push(first);
        ordered
    };
    (delay_to_the_end(protected), delay_to_the_end(packets))
}

/// The window decides how far a packet may be reordered before the decoder
/// treats it as too old to judge.
#[tokio::test]
async fn the_replay_window_sizes_how_far_a_packet_may_be_reordered() {
    assert!(
        (MINIMUM_REPLAY_WINDOW..DEFAULT_REPLAY_WINDOW).contains(&REORDER_DISTANCE),
        "the reorder has to fall between the two windows under test"
    );
    let (reordered, expected) = reordered_packets(REORDER_DISTANCE + 1).await;

    let wide = run_one(
        &mut decoder(MASTER_KEY_HEX),
        protected_caps(SrtpFlow::Rtp),
        reordered.clone(),
    )
    .await;
    assert_eq!(
        wide, expected,
        "the default window still covers a packet {REORDER_DISTANCE} behind"
    );

    let mut narrow = decoder(MASTER_KEY_HEX)
        .with_replay_window(MINIMUM_REPLAY_WINDOW)
        .expect("the minimum is an accepted window");
    let recovered = run_one(&mut narrow, protected_caps(SrtpFlow::Rtp), reordered).await;
    assert_eq!(
        recovered,
        expected[..expected.len() - 1].to_vec(),
        "a {MINIMUM_REPLAY_WINDOW}-packet window drops the delayed packet"
    );
    assert_eq!(narrow.stats().packets_dropped, 1);
}

/// The same window is settable from a launch line, where it has to reach the
/// context the first packet creates.
#[tokio::test]
async fn the_replay_window_property_reaches_the_context() {
    let (reordered, expected) = reordered_packets(REORDER_DISTANCE + 1).await;
    let mut narrow = decoder(MASTER_KEY_HEX);
    narrow
        .set_property("replay-window-size", window_property(MINIMUM_REPLAY_WINDOW))
        .expect("the minimum is an accepted window");

    let recovered = run_one(&mut narrow, protected_caps(SrtpFlow::Rtp), reordered).await;
    assert_eq!(recovered, expected[..expected.len() - 1].to_vec());
}

/// `allow-repeat-tx` decides whether a re-sent packet is protected again or
/// dropped. libsrtp reuses the key stream, so the bytes have to come out
/// identical.
#[tokio::test]
async fn repeated_transmission_protects_the_same_packet_twice() {
    let packets = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    let mut resent = packets.clone();
    resent.push(packets[0].clone());

    let mut refusing = encoder(MASTER_KEY_HEX);
    let dropped = run_one(&mut refusing, plain_caps(SrtpFlow::Rtp), resent.clone()).await;
    assert_eq!(dropped.len(), packets.len(), "the repeat was dropped");
    assert_eq!(
        refusing.stats(),
        SrtpEncStats {
            packets_protected: packets.len() as u64,
            packets_dropped: 1,
        }
    );

    let mut allowing = encoder(MASTER_KEY_HEX).with_repeat_transmission(true);
    let protected = run_one(&mut allowing, plain_caps(SrtpFlow::Rtp), resent).await;
    assert_eq!(protected.len(), packets.len() + 1);
    assert_eq!(
        protected[0],
        protected[packets.len()],
        "the repeat has to be byte-identical: it reuses the AES-GCM nonce"
    );
    assert_eq!(allowing.stats().packets_dropped, 0);
}

/// The property half of the same switch, for a launch line.
#[tokio::test]
async fn the_repeat_transmission_property_reaches_the_sender() {
    let packets = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    let mut resent = packets.clone();
    resent.push(packets[0].clone());

    let mut element = encoder(MASTER_KEY_HEX);
    element
        .set_property("allow-repeat-tx", PropValue::Bool(true))
        .expect("a boolean property");
    let protected = run_one(&mut element, plain_caps(SrtpFlow::Rtp), resent).await;
    assert_eq!(protected.len(), packets.len() + 1);
    assert_eq!(protected[0], protected[packets.len()]);
}

/// Two keys on the decoder, each tagged with its own MKI, and the sender's MKI
/// picks between them. The MKI is stripped from what the decoder hands on.
#[tokio::test]
async fn an_mki_picks_the_decoder_key_for_rtp_and_rtcp() {
    for flow in [SrtpFlow::Rtp, SrtpFlow::Rtcp] {
        let packets = match flow {
            SrtpFlow::Rtp => rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE),
            SrtpFlow::Rtcp => rtcp_packets(PACKET_COUNT, SYNCHRONIZATION_SOURCE),
        };
        let protected = run_one(
            &mut encoder(SECOND_KEY_HEX)
                .with_mki(SECOND_MKI)
                .expect("a four-byte MKI"),
            plain_caps(flow),
            packets.clone(),
        )
        .await;
        assert_eq!(
            &protected[0][protected[0].len() - SECOND_MKI.len()..],
            SECOND_MKI,
            "the MKI is the last field of a protected {flow:?} packet"
        );

        let recovered = run_one(
            &mut two_key_decoder(),
            protected_caps(flow),
            protected.clone(),
        )
        .await;
        assert_eq!(
            recovered, packets,
            "{flow:?} did not round-trip under an MKI"
        );

        // Re-tagging one packet with an MKI no key claims drops just that one.
        let mut retagged = protected.clone();
        let body = retagged[TAMPERED_PACKET].len() - SECOND_MKI.len();
        retagged[TAMPERED_PACKET].truncate(body);
        retagged[TAMPERED_PACKET].extend_from_slice(UNKNOWN_MKI);
        let mut element = two_key_decoder();
        let recovered = run_one(&mut element, protected_caps(flow), retagged).await;
        let mut expected = packets.clone();
        expected.remove(TAMPERED_PACKET);
        assert_eq!(recovered, expected);
        assert_eq!(element.stats().packets_dropped, 1);
    }
}

/// A decoder keyed with both MKI-tagged keys through its key provider, the only
/// route to more than one key (gst has no property for this either).
fn two_key_decoder() -> SrtpDec {
    let provider = move |_source: u32| {
        let key = |hexadecimal: &str, mki: &[u8]| -> SrtpKeyingMaterial {
            master_key(hexadecimal)
                .with_mki(mki)
                .expect("a four-byte MKI")
                .keying_material(0)
                .expect("valid key material")
        };
        Vec::from([
            key(MASTER_KEY_HEX, FIRST_MKI),
            key(SECOND_KEY_HEX, SECOND_MKI),
        ])
    };
    SrtpDec::default().with_key_provider(Box::new(provider) as Box<dyn SrtpKeyProvider + Send>)
}

/// The MKI travels the whole link, chosen through the `mki` property.
#[tokio::test]
async fn the_mki_property_reaches_the_wire() {
    let packets = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    let mut element = encoder(MASTER_KEY_HEX);
    element
        .set_property("mki", PropValue::Str(FIRST_MKI_HEX.into()))
        .expect("hexadecimal MKI");
    let protected = run_one(&mut element, plain_caps(SrtpFlow::Rtp), packets.clone()).await;

    assert!(protected.iter().all(|packet| packet.ends_with(FIRST_MKI)));
    let recovered = run_one(
        &mut two_key_decoder(),
        protected_caps(SrtpFlow::Rtp),
        protected,
    )
    .await;
    assert_eq!(recovered, packets);
}

/// The counters both elements keep, over a run whose middle packet is damaged
/// after protection.
#[tokio::test]
async fn statistics_count_what_each_element_handled() {
    let packets = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    let mut protector = encoder(MASTER_KEY_HEX);
    let mut protected = run_one(&mut protector, plain_caps(SrtpFlow::Rtp), packets.clone()).await;
    assert_eq!(
        protector.stats(),
        SrtpEncStats {
            packets_protected: packets.len() as u64,
            packets_dropped: 0,
        }
    );

    // One packet no longer authenticates, and one comes from a second source.
    // The `key` property answers for every source, so the second one gets a
    // context of its own and then fails to authenticate under that key.
    *protected[TAMPERED_PACKET]
        .last_mut()
        .expect("a protected packet is never empty") ^= 1;
    let stray_source = !SYNCHRONIZATION_SOURCE;
    protected.push(rtp_packet(0, stray_source, b"another stream"));

    let mut unprotector = decoder(MASTER_KEY_HEX);
    let recovered = run_one(
        &mut unprotector,
        protected_caps(SrtpFlow::Rtp),
        protected.clone(),
    )
    .await;
    assert_eq!(recovered.len(), packets.len() - 1);
    assert_eq!(
        unprotector.stats(),
        SrtpDecStats {
            packets_received: protected.len() as u64,
            packets_dropped: 2,
            streams: Vec::from([
                SrtpStreamStats {
                    synchronization_source: SYNCHRONIZATION_SOURCE,
                    rollover_counter: 0,
                },
                SrtpStreamStats {
                    synchronization_source: stray_source,
                    rollover_counter: 0,
                },
            ]),
        }
    );

    // The counter a stream reports is the live one, not the one it started at.
    let wrapping = rtp_packets(WRAP_START_SEQUENCE, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    let protected = run_one(
        &mut encoder(MASTER_KEY_HEX),
        plain_caps(SrtpFlow::Rtp),
        wrapping,
    )
    .await;
    let mut past_the_wrap = decoder(MASTER_KEY_HEX);
    run_one(&mut past_the_wrap, protected_caps(SrtpFlow::Rtp), protected).await;
    assert_eq!(
        past_the_wrap.stats().streams,
        Vec::from([SrtpStreamStats {
            synchronization_source: SYNCHRONIZATION_SOURCE,
            rollover_counter: 1,
        }])
    );
}

// GStreamer interop for the MKI. gst's `srtpdec` has no `mki` property: an MKI
// key set reaches it only through the sink-pad caps, as `srtp-key` + `mki` and
// then `srtp-key2` + `mki2` for the second one. The caps also have to name the
// stream's `ssrc`, or the element asks its `request-key` signal instead.

/// gst protects with an MKI, `srtpdec` recovers with a two-key context.
#[tokio::test]
#[ignore = "needs gst-launch-1.0 with an AES-GCM libsrtp"]
async fn gst_protects_with_an_mki_and_srtpdec_picks_the_key() {
    let Some(directory) = peer_directory("g2g_m1099_gst_mki_to_g2g") else {
        return;
    };
    run_peer(
        &directory,
        &format!(
            "audiotestsrc num-buffers={PEER_BUFFERS} ! audioconvert ! rtpL16pay \
             ! tee name=t ! queue ! multifilesink location=plain%05d.bin \
             t. ! queue ! srtpenc {} mki={FIRST_MKI_HEX} \
             ! multifilesink location=srtp%05d.bin",
            peer_cipher_arguments()
        ),
    );

    let plain = numbered_files(&directory, "plain");
    let protected = numbered_files(&directory, "srtp");
    assert!(!plain.is_empty(), "gst wrote no packets");
    assert!(
        protected
            .iter()
            .zip(&plain)
            .all(|(protected, plain)| protected.len()
                == plain.len() + AUTHENTICATION_TAG_LENGTH + FIRST_MKI.len()),
        "libsrtp appends the MKI after the AES-GCM tag"
    );

    let recovered = run_one(
        &mut two_key_decoder(),
        protected_caps(SrtpFlow::Rtp),
        protected,
    )
    .await;
    assert_eq!(recovered, plain, "srtpdec did not recover gst's MKI stream");
    println!(
        "gst -> g2g: {} MKI-tagged packets recovered byte-exact",
        plain.len()
    );
}

/// The same for SRTCP, where the MKI sits after the E-flag and SRTCP index word
/// rather than at the end of the ciphertext (RFC 7714 figure 5). gst's `srtpenc`
/// takes a bare RTCP packet on its `rtcp_sink` pad through a caps filter.
#[tokio::test]
#[ignore = "needs gst-launch-1.0 with an AES-GCM libsrtp"]
async fn gst_protects_rtcp_with_an_mki_and_srtpdec_picks_the_key() {
    /// Where the peer's protected RTCP lands, and what it reads.
    const PEER_PLAIN_FILE: &str = "plain.bin";
    const PEER_PROTECTED_FILE: &str = "srtcp.bin";

    let Some(directory) = peer_directory("g2g_m1099_gst_rtcp_mki_to_g2g") else {
        return;
    };
    let plain = rtcp_packets(1, SYNCHRONIZATION_SOURCE);
    std::fs::write(directory.join(PEER_PLAIN_FILE), &plain[0]).expect("write the RTCP packet");
    run_peer(
        &directory,
        &format!(
            "filesrc location={PEER_PLAIN_FILE} ! application/x-rtcp \
             ! srtpenc {} mki={FIRST_MKI_HEX} ! filesink location={PEER_PROTECTED_FILE}",
            peer_cipher_arguments()
        ),
    );

    let protected =
        std::fs::read(directory.join(PEER_PROTECTED_FILE)).expect("read the protected RTCP");
    assert_eq!(
        protected.len(),
        plain[0].len() + AUTHENTICATION_TAG_LENGTH + SRTCP_INDEX_LENGTH + FIRST_MKI.len()
    );
    assert_eq!(
        &protected[protected.len() - FIRST_MKI.len()..],
        FIRST_MKI,
        "libsrtp appends the SRTCP MKI after the index word"
    );

    let recovered = run_one(
        &mut two_key_decoder(),
        protected_caps(SrtpFlow::Rtcp),
        Vec::from([protected]),
    )
    .await;
    assert_eq!(recovered, plain, "srtpdec did not recover gst's SRTCP MKI");
    println!("gst -> g2g: one MKI-tagged SRTCP packet recovered byte-exact");
}

/// `srtpenc` protects with the second MKI, gst recovers with both keys in its
/// sink caps, so libsrtp itself has to pick the one the MKI names.
#[tokio::test]
#[ignore = "needs gst-launch-1.0 with an AES-GCM libsrtp"]
async fn srtpenc_protects_with_an_mki_and_gst_picks_the_key() {
    let Some(directory) = peer_directory("g2g_m1099_g2g_mki_to_gst") else {
        return;
    };
    run_peer(
        &directory,
        &format!(
            "audiotestsrc num-buffers={PEER_BUFFERS} ! audioconvert ! rtpL16pay \
             ! multifilesink location=plain%05d.bin"
        ),
    );

    let plain = numbered_files(&directory, "plain");
    assert!(!plain.is_empty(), "gst wrote no packets");
    let synchronization_source = RtpHeader::parse(&plain[0])
        .expect("gst wrote valid RTP")
        .header
        .ssrc;

    let protected = run_one(
        &mut encoder(SECOND_KEY_HEX)
            .with_mki(SECOND_MKI)
            .expect("a four-byte MKI"),
        plain_caps(SrtpFlow::Rtp),
        plain.clone(),
    )
    .await;
    assert_eq!(protected.len(), plain.len());
    write_numbered_files(&directory, "srtp", &protected);

    run_peer(
        &directory,
        &format!(
            "multifilesrc location=srtp%05d.bin \
             caps=application/x-srtp,ssrc=(uint){synchronization_source},\
srtp-key=(buffer){MASTER_KEY_HEX},mki=(buffer){FIRST_MKI_HEX},\
srtp-key2=(buffer){SECOND_KEY_HEX},mki2=(buffer){SECOND_MKI_HEX},\
srtp-cipher=(string)aes-128-gcm,srtp-auth=(string)null,\
srtcp-cipher=(string)aes-128-gcm,srtcp-auth=(string)null,roc=(uint)0 \
             ! srtpdec ! multifilesink location=recovered%05d.bin"
        ),
    );

    let recovered = numbered_files(&directory, "recovered");
    assert_eq!(recovered, plain, "gst did not recover srtpenc's MKI stream");
    println!(
        "g2g -> gst: {} MKI-tagged packets recovered byte-exact",
        plain.len()
    );
}
