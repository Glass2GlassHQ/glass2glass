//! M1098: the `srtpenc` / `srtpdec` elements, RFC 7714 AES-GCM protection on a
//! pipeline link. Every leg runs the real runner with the elements linked in a
//! graph.
//!
//! ```sh
//! cargo test -p g2g-plugins --features srtp,std --test m1098_srtp
//! ```
//!
//! The `#[ignore]`d legs pair an element with `gst-launch-1.0`, which a g2g to
//! g2g loopback cannot check: both ends would share a wire-format bug. They need
//! a `gst-plugins-bad` whose libsrtp does AES-GCM. Run:
//!
//! ```sh
//! cargo test -p g2g-plugins --features srtp,std --test m1098_srtp -- --ignored --nocapture
//! ```
#![cfg(all(feature = "srtp", feature = "std"))]

use g2g_core::rtp::RtpHeader;
use g2g_core::runtime::{parse_launch, run_graph, run_source_transform_sink};
use g2g_core::{AsyncElement, Bus, BusMessage, G2gError, PropValue};

use g2g_plugins::registry::default_registry;
use g2g_plugins::srtp::{
    KeyUsage, SrtpAuthentication, SrtpCipher, SrtpFlow, SrtpKeyProvider, SrtpPolicy, SrtpSoftLimits,
};
use g2g_plugins::srtpdec::SrtpDec;
use g2g_plugins::srtpenc::SrtpEnc;

mod srtp_common;
use srtp_common::{
    decoder, encoder, master_key, numbered_files, peer_cipher_arguments, peer_decoder_caps,
    peer_directory, plain_caps, protected_caps, rekey, rtcp_packets, rtp_packet, rtp_packets,
    run_one, run_peer, run_protected_link, scratch_path, write_numbered_files, CollectingSink,
    PacketSource, ZeroClock, AES_256_GCM_KEY_HEX, GST_AES_128_GCM, GST_NULL, LINK_CAPACITY,
    MASTER_KEY_HEX, PACKET_COUNT, PEER_BUFFERS, SECOND_KEY_HEX, SECOND_SYNCHRONIZATION_SOURCE,
    SYNCHRONIZATION_SOURCE, WRAP_START_SEQUENCE,
};

/// The replacement a rekey installs.
const REPLACEMENT_KEY_HEX: &str = "0f0e0d0c0b0a09080706050403020100517569642070726f2071756f";

/// Where a rekey leg splits its stream.
const REKEY_AFTER: usize = 4;

#[tokio::test]
async fn rtp_round_trips_byte_exact_across_a_sequence_wrap() {
    let packets = rtp_packets(WRAP_START_SEQUENCE, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    let recovered = run_protected_link(
        SrtpFlow::Rtp,
        &mut encoder(MASTER_KEY_HEX),
        &mut decoder(MASTER_KEY_HEX),
        packets.clone(),
    )
    .await;
    assert_eq!(recovered, packets);

    // The run really crossed the wrap, so the rollover counter advanced.
    let sequences: Vec<u16> = packets
        .iter()
        .map(|packet| RtpHeader::parse(packet).expect("valid RTP").header.sequence)
        .collect();
    assert!(
        sequences.windows(2).any(|pair| pair[1] < pair[0]),
        "{sequences:?} never wraps"
    );
}

#[tokio::test]
async fn a_packet_from_another_source_is_dropped_and_the_stream_continues() {
    let mut packets = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    // One stray packet in the middle, and one frame that is not RTP at all.
    let stray = rtp_packet(0, SECOND_SYNCHRONIZATION_SOURCE, b"another stream");
    let malformed = b"not a packet".to_vec();
    let expected = packets.clone();
    packets.insert(REKEY_AFTER, stray);
    packets.insert(REKEY_AFTER, malformed);

    let recovered = run_protected_link(
        SrtpFlow::Rtp,
        &mut encoder(MASTER_KEY_HEX),
        &mut decoder(MASTER_KEY_HEX),
        packets,
    )
    .await;
    assert_eq!(recovered, expected);
}

#[tokio::test]
async fn a_missing_key_refuses_to_configure() {
    let mut source = PacketSource::new(
        rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE),
        plain_caps(SrtpFlow::Rtp),
    );
    let mut unkeyed = SrtpEnc::default();
    let mut sink = CollectingSink::default();
    assert_eq!(
        run_source_transform_sink(
            &mut source,
            &mut unkeyed,
            &mut sink,
            &ZeroClock,
            LINK_CAPACITY
        )
        .await,
        Err(G2gError::NotConfigured)
    );
    assert!(sink.packets.is_empty());
}

/// Reaching the soft limit posts one bus `Info`, the gst `soft-limit` signal's
/// analog, and the stream keeps flowing: only the RFC hard limit stops it.
#[tokio::test]
async fn the_soft_limit_posts_one_notice_and_the_stream_continues() {
    /// Packets this key may protect before a replacement is due.
    const SOFT_LIMIT_PACKETS: u64 = 2;

    let (bus, handle) = Bus::new(PACKET_COUNT);
    let packets = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    let mut encoder = encoder(MASTER_KEY_HEX)
        .with_bus(handle)
        .with_soft_limits(SrtpSoftLimits {
            srtp_packets: SOFT_LIMIT_PACKETS,
            ..SrtpSoftLimits::default()
        });
    let mut source = PacketSource::new(packets.clone(), plain_caps(SrtpFlow::Rtp));
    let mut sink = CollectingSink::default();
    run_source_transform_sink(
        &mut source,
        &mut encoder,
        &mut sink,
        &ZeroClock,
        LINK_CAPACITY,
    )
    .await
    .expect("the pipeline runs past the soft limit");

    assert_eq!(sink.packets.len(), packets.len());
    assert_eq!(
        encoder.key_usage().map(|usage| usage.srtp),
        Some(KeyUsage::SoftLimitReached)
    );
    let notices: Vec<BusMessage> = std::iter::from_fn(|| bus.try_recv())
        .filter(|message| matches!(message, BusMessage::Info(_)))
        .collect();
    assert_eq!(notices.len(), 1, "{notices:?}");
}

#[tokio::test]
async fn the_aes_256_profile_follows_the_key_length() {
    let packets = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    assert_eq!(
        master_key(AES_256_GCM_KEY_HEX).policy(),
        SrtpPolicy {
            cipher: SrtpCipher::Aes256Gcm,
            authentication: SrtpAuthentication::Null,
        }
    );
    let recovered = run_protected_link(
        SrtpFlow::Rtp,
        &mut encoder(AES_256_GCM_KEY_HEX),
        &mut decoder(AES_256_GCM_KEY_HEX),
        packets.clone(),
    )
    .await;
    assert_eq!(recovered, packets);
}

#[tokio::test]
async fn rtcp_round_trips_byte_exact_encrypted_and_authenticated_only() {
    /// Where an RTCP packet's body starts: the two authenticated header words.
    const RTCP_BODY_OFFSET: usize = 8;

    for encrypt in [true, false] {
        let packets = rtcp_packets(PACKET_COUNT, SYNCHRONIZATION_SOURCE);
        let recovered = run_protected_link(
            SrtpFlow::Rtcp,
            &mut encoder(MASTER_KEY_HEX).with_rtcp_encryption(encrypt),
            &mut decoder(MASTER_KEY_HEX),
            packets.clone(),
        )
        .await;
        assert_eq!(recovered, packets, "rtcp-encrypt={encrypt}");

        // The switch has to reach the wire: only the encrypting run hides the
        // body an authentication-only run leaves in the clear.
        let protected = run_one(
            &mut encoder(MASTER_KEY_HEX).with_rtcp_encryption(encrypt),
            plain_caps(SrtpFlow::Rtcp),
            packets.clone(),
        )
        .await;
        let plain_body = &packets[0][RTCP_BODY_OFFSET..];
        let protected_body = &protected[0][RTCP_BODY_OFFSET..packets[0].len()];
        assert_eq!(
            protected_body == plain_body,
            !encrypt,
            "rtcp-encrypt={encrypt} did not reach the wire"
        );
    }
}

#[tokio::test]
async fn a_tampered_packet_is_dropped_and_the_stream_continues() {
    /// The packet the leg damages, and the byte of it that is flipped: one past
    /// the RTP header, so the ciphertext and not the authenticated header
    /// changes.
    const DAMAGED_PACKET: usize = 2;

    let packets = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    let mut protected = run_one(
        &mut encoder(MASTER_KEY_HEX),
        plain_caps(SrtpFlow::Rtp),
        packets.clone(),
    )
    .await;
    let header_length = RtpHeader::parse(&packets[DAMAGED_PACKET])
        .expect("valid RTP")
        .payload_offset;
    protected[DAMAGED_PACKET][header_length] ^= 1;

    let recovered = run_one(
        &mut decoder(MASTER_KEY_HEX),
        protected_caps(SrtpFlow::Rtp),
        protected,
    )
    .await;
    let expected: Vec<Vec<u8>> = packets
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != DAMAGED_PACKET)
        .map(|(_, packet)| packet.clone())
        .collect();
    assert_eq!(recovered, expected);
}

#[tokio::test]
async fn a_replayed_packet_is_dropped() {
    let packets = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    let protected = run_one(
        &mut encoder(MASTER_KEY_HEX),
        plain_caps(SrtpFlow::Rtp),
        packets.clone(),
    )
    .await;

    let mut replayed = protected.clone();
    replayed.push(protected[0].clone());
    let recovered = run_one(
        &mut decoder(MASTER_KEY_HEX),
        protected_caps(SrtpFlow::Rtp),
        replayed,
    )
    .await;
    assert_eq!(recovered, packets, "the repeat was not delivered twice");
}

#[tokio::test]
async fn a_mid_stream_rekey_keeps_the_stream_going() {
    let packets = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    let (first, second) = packets.split_at(REKEY_AFTER);

    let mut encoder = encoder(MASTER_KEY_HEX);
    let mut decoder = decoder(MASTER_KEY_HEX);
    let before = run_one(&mut encoder, plain_caps(SrtpFlow::Rtp), first.to_vec()).await;
    assert_eq!(
        run_one(&mut decoder, protected_caps(SrtpFlow::Rtp), before).await,
        first
    );

    let (profile, key, salt) = rekey(REPLACEMENT_KEY_HEX);
    encoder
        .replace_key(profile, &key, &salt)
        .expect("re-key the encoder");
    decoder
        .replace_key(profile, &key, &salt)
        .expect("re-key the decoder");

    // The same instances carry on, their packet indices continued across the key change.
    let after = run_one(&mut encoder, plain_caps(SrtpFlow::Rtp), second.to_vec()).await;
    assert_eq!(
        run_one(&mut decoder, protected_caps(SrtpFlow::Rtp), after.clone()).await,
        second
    );

    // A decoder still holding the old key recovers none of it.
    let mut stale = self::decoder(MASTER_KEY_HEX);
    assert!(run_one(&mut stale, protected_caps(SrtpFlow::Rtp), after)
        .await
        .is_empty());
}

#[tokio::test]
async fn setting_the_decoder_key_property_rekeys_an_existing_context() {
    let packets = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    let (first, second) = packets.split_at(REKEY_AFTER);

    let mut encoder = encoder(MASTER_KEY_HEX);
    let mut decoder = decoder(MASTER_KEY_HEX);
    let before = run_one(&mut encoder, plain_caps(SrtpFlow::Rtp), first.to_vec()).await;
    // The context for this source now exists, so the property has to reach it.
    assert_eq!(
        run_one(&mut decoder, protected_caps(SrtpFlow::Rtp), before).await,
        first
    );

    encoder
        .set_property("key", PropValue::Str(REPLACEMENT_KEY_HEX.into()))
        .expect("the encoder takes the replacement key");
    decoder
        .set_property("key", PropValue::Str(REPLACEMENT_KEY_HEX.into()))
        .expect("the decoder takes the replacement key");

    let after = run_one(&mut encoder, plain_caps(SrtpFlow::Rtp), second.to_vec()).await;
    assert_eq!(
        run_one(&mut decoder, protected_caps(SrtpFlow::Rtp), after).await,
        second
    );
}

#[tokio::test]
async fn a_key_provider_serves_two_sources_with_different_keys() {
    let first = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    let second = rtp_packets(0, PACKET_COUNT, SECOND_SYNCHRONIZATION_SOURCE);
    let first_protected = run_one(
        &mut encoder(MASTER_KEY_HEX),
        plain_caps(SrtpFlow::Rtp),
        first.clone(),
    )
    .await;
    let second_protected = run_one(
        &mut encoder(SECOND_KEY_HEX),
        plain_caps(SrtpFlow::Rtp),
        second.clone(),
    )
    .await;

    let mut interleaved = Vec::new();
    let mut expected = Vec::new();
    for index in 0..PACKET_COUNT {
        interleaved.push(first_protected[index].clone());
        interleaved.push(second_protected[index].clone());
        expected.push(first[index].clone());
        expected.push(second[index].clone());
    }

    let provider = move |source: u32| {
        let hexadecimal = match source {
            SYNCHRONIZATION_SOURCE => MASTER_KEY_HEX,
            SECOND_SYNCHRONIZATION_SOURCE => SECOND_KEY_HEX,
            _ => return Vec::new(),
        };
        let key = master_key(hexadecimal);
        key.keying_material(0).into_iter().collect()
    };
    let mut decoder =
        SrtpDec::default().with_key_provider(Box::new(provider) as Box<dyn SrtpKeyProvider + Send>);
    let recovered = run_one(&mut decoder, protected_caps(SrtpFlow::Rtp), interleaved).await;
    assert_eq!(recovered, expected);
}

#[tokio::test]
async fn the_roc_property_joins_a_stream_past_a_wrap() {
    /// The sender's first packet, so its next one wraps the sequence and moves
    /// the stream onto rollover counter 1.
    const LAST_SEQUENCE_OF_ROLLOVER_ZERO: u16 = u16::MAX;
    const ROLLOVER_COUNTER_AFTER_THE_WRAP: u64 = 1;

    let mut sent = vec![rtp_packet(
        LAST_SEQUENCE_OF_ROLLOVER_ZERO,
        SYNCHRONIZATION_SOURCE,
        b"the packet before the wrap",
    )];
    sent.extend(rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE));
    let protected = run_one(
        &mut encoder(MASTER_KEY_HEX),
        plain_caps(SrtpFlow::Rtp),
        sent.clone(),
    )
    .await;

    // A receiver joining here never saw the wrap, so only the `roc` property
    // tells it which rollover counter these packets belong to.
    let joined: Vec<Vec<u8>> = protected[1..].to_vec();
    let expected: Vec<Vec<u8>> = sent[1..].to_vec();

    let mut ready = decoder(MASTER_KEY_HEX);
    ready
        .set_property("roc", PropValue::Uint(ROLLOVER_COUNTER_AFTER_THE_WRAP))
        .expect("the decoder takes a rollover counter");
    assert_eq!(
        run_one(&mut ready, protected_caps(SrtpFlow::Rtp), joined.clone()).await,
        expected
    );

    // The default rollover counter authenticates none of them.
    let mut from_zero = decoder(MASTER_KEY_HEX);
    assert_eq!(
        from_zero.get_property("roc"),
        Some(PropValue::Uint(0)),
        "the default is the start of the stream"
    );
    assert!(
        run_one(&mut from_zero, protected_caps(SrtpFlow::Rtp), joined)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn a_launch_line_protects_and_recovers_a_file_of_packets() {
    let packets = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    let packet_length = packets[0].len();
    assert!(
        packets.iter().all(|packet| packet.len() == packet_length),
        "one blocksize has to frame every packet"
    );
    let sent: Vec<u8> = packets.concat();

    let source_path = scratch_path("g2g_m1098_launch_in.rtp");
    let sink_path = scratch_path("g2g_m1098_launch_out.rtp");
    std::fs::write(&source_path, &sent).expect("write the packets");

    let registry = default_registry();
    let graph = parse_launch(
        &registry,
        &format!(
            "filesrc location={} bytestream-format=rtp blocksize={packet_length} \
             ! srtpenc key={MASTER_KEY_HEX} ! srtpdec key={MASTER_KEY_HEX} \
             ! filesink location={}",
            source_path.display(),
            sink_path.display()
        ),
    )
    .expect("the launch line parses");
    run_graph(graph, &ZeroClock, LINK_CAPACITY)
        .await
        .expect("the launch pipeline runs");

    assert_eq!(
        std::fs::read(&sink_path).expect("read the recovered packets"),
        sent
    );
}

/// gst protects, `srtpdec` recovers: the packets g2g hands back must equal the
/// ones gst protected.
#[tokio::test]
#[ignore = "needs gst-launch-1.0 with an AES-GCM libsrtp"]
async fn gst_protects_and_srtpdec_recovers() {
    let Some(directory) = peer_directory("g2g_m1098_gst_to_g2g") else {
        return;
    };
    run_peer(
        &directory,
        &format!(
            "audiotestsrc num-buffers={PEER_BUFFERS} ! audioconvert ! rtpL16pay \
             ! tee name=t ! queue ! multifilesink location=plain%05d.bin \
             t. ! queue ! srtpenc {} ! multifilesink location=srtp%05d.bin",
            peer_cipher_arguments()
        ),
    );

    let plain = numbered_files(&directory, "plain");
    let protected = numbered_files(&directory, "srtp");
    assert_eq!(plain.len(), protected.len());
    assert!(!plain.is_empty(), "gst wrote no packets");

    let recovered = run_one(
        &mut decoder(MASTER_KEY_HEX),
        protected_caps(SrtpFlow::Rtp),
        protected,
    )
    .await;
    assert_eq!(recovered, plain, "srtpdec did not recover gst's SRTP");
    println!("gst -> g2g: {} packets recovered byte-exact", plain.len());
}

/// `srtpenc` protects, gst recovers. gst's `srtpdec` has no key property: it
/// takes the key from its sink-pad caps, which must also carry the stream's
/// `ssrc` (without it the element asks its `request-key` signal instead and
/// drops every packet).
#[tokio::test]
#[ignore = "needs gst-launch-1.0 with an AES-GCM libsrtp"]
async fn srtpenc_protects_and_gst_recovers() {
    let Some(directory) = peer_directory("g2g_m1098_g2g_to_gst") else {
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
        &mut encoder(MASTER_KEY_HEX),
        plain_caps(SrtpFlow::Rtp),
        plain.clone(),
    )
    .await;
    assert_eq!(protected.len(), plain.len());
    write_numbered_files(&directory, "srtp", &protected);

    run_peer(
        &directory,
        &format!(
            "multifilesrc location=srtp%05d.bin caps={} \
             ! srtpdec ! multifilesink location=recovered%05d.bin",
            peer_decoder_caps(
                SrtpFlow::Rtp,
                synchronization_source,
                MASTER_KEY_HEX,
                GST_AES_128_GCM,
                GST_NULL
            )
        ),
    );

    let recovered = numbered_files(&directory, "recovered");
    assert_eq!(recovered, plain, "gst did not recover srtpenc's SRTP");
    println!("g2g -> gst: {} packets recovered byte-exact", plain.len());
}
