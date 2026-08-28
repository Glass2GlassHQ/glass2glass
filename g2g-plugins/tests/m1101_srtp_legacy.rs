//! M1101: the RFC 3711 protection policies on `srtpenc` / `srtpdec`, so a peer
//! that never negotiates RFC 7714 still works: AES counter mode, HMAC-SHA1 at
//! both tag lengths, and the NULL cipher.
//!
//! ```sh
//! cargo test -p g2g-plugins --features srtp,std --test m1101_srtp_legacy
//! ```
//!
//! The `#[ignore]`d legs pair an element with `gst-launch-1.0`'s libsrtp, which
//! a g2g to g2g loopback cannot check: both ends would share a wire-format bug.
//! Run:
//!
//! ```sh
//! cargo test -p g2g-plugins --features srtp,std --test m1101_srtp_legacy \
//!     -- --ignored --nocapture
//! ```
#![cfg(all(feature = "srtp", feature = "std"))]

use g2g_core::rtp::RtpHeader;
use g2g_core::{AsyncElement, PropValue};

use g2g_plugins::srtp::{
    SrtpAuthentication, SrtpCipher, SrtpFlow, SrtpKeyProvider, SrtpMasterKey,
    HMAC_SHA1_32_TAG_LENGTH, HMAC_SHA1_80_TAG_LENGTH, SRTCP_INDEX_LENGTH,
};
use g2g_plugins::srtpdec::SrtpDec;
use g2g_plugins::srtpenc::SrtpEnc;

mod srtp_common;
use srtp_common::{
    numbered_files, peer_arguments, peer_decoder_caps, peer_directory, plain_caps, protected_caps,
    rtcp_packets, rtp_packets, run_one, run_peer, run_protected_link, write_numbered_files,
    COUNTER_MODE_256_KEY_HEX, COUNTER_MODE_KEY_HEX, GST_AES_128_COUNTER_MODE,
    GST_AES_256_COUNTER_MODE, GST_HMAC_SHA1_32, GST_HMAC_SHA1_80, GST_NULL, MASTER_KEY_HEX,
    PACKET_COUNT, PEER_BUFFERS, SYNCHRONIZATION_SOURCE,
};

/// The MKI an interop leg tags its packets with, and the same value as the
/// hexadecimal digits a property carries.
const MKI: &[u8] = &[0xa1, 0xb2, 0xc3, 0xd4];
const MKI_HEX: &str = "a1b2c3d4";

/// The E-flag of the SRTCP index word, set only when the body is encrypted.
const SRTCP_ENCRYPTION_FLAG: u32 = 1 << 31;
/// Where an RTCP packet's body starts: the two authenticated header words.
const RTCP_BODY_OFFSET: usize = 8;

/// Every RFC 3711 cipher with the key length that keys it, and the property
/// value gst spells for it.
const CIPHERS: &[(&str, &str, SrtpCipher)] = &[
    (
        COUNTER_MODE_KEY_HEX,
        GST_AES_128_COUNTER_MODE,
        SrtpCipher::Aes128CounterMode,
    ),
    (
        COUNTER_MODE_256_KEY_HEX,
        GST_AES_256_COUNTER_MODE,
        SrtpCipher::Aes256CounterMode,
    ),
    (COUNTER_MODE_KEY_HEX, GST_NULL, SrtpCipher::Null),
];

/// Every authentication transform with the tag it appends.
const AUTHENTICATIONS: &[(&str, usize, SrtpAuthentication)] = &[
    (
        GST_HMAC_SHA1_80,
        HMAC_SHA1_80_TAG_LENGTH,
        SrtpAuthentication::HmacSha1Tag80,
    ),
    (
        GST_HMAC_SHA1_32,
        HMAC_SHA1_32_TAG_LENGTH,
        SrtpAuthentication::HmacSha1Tag32,
    ),
    (GST_NULL, 0, SrtpAuthentication::Null),
];

fn cipher_property(flow: SrtpFlow) -> &'static str {
    match flow {
        SrtpFlow::Rtp => "rtp-cipher",
        SrtpFlow::Rtcp => "rtcp-cipher",
    }
}

fn authentication_property(flow: SrtpFlow) -> &'static str {
    match flow {
        SrtpFlow::Rtp => "rtp-auth",
        SrtpFlow::Rtcp => "rtcp-auth",
    }
}

/// Set the key and this flow's protection pair the way a launch line does, so
/// the legs exercise the property path rather than a constructor.
fn configure<E: AsyncElement>(
    element: &mut E,
    key: &str,
    flow: SrtpFlow,
    cipher: &str,
    authentication: &str,
) {
    element
        .set_property("key", PropValue::Str(key.into()))
        .expect("the element takes the key");
    element
        .set_property(cipher_property(flow), PropValue::Str(cipher.into()))
        .expect("the element takes the cipher");
    element
        .set_property(
            authentication_property(flow),
            PropValue::Str(authentication.into()),
        )
        .expect("the element takes the authentication");
}

fn encoder_with(key: &str, flow: SrtpFlow, cipher: &str, authentication: &str) -> SrtpEnc {
    let mut element = SrtpEnc::default();
    configure(&mut element, key, flow, cipher, authentication);
    element
}

fn decoder_with(key: &str, flow: SrtpFlow, cipher: &str, authentication: &str) -> SrtpDec {
    let mut element = SrtpDec::default();
    configure(&mut element, key, flow, cipher, authentication);
    element
}

/// An `srtpdec` keyed by one MKI-tagged counter-mode key. gst's `srtpdec` has
/// no `mki` property, and g2g's takes an MKI only through a key provider.
fn mki_decoder() -> SrtpDec {
    let key = SrtpMasterKey::from_hexadecimal(COUNTER_MODE_KEY_HEX)
        .expect("the constant is valid key material")
        .with_mki(MKI)
        .expect("a four-byte MKI");
    let provider = move |_source: u32| key.keying_material(0).into_iter().collect();
    SrtpDec::default().with_key_provider(Box::new(provider) as Box<dyn SrtpKeyProvider + Send>)
}

/// Every cipher crossed with every authentication transform round trips
/// byte-exact on the real runner, for both flows.
#[tokio::test]
async fn every_rfc_3711_policy_round_trips_rtp_and_rtcp() {
    for (key, cipher, _) in CIPHERS {
        for (authentication, _, _) in AUTHENTICATIONS {
            let packets = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
            let recovered = run_protected_link(
                SrtpFlow::Rtp,
                &mut encoder_with(key, SrtpFlow::Rtp, cipher, authentication),
                &mut decoder_with(key, SrtpFlow::Rtp, cipher, authentication),
                packets.clone(),
            )
            .await;
            assert_eq!(recovered, packets, "rtp {cipher} + {authentication}");

            let reports = rtcp_packets(PACKET_COUNT, SYNCHRONIZATION_SOURCE);
            let recovered = run_protected_link(
                SrtpFlow::Rtcp,
                &mut encoder_with(key, SrtpFlow::Rtcp, cipher, authentication),
                &mut decoder_with(key, SrtpFlow::Rtcp, cipher, authentication),
                reports.clone(),
            )
            .await;
            assert_eq!(recovered, reports, "rtcp {cipher} + {authentication}");
        }
    }
}

/// The wire shows what the policy asked for: the payload is hidden unless the
/// cipher is NULL, and the packet grows by exactly the declared tag.
#[tokio::test]
async fn the_policy_reaches_the_wire() {
    let packets = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    let header_length = RtpHeader::parse(&packets[0])
        .expect("valid RTP")
        .payload_offset;

    for (key, cipher, kind) in CIPHERS {
        for (authentication, tag_length, _) in AUTHENTICATIONS {
            let protected = run_one(
                &mut encoder_with(key, SrtpFlow::Rtp, cipher, authentication),
                plain_caps(SrtpFlow::Rtp),
                packets.clone(),
            )
            .await;
            assert_eq!(
                protected[0].len(),
                packets[0].len() + tag_length,
                "{cipher} + {authentication} packet growth"
            );
            assert_eq!(
                protected[0][header_length..packets[0].len()] == packets[0][header_length..],
                *kind == SrtpCipher::Null,
                "{cipher} + {authentication} encrypted the payload"
            );
            assert_eq!(
                protected[0][..header_length],
                packets[0][..header_length],
                "the RTP header stays in the clear"
            );
        }
    }
}

/// A flipped byte is caught at both tag lengths and the stream carries on, and
/// with no authentication the same packet is handed on unchecked.
#[tokio::test]
async fn hmac_sha1_drops_a_tampered_packet_at_both_tag_lengths() {
    /// The packet a leg damages, and the byte of it: one past the RTP header,
    /// so the ciphertext and not the authenticated header changes.
    const DAMAGED_PACKET: usize = 2;

    let packets = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    let header_length = RtpHeader::parse(&packets[DAMAGED_PACKET])
        .expect("valid RTP")
        .payload_offset;

    for (authentication, tag_length, _) in AUTHENTICATIONS {
        let mut protected = run_one(
            &mut encoder_with(
                COUNTER_MODE_KEY_HEX,
                SrtpFlow::Rtp,
                GST_AES_128_COUNTER_MODE,
                authentication,
            ),
            plain_caps(SrtpFlow::Rtp),
            packets.clone(),
        )
        .await;
        protected[DAMAGED_PACKET][header_length] ^= 1;

        let mut unprotector = decoder_with(
            COUNTER_MODE_KEY_HEX,
            SrtpFlow::Rtp,
            GST_AES_128_COUNTER_MODE,
            authentication,
        );
        let recovered = run_one(&mut unprotector, protected_caps(SrtpFlow::Rtp), protected).await;
        if *tag_length == 0 {
            assert_eq!(
                recovered.len(),
                packets.len(),
                "with no tag every packet is handed on, forged or not"
            );
            assert_ne!(
                recovered[DAMAGED_PACKET], packets[DAMAGED_PACKET],
                "the forged payload reached the sink unaltered"
            );
            continue;
        }
        let expected: Vec<Vec<u8>> = packets
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != DAMAGED_PACKET)
            .map(|(_, packet)| packet.clone())
            .collect();
        assert_eq!(recovered, expected, "{authentication}");
        assert_eq!(unprotector.stats().packets_dropped, 1, "{authentication}");
    }
}

/// `rtcp-encrypt=false` under a counter-mode cipher: the E flag is 0 and the
/// body is in the clear, but the packet is still authenticated, so a forged one
/// is dropped.
#[tokio::test]
async fn an_unencrypted_srtcp_body_is_still_authenticated() {
    let reports = rtcp_packets(PACKET_COUNT, SYNCHRONIZATION_SOURCE);

    for encrypt in [true, false] {
        let mut protector = encoder_with(
            COUNTER_MODE_KEY_HEX,
            SrtpFlow::Rtcp,
            GST_AES_128_COUNTER_MODE,
            GST_HMAC_SHA1_80,
        );
        protector
            .set_property("rtcp-encrypt", PropValue::Bool(encrypt))
            .expect("the element takes rtcp-encrypt");
        let mut protected =
            run_one(&mut protector, plain_caps(SrtpFlow::Rtcp), reports.clone()).await;

        let plain = &reports[0];
        assert_eq!(
            protected[0].len(),
            plain.len() + SRTCP_INDEX_LENGTH + HMAC_SHA1_80_TAG_LENGTH
        );
        let index_word = u32::from_be_bytes(
            protected[0][plain.len()..plain.len() + SRTCP_INDEX_LENGTH]
                .try_into()
                .expect("the index word"),
        );
        assert_eq!(
            index_word & SRTCP_ENCRYPTION_FLAG != 0,
            encrypt,
            "rtcp-encrypt={encrypt} did not reach the E flag"
        );
        assert_eq!(
            protected[0][RTCP_BODY_OFFSET..plain.len()] == plain[RTCP_BODY_OFFSET..],
            !encrypt,
            "rtcp-encrypt={encrypt} did not reach the body"
        );

        let mut unprotector = decoder_with(
            COUNTER_MODE_KEY_HEX,
            SrtpFlow::Rtcp,
            GST_AES_128_COUNTER_MODE,
            GST_HMAC_SHA1_80,
        );
        assert_eq!(
            run_one(
                &mut unprotector,
                protected_caps(SrtpFlow::Rtcp),
                protected.clone()
            )
            .await,
            reports,
            "rtcp-encrypt={encrypt}"
        );

        // The tag covers the body whether it was encrypted or not.
        protected[0][RTCP_BODY_OFFSET] ^= 1;
        let mut unprotector = decoder_with(
            COUNTER_MODE_KEY_HEX,
            SrtpFlow::Rtcp,
            GST_AES_128_COUNTER_MODE,
            GST_HMAC_SHA1_80,
        );
        let recovered = run_one(&mut unprotector, protected_caps(SrtpFlow::Rtcp), protected).await;
        assert_eq!(recovered.len(), reports.len() - 1, "rtcp-encrypt={encrypt}");
    }
}

/// Under the RFC 3711 profiles the MKI sits between the protected body and the
/// authentication tag, not after it, and the decoder finds it there.
#[tokio::test]
async fn an_mki_sits_before_the_tag_under_a_counter_mode_policy() {
    let packets = rtp_packets(0, PACKET_COUNT, SYNCHRONIZATION_SOURCE);
    let mut protector = encoder_with(
        COUNTER_MODE_KEY_HEX,
        SrtpFlow::Rtp,
        GST_AES_128_COUNTER_MODE,
        GST_HMAC_SHA1_80,
    );
    protector
        .set_property("mki", PropValue::Str(MKI_HEX.into()))
        .expect("the element takes the MKI");
    let protected = run_one(&mut protector, plain_caps(SrtpFlow::Rtp), packets.clone()).await;

    assert_eq!(
        protected[0].len(),
        packets[0].len() + MKI.len() + HMAC_SHA1_80_TAG_LENGTH
    );
    let mki_end = protected[0].len() - HMAC_SHA1_80_TAG_LENGTH;
    assert_eq!(
        &protected[0][mki_end - MKI.len()..mki_end],
        MKI,
        "the MKI is between the ciphertext and the tag"
    );

    let mut unprotector = mki_decoder();
    assert_eq!(
        run_one(&mut unprotector, protected_caps(SrtpFlow::Rtp), protected).await,
        packets
    );
}

/// The property rules: the cipher and authentication follow the key length
/// unless a value names them, a contradiction is refused, and each flow's pair
/// is read on an instance carrying that flow.
#[test]
fn the_protection_properties_resolve_per_flow() {
    let mut element = SrtpEnc::default();
    let read = |element: &SrtpEnc, name: &str| element.get_property(name);

    // No key yet: the gst defaults.
    assert_eq!(
        read(&element, "rtp-cipher"),
        Some(PropValue::Str(GST_AES_128_COUNTER_MODE.into()))
    );
    assert_eq!(
        read(&element, "rtcp-auth"),
        Some(PropValue::Str(GST_HMAC_SHA1_80.into()))
    );

    // An AES-GCM key length moves both flows, and turns the authentication off.
    element
        .set_property("key", PropValue::Str(MASTER_KEY_HEX.into()))
        .unwrap();
    for name in ["rtp-cipher", "rtcp-cipher"] {
        assert_eq!(
            read(&element, name),
            Some(PropValue::Str("aes-128-gcm".into())),
            "{name}"
        );
    }
    for name in ["rtp-auth", "rtcp-auth"] {
        assert_eq!(
            read(&element, name),
            Some(PropValue::Str(GST_NULL.into())),
            "{name}"
        );
    }
    // An AEAD cipher carries its own tag, so a second transform is refused.
    assert!(element
        .set_property("rtp-auth", PropValue::Str(GST_HMAC_SHA1_80.into()))
        .is_err());
    // And a cipher this key length cannot key.
    assert!(element
        .set_property(
            "rtp-cipher",
            PropValue::Str(GST_AES_128_COUNTER_MODE.into())
        )
        .is_err());

    // A counter-mode key length moves them back.
    element
        .set_property("key", PropValue::Str(COUNTER_MODE_KEY_HEX.into()))
        .unwrap();
    assert_eq!(
        read(&element, "rtp-cipher"),
        Some(PropValue::Str(GST_AES_128_COUNTER_MODE.into()))
    );
    assert_eq!(
        read(&element, "rtp-auth"),
        Some(PropValue::Str(GST_HMAC_SHA1_80.into()))
    );
    // Only the flow that was set moves.
    element
        .set_property("rtcp-auth", PropValue::Str(GST_HMAC_SHA1_32.into()))
        .unwrap();
    assert_eq!(
        read(&element, "rtp-auth"),
        Some(PropValue::Str(GST_HMAC_SHA1_80.into()))
    );
    assert_eq!(
        read(&element, "rtcp-auth"),
        Some(PropValue::Str(GST_HMAC_SHA1_32.into()))
    );
    // The NULL cipher takes either counter-mode key length.
    element
        .set_property("rtp-cipher", PropValue::Str(GST_NULL.into()))
        .unwrap();
    element
        .set_property("key", PropValue::Str(COUNTER_MODE_256_KEY_HEX.into()))
        .unwrap();
    assert_eq!(
        read(&element, "rtp-cipher"),
        Some(PropValue::Str(GST_NULL.into()))
    );
    // An AES-256 cipher needs the 46-byte key, and refuses the 30-byte one.
    element
        .set_property(
            "rtp-cipher",
            PropValue::Str(GST_AES_256_COUNTER_MODE.into()),
        )
        .unwrap();
    assert!(element
        .set_property("key", PropValue::Str(COUNTER_MODE_KEY_HEX.into()))
        .is_err());
    // A value neither enum names.
    assert!(element
        .set_property("rtp-cipher", PropValue::Str("aes-192-icm".into()))
        .is_err());
    assert!(element
        .set_property("rtp-auth", PropValue::Str("hmac-sha256-80".into()))
        .is_err());
}

// GStreamer interop. A loopback between two g2g ends cannot catch a wire-format
// bug, so each leg puts `gst-launch-1.0`'s libsrtp on the other side. Every
// combination runs in both directions.

/// The combinations both stacks can key, spelled the way gst spells them.
const INTEROP_POLICIES: &[(&str, &str, &str)] = &[
    (
        COUNTER_MODE_KEY_HEX,
        GST_AES_128_COUNTER_MODE,
        GST_HMAC_SHA1_80,
    ),
    (
        COUNTER_MODE_KEY_HEX,
        GST_AES_128_COUNTER_MODE,
        GST_HMAC_SHA1_32,
    ),
    (
        COUNTER_MODE_256_KEY_HEX,
        GST_AES_256_COUNTER_MODE,
        GST_HMAC_SHA1_80,
    ),
    (COUNTER_MODE_KEY_HEX, GST_NULL, GST_HMAC_SHA1_80),
];

/// gst protects, `srtpdec` recovers: the packets g2g hands back must equal the
/// ones gst protected, under every RFC 3711 policy.
#[tokio::test]
#[ignore = "needs gst-launch-1.0 with libsrtp"]
async fn gst_protects_and_srtpdec_recovers_every_legacy_policy() {
    const LEG_FLOW: SrtpFlow = SrtpFlow::Rtp;

    for (key, cipher, authentication) in INTEROP_POLICIES {
        let Some(directory) =
            peer_directory(&format!("g2g_m1101_gst_to_g2g_{cipher}_{authentication}"))
        else {
            return;
        };
        run_peer(
            &directory,
            &format!(
                "audiotestsrc num-buffers={PEER_BUFFERS} ! audioconvert ! rtpL16pay \
                 ! tee name=t ! queue ! multifilesink location=plain%05d.bin \
                 t. ! queue ! srtpenc {} ! multifilesink location=srtp%05d.bin",
                peer_arguments(key, LEG_FLOW, cipher, authentication)
            ),
        );

        let plain = numbered_files(&directory, "plain");
        let protected = numbered_files(&directory, "srtp");
        assert_eq!(plain.len(), protected.len());
        assert!(!plain.is_empty(), "gst wrote no packets");

        let recovered = run_one(
            &mut decoder_with(key, SrtpFlow::Rtp, cipher, authentication),
            protected_caps(SrtpFlow::Rtp),
            protected,
        )
        .await;
        assert_eq!(
            recovered, plain,
            "srtpdec did not recover gst's {cipher} + {authentication} SRTP"
        );
        println!(
            "gst -> g2g: {} packets recovered byte-exact, {cipher} + {authentication}",
            plain.len()
        );
    }
}

/// `srtpenc` protects, gst recovers, under every RFC 3711 policy.
#[tokio::test]
#[ignore = "needs gst-launch-1.0 with libsrtp"]
async fn srtpenc_protects_and_gst_recovers_every_legacy_policy() {
    const LEG_FLOW: SrtpFlow = SrtpFlow::Rtp;

    for (key, cipher, authentication) in INTEROP_POLICIES {
        let Some(directory) =
            peer_directory(&format!("g2g_m1101_g2g_to_gst_{cipher}_{authentication}"))
        else {
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
            &mut encoder_with(key, SrtpFlow::Rtp, cipher, authentication),
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
                    LEG_FLOW,
                    synchronization_source,
                    key,
                    cipher,
                    authentication
                )
            ),
        );

        let recovered = numbered_files(&directory, "recovered");
        assert_eq!(
            recovered, plain,
            "gst did not recover srtpenc's {cipher} + {authentication} SRTP"
        );
        println!(
            "g2g -> gst: {} packets recovered byte-exact, {cipher} + {authentication}",
            plain.len()
        );
    }
}

/// The SRTCP direction: gst protects one RTCP packet under each policy and
/// `srtpdec` recovers it, so the E flag, index word and tag placement match.
#[tokio::test]
#[ignore = "needs gst-launch-1.0 with libsrtp"]
async fn gst_protects_rtcp_and_srtpdec_recovers_every_legacy_policy() {
    /// Where the peer's plain RTCP is written, and where it writes the result.
    const PEER_PLAIN_FILE: &str = "plain.bin";
    const PEER_PROTECTED_FILE: &str = "srtcp.bin";
    const LEG_FLOW: SrtpFlow = SrtpFlow::Rtcp;

    for (key, cipher, authentication) in INTEROP_POLICIES {
        let Some(directory) = peer_directory(&format!(
            "g2g_m1101_gst_rtcp_to_g2g_{cipher}_{authentication}"
        )) else {
            return;
        };
        let plain = rtcp_packets(1, SYNCHRONIZATION_SOURCE);
        std::fs::write(directory.join(PEER_PLAIN_FILE), &plain[0]).expect("write the RTCP packet");
        run_peer(
            &directory,
            &format!(
                "filesrc location={PEER_PLAIN_FILE} ! application/x-rtcp \
                 ! srtpenc {} ! filesink location={PEER_PROTECTED_FILE}",
                peer_arguments(key, LEG_FLOW, cipher, authentication)
            ),
        );

        let protected =
            std::fs::read(directory.join(PEER_PROTECTED_FILE)).expect("read the protected RTCP");
        let recovered = run_one(
            &mut decoder_with(key, SrtpFlow::Rtcp, cipher, authentication),
            protected_caps(SrtpFlow::Rtcp),
            Vec::from([protected]),
        )
        .await;
        assert_eq!(
            recovered, plain,
            "srtpdec did not recover gst's {cipher} + {authentication} SRTCP"
        );
        println!("gst -> g2g: one SRTCP packet recovered byte-exact, {cipher} + {authentication}");
    }
}

/// The SRTCP direction the other way, so gst's libsrtp checks the tag `srtpenc`
/// wrote after the index word.
#[tokio::test]
#[ignore = "needs gst-launch-1.0 with libsrtp"]
async fn srtpenc_protects_rtcp_and_gst_recovers_every_legacy_policy() {
    const PEER_PROTECTED_FILE: &str = "srtcp.bin";
    const PEER_RECOVERED_FILE: &str = "recovered.bin";
    const LEG_FLOW: SrtpFlow = SrtpFlow::Rtcp;

    for (key, cipher, authentication) in INTEROP_POLICIES {
        let Some(directory) = peer_directory(&format!(
            "g2g_m1101_g2g_rtcp_to_gst_{cipher}_{authentication}"
        )) else {
            return;
        };
        let plain = rtcp_packets(1, SYNCHRONIZATION_SOURCE);
        let protected = run_one(
            &mut encoder_with(key, SrtpFlow::Rtcp, cipher, authentication),
            plain_caps(SrtpFlow::Rtcp),
            plain.clone(),
        )
        .await;
        std::fs::write(directory.join(PEER_PROTECTED_FILE), &protected[0])
            .expect("write the protected RTCP");

        run_peer(
            &directory,
            &format!(
                "filesrc location={PEER_PROTECTED_FILE} ! {} ! srtpdec name=d \
                 d.rtcp_src ! filesink location={PEER_RECOVERED_FILE}",
                peer_decoder_caps(
                    LEG_FLOW,
                    SYNCHRONIZATION_SOURCE,
                    key,
                    cipher,
                    authentication
                )
            ),
        );

        let recovered =
            std::fs::read(directory.join(PEER_RECOVERED_FILE)).expect("read the recovered RTCP");
        assert_eq!(
            recovered, plain[0],
            "gst did not recover srtpenc's {cipher} + {authentication} SRTCP"
        );
        println!("g2g -> gst: one SRTCP packet recovered byte-exact, {cipher} + {authentication}");
    }
}

/// The MKI under a counter-mode policy, where it sits before the tag rather
/// than after it: libsrtp has to find it at the same offset g2g wrote it.
#[tokio::test]
#[ignore = "needs gst-launch-1.0 with libsrtp"]
async fn gst_protects_with_an_mki_under_a_counter_mode_policy() {
    let Some(directory) = peer_directory("g2g_m1101_gst_mki_to_g2g") else {
        return;
    };
    run_peer(
        &directory,
        &format!(
            "audiotestsrc num-buffers={PEER_BUFFERS} ! audioconvert ! rtpL16pay \
             ! tee name=t ! queue ! multifilesink location=plain%05d.bin \
             t. ! queue ! srtpenc {} mki={MKI_HEX} ! multifilesink location=srtp%05d.bin",
            peer_arguments(
                COUNTER_MODE_KEY_HEX,
                SrtpFlow::Rtp,
                GST_AES_128_COUNTER_MODE,
                GST_HMAC_SHA1_80
            )
        ),
    );

    let plain = numbered_files(&directory, "plain");
    let protected = numbered_files(&directory, "srtp");
    assert!(!plain.is_empty(), "gst wrote no packets");
    let mki_end = protected[0].len() - HMAC_SHA1_80_TAG_LENGTH;
    assert_eq!(
        &protected[0][mki_end - MKI.len()..mki_end],
        MKI,
        "libsrtp puts the MKI between the ciphertext and the tag"
    );

    let mut unprotector = mki_decoder();
    let recovered = run_one(&mut unprotector, protected_caps(SrtpFlow::Rtp), protected).await;
    assert_eq!(recovered, plain, "srtpdec did not recover gst's MKI stream");
    println!(
        "gst -> g2g: {} MKI-tagged counter-mode packets recovered byte-exact",
        plain.len()
    );
}
