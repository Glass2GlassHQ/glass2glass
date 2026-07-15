//! M644: the `convert` stage (32-bit capture slots -> S16LE). The narrowing
//! is keep-the-top-16-bits (left-justified I2S/SAI slots), so the reference
//! is trivial and the tests focus on the wire framing: slot boundaries,
//! interleaving preservation, and torn-slot rejection.

mod util;

use g2g_core::error::G2gError;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticTransform;
use g2g_mcu::pcm::s32_slot_to_s16;
use g2g_mcu::PcmConvert;
use util::{block_on, frame_of, payload};

fn slots_le(slots: &[i32]) -> Vec<u8> {
    slots.iter().flat_map(|s| s.to_le_bytes()).collect()
}

#[test]
fn narrows_left_justified_slots() {
    // A 24-bit sample sits in the top 3 bytes; the top 16 bits survive.
    assert_eq!(s32_slot_to_s16(0x1234_5600), 0x1234);
    assert_eq!(s32_slot_to_s16(-0x1234_5600), -0x1235, "arithmetic shift floors toward -inf");
    assert_eq!(s32_slot_to_s16(i32::MIN), i16::MIN);
    assert_eq!(s32_slot_to_s16(i32::MAX), i16::MAX);
    assert_eq!(s32_slot_to_s16(0xFFFF), 0, "low-16 dither never leaks");
}

#[test]
fn element_converts_and_inherits_timing() {
    let input_ring: StaticLendRing<1, 64> = StaticLendRing::new();
    let out_ring: StaticLendRing<1, 32> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut conv = unsafe { PcmConvert::with_ring(&out_ring) };

    let slots = [0x7FFF_0000i32, -0x8000_0000, 0x0001_8000, -0x0001_8000];
    let input = frame_of(&input_ring, &slots_le(&slots), 555, 7);
    let out = block_on(conv.process(input)).expect("convert").expect("frame");
    let samples: Vec<i16> =
        payload(&out).chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
    let expect: Vec<i16> = slots.iter().map(|&s| s32_slot_to_s16(s)).collect();
    assert_eq!(samples, expect, "each slot narrows independently, order preserved");
    assert_eq!(out.timing.pts_ns, 555, "timing inherited");
    assert_eq!(out.sequence, 7, "sequence inherited");
}

#[test]
fn torn_slots_are_rejected() {
    let input_ring: StaticLendRing<1, 64> = StaticLendRing::new();
    let out_ring: StaticLendRing<1, 32> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut conv = unsafe { PcmConvert::with_ring(&out_ring) };
    let input = frame_of(&input_ring, &[1u8, 2, 3, 4, 5, 6], 0, 0);
    assert!(
        matches!(block_on(conv.process(input)), Err(G2gError::CapsMismatch)),
        "a payload that is not whole 32-bit slots is a framing bug"
    );
}
