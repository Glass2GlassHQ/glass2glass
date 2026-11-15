// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

mod util;

use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticTransform;
use g2g_mcu::logmel::MICRO_SPEECH_SETTINGS;
use g2g_mcu::MicroFrontend;
use util::{block_on, frame_of, payload};

const CHANNELS: usize = 40;
const FEATURE_BYTES: usize = MICRO_SPEECH_SETTINGS.frames * CHANNELS * 4;
const CLIP: &[u8] = include_bytes!("fixtures/keyword/yes.s16le");
const CLIP_BYTES: usize = MICRO_SPEECH_SETTINGS.clip_samples() * 2;
// TensorFlow 2.15.1 audio_microfrontend with the speech_commands settings, times 10/256.
const REFERENCE: &[u8] = include_bytes!("fixtures/keyword/frontend.f32le");

fn frontend_features(
    frontend: &mut MicroFrontend<'_, CHANNELS, 1, FEATURE_BYTES>,
    clip_ring: &StaticLendRing<1, CLIP_BYTES>,
) -> Vec<u8> {
    let output = block_on(frontend.process(frame_of(clip_ring, CLIP, 0, 0)))
        .expect("frontend process")
        .expect("one feature frame");
    payload(&output).to_vec()
}

#[test]
fn micro_frontend_matches_tensorflow_bit_for_bit() {
    let clip_ring: StaticLendRing<1, CLIP_BYTES> = StaticLendRing::new();
    let ring: StaticLendRing<1, FEATURE_BYTES> = StaticLendRing::new();
    // SAFETY: the ring outlives every frame in this test.
    let mut frontend = unsafe { MicroFrontend::with_ring(MICRO_SPEECH_SETTINGS, &ring) }.unwrap();
    let features = frontend_features(&mut frontend, &clip_ring);
    assert_eq!(features, REFERENCE, "features differ from frontend.f32le");
    let again = frontend_features(&mut frontend, &clip_ring);
    assert_eq!(
        again, REFERENCE,
        "noise state leaked from the previous clip"
    );
}
