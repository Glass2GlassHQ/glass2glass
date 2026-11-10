// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M32: `WasapiSrc` captures PCM from the default audio endpoint through the
//! source loop. The input mirror of the M29 `WasapiSink` test.
//!
//! Run with:
//!
//! ```powershell
//! cargo test -p g2g-plugins --features wasapi-src --test m32_wasapi_capture
//! ```
//!
//! Opens a real capture endpoint, so on a headless host (no microphone /
//! capture device) it skips: the probe returns `DeviceNotFound` or a
//! `Hardware` error.

#![cfg(all(target_os = "windows", feature = "wasapi-src"))]

use g2g_core::element::{OutputSink, PushOutcome};
use g2g_core::frame::PipelinePacket;
use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, G2gError};
use g2g_plugins::wasapisrc::WasapiSrc;

#[derive(Default)]
struct Collect {
    data_frames: usize,
    sample_bytes: usize,
    eos: bool,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(slice) = f.domain.as_system_slice() {
                        self.sample_bytes += slice.len();
                    }
                    self.data_frames += 1;
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[tokio::test]
async fn captures_pcm_from_the_default_endpoint_or_skips() {
    const BUFFERS: u64 = 5;

    let mut src = WasapiSrc::new(BUFFERS);
    let caps = match src.intercept_caps().await {
        Ok(caps) => caps,
        Err(G2gError::Hardware(_) | G2gError::DeviceNotFound) => {
            std::eprintln!("skipping: no audio capture endpoint on this host");
            return;
        }
        Err(e) => panic!("unexpected probe error: {e:?}"),
    };
    assert!(
        matches!(caps, Caps::Audio { .. }),
        "capture caps must be PCM audio, got {caps:?}"
    );

    src.configure_pipeline(&caps).expect("configure source");

    let mut out = Collect::default();
    let emitted = src.run(&mut out).await.expect("capture runs");

    assert!(out.eos, "source ends with Eos");
    assert_eq!(out.data_frames as u64, emitted, "every frame counted");
    assert_eq!(emitted, BUFFERS, "captured the requested number of buffers");
    assert!(out.sample_bytes > 0, "captured non-empty PCM");
}

#[tokio::test]
async fn unknown_endpoint_id_fails_as_not_found() {
    for id in ["{0.0.1.00000000}.{no-such-endpoint}", "bogus"] {
        let mut src = WasapiSrc::new(1).with_device(id);
        assert_eq!(
            src.intercept_caps().await,
            Err(G2gError::DeviceNotFound),
            "{id}"
        );
    }
}
