// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![allow(dead_code)] // each test file uses a subset

use core::future::Future;
use core::pin::Pin;
use std::io::Write;
use std::path::{Path, PathBuf};

use g2g_core::frame::Frame;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    OutputSink, PipelineClock, PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};

// ---------------------------------------------------------------- test scaffold

pub(crate) struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

pub(crate) struct NullOut;
impl OutputSink for NullOut {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// Collects the caps and each frame's (sequence, timing, bytes) so a test can
/// assert the whole stream crossed the boundary intact.
#[derive(Default)]
pub(crate) struct CollectSink {
    pub(crate) caps: Vec<Caps>,
    pub(crate) frames: Vec<(u64, FrameTiming, Vec<u8>)>,
    pub(crate) eos: bool,
}

impl AsyncElement for CollectSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn configure_pipeline(&mut self, c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // The initial caps reach a sink via configure_pipeline, not process.
        self.caps.push(c.clone());
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if let Some(slice) = frame.domain.as_system_slice() {
                        self.frames
                            .push((frame.sequence, frame.timing, slice.to_vec()));
                    }
                }
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Ok(())
        })
    }
}

/// Emits `n` RGBA frames (each byte = frame index) then EOS.
pub(crate) struct CountSrc {
    pub(crate) n: u8,
}

impl SourceLoop for CountSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(test_caps()))
    }
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(test_caps()))))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(test_caps())).await?;
            for i in 0..self.n {
                out.push(PipelinePacket::DataFrame(test_frame(i, FRAME_LEN)))
                    .await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.n as u64)
        })
    }
}

pub(crate) const FRAME_LEN: usize = 2 * 2 * 4; // 2x2 RGBA

pub(crate) fn test_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    }
}

/// A frame tagged twice: byte 0 is the index and byte 1 a fixed marker, so a
/// mis-framed read (the length prefix is the only message boundary) is obvious.
pub(crate) fn test_frame(i: u8, len: usize) -> Frame {
    let mut bytes = vec![i; len];
    bytes[1] = 0xAB;
    Frame::from_system_bytes(
        bytes,
        FrameTiming {
            pts_ns: i as u64 * 1_000_000,
            dts_ns: i as u64 * 1_000_000,
            duration_ns: 33_000,
            keyframe: i == 0,
            ..FrameTiming::default()
        },
        i as u64,
    )
}

// ------------------------------------------------------------------------ TLS

/// A self-signed certificate for the loopback server, written out as the PEM file
/// pair the `certificate` / `private-key` properties take, plus the hex SHA-256
/// digest a client passes as `server-certificate-hashes` (no system root covers a
/// throwaway certificate, exactly as for a browser's `serverCertificateHashes`).
pub(crate) struct TestCert {
    pub(crate) cert_path: PathBuf,
    pub(crate) key_path: PathBuf,
    pub(crate) hash_hex: String,
}

impl TestCert {
    pub(crate) fn generate(tag: &str) -> Self {
        let issued =
            g2g_plugins::remotewtio::generate_certificate().expect("self-signed certificate");
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let cert_path = dir.join(format!("g2g-m901-{pid}-{tag}.crt"));
        let key_path = dir.join(format!("g2g-m901-{pid}-{tag}.key"));
        write_file(&cert_path, issued.certificate_pem.as_bytes());
        write_file(&key_path, issued.private_key_pem.as_bytes());
        let hash_hex = issued.sha256_hex;

        Self {
            cert_path,
            key_path,
            hash_hex,
        }
    }

    pub(crate) fn cert(&self) -> String {
        self.cert_path.display().to_string()
    }

    pub(crate) fn key(&self) -> String {
        self.key_path.display().to_string()
    }
}

impl Drop for TestCert {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.cert_path);
        let _ = std::fs::remove_file(&self.key_path);
    }
}

pub(crate) fn write_file(path: &Path, bytes: &[u8]) {
    let mut f = std::fs::File::create(path).expect("create pem");
    f.write_all(bytes).expect("write pem");
}
