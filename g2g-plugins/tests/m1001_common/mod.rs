//! Helpers shared by the M1001 codec-quality batteries (`m1001_golden_decode`,
//! `m1001_encode_psnr`, `m1001_psnr_oracle`): the collect-into-Vec sink, the
//! deterministic synthetic source the encode batteries measure against, the AV1
//! temporal-unit split a decoder needs from a bare OBU stream, and the evidence-log
//! scoping every battery does before persisting its `Quality` rows. One definition,
//! included per test binary via `mod m1001_common;`.
#![allow(dead_code)] // no one battery uses every helper here

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;

use g2g_core::conformance::Evidence;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, G2gError, OutputSink, PushOutcome};
use g2g_plugins::conformance::persist;

/// Collects the caps and the system-memory payload of every frame an element pushes.
#[derive(Default)]
pub(crate) struct CaptureSink {
    pub(crate) caps: Vec<Caps>,
    pub(crate) frames: Vec<Vec<u8>>,
}

impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.frames.push(s.to_vec());
                    }
                }
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

impl CaptureSink {
    /// Every collected frame end to end, the buffer a golden digest is taken over.
    pub(crate) fn concatenated(&self) -> Vec<u8> {
        self.frames.concat()
    }
}

/// One `DataFrame` carrying `data` at `pts_ns`.
pub(crate) fn data_frame(data: Vec<u8>, pts_ns: u64, sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        sequence,
    ))
}

/// The path of a committed fixture under `tests/fixtures`.
pub(crate) fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Scopes where a battery's `Quality` / `Oracle` evidence lands: a dedicated,
/// freshly-truncated log per test binary, unless a run already set
/// `$G2G_CONFORMANCE_LOG` (the CI conformance job aggregates every battery into one),
/// in which case rows are appended to that shared log and left in place.
pub(crate) struct EvidenceLog {
    dedicated: Option<PathBuf>,
}

impl EvidenceLog {
    /// Point the log at `<tempdir>/g2g-conformance-<name>.tsv` unless one is set.
    pub(crate) fn scoped(name: &str) -> Self {
        if std::env::var_os("G2G_CONFORMANCE_LOG").is_some() {
            return Self { dedicated: None };
        }
        let path = std::env::temp_dir().join(format!("g2g-conformance-{name}.tsv"));
        std::env::set_var("G2G_CONFORMANCE_LOG", &path);
        let _ = std::fs::remove_file(&path);
        Self {
            dedicated: Some(path),
        }
    }

    /// Append one piece of evidence for `element`, failing loud if the log cannot be
    /// written (silently losing evidence would overstate nothing but hide a pass).
    pub(crate) fn record(&self, element: &str, evidence: &Evidence) {
        persist::record_evidence(element, evidence).expect("append conformance evidence");
    }
}

impl Drop for EvidenceLog {
    fn drop(&mut self) {
        if let Some(path) = &self.dedicated {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// LEB128 unsigned (AV1 4.10.5) at `offset`: the value and the bytes it occupied.
fn leb128(stream: &[u8], offset: usize) -> Option<(u64, usize)> {
    let mut value = 0u64;
    let mut p = offset;
    for i in 0..8 {
        let b = *stream.get(p)?;
        p += 1;
        value |= ((b & 0x7f) as u64) << (7 * i);
        if b & 0x80 == 0 {
            return Some((value, p - offset));
        }
    }
    None
}

/// Split a low-overhead AV1 OBU stream into temporal units, each starting at its
/// temporal-delimiter OBU. This is the chunking a decoder expects: one call per
/// temporal unit, sequence header included in the unit that carries it.
pub(crate) fn split_temporal_units(stream: &[u8]) -> Vec<&[u8]> {
    const OBU_TEMPORAL_DELIMITER: u8 = 2;
    let mut units = Vec::new();
    let mut unit_start = 0usize;
    let mut p = 0usize;
    while p < stream.len() {
        let obu_start = p;
        let header = stream[p];
        let obu_type = (header >> 3) & 0xf;
        let has_extension = (header >> 2) & 1 == 1;
        let has_size = (header >> 1) & 1 == 1;
        p += 1;
        if has_extension {
            p += 1;
        }
        let payload = if has_size {
            let (size, read) = leb128(stream, p).expect("valid leb128 obu_size");
            p += read;
            size as usize
        } else {
            stream.len().saturating_sub(p)
        };
        p = (p + payload).min(stream.len());
        if obu_type == OBU_TEMPORAL_DELIMITER && obu_start > unit_start {
            units.push(&stream[unit_start..obu_start]);
            unit_start = obu_start;
        }
    }
    if unit_start < stream.len() {
        units.push(&stream[unit_start..]);
    }
    units
}

/// Split an ADTS AAC elementary stream into one chunk per access unit. `AacParse`
/// refines caps but forwards the buffer whole, and a decoder needs one access unit
/// per packet, so the batteries do the framing from the 13-bit `aac_frame_length`.
pub(crate) fn split_adts_frames(stream: &[u8]) -> Vec<&[u8]> {
    const HEADER: usize = 7;
    let mut frames = Vec::new();
    let mut p = 0usize;
    while p + HEADER <= stream.len() {
        if stream[p] != 0xff || stream[p + 1] & 0xf0 != 0xf0 {
            break; // lost sync: stop rather than guess at a resync point
        }
        let length = ((stream[p + 3] as usize & 0x03) << 11)
            | ((stream[p + 4] as usize) << 3)
            | (stream[p + 5] as usize >> 5);
        if length < HEADER || p + length > stream.len() {
            break;
        }
        frames.push(&stream[p..p + length]);
        p += length;
    }
    frames
}

/// A deterministic synthetic I420 frame: a diagonal luma gradient with a 16-pixel
/// checkerboard and a vertical bar that walks with `phase`, plus chroma gradients.
/// The detail matters: a flat image would give any encoder an unrealistically high
/// PSNR, so the pattern carries both smooth and hard edges.
pub(crate) fn synthetic_i420(width: usize, height: usize, phase: usize) -> Vec<u8> {
    assert!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "I420 needs even geometry"
    );
    let mut buffer = Vec::with_capacity(width * height * 3 / 2);
    for y in 0..height {
        for x in 0..width {
            let gradient = ((x * 255) / width + (y * 255) / height) / 2;
            let checker = if (x / 16 + y / 16).is_multiple_of(2) {
                32
            } else {
                0
            };
            let bar = if (x + phase * 8) % 64 < 4 { 96 } else { 0 };
            buffer.push((gradient + checker + bar).min(255) as u8);
        }
    }
    for plane in 0..2u8 {
        for y in 0..height / 2 {
            for x in 0..width / 2 {
                let base = if plane == 0 { x * 2 } else { y * 2 };
                buffer.push(((base * 200) / width.max(height) + 28).min(255) as u8);
            }
        }
    }
    buffer
}

/// The same pattern as [`synthetic_i420`] in packed RGBA8: the luma pattern on the
/// green channel, the two chroma ramps on red and blue, alpha opaque.
pub(crate) fn synthetic_rgba8(width: usize, height: usize, phase: usize) -> Vec<u8> {
    let planar = synthetic_i420(width, height, phase);
    let mut buffer = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        for x in 0..width {
            let luma = planar[y * width + x];
            buffer.push(((x * 255) / width) as u8);
            buffer.push(luma);
            buffer.push(((y * 255) / height) as u8);
            buffer.push(255);
        }
    }
    buffer
}

/// The R, G and B channels of a packed RGBA8 buffer as three planes. Alpha is left
/// out: it is constant in the synthetic source and would flatter any PSNR that
/// included it.
pub(crate) fn deinterleave_rgb(rgba: &[u8]) -> [Vec<u8>; 3] {
    let pixels = rgba.len() / 4;
    let mut channels = [
        Vec::with_capacity(pixels),
        Vec::with_capacity(pixels),
        Vec::with_capacity(pixels),
    ];
    for pixel in rgba.chunks_exact(4) {
        for (channel, value) in channels.iter_mut().zip(&pixel[..3]) {
            channel.push(*value);
        }
    }
    channels
}

/// A deterministic synthetic S16LE tone: two summed sinusoids so a lossy audio
/// encoder has more than one partial to preserve. `channels` interleaved.
pub(crate) fn synthetic_pcm_s16le(channels: usize, samples: usize, sample_rate: u32) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(samples * channels * 2);
    for n in 0..samples {
        let t = n as f64 / sample_rate as f64;
        for channel in 0..channels {
            let detune = 1.0 + channel as f64 * 0.01;
            let value = 0.45 * (2.0 * core::f64::consts::PI * 440.0 * detune * t).sin()
                + 0.25 * (2.0 * core::f64::consts::PI * 1_970.0 * detune * t).sin();
            let sample = (value * i16::MAX as f64).round().clamp(-32768.0, 32767.0) as i16;
            buffer.extend_from_slice(&sample.to_le_bytes());
        }
    }
    buffer
}
