//! Helpers shared by the cue-rendering tests (`m1055_cue_text_style`,
//! `m1057_cue_css`): the one-cue WebVTT document builder, the black frame and
//! sink the overlay renders into, the system fonts to render with, and the pixel
//! predicates and scanners the assertions read the result through. One
//! definition, included per test binary via `mod cue_render_common;`.
#![allow(dead_code)] // no one test file uses every helper here

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::subparse::parse_webvtt;
use g2g_plugins::textoverlay::TextOverlay;

pub(crate) const W: u32 = 480;
pub(crate) const H: u32 = 160;
pub(crate) const FONT_PX: u32 = 32;

/// Cue-wide declarations every document here starts from: the backing box off,
/// so the only painted pixels are the ones under test.
pub(crate) const NO_BOX: &str = "::cue { background-color: transparent; }";

/// First available Latin system font, or `None` to skip (a host with no fonts).
/// These are the Fedora paths the dev host has.
pub(crate) fn latin_font() -> Option<Vec<u8>> {
    read_first(&[
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
    ])
}

/// First available `wght`-variable system font, or `None` to skip: a weight the
/// font database cannot satisfy with a real bold face reaches that axis.
pub(crate) fn variable_font() -> Option<Vec<u8>> {
    read_first(&[
        "/usr/share/fonts/abattis-cantarell-vf-fonts/Cantarell-VF.otf",
        "/usr/share/fonts/vazirmatn-vf-fonts/Vazirmatn[wght].ttf",
        "/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc",
    ])
}

pub(crate) fn read_first(paths: &[&str]) -> Option<Vec<u8>> {
    paths.iter().find_map(|path| std::fs::read(path).ok())
}

/// The file `fc-match` resolves a query to, or `None` where fontconfig is not
/// installed.
pub(crate) fn fc_match_file(query: &str) -> Option<String> {
    let out = std::process::Command::new("fc-match")
        .args([query, "file"])
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub(crate) fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// A one-cue WebVTT document: `style` declarations, then `text` as the cue body.
pub(crate) fn document(style: &str, text: &str) -> String {
    format!("WEBVTT\n\nSTYLE\n{style}\n\n00:00:00.000 --> 00:00:10.000\n{text}\n")
}

pub(crate) fn black_frame() -> Frame {
    let mut bytes = Vec::with_capacity((W * H * 4) as usize);
    for _ in 0..W * H {
        bytes.extend_from_slice(&[0, 0, 0, 255]);
    }
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    )
}

#[derive(Default)]
pub(crate) struct FrameSink {
    pub(crate) last: Option<Frame>,
}
impl OutputSink for FrameSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet.take() {
            self.last = Some(frame);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// Render the document's cue over a black frame, as RGBA8 bytes. With no `font`
/// the shaper picks the system sans-serif itself.
pub(crate) async fn render(font: Option<&[u8]>, vtt: &str) -> Vec<u8> {
    let mut overlay = TextOverlay::new()
        .with_cues(parse_webvtt(vtt))
        .with_font_size(FONT_PX);
    if let Some(font) = font {
        overlay = overlay.with_font_bytes(font, 0).expect("font parses");
    }
    overlay.configure_pipeline(&caps()).expect("caps accepted");
    let mut sink = FrameSink::default();
    overlay
        .process(PipelinePacket::DataFrame(black_frame()), &mut sink)
        .await
        .expect("frame rendered");
    sink.last
        .expect("frame forwarded")
        .domain
        .as_system_slice()
        .expect("system memory out")
        .to_vec()
}

/// Whether a pixel is dominated by one channel, which is how a coloured glyph or
/// bar reads once it is alpha-blended onto the black frame.
pub(crate) fn is_red(px: &[u8]) -> bool {
    dominates(px[0], px[1], px[2])
}
pub(crate) fn is_green(px: &[u8]) -> bool {
    dominates(px[1], px[0], px[2])
}
pub(crate) fn is_blue(px: &[u8]) -> bool {
    dominates(px[2], px[0], px[1])
}
pub(crate) fn dominates(channel: u8, other: u8, third: u8) -> bool {
    let (channel, other, third) = (u32::from(channel), u32::from(other), u32::from(third));
    channel > 60 && other * 3 < channel && third * 3 < channel
}

/// Whether a pixel is painted at all, so it counts as ink.
pub(crate) fn is_ink(px: &[u8]) -> bool {
    px[0] != 0 || px[1] != 0 || px[2] != 0
}

/// Count of painted (non-black) pixels, a proxy for how much ink a weight or a
/// slant puts on the canvas.
pub(crate) fn ink(pixels: &[u8]) -> usize {
    pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| is_ink(*px))
        .count()
}

/// Bounding box `(left, top, right, bottom)` of the pixels `pick` accepts,
/// inclusive on every edge. `None` when it accepted none.
pub(crate) fn bounds(pixels: &[u8], pick: fn(&[u8]) -> bool) -> Option<(u32, u32, u32, u32)> {
    let mut found: Option<(u32, u32, u32, u32)> = None;
    for (i, px) in pixels.as_chunks::<4>().0.iter().enumerate() {
        if !pick(px) {
            continue;
        }
        let (x, y) = (i as u32 % W, i as u32 / W);
        found = Some(match found {
            None => (x, y, x, y),
            Some((l, t, r, b)) => (l.min(x), t.min(y), r.max(x), b.max(y)),
        });
    }
    found
}

/// The longest unbroken run of `pick` pixels along a row, as `(row, first, last)`.
/// The underline bar is the widest one a cue draws: it spans its whole run,
/// where a glyph only ever fills part of one letter.
pub(crate) fn widest_row_run(pixels: &[u8], pick: fn(&[u8]) -> bool) -> Option<(u32, u32, u32)> {
    longest_run(pixels, pick, |offset, line| (offset, line), W, H)
}

/// The longest unbroken run of `pick` pixels down a column, as
/// `(column, first, last)`.
pub(crate) fn tallest_column_run(
    pixels: &[u8],
    pick: fn(&[u8]) -> bool,
) -> Option<(u32, u32, u32)> {
    longest_run(pixels, pick, |offset, line| (line, offset), H, W)
}

/// The longest unbroken run of `pick` pixels along one axis: `lines` scans of
/// `len` pixels each, `at` mapping a scan position to `(line, offset)`.
pub(crate) fn longest_run(
    pixels: &[u8],
    pick: fn(&[u8]) -> bool,
    at: fn(u32, u32) -> (u32, u32),
    len: u32,
    lines: u32,
) -> Option<(u32, u32, u32)> {
    let mut best: Option<(u32, u32, u32)> = None;
    for line in 0..lines {
        let mut run_start: Option<u32> = None;
        for offset in 0..=len {
            let hit = offset < len && {
                let (x, y) = at(offset, line);
                pick(&pixels[((y * W + x) * 4) as usize..][..4])
            };
            match (hit, run_start) {
                (true, None) => run_start = Some(offset),
                (false, Some(start)) => {
                    let run = (line, start, offset - 1);
                    if best.is_none_or(|(_, b_start, b_end)| b_end - b_start < run.2 - run.1) {
                        best = Some(run);
                    }
                    run_start = None;
                }
                _ => {}
            }
        }
    }
    best
}
