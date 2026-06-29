//! Subtitle cue parsing (M171, `no_std`): SRT (SubRip) and WebVTT text into a
//! common timed-text [`Cue`] list, the `subparse` analog. Pure parsing with no
//! OS dependency, so it sits on the `no_std + alloc` baseline; the `TextOverlay`
//! element ([`crate::textoverlay`]) renders the cues, and the `std`-gated
//! `location=` property loads a file through these parsers.
//!
//! Both formats are a sequence of blank-line-separated blocks; a cue block has a
//! timing line (`start --> end`) with the text on the lines after it. The shared
//! block walker tolerates the differences:
//!
//! - **Timestamps.** SRT uses `HH:MM:SS,mmm` (comma); WebVTT uses `.` and allows
//!   the hours to be omitted (`MM:SS.mmm`). [`parse_timestamp`] accepts either
//!   separator and a 2- or 3-component clock.
//! - **Leading lines.** A block may open with an SRT sequence number or a WebVTT
//!   cue identifier before the timing line; everything before the `-->` line is
//!   ignored, so both are handled without a format flag.
//! - **WebVTT structure.** The `WEBVTT` header block and `NOTE` / `STYLE` /
//!   `REGION` blocks are skipped (a `NOTE` may itself contain `-->`).
//! - **Inline tags.** `<i>`, `<b>`, `<c.classname>`, and `<00:00:01.000>` cue
//!   timestamps are stripped to plain text (the bitmap overlay has no styling).
//! - **Cue settings.** Tokens after the end timestamp on the timing line
//!   (`position:50% align:start`) are ignored.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, FrameTiming,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, TextFormat,
};

/// Horizontal text alignment within a cue box (the WebVTT `align:` setting;
/// `left`/`right` map to `Start`/`End`). The default is `Center`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextAlign {
    /// Left-aligned (`align:start` / `align:left`).
    Start,
    /// Centred (`align:center`, the default).
    #[default]
    Center,
    /// Right-aligned (`align:end` / `align:right`).
    End,
}

/// WebVTT cue placement settings, the subset the bitmap overlay honours. `None`
/// fields mean "auto": auto `line` stacks the cue from the bottom, auto
/// `position` centres it. SRT cues always carry the default (no positioning).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CueSettings {
    /// Horizontal anchor as a percent `0..=100` of the frame width
    /// (WebVTT `position:`). `None` = auto (centre).
    pub position: Option<u8>,
    /// Vertical placement as a percent `0..=100` of the frame height
    /// (WebVTT `line:` in its percentage form). `None` = auto (stack from bottom).
    pub line: Option<u8>,
    /// Text alignment within the box (WebVTT `align:`).
    pub align: TextAlign,
}

/// One timed subtitle cue: a half-open `[start_ns, end_ns)` running-time span, its
/// text, and its placement settings. Multi-line text keeps its `\n` line breaks
/// (the overlay renders one row per line).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cue {
    /// Cue onset, nanoseconds from the stream origin.
    pub start_ns: u64,
    /// Cue end, nanoseconds. The cue shows while `start_ns <= t < end_ns`.
    pub end_ns: u64,
    /// Plain text, with `\n` separating wrapped lines. Markup is stripped.
    pub text: String,
    /// Placement from the WebVTT cue settings (default for SRT).
    pub settings: CueSettings,
}

impl Cue {
    /// Whether this cue is visible at running time `t_ns` (`[start, end)`).
    pub fn covers(&self, t_ns: u64) -> bool {
        t_ns >= self.start_ns && t_ns < self.end_ns
    }
}

/// Parse SubRip (`.srt`) text into cues, in file order. Malformed blocks (no
/// timing line, unparseable timestamps) are skipped rather than failing the whole
/// parse, matching how players tolerate dirty subtitle files.
pub fn parse_srt(input: &str) -> Vec<Cue> {
    parse_blocks(input, false)
}

/// Parse WebVTT (`.vtt`) text into cues, in file order. The `WEBVTT` header and
/// `NOTE` / `STYLE` / `REGION` blocks are skipped; inline markup is removed.
pub fn parse_webvtt(input: &str) -> Vec<Cue> {
    parse_blocks(input, true)
}

/// Auto-detect the format from the content and parse: a leading `WEBVTT`
/// signature selects WebVTT, a leading `[` section header selects SSA/ASS,
/// otherwise SRT (all after an optional BOM).
pub fn parse_auto(input: &str) -> Vec<Cue> {
    let trimmed = input.strip_prefix('\u{feff}').unwrap_or(input).trim_start();
    if trimmed.starts_with("WEBVTT") {
        parse_webvtt(input)
    } else if trimmed.starts_with('[') {
        // SSA/ASS always opens with a section header (`[Script Info]`); SRT /
        // WebVTT never start with `[`.
        parse_ssa(input)
    } else {
        parse_srt(input)
    }
}

/// Parse SubStation Alpha / Advanced SSA (`.ssa` / `.ass`) into cues, in file
/// order. Only the `[Events]` section is read: its `Format:` line gives the
/// column order, and each `Dialogue:` line is split accordingly (the `Text`
/// column is last and may itself contain commas). Inline override blocks
/// (`{\i1}`...) are stripped and `\N` / `\n` line breaks become real newlines,
/// like the SRT / WebVTT tag handling. Malformed dialogue lines are skipped.
pub fn parse_ssa(input: &str) -> Vec<Cue> {
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let mut cues = Vec::new();
    let mut in_events = false;
    // V4+ default column order, used until an explicit `Format:` line overrides
    // it: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text.
    let (mut i_start, mut i_end, mut i_text) = (1usize, 2usize, 9usize);
    for line in input.lines() {
        let line = line.trim();
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            in_events = name.eq_ignore_ascii_case("Events");
            continue;
        }
        if !in_events {
            continue;
        }
        if let Some(rest) = strip_prefix_ci(line, "Format:") {
            let cols: Vec<&str> = rest.split(',').map(str::trim).collect();
            i_start = col_index(&cols, "Start").unwrap_or(i_start);
            i_end = col_index(&cols, "End").unwrap_or(i_end);
            // Text is the last column by spec; fall back to that if unnamed.
            i_text = col_index(&cols, "Text").unwrap_or(cols.len().saturating_sub(1));
        } else if let Some(rest) = strip_prefix_ci(line, "Dialogue:") {
            if let Some(cue) = parse_ass_dialogue(rest, i_start, i_end, i_text) {
                cues.push(cue);
            }
        }
    }
    cues
}

/// Case-insensitive `name -> column index` lookup in a `Format:` column list.
fn col_index(cols: &[&str], name: &str) -> Option<usize> {
    cols.iter().position(|c| c.eq_ignore_ascii_case(name))
}

/// Case-insensitively strip an ASCII keyword prefix (`"Dialogue:"`), returning
/// the remainder. `str::get` keeps the byte-slice on a char boundary so a line
/// opening with a multi-byte char never panics.
fn strip_prefix_ci<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let head = line.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix).then(|| &line[prefix.len()..])
}

/// Parse one `Dialogue:` body into a cue using the resolved column indices. The
/// `Text` column is last, so we split on only the leading commas and keep its
/// remainder (commas and all) intact.
fn parse_ass_dialogue(body: &str, i_start: usize, i_end: usize, i_text: usize) -> Option<Cue> {
    // splitn keeps everything after the i_text-th comma as the final field.
    let fields: Vec<&str> = body.splitn(i_text + 1, ',').collect();
    if fields.len() <= i_text {
        return None;
    }
    let start_ns = parse_timestamp(fields.get(i_start)?.trim())?;
    let end_ns = parse_timestamp(fields.get(i_end)?.trim())?;
    let text = strip_ass_markup(fields[i_text]);
    if text.trim().is_empty() {
        return None;
    }
    Some(Cue { start_ns, end_ns, text, settings: CueSettings::default() })
}

/// Strip ASS override blocks (`{...}`) and turn the `\N` / `\n` line breaks and
/// `\h` hard space into plain text, the SSA analog of [`push_stripped`].
fn strip_ass_markup(raw: &str) -> String {
    let mut out = String::new();
    let mut chars = raw.chars().peekable();
    let mut in_brace = false;
    while let Some(c) = chars.next() {
        match c {
            '{' => in_brace = true,
            '}' => in_brace = false,
            _ if in_brace => {}
            '\\' => match chars.peek() {
                // `\N` hard break and `\n` soft break both render as a newline.
                Some('N') | Some('n') => {
                    out.push('\n');
                    chars.next();
                }
                Some('h') => {
                    out.push(' ');
                    chars.next();
                }
                _ => out.push('\\'),
            },
            _ => out.push(c),
        }
    }
    out
}

/// Walk blank-line-separated blocks, turning each cue block into a [`Cue`].
/// `webvtt` enables the WebVTT-only block skips. `str::lines` already strips a
/// trailing `\r`, so CRLF input is handled without extra work.
fn parse_blocks(input: &str, webvtt: bool) -> Vec<Cue> {
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let mut cues = Vec::new();
    let mut block: Vec<&str> = Vec::new();
    for line in input.lines() {
        if line.trim().is_empty() {
            if let Some(cue) = block_to_cue(&block, webvtt) {
                cues.push(cue);
            }
            block.clear();
        } else {
            block.push(line);
        }
    }
    if let Some(cue) = block_to_cue(&block, webvtt) {
        cues.push(cue);
    }
    cues
}

/// Turn one non-empty block into a cue, or `None` if it is not a cue (a WebVTT
/// header / NOTE / STYLE / REGION block, or a block with no timing line).
fn block_to_cue(block: &[&str], webvtt: bool) -> Option<Cue> {
    if block.is_empty() {
        return None;
    }
    if webvtt {
        let first = block[0].trim_start();
        // The header block and the non-cue WebVTT blocks. `STYLE` / `NOTE` may
        // contain text that looks like a timing line, so skip them explicitly.
        if first == "WEBVTT"
            || first.starts_with("WEBVTT ")
            || first.starts_with("WEBVTT\t")
            || first.starts_with("NOTE")
            || first.starts_with("STYLE")
            || first.starts_with("REGION")
        {
            return None;
        }
    }
    // The timing line is the first line containing the `-->` cue arrow; any lines
    // before it are an SRT index or a WebVTT cue identifier.
    let timing_idx = block.iter().position(|l| l.contains("-->"))?;
    let (start_ns, end_ns, settings) = parse_timing(block[timing_idx])?;

    let mut text = String::new();
    for (i, raw) in block[timing_idx + 1..].iter().enumerate() {
        if i > 0 {
            text.push('\n');
        }
        push_stripped(raw, &mut text);
    }
    // Drop a fully empty payload (a timing line with no following text).
    if text.trim().is_empty() {
        return None;
    }
    Some(Cue { start_ns, end_ns, text, settings })
}

/// Parse a `start --> end [settings...]` timing line into a nanosecond span plus
/// the WebVTT cue settings (the tokens after the end timestamp).
fn parse_timing(line: &str) -> Option<(u64, u64, CueSettings)> {
    let (lhs, rhs) = line.split_once("-->")?;
    let start = parse_timestamp(lhs.trim())?;
    // The end timestamp is the first whitespace token; cue settings follow it.
    let mut toks = rhs.split_whitespace();
    let end = parse_timestamp(toks.next()?)?;
    Some((start, end, parse_settings(toks)))
}

/// Parse the `name:value` cue-setting tokens that follow the end timestamp.
/// Recognises `position`, `line` (percentage form), and `align`; `size`,
/// `vertical`, and `region` are accepted but not applied by the bitmap overlay.
fn parse_settings<'a>(tokens: impl Iterator<Item = &'a str>) -> CueSettings {
    let mut s = CueSettings::default();
    for tok in tokens {
        let Some((key, val)) = tok.split_once(':') else {
            continue;
        };
        match key {
            "position" => s.position = parse_percent(val),
            // Only the percentage form of `line:` maps to our model; a bare line
            // number stays auto (bottom-stacked).
            "line" => s.line = parse_percent(val),
            "align" => {
                if let Some(a) = parse_align(val) {
                    s.align = a;
                }
            }
            _ => {}
        }
    }
    s
}

/// Parse a percentage setting value (`"50%"`, or `"50%,start"` with an extra
/// keyword) into `0..=100`. `None` if it is not a percentage.
fn parse_percent(v: &str) -> Option<u8> {
    let v = v.split(',').next()?.trim().strip_suffix('%')?;
    let n: i32 = v.parse().ok()?;
    Some(n.clamp(0, 100) as u8)
}

/// Parse an `align:` value (the part before any `,`): `start`/`left`,
/// `center`/`middle`, `end`/`right`.
fn parse_align(v: &str) -> Option<TextAlign> {
    match v.split(',').next()?.trim() {
        "start" | "left" => Some(TextAlign::Start),
        "center" | "middle" => Some(TextAlign::Center),
        "end" | "right" => Some(TextAlign::End),
        _ => None,
    }
}

/// Parse one timestamp to nanoseconds. Accepts `HH:MM:SS,mmm`, `HH:MM:SS.mmm`,
/// and the WebVTT short form `MM:SS.mmm`; the fractional part may be `,` or `.`
/// separated and 1-3 digits.
pub fn parse_timestamp(s: &str) -> Option<u64> {
    let s = s.trim();
    let (clock, frac) = match s.find(['.', ',']) {
        Some(i) => (&s[..i], &s[i + 1..]),
        None => (s, ""),
    };
    // Components are seconds, then minutes, then (optional) hours, right to left.
    let mut it = clock.split(':').rev();
    let secs: u64 = it.next()?.trim().parse().ok()?;
    let mins: u64 = match it.next() {
        Some(p) => p.trim().parse().ok()?,
        None => 0,
    };
    let hours: u64 = match it.next() {
        Some(p) => p.trim().parse().ok()?,
        None => 0,
    };
    // No more than three clock fields.
    if it.next().is_some() || secs >= 60 || mins >= 60 {
        return None;
    }
    let millis = frac_millis(frac);
    // hours is untrusted and unbounded; fold with checked arithmetic so a huge
    // value fails the parse (None) instead of overflowing. mins/secs are < 60
    // and millis < 1000, so their sub-products cannot overflow.
    let total_secs = hours.checked_mul(3600)?.checked_add(mins * 60 + secs)?;
    let total_ms = total_secs.checked_mul(1000)?.checked_add(millis)?;
    total_ms.checked_mul(1_000_000)
}

/// Interpret a fractional-second digit string as milliseconds: take up to three
/// leading digits, right-padding to thousandths (`5` -> 500, `25` -> 250).
fn frac_millis(frac: &str) -> u64 {
    let mut ms = 0u64;
    let mut count = 0;
    for c in frac.chars() {
        if !c.is_ascii_digit() || count == 3 {
            break;
        }
        ms = ms * 10 + (c as u64 - '0' as u64);
        count += 1;
    }
    while count < 3 {
        ms *= 10;
        count += 1;
    }
    ms
}

/// Append `line` to `out` with any `<...>` markup removed. Handles `<i>`, `<b>`,
/// `<c.class>`, and `<00:00:01.000>` inline cue timestamps uniformly.
fn push_stripped(line: &str, out: &mut String) {
    let mut in_tag = false;
    for c in line.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
}

/// Subtitle parser element (M400): a `Caps::Text{Srt}` / `Caps::Text{WebVtt}`
/// byte stream in, timed `Caps::Text{Utf8}` cues out, one frame per cue with the
/// cue window carried as PTS + duration. The pipeline-native counterpart of the
/// `parse_srt` / `parse_webvtt` functions above (which `TextOverlay` calls
/// directly on an out-of-band file): this lets subtitle text *flow as a stream*,
/// so a demuxed subtitle track or a network rendition can feed an overlay / sink
/// like any other media. The text-domain analog of a codec decoder, refining the
/// media type from a structured subtitle format to plain UTF-8 via the same
/// [`CapsConstraint::DerivedOutput`] negotiation.
///
/// Scope (v1): a `.srt` / `.vtt` is a small whole document, so this buffers the
/// sink bytes and emits every cue at `Eos` (batch, like `Mp4DemuxN`); incremental
/// cue-by-cue streaming is a follow-up. WebVTT cue positioning ([`CueSettings`])
/// is parsed but not carried on the frame yet (no text frame-meta); the payload
/// is the plain cue text.
#[derive(Debug, Default)]
pub struct SubParse {
    /// Input subtitle format, fixed at `configure_pipeline`.
    format: Option<TextFormat>,
    /// Accumulated sink bytes, parsed at `Eos`.
    buf: Vec<u8>,
    /// Whether the output `Caps::Text{Utf8}` has been announced downstream.
    caps_emitted: bool,
    sequence: u64,
}

impl SubParse {
    pub fn new() -> Self {
        Self::default()
    }

    /// The structured subtitle formats this element parses (its sink pad).
    fn input_alternatives() -> CapsSet {
        CapsSet::from_alternatives(Vec::from([
            Caps::Text { format: TextFormat::Srt },
            Caps::Text { format: TextFormat::WebVtt },
            Caps::Text { format: TextFormat::Ssa },
        ]))
    }

    fn output_caps() -> Caps {
        Caps::Text { format: TextFormat::Utf8 }
    }

    /// Parse the buffered document with the parser for the configured format.
    fn parse_buffered(&self) -> Vec<Cue> {
        let doc = String::from_utf8_lossy(&self.buf);
        match self.format {
            Some(TextFormat::WebVtt) => parse_webvtt(&doc),
            Some(TextFormat::Ssa) => parse_ssa(&doc),
            // SubRip is the default; the constraint admits only Srt / WebVtt / Ssa.
            _ => parse_srt(&doc),
        }
    }
}

impl AsyncElement for SubParse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Text { format: TextFormat::Srt | TextFormat::WebVtt | TextFormat::Ssa } => {
                Ok(upstream_caps.clone())
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Decoder-style: the output media type is derived from the input. A
    /// structured subtitle format in, plain UTF-8 out, so the solver negotiates
    /// `Text{Utf8}` onto the downstream link while the sink pad takes the SRT /
    /// WebVTT / SSA document.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Text { format: TextFormat::Srt | TextFormat::WebVtt | TextFormat::Ssa } => {
                CapsSet::one(Self::output_caps())
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::Text { format: format @ (TextFormat::Srt | TextFormat::WebVtt | TextFormat::Ssa) } => {
                self.format = Some(*format);
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Subtitle parser",
            "Codec/Parser/Subtitle",
            "Parses a SubRip / WebVTT document into timed UTF-8 text cues",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if self.format.is_none() {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if let MemoryDomain::System(slice) = &frame.domain {
                        // The parsers handle CRLF / BOM, so accumulate raw bytes.
                        self.buf.extend_from_slice(slice.as_slice());
                    }
                }
                // Output caps are negotiated up front (DerivedOutput) and announced
                // at the first cue; an inbound caps change on the SRT side is absorbed.
                PipelinePacket::CapsChanged(_) => {}
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Flush => {
                    self.buf.clear();
                    out.push(PipelinePacket::Flush).await?;
                }
                // Batch parse: the whole document is buffered by now. Emit one timed
                // UTF-8 frame per cue (the runner arm forwards the trailing Eos).
                PipelinePacket::Eos => {
                    for cue in self.parse_buffered() {
                        if !self.caps_emitted {
                            out.push(PipelinePacket::CapsChanged(Self::output_caps())).await?;
                            self.caps_emitted = true;
                        }
                        let timing = FrameTiming {
                            pts_ns: cue.start_ns,
                            duration_ns: cue.end_ns.saturating_sub(cue.start_ns),
                            ..Default::default()
                        };
                        let payload = cue.text.into_bytes().into_boxed_slice();
                        let frame = Frame::new(
                            MemoryDomain::System(SystemSlice::from_boxed(payload)),
                            timing,
                            self.sequence,
                        );
                        self.sequence += 1;
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                    }
                    self.buf.clear();
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for SubParse {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(Self::input_alternatives()),
            PadTemplate::source(CapsSet::one(Self::output_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_srt_and_webvtt_forms() {
        // SRT comma, full clock.
        assert_eq!(parse_timestamp("00:00:01,000"), Some(1_000_000_000));
        // WebVTT dot, full clock with millis.
        assert_eq!(parse_timestamp("01:02:03.500"), Some((3600 + 120 + 3) * 1_000_000_000 + 500_000_000));
        // WebVTT short form (no hours).
        assert_eq!(parse_timestamp("00:04.250"), Some(4_250_000_000));
        // Short fractional digits right-pad to millis.
        assert_eq!(parse_timestamp("00:00:00.5"), Some(500_000_000));
        // Out-of-range fields rejected.
        assert_eq!(parse_timestamp("00:99:00,000"), None);
        // An untrusted, unbounded hours field overflows to None, not a panic.
        assert_eq!(parse_timestamp("9999999999999999:00:00,000"), None);
    }

    #[test]
    fn srt_two_cues_with_multiline_text() {
        let input = "1\n00:00:01,000 --> 00:00:04,000\nHello world\n\n2\n00:00:05,000 --> 00:00:08,500\nSecond cue\nacross two lines\n";
        let cues = parse_srt(input);
        assert_eq!(cues.len(), 2);
        assert_eq!(
            cues[0],
            Cue {
                start_ns: 1_000_000_000,
                end_ns: 4_000_000_000,
                text: "Hello world".into(),
                settings: CueSettings::default(),
            }
        );
        assert_eq!(cues[1].text, "Second cue\nacross two lines");
        assert_eq!(cues[1].start_ns, 5_000_000_000);
        assert_eq!(cues[1].end_ns, 8_500_000_000);
    }

    #[test]
    fn webvtt_cue_settings_are_parsed() {
        let input = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000 position:20% line:80% align:start\ntop-left-ish\n\n00:00:02.000 --> 00:00:03.000 align:right\nright\n";
        let cues = parse_webvtt(input);
        assert_eq!(cues.len(), 2);
        assert_eq!(
            cues[0].settings,
            CueSettings { position: Some(20), line: Some(80), align: TextAlign::Start }
        );
        // Bare `align:right` maps to End; position / line stay auto.
        assert_eq!(
            cues[1].settings,
            CueSettings { position: None, line: None, align: TextAlign::End }
        );
    }

    #[test]
    fn webvtt_skips_header_note_and_strips_tags() {
        let input = "WEBVTT - Demo\nKind: captions\n\nNOTE this is a comment\nwith -->  a fake arrow\n\nintro\n00:00:00.000 --> 00:00:02.000 position:50%\n<v Speaker><i>Italic</i> text\n";
        let cues = parse_webvtt(input);
        assert_eq!(cues.len(), 1, "header and NOTE blocks skipped");
        assert_eq!(cues[0].start_ns, 0);
        assert_eq!(cues[0].end_ns, 2_000_000_000);
        assert_eq!(cues[0].text, "Italic text", "tags stripped, settings ignored");
    }

    #[test]
    fn crlf_and_bom_are_tolerated() {
        let input = "\u{feff}1\r\n00:00:01,000 --> 00:00:02,000\r\nLine\r\n";
        let cues = parse_srt(input);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "Line");
    }

    #[test]
    fn auto_detects_format() {
        assert_eq!(parse_auto("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n").len(), 1);
        assert_eq!(parse_auto("1\n00:00:01,000 --> 00:00:02,000\nhi\n").len(), 1);
        assert_eq!(
            parse_auto("[Script Info]\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,hi\n").len(),
            1,
            "a leading [ section header selects SSA",
        );
    }

    const ASS: &str = "[Script Info]\n\
        Title: demo\n\
        \n\
        [V4+ Styles]\n\
        Format: Name, Fontname\n\
        Style: Default,Arial\n\
        \n\
        [Events]\n\
        Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\
        Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,{\\i1}Hello{\\i0}, world\n\
        Dialogue: 0,0:01:02.50,0:01:05.00,Default,Bob,0,0,0,,Line one\\NLine two\n";

    #[test]
    fn ssa_reads_events_format_and_dialogue() {
        let cues = parse_ssa(ASS);
        assert_eq!(cues.len(), 2);
        // Centisecond fraction (.00) -> 0 ms; override tags stripped; the comma
        // inside the Text field is preserved (Text is the last column).
        assert_eq!(
            cues[0],
            Cue {
                start_ns: 1_000_000_000,
                end_ns: 4_000_000_000,
                text: "Hello, world".into(),
                settings: CueSettings::default(),
            }
        );
        // `\N` becomes a real line break; .50 centiseconds -> 500 ms.
        assert_eq!(cues[1].start_ns, 62_500_000_000);
        assert_eq!(cues[1].text, "Line one\nLine two");
    }

    #[test]
    fn ssa_honors_reordered_format_columns() {
        // Text not last in name order? It still must be the final column per spec;
        // here Start/End sit at non-default indices and are looked up by name.
        let doc = "[Events]\n\
            Format: Start, End, Text\n\
            Dialogue: 0:00:00.00,0:00:01.00,hi there\n";
        let cues = parse_ssa(doc);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].end_ns, 1_000_000_000);
        assert_eq!(cues[0].text, "hi there");
    }

    #[test]
    fn ssa_ignores_lines_outside_events() {
        // A `Format:` outside [Events] (the styles block) must not be mistaken
        // for the dialogue column order, and there are no dialogue lines.
        let doc = "[V4+ Styles]\nFormat: Name, Fontname\nStyle: Default,Arial\n";
        assert!(parse_ssa(doc).is_empty());
    }

    #[test]
    fn covers_is_half_open() {
        let cue = Cue { start_ns: 1000, end_ns: 2000, text: "x".into(), settings: CueSettings::default() };
        assert!(!cue.covers(999));
        assert!(cue.covers(1000));
        assert!(cue.covers(1999));
        assert!(!cue.covers(2000));
    }

    // -- SubParse element: drive process() directly. --------------------------

    use g2g_core::PushOutcome;

    #[derive(Default)]
    struct RecordingSink {
        packets: Vec<PipelinePacket>,
    }

    impl OutputSink for RecordingSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                self.packets.push(packet);
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn srt_bytes_frame(bytes: &[u8]) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        ))
    }

    #[test]
    fn element_negotiates_srt_to_utf8() {
        let el = SubParse::new();
        // Decoder-style: SRT/WebVTT in on the sink, UTF-8 derived on the source.
        assert_eq!(el.intercept_caps(&Caps::Text { format: TextFormat::Srt }).unwrap(),
            Caps::Text { format: TextFormat::Srt });
        assert!(el.intercept_caps(&Caps::Text { format: TextFormat::Utf8 }).is_err());
        let CapsConstraint::DerivedOutput(derive) = el.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let out = derive(&Caps::Text { format: TextFormat::WebVtt });
        assert_eq!(out.alternatives(), &[Caps::Text { format: TextFormat::Utf8 }]);
        // SSA negotiates the same way (also -> Utf8).
        assert_eq!(
            el.intercept_caps(&Caps::Text { format: TextFormat::Ssa }).unwrap(),
            Caps::Text { format: TextFormat::Ssa }
        );
    }

    #[tokio::test]
    async fn element_parses_ssa_to_timed_utf8() {
        let mut el = SubParse::new();
        el.configure_pipeline(&Caps::Text { format: TextFormat::Ssa }).expect("accepts SSA");

        let mut sink = RecordingSink::default();
        el.process(srt_bytes_frame(ASS.as_bytes()), &mut sink).await.unwrap();
        el.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let frames: Vec<&Frame> = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].timing.pts_ns, 1_000_000_000);
        if let MemoryDomain::System(s) = &frames[0].domain {
            assert_eq!(s.as_slice(), b"Hello, world");
        } else {
            panic!("cue payload must be a system buffer");
        }
    }

    #[tokio::test]
    async fn element_emits_caps_then_timed_cue_frames() {
        let doc = "1\n00:00:01,000 --> 00:00:04,000\nHello world\n\n\
                   2\n00:01:02,500 --> 00:01:05,000\nSecond cue\nacross two lines\n";
        let mut el = SubParse::new();
        el.configure_pipeline(&Caps::Text { format: TextFormat::Srt }).expect("accepts SRT");

        let mut sink = RecordingSink::default();
        // Two chunks then EOS, exercising the byte buffer.
        let (a, b) = doc.as_bytes().split_at(25);
        el.process(srt_bytes_frame(a), &mut sink).await.unwrap();
        el.process(srt_bytes_frame(b), &mut sink).await.unwrap();
        el.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        assert!(matches!(
            sink.packets.first(),
            Some(PipelinePacket::CapsChanged(Caps::Text { format: TextFormat::Utf8 }))
        ));
        let frames: Vec<&Frame> = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect();
        assert_eq!(frames.len(), 2, "one frame per cue");
        assert_eq!(frames[0].timing.pts_ns, 1_000_000_000);
        assert_eq!(frames[0].timing.duration_ns, 3_000_000_000);
        if let MemoryDomain::System(s) = &frames[0].domain {
            assert_eq!(s.as_slice(), b"Hello world");
        } else {
            panic!("cue payload must be a system buffer");
        }
        assert_eq!(frames[1].timing.pts_ns, 62_500_000_000);
    }

    #[test]
    fn unconfigured_element_errors() {
        // process() before configure_pipeline must fail loud, not silently buffer.
        let mut el = SubParse::new();
        let mut sink = RecordingSink::default();
        let r = futures_lite_block(el.process(PipelinePacket::Eos, &mut sink));
        assert!(matches!(r, Err(G2gError::NotConfigured)));
    }

    /// Minimal block-on for the single-poll futures these element calls produce
    /// (RecordingSink resolves immediately), avoiding a runtime dep in this test.
    fn futures_lite_block<F: Future>(fut: F) -> F::Output {
        use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        static VT: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(core::ptr::null(), &VT),
            |_| {},
            |_| {},
            |_| {},
        );
        // SAFETY: the vtable functions are no-ops that never deref the data pointer.
        let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VT)) };
        let mut cx = Context::from_waker(&waker);
        let mut fut = Box::pin(fut);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => v,
            Poll::Pending => panic!("future unexpectedly pending"),
        }
    }
}
