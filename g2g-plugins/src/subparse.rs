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

use alloc::string::String;
use alloc::vec::Vec;

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
/// signature (after an optional BOM) selects WebVTT, otherwise SRT.
pub fn parse_auto(input: &str) -> Vec<Cue> {
    let trimmed = input.strip_prefix('\u{feff}').unwrap_or(input).trim_start();
    if trimmed.starts_with("WEBVTT") {
        parse_webvtt(input)
    } else {
        parse_srt(input)
    }
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
    Some(total_ms.checked_mul(1_000_000)?)
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
    }

    #[test]
    fn covers_is_half_open() {
        let cue = Cue { start_ns: 1000, end_ns: 2000, text: "x".into(), settings: CueSettings::default() };
        assert!(!cue.covers(999));
        assert!(cue.covers(1000));
        assert!(cue.covers(1999));
        assert!(!cue.covers(2000));
    }
}
