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

/// Writing mode for a cue (the WebVTT `vertical:` setting). The default is
/// horizontal (top-to-bottom lines, left-to-right text); the vertical modes are
/// used by CJK subtitles, growing columns right-to-left (`rl`) or left-to-right
/// (`lr`). Parsed and carried on the cue; the bitmap overlay does not yet lay
/// text out vertically, so it renders these horizontally for now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WritingMode {
    /// Horizontal lines (no `vertical:` setting), the default.
    #[default]
    Horizontal,
    /// Vertical columns growing right-to-left (`vertical:rl`).
    VerticalRl,
    /// Vertical columns growing left-to-right (`vertical:lr`).
    VerticalLr,
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
    /// Writing mode (WebVTT `vertical:`); default horizontal. Carried for CJK
    /// subtitles even though the bitmap overlay still lays text out horizontally.
    pub vertical: WritingMode,
    /// Opaque text RGBA from a WebVTT `STYLE` `::cue` `color:` rule, if any
    /// (resolved at parse time). `None` = the overlay's default text colour.
    pub color: Option<[u8; 4]>,
    /// Backing-box RGBA from a `::cue` `background-color:` rule, if any. A zero
    /// alpha (e.g. `transparent`) draws no box. `None` = the overlay's default.
    pub background: Option<[u8; 4]>,
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

/// Per-frame metadata carrying a streamed cue's placement (M406): the
/// [`CueSettings`] that `SubParse` parses but cannot put in the plain UTF-8
/// payload, so a downstream overlay recovers WebVTT / SSA positioning instead of
/// drawing every streamed cue bottom-centre. Gated behind the `metadata` feature
/// (a [`FrameMeta`] needs the typed container); the baseline carries no meta.
///
/// [`FrameMeta`]: g2g_core::FrameMeta
#[cfg(feature = "metadata")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextCueMeta {
    /// The cue's placement, as parsed from the subtitle format.
    pub settings: CueSettings,
}

#[cfg(feature = "metadata")]
impl g2g_core::FrameMeta for TextCueMeta {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn core::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn g2g_core::FrameMeta> {
        Box::new(*self)
    }
    // Placement is normalized (percent of frame width / height), so it survives
    // a downstream scale / crop unchanged; the default `Keep` is correct.
}

/// Parse SubRip (`.srt`) text into cues, in file order. Malformed blocks (no
/// timing line, unparseable timestamps) are skipped rather than failing the whole
/// parse, matching how players tolerate dirty subtitle files.
pub fn parse_srt(input: &str) -> Vec<Cue> {
    parse_blocks(input, false)
}

/// Parse WebVTT (`.vtt`) text into cues, in file order. The `WEBVTT` header and
/// `NOTE` / `REGION` blocks are skipped and inline markup is removed; `STYLE`
/// blocks are read for `::cue` / `::cue(#id)` `color` / `background-color` rules,
/// which are resolved onto each cue's [`CueSettings`] (the subset the overlay can
/// apply; other CSS properties and `::cue(.class)` span selectors are ignored).
pub fn parse_webvtt(input: &str) -> Vec<Cue> {
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);

    // Split into blank-line-separated blocks (kept whole for the two passes).
    let mut blocks: Vec<Vec<&str>> = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    for line in input.lines() {
        if line.trim().is_empty() {
            if !cur.is_empty() {
                blocks.push(core::mem::take(&mut cur));
            }
        } else {
            cur.push(line);
        }
    }
    if !cur.is_empty() {
        blocks.push(cur);
    }

    // Pass 1: collect the CSS from every `STYLE` block.
    let mut css = String::new();
    for b in &blocks {
        if b[0].trim_start().starts_with("STYLE") {
            for line in &b[1..] {
                css.push_str(line);
                css.push('\n');
            }
        }
    }
    let sheet = parse_cue_styles(&css);

    // Pass 2: parse the cue blocks, resolving each cue's style by its identifier.
    let mut cues = Vec::new();
    for b in &blocks {
        if let Some(mut cue) = block_to_cue(b, true) {
            if !sheet.is_empty() {
                apply_cue_style(&sheet, block_cue_id(b), &mut cue.settings);
            }
            cues.push(cue);
        }
    }
    cues
}

/// The WebVTT cue identifier (the line just before the timing line), if any.
/// `None` when the timing line opens the block (no identifier).
fn block_cue_id<'a>(block: &[&'a str]) -> Option<&'a str> {
    let timing_idx = block.iter().position(|l| l.contains("-->"))?;
    (timing_idx > 0).then(|| block[timing_idx - 1].trim())
}

/// A `::cue` selector we apply: all cues (`::cue`) or one identifier
/// (`::cue(#id)`). `::cue(.class)` / element selectors are not supported.
#[derive(Debug)]
enum CueSelector {
    All,
    Id(String),
}

/// One parsed `::cue` rule: its selectors and the `color` / `background-color`
/// it sets (the only properties the overlay can honour).
#[derive(Debug)]
struct CueStyleRule {
    selectors: Vec<CueSelector>,
    color: Option<[u8; 4]>,
    background: Option<[u8; 4]>,
}

/// Parse the WebVTT `STYLE` CSS into the supported `::cue` rules. Comments are
/// stripped; rules with no understood selector (a class / element rule) are
/// dropped. Hand-rolled (no CSS dep, stays on the `no_std` baseline).
fn parse_cue_styles(css: &str) -> Vec<CueStyleRule> {
    let css = strip_css_comments(css);
    let mut rules = Vec::new();
    let mut rest = css.as_str();
    while let Some(open) = rest.find('{') {
        let sel_str = &rest[..open];
        let after = &rest[open + 1..];
        let Some(close) = after.find('}') else { break };
        let decl_str = &after[..close];
        rest = &after[close + 1..];

        let selectors: Vec<CueSelector> =
            sel_str.split(',').filter_map(|s| parse_cue_selector(s.trim())).collect();
        if selectors.is_empty() {
            continue;
        }
        let mut color = None;
        let mut background = None;
        for decl in decl_str.split(';') {
            let Some((prop, val)) = decl.split_once(':') else {
                continue;
            };
            match prop.trim().to_ascii_lowercase().as_str() {
                "color" => color = parse_css_color(val.trim()),
                "background-color" | "background" => background = parse_css_color(val.trim()),
                _ => {}
            }
        }
        rules.push(CueStyleRule { selectors, color, background });
    }
    rules
}

/// A `::cue` or `::cue(#id)` selector, or `None` for anything else.
fn parse_cue_selector(sel: &str) -> Option<CueSelector> {
    if sel == "::cue" {
        return Some(CueSelector::All);
    }
    let inner = sel.strip_prefix("::cue(")?.strip_suffix(')')?.trim();
    let id = inner.strip_prefix('#')?;
    // Only a bare `#id` is supported (no compound / descendant selectors).
    (!id.is_empty() && !id.contains(|c: char| c.is_whitespace()))
        .then(|| CueSelector::Id(id.into()))
}

/// Resolve a cue's `color` / `background` from the sheet: global `::cue` rules
/// first, then `::cue(#id)` so an id rule overrides the global one.
fn apply_cue_style(sheet: &[CueStyleRule], id: Option<&str>, settings: &mut CueSettings) {
    let apply = |rule: &CueStyleRule, s: &mut CueSettings| {
        if rule.color.is_some() {
            s.color = rule.color;
        }
        if rule.background.is_some() {
            s.background = rule.background;
        }
    };
    for rule in sheet {
        if rule.selectors.iter().any(|sel| matches!(sel, CueSelector::All)) {
            apply(rule, settings);
        }
    }
    if let Some(id) = id {
        for rule in sheet {
            if rule.selectors.iter().any(|sel| matches!(sel, CueSelector::Id(rid) if rid == id)) {
                apply(rule, settings);
            }
        }
    }
}

/// Strip `/* ... */` comments from a CSS string.
fn strip_css_comments(css: &str) -> String {
    let mut out = String::with_capacity(css.len());
    let mut rest = css;
    while let Some(start) = rest.find("/*") {
        out.push_str(&rest[..start]);
        match rest[start + 2..].find("*/") {
            Some(end) => rest = &rest[start + 2 + end + 2..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Parse a CSS colour value to opaque-or-alpha RGBA: `transparent`, `#rgb` /
/// `#rrggbb`, `rgb(...)` / `rgba(...)`, or a small set of named colours. `None`
/// for anything unrecognised (the cue keeps the overlay default).
fn parse_css_color(v: &str) -> Option<[u8; 4]> {
    let v = v.trim();
    if v.eq_ignore_ascii_case("transparent") {
        return Some([0, 0, 0, 0]);
    }
    if let Some(hex) = v.strip_prefix('#') {
        return parse_hex_color(hex);
    }
    if let Some(rest) = v.strip_prefix("rgba(").or_else(|| v.strip_prefix("rgb(")) {
        let rest = rest.strip_suffix(')')?;
        let mut it = rest.split(',');
        let mut chan = || it.next()?.trim().parse::<u32>().ok().map(|n| n.min(255) as u8);
        let r = chan()?;
        let g = chan()?;
        let b = chan()?;
        let a = match it.next() {
            Some(a) => (a.trim().parse::<f32>().ok()?.clamp(0.0, 1.0) * 255.0) as u8,
            None => 255,
        };
        return Some([r, g, b, a]);
    }
    named_css_color(v)
}

/// Parse a `#rgb` or `#rrggbb` hex colour to opaque RGBA.
fn parse_hex_color(hex: &str) -> Option<[u8; 4]> {
    let hex = hex.trim();
    match hex.len() {
        6 => Some([
            u8::from_str_radix(&hex[0..2], 16).ok()?,
            u8::from_str_radix(&hex[2..4], 16).ok()?,
            u8::from_str_radix(&hex[4..6], 16).ok()?,
            255,
        ]),
        3 => {
            let dup = |c: &str| u8::from_str_radix(c, 16).ok().map(|v| v * 16 + v);
            Some([dup(&hex[0..1])?, dup(&hex[1..2])?, dup(&hex[2..3])?, 255])
        }
        _ => None,
    }
}

/// A small set of CSS named colours (the ones subtitles realistically use).
fn named_css_color(name: &str) -> Option<[u8; 4]> {
    Some(match name.to_ascii_lowercase().as_str() {
        "black" => [0, 0, 0, 255],
        "white" => [255, 255, 255, 255],
        "red" => [255, 0, 0, 255],
        "lime" => [0, 255, 0, 255],
        "green" => [0, 128, 0, 255],
        "blue" => [0, 0, 255, 255],
        "yellow" => [255, 255, 0, 255],
        "cyan" | "aqua" => [0, 255, 255, 255],
        "magenta" | "fuchsia" => [255, 0, 255, 255],
        "gray" | "grey" => [128, 128, 128, 255],
        _ => return None,
    })
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
    } else if trimmed.starts_with('<') {
        // TTML / DFXP is XML, opening with `<?xml ...` or the `<tt>` root.
        parse_ttml(input)
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
    let mut state = SsaState::default();
    let mut cues = Vec::new();
    for line in input.lines() {
        if let Some(cue) = state.feed_line(line) {
            cues.push(cue);
        }
    }
    cues
}

/// Per-line SSA parse state, shared by the whole-document [`parse_ssa`] and the
/// streaming `SubParse` element: whether the scan is inside the `[Events]`
/// section and the resolved Start / End / Text column indices. Held across
/// chunks so a `Dialogue:` line in a later chunk parses with the column order
/// declared by an earlier one.
#[derive(Debug, Clone)]
struct SsaState {
    in_events: bool,
    i_start: usize,
    i_end: usize,
    i_text: usize,
}

impl Default for SsaState {
    fn default() -> Self {
        // V4+ default column order, used until an explicit `Format:` line
        // overrides it: Layer, Start, End, Style, Name, MarginL, MarginR,
        // MarginV, Effect, Text.
        Self { in_events: false, i_start: 1, i_end: 2, i_text: 9 }
    }
}

impl SsaState {
    /// Feed one SSA line, returning a cue if it is a `Dialogue:` line in the
    /// `[Events]` section. Section headers and `Format:` lines update the state.
    fn feed_line(&mut self, line: &str) -> Option<Cue> {
        let line = line.trim();
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            self.in_events = name.eq_ignore_ascii_case("Events");
            return None;
        }
        if !self.in_events {
            return None;
        }
        if let Some(rest) = strip_prefix_ci(line, "Format:") {
            let cols: Vec<&str> = rest.split(',').map(str::trim).collect();
            self.i_start = col_index(&cols, "Start").unwrap_or(self.i_start);
            self.i_end = col_index(&cols, "End").unwrap_or(self.i_end);
            // Text is the last column by spec; fall back to that if unnamed.
            self.i_text = col_index(&cols, "Text").unwrap_or(cols.len().saturating_sub(1));
            None
        } else if let Some(rest) = strip_prefix_ci(line, "Dialogue:") {
            parse_ass_dialogue(rest, self.i_start, self.i_end, self.i_text)
        } else {
            None
        }
    }
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

/// Parse TTML / DFXP (W3C Timed Text, also SMPTE-TT / EBU-TT / IMSC) into cues.
/// TTML is XML; rather than a full parser this scans for `<p>` paragraph elements
/// (any namespace prefix), reading their `begin` / `end` time attributes and text
/// content. Inline markup (`<span>`...) is stripped, `<br/>` becomes a newline,
/// XML entities (`&amp;` / `&#10;`...) are decoded, and insignificant XML
/// whitespace is collapsed (TTML default `xml:space`). Times accept clock-time
/// (`HH:MM:SS.fff`) and offset-time (`5s` / `1.5s` / `400ms` / `2m` / `1h`);
/// frame / tick offsets (need a frame / tick rate) are not supported. Malformed
/// paragraphs are skipped, and the scan never indexes off a char boundary, so
/// untrusted markup fails safe rather than panicking.
pub fn parse_ttml(input: &str) -> Vec<Cue> {
    let mut cues = Vec::new();
    let mut rest = input;
    // Walk `<p ...> ... </p>` spans. `<p>` does not nest in TTML, so the next
    // close tag terminates the current paragraph.
    while let Some((attrs, body, after)) = next_paragraph(rest) {
        rest = after;
        let (Some(begin), Some(end)) = (xml_attr(attrs, "begin"), xml_attr(attrs, "end")) else {
            continue;
        };
        let (Some(start_ns), Some(end_ns)) = (parse_ttml_time(begin), parse_ttml_time(end)) else {
            continue;
        };
        let text = ttml_text(body);
        if !text.trim().is_empty() {
            cues.push(Cue { start_ns, end_ns, text, settings: CueSettings::default() });
        }
    }
    cues
}

/// Find the next `<p ...>` paragraph: returns its attribute string, its inner
/// content (up to the matching `</p>`), and the remainder after the close tag.
/// Matches the `p` local name under any namespace prefix; skips self-closing
/// `<p/>` (no content) and any non-`p` tag.
fn next_paragraph(input: &str) -> Option<(&str, &str, &str)> {
    let mut scan = input;
    loop {
        let lt = scan.find('<')?;
        let after_lt = &scan[lt + 1..];
        let gt = after_lt.find('>')?;
        let tag = &after_lt[..gt]; // between '<' and '>'
        let after_tag = &after_lt[gt + 1..];
        // Tag name is up to the first whitespace / '/' ; strip a namespace prefix.
        let name = tag.trim_start_matches('/');
        let name = name.split([' ', '\t', '\r', '\n', '/']).next().unwrap_or("");
        let local = name.rsplit(':').next().unwrap_or(name);
        if local.eq_ignore_ascii_case("p") && !tag.starts_with('/') {
            // Open <p ...>. Self-closing (`<p/>`) has no body.
            if tag.ends_with('/') {
                scan = after_tag;
                continue;
            }
            let attrs = &tag[name.len()..];
            let close = find_paragraph_close(after_tag)?;
            let body = &after_tag[..close.0];
            return Some((attrs, body, &after_tag[close.1..]));
        }
        scan = after_tag;
    }
}

/// Find the next `</p>` close tag (any namespace prefix) in `s`; returns
/// `(content_end, after_close)` byte offsets into `s`.
fn find_paragraph_close(s: &str) -> Option<(usize, usize)> {
    let mut from = 0;
    loop {
        let lt = s[from..].find("</")? + from;
        let after = &s[lt + 2..];
        let gt = after.find('>')?;
        let name = after[..gt].trim();
        let local = name.rsplit(':').next().unwrap_or(name);
        if local.eq_ignore_ascii_case("p") {
            return Some((lt, lt + 2 + gt + 1));
        }
        from = lt + 2;
    }
}

/// Read an XML attribute value (`name="..."` or `name='...'`) from a tag's
/// attribute string. `None` if the attribute is absent or unquoted.
fn xml_attr<'a>(attrs: &'a str, name: &str) -> Option<&'a str> {
    let mut from = 0;
    while let Some(pos) = attrs[from..].find(name) {
        let at = from + pos;
        let before_ok = at == 0
            || attrs[..at].chars().next_back().map(|c| c.is_whitespace()).unwrap_or(true);
        let after = attrs[at + name.len()..].trim_start();
        if before_ok {
            if let Some(rest) = after.strip_prefix('=') {
                let rest = rest.trim_start();
                let quote = rest.chars().next()?;
                if quote == '"' || quote == '\'' {
                    let val = &rest[1..];
                    let end = val.find(quote)?;
                    return Some(&val[..end]);
                }
            }
        }
        from = at + name.len();
    }
    None
}

/// Extract the plain text of a TTML paragraph body: strip inline tags, map
/// `<br/>` to a newline, decode entities, and collapse insignificant whitespace.
fn ttml_text(body: &str) -> String {
    let mut out = String::new();
    let mut rest = body;
    while let Some(lt) = rest.find('<') {
        push_collapsed(&mut out, &decode_entities(&rest[..lt]));
        let after = &rest[lt + 1..];
        let Some(gt) = after.find('>') else { break };
        let tag = &after[..gt];
        let name = tag.trim_start_matches('/');
        let name = name.split([' ', '\t', '\r', '\n', '/']).next().unwrap_or("");
        let local = name.rsplit(':').next().unwrap_or(name);
        if local.eq_ignore_ascii_case("br") {
            // A hard line break: drop a trailing collapse-space first.
            while out.ends_with(' ') {
                out.pop();
            }
            out.push('\n');
        }
        rest = &after[gt + 1..];
    }
    push_collapsed(&mut out, &decode_entities(rest));
    out.trim().into()
}

/// Append `text` to `out` collapsing every run of XML whitespace (spaces, tabs,
/// and the newlines / indentation of pretty-printed markup) to a single space.
/// A `\n` already in `out` (from a `<br/>`) suppresses the leading space.
fn push_collapsed(out: &mut String, text: &str) {
    for c in text.chars() {
        if c.is_whitespace() {
            if !out.is_empty() && !out.ends_with(' ') && !out.ends_with('\n') {
                out.push(' ');
            }
        } else {
            out.push(c);
        }
    }
}

/// Decode the XML predefined entities and numeric character references. An
/// unrecognised `&...;` is left verbatim (a lone `&` is common in dirty input).
fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.into();
    }
    let mut out = String::new();
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let after = &rest[amp + 1..];
        match after.find(';') {
            Some(semi) => {
                let ent = &after[..semi];
                let decoded = match ent {
                    "amp" => Some('&'),
                    "lt" => Some('<'),
                    "gt" => Some('>'),
                    "quot" => Some('"'),
                    "apos" => Some('\''),
                    _ if ent.starts_with("#x") || ent.starts_with("#X") => {
                        u32::from_str_radix(&ent[2..], 16).ok().and_then(char::from_u32)
                    }
                    _ if ent.starts_with('#') => {
                        ent[1..].parse::<u32>().ok().and_then(char::from_u32)
                    }
                    _ => None,
                };
                match decoded {
                    Some(c) => {
                        out.push(c);
                        rest = &after[semi + 1..];
                    }
                    None => {
                        out.push('&');
                        rest = after;
                    }
                }
            }
            None => {
                out.push('&');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Parse a TTML time expression to nanoseconds: clock-time (`HH:MM:SS.fff`, via
/// the shared [`parse_timestamp`]) or offset-time (`<value><metric>` with metric
/// `h` / `m` / `s` / `ms`). Frame (`f`) and tick (`t`) metrics need a rate and
/// are unsupported (`None`). Untrusted: folded with checked / `u128` arithmetic.
fn parse_ttml_time(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.contains(':') {
        return parse_timestamp(s);
    }
    let metric_at = s.find(|c: char| c.is_ascii_alphabetic())?;
    let (num, metric) = s.split_at(metric_at);
    let unit_ns: u64 = match metric {
        "h" => 3_600_000_000_000,
        "m" => 60_000_000_000,
        "s" => 1_000_000_000,
        "ms" => 1_000_000,
        _ => return None,
    };
    let (int_part, frac_part) = num.split_once('.').unwrap_or((num, ""));
    let whole = int_part.parse::<u64>().ok()?.checked_mul(unit_ns)?;
    let frac_ns = frac_of_unit_ns(frac_part, unit_ns);
    whole.checked_add(frac_ns)
}

/// Fractional part of an offset-time as nanoseconds: `0.frac * unit_ns`, computed
/// in `u128` to avoid overflow, with the fraction capped at 9 digits.
fn frac_of_unit_ns(frac: &str, unit_ns: u64) -> u64 {
    let frac: alloc::string::String = frac.chars().take_while(|c| c.is_ascii_digit()).take(9).collect();
    if frac.is_empty() {
        return 0;
    }
    let Ok(frac_int) = frac.parse::<u64>() else { return 0 };
    let denom = 10u128.pow(frac.len() as u32);
    ((unit_ns as u128 * frac_int as u128) / denom) as u64
}

/// Length of the longest valid-UTF8 prefix of `buf`. A multi-byte char may be
/// split across a chunk boundary; only this prefix is safe to parse, and the
/// trailing partial-char bytes wait in the buffer for the next chunk.
fn utf8_prefix_len(buf: &[u8]) -> usize {
    match core::str::from_utf8(buf) {
        Ok(s) => s.len(),
        Err(e) => e.valid_up_to(),
    }
}

/// Byte offset just past the last blank-line separator in `s` (a line empty
/// after trimming), i.e. the end of the last fully terminated block. `None` if
/// no blank line has arrived yet, so no block is known complete.
fn last_block_boundary(s: &str) -> Option<usize> {
    let mut boundary = None;
    let mut line_start = 0;
    for (i, &b) in s.as_bytes().iter().enumerate() {
        if b == b'\n' {
            if s[line_start..i].trim().is_empty() {
                boundary = Some(i + 1);
            }
            line_start = i + 1;
        }
    }
    boundary
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
/// Recognises `position`, `line` (percentage form), `align`, and `vertical`
/// (writing mode); `size` and `region` are accepted but not applied.
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
            "vertical" => {
                if let Some(v) = parse_vertical(val) {
                    s.vertical = v;
                }
            }
            _ => {}
        }
    }
    s
}

/// Parse a `vertical:` value: `rl` (right-to-left columns) or `lr`. An
/// unrecognised value leaves the cue horizontal.
fn parse_vertical(v: &str) -> Option<WritingMode> {
    match v.split(',').next()?.trim() {
        "rl" => Some(WritingMode::VerticalRl),
        "lr" => Some(WritingMode::VerticalLr),
        _ => None,
    }
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
/// Streaming (M405): SRT / WebVTT / SSA are line based, so complete cues are
/// emitted as the bytes arrive. Each `process` call drains the blocks (or SSA
/// `Dialogue:` lines) that are fully terminated, retains the trailing partial
/// block in the buffer, and flushes the remainder at `Eos`. This unblocks a
/// downstream overlay, which no longer has to buffer video until the subtitle
/// stream ends. TTML is XML with no blank-line block boundary, so it stays
/// batch: its cues are parsed at `Eos`. WebVTT cue positioning ([`CueSettings`])
/// is parsed but not yet carried on the frame (no text frame-meta); the payload
/// is the plain cue text.
#[derive(Debug, Default)]
pub struct SubParse {
    /// Input subtitle format, fixed at `configure_pipeline`.
    format: Option<TextFormat>,
    /// Sink bytes not yet forming a complete cue, carried to the next chunk.
    buf: Vec<u8>,
    /// SSA `[Events]` / column-order state, persisted across chunks.
    ssa: SsaState,
    /// Whether a leading UTF-8 BOM has been resolved (consumed or ruled out).
    bom_stripped: bool,
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
            Caps::Text { format: TextFormat::Ttml },
        ]))
    }

    fn output_caps() -> Caps {
        Caps::Text { format: TextFormat::Utf8 }
    }

    /// Consume a leading UTF-8 BOM from the buffer the first time enough bytes
    /// (3) have arrived to recognise it; mark it resolved either way.
    fn strip_bom(&mut self) {
        if self.bom_stripped {
            return;
        }
        if self.buf.starts_with(&[0xEF, 0xBB, 0xBF]) {
            self.buf.drain(..3);
            self.bom_stripped = true;
        } else if self.buf.len() >= 3 {
            // Three bytes in and not a BOM, so there is none to strip.
            self.bom_stripped = true;
        }
    }

    /// Drain the cues now known complete, leaving any partial trailing block in
    /// the buffer. At `final_flush` (`Eos`) the whole remainder is parsed and the
    /// buffer cleared. TTML is XML (no blank-line boundary) and only parses at
    /// the flush; the line-based formats stream incrementally.
    fn drain_cues(&mut self, final_flush: bool) -> Vec<Cue> {
        match self.format {
            Some(TextFormat::Ttml) => {
                if !final_flush {
                    return Vec::new();
                }
                let doc = String::from_utf8_lossy(&self.buf);
                let cues = parse_ttml(&doc);
                self.buf.clear();
                cues
            }
            Some(TextFormat::Ssa) => self.drain_ssa(final_flush),
            Some(TextFormat::WebVtt) => self.drain_blocks(true, final_flush),
            // SubRip is the default; the constraint admits only the four formats.
            _ => self.drain_blocks(false, final_flush),
        }
    }

    /// Drain complete blank-line-separated blocks (SRT / WebVTT). On `final_flush`
    /// the whole buffer is one last parse; otherwise only blocks terminated by a
    /// blank line are taken and the partial tail is retained.
    fn drain_blocks(&mut self, webvtt: bool, final_flush: bool) -> Vec<Cue> {
        if final_flush {
            let doc = String::from_utf8_lossy(&self.buf);
            let cues = parse_blocks(&doc, webvtt);
            self.buf.clear();
            return cues;
        }
        let valid = utf8_prefix_len(&self.buf);
        let s = core::str::from_utf8(&self.buf[..valid]).expect("valid_up_to is a char boundary");
        let Some(boundary) = last_block_boundary(s) else {
            return Vec::new();
        };
        let cues = parse_blocks(&s[..boundary], webvtt);
        self.buf.drain(..boundary);
        cues
    }

    /// Drain complete SSA lines (newline-terminated), keeping the column-order
    /// state across chunks. On `final_flush` the partial tail line is parsed too
    /// (end of input terminates it).
    fn drain_ssa(&mut self, final_flush: bool) -> Vec<Cue> {
        let mut cues = Vec::new();
        if final_flush {
            let doc = String::from_utf8_lossy(&self.buf);
            for line in doc.lines() {
                if let Some(cue) = self.ssa.feed_line(line) {
                    cues.push(cue);
                }
            }
            self.buf.clear();
            return cues;
        }
        let valid = utf8_prefix_len(&self.buf);
        let s = core::str::from_utf8(&self.buf[..valid]).expect("valid_up_to is a char boundary");
        let Some(nl) = s.rfind('\n') else {
            return Vec::new();
        };
        for line in s[..=nl].lines() {
            if let Some(cue) = self.ssa.feed_line(line) {
                cues.push(cue);
            }
        }
        self.buf.drain(..=nl);
        cues
    }
}

impl AsyncElement for SubParse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Text {
                format: TextFormat::Srt | TextFormat::WebVtt | TextFormat::Ssa | TextFormat::Ttml,
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Decoder-style: the output media type is derived from the input. A
    /// structured subtitle format in, plain UTF-8 out, so the solver negotiates
    /// `Text{Utf8}` onto the downstream link while the sink pad takes the SRT /
    /// WebVTT / SSA document.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Text {
                format: TextFormat::Srt | TextFormat::WebVtt | TextFormat::Ssa | TextFormat::Ttml,
            } => CapsSet::one(Self::output_caps()),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::Text {
                format:
                    format @ (TextFormat::Srt
                    | TextFormat::WebVtt
                    | TextFormat::Ssa
                    | TextFormat::Ttml),
            } => {
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
            // Drain whatever cues are now complete; emit them below. A DataFrame
            // streams the just-terminated cues, Eos flushes the trailing block.
            let cues = match packet {
                PipelinePacket::DataFrame(frame) => {
                    if let MemoryDomain::System(slice) = &frame.domain {
                        // The parsers handle CRLF / BOM, so accumulate raw bytes.
                        self.buf.extend_from_slice(slice.as_slice());
                    }
                    self.strip_bom();
                    self.drain_cues(false)
                }
                // Output caps are negotiated up front (DerivedOutput) and announced
                // at the first cue; an inbound caps change on the SRT side is absorbed.
                PipelinePacket::CapsChanged(_) => Vec::new(),
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                    Vec::new()
                }
                PipelinePacket::Flush => {
                    self.buf.clear();
                    self.ssa = SsaState::default();
                    self.bom_stripped = false;
                    out.push(PipelinePacket::Flush).await?;
                    Vec::new()
                }
                // The trailing partial block is parsed now; the runner arm forwards
                // the trailing Eos.
                PipelinePacket::Eos => self.drain_cues(true),
            };
            for cue in cues {
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
                #[cfg_attr(not(feature = "metadata"), allow(unused_mut))]
                let mut frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(payload)),
                    timing,
                    self.sequence,
                );
                // Carry the cue placement as frame-meta so an overlay can honour
                // WebVTT / SSA positioning (no-op on the ZST baseline).
                #[cfg(feature = "metadata")]
                frame.meta.attach(TextCueMeta { settings: cue.settings });
                self.sequence += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
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
            CueSettings {
                position: Some(20),
                line: Some(80),
                align: TextAlign::Start,
                vertical: WritingMode::Horizontal,
                color: None,
                background: None,
            }
        );
        // Bare `align:right` maps to End; position / line stay auto.
        assert_eq!(
            cues[1].settings,
            CueSettings {
                position: None,
                line: None,
                align: TextAlign::End,
                vertical: WritingMode::Horizontal,
                color: None,
                background: None,
            }
        );
    }

    #[test]
    fn webvtt_cue_style_color_and_background() {
        // A global `::cue` rule plus an id override (the `id_selectors` shape),
        // with a CSS comment, hex, rgba, named, and transparent colours.
        let input = "WEBVTT\n\n\
            STYLE\n\
            ::cue { color: black; background-color: transparent; }\n\
            ::cue(#cue1), ::cue(#cue2) { color: white; background-color: rgba(0,0,0,1.0); }\n\
            ::cue(#cue3) { /* gold */ color: #A28849; }\n\
            .ignored { color: red; }\n\n\
            cue1\n00:00:00.000 --> 00:00:01.000\nwhite on black\n\n\
            cue3\n00:00:01.000 --> 00:00:02.000\ngold, no box\n\n\
            cue9\n00:00:02.000 --> 00:00:03.000\nglobal black\n";
        let cues = parse_webvtt(input);
        assert_eq!(cues.len(), 3);
        // cue1: id override -> white text on opaque black box.
        assert_eq!(cues[0].settings.color, Some([255, 255, 255, 255]));
        assert_eq!(cues[0].settings.background, Some([0, 0, 0, 255]));
        // cue3: gold text; background falls back to the global transparent.
        assert_eq!(cues[1].settings.color, Some([0xA2, 0x88, 0x49, 255]));
        assert_eq!(cues[1].settings.background, Some([0, 0, 0, 0]));
        // cue9: no id rule -> the global ::cue (black on transparent).
        assert_eq!(cues[2].settings.color, Some([0, 0, 0, 255]));
        assert_eq!(cues[2].settings.background, Some([0, 0, 0, 0]));
    }

    #[test]
    fn webvtt_without_style_has_no_cue_colors() {
        let cues = parse_webvtt("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nplain\n");
        assert_eq!(cues[0].settings.color, None);
        assert_eq!(cues[0].settings.background, None);
    }

    #[test]
    fn webvtt_vertical_writing_mode_is_parsed() {
        // The CJK case: `vertical:rl` columns plus placement, as in the real
        // Japanese fixture. The token is carried even though the bitmap overlay
        // still lays text out horizontally.
        let input = "WEBVTT\n\n00:00:05.000 --> 00:00:10.000 position:90% align:end line:10% vertical:rl\n縦書き\n\n00:00:10.000 --> 00:00:12.000 vertical:lr\n左書き\n";
        let cues = parse_webvtt(input);
        assert_eq!(cues.len(), 2);
        assert_eq!(
            cues[0].settings,
            CueSettings {
                position: Some(90),
                line: Some(10),
                align: TextAlign::End,
                vertical: WritingMode::VerticalRl,
                color: None,
                background: None,
            }
        );
        assert_eq!(cues[1].settings.vertical, WritingMode::VerticalLr);
        // A cue with no `vertical:` token stays horizontal.
        let plain = parse_webvtt("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nx\n");
        assert_eq!(plain[0].settings.vertical, WritingMode::Horizontal);
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

    const TTML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<tt xmlns="http://www.w3.org/ns/ttml" xml:lang="en">
  <body>
    <div>
      <p begin="00:00:01.000" end="00:00:04.000">Hello &amp; <span>world</span></p>
      <p begin="00:01:02.500" end="00:01:05.000">Line one<br/>Line two</p>
      <p begin="5s" end="7.5s">offset time</p>
    </div>
  </body>
</tt>"#;

    #[test]
    fn ttml_reads_paragraph_cues() {
        let cues = parse_ttml(TTML);
        assert_eq!(cues.len(), 3);
        // Entity decoded, inline <span> stripped, XML whitespace collapsed.
        assert_eq!(
            cues[0],
            Cue {
                start_ns: 1_000_000_000,
                end_ns: 4_000_000_000,
                text: "Hello & world".into(),
                settings: CueSettings::default(),
            }
        );
        // <br/> -> newline.
        assert_eq!(cues[1].start_ns, 62_500_000_000);
        assert_eq!(cues[1].text, "Line one\nLine two");
    }

    #[test]
    fn ttml_offset_time() {
        // The third cue uses offset-time (5s .. 7.5s).
        let cues = parse_ttml(TTML);
        assert_eq!(cues[2].start_ns, 5_000_000_000);
        assert_eq!(cues[2].end_ns, 7_500_000_000);
        assert_eq!(cues[2].text, "offset time");
    }

    #[test]
    fn ttml_namespace_prefixed_tags() {
        // A `tt:` prefix on the paragraph + break must still match (local name).
        let doc = r#"<tt:tt><tt:body><tt:p begin="0s" end="1s">hi<tt:br/>there</tt:p></tt:body></tt:tt>"#;
        let cues = parse_ttml(doc);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "hi\nthere");
    }

    #[test]
    fn ttml_skips_paragraph_with_bad_time() {
        let doc = r#"<p begin="nope" end="1s">x</p><p begin="0s" end="1s">ok</p>"#;
        let cues = parse_ttml(doc);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "ok");
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
        // SSA and TTML negotiate the same way (also -> Utf8).
        assert_eq!(
            el.intercept_caps(&Caps::Text { format: TextFormat::Ssa }).unwrap(),
            Caps::Text { format: TextFormat::Ssa }
        );
        assert_eq!(
            el.intercept_caps(&Caps::Text { format: TextFormat::Ttml }).unwrap(),
            Caps::Text { format: TextFormat::Ttml }
        );
    }

    #[tokio::test]
    async fn element_parses_ttml_to_timed_utf8() {
        let mut el = SubParse::new();
        el.configure_pipeline(&Caps::Text { format: TextFormat::Ttml }).expect("accepts TTML");

        let mut sink = RecordingSink::default();
        el.process(srt_bytes_frame(TTML.as_bytes()), &mut sink).await.unwrap();
        el.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let frames: Vec<&Frame> = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].timing.pts_ns, 1_000_000_000);
        if let MemoryDomain::System(s) = &frames[0].domain {
            assert_eq!(s.as_slice(), b"Hello & world");
        } else {
            panic!("cue payload must be a system buffer");
        }
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

    #[tokio::test]
    async fn element_streams_terminated_cue_before_eos() {
        // A complete first cue (terminated by a blank line) then a dangling
        // second cue arrive in one chunk; the complete one streams out at once.
        let mut el = SubParse::new();
        el.configure_pipeline(&Caps::Text { format: TextFormat::Srt }).unwrap();
        let mut sink = RecordingSink::default();

        el.process(
            srt_bytes_frame(b"1\n00:00:01,000 --> 00:00:02,000\nfirst\n\n2\n00:00:03,000 -->"),
            &mut sink,
        )
        .await
        .unwrap();

        let count = |sink: &RecordingSink| {
            sink.packets.iter().filter(|p| matches!(p, PipelinePacket::DataFrame(_))).count()
        };
        assert_eq!(count(&sink), 1, "the terminated cue is emitted before Eos");

        // Eos cannot complete the dangling second cue (no end timestamp/text).
        el.process(PipelinePacket::Eos, &mut sink).await.unwrap();
        assert_eq!(count(&sink), 1);
    }

    #[tokio::test]
    async fn element_streams_across_utf8_char_split() {
        // A multi-byte char split across the chunk boundary must not corrupt the
        // cue, and the earlier complete cue must still stream immediately.
        let mut el = SubParse::new();
        el.configure_pipeline(&Caps::Text { format: TextFormat::Srt }).unwrap();
        let mut sink = RecordingSink::default();

        let mut chunk1 = Vec::from(
            &b"1\n00:00:01,000 --> 00:00:02,000\nokay\n\n2\n00:00:03,000 --> 00:00:04,000\ncaf"[..],
        );
        chunk1.push(0xC3); // first byte of 'e-acute', completed in the next chunk
        el.process(srt_bytes_frame(&chunk1), &mut sink).await.unwrap();
        let after_chunk1 =
            sink.packets.iter().filter(|p| matches!(p, PipelinePacket::DataFrame(_))).count();
        assert_eq!(after_chunk1, 1, "the terminated cue streams before the rest arrives");

        el.process(srt_bytes_frame(&[0xA9, b'\n', b'\n']), &mut sink).await.unwrap();
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
        if let MemoryDomain::System(s) = &frames[1].domain {
            assert_eq!(core::str::from_utf8(s.as_slice()).unwrap(), "café");
        } else {
            panic!("cue payload must be a system buffer");
        }
    }

    #[cfg(feature = "metadata")]
    #[tokio::test]
    async fn element_attaches_cue_positioning_meta() {
        // WebVTT placement is parsed into CueSettings; the element carries it on
        // the cue frame as TextCueMeta so an overlay recovers it (M406).
        let doc = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000 position:20% line:80% align:start\nplaced\n\n";
        let mut el = SubParse::new();
        el.configure_pipeline(&Caps::Text { format: TextFormat::WebVtt }).unwrap();
        let mut sink = RecordingSink::default();
        el.process(srt_bytes_frame(doc.as_bytes()), &mut sink).await.unwrap();
        el.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let frame = sink
            .packets
            .iter()
            .find_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .expect("a cue frame");
        let meta = frame.meta.get::<TextCueMeta>().expect("cue carries placement meta");
        assert_eq!(
            meta.settings,
            CueSettings {
                position: Some(20),
                line: Some(80),
                align: TextAlign::Start,
                vertical: WritingMode::Horizontal,
                color: None,
                background: None,
            }
        );
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
