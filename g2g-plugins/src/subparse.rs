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
//!   `REGION` blocks are skipped (a `NOTE` may itself contain `-->`). A header
//!   block's `X-TIMESTAMP-MAP` rebases the cue times that follow it.
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

/// One styled run of a cue's text: the `[start, end)` byte range of
/// [`Cue::text`] that a class-carrying WebVTT span (`<c.loud>...</c>`) covers,
/// and the colour the `::cue(.loud)` rules resolved for it. Runs are in document
/// order and may nest, so where two overlap the later one wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanStyle {
    /// Byte offset of the span's first character in [`Cue::text`].
    pub start: usize,
    /// Byte offset just past the span's last character.
    pub end: usize,
    /// Text RGBA for this run, overriding the cue-wide colour.
    pub color: [u8; 4],
}

/// WebVTT cue placement settings, the subset the bitmap overlay honours. `None`
/// fields mean "auto": auto `line` stacks the cue from the bottom, auto
/// `position` centres it. SRT cues always carry the default (no positioning).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
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
    /// A cue has one backing box, so a span-scoped rule's `background-color`
    /// lands here too rather than behind its span alone.
    pub background: Option<[u8; 4]>,
    /// Per-span text colours from `::cue(.class)` rules, each covering only the
    /// span it came from. Empty unless a span-scoped rule matched.
    pub spans: Vec<SpanStyle>,
}

impl CueSettings {
    /// The colour to draw the character at byte offset `at` in the cue text: the
    /// innermost span run covering it, else the cue-wide `color`.
    pub fn color_at(&self, at: usize) -> Option<[u8; 4]> {
        self.spans
            .iter()
            .rev()
            .find(|s| at >= s.start && at < s.end)
            .map(|s| s.color)
            .or(self.color)
    }
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
        Box::new(self.clone())
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
/// blocks are read for `::cue`, `::cue(#id)` and `::cue(.a[.b...])`
/// `color` / `background-color` rules, which are resolved onto each cue's
/// [`CueSettings`] (the subset the overlay can apply; other CSS properties are
/// ignored). A span-scoped `::cue(.class)` colour covers only the `<c.class>`
/// span it matched, as a [`SpanStyle`] run. A header block's `X-TIMESTAMP-MAP` (RFC 8216 §3.5) rebases the cue
/// times that follow it onto the MPEG-2 media timeline, so the concatenated
/// segments of an HLS rendition land where the video does.
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
    // A header block's `X-TIMESTAMP-MAP` rebases every cue after it (each HLS
    // segment carries its own header, so the offset changes mid-document).
    let mut cues = Vec::new();
    let mut offset_ns = 0i64;
    for b in &blocks {
        if let Some(off) = block_timestamp_offset(b) {
            offset_ns = off;
        }
        if let Some((mut cue, spans)) = block_to_cue(b, true) {
            rebase_cue(&mut cue, offset_ns);
            if !sheet.is_empty() {
                apply_cue_style(&sheet, block_cue_id(b), &spans, &mut cue.settings);
            }
            cues.push(cue);
        }
    }
    cues
}

/// The cue-time rebase a WebVTT header block's `X-TIMESTAMP-MAP` puts in effect
/// (RFC 8216 §3.5), or `None` when the block is not a header or carries no
/// usable map (a malformed one is skipped, leaving the previous offset).
fn block_timestamp_offset(block: &[&str]) -> Option<i64> {
    let first = block.first()?.trim_start();
    if !(first == "WEBVTT" || first.starts_with("WEBVTT ") || first.starts_with("WEBVTT\t")) {
        return None;
    }
    block.iter().find_map(|l| parse_timestamp_map(l.trim()))
}

/// Parse an `X-TIMESTAMP-MAP` line into the nanoseconds to add to every cue time
/// of the segment carrying it: `MPEGTS/90000 - LOCAL`, i.e. where on the MPEG-2
/// (90 kHz) media timeline the segment's cue-time origin sits. The two fields are
/// named, so either order parses; the RFC's example writes `LOCAL` first. Both are
/// required, and both are untrusted, so a missing / unparseable / out-of-range
/// field yields `None` (the header is skipped, as it was before the map was read).
fn parse_timestamp_map(line: &str) -> Option<i64> {
    let attrs = line.strip_prefix("X-TIMESTAMP-MAP")?.trim_start();
    let mut mpegts: Option<u64> = None;
    let mut local: Option<u64> = None;
    for field in attrs.strip_prefix('=')?.split(',') {
        // LOCAL's value is itself colon-separated (`00:00:00.000`), so split on
        // the first colon only.
        let (key, value) = field.trim().split_once(':')?;
        match key.trim() {
            "MPEGTS" => mpegts = value.trim().parse().ok(),
            "LOCAL" => local = parse_timestamp(value),
            _ => {}
        }
    }
    // 90 kHz ticks to ns, in i128 so an attacker-supplied MPEGTS cannot overflow.
    let mpegts_ns = i128::from(mpegts?) * 1_000_000_000 / 90_000;
    i64::try_from(mpegts_ns - i128::from(local?)).ok()
}

/// Shift a cue's window by an `X-TIMESTAMP-MAP` offset, clamping at zero (a
/// segment whose map moves cues before the origin shows them at time 0).
fn rebase_cue(cue: &mut Cue, offset_ns: i64) {
    if offset_ns == 0 {
        return;
    }
    cue.start_ns = cue.start_ns.saturating_add_signed(offset_ns);
    cue.end_ns = cue.end_ns.saturating_add_signed(offset_ns);
}

/// The WebVTT cue identifier (the line just before the timing line), if any.
/// `None` when the timing line opens the block (no identifier).
fn block_cue_id<'a>(block: &[&'a str]) -> Option<&'a str> {
    let timing_idx = block.iter().position(|l| l.contains("-->"))?;
    (timing_idx > 0).then(|| block[timing_idx - 1].trim())
}

/// One class-carrying span of a cue's text: where it lands in the stripped text
/// and the classes its tag named (`<c.loud.narrator>`, `<v.loud Bob>`). What
/// `::cue(.class)` selects.
#[derive(Debug)]
struct CueSpan {
    start: usize,
    end: usize,
    classes: Vec<String>,
}

/// The `.class` parts of an open span tag's name (`c.loud.narrator`), empty for
/// a tag that carries none. A tag is `<name.class1.class2 annotation>`, so only
/// the part before the annotation counts; a cue timestamp (`<00:00:01.000>`) has
/// no tag name and is not a span.
fn tag_classes(tag: &str) -> Vec<String> {
    if !is_span_tag(tag) {
        return Vec::new();
    }
    tag.split_whitespace()
        .next()
        .unwrap_or("")
        .split('.')
        .skip(1)
        .filter(|p| !p.is_empty())
        .map(String::from)
        .collect()
}

/// A `::cue` selector we apply: all cues (`::cue`), one identifier
/// (`::cue(#id)`), or a span class list (`::cue(.a)`, `::cue(.a.b)` requiring
/// both). Element and descendant selectors are not supported.
#[derive(Debug)]
enum CueSelector {
    All,
    Id(String),
    Classes(Vec<String>),
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
/// stripped; rules with no understood selector (an element or compound rule) are
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

        let selectors: Vec<CueSelector> = sel_str
            .split(',')
            .filter_map(|s| parse_cue_selector(s.trim()))
            .collect();
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
        rules.push(CueStyleRule {
            selectors,
            color,
            background,
        });
    }
    rules
}

/// A `::cue`, `::cue(#id)` or `::cue(.a[.b...])` selector, or `None` for
/// anything else.
fn parse_cue_selector(sel: &str) -> Option<CueSelector> {
    if sel == "::cue" {
        return Some(CueSelector::All);
    }
    let inner = sel.strip_prefix("::cue(")?.strip_suffix(')')?.trim();
    // Only a bare `#id` / `.class` chain is supported (no element or descendant
    // selectors), so a name must be non-empty and unbroken.
    let named = |n: &str| !n.is_empty() && !n.contains(|c: char| c.is_whitespace());
    if let Some(id) = inner.strip_prefix('#') {
        return named(id).then(|| CueSelector::Id(id.into()));
    }
    let classes: Vec<String> = inner
        .strip_prefix('.')?
        .split('.')
        .map(String::from)
        .collect();
    classes
        .iter()
        .all(|c| named(c))
        .then_some(CueSelector::Classes(classes))
}

/// Resolve a cue's style from the sheet. Whole-cue rules (`::cue`, then
/// `::cue(#id)`, in increasing specificity) set the cue's `color` /
/// `background`; a span-scoped `::cue(.class)` rule sets the colour of just the
/// spans that carry its classes, as a [`SpanStyle`] run. A cue has one backing
/// box, so a span rule's `background-color` still applies cue-wide.
fn apply_cue_style(
    sheet: &[CueStyleRule],
    id: Option<&str>,
    spans: &[CueSpan],
    settings: &mut CueSettings,
) {
    apply_matching(sheet, settings, |sel| matches!(sel, CueSelector::All));
    for span in spans {
        // CSS specificity: a compound `::cue(.a.b)` beats a one-class rule
        // whatever the sheet order; equal specificity falls back to that order.
        let mut matched: Vec<(usize, &CueStyleRule)> = sheet
            .iter()
            .filter_map(|rule| {
                rule.selectors
                    .iter()
                    .filter_map(|sel| match sel {
                        CueSelector::Classes(want)
                            if want.iter().all(|w| span.classes.contains(w)) =>
                        {
                            Some(want.len())
                        }
                        _ => None,
                    })
                    .max()
                    .map(|specificity| (specificity, rule))
            })
            .collect();
        matched.sort_by_key(|(specificity, _)| *specificity);
        let mut run = CueSettings::default();
        for (_, rule) in matched {
            fold_rule(rule, &mut run);
        }
        if let Some(color) = run.color {
            settings.spans.push(SpanStyle {
                start: span.start,
                end: span.end,
                color,
            });
        }
        if run.background.is_some() {
            settings.background = run.background;
        }
    }
    if let Some(id) = id {
        apply_matching(
            sheet,
            settings,
            |sel| matches!(sel, CueSelector::Id(rid) if rid == id),
        );
    }
}

/// Fold the `color` / `background` of every rule with a selector satisfying
/// `pred` onto `settings`, in sheet order (a later rule wins).
fn apply_matching(
    sheet: &[CueStyleRule],
    settings: &mut CueSettings,
    pred: impl Fn(&CueSelector) -> bool,
) {
    for rule in sheet {
        if rule.selectors.iter().any(&pred) {
            fold_rule(rule, settings);
        }
    }
}

/// Fold one rule's `color` / `background` onto `settings` (a property the rule
/// does not set leaves the current value).
fn fold_rule(rule: &CueStyleRule, settings: &mut CueSettings) {
    if rule.color.is_some() {
        settings.color = rule.color;
    }
    if rule.background.is_some() {
        settings.background = rule.background;
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
        let mut chan = || {
            it.next()?
                .trim()
                .parse::<u32>()
                .ok()
                .map(|n| n.min(255) as u8)
        };
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
/// order. The `[Events]` section carries the cues: its `Format:` line gives the
/// column order, and each `Dialogue:` line is split accordingly (the `Text`
/// column is last and may itself contain commas). Inline override blocks
/// (`{\i1}`...) are stripped and `\N` / `\n` line breaks become real newlines,
/// like the SRT / WebVTT tag handling. Malformed dialogue lines are skipped.
///
/// Placement comes from the `Alignment` of the dialogue's style (read from the
/// `[V4 Styles]` / `[V4+ Styles]` section), overridden by an inline `{\an8}` /
/// legacy `{\a6}` tag, mapped onto the same [`CueSettings`] the WebVTT path
/// fills. Pixel-space placement is mapped through the `[Script Info]`
/// `PlayResX` / `PlayResY`: an inline `{\pos(x,y)}` places the cue at that point
/// as a percentage of the script canvas, and otherwise the `MarginL` / `MarginR`
/// / `MarginV` columns (the dialogue's own, falling back to its style's) inset
/// it. Without a `PlayRes` the pixel values mean nothing, so only the alignment
/// is used.
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

/// One style's placement defaults: its `\an` alignment (1-9) and its
/// `MarginL` / `MarginR` / `MarginV` insets in script pixels, which a dialogue
/// line inherits wherever its own margin column is `0`.
#[derive(Debug, Clone, Copy, Default)]
struct SsaStyle {
    align: Option<u8>,
    margins: SsaMargins,
}

/// Margin insets in script (`PlayRes`) pixels: from the left and right edges and
/// from the top or bottom (whichever the alignment anchors to). `0` means "not
/// set" in both SSA and ASS, which is why a dialogue line's zero falls back to
/// its style's value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SsaMargins {
    left: u32,
    right: u32,
    vertical: u32,
}

impl SsaMargins {
    /// This line's margins with each unset (`0`) component taken from `style`.
    fn or(self, style: SsaMargins) -> SsaMargins {
        SsaMargins {
            left: if self.left != 0 {
                self.left
            } else {
                style.left
            },
            right: if self.right != 0 {
                self.right
            } else {
                style.right
            },
            vertical: if self.vertical != 0 {
                self.vertical
            } else {
                style.vertical
            },
        }
    }
}

/// Per-line SSA parse state, shared by the whole-document [`parse_ssa`] and the
/// streaming `SubParse` element: which section the scan is in, the resolved
/// `[Events]` column indices, the script canvas (`PlayResX` / `PlayResY`), and
/// the style table collected from the styles section. Held across chunks so a
/// `Dialogue:` line in a later chunk parses with the column order, canvas and
/// styles declared by an earlier one.
#[derive(Debug, Clone)]
struct SsaState {
    in_events: bool,
    /// Inside `[V4 Styles]` / `[V4+ Styles]`, whose `Style:` lines are read.
    in_styles: bool,
    /// Inside `[Script Info]`, whose `PlayResX` / `PlayResY` give the canvas the
    /// pixel placements are expressed in.
    in_script_info: bool,
    /// The styles section is legacy `[V4 Styles]`, whose `Alignment` column uses
    /// the old `\a` numbering rather than the `\an` numpad one.
    styles_legacy: bool,
    /// `PlayResX` / `PlayResY`, `0` until the script declares them (without both,
    /// pixel placement cannot be made frame-relative and is ignored).
    play_res: (u32, u32),
    i_start: usize,
    i_end: usize,
    i_text: usize,
    i_style: usize,
    /// `MarginL` / `MarginR` / `MarginV` columns of the `[Events]` `Format:` line.
    i_margins: (usize, usize, usize),
    /// `Name` / `Alignment` columns of the styles `Format:` line; alignment is
    /// read only once a `Format:` names it.
    i_style_name: usize,
    i_style_align: Option<usize>,
    /// `MarginL` / `MarginR` / `MarginV` columns of the styles `Format:` line.
    i_style_margins: (Option<usize>, Option<usize>, Option<usize>),
    /// Style name -> its placement defaults, in declaration order.
    styles: Vec<(String, SsaStyle)>,
}

impl Default for SsaState {
    fn default() -> Self {
        // V4+ default column order, used until an explicit `Format:` line
        // overrides it: Layer, Start, End, Style, Name, MarginL, MarginR,
        // MarginV, Effect, Text.
        Self {
            in_events: false,
            in_styles: false,
            in_script_info: false,
            styles_legacy: false,
            play_res: (0, 0),
            i_start: 1,
            i_end: 2,
            i_text: 9,
            i_style: 3,
            i_margins: (5, 6, 7),
            i_style_name: 0,
            i_style_align: None,
            i_style_margins: (None, None, None),
            styles: Vec::new(),
        }
    }
}

impl SsaState {
    /// Feed one SSA line, returning a cue if it is a `Dialogue:` line in the
    /// `[Events]` section. Section headers, `Format:` and `Style:` lines update
    /// the state.
    fn feed_line(&mut self, line: &str) -> Option<Cue> {
        let line = line.trim();
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            let name = name.trim();
            self.in_events = name.eq_ignore_ascii_case("Events");
            self.in_script_info = name.eq_ignore_ascii_case("Script Info");
            // Both `[V4 Styles]` and `[V4+ Styles]` (and the `[V4+ Styles]`
            // localizations that keep the suffix) carry the style table.
            self.in_styles = name.to_ascii_lowercase().ends_with("styles");
            self.styles_legacy = self.in_styles && !name.contains('+');
            return None;
        }
        if self.in_script_info {
            // `PlayResX: 1920`. An unparseable or zero value leaves the canvas
            // unset, so pixel placement stays unmapped rather than dividing by 0.
            if let Some(v) = strip_prefix_ci(line, "PlayResX:").and_then(parse_ssa_u32) {
                self.play_res.0 = v;
            } else if let Some(v) = strip_prefix_ci(line, "PlayResY:").and_then(parse_ssa_u32) {
                self.play_res.1 = v;
            }
            return None;
        }
        if self.in_styles {
            self.feed_style_line(line);
            return None;
        }
        if !self.in_events {
            return None;
        }
        if let Some(rest) = strip_prefix_ci(line, "Format:") {
            let cols: Vec<&str> = rest.split(',').map(str::trim).collect();
            self.i_start = col_index(&cols, "Start").unwrap_or(self.i_start);
            self.i_end = col_index(&cols, "End").unwrap_or(self.i_end);
            self.i_style = col_index(&cols, "Style").unwrap_or(self.i_style);
            self.i_margins = (
                col_index(&cols, "MarginL").unwrap_or(self.i_margins.0),
                col_index(&cols, "MarginR").unwrap_or(self.i_margins.1),
                col_index(&cols, "MarginV").unwrap_or(self.i_margins.2),
            );
            // Text is the last column by spec; fall back to that if unnamed.
            self.i_text = col_index(&cols, "Text").unwrap_or(cols.len().saturating_sub(1));
            None
        } else if let Some(rest) = strip_prefix_ci(line, "Dialogue:") {
            self.parse_dialogue(rest)
        } else {
            None
        }
    }

    /// Read a `Format:` / `Style:` line of the styles section into the table.
    fn feed_style_line(&mut self, line: &str) {
        if let Some(rest) = strip_prefix_ci(line, "Format:") {
            let cols: Vec<&str> = rest.split(',').map(str::trim).collect();
            self.i_style_name = col_index(&cols, "Name").unwrap_or(0);
            self.i_style_align = col_index(&cols, "Alignment");
            self.i_style_margins = (
                col_index(&cols, "MarginL"),
                col_index(&cols, "MarginR"),
                col_index(&cols, "MarginV"),
            );
            return;
        }
        let Some(rest) = strip_prefix_ci(line, "Style:") else {
            return;
        };
        let cols: Vec<&str> = rest.split(',').map(str::trim).collect();
        let Some(name) = cols.get(self.i_style_name) else {
            return;
        };
        let align = self
            .i_style_align
            .and_then(|i| cols.get(i))
            .and_then(|a| a.parse::<u8>().ok())
            .and_then(|a| {
                if self.styles_legacy {
                    legacy_alignment(a)
                } else {
                    (1..=9).contains(&a).then_some(a)
                }
            });
        let column = |i: Option<usize>| i.and_then(|i| cols.get(i)).and_then(|v| parse_ssa_u32(v));
        let margins = SsaMargins {
            left: column(self.i_style_margins.0).unwrap_or(0),
            right: column(self.i_style_margins.1).unwrap_or(0),
            vertical: column(self.i_style_margins.2).unwrap_or(0),
        };
        if align.is_none() && margins == SsaMargins::default() {
            return; // nothing placeable in this style
        }
        self.styles
            .push((String::from(*name), SsaStyle { align, margins }));
    }

    /// Parse one `Dialogue:` body into a cue using the resolved column indices.
    /// The `Text` column is last, so we split on only the leading commas and keep
    /// its remainder (commas and all) intact. Alignment is the inline `{\an}`
    /// override if present, else the dialogue style's; pixel placement is the
    /// inline `{\pos(x,y)}` if present, else the margin columns, both scaled by
    /// the script's `PlayRes`.
    fn parse_dialogue(&self, body: &str) -> Option<Cue> {
        // splitn keeps everything after the i_text-th comma as the final field.
        let fields: Vec<&str> = body.splitn(self.i_text + 1, ',').collect();
        if fields.len() <= self.i_text {
            return None;
        }
        let start_ns = parse_timestamp(fields.get(self.i_start)?.trim())?;
        let end_ns = parse_timestamp(fields.get(self.i_end)?.trim())?;
        let raw = fields[self.i_text];
        let text = strip_ass_markup(raw);
        if text.trim().is_empty() {
            return None;
        }
        let style = fields
            .get(self.i_style)
            .map(|s| s.trim())
            .and_then(|name| {
                self.styles
                    .iter()
                    .find(|(n, _)| n.eq_ignore_ascii_case(name))
            })
            .map(|(_, s)| *s)
            .unwrap_or_default();
        let an = ass_inline_alignment(raw).or(style.align);
        let mut settings = an.map(ass_alignment_settings).unwrap_or_default();

        // Pixel placement, in the script's own canvas. `\pos` wins over margins,
        // which are the dialogue's own falling back to its style's.
        let column = |i: usize| fields.get(i).and_then(|v| parse_ssa_u32(v)).unwrap_or(0);
        let margins = SsaMargins {
            left: column(self.i_margins.0),
            right: column(self.i_margins.1),
            vertical: column(self.i_margins.2),
        }
        .or(style.margins);
        match ass_inline_pos(raw) {
            Some((x, y)) => ass_apply_pos(&mut settings, self.play_res, x, y),
            None => ass_apply_margins(&mut settings, self.play_res, an.unwrap_or(2), margins),
        }

        Some(Cue {
            start_ns,
            end_ns,
            text,
            settings,
        })
    }
}

/// Place a cue at an `{\pos(x,y)}` point, as a percentage of the script canvas.
/// Without a `PlayRes` the pixels mean nothing and the placement is left alone.
/// `y` sets the cue's line, which this model anchors at the block's top edge, so
/// a bottom-anchored `\pos` lands the block a block-height low.
fn ass_apply_pos(settings: &mut CueSettings, play_res: (u32, u32), x: u32, y: u32) {
    let (res_x, res_y) = play_res;
    if res_x != 0 {
        settings.position = Some(pixel_percent(x, res_x));
    }
    if res_y != 0 {
        settings.line = Some(pixel_percent(y, res_y));
    }
}

/// Inset a cue by its `MarginL` / `MarginR` / `MarginV`, as a percentage of the
/// script canvas: the horizontal margins move the anchor the alignment's column
/// uses, and `MarginV` sets the line for a top-anchored alignment. A
/// bottom-anchored one keeps the auto bottom stack (this model's `line` is the
/// block's top, so a bottom inset has no exact expression) and a middle-anchored
/// one stays centred. Without a `PlayRes`, nothing is mapped.
fn ass_apply_margins(
    settings: &mut CueSettings,
    play_res: (u32, u32),
    an: u8,
    margins: SsaMargins,
) {
    let (res_x, res_y) = play_res;
    if res_x != 0 {
        let left = pixel_percent(margins.left, res_x);
        let right = 100u8.saturating_sub(pixel_percent(margins.right, res_x));
        settings.position = match an % 3 {
            1 => Some(left),
            0 => Some(right),
            // A centred cue sits midway between the two insets; with neither set
            // that is the frame centre, which is the auto position already.
            _ if margins.left == 0 && margins.right == 0 => settings.position,
            _ => Some(left.saturating_add(right) / 2),
        };
    }
    if res_y != 0 && margins.vertical != 0 && (7..=9).contains(&an) {
        settings.line = Some(pixel_percent(margins.vertical, res_y));
    }
}

/// A script-pixel coordinate as a percentage `0..=100` of the canvas extent.
/// `extent` is non-zero (the callers check), and a value past the canvas clamps.
fn pixel_percent(value: u32, extent: u32) -> u8 {
    ((value as u64 * 100 / extent as u64).min(100)) as u8
}

/// Parse an SSA integer field (a `PlayRes` or a margin). `0` reads as unset, and
/// so does an empty, negative, or out-of-range value: untrusted numbers are
/// dropped rather than wrapping.
fn parse_ssa_u32(v: &str) -> Option<u32> {
    v.trim().parse::<u32>().ok().filter(|n| *n != 0)
}

/// The `{\pos(x,y)}` override in an ASS text field, if any (the last one wins,
/// like the alignment overrides). Fractional coordinates are truncated; a
/// malformed or out-of-range pair is ignored.
fn ass_inline_pos(raw: &str) -> Option<(u32, u32)> {
    let mut out = None;
    let mut rest = raw;
    while let Some(open) = rest.find('{') {
        let after = &rest[open + 1..];
        let end = after.find('}').unwrap_or(after.len());
        let mut block = &after[..end];
        while let Some(at) = block.find("\\pos(") {
            let tail = &block[at + 5..];
            block = tail;
            let Some(close) = tail.find(')') else { break };
            let mut parts = tail[..close].split(',');
            let mut coord = || {
                let v = parts.next()?.trim();
                let int = v.split_once('.').map_or(v, |(i, _)| i);
                int.parse::<u32>().ok()
            };
            if let (Some(x), Some(y)) = (coord(), coord()) {
                out = Some((x, y));
            }
        }
        rest = &after[end..];
    }
    out
}

/// Map an ASS `\an` alignment (numpad 1-9) onto the cue placement model: the
/// column picks `align` and the horizontal anchor, the row picks `line`. The
/// bottom row stays auto (`None`) so bottom cues stack like any default cue.
fn ass_alignment_settings(an: u8) -> CueSettings {
    let (align, position) = match an % 3 {
        1 => (TextAlign::Start, Some(0)),
        2 => (TextAlign::Center, None),
        _ => (TextAlign::End, Some(100)),
    };
    let line = match (an - 1) / 3 {
        0 => None,
        1 => Some(50),
        _ => Some(0),
    };
    CueSettings {
        position,
        line,
        align,
        ..CueSettings::default()
    }
}

/// Map a legacy SSA `\a` alignment code to the ASS numpad `\an` value: the low
/// two bits pick the column, bit 2 the top row and bit 3 the middle row.
fn legacy_alignment(a: u8) -> Option<u8> {
    let col = match a & 3 {
        1 => 1,
        2 => 2,
        3 => 3,
        _ => return None,
    };
    Some(if a & 8 != 0 {
        col + 3
    } else if a & 4 != 0 {
        col + 6
    } else {
        col
    })
}

/// The alignment override in an ASS text field's `{...}` blocks: `\an<1-9>` or
/// the legacy `\a<code>`. The last one wins, matching the order a renderer
/// applies overrides in.
fn ass_inline_alignment(raw: &str) -> Option<u8> {
    let mut out = None;
    let mut rest = raw;
    while let Some(open) = rest.find('{') {
        let after = &rest[open + 1..];
        let end = after.find('}').unwrap_or(after.len());
        let mut block = &after[..end];
        while let Some(at) = block.find("\\a") {
            let tail = &block[at + 2..];
            let (digits, numpad) = match tail.strip_prefix('n') {
                Some(d) => (d, true),
                None => (tail, false),
            };
            let n: String = digits.chars().take_while(char::is_ascii_digit).collect();
            if let Ok(v) = n.parse::<u8>() {
                let an = if numpad {
                    (1..=9).contains(&v).then_some(v)
                } else {
                    legacy_alignment(v)
                };
                out = an.or(out);
            }
            block = tail;
        }
        rest = &after[end..];
    }
    out
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
    head.eq_ignore_ascii_case(prefix)
        .then(|| &line[prefix.len()..])
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

/// De-frame one Matroska subtitle block payload to plain display text (M415). The
/// block carries a single cue and its timing rides the Matroska block (timestamp +
/// `BlockDuration`), so only the text is extracted, by the track's source format:
/// - [`TextFormat::Utf8`] (`S_TEXT/UTF8`): the payload is the text already.
/// - [`TextFormat::Ssa`] (`S_TEXT/ASS`): the block is the comma-separated fields
///   `ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text`; the `Text`
///   field (everything after the 8th comma) is taken and its `{...}` override tags
///   / `\N` line breaks resolved (via [`strip_ass_markup`]).
/// - [`TextFormat::WebVtt`] (`S_TEXT/WEBVTT`): the payload is the cue text with
///   inline `<...>` tags, which are stripped.
///
/// Any other format returns the payload unchanged. The Matroska ASS framing omits
/// `Dialogue:` and the timing fields (the block carries those), so it cannot be fed
/// to the document `SubParse`; de-framing to plain UTF-8 here matches how the MP4
/// `tx3g` / `wvtt` paths forward container-timed text.
pub fn deframe_subtitle_block(payload: &str, format: TextFormat) -> String {
    match format {
        TextFormat::Ssa => {
            // Skip the 8 leading fields (ReadOrder..Effect) to the Text field, which
            // may itself contain commas, so split into at most 9 parts.
            let text = payload.splitn(9, ',').nth(8).unwrap_or("");
            strip_ass_markup(text)
        }
        TextFormat::WebVtt => {
            let mut out = String::new();
            for (i, line) in payload.lines().enumerate() {
                if i > 0 {
                    out.push('\n');
                }
                push_stripped(line, &mut out);
            }
            out
        }
        _ => String::from(payload),
    }
}

/// The `CodecPrivate` an `S_TEXT/ASS` track carries: the script header up to and
/// including the `[Events]` `Format:` line, which is what names the fields every
/// block payload then holds (an ASS reader rejects the track without it). One
/// `Default` style, since the cues framed by [`frame_subtitle_block`] are plain
/// text and reference no other.
pub const ASS_SCRIPT_HEADER: &str = "[Script Info]\r\n\
ScriptType: v4.00+\r\n\
\r\n\
[V4+ Styles]\r\n\
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\r\n\
Style: Default,Arial,16,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1\r\n\
\r\n\
[Events]\r\n\
Format: ReadOrder, Layer, Style, Name, MarginL, MarginR, MarginV, Effect, Text\r\n";

/// Frame plain UTF-8 cue text as the block payload an `S_TEXT/*` Matroska track
/// carries (M898), the inverse of [`deframe_subtitle_block`]. The cue's window is
/// the block timestamp + `BlockDuration`, so only the text is framed:
/// - [`TextFormat::Utf8`] / [`TextFormat::WebVtt`]: the text is the payload.
/// - [`TextFormat::Ssa`]: the eight fields the de-frame skips are written ahead of
///   the text, with the line breaks as `\N`. `read_order` is the event index, which
///   a reader uses to order events and drop repeats, so it must rise per cue.
pub fn frame_subtitle_block(text: &str, format: TextFormat, read_order: u64) -> String {
    match format {
        TextFormat::Ssa => {
            let mut out = alloc::format!("{read_order},0,Default,,0,0,0,,");
            for (i, line) in text.lines().enumerate() {
                if i > 0 {
                    out.push_str("\\N");
                }
                out.push_str(line);
            }
            out
        }
        _ => String::from(text),
    }
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
///
/// Placement comes from the paragraph's `tts:textAlign` and the `<region>` it
/// uses (its own or the one inherited from the enclosing `<div>` / `<body>`),
/// whose percentage `tts:origin` / `tts:extent` / `tts:displayAlign` map onto the
/// same [`CueSettings`] the WebVTT path fills. Non-percentage region lengths
/// (pixels, cells) are not mapped; such a paragraph keeps the default placement.
pub fn parse_ttml(input: &str) -> Vec<Cue> {
    let regions = parse_ttml_regions(input);
    let mut cues = Vec::new();
    let mut paragraphs = TtmlParagraphs::new(input);
    while let Some((attrs, body, inherited)) = paragraphs.next() {
        let (Some(begin), Some(end)) = (xml_attr(attrs, "begin"), xml_attr(attrs, "end")) else {
            continue;
        };
        let (Some(start_ns), Some(end_ns)) = (parse_ttml_time(begin), parse_ttml_time(end)) else {
            continue;
        };
        let text = ttml_text(body);
        if !text.trim().is_empty() {
            cues.push(Cue {
                start_ns,
                end_ns,
                text,
                settings: ttml_settings(attrs, inherited, &regions),
            });
        }
    }
    cues
}

/// Split a tag body (what sits between `<` and `>`) into its local element name
/// (leading `/` and any namespace prefix removed) and the attribute remainder.
fn split_tag(tag: &str) -> (&str, &str) {
    let body = tag.trim_start_matches('/');
    let skipped = tag.len() - body.len();
    let name = body
        .split([' ', '\t', '\r', '\n', '/'])
        .next()
        .unwrap_or("");
    let local = name.rsplit(':').next().unwrap_or(name);
    (local, &tag[skipped + name.len()..])
}

/// Walks a TTML document's `<p ...> ... </p>` paragraphs, carrying the region
/// inherited from the enclosing `<div>` / `<body>` (TTML lets either hold the
/// `region` attribute). `<p>` does not nest, so the next close tag terminates the
/// current paragraph; self-closing `<p/>` has no content and is skipped.
#[derive(Debug)]
struct TtmlParagraphs<'a> {
    rest: &'a str,
    inherited: Option<&'a str>,
}

impl<'a> TtmlParagraphs<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            rest: input,
            inherited: None,
        }
    }

    /// The next paragraph's attribute string, inner content, and inherited
    /// region id.
    fn next(&mut self) -> Option<(&'a str, &'a str, Option<&'a str>)> {
        loop {
            let lt = self.rest.find('<')?;
            let after_lt = &self.rest[lt + 1..];
            let gt = after_lt.find('>')?;
            let tag = &after_lt[..gt];
            let after_tag = &after_lt[gt + 1..];
            self.rest = after_tag;
            let (local, attrs) = split_tag(tag);
            let closing = tag.starts_with('/');
            if local.eq_ignore_ascii_case("p") && !closing {
                if tag.ends_with('/') {
                    continue;
                }
                let (content_end, after_close) = find_paragraph_close(after_tag)?;
                self.rest = &after_tag[after_close..];
                return Some((attrs, &after_tag[..content_end], self.inherited));
            }
            if local.eq_ignore_ascii_case("div") || local.eq_ignore_ascii_case("body") {
                self.inherited = if closing {
                    None
                } else {
                    xml_attr(attrs, "region").or(self.inherited)
                };
            }
        }
    }
}

/// Where a region's content sits inside it vertically (`tts:displayAlign`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum DisplayAlign {
    /// Top of the region, the TTML default.
    #[default]
    Before,
    Center,
    /// Bottom of the region, the usual subtitle setting.
    After,
}

/// A TTML `<region>`'s placement: percentage `tts:origin` / `tts:extent` plus the
/// alignment defaults it gives the paragraphs that use it.
#[derive(Debug)]
struct TtmlRegion {
    id: String,
    origin: Option<(u8, u8)>,
    extent: Option<(u8, u8)>,
    display_align: DisplayAlign,
    text_align: Option<TextAlign>,
}

/// Collect the document's `<region>` definitions (they live in `<head><layout>`,
/// but any `region` element is read; a region with no `xml:id` is unusable).
fn parse_ttml_regions(input: &str) -> Vec<TtmlRegion> {
    let mut regions = Vec::new();
    let mut rest = input;
    while let Some(lt) = rest.find('<') {
        let after_lt = &rest[lt + 1..];
        let Some(gt) = after_lt.find('>') else { break };
        let tag = &after_lt[..gt];
        rest = &after_lt[gt + 1..];
        let (local, attrs) = split_tag(tag);
        if !local.eq_ignore_ascii_case("region") || tag.starts_with('/') {
            continue;
        }
        let Some(id) = xml_attr(attrs, "id") else {
            continue;
        };
        regions.push(TtmlRegion {
            id: id.into(),
            origin: percent_pair(xml_attr(attrs, "origin")),
            extent: percent_pair(xml_attr(attrs, "extent")),
            display_align: match xml_attr(attrs, "displayAlign").map(str::trim) {
                Some("center") => DisplayAlign::Center,
                Some("after") => DisplayAlign::After,
                _ => DisplayAlign::Before,
            },
            text_align: xml_attr(attrs, "textAlign").and_then(parse_ttml_align),
        });
    }
    regions
}

/// Map a paragraph's `tts:textAlign` plus the region it uses onto the cue
/// placement model. A paragraph with neither keeps the default (auto placement,
/// centred), so an unpositioned document renders exactly as before.
fn ttml_settings(attrs: &str, inherited: Option<&str>, regions: &[TtmlRegion]) -> CueSettings {
    let own_align = xml_attr(attrs, "textAlign").and_then(parse_ttml_align);
    let region = xml_attr(attrs, "region")
        .or(inherited)
        .and_then(|id| regions.iter().find(|r| r.id == id))
        // A region with no percentage geometry (pixel lengths, or none at all)
        // says nothing this model can use; leave the cue at the default.
        .filter(|r| r.origin.is_some() || r.extent.is_some());
    let Some(region) = region else {
        return CueSettings {
            align: own_align.unwrap_or_default(),
            ..CueSettings::default()
        };
    };
    let align = own_align.or(region.text_align).unwrap_or_default();
    let (x, y) = region.origin.unwrap_or((0, 0));
    let (w, h) = region
        .extent
        .unwrap_or((100u8.saturating_sub(x), 100u8.saturating_sub(y)));
    let bottom = y.saturating_add(h).min(100);
    let position = match align {
        TextAlign::Start => x,
        TextAlign::Center => x.saturating_add(w / 2).min(100),
        TextAlign::End => x.saturating_add(w).min(100),
    };
    let line = match region.display_align {
        DisplayAlign::Before => Some(y),
        DisplayAlign::Center => Some(y.saturating_add(h / 2).min(100)),
        // `after` anchors the block's bottom to the region's, which this model
        // expresses only as the auto bottom stack; a region stopping short of the
        // frame bottom falls back to placing the block at its lower edge.
        DisplayAlign::After if bottom >= 100 => None,
        DisplayAlign::After => Some(bottom),
    };
    CueSettings {
        position: Some(position),
        line,
        align,
        ..CueSettings::default()
    }
}

/// Parse a `tts:textAlign` value. `justify` has no analog in the cue model and
/// falls back to the default.
fn parse_ttml_align(v: &str) -> Option<TextAlign> {
    match v.trim() {
        "left" | "start" => Some(TextAlign::Start),
        "center" => Some(TextAlign::Center),
        "right" | "end" => Some(TextAlign::End),
        _ => None,
    }
}

/// Parse a two-length TTML attribute (`tts:origin="10% 80%"`) to a percentage
/// pair. `None` unless both lengths are percentages, since the cue model is
/// frame-relative.
fn percent_pair(v: Option<&str>) -> Option<(u8, u8)> {
    let mut parts = v?.split_whitespace();
    let x = percent_len(parts.next()?)?;
    let y = percent_len(parts.next()?)?;
    Some((x, y))
}

/// Parse one `<n>%` length to `0..=100`, accepting a fractional value.
fn percent_len(v: &str) -> Option<u8> {
    let n: f32 = v.trim().strip_suffix('%')?.parse().ok()?;
    Some(n.clamp(0.0, 100.0) as u8)
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
/// attribute string. The name is matched on its local part, so any namespace
/// prefix binding is accepted (`tts:origin` for `origin`, `xml:id` for `id`).
/// `None` if the attribute is absent or unquoted.
fn xml_attr<'a>(attrs: &'a str, name: &str) -> Option<&'a str> {
    let mut from = 0;
    while let Some(pos) = attrs[from..].find(name) {
        let at = from + pos;
        let before_ok = at == 0
            || attrs[..at]
                .chars()
                .next_back()
                .map(|c| c.is_whitespace() || c == ':')
                .unwrap_or(true);
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
        let (local, _) = split_tag(tag);
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
                        u32::from_str_radix(&ent[2..], 16)
                            .ok()
                            .and_then(char::from_u32)
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
    let frac: alloc::string::String = frac
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .take(9)
        .collect();
    if frac.is_empty() {
        return 0;
    }
    let Ok(frac_int) = frac.parse::<u64>() else {
        return 0;
    };
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
    parse_blocks_rebased(input, webvtt, &mut 0)
}

/// [`parse_blocks`] with the `X-TIMESTAMP-MAP` offset held by the caller, so the
/// streaming [`SubParse`] keeps a segment's rebase in effect across the chunk
/// boundary that may fall between its header block and its cues.
fn parse_blocks_rebased(input: &str, webvtt: bool, offset_ns: &mut i64) -> Vec<Cue> {
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let mut cues = Vec::new();
    let mut block: Vec<&str> = Vec::new();
    let mut take = |block: &[&str], cues: &mut Vec<Cue>| {
        if webvtt {
            if let Some(off) = block_timestamp_offset(block) {
                *offset_ns = off;
            }
        }
        if let Some((mut cue, _spans)) = block_to_cue(block, webvtt) {
            rebase_cue(&mut cue, *offset_ns);
            cues.push(cue);
        }
    };
    for line in input.lines() {
        if line.trim().is_empty() {
            take(&block, &mut cues);
            block.clear();
        } else {
            block.push(line);
        }
    }
    take(&block, &mut cues);
    cues
}

/// Turn one non-empty block into a cue plus the class-carrying spans of its text
/// (what `::cue(.class)` selects), or `None` if it is not a cue (a WebVTT header
/// / NOTE / STYLE / REGION block, or a block with no timing line).
fn block_to_cue(block: &[&str], webvtt: bool) -> Option<(Cue, Vec<CueSpan>)> {
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

    let (text, spans) = strip_cue_text(&block[timing_idx + 1..]);
    // Drop a fully empty payload (a timing line with no following text).
    if text.trim().is_empty() {
        return None;
    }
    Some((
        Cue {
            start_ns,
            end_ns,
            text,
            settings,
        },
        spans,
    ))
}

/// Join a cue's text lines with the `<...>` markup removed, recording the byte
/// range each class-carrying span covers in the result. Spans nest, so the stack
/// pairs each close tag with the innermost open one; a span left unclosed at the
/// end of the cue runs to the end of the text (a stray `</c>` is ignored).
fn strip_cue_text(lines: &[&str]) -> (String, Vec<CueSpan>) {
    let mut text = String::new();
    // Spans in document order (an enclosing span before the ones it contains),
    // with `open` holding the indices of the ones still to be closed.
    let mut spans: Vec<CueSpan> = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    for (i, raw) in lines.iter().enumerate() {
        if i > 0 {
            text.push('\n');
        }
        let mut rest = *raw;
        while let Some(lt) = rest.find('<') {
            text.push_str(&rest[..lt]);
            let after = &rest[lt + 1..];
            let Some(gt) = after.find('>') else {
                rest = "";
                break;
            };
            let tag = &after[..gt];
            rest = &after[gt + 1..];
            if tag.starts_with('/') {
                if let Some(idx) = open.pop() {
                    spans[idx].end = text.len();
                }
            } else if is_span_tag(tag) {
                open.push(spans.len());
                spans.push(CueSpan {
                    start: text.len(),
                    end: text.len(),
                    classes: tag_classes(tag),
                });
            }
        }
        text.push_str(rest);
    }
    // A span the cue never closes (the common unclosed `<v Speaker>`) runs to the end.
    for idx in open {
        spans[idx].end = text.len();
    }
    spans.retain(|s| !s.classes.is_empty());
    (text, spans)
}

/// Whether an open tag is a span (`c`, `i`, `v.narrator`, ...) rather than an
/// inline cue timestamp (`<00:00:01.000>`), which is a void element and must not
/// be paired with a following close tag.
fn is_span_tag(tag: &str) -> bool {
    let name = tag_name(tag);
    !name.is_empty() && !name.contains(':')
}

/// A tag's element name: the part before any `.class` chain or annotation.
fn tag_name(tag: &str) -> &str {
    tag.split_whitespace()
        .next()
        .unwrap_or("")
        .split('.')
        .next()
        .unwrap_or("")
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
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::subparse::SubParse;
///
/// let parser = SubParse::new();
/// ```
#[derive(Debug, Default)]
pub struct SubParse {
    /// Input subtitle format, fixed at `configure_pipeline`.
    format: Option<TextFormat>,
    /// Sink bytes not yet forming a complete cue, carried to the next chunk.
    buf: Vec<u8>,
    /// SSA `[Events]` / column-order state, persisted across chunks.
    ssa: SsaState,
    /// The `X-TIMESTAMP-MAP` rebase the last WebVTT header block put in effect,
    /// held across chunks (and across the segments of an HLS rendition, each of
    /// which may carry its own map).
    map_offset_ns: i64,
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
            Caps::Text {
                format: TextFormat::Srt,
            },
            Caps::Text {
                format: TextFormat::WebVtt,
            },
            Caps::Text {
                format: TextFormat::Ssa,
            },
            Caps::Text {
                format: TextFormat::Ttml,
            },
        ]))
    }

    fn output_caps() -> Caps {
        Caps::Text {
            format: TextFormat::Utf8,
        }
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
            let cues = parse_blocks_rebased(&doc, webvtt, &mut self.map_offset_ns);
            self.buf.clear();
            return cues;
        }
        let valid = utf8_prefix_len(&self.buf);
        let s = core::str::from_utf8(&self.buf[..valid]).expect("valid_up_to is a char boundary");
        let Some(boundary) = last_block_boundary(s) else {
            return Vec::new();
        };
        let cues = parse_blocks_rebased(&s[..boundary], webvtt, &mut self.map_offset_ns);
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
                    format @ (TextFormat::Srt | TextFormat::WebVtt | TextFormat::Ssa | TextFormat::Ttml),
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
                    if let Some(slice) = frame.domain.as_system_slice() {
                        // The parsers handle CRLF / BOM, so accumulate raw bytes.
                        self.buf.extend_from_slice(slice);
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
                    self.map_offset_ns = 0;
                    self.bom_stripped = false;
                    out.push(PipelinePacket::Flush).await?;
                    Vec::new()
                }
                // The trailing partial block is parsed now; the runner arm forwards
                // the trailing Eos.
                PipelinePacket::Eos => self.drain_cues(true),
                other => {
                    out.push(other).await?;
                    Vec::new()
                }
            };
            for cue in cues {
                if !self.caps_emitted {
                    out.push(PipelinePacket::CapsChanged(Self::output_caps()))
                        .await?;
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
                frame.meta.attach(TextCueMeta {
                    settings: cue.settings,
                });
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
    fn deframe_subtitle_block_per_source_format() {
        // S_TEXT/UTF8: the block is the text already.
        assert_eq!(
            deframe_subtitle_block("Hello\nworld", TextFormat::Utf8),
            "Hello\nworld"
        );

        // S_TEXT/ASS: take the Text field (after 8 commas), strip {...} and \N.
        let ass = "0,0,Default,,0,0,0,,{\\i1}Hello{\\i0}\\Nworld";
        assert_eq!(deframe_subtitle_block(ass, TextFormat::Ssa), "Hello\nworld");
        // A Text field that itself contains commas survives (split into 9 parts).
        let ass_commas = "1,0,Default,,0,0,0,,one, two, three";
        assert_eq!(
            deframe_subtitle_block(ass_commas, TextFormat::Ssa),
            "one, two, three"
        );

        // S_TEXT/WEBVTT: the block is cue text with inline tags, which are stripped.
        assert_eq!(
            deframe_subtitle_block("<c.yellow>Hi</c> there", TextFormat::WebVtt),
            "Hi there"
        );
    }

    /// The framing pairs with the de-framing: what a muxer writes for a cue is
    /// what a demuxer reads back as that cue's text.
    #[test]
    fn frame_subtitle_block_is_the_inverse_of_the_de_frame() {
        let ass = frame_subtitle_block("Hello\nworld", TextFormat::Ssa, 7);
        assert_eq!(ass, "7,0,Default,,0,0,0,,Hello\\Nworld");
        assert_eq!(
            deframe_subtitle_block(&ass, TextFormat::Ssa),
            "Hello\nworld"
        );
        // A cue whose text contains commas survives the field split.
        let commas = frame_subtitle_block("one, two", TextFormat::Ssa, 0);
        assert_eq!(deframe_subtitle_block(&commas, TextFormat::Ssa), "one, two");
        // Plain UTF-8 is the payload as-is.
        assert_eq!(
            frame_subtitle_block("Hi", TextFormat::Utf8, 3),
            String::from("Hi")
        );
    }

    #[test]
    fn segmented_webvtt_skips_repeated_headers() {
        // Two concatenated HLS WebVTT segments (M419): each opens with a
        // `WEBVTT` + `X-TIMESTAMP-MAP` header block. Those blocks have no `-->`
        // timing line, so they parse as non-cues and are skipped; both cues survive.
        let segmented = "WEBVTT\n\
            X-TIMESTAMP-MAP=MPEGTS:0,LOCAL:00:00:00.000\n\n\
            00:00:01.000 --> 00:00:02.000\nHello\n\n\
            WEBVTT\n\
            X-TIMESTAMP-MAP=MPEGTS:0,LOCAL:00:00:00.000\n\n\
            00:00:03.000 --> 00:00:04.000\nWorld\n";
        let cues = parse_webvtt(segmented);
        assert_eq!(cues.len(), 2, "both cues parse across the segment boundary");
        assert_eq!(cues[0].text, "Hello");
        assert_eq!(cues[0].start_ns, 1_000_000_000);
        assert_eq!(cues[1].text, "World");
        assert_eq!(cues[1].start_ns, 3_000_000_000);
    }

    #[test]
    fn segment_timestamp_maps_rebase_their_own_cues() {
        // Two HLS WebVTT segments, each with its own X-TIMESTAMP-MAP, in the two
        // field orders that occur in the wild (the RFC's example writes LOCAL
        // first). Segment 1: MPEGTS 900000 (10s) maps cue time 0, so its cue
        // shifts +10s. Segment 2: MPEGTS 1800000 (20s) maps cue time 5s, so the
        // offset is +15s.
        let segmented = "WEBVTT\n\
            X-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:900000\n\n\
            00:00:01.000 --> 00:00:02.000\nHello\n\n\
            WEBVTT\n\
            X-TIMESTAMP-MAP=MPEGTS:1800000,LOCAL:00:00:05.000\n\n\
            00:00:06.000 --> 00:00:07.000\nWorld\n";
        let cues = parse_webvtt(segmented);
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, "Hello");
        assert_eq!(cues[0].start_ns, 11_000_000_000);
        assert_eq!(cues[0].end_ns, 12_000_000_000);
        assert_eq!(cues[1].text, "World");
        assert_eq!(cues[1].start_ns, 21_000_000_000);
        assert_eq!(cues[1].end_ns, 22_000_000_000);
    }

    #[test]
    fn timestamp_map_negative_offset_clamps_at_zero() {
        // LOCAL ahead of MPEGTS shifts cues earlier; a cue that would land before
        // the origin clamps to 0 rather than wrapping.
        let input = "WEBVTT\n\
            X-TIMESTAMP-MAP=MPEGTS:90000,LOCAL:00:00:05.000\n\n\
            00:00:06.000 --> 00:00:07.000\nlate\n\n\
            00:00:01.000 --> 00:00:02.000\nearly\n";
        let cues = parse_webvtt(input);
        assert_eq!(cues[0].start_ns, 2_000_000_000, "6s - 4s offset");
        assert_eq!(cues[1].start_ns, 0, "1s - 4s clamps at the origin");
        assert_eq!(cues[1].end_ns, 0);
    }

    #[test]
    fn malformed_timestamp_maps_leave_cues_untouched() {
        // A map missing a field, with a non-numeric MPEGTS, with an unparseable
        // LOCAL, or with an out-of-range MPEGTS is skipped: the cue keeps its own
        // time and nothing panics.
        for header in [
            "X-TIMESTAMP-MAP=MPEGTS:900000",
            "X-TIMESTAMP-MAP=LOCAL:00:00:00.000",
            "X-TIMESTAMP-MAP=MPEGTS:nope,LOCAL:00:00:00.000",
            "X-TIMESTAMP-MAP=MPEGTS:900000,LOCAL:99:99:99.999",
            "X-TIMESTAMP-MAP=MPEGTS:18446744073709551615,LOCAL:00:00:00.000",
            "X-TIMESTAMP-MAP=",
            "X-TIMESTAMP-MAP",
        ] {
            let input = alloc::format!("WEBVTT\n{header}\n\n00:00:01.000 --> 00:00:02.000\ncue\n");
            let cues = parse_webvtt(&input);
            assert_eq!(cues.len(), 1, "{header}");
            assert_eq!(cues[0].start_ns, 1_000_000_000, "{header}");
        }
    }

    #[test]
    fn timestamp_srt_and_webvtt_forms() {
        // SRT comma, full clock.
        assert_eq!(parse_timestamp("00:00:01,000"), Some(1_000_000_000));
        // WebVTT dot, full clock with millis.
        assert_eq!(
            parse_timestamp("01:02:03.500"),
            Some((3600 + 120 + 3) * 1_000_000_000 + 500_000_000)
        );
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
                spans: Vec::new(),
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
                spans: Vec::new(),
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
    fn webvtt_cue_class_selector_styles_only_its_span() {
        // `::cue(.class)` matches the classes on the cue's span tags: `<c.loud>`
        // and the voice form `<v.narrator Bob>`. The colour covers just that
        // span's byte range; the rest of the cue keeps the global `::cue` colour.
        let input = "WEBVTT\n\n\
            STYLE\n\
            ::cue { color: white; }\n\
            ::cue(.loud) { color: red; background-color: black; }\n\
            ::cue(.narrator) { color: cyan; }\n\
            ::cue(#tagged) { color: lime; }\n\n\
            00:00:00.000 --> 00:00:01.000\nsay <c.loud>SHOUT</c> now\n\n\
            00:00:01.000 --> 00:00:02.000\n<v.narrator Bob>calm\n\n\
            tagged\n00:00:02.000 --> 00:00:03.000\n<c.loud>span</c> plus id\n\n\
            00:00:03.000 --> 00:00:04.000\nplain\n";
        let cues = parse_webvtt(input);
        assert_eq!(cues.len(), 4);

        // .loud recolours "SHOUT" only; the cue-wide colour stays the global one,
        // and the box (one per cue) takes the rule's background.
        assert_eq!(cues[0].text, "say SHOUT now");
        assert_eq!(
            cues[0].settings.spans,
            alloc::vec![SpanStyle {
                start: 4,
                end: 9,
                color: [255, 0, 0, 255],
            }]
        );
        assert_eq!(cues[0].settings.color, Some([255, 255, 255, 255]));
        assert_eq!(cues[0].settings.background, Some([0, 0, 0, 255]));
        assert_eq!(cues[0].settings.color_at(0), Some([255, 255, 255, 255]));
        assert_eq!(cues[0].settings.color_at(4), Some([255, 0, 0, 255]));
        assert_eq!(cues[0].settings.color_at(9), Some([255, 255, 255, 255]));

        // An unclosed voice span runs to the end of the cue.
        assert_eq!(
            cues[1].settings.spans,
            alloc::vec![SpanStyle {
                start: 0,
                end: 4,
                color: [0, 255, 255, 255],
            }]
        );

        // A span rule and an id rule coexist: the id sets the cue colour, the
        // span keeps its own.
        assert_eq!(cues[2].settings.color, Some([0, 255, 0, 255]));
        assert_eq!(cues[2].settings.color_at(0), Some([255, 0, 0, 255]));
        assert_eq!(cues[2].settings.color_at(5), Some([0, 255, 0, 255]));

        // No span on the cue -> only the global rule, everywhere.
        assert!(cues[3].settings.spans.is_empty());
        assert_eq!(cues[3].settings.color_at(0), Some([255, 255, 255, 255]));
        assert_eq!(cues[3].settings.background, None);
    }

    #[test]
    fn webvtt_compound_class_selector_needs_every_class() {
        // `::cue(.a.b)` matches only a span carrying both classes, in any order;
        // a span with one of them is left alone. Nested spans stack, the inner
        // one winning where they overlap.
        let input = "WEBVTT\n\n\
            STYLE\n\
            ::cue(.loud.angry) { color: red; }\n\
            ::cue(.loud) { color: blue; }\n\n\
            00:00:00.000 --> 00:00:01.000\n\
            <c.loud>calm <c.angry.loud>MAD</c></c>\n\n\
            00:00:01.000 --> 00:00:02.000\n<c.angry>only angry</c>\n";
        let cues = parse_webvtt(input);
        // The outer `.loud` span is blue over the whole line; the inner span
        // carries both classes, so the compound rule recolours "MAD" red.
        assert_eq!(cues[0].text, "calm MAD");
        assert_eq!(
            cues[0].settings.spans,
            alloc::vec![
                SpanStyle {
                    start: 0,
                    end: 8,
                    color: [0, 0, 255, 255],
                },
                SpanStyle {
                    start: 5,
                    end: 8,
                    color: [255, 0, 0, 255],
                },
            ]
        );
        assert_eq!(cues[0].settings.color_at(0), Some([0, 0, 255, 255]));
        assert_eq!(cues[0].settings.color_at(5), Some([255, 0, 0, 255]));
        // `.angry` alone matches neither rule (the compound needs `.loud` too).
        assert!(cues[1].settings.spans.is_empty());
        assert_eq!(cues[1].settings.color_at(0), None);
    }

    #[test]
    fn webvtt_cue_timestamp_tag_is_not_a_class() {
        // An inline cue timestamp (`<00:00:01.000>`) must not be mistaken for a
        // span whose "class" is the millisecond part.
        let input = "WEBVTT\n\n\
            STYLE\n::cue(.000) { color: red; }\n\n\
            00:00:00.000 --> 00:00:02.000\nkara<00:00:01.000>oke\n";
        let cues = parse_webvtt(input);
        assert_eq!(cues[0].text, "karaoke");
        assert_eq!(cues[0].settings.color, None);
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
                spans: Vec::new(),
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
        assert_eq!(
            cues[0].text, "Italic text",
            "tags stripped, settings ignored"
        );
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
        assert_eq!(
            parse_auto("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n").len(),
            1
        );
        assert_eq!(
            parse_auto("1\n00:00:01,000 --> 00:00:02,000\nhi\n").len(),
            1
        );
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
    fn ssa_style_alignment_becomes_cue_placement() {
        let doc = "[Script Info]\n\
            Title: demo\n\
            \n\
            [V4+ Styles]\n\
            Format: Name, Fontname, Alignment, MarginL\n\
            Style: Top,Arial,7,0\n\
            Style: Default,Arial,2,0\n\
            \n\
            [Events]\n\
            Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\
            Dialogue: 0,0:00:00.00,0:00:01.00,Top,,0,0,0,,top left\n\
            Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,bottom centre\n\
            Dialogue: 0,0:00:02.00,0:00:03.00,Default,,0,0,0,,{\\i1}{\\an9}top right\n";
        let cues = parse_ssa(doc);
        assert_eq!(cues.len(), 3);
        // \an7: top row, left column.
        assert_eq!(
            cues[0].settings,
            CueSettings {
                position: Some(0),
                line: Some(0),
                align: TextAlign::Start,
                ..CueSettings::default()
            }
        );
        // \an2 is the renderer default: auto placement, centred.
        assert_eq!(cues[1].settings, CueSettings::default());
        // The inline override beats the style's \an2.
        assert_eq!(
            cues[2].settings,
            CueSettings {
                position: Some(100),
                line: Some(0),
                align: TextAlign::End,
                ..CueSettings::default()
            }
        );
        assert_eq!(cues[2].text, "top right", "override tags still stripped");
    }

    #[test]
    fn ssa_legacy_alignment_codes() {
        // `[V4 Styles]` numbers Alignment the old way: 6 = 2 (centre) | 4 (top).
        let doc = "[V4 Styles]\n\
            Format: Name, Fontname, Alignment\n\
            Style: Legacy,Arial,6\n\
            \n\
            [Events]\n\
            Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\
            Dialogue: 0,0:00:00.00,0:00:01.00,Legacy,,0,0,0,,top centre\n\
            Dialogue: 0,0:00:01.00,0:00:02.00,Legacy,,0,0,0,,{\\a1}bottom left\n";
        let cues = parse_ssa(doc);
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].settings.line, Some(0), "bit 2 is the top row");
        assert_eq!(cues[0].settings.align, TextAlign::Center);
        // Legacy \a1: bottom row (auto), left column.
        assert_eq!(
            cues[1].settings,
            CueSettings {
                position: Some(0),
                line: None,
                align: TextAlign::Start,
                ..CueSettings::default()
            }
        );
    }

    #[test]
    fn ssa_inline_pos_maps_through_playres() {
        // `{\pos(x,y)}` is in script-canvas pixels: 480/1920 = 25% across,
        // 810/1080 = 75% down. The last override on the line wins, and a cue
        // without one keeps its alignment-derived placement.
        let doc = "[Script Info]\n\
            PlayResX: 1920\n\
            PlayResY: 1080\n\
            \n\
            [Events]\n\
            Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\
            Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\\pos(480,810)}placed\n\
            Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\an7}{\\pos(1920,0)}corner\n\
            Dialogue: 0,0:00:02.00,0:00:03.00,Default,,0,0,0,,plain\n\
            Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,{\\pos(9999,9999)}off-canvas\n";
        let cues = parse_ssa(doc);
        assert_eq!(cues.len(), 4);
        assert_eq!(cues[0].settings.position, Some(25));
        assert_eq!(cues[0].settings.line, Some(75));
        assert_eq!(cues[0].text, "placed", "the override tag is stripped");
        // `\pos` places, `\an` still picks how the block hangs off the point.
        assert_eq!(cues[1].settings.position, Some(100));
        assert_eq!(cues[1].settings.line, Some(0));
        assert_eq!(cues[1].settings.align, TextAlign::Start);
        // No `\pos`, no margins: the default auto placement.
        assert_eq!(cues[2].settings, CueSettings::default());
        // A point outside the canvas clamps to the frame edge.
        assert_eq!(cues[3].settings.position, Some(100));
        assert_eq!(cues[3].settings.line, Some(100));
    }

    #[test]
    fn ssa_margins_inset_the_cue() {
        // MarginL/R inset the horizontal anchor of the alignment's column and
        // MarginV the line of a top-anchored one, both as a percentage of the
        // canvas. A dialogue's own 0 falls back to its style's margin.
        let doc = "[Script Info]\n\
            PlayResX: 1920\n\
            PlayResY: 1080\n\
            \n\
            [V4+ Styles]\n\
            Format: Name, Fontname, Alignment, MarginL, MarginR, MarginV\n\
            Style: Left,Arial,1,192,0,0\n\
            Style: Right,Arial,3,0,384,0\n\
            Style: Top,Arial,8,0,0,108\n\
            \n\
            [Events]\n\
            Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\
            Dialogue: 0,0:00:00.00,0:00:01.00,Left,,0,0,0,,style margin\n\
            Dialogue: 0,0:00:01.00,0:00:02.00,Left,,960,0,0,,line margin wins\n\
            Dialogue: 0,0:00:02.00,0:00:03.00,Right,,0,0,0,,from the right\n\
            Dialogue: 0,0:00:03.00,0:00:04.00,Top,,0,0,0,,top inset\n\
            Dialogue: 0,0:00:04.00,0:00:05.00,Top,,192,192,0,,centred between\n";
        let cues = parse_ssa(doc);
        assert_eq!(cues.len(), 5);
        // \an1 (bottom left) + MarginL 192 = 10% in from the left.
        assert_eq!(cues[0].settings.position, Some(10));
        assert_eq!(cues[0].settings.align, TextAlign::Start);
        assert_eq!(cues[0].settings.line, None, "bottom rows still auto-stack");
        // The dialogue's own MarginL overrides the style's.
        assert_eq!(cues[1].settings.position, Some(50));
        // \an3 (bottom right) + MarginR 384 = 20% in from the right.
        assert_eq!(cues[2].settings.position, Some(80));
        assert_eq!(cues[2].settings.align, TextAlign::End);
        // \an8 (top centre) + MarginV 108 = 10% down.
        assert_eq!(cues[3].settings.line, Some(10));
        assert_eq!(
            cues[3].settings.position, None,
            "centred: no inset either side"
        );
        // A centred cue with both insets sits midway between them.
        assert_eq!(cues[4].settings.position, Some(50));
    }

    #[test]
    fn ssa_pixel_placement_needs_a_playres() {
        // Without `PlayResX` / `PlayResY` (or with a bogus one) the pixel values
        // have no canvas to scale against, so only the alignment is mapped.
        let doc = "[Script Info]\n\
            PlayResX: nonsense\n\
            \n\
            [Events]\n\
            Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\
            Dialogue: 0,0:00:00.00,0:00:01.00,Default,,300,0,0,,{\\pos(100,200)}unmapped\n";
        let cues = parse_ssa(doc);
        assert_eq!(cues[0].settings, CueSettings::default());
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
        let doc =
            r#"<tt:tt><tt:body><tt:p begin="0s" end="1s">hi<tt:br/>there</tt:p></tt:body></tt:tt>"#;
        let cues = parse_ttml(doc);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "hi\nthere");
    }

    const TTML_REGIONS: &str = r#"<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling">
  <head><layout>
    <region xml:id="top" tts:origin="0% 0%" tts:extent="50% 20%" tts:displayAlign="before" tts:textAlign="left"/>
    <region xml:id="bottom" tts:origin="10% 80%" tts:extent="80% 20%" tts:displayAlign="after"/>
    <region xml:id="pixels" tts:origin="16px 16px" tts:extent="320px 40px"/>
  </layout></head>
  <body>
    <div region="bottom">
      <p begin="0s" end="1s">inherited region</p>
      <p begin="1s" end="2s" region="top">own region</p>
      <p begin="2s" end="3s" tts:textAlign="right">aligned in the div region</p>
      <p begin="3s" end="4s" region="pixels">pixel region</p>
    </div>
  </body>
</tt>"#;

    #[test]
    fn ttml_region_placement_becomes_cue_settings() {
        let cues = parse_ttml(TTML_REGIONS);
        assert_eq!(cues.len(), 4);
        // The div's region: bottom-anchored and reaching the frame edge, so the
        // cue keeps the auto bottom stack; centred within 10%..90%.
        assert_eq!(
            cues[0].settings,
            CueSettings {
                position: Some(50),
                line: None,
                align: TextAlign::Center,
                ..CueSettings::default()
            }
        );
        // The paragraph's own region wins over the div's: top-left, left-aligned
        // from the region's tts:textAlign.
        assert_eq!(
            cues[1].settings,
            CueSettings {
                position: Some(0),
                line: Some(0),
                align: TextAlign::Start,
                ..CueSettings::default()
            }
        );
        // Paragraph textAlign wins over the region's, and moves the anchor to the
        // region's right edge.
        assert_eq!(
            cues[2].settings,
            CueSettings {
                position: Some(90),
                line: None,
                align: TextAlign::End,
                ..CueSettings::default()
            }
        );
        // Pixel lengths are not frame-relative, so that region places nothing.
        assert_eq!(
            cues[3].settings,
            CueSettings::default(),
            "a pixel region leaves the cue at the default placement"
        );
    }

    #[test]
    fn ttml_without_regions_keeps_default_placement() {
        // The unpositioned document (no region, no textAlign) is unchanged.
        assert!(parse_ttml(TTML)
            .iter()
            .all(|c| c.settings == CueSettings::default()));
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
        let cue = Cue {
            start_ns: 1000,
            end_ns: 2000,
            text: "x".into(),
            settings: CueSettings::default(),
        };
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
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
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
        assert_eq!(
            el.intercept_caps(&Caps::Text {
                format: TextFormat::Srt
            })
            .unwrap(),
            Caps::Text {
                format: TextFormat::Srt
            }
        );
        assert!(el
            .intercept_caps(&Caps::Text {
                format: TextFormat::Utf8
            })
            .is_err());
        let CapsConstraint::DerivedOutput(derive) = el.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let out = derive(&Caps::Text {
            format: TextFormat::WebVtt,
        });
        assert_eq!(
            out.alternatives(),
            &[Caps::Text {
                format: TextFormat::Utf8
            }]
        );
        // SSA and TTML negotiate the same way (also -> Utf8).
        assert_eq!(
            el.intercept_caps(&Caps::Text {
                format: TextFormat::Ssa
            })
            .unwrap(),
            Caps::Text {
                format: TextFormat::Ssa
            }
        );
        assert_eq!(
            el.intercept_caps(&Caps::Text {
                format: TextFormat::Ttml
            })
            .unwrap(),
            Caps::Text {
                format: TextFormat::Ttml
            }
        );
    }

    #[tokio::test]
    async fn element_parses_ttml_to_timed_utf8() {
        let mut el = SubParse::new();
        el.configure_pipeline(&Caps::Text {
            format: TextFormat::Ttml,
        })
        .expect("accepts TTML");

        let mut sink = RecordingSink::default();
        el.process(srt_bytes_frame(TTML.as_bytes()), &mut sink)
            .await
            .unwrap();
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
        if let Some(s) = frames[0].domain.as_system_slice() {
            assert_eq!(s, b"Hello & world");
        } else {
            panic!("cue payload must be a system buffer");
        }
    }

    #[tokio::test]
    async fn element_parses_ssa_to_timed_utf8() {
        let mut el = SubParse::new();
        el.configure_pipeline(&Caps::Text {
            format: TextFormat::Ssa,
        })
        .expect("accepts SSA");

        let mut sink = RecordingSink::default();
        el.process(srt_bytes_frame(ASS.as_bytes()), &mut sink)
            .await
            .unwrap();
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
        if let Some(s) = frames[0].domain.as_system_slice() {
            assert_eq!(s, b"Hello, world");
        } else {
            panic!("cue payload must be a system buffer");
        }
    }

    #[tokio::test]
    async fn element_emits_caps_then_timed_cue_frames() {
        let doc = "1\n00:00:01,000 --> 00:00:04,000\nHello world\n\n\
                   2\n00:01:02,500 --> 00:01:05,000\nSecond cue\nacross two lines\n";
        let mut el = SubParse::new();
        el.configure_pipeline(&Caps::Text {
            format: TextFormat::Srt,
        })
        .expect("accepts SRT");

        let mut sink = RecordingSink::default();
        // Two chunks then EOS, exercising the byte buffer.
        let (a, b) = doc.as_bytes().split_at(25);
        el.process(srt_bytes_frame(a), &mut sink).await.unwrap();
        el.process(srt_bytes_frame(b), &mut sink).await.unwrap();
        el.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        assert!(matches!(
            sink.packets.first(),
            Some(PipelinePacket::CapsChanged(Caps::Text {
                format: TextFormat::Utf8
            }))
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
        if let Some(s) = frames[0].domain.as_system_slice() {
            assert_eq!(s, b"Hello world");
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
        el.configure_pipeline(&Caps::Text {
            format: TextFormat::Srt,
        })
        .unwrap();
        let mut sink = RecordingSink::default();

        el.process(
            srt_bytes_frame(b"1\n00:00:01,000 --> 00:00:02,000\nfirst\n\n2\n00:00:03,000 -->"),
            &mut sink,
        )
        .await
        .unwrap();

        let count = |sink: &RecordingSink| {
            sink.packets
                .iter()
                .filter(|p| matches!(p, PipelinePacket::DataFrame(_)))
                .count()
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
        el.configure_pipeline(&Caps::Text {
            format: TextFormat::Srt,
        })
        .unwrap();
        let mut sink = RecordingSink::default();

        let mut chunk1 = Vec::from(
            &b"1\n00:00:01,000 --> 00:00:02,000\nokay\n\n2\n00:00:03,000 --> 00:00:04,000\ncaf"[..],
        );
        chunk1.push(0xC3); // first byte of 'e-acute', completed in the next chunk
        el.process(srt_bytes_frame(&chunk1), &mut sink)
            .await
            .unwrap();
        let after_chunk1 = sink
            .packets
            .iter()
            .filter(|p| matches!(p, PipelinePacket::DataFrame(_)))
            .count();
        assert_eq!(
            after_chunk1, 1,
            "the terminated cue streams before the rest arrives"
        );

        el.process(srt_bytes_frame(&[0xA9, b'\n', b'\n']), &mut sink)
            .await
            .unwrap();
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
        if let Some(s) = frames[1].domain.as_system_slice() {
            assert_eq!(core::str::from_utf8(s).unwrap(), "café");
        } else {
            panic!("cue payload must be a system buffer");
        }
    }

    #[cfg(feature = "metadata")]
    #[tokio::test]
    async fn element_attaches_cue_positioning_meta() {
        // WebVTT placement is parsed into CueSettings; the element carries it on
        // the cue frame as TextCueMeta so an overlay recovers it (M406).
        let doc =
            "WEBVTT\n\n00:00:00.000 --> 00:00:02.000 position:20% line:80% align:start\nplaced\n\n";
        let mut el = SubParse::new();
        el.configure_pipeline(&Caps::Text {
            format: TextFormat::WebVtt,
        })
        .unwrap();
        let mut sink = RecordingSink::default();
        el.process(srt_bytes_frame(doc.as_bytes()), &mut sink)
            .await
            .unwrap();
        el.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let frame = sink
            .packets
            .iter()
            .find_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .expect("a cue frame");
        let meta = frame
            .meta
            .get::<TextCueMeta>()
            .expect("cue carries placement meta");
        assert_eq!(
            meta.settings,
            CueSettings {
                position: Some(20),
                line: Some(80),
                align: TextAlign::Start,
                vertical: WritingMode::Horizontal,
                color: None,
                background: None,
                spans: Vec::new(),
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
