//! GStreamer-to-g2g porting helpers (M200): a `gst`-element-name map and a
//! launch-line linter that turns parse failures into porting guidance.
//!
//! These back `g2g-inspect --gst <name>` and `g2g-launch`'s explain-on-error,
//! and are the programmatic surface a porting tool builds on. They complement
//! [`parse_launch`](g2g_core::runtime::parse_launch) (the authoritative parse):
//! the linter runs it and enriches the first error with a gst->g2g suggestion,
//! so porting is fix-and-rerun rather than decode-the-error.

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::runtime::{parse_launch, ParseError, Registry};

/// What a GStreamer element name maps to in g2g.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GstEquivalent {
    /// A registered g2g element (possibly via an alias) or a launch keyword
    /// (`tee`, `queue`, `decodebin`, ...) uses this exact name.
    Available,
    /// g2g has an equivalent under a different name (the suggestion). The target
    /// may be feature-gated, so it is advice, not a guarantee it is compiled in.
    Renamed(&'static str),
    /// No g2g element; the hint explains the closest path.
    Unsupported(&'static str),
    /// g2g has this exact element, but the cargo feature that compiles it (the
    /// payload) is off in this build.
    NotCompiled(&'static str),
    /// Unknown, but close enough to a name this build does have (the payload)
    /// to be a spelling mistake.
    DidYouMean(&'static str),
    /// Unknown to both the registry and the gst-compat table: cannot advise.
    Unknown,
}

/// Launch keywords the parser handles that are not registry elements.
static LAUNCH_KEYWORDS: &[&str] = &[
    "decodebin",
    "uridecodebin",
    "playbin",
    "queue",
    "queue2",
    "tee",
];

/// gst element name -> guidance, for names NOT registered under the same name.
/// Registered names (incl. aliases like `avdec_h264` -> `ffmpegdec`) resolve to
/// `Available` before this table is consulted; keep this for the gst names that
/// have no same-name g2g element. Extend freely.
static GST_MAP: &[(&str, GstEquivalent)] = &[
    ("x264enc", GstEquivalent::Unsupported(
        "software H.264 encode (`x264enc`, libx264) needs the `ffmpeg` feature on Linux; \
         otherwise `nvenc` (NVIDIA), `mfencode` (Windows), or encode AV1/VP8/VP9 with `av1enc`/`vpxenc`",
    )),
    ("x265enc", GstEquivalent::Unsupported("no software H.265 encoder; use `nvenc` (NVIDIA HEVC) or `av1enc`")),
    ("theoraenc", GstEquivalent::Unsupported("no Theora encoder; use `vpxenc` (VP8/VP9) or `av1enc`")),
    ("avdec_h264", GstEquivalent::Renamed("ffmpegdec")),
    ("avdec_h265", GstEquivalent::Renamed("ffmpegdec")),
    // NVIDIA hardware codecs map to the native NVDEC / NVENC elements (their
    // features are CI-excluded but the names are the direct equivalents, like the
    // VAAPI rows below); `ffmpegdec`'s cuvid backend is the software-feature fallback.
    ("nvh264dec", GstEquivalent::Renamed("nvdec")),
    ("nvh265dec", GstEquivalent::Renamed("nvdec")),
    ("nvh264enc", GstEquivalent::Renamed("nvenc")),
    ("nvh265enc", GstEquivalent::Renamed("nvenc")),
    ("vaapih264dec", GstEquivalent::Renamed("vaapidec")),
    ("vah264dec", GstEquivalent::Renamed("vaapidec")),
    ("vp8enc", GstEquivalent::Renamed("vpxenc")),
    ("vp9enc", GstEquivalent::Renamed("vpxenc")),
    ("jpegenc", GstEquivalent::Renamed("mjpegenc")),
    ("jpegdec", GstEquivalent::Renamed("mjpegdec")),
    ("avenc_aac", GstEquivalent::Renamed("mfaacencode")),
    ("faac", GstEquivalent::Renamed("mfaacencode")),
    ("souphttpsrc", GstEquivalent::Renamed("httpsrc")),
    // appsrc / appsink are registered elements, so gst_equivalent resolves them
    // to Available before this table; no row is needed (and an Unsupported one
    // would contradict reality).
    ("rtph264depay", GstEquivalent::Unsupported("RTP depayloading is built into `udpsrc` / `rtspsrc`")),
    ("rtph264pay", GstEquivalent::Unsupported("RTP payloading is built into `udpsink`")),
    // `equalizer-3bands` / `spectrum` / `clockoverlay` / `splitmuxsink` are
    // registered elements, so gst_equivalent resolves them to Available before this
    // table; only the wider N-band equalizers need a pointer.
    ("equalizer-10bands", GstEquivalent::Renamed("equalizer-3bands")),
    ("equalizer-nbands", GstEquivalent::Renamed("equalizer-3bands")),
];

/// Map a GStreamer element name to its g2g equivalent, consulting the live
/// `registry` first (so aliases resolve and feature-gated elements that ARE
/// compiled in show as `Available`), then the launch keywords, then the static
/// guidance table, then the feature catalog (the name is a g2g element this build
/// left out), and finally the nearest known name (a spelling mistake).
///
/// The hand-written table outranks the feature catalog: both know `x264enc`, and
/// the table's entry also lists the alternatives for a platform where the feature
/// cannot be built.
pub fn gst_equivalent(registry: &Registry, gst_name: &str) -> GstEquivalent {
    if registry_has(registry, gst_name) || LAUNCH_KEYWORDS.contains(&gst_name) {
        return GstEquivalent::Available;
    }
    if let Some((_, equivalent)) = GST_MAP.iter().find(|(name, _)| *name == gst_name) {
        return equivalent.clone();
    }
    if let Some(feature) = crate::registry::required_feature(gst_name) {
        return GstEquivalent::NotCompiled(feature);
    }
    match nearest_known_name(registry, gst_name) {
        Some(near) => GstEquivalent::DidYouMean(near),
        None => GstEquivalent::Unknown,
    }
}

/// How many single-character insertions, deletions, or substitutions turn `left`
/// into `right`, comparing ASCII case-insensitively (element names are lowercase,
/// so `FileSrc` should still read as `filesrc`).
fn edit_distance(left: &str, right: &str) -> usize {
    let right: Vec<u8> = right.bytes().map(|b| b.to_ascii_lowercase()).collect();
    // One row of the edit matrix: `previous[j]` is the distance from the prefix
    // of `left` handled so far to the first `j` bytes of `right`.
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = alloc::vec![0usize; right.len() + 1];
    for (i, l) in left.bytes().map(|b| b.to_ascii_lowercase()).enumerate() {
        current[0] = i + 1;
        for (j, r) in right.iter().enumerate() {
            let substitute = previous[j] + usize::from(l != *r);
            current[j + 1] = substitute.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        core::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

/// A typo suggestion allows one edit per this many characters of the unknown
/// name, so a name under this length gets no suggestion at all.
const TYPO_CHARS_PER_EDIT: usize = 4;

/// The edit allowance cap, so a long garbage token cannot reach a real name.
const TYPO_MAX_EDITS: usize = 2;

/// The name closest to `name` among everything a launch line can reference, when
/// one is close enough to be a typo of it (see [`TYPO_CHARS_PER_EDIT`] /
/// [`TYPO_MAX_EDITS`]), so a garbage token gets no suggestion. Ties go to the
/// earliest candidate, registered elements before keywords before gst names.
fn nearest_known_name(registry: &Registry, name: &str) -> Option<&'static str> {
    let mut best: Option<(usize, &'static str)> = None;
    let candidates = registry
        .element_names()
        .into_iter()
        .chain(LAUNCH_KEYWORDS.iter().copied())
        .chain(GST_MAP.iter().map(|(gst_name, _)| *gst_name));
    for candidate in candidates {
        let distance = edit_distance(name, candidate);
        if best.is_none_or(|(best_distance, _)| distance < best_distance) {
            best = Some((distance, candidate));
        }
    }
    let (distance, candidate) = best?;
    let allowed = (name.len() / TYPO_CHARS_PER_EDIT).min(TYPO_MAX_EDITS);
    (distance <= allowed).then_some(candidate)
}

/// Every GStreamer element name g2g's runtime reports under a different name,
/// as `(gst name, g2g runtime name)`.
///
/// The runtime name is what a graph dump calls the element, its log category,
/// which is the Rust type name and so often not the launch name: gst's
/// `h264parse` is g2g's `NalParse`. A tool comparing the two engines' graphs
/// pairs elements with this; names that already read the same on both sides
/// (`filesrc` against `FileSrc`) are left out, since pairing those needs no
/// table. Backs `g2g-inspect --gst-map`.
pub fn gst_name_synonyms(registry: &Registry) -> Vec<(&'static str, &'static str)> {
    let mut pairs = Vec::new();
    let mut add = |gst_name: &'static str, g2g_name: &str| {
        let Some(runtime) = runtime_name(registry, g2g_name) else {
            return;
        };
        if !same_word(gst_name, runtime) {
            pairs.push((gst_name, runtime));
        }
    };
    for name in registry.element_names() {
        add(name, name);
    }
    for (gst_name, equivalent) in GST_MAP {
        if let GstEquivalent::Renamed(g2g_name) = equivalent {
            add(gst_name, g2g_name);
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// What the runtime calls the element registered as `name`: its log category,
/// which the runner suffixes with an instance number to name a graph node.
fn runtime_name(registry: &Registry, name: &str) -> Option<&'static str> {
    if let Some(element) = registry.make_element(name) {
        return Some(element.log_category());
    }
    registry.make_source(name).map(|s| s.log_category())
}

/// Whether two element names are the same word once case and punctuation are
/// dropped, which is how a graph comparison pairs `filesrc0` with `FileSrc0`.
fn same_word(left: &str, right: &str) -> bool {
    let word = |s: &str| -> String {
        s.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect()
    };
    word(left) == word(right)
}

/// Whether `name` resolves to a registered element of any role (transform/sink,
/// source, or muxer), aliases included.
fn registry_has(registry: &Registry, name: &str) -> bool {
    registry.make_element(name).is_some()
        || registry.make_source(name).is_some()
        || registry.make_muxer(name, 2).is_some()
}

/// The result of linting a `gst-launch` line for g2g portability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintReport {
    /// True when the line is portable as written (every element resolves and it
    /// parses against `registry`).
    pub ok: bool,
    /// Porting guidance, one per issue. Empty when `ok`. Unportable elements are
    /// reported together (every renamed / unsupported / unknown element in the
    /// line, not just the first), so a port is one pass rather than
    /// fix-one-rerun; a structural / property error is reported on its own once
    /// the element names all resolve.
    pub findings: Vec<String>,
}

/// Every element name a `gst-launch` line references, best-effort: the first
/// token of each `!`-separated segment, skipping inline caps filters
/// (`video/x-raw,...`, which contain `/`), tee branch references (`t.`), and
/// stray `key=value` tokens. Good enough for a portability scan; the
/// authoritative element set is whatever [`parse_launch`] builds.
fn element_names(line: &str) -> Vec<&str> {
    let mut names = Vec::new();
    for segment in line.split('!') {
        let Some(first) = segment.split_whitespace().next() else {
            continue;
        };
        // Inline caps filter (media/type,fields) or a branch reference (`t.`) or
        // a bare property token, none of which is an element to look up.
        if first.contains('/') || first.ends_with('.') || first.contains('=') {
            continue;
        }
        names.push(first);
    }
    names
}

/// The porting guidance for one element name, `None` when it is portable as
/// written. Shared by the launch linter, the source scanner, and the parse-error
/// explainer, so all three word the same problem identically.
fn finding(name: &str, equivalent: &GstEquivalent) -> Option<String> {
    match equivalent {
        GstEquivalent::Available => None,
        GstEquivalent::Renamed(g) => Some(format!(
            "`{name}` is not a g2g element name; g2g calls it `{g}` (see `g2g-inspect {g}`)"
        )),
        GstEquivalent::Unsupported(hint) => Some(format!("`{name}` has no g2g element: {hint}")),
        GstEquivalent::NotCompiled(feature) => Some(format!(
            "`{name}` is a g2g element but is not compiled into this build; \
             rebuild with `--features {feature}`"
        )),
        GstEquivalent::DidYouMean(near) => Some(format!(
            "`{name}` is not a g2g element; did you mean `{near}`?"
        )),
        GstEquivalent::Unknown => Some(format!(
            "`{name}` is unknown to g2g with no known equivalent; list elements with `g2g-inspect`"
        )),
    }
}

/// The porting findings for a set of element names, in order, portable names
/// skipped. Shared by the linter, the source scanner, and the parse-error
/// explainer.
fn name_findings<'a>(registry: &Registry, names: impl Iterator<Item = &'a str>) -> Vec<String> {
    names
        .filter_map(|name| finding(name, &gst_equivalent(registry, name)))
        .collect()
}

/// Guidance for a [`ParseError`] that `line` already produced, without re-running
/// the parse (a re-parse would repeat its side effects, like a `uridecodebin`
/// file probe logging the same unreadable path twice). Element-name findings when
/// a name is at fault, else the explained error; empty when the explanation would
/// only restate the error's own message.
pub fn explain_parse_error(registry: &Registry, line: &str, error: &ParseError) -> Vec<String> {
    let findings = name_findings(registry, element_names(line).into_iter());
    if !findings.is_empty() {
        return findings;
    }
    let explained = explain(registry, error);
    if explained == error.to_string() {
        return Vec::new();
    }
    Vec::from([explained])
}

/// Lint a `gst-launch` line for g2g portability. First scans every element name
/// and collects guidance for all that are not portable as-is (renamed,
/// unsupported, or unknown); if all elements resolve, runs the authoritative
/// [`parse_launch`] and, on failure, explains that structural / property error.
pub fn lint_launch(registry: &Registry, line: &str) -> LintReport {
    let findings = name_findings(registry, element_names(line).into_iter());
    if !findings.is_empty() {
        return LintReport {
            ok: false,
            findings,
        };
    }
    // Elements all resolve: let the parser catch caps / property / topology
    // issues (one authoritative error).
    match parse_launch(registry, line) {
        Ok(_) => LintReport {
            ok: true,
            findings: Vec::new(),
        },
        Err(e) => LintReport {
            ok: false,
            findings: Vec::from([explain(registry, &e)]),
        },
    }
}

/// The result of scanning GStreamer application source for g2g portability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceScanReport {
    /// Porting guidance for each distinct non-portable element factory the
    /// source instantiates (renamed / unsupported / unknown), deduplicated and
    /// sorted. Empty when every element resolves.
    pub findings: Vec<String>,
    /// Advisories for dynamic-pipeline APIs the source uses (pad-added relink,
    /// pad probes, appsrc/appsink), each pointing at the porting guidance. These
    /// are not errors: they flag idioms that map to a different g2g primitive.
    pub notes: Vec<String>,
}

/// The quoted string argument immediately following each occurrence of `anchor`,
/// best-effort: only when a `"..."` opens before any `)` / `;` / newline, so a
/// call passing a *variable* (e.g. `gst_parse_launch(pipeline, &err)`) is
/// skipped rather than grabbing an unrelated later literal.
fn quoted_args_after(source: &str, anchor: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = source;
    while let Some(pos) = rest.find(anchor) {
        let after = &rest[pos + anchor.len()..];
        if let Some(q) = after.find('"') {
            let pre = &after[..q];
            if !pre.contains(';') && !pre.contains('\n') && !pre.contains(')') {
                let tail = &after[q + 1..];
                if let Some(end) = tail.find('"') {
                    out.push(tail[..end].to_string());
                }
            }
        }
        rest = after; // strictly shorter, so this terminates
    }
    out
}

/// Scan GStreamer *application source* (C or Python) for g2g portability: the
/// element factories it instantiates (`gst_element_factory_make("x", ...)`,
/// `Gst.ElementFactory.make("x")`, and the elements inside any
/// `gst_parse_launch("...")` / `Gst.parse_launch("...")` string) and the
/// dynamic-pipeline APIs it uses. Best-effort and static, it complements
/// [`lint_launch`] (a single launch string) for apps that build pipelines in
/// code; the authoritative check is still running the ported pipeline.
pub fn scan_source(registry: &Registry, source: &str) -> SourceScanReport {
    // Element factories: the first quoted arg of each make-call, plus every
    // element inside each parse_launch string.
    let mut names: BTreeSet<String> = BTreeSet::new();
    for anchor in ["factory_make", "ElementFactory.make"] {
        for name in quoted_args_after(source, anchor) {
            names.insert(name);
        }
    }
    for line in quoted_args_after(source, "parse_launch") {
        for name in element_names(&line) {
            names.insert(name.to_string());
        }
    }

    let findings = name_findings(registry, names.iter().map(String::as_str));

    // Dynamic-pipeline idioms: map each to its g2g primitive (PORTING.md §5.1).
    let mut notes = Vec::new();
    let has = |needle: &str| source.contains(needle);
    if has("pad-added") {
        notes.push(
            "uses `pad-added` dynamic relink: in g2g use `decodebin`/`uridecodebin` auto-plug, \
             or `StreamDemux` / `register_demux` with typed output ports (PORTING.md §5.1)"
                .to_string(),
        );
    }
    if has("add_probe") || has("pad_add_probe") {
        notes.push(
            "uses pad probes: in g2g register a `LinkInterceptor` on the slot (PORTING.md §5.1)"
                .to_string(),
        );
    }
    if has("appsrc") || has("need-data") || has("push-buffer") {
        notes.push(
            "uses appsrc: g2g has `appsrc channel=<name>` + `register_appsrc`, or `g2g-bridge` \
             for a whole embedded sub-graph (PORTING.md §5.1)"
                .to_string(),
        );
    }
    if has("appsink") || has("new-sample") || has("pull-sample") {
        notes.push(
            "uses appsink: g2g has `appsink channel=<name>` + `set_appsink_callback` (callback) \
             or `register_appsink_pull` (pull) (PORTING.md §5.1)"
                .to_string(),
        );
    }

    SourceScanReport { findings, notes }
}

/// Turn a [`ParseError`] into porting-oriented guidance.
fn explain(registry: &Registry, e: &ParseError) -> String {
    match e {
        ParseError::UnknownElement(n) | ParseError::UnknownSource(n) => {
            let equivalent = gst_equivalent(registry, n);
            finding(n, &equivalent).unwrap_or_else(|| {
                format!(
                    "`{n}` is available; re-check spelling or whether its feature is compiled in"
                )
            })
        }
        ParseError::UnknownProperty { element, key } => {
            format!("`{element}` has no property `{key}`; run `g2g-inspect {element}` for its properties")
        }
        ParseError::BadValue {
            element,
            key,
            value,
        } => {
            format!("`{element}` property `{key}` rejects `{value}`; check its type with `g2g-inspect {element}`")
        }
        ParseError::NotAMuxer(n) => {
            format!("`{n}` has several inputs but is not a registered muxer; use a g2g muxer (`funnel`, `audiomixer`, `mpegtsmux`, ...)")
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capsfilter::parse_caps;
    use crate::registry::default_registry;
    use g2g_core::{Caps, Dim, Rate, RawVideoFormat};

    #[test]
    fn caps_string_round_trips_through_the_parser() {
        let c = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
        };
        assert_eq!(parse_caps(&c.to_gst_string()), Some(c));
    }

    #[test]
    fn the_synonym_table_names_the_elements_the_two_engines_disagree_about() {
        let reg = default_registry();
        let pairs = gst_name_synonyms(&reg);
        // The case that makes the table necessary: gst's parser is a type g2g
        // shares between codecs, so a graph dump never pairs the two by name.
        assert!(pairs.contains(&("h264parse", "NalParse")), "got {pairs:?}");
        for (gst_name, g2g_name) in &pairs {
            assert!(
                !same_word(gst_name, g2g_name),
                "{gst_name} and {g2g_name} already pair without the table"
            );
            assert!(
                runtime_name(&reg, gst_name).is_some_and(|n| n == *g2g_name)
                    || matches!(gst_equivalent(&reg, gst_name), GstEquivalent::Renamed(_)),
                "{gst_name} maps to {g2g_name} through the registry or the rename table"
            );
        }
    }

    #[test]
    fn clean_line_lints_ok() {
        let reg = default_registry();
        let r = lint_launch(&reg, "videotestsrc num-buffers=1 ! videoconvert ! fakesink");
        assert!(r.ok, "findings: {:?}", r.findings);
    }

    // Only meaningful when `x264enc` is NOT compiled in: with the `ffmpeg`
    // feature it is a registered element, so the lint reports no finding.
    #[cfg(not(feature = "ffmpeg"))]
    #[test]
    fn unknown_encoder_gets_a_suggestion() {
        let reg = default_registry();
        let r = lint_launch(&reg, "videotestsrc ! x264enc ! fakesink");
        assert!(!r.ok);
        let msg = &r.findings[0];
        assert!(
            msg.contains("x264enc") && (msg.contains("mfencode") || msg.contains("av1enc")),
            "{msg}"
        );
    }

    #[test]
    fn renamed_element_maps_to_g2g_name() {
        let reg = default_registry();
        assert_eq!(
            gst_equivalent(&reg, "jpegdec"),
            GstEquivalent::Renamed("mjpegdec")
        );
    }

    #[test]
    fn reports_every_unportable_element_not_just_the_first() {
        let reg = default_registry();
        // Two unsupported encoders (feature-independent) in one line: both must
        // appear, so a port is one pass rather than fix-one-rerun.
        let r = lint_launch(&reg, "videotestsrc ! theoraenc ! x265enc ! fakesink");
        assert!(!r.ok);
        assert_eq!(r.findings.len(), 2, "both flagged: {:?}", r.findings);
        assert!(
            r.findings.iter().any(|m| m.contains("theoraenc")),
            "{:?}",
            r.findings
        );
        assert!(
            r.findings.iter().any(|m| m.contains("x265enc")),
            "{:?}",
            r.findings
        );
    }

    #[test]
    fn renamed_element_in_a_line_is_flagged_with_its_g2g_name() {
        let reg = default_registry();
        let r = lint_launch(&reg, "filesrc location=x ! jpegdec ! fakesink");
        assert!(!r.ok);
        assert_eq!(r.findings.len(), 1, "{:?}", r.findings);
        assert!(
            r.findings[0].contains("jpegdec") && r.findings[0].contains("mjpegdec"),
            "{:?}",
            r.findings
        );
    }

    #[test]
    fn caps_filters_and_tee_branches_are_not_mistaken_for_elements() {
        let reg = default_registry();
        // Inline caps filter and a tee branch ref must not be linted as unknown
        // elements; this well-formed line is portable.
        let r = lint_launch(
            &reg,
            "videotestsrc ! video/x-raw,width=320,height=240 ! tee name=t \
             ! queue ! fakesink t. ! queue ! fakesink",
        );
        assert!(r.ok, "findings: {:?}", r.findings);
    }

    #[test]
    fn keyword_and_unknown_classify() {
        let reg = default_registry();
        assert_eq!(gst_equivalent(&reg, "tee"), GstEquivalent::Available);
        assert_eq!(
            gst_equivalent(&reg, "videoconvert"),
            GstEquivalent::Available
        );
        assert_eq!(
            gst_equivalent(&reg, "totally-made-up"),
            GstEquivalent::Unknown
        );
    }

    #[test]
    fn a_misspelled_element_gets_a_suggestion() {
        let reg = default_registry();
        assert_eq!(
            gst_equivalent(&reg, "filesrcc"),
            GstEquivalent::DidYouMean("filesrc")
        );
        let r = lint_launch(&reg, "filesrcc location=x ! fakesink");
        assert!(!r.ok);
        assert_eq!(r.findings.len(), 1, "{:?}", r.findings);
        assert!(
            r.findings[0].contains("did you mean `filesrc`"),
            "{:?}",
            r.findings
        );
    }

    #[test]
    fn a_garbage_token_gets_no_suggestion() {
        let reg = default_registry();
        for name in ["totally-made-up", "zzzz", "xyzzy", "qqqqqqqqqqqq"] {
            assert_eq!(
                gst_equivalent(&reg, name),
                GstEquivalent::Unknown,
                "`{name}` must not get a suggestion"
            );
        }
    }

    // Only meaningful when `srt` is NOT compiled in: with the feature the element
    // is registered, so it resolves as `Available`.
    #[cfg(not(feature = "srt"))]
    #[test]
    fn a_feature_gated_element_names_its_feature() {
        let reg = default_registry();
        assert_eq!(
            gst_equivalent(&reg, "srtsink"),
            GstEquivalent::NotCompiled("srt")
        );
        let r = lint_launch(&reg, "filesrc location=x ! srtsink");
        assert!(!r.ok);
        assert!(r.findings[0].contains("--features srt"), "{:?}", r.findings);
    }

    #[test]
    fn registered_appsrc_appsink_are_available_not_unsupported() {
        let reg = default_registry();
        assert_eq!(gst_equivalent(&reg, "appsrc"), GstEquivalent::Available);
        assert_eq!(gst_equivalent(&reg, "appsink"), GstEquivalent::Available);
    }

    #[test]
    fn scans_c_source_for_factories_and_dynamic_apis() {
        let reg = default_registry();
        // A snippet of a C GStreamer app: factory_make calls (one renamed), a
        // parse_launch string (one unsupported element), a pad-added handler.
        let src = r#"
            GstElement *conv = gst_element_factory_make("videoconvert", "c");
            GstElement *dec  = gst_element_factory_make("jpegdec", "d");
            pipeline = gst_parse_launch("videotestsrc ! theoraenc ! fakesink", &err);
            g_signal_connect(demux, "pad-added", G_CALLBACK(on_pad_added), NULL);
        "#;
        let r = scan_source(&reg, src);
        // videoconvert is available (no finding); jpegdec renamed; theoraenc unsupported.
        assert!(
            r.findings
                .iter()
                .any(|m| m.contains("jpegdec") && m.contains("mjpegdec")),
            "{:?}",
            r.findings
        );
        assert!(
            r.findings.iter().any(|m| m.contains("theoraenc")),
            "{:?}",
            r.findings
        );
        assert!(
            !r.findings.iter().any(|m| m.contains("videoconvert")),
            "available element flagged: {:?}",
            r.findings
        );
        assert!(
            r.notes.iter().any(|n| n.contains("pad-added")),
            "notes: {:?}",
            r.notes
        );
    }

    #[test]
    fn scans_python_source_and_ignores_variable_parse_launch() {
        let reg = default_registry();
        let src = r#"
            conv = Gst.ElementFactory.make("videoconvert", "conv")
            sink = Gst.ElementFactory.make("appsink", "sink")
            pipeline = Gst.parse_launch(user_supplied_string)  # variable, not a literal
        "#;
        let r = scan_source(&reg, src);
        // appsink resolves (registered); videoconvert too; the variable
        // parse_launch yields no phantom element findings.
        assert!(
            r.findings.is_empty(),
            "unexpected findings: {:?}",
            r.findings
        );
        // appsink triggers the dynamic-API note.
        assert!(
            r.notes.iter().any(|n| n.contains("appsink")),
            "notes: {:?}",
            r.notes
        );
    }
}
