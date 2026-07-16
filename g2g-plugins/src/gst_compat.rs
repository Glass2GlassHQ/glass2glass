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
    /// Unknown to both the registry and the gst-compat table: cannot advise.
    Unknown,
}

/// Launch keywords the parser handles that are not registry elements.
static LAUNCH_KEYWORDS: &[&str] =
    &["decodebin", "uridecodebin", "playbin", "queue", "queue2", "tee"];

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
];

/// Map a GStreamer element name to its g2g equivalent, consulting the live
/// `registry` first (so aliases resolve and feature-gated elements that ARE
/// compiled in show as `Available`), then the launch keywords, then the static
/// guidance table.
pub fn gst_equivalent(registry: &Registry, gst_name: &str) -> GstEquivalent {
    if registry_has(registry, gst_name) || LAUNCH_KEYWORDS.contains(&gst_name) {
        return GstEquivalent::Available;
    }
    GST_MAP
        .iter()
        .find(|(name, _)| *name == gst_name)
        .map(|(_, eq)| eq.clone())
        .unwrap_or(GstEquivalent::Unknown)
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
        let Some(first) = segment.split_whitespace().next() else { continue };
        // Inline caps filter (media/type,fields) or a branch reference (`t.`) or
        // a bare property token, none of which is an element to look up.
        if first.contains('/') || first.ends_with('.') || first.contains('=') {
            continue;
        }
        names.push(first);
    }
    names
}

/// Lint a `gst-launch` line for g2g portability. First scans every element name
/// and collects guidance for all that are not portable as-is (renamed,
/// unsupported, or unknown); if all elements resolve, runs the authoritative
/// [`parse_launch`] and, on failure, explains that structural / property error.
pub fn lint_launch(registry: &Registry, line: &str) -> LintReport {
    let mut findings = Vec::new();
    for name in element_names(line) {
        match gst_equivalent(registry, name) {
            GstEquivalent::Available => {}
            GstEquivalent::Renamed(g) => findings.push(format!(
                "`{name}` is not a g2g element name; g2g calls it `{g}` (see `g2g-inspect {g}`)"
            )),
            GstEquivalent::Unsupported(hint) => {
                findings.push(format!("`{name}` has no g2g element: {hint}"))
            }
            GstEquivalent::Unknown => findings.push(format!(
                "`{name}` is unknown to g2g with no known equivalent; list elements with `g2g-inspect`"
            )),
        }
    }
    if !findings.is_empty() {
        return LintReport { ok: false, findings };
    }
    // Elements all resolve: let the parser catch caps / property / topology
    // issues (one authoritative error).
    match parse_launch(registry, line) {
        Ok(_) => LintReport { ok: true, findings: Vec::new() },
        Err(e) => LintReport { ok: false, findings: Vec::from([explain(registry, &e)]) },
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

    let mut findings = Vec::new();
    for name in &names {
        match gst_equivalent(registry, name) {
            GstEquivalent::Available => {}
            GstEquivalent::Renamed(g) => findings.push(format!(
                "`{name}` is not a g2g element name; g2g calls it `{g}` (see `g2g-inspect {g}`)"
            )),
            GstEquivalent::Unsupported(hint) => {
                findings.push(format!("`{name}` has no g2g element: {hint}"))
            }
            GstEquivalent::Unknown => findings.push(format!(
                "`{name}` is unknown to g2g with no known equivalent; list elements with `g2g-inspect`"
            )),
        }
    }

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
        ParseError::UnknownElement(n) | ParseError::UnknownSource(n) => match gst_equivalent(registry, n) {
            GstEquivalent::Renamed(g) => {
                format!("`{n}` is not a g2g element name; g2g calls it `{g}` (see `g2g-inspect {g}`)")
            }
            GstEquivalent::Unsupported(hint) => format!("`{n}` has no g2g element: {hint}"),
            GstEquivalent::Available => {
                format!("`{n}` is available; re-check spelling or whether its feature is compiled in")
            }
            GstEquivalent::Unknown => {
                format!("`{n}` is unknown to g2g with no known equivalent; list elements with `g2g-inspect`")
            }
        },
        ParseError::UnknownProperty { element, key } => {
            format!("`{element}` has no property `{key}`; run `g2g-inspect {element}` for its properties")
        }
        ParseError::BadValue { element, key, value } => {
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
        };
        assert_eq!(parse_caps(&c.to_gst_string()), Some(c));
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
        assert!(msg.contains("x264enc") && (msg.contains("mfencode") || msg.contains("av1enc")), "{msg}");
    }

    #[test]
    fn renamed_element_maps_to_g2g_name() {
        let reg = default_registry();
        assert_eq!(gst_equivalent(&reg, "jpegdec"), GstEquivalent::Renamed("mjpegdec"));
    }

    #[test]
    fn reports_every_unportable_element_not_just_the_first() {
        let reg = default_registry();
        // Two unsupported encoders (feature-independent) in one line: both must
        // appear, so a port is one pass rather than fix-one-rerun.
        let r = lint_launch(&reg, "videotestsrc ! theoraenc ! x265enc ! fakesink");
        assert!(!r.ok);
        assert_eq!(r.findings.len(), 2, "both flagged: {:?}", r.findings);
        assert!(r.findings.iter().any(|m| m.contains("theoraenc")), "{:?}", r.findings);
        assert!(r.findings.iter().any(|m| m.contains("x265enc")), "{:?}", r.findings);
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
        assert_eq!(gst_equivalent(&reg, "videoconvert"), GstEquivalent::Available);
        assert_eq!(gst_equivalent(&reg, "totally-made-up"), GstEquivalent::Unknown);
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
        assert!(r.findings.iter().any(|m| m.contains("jpegdec") && m.contains("mjpegdec")), "{:?}", r.findings);
        assert!(r.findings.iter().any(|m| m.contains("theoraenc")), "{:?}", r.findings);
        assert!(!r.findings.iter().any(|m| m.contains("videoconvert")), "available element flagged: {:?}", r.findings);
        assert!(r.notes.iter().any(|n| n.contains("pad-added")), "notes: {:?}", r.notes);
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
        assert!(r.findings.is_empty(), "unexpected findings: {:?}", r.findings);
        // appsink triggers the dynamic-API note.
        assert!(r.notes.iter().any(|n| n.contains("appsink")), "notes: {:?}", r.notes);
    }
}
