//! GStreamer-to-g2g porting helpers (M200): a `gst`-element-name map and a
//! launch-line linter that turns parse failures into porting guidance.
//!
//! These back `g2g-inspect --gst <name>` and `g2g-launch`'s explain-on-error,
//! and are the programmatic surface a porting tool builds on. They complement
//! [`parse_launch`](g2g_core::runtime::parse_launch) (the authoritative parse):
//! the linter runs it and enriches the first error with a gst->g2g suggestion,
//! so porting is fix-and-rerun rather than decode-the-error.

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
        "no software H.264 encoder; use `mfencode` (Windows), or encode AV1/VP8/VP9 with `av1enc`/`vpxenc`",
    )),
    ("x265enc", GstEquivalent::Unsupported("no H.265 encoder; consider `av1enc`")),
    ("theoraenc", GstEquivalent::Unsupported("no Theora encoder; use `vpxenc` (VP8/VP9) or `av1enc`")),
    ("avdec_h264", GstEquivalent::Renamed("ffmpegdec")),
    ("avdec_h265", GstEquivalent::Renamed("ffmpegdec")),
    ("nvh264dec", GstEquivalent::Renamed("ffmpegdec")),
    ("nvh264enc", GstEquivalent::Unsupported("no NVENC encode element; software / AV1 paths only")),
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
    /// True when the line parses against `registry` (it is portable as written).
    pub ok: bool,
    /// Porting guidance, one per issue. Empty when `ok`. The authoritative parser
    /// stops at the first error, so this reports one issue at a time
    /// (fix-and-rerun).
    pub findings: Vec<String>,
}

/// Lint a `gst-launch` line for g2g portability: parse it, and on failure turn
/// the error into a gst->g2g porting suggestion.
pub fn lint_launch(registry: &Registry, line: &str) -> LintReport {
    match parse_launch(registry, line) {
        Ok(_) => LintReport { ok: true, findings: Vec::new() },
        Err(e) => LintReport { ok: false, findings: Vec::from([explain(registry, &e)]) },
    }
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
        ParseError::FanOutWithoutTee(n) => {
            format!("`{n}` feeds several branches; g2g needs an explicit `tee` (gst auto-inserts one, g2g does not)")
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
}
