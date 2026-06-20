//! DASH MPD manifest parser (a subset of ISO/IEC 23009-1), driven by
//! [`DashSrc`](crate::dashsrc). Pure (no I/O), so it is fully unit-testable.
//!
//! Scope: static (VOD) manifests using `SegmentTemplate` with `$Number$`
//! addressing and an explicit `@duration` (the dominant DASH-IF profile).
//! `SegmentTimeline`, `SegmentList`, `SegmentBase` byte-ranges, and dynamic
//! (live) manifests are follow-ups. Attribute inheritance (geometry / codecs /
//! the `SegmentTemplate` itself declared on the `AdaptationSet` and shared by its
//! `Representation`s) is resolved by walking ancestors.

use alloc::string::String;
use alloc::vec::Vec;

use roxmltree::{Document, Node};

/// One selectable Representation (a single quality rendition).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Representation {
    pub id: String,
    pub bandwidth: u64,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub codecs: Option<String>,
    pub mime_type: Option<String>,
    pub template: SegmentTemplate,
}

/// `SegmentTemplate` with `$Number$` addressing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentTemplate {
    /// Init-segment template (resolves `$RepresentationID$`).
    pub initialization: Option<String>,
    /// Media-segment template (resolves `$RepresentationID$` and `$Number$`).
    pub media: String,
    pub start_number: u64,
    /// Segment duration in `timescale` units.
    pub duration: u64,
    pub timescale: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Mpd {
    pub base_url: Option<String>,
    /// `mediaPresentationDuration` in seconds (for the VOD segment count).
    pub duration_secs: f64,
    pub representations: Vec<Representation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpdError {
    /// XML did not parse, or no usable Representation was found.
    Invalid,
}

impl SegmentTemplate {
    /// Number of segments for a VOD presentation of `total_secs`.
    pub fn segment_count(&self, total_secs: f64) -> u64 {
        if self.duration == 0 || self.timescale == 0 {
            return 0;
        }
        let seg_secs = self.duration as f64 / self.timescale as f64;
        if seg_secs <= 0.0 {
            return 0;
        }
        (total_secs / seg_secs).ceil() as u64
    }

    /// The init-segment URL template expanded for `rep_id`.
    pub fn init_url(&self, rep_id: &str) -> Option<String> {
        self.initialization.as_ref().map(|t| expand(t, rep_id, None))
    }

    /// The media-segment URL template expanded for `rep_id` and `number`.
    pub fn media_url(&self, rep_id: &str, number: u64) -> String {
        expand(&self.media, rep_id, Some(number))
    }
}

impl Mpd {
    /// Pick the highest-bandwidth Representation at or below `max_bandwidth`
    /// (or the overall highest when `None` / nothing fits).
    pub fn select(&self, max_bandwidth: Option<u64>) -> Option<&Representation> {
        let under = |r: &&Representation| max_bandwidth.map_or(true, |cap| r.bandwidth <= cap);
        self.representations
            .iter()
            .filter(under)
            .max_by_key(|r| r.bandwidth)
            .or_else(|| self.representations.iter().min_by_key(|r| r.bandwidth))
    }
}

/// Parse an MPD manifest.
pub fn parse(xml: &str) -> Result<Mpd, MpdError> {
    let doc = Document::parse(xml).map_err(|_| MpdError::Invalid)?;
    let root = doc.root_element();

    let duration_secs =
        root.attribute("mediaPresentationDuration").and_then(parse_iso_duration).unwrap_or(0.0);
    let base_url = root
        .descendants()
        .find(|n| n.has_tag_name("BaseURL"))
        .and_then(|n| n.text())
        .map(|s| String::from(s.trim()));

    let mut representations = Vec::new();
    for rep in root.descendants().filter(|n| n.has_tag_name("Representation")) {
        let Some(id) = rep.attribute("id") else { continue };
        let Some(template) = segment_template(rep) else { continue };
        representations.push(Representation {
            id: String::from(id),
            bandwidth: inherited(rep, "bandwidth").and_then(|s| s.parse().ok()).unwrap_or(0),
            width: inherited(rep, "width").and_then(|s| s.parse().ok()),
            height: inherited(rep, "height").and_then(|s| s.parse().ok()),
            codecs: inherited(rep, "codecs").map(String::from),
            mime_type: inherited(rep, "mimeType").map(String::from),
            template,
        });
    }

    if representations.is_empty() {
        return Err(MpdError::Invalid);
    }
    Ok(Mpd { base_url, duration_secs, representations })
}

/// The nearest `SegmentTemplate` for a Representation (its own, else inherited
/// from an ancestor AdaptationSet / Period), parsed into a [`SegmentTemplate`].
/// Requires a `media` attribute (number addressing).
fn segment_template(rep: Node) -> Option<SegmentTemplate> {
    let st = rep
        .ancestors()
        .find_map(|n| n.children().find(|c| c.is_element() && c.has_tag_name("SegmentTemplate")))?;
    Some(SegmentTemplate {
        initialization: st.attribute("initialization").map(String::from),
        media: String::from(st.attribute("media")?),
        start_number: st.attribute("startNumber").and_then(|s| s.parse().ok()).unwrap_or(1),
        duration: st.attribute("duration").and_then(|s| s.parse().ok()).unwrap_or(0),
        timescale: st.attribute("timescale").and_then(|s| s.parse().ok()).unwrap_or(1),
    })
}

/// An attribute on `node` or the nearest ancestor that carries it.
fn inherited<'a>(node: Node<'a, '_>, name: &str) -> Option<&'a str> {
    node.ancestors().find_map(|n| n.attribute(name))
}

/// Expand a `SegmentTemplate` URL: `$$` -> `$`, `$RepresentationID$` -> id,
/// `$Number$` / `$Number%0Nd$` -> the segment number (zero-padded).
fn expand(tmpl: &str, rep_id: &str, number: Option<u64>) -> String {
    let mut out = String::new();
    for (i, part) in tmpl.split('$').enumerate() {
        if i % 2 == 0 {
            out.push_str(part);
        } else if part.is_empty() {
            out.push('$'); // "$$"
        } else if part == "RepresentationID" {
            out.push_str(rep_id);
        } else if let Some(fmt) = part.strip_prefix("Number") {
            out.push_str(&format_number(fmt, number.unwrap_or(0)));
        }
        // unknown identifier (eg $Time$ in a timeline profile): dropped
    }
    out
}

/// Format a `$Number...$` value, honoring a `%0Nd` zero-pad width.
fn format_number(fmt: &str, n: u64) -> String {
    if let Some(width) = fmt.strip_prefix("%0").and_then(|s| s.strip_suffix('d')).and_then(|s| s.parse::<usize>().ok()) {
        alloc::format!("{n:0width$}")
    } else {
        alloc::format!("{n}")
    }
}

/// Parse an ISO 8601 duration's time component (`PT1H2M3.5S`) to seconds. The
/// date part (years/months/days before `T`) is not expected in media durations
/// and is ignored.
fn parse_iso_duration(s: &str) -> Option<f64> {
    let time = s.split_once('T').map(|(_, t)| t).unwrap_or("");
    let mut secs = 0.0f64;
    let mut num = String::new();
    for ch in time.chars() {
        match ch {
            '0'..='9' | '.' => num.push(ch),
            'H' => {
                secs += num.parse::<f64>().ok()? * 3600.0;
                num.clear();
            }
            'M' => {
                secs += num.parse::<f64>().ok()? * 60.0;
                num.clear();
            }
            'S' => {
                secs += num.parse::<f64>().ok()?;
                num.clear();
            }
            _ => return None,
        }
    }
    Some(secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MPD: &str = r#"<?xml version="1.0"?>
<MPD mediaPresentationDuration="PT0H0M12.0S" type="static">
  <Period>
    <AdaptationSet mimeType="video/mp4" codecs="avc1.4d401f">
      <SegmentTemplate initialization="init-$RepresentationID$.mp4"
                       media="seg-$RepresentationID$-$Number%03d$.m4s"
                       startNumber="1" duration="4000" timescale="1000"/>
      <Representation id="low" bandwidth="800000" width="640" height="360"/>
      <Representation id="high" bandwidth="2400000" width="1280" height="720"/>
    </AdaptationSet>
  </Period>
</MPD>"#;

    #[test]
    fn parses_representations_with_inherited_template_and_geometry() {
        let mpd = parse(MPD).unwrap();
        assert!((mpd.duration_secs - 12.0).abs() < 1e-6);
        assert_eq!(mpd.representations.len(), 2);
        let high = mpd.select(None).unwrap();
        assert_eq!(high.id, "high");
        assert_eq!(high.bandwidth, 2_400_000);
        assert_eq!(high.width, Some(1280));
        // codecs inherited from the AdaptationSet
        assert_eq!(high.codecs.as_deref(), Some("avc1.4d401f"));
        assert_eq!(high.template.timescale, 1000);
        assert_eq!(high.template.duration, 4000);
    }

    #[test]
    fn abr_caps_selection() {
        let mpd = parse(MPD).unwrap();
        assert_eq!(mpd.select(Some(1_000_000)).unwrap().id, "low");
        assert_eq!(mpd.select(Some(1)).unwrap().id, "low"); // fallback to lowest
    }

    #[test]
    fn segment_count_and_url_templating() {
        let mpd = parse(MPD).unwrap();
        let rep = mpd.select(None).unwrap();
        // 12s / 4s = 3 segments.
        assert_eq!(rep.template.segment_count(mpd.duration_secs), 3);
        assert_eq!(rep.template.init_url(&rep.id).as_deref(), Some("init-high.mp4"));
        assert_eq!(rep.template.media_url(&rep.id, 1), "seg-high-001.m4s");
        assert_eq!(rep.template.media_url(&rep.id, 12), "seg-high-012.m4s");
    }

    #[test]
    fn iso_duration_forms() {
        assert_eq!(parse_iso_duration("PT12.0S"), Some(12.0));
        assert_eq!(parse_iso_duration("PT1H2M3S"), Some(3723.0));
        assert_eq!(parse_iso_duration("PT0.5S"), Some(0.5));
    }

    #[test]
    fn rejects_non_mpd() {
        assert_eq!(parse("not xml at all <<<"), Err(MpdError::Invalid));
    }
}
