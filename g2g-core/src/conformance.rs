//! Conformance vocabulary + derived maturity (M614).
//!
//! g2g grows fast under agent-driven development, so "how validated is this
//! element?" must be answerable honestly. The trap is a hand-authored maturity
//! field: under fast iteration it drifts into an overclaim (a level bumped in the
//! same change that adds the feature). This module removes that trap by making
//! maturity a **pure function of evidence**, where evidence is produced only by a
//! conformance case that actually ran and passed.
//!
//! An element's [`MaturityRecord`] is a bag of [`Evidence`], each tagging one
//! [`ConformanceDimension`] that was verified (optionally with the platform, codec,
//! or external peer it was verified against). [`MaturityRecord::level`] derives a
//! conservative headline [`MaturityLevel`] from that bag: you cannot reach
//! [`InteropTested`](MaturityLevel::InteropTested) without an [`Oracle`] evidence
//! naming a *peer*, nor [`HardwareValidated`](MaturityLevel::HardwareValidated)
//! without a [`Hardware`] evidence naming a *platform*. There is no setter for the
//! level, so the record cannot claim more than its evidence supports.
//!
//! Crucially, the *absence* of evidence is itself the honest signal: an element that
//! round-trips in-process but has never been checked against an external
//! implementation carries no `Oracle` evidence and so lands at
//! [`UnitTested`](MaturityLevel::UnitTested), which is exactly the
//! "loopback-tested, not interop-validated" caveat expressed as data rather than a
//! comment.
//!
//! [`Oracle`]: ConformanceDimension::Oracle
//! [`Hardware`]: ConformanceDimension::Hardware

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// A kind of check a conformance case can verify about an element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConformanceDimension {
    /// The element constructs and advertises its metadata / pad caps.
    Instantiate,
    /// Every advertised property round-trips through `set` / `get` and rejects a
    /// bad value.
    Properties,
    /// Data survives the element (or a packetize / depacketize, encode / decode
    /// pair) intact: a self-contained behavioral check.
    RoundTrip,
    /// The element recovers from lost / reordered / duplicated input (e.g. a
    /// depacketizer through dropped packets, a -7 seamless merge).
    LossResilience,
    /// A graph built around the element is proven zero-copy by the copy plan
    /// (`crate::copyplan`): no host round-trip of a raw frame.
    ZeroCopy,
    /// A measured latency / throughput figure (informational; does not raise the
    /// maturity level on its own).
    Latency,
    /// Validated against an external reference implementation (ffmpeg, GStreamer, a
    /// hardware peer). Only counts toward maturity when it names the `peer`.
    Oracle,
    /// Exercised on real hardware / a real device. Only counts toward maturity when
    /// it names the `platform`.
    Hardware,
}

impl ConformanceDimension {
    /// Every dimension, in reporting order.
    pub const ALL: [ConformanceDimension; 8] = [
        ConformanceDimension::Instantiate,
        ConformanceDimension::Properties,
        ConformanceDimension::RoundTrip,
        ConformanceDimension::LossResilience,
        ConformanceDimension::ZeroCopy,
        ConformanceDimension::Latency,
        ConformanceDimension::Oracle,
        ConformanceDimension::Hardware,
    ];

    /// A short kebab-case label.
    pub fn label(self) -> &'static str {
        match self {
            ConformanceDimension::Instantiate => "instantiate",
            ConformanceDimension::Properties => "properties",
            ConformanceDimension::RoundTrip => "round-trip",
            ConformanceDimension::LossResilience => "loss-resilience",
            ConformanceDimension::ZeroCopy => "zero-copy",
            ConformanceDimension::Latency => "latency",
            ConformanceDimension::Oracle => "oracle",
            ConformanceDimension::Hardware => "hardware",
        }
    }

    /// Parse a [`label`](Self::label) back (for the persisted evidence log). `None`
    /// for an unknown token, so a malformed log line is skipped, not trusted.
    pub fn from_label(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|d| d.label() == s)
    }
}

/// One passed conformance check: the dimension it verified, plus the context it was
/// verified in. Build with [`Evidence::new`] and the context setters; construct one
/// only when the check actually passed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evidence {
    /// What was verified.
    pub dimension: ConformanceDimension,
    /// The platform it ran on (e.g. `"linux"`, `"pixel-10a"`). Required for
    /// [`Hardware`](ConformanceDimension::Hardware) to count toward maturity.
    pub platform: Option<String>,
    /// The codec / format the check covered (e.g. `"h264"`, `"rgba8"`).
    pub codec: Option<String>,
    /// The external implementation it was validated against (e.g. `"ffmpeg"`).
    /// Required for [`Oracle`](ConformanceDimension::Oracle) to count.
    pub peer: Option<String>,
    /// A free-text note (fixture name, measured figure, caveat).
    pub detail: Option<String>,
}

impl Evidence {
    /// Evidence for `dimension` with no context yet.
    pub fn new(dimension: ConformanceDimension) -> Self {
        Self {
            dimension,
            platform: None,
            codec: None,
            peer: None,
            detail: None,
        }
    }

    /// Tag the platform this ran on.
    pub fn platform(mut self, p: impl Into<String>) -> Self {
        self.platform = Some(p.into());
        self
    }

    /// Tag the codec / format covered.
    pub fn codec(mut self, c: impl Into<String>) -> Self {
        self.codec = Some(c.into());
        self
    }

    /// Tag the external peer validated against.
    pub fn peer(mut self, p: impl Into<String>) -> Self {
        self.peer = Some(p.into());
        self
    }

    /// Attach a free-text note.
    pub fn detail(mut self, d: impl Into<String>) -> Self {
        self.detail = Some(d.into());
        self
    }
}

/// The conservative headline maturity of an element, derived from its evidence.
/// Ordered: a higher level strictly implies more validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MaturityLevel {
    /// No conformance evidence.
    Unverified,
    /// Constructs and advertises its interface (caps / properties).
    Instantiated,
    /// Passes a self-contained behavioral check (round-trip, loss resilience, or a
    /// zero-copy graph), but has not been validated against an external peer.
    UnitTested,
    /// Validated against an external reference implementation (a named peer).
    InteropTested,
    /// Validated on real hardware (a named platform).
    HardwareValidated,
}

impl MaturityLevel {
    /// A short label.
    pub fn label(self) -> &'static str {
        match self {
            MaturityLevel::Unverified => "unverified",
            MaturityLevel::Instantiated => "instantiated",
            MaturityLevel::UnitTested => "unit-tested",
            MaturityLevel::InteropTested => "interop-tested",
            MaturityLevel::HardwareValidated => "hardware-validated",
        }
    }
}

/// One element's conformance evidence, from which its [`MaturityLevel`] is derived.
/// There is deliberately no way to set the level directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaturityRecord {
    /// The element's name (its registry / log category).
    pub element: String,
    /// Every passed check.
    pub evidence: Vec<Evidence>,
}

impl MaturityRecord {
    /// A record for `element` with no evidence yet (maturity [`Unverified`]).
    ///
    /// [`Unverified`]: MaturityLevel::Unverified
    pub fn new(element: impl Into<String>) -> Self {
        Self {
            element: element.into(),
            evidence: Vec::new(),
        }
    }

    /// Add one piece of evidence (builder style).
    pub fn with(mut self, e: Evidence) -> Self {
        self.evidence.push(e);
        self
    }

    /// Add one piece of evidence.
    pub fn add(&mut self, e: Evidence) {
        self.evidence.push(e);
    }

    /// Whether any evidence covers `dimension`.
    pub fn has(&self, dimension: ConformanceDimension) -> bool {
        self.evidence.iter().any(|e| e.dimension == dimension)
    }

    /// The distinct dimensions with evidence, in reporting order.
    pub fn dimensions(&self) -> Vec<ConformanceDimension> {
        ConformanceDimension::ALL
            .into_iter()
            .filter(|&d| self.has(d))
            .collect()
    }

    /// The distinct external peers this element was validated against.
    pub fn peers(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self
            .evidence
            .iter()
            .filter_map(|e| e.peer.as_deref())
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The distinct platforms this element was validated on.
    pub fn platforms(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self
            .evidence
            .iter()
            .filter_map(|e| e.platform.as_deref())
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The derived headline maturity. `Oracle` counts only with a named peer and
    /// `Hardware` only with a named platform, so the level never overstates what the
    /// evidence supports.
    pub fn level(&self) -> MaturityLevel {
        use ConformanceDimension as D;
        let has_hardware = self
            .evidence
            .iter()
            .any(|e| e.dimension == D::Hardware && e.platform.is_some());
        let has_interop = self
            .evidence
            .iter()
            .any(|e| e.dimension == D::Oracle && e.peer.is_some());
        let behavioral =
            self.has(D::RoundTrip) || self.has(D::LossResilience) || self.has(D::ZeroCopy);
        let advertised = self.has(D::Instantiate) || self.has(D::Properties);
        if has_hardware {
            MaturityLevel::HardwareValidated
        } else if has_interop {
            MaturityLevel::InteropTested
        } else if behavioral {
            MaturityLevel::UnitTested
        } else if advertised {
            MaturityLevel::Instantiated
        } else {
            MaturityLevel::Unverified
        }
    }
}

/// A collection of [`MaturityRecord`]s, rendered as a matrix table.
#[derive(Debug, Clone, Default)]
pub struct ConformanceReport {
    /// Per-element records.
    pub records: Vec<MaturityRecord>,
}

impl ConformanceReport {
    /// An empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a record.
    pub fn push(&mut self, record: MaturityRecord) {
        self.records.push(record);
    }

    /// The record for `element`, inserting an empty one if absent.
    pub fn record_mut(&mut self, element: &str) -> &mut MaturityRecord {
        if let Some(i) = self.records.iter().position(|r| r.element == element) {
            &mut self.records[i]
        } else {
            self.records.push(MaturityRecord::new(element));
            self.records.last_mut().expect("just pushed")
        }
    }

    /// Merge another report into this one: each record's evidence is unioned into the
    /// matching element (deduplicating identical evidence). Used to fold persisted
    /// `Oracle` / `Hardware` evidence (from the resource-owning tests) into the
    /// in-process battery report, so the derived level rises to reflect it.
    pub fn absorb(&mut self, other: ConformanceReport) {
        for record in other.records {
            let dst = self.record_mut(&record.element);
            for ev in record.evidence {
                if !dst.evidence.contains(&ev) {
                    dst.evidence.push(ev);
                }
            }
        }
    }

    /// The lowest maturity level across all records (the weakest link), or
    /// [`Unverified`](MaturityLevel::Unverified) for an empty report.
    pub fn min_level(&self) -> MaturityLevel {
        self.records
            .iter()
            .map(MaturityRecord::level)
            .min()
            .unwrap_or(MaturityLevel::Unverified)
    }

    /// Render the report as an aligned text table: element, derived level, the
    /// dimensions with evidence, and any peers / platforms.
    pub fn to_table(&self) -> String {
        let rows: Vec<(String, String, String, String)> = self
            .records
            .iter()
            .map(|r| {
                let dims = r
                    .dimensions()
                    .iter()
                    .map(|d| d.label())
                    .collect::<Vec<_>>()
                    .join(", ");
                let mut context = Vec::new();
                let peers = r.peers();
                if !peers.is_empty() {
                    context.push(format!("peers: {}", peers.join(", ")));
                }
                let plats = r.platforms();
                if !plats.is_empty() {
                    context.push(format!("platforms: {}", plats.join(", ")));
                }
                (
                    r.element.clone(),
                    r.level().label().to_string(),
                    dims,
                    context.join("; "),
                )
            })
            .collect();

        let w_el = rows.iter().map(|r| r.0.len()).chain([7]).max().unwrap_or(7);
        let w_lv = rows.iter().map(|r| r.1.len()).chain([5]).max().unwrap_or(5);
        let mut s = String::new();
        s.push_str(&format!(
            "{:<w_el$}  {:<w_lv$}  dimensions\n",
            "element", "maturity"
        ));
        for (el, lv, dims, ctx) in &rows {
            s.push_str(&format!("{el:<w_el$}  {lv:<w_lv$}  {dims}\n"));
            if !ctx.is_empty() {
                s.push_str(&format!("{:<w_el$}  {:<w_lv$}  ({ctx})\n", "", ""));
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_record_is_unverified() {
        assert_eq!(MaturityRecord::new("x").level(), MaturityLevel::Unverified);
    }

    #[test]
    fn instantiate_alone_is_instantiated() {
        let r = MaturityRecord::new("capsfilter")
            .with(Evidence::new(ConformanceDimension::Instantiate))
            .with(Evidence::new(ConformanceDimension::Properties));
        assert_eq!(r.level(), MaturityLevel::Instantiated);
    }

    #[test]
    fn a_round_trip_reaches_unit_tested_but_not_interop() {
        // The honesty case: a loopback round-trip proves behavior but is NOT interop
        // validation, so with no peer-tagged oracle the element stays UnitTested.
        let r = MaturityRecord::new("st2110video")
            .with(Evidence::new(ConformanceDimension::Instantiate))
            .with(Evidence::new(ConformanceDimension::RoundTrip).codec("rgba8"));
        assert_eq!(r.level(), MaturityLevel::UnitTested);
        assert!(!r.has(ConformanceDimension::Oracle), "no interop claim");
    }

    #[test]
    fn oracle_without_a_peer_does_not_reach_interop() {
        // A bare Oracle evidence with no named peer is a hollow claim and must not
        // raise the level past UnitTested.
        let r = MaturityRecord::new("h264enc")
            .with(Evidence::new(ConformanceDimension::RoundTrip))
            .with(Evidence::new(ConformanceDimension::Oracle));
        assert_eq!(r.level(), MaturityLevel::UnitTested);
    }

    #[test]
    fn oracle_with_a_peer_reaches_interop() {
        let r = MaturityRecord::new("h264enc")
            .with(Evidence::new(ConformanceDimension::RoundTrip))
            .with(
                Evidence::new(ConformanceDimension::Oracle)
                    .peer("ffmpeg")
                    .codec("h264"),
            );
        assert_eq!(r.level(), MaturityLevel::InteropTested);
        assert_eq!(r.peers(), alloc::vec!["ffmpeg"]);
    }

    #[test]
    fn hardware_with_a_platform_is_the_top_level() {
        let r = MaturityRecord::new("nvh264dec")
            .with(Evidence::new(ConformanceDimension::Oracle).peer("ffmpeg"))
            .with(Evidence::new(ConformanceDimension::Hardware).platform("rtx-3060"));
        assert_eq!(r.level(), MaturityLevel::HardwareValidated);
        assert_eq!(r.platforms(), alloc::vec!["rtx-3060"]);
    }

    #[test]
    fn levels_are_ordered() {
        assert!(MaturityLevel::Unverified < MaturityLevel::Instantiated);
        assert!(MaturityLevel::UnitTested < MaturityLevel::InteropTested);
        assert!(MaturityLevel::InteropTested < MaturityLevel::HardwareValidated);
    }

    #[test]
    fn dimension_labels_round_trip() {
        for d in ConformanceDimension::ALL {
            assert_eq!(ConformanceDimension::from_label(d.label()), Some(d));
        }
        assert_eq!(ConformanceDimension::from_label("bogus"), None);
    }

    #[test]
    fn absorb_merges_persisted_evidence_and_raises_the_level() {
        // The in-process battery derives UnitTested; a persisted Oracle (with a peer)
        // folded in via absorb raises the same element to InteropTested.
        let mut base = ConformanceReport::new();
        base.push(
            MaturityRecord::new("mp4mux")
                .with(Evidence::new(ConformanceDimension::Instantiate))
                .with(Evidence::new(ConformanceDimension::RoundTrip)),
        );
        assert_eq!(base.record_mut("mp4mux").level(), MaturityLevel::UnitTested);

        let mut persisted = ConformanceReport::new();
        persisted.push(
            MaturityRecord::new("mp4mux").with(
                Evidence::new(ConformanceDimension::Oracle)
                    .peer("ffmpeg")
                    .codec("h264"),
            ),
        );
        base.absorb(persisted);
        assert_eq!(
            base.record_mut("mp4mux").level(),
            MaturityLevel::InteropTested
        );
        assert_eq!(
            base.records.len(),
            1,
            "merged into the existing element, not duplicated"
        );
    }

    #[test]
    fn absorb_deduplicates_identical_evidence() {
        let mut a = ConformanceReport::new();
        a.push(MaturityRecord::new("x").with(Evidence::new(ConformanceDimension::RoundTrip)));
        let mut b = ConformanceReport::new();
        b.push(MaturityRecord::new("x").with(Evidence::new(ConformanceDimension::RoundTrip)));
        a.absorb(b);
        assert_eq!(
            a.record_mut("x").evidence.len(),
            1,
            "identical evidence is not doubled"
        );
    }

    #[test]
    fn report_table_lists_rows_and_min_level() {
        let mut report = ConformanceReport::new();
        report.push(
            MaturityRecord::new("st2110video")
                .with(Evidence::new(ConformanceDimension::Instantiate))
                .with(Evidence::new(ConformanceDimension::RoundTrip)),
        );
        report.push(MaturityRecord::new("unchecked"));
        let table = report.to_table();
        assert!(table.contains("st2110video"), "row present:\n{table}");
        assert!(
            table.contains("unit-tested"),
            "derived level shown:\n{table}"
        );
        assert!(table.contains("round-trip"), "dimension shown:\n{table}");
        assert_eq!(
            report.min_level(),
            MaturityLevel::Unverified,
            "the unchecked element drags the min down"
        );
    }
}
