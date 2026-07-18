//! Conformance batteries (M614): the cases that produce the [`MaturityRecord`]s the
//! maturity report is derived from.
//!
//! Each battery here exercises a *real* element (never a mock) with cheap,
//! self-contained checks and adds a piece of [`Evidence`] only when a check actually
//! passes. So the maturity a battery reports is computed from behavior observed in
//! this process, not asserted by hand: a regression that breaks a round-trip drops
//! the derived level, and the honest ceiling of a loopback-only check is
//! `UnitTested` (no external-peer `Oracle` evidence is emitted, so the ST 2110 cores
//! surface as "unit-tested, interop pending" rather than claiming more).
//!
//! Only in-process, dependency-free checks run here, so `g2g-inspect --maturity` can
//! run the whole battery live. `Oracle` (ffmpeg / reference-gear) and `Hardware`
//! (GPU / device) evidence is produced by the feature-gated / host-gated integration
//! tests that own those resources, not by this always-on battery.

use alloc::vec::Vec;

use g2g_core::conformance::{
    ConformanceDimension as D, ConformanceReport, Evidence, MaturityRecord,
};
use g2g_core::RawVideoFormat;

use crate::st2110dup::SeamlessDedup;
use crate::st2110video::{Sampling, St2110VideoDepacketizer, St2110VideoPacketizer};

/// Conformance of the ST 2110-20 (RFC 4175) video packetizer / depacketizer core.
///
/// Verifies it constructs (`Instantiate`), round-trips a frame byte-exact through
/// packetize -> depacketize (`RoundTrip`), and reconstructs a frame under ST 2110-7
/// redundant-path loss via the sequence-number merge (`LossResilience`). It emits no
/// `Oracle` evidence: it has not been validated against reference -20 gear, so it
/// tops out at `UnitTested`.
pub fn st2110_video() -> MaturityRecord {
    let mut rec = MaturityRecord::new("st2110video");
    let (w, h) = (8usize, 8usize);

    // Instantiate: the packetizer / depacketizer construct for a real sampling.
    if St2110VideoDepacketizer::new(RawVideoFormat::Rgba8, w, h).is_some() {
        rec.add(Evidence::new(D::Instantiate));
    }

    // RoundTrip: an RGBA frame survives packetize -> depacketize byte-exact.
    let frame: Vec<u8> = (0..w * 4 * h).map(|i| (i * 7 + 1) as u8).collect();
    let mut tx = St2110VideoPacketizer::new(96, 0xABCD, Sampling::Rgba8, 60);
    if let Some(packets) = tx.packetize(&frame, w, h, 1_000_000_000) {
        if let Some(mut rx) = St2110VideoDepacketizer::new(RawVideoFormat::Rgba8, w, h) {
            let mut out = None;
            for p in &packets {
                if let Some(f) = rx.depacketize(p) {
                    out = Some(f.bytes);
                }
            }
            if out.as_deref() == Some(frame.as_slice()) {
                rec.add(
                    Evidence::new(D::RoundTrip)
                        .codec("rgba8")
                        .detail("packetize/depacketize loopback"),
                );
            }
        }
    }

    // LossResilience: the same frame reconstructs when each of two redundant paths
    // drops a different subset of packets (never the marker, none lost on both),
    // merged by the -7 SeamlessDedup. This is the M610 receive path without sockets.
    if reconstructs_through_redundant_loss(&frame, w, h) {
        rec.add(
            Evidence::new(D::LossResilience)
                .detail("ST 2110-7 seamless merge through per-path drops"),
        );
    }

    rec
}

/// Reconstruct a frame through two lossy redundant paths merged by [`SeamlessDedup`],
/// returning whether the result is byte-exact.
fn reconstructs_through_redundant_loss(frame: &[u8], w: usize, h: usize) -> bool {
    let mut tx = St2110VideoPacketizer::new(96, 0xBEEF, Sampling::Rgba8, 60);
    let Some(packets) = tx.packetize(frame, w, h, 1_000_000_000) else {
        return false;
    };
    if packets.len() < 6 {
        return false; // too few packets to model a meaningful loss split
    }
    let last = packets.len() - 1;
    let mut dedup = SeamlessDedup::new();
    let Some(mut rx) = St2110VideoDepacketizer::new(RawVideoFormat::Rgba8, w, h) else {
        return false;
    };
    let mut done = None;
    for (i, p) in packets.iter().enumerate() {
        // Path A drops packet 2, path B drops 3 and 5; the marker (last) is on both.
        let on_a = i == last || i != 2;
        let on_b = i == last || (i != 3 && i != 5);
        for present in [on_a, on_b] {
            if present && dedup.accept(p) {
                if let Some(f) = rx.depacketize(p) {
                    done = Some(f.bytes);
                }
            }
        }
    }
    done.as_deref() == Some(frame)
}

/// Conformance of the ST 2110-30 (AES67) PCM audio packetizer / depacketizer core.
///
/// Verifies it constructs (`Instantiate`) and round-trips interleaved PCM byte-exact
/// through packetize -> depacketize (`RoundTrip`). Like the video core it emits no
/// `Oracle` evidence, so it tops out at `UnitTested`.
pub fn st2110_audio() -> MaturityRecord {
    use crate::st2110audio::{SampleDepth, St2110AudioDepacketizer, St2110AudioPacketizer};

    let mut rec = MaturityRecord::new("st2110audio");
    rec.add(Evidence::new(D::Instantiate));

    // Stereo L16 samples across the signed range so the round-trip is exact.
    let samples: Vec<i32> = (0..96i32).map(|i| ((i * 331) % 30_000) - 15_000).collect();
    let mut tx = St2110AudioPacketizer::new(96, 0x1234, 48_000, 2, SampleDepth::L16, 48);
    let packets = tx.packetize(&samples, 0);
    let rx = St2110AudioDepacketizer::new(2, SampleDepth::L16);
    let mut got: Vec<i32> = Vec::new();
    for p in &packets {
        if let Some(pkt) = rx.depacketize(p) {
            got.extend(pkt.samples);
        }
    }
    if got == samples {
        rec.add(
            Evidence::new(D::RoundTrip)
                .codec("l16")
                .detail("packetize/depacketize loopback"),
        );
    }

    rec
}

/// The in-process conformance report: run every always-on battery and collect its
/// derived [`MaturityRecord`]. These are the checks that run anywhere (no ffmpeg, no
/// GPU), so they top out at `UnitTested`.
pub fn report() -> ConformanceReport {
    let mut report = ConformanceReport::new();
    report.push(st2110_video());
    report.push(st2110_audio());
    report
}

/// Persisted conformance evidence (M615): the `Oracle` (reference-implementation) and
/// `Hardware` (device) checks that cannot run in-process are produced by the
/// integration tests that own those resources (ffmpeg, a GPU), which append their
/// evidence to a shared log. [`full_report`] folds that log into the in-process
/// [`report`] so `g2g-inspect --maturity` shows the `InteropTested` /
/// `HardwareValidated` rows those tests earned, without inspect itself needing the
/// resources.
///
/// The log is a simple tab-separated append-only file (one evidence line each), so
/// concurrent tests can append without coordination and it stays greppable. Its path
/// is `$G2G_CONFORMANCE_LOG`, or a default under the temp dir.
#[cfg(feature = "std")]
pub mod persist {
    use super::*;
    use alloc::format;
    use alloc::string::{String, ToString};
    use g2g_core::conformance::ConformanceDimension;
    use std::io::Write;
    use std::path::PathBuf;

    /// The evidence log path: `$G2G_CONFORMANCE_LOG` or `<tempdir>/g2g-conformance.tsv`.
    pub fn evidence_log_path() -> PathBuf {
        match std::env::var_os("G2G_CONFORMANCE_LOG") {
            Some(p) => PathBuf::from(p),
            None => std::env::temp_dir().join("g2g-conformance.tsv"),
        }
    }

    /// A field for the TSV line: `-` for absent, tabs / newlines flattened to spaces.
    fn field(v: Option<&str>) -> String {
        match v {
            None => "-".into(),
            Some(s) => s.replace(['\t', '\n', '\r'], " "),
        }
    }

    /// Parse a `-`/value TSV field back to an `Option`.
    fn unfield(s: &str) -> Option<String> {
        if s == "-" {
            None
        } else {
            Some(s.to_string())
        }
    }

    /// Append one piece of evidence for `element` to the log (creating it if needed).
    /// Called by a resource-owning conformance test when a check passes.
    pub fn record_evidence(element: &str, ev: &Evidence) -> std::io::Result<()> {
        let line = format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            field(Some(element)),
            ev.dimension.label(),
            field(ev.platform.as_deref()),
            field(ev.codec.as_deref()),
            field(ev.peer.as_deref()),
            field(ev.detail.as_deref()),
        );
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(evidence_log_path())?;
        f.write_all(line.as_bytes())
    }

    /// Load the persisted evidence into a report (empty if the log is absent). A line
    /// with an unknown dimension or too few fields is skipped, never trusted.
    pub fn load_persisted() -> ConformanceReport {
        let mut report = ConformanceReport::new();
        let Ok(text) = std::fs::read_to_string(evidence_log_path()) else {
            return report;
        };
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 6 {
                continue;
            }
            let Some(dimension) = ConformanceDimension::from_label(f[1]) else {
                continue;
            };
            let mut ev = Evidence::new(dimension);
            ev.platform = unfield(f[2]);
            ev.codec = unfield(f[3]);
            ev.peer = unfield(f[4]);
            ev.detail = unfield(f[5]);
            report.record_mut(f[0]).add(ev);
        }
        report
    }

    /// The in-process [`report`](super::report) with the persisted `Oracle` /
    /// `Hardware` evidence folded in. This is what `g2g-inspect --maturity` renders.
    pub fn full_report() -> ConformanceReport {
        let mut report = super::report();
        report.absorb(load_persisted());
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::conformance::MaturityLevel;

    #[test]
    fn video_battery_is_unit_tested_not_interop() {
        let rec = st2110_video();
        assert!(rec.has(D::Instantiate));
        assert!(rec.has(D::RoundTrip), "loopback round-trip passed");
        assert!(rec.has(D::LossResilience), "-7 reconstruction passed");
        // The honesty contract: loopback validation is NOT interop validation.
        assert!(!rec.has(D::Oracle), "no reference-gear oracle evidence");
        assert_eq!(rec.level(), MaturityLevel::UnitTested);
    }

    #[test]
    fn audio_battery_round_trips() {
        let rec = st2110_audio();
        assert!(rec.has(D::RoundTrip), "PCM loopback passed");
        assert_eq!(rec.level(), MaturityLevel::UnitTested);
        assert!(!rec.has(D::Oracle));
    }

    #[test]
    fn report_renders_every_battery() {
        let report = report();
        let table = report.to_table();
        assert!(table.contains("st2110video"), "video row:\n{table}");
        assert!(table.contains("st2110audio"), "audio row:\n{table}");
        assert!(
            table.contains("unit-tested"),
            "derived levels shown:\n{table}"
        );
        // Every element is at least unit-tested (nothing regressed to instantiated).
        assert_eq!(report.min_level(), MaturityLevel::UnitTested);
    }

    #[cfg(feature = "std")]
    #[test]
    fn persisted_evidence_round_trips_and_merges_into_full_report() {
        // Record an Oracle datapoint (no ffmpeg needed for the persistence path
        // itself), reload it, and confirm it derives InteropTested and folds into the
        // in-process batteries via full_report.
        use g2g_core::conformance::ConformanceDimension;
        let log = std::env::temp_dir().join("g2g-conformance-unit-roundtrip.tsv");
        std::env::set_var("G2G_CONFORMANCE_LOG", &log);
        let _ = std::fs::remove_file(&log);

        persist::record_evidence(
            "x264enc",
            &Evidence::new(ConformanceDimension::Oracle)
                .peer("ffmpeg")
                .codec("h264")
                .detail("decoded by ffmpeg"),
        )
        .unwrap();

        let loaded = persist::load_persisted();
        let rec = loaded
            .records
            .iter()
            .find(|r| r.element == "x264enc")
            .expect("persisted");
        assert_eq!(rec.level(), MaturityLevel::InteropTested);
        assert_eq!(rec.peers(), alloc::vec!["ffmpeg"]);
        assert_eq!(
            rec.evidence[0].detail.as_deref(),
            Some("decoded by ffmpeg"),
            "detail survives"
        );

        // full_report carries both the in-process batteries and the persisted row.
        let full = persist::full_report();
        assert!(
            full.records.iter().any(|r| r.element == "st2110video"),
            "battery present"
        );
        assert!(
            full.records.iter().any(|r| r.element == "x264enc"),
            "persisted present"
        );

        let _ = std::fs::remove_file(&log);
    }
}
