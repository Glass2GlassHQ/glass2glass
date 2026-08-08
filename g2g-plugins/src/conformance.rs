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

/// An Annex-B access unit: a 4-byte start code, `header` as the NAL header byte,
/// then `body` bytes of a repeating pattern (so a truncated reassembly shows up).
fn annexb_nal(header: u8, body: usize) -> Vec<u8> {
    let mut au = alloc::vec![0, 0, 0, 1, header];
    au.extend((0..body).map(|i| (i * 31 + 7) as u8));
    au
}

/// Conformance of the H.264 RTP payload core (RFC 6184): the `rtppay`
/// packetizer and the `rtpdepay` depayloader that `udpsink` / `udpsrc`, the RTSP
/// server, and the WebRTC path all share.
///
/// Verifies it constructs (`Instantiate`), round-trips an access unit whose slice
/// NAL exceeds the MTU through FU-A fragmentation and reassembly byte-exact
/// (`RoundTrip`), and that a dropped fragment costs only its own access unit: the
/// sequence gap discards the damaged one instead of welding it to the next, which
/// still depayloads intact (`LossResilience`). No `Oracle` evidence: the
/// ffmpeg-peer check for this path is `udpsrc`'s, not this core's.
pub fn rtp_h264() -> MaturityRecord {
    use crate::rtpdepay::RtpH264Depayloader;
    use crate::rtppay::RtpH264Packetizer;

    let mut rec = MaturityRecord::new("rtph264");
    rec.add(Evidence::new(D::Instantiate));

    // A parameter-set NAL plus a slice far past the 200-byte MTU, so the access
    // unit spans a single-NAL packet and a run of FU-A fragments.
    let mut au = annexb_nal(0x67, 8);
    au.extend(annexb_nal(0x65, 900));
    let mut tx = RtpH264Packetizer::new(96, 0x5EED).with_max_payload(200);
    let packets = tx.packetize(&au, 9000);
    let mut rx = RtpH264Depayloader::new();
    let mut out = None;
    for p in &packets {
        if let Some(unit) = rx.depacketize(p) {
            out = Some(unit);
        }
    }
    if out.as_ref().map(|u| u.data.as_slice()) == Some(au.as_slice()) {
        rec.add(
            Evidence::new(D::RoundTrip)
                .codec("h264")
                .detail("FU-A fragmented access unit reassembled byte-exact"),
        );
    }

    if survives_a_dropped_fragment(&au) {
        rec.add(
            Evidence::new(D::LossResilience)
                .codec("h264")
                .detail("dropped FU-A fragment discards only its own access unit"),
        );
    }

    rec
}

/// Payloadize `au` twice (two access units on one sequence run), drop a fragment
/// of the first, and report whether exactly the second comes out intact.
fn survives_a_dropped_fragment(au: &[u8]) -> bool {
    use crate::rtpdepay::RtpH264Depayloader;
    use crate::rtppay::RtpH264Packetizer;

    let mut tx = RtpH264Packetizer::new(96, 0x5EED).with_max_payload(200);
    let first = tx.packetize(au, 9000);
    let second = tx.packetize(au, 12000);
    if first.len() < 4 {
        return false; // no fragment run to damage
    }
    let mut rx = RtpH264Depayloader::new();
    let mut units = Vec::new();
    for (i, p) in first.iter().enumerate() {
        if i == 2 {
            continue; // the lost fragment
        }
        if let Some(u) = rx.depacketize(p) {
            units.push(u.data);
        }
    }
    for p in &second {
        if let Some(u) = rx.depacketize(p) {
            units.push(u.data);
        }
    }
    units.len() == 1 && units[0] == au
}

/// Conformance of the RTP jitter buffer (`rtpjitter`), the receive-side reorder
/// stage shared by `udpsrc`, `rtspsrc`, and the RTSP server.
///
/// Verifies it constructs (`Instantiate`) and that a wire which reorders packets
/// and loses one still yields intact access units (`LossResilience`): the
/// reordered packets are released in sequence order, the hole is reported for NACK
/// and then declared lost once it is overdue rather than stalling the stream.
pub fn rtp_jitter() -> MaturityRecord {
    use crate::rtpdepay::RtpH264Depayloader;
    use crate::rtpjitter::{JitterConfig, RtpJitterBuffer};
    use crate::rtppay::RtpH264Packetizer;

    let mut rec = MaturityRecord::new("rtpjitter");
    let config = JitterConfig::new(50, 64);
    rec.add(Evidence::new(D::Instantiate));

    let mut au = annexb_nal(0x67, 8);
    au.extend(annexb_nal(0x65, 900));
    let mut tx = RtpH264Packetizer::new(96, 0x1EAF).with_max_payload(200);
    let mut wire = tx.packetize(&au, 9000);
    let first_len = wire.len();
    wire.extend(tx.packetize(&au, 12000));
    if first_len < 4 {
        return rec;
    }

    // Arrival order: swap two adjacent pairs (reorder) and drop one packet of the
    // first access unit (loss).
    let lost_seq = 1u16;
    let mut jb = RtpJitterBuffer::new(config);
    let mut arrival: Vec<&Vec<u8>> = wire.iter().collect();
    arrival.swap(2, 3);
    arrival.swap(first_len, first_len + 1);
    for (i, p) in arrival.into_iter().enumerate() {
        if u16::from_be_bytes([p[2], p[3]]) == lost_seq {
            continue;
        }
        jb.push(p, i as u64 * 1_000_000);
    }
    let hole_seen = jb.missing_seqs().contains(&lost_seq);

    // Drain past the hold bound: the missing head is declared lost and the rest
    // releases in sequence order.
    let mut rx = RtpH264Depayloader::new();
    let mut units = Vec::new();
    let now = 10 * config.max_hold_ns;
    while let Some(p) = jb.pop(now) {
        if let Some(u) = rx.depacketize(&p) {
            units.push(u.data);
        }
    }
    let stats = jb.stats();
    if hole_seen && stats.reordered > 0 && stats.lost > 0 && units == alloc::vec![au] {
        rec.add(
            Evidence::new(D::LossResilience)
                .codec("h264")
                .detail("reordered arrival released in sequence order, lost packet skipped"),
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
    report.push(rtp_h264());
    report.push(rtp_jitter());
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

    /// The platform tag for a `Hardware` evidence row: `$G2G_CONFORMANCE_PLATFORM`
    /// when the runner names itself, else the device name `name_device` finds,
    /// else the host os / arch (which names no device, so it cannot pass for a
    /// device claim).
    #[cfg(all(
        target_os = "linux",
        any(feature = "cuda", feature = "v4l2", feature = "libcamera")
    ))]
    fn platform_tag(name_device: impl FnOnce() -> Option<String>) -> String {
        if let Some(name) = std::env::var_os("G2G_CONFORMANCE_PLATFORM") {
            return name.to_string_lossy().into_owned();
        }
        name_device()
            .unwrap_or_else(|| format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH))
    }

    /// The platform tag a CUDA-bound test should put on its `Hardware` evidence:
    /// the name the driver gives CUDA device 0 (the device the CUDA elements
    /// bind). Named per family rather than "the first GPU": a box with an
    /// integrated and a discrete GPU would otherwise tag a CUDA run with
    /// whichever adapter enumerated first.
    ///
    /// A test that already holds its own device name (Vulkan Video, a wgpu
    /// adapter it opened itself) passes that instead.
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub fn cuda_platform_tag() -> String {
        use g2g_core::runtime::DeviceProvider;
        platform_tag(|| {
            crate::gpudevice::GpuDeviceProvider::new()
                .probe()
                .ok()?
                .into_iter()
                .find(|d| d.persistent_id.starts_with("cuda:0:"))
                .map(|d| d.display_name)
        })
    }

    /// The platform tag a V4L2 capture test should put on its `Hardware`
    /// evidence: the card name the driver reports for the node it captured from
    /// (the camera model), which says which camera the evidence came from where
    /// `/dev/videoN` alone would not survive a replug.
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    pub fn v4l2_platform_tag(device: &str) -> String {
        use g2g_core::runtime::DeviceProvider;
        platform_tag(|| {
            crate::v4l2device::V4l2DeviceProvider::new()
                .probe()
                .ok()?
                .into_iter()
                .find(|d| d.props.iter().any(|(k, v)| k == "device" && v == device))
                .map(|d| d.display_name)
        })
    }

    /// The platform tag a libcamera capture test should put on its `Hardware`
    /// evidence: the id libcamera gives the camera at `index`, which identifies
    /// the physical device (bus path plus USB vendor / product).
    #[cfg(all(target_os = "linux", feature = "libcamera"))]
    pub fn libcamera_platform_tag(index: usize) -> String {
        platform_tag(|| {
            let manager = libcamera::camera_manager::CameraManager::new().ok()?;
            let cameras = manager.cameras();
            cameras.get(index).map(|camera| camera.id().to_string())
        })
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
    fn rtp_h264_battery_round_trips_and_survives_loss() {
        let rec = rtp_h264();
        assert!(rec.has(D::RoundTrip), "FU-A reassembly is byte-exact");
        assert!(
            rec.has(D::LossResilience),
            "a dropped fragment costs only its own access unit"
        );
        assert_eq!(rec.level(), MaturityLevel::UnitTested);
        assert!(!rec.has(D::Oracle), "no peer validated this core");
    }

    #[test]
    fn rtp_jitter_battery_reorders_and_skips_a_hole() {
        let rec = rtp_jitter();
        assert!(
            rec.has(D::LossResilience),
            "reordered arrival reconstructs, the hole is skipped not stalled"
        );
        assert_eq!(rec.level(), MaturityLevel::UnitTested);
    }

    #[test]
    fn report_renders_every_battery() {
        let report = report();
        let table = report.to_table();
        assert!(table.contains("st2110video"), "video row:\n{table}");
        assert!(table.contains("st2110audio"), "audio row:\n{table}");
        assert!(table.contains("rtph264"), "rtp payload row:\n{table}");
        assert!(table.contains("rtpjitter"), "jitter row:\n{table}");
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
