//! Boundary-scoped time newtypes (M618): distinct types at the clock / PTP / RTP
//! seam, where three different "just an integer" times meet and are easy to mix up.
//!
//! ST 2110 and PTP juggle several clocks: the pipeline's monotonic reference, PTP /
//! TAI absolute time, and the 32-bit wrapping RTP media-clock timestamp on the wire.
//! They are all integers, so nothing stops a `reference_ns` being passed where a TAI
//! time is wanted, or an RTP timestamp being treated as nanoseconds, exactly the
//! confusion the PTP servo work hit (a monotonic reference minus a TAI master is a
//! meaningless offset). These newtypes make the seam explicit: [`MediaClock`] takes a
//! [`TaiNs`] and returns an [`RtpTs`], so the compiler rejects handing it the wrong
//! clock. Deliberately narrow, this is the RTP / TAI boundary, not a `Frame.timing`
//! retrofit (PTS stays a plain `u64` ns in the pipeline's own timeline).
//!
//! [`MediaClock`]: crate::mediaclock::MediaClock

/// PTP / TAI time in nanoseconds since the PTP epoch (absolute wall time all
/// grandmaster-locked endpoints agree on). Distinct from the pipeline's monotonic
/// `pts_ns` (a relative timeline) and from a raw duration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TaiNs(pub u64);

impl TaiNs {
    /// The underlying nanosecond count.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// This instant advanced by `ns` nanoseconds (saturating), e.g. a base time plus
    /// a frame's PTS offset.
    pub fn saturating_add_ns(self, ns: u64) -> Self {
        Self(self.0.saturating_add(ns))
    }
}

impl From<u64> for TaiNs {
    fn from(ns: u64) -> Self {
        Self(ns)
    }
}

/// The pipeline's monotonic reference time in nanoseconds: a relative timeline with
/// an arbitrary epoch (whatever the platform clock started at), *not* absolute wall
/// time. Distinct from [`TaiNs`]: a PTP servo disciplines a `RefNs` reading to a
/// `TaiNs` master, and subtracting one from the other directly is the meaningless
/// offset the servo work hit. The type keeps the two apart at the servo seam
/// (`observe_master` / `sync_exchange` take a `RefNs` for `t2` / `t3` and a `TaiNs`
/// for `t1` / `t4`), so the master and reference roles can no longer be swapped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct RefNs(pub u64);

impl RefNs {
    /// The underlying nanosecond count.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for RefNs {
    fn from(ns: u64) -> Self {
        Self(ns)
    }
}

/// A 32-bit RTP media-clock timestamp as it appears on the wire (wraps at `2^32`).
/// Distinct from a nanosecond time: it counts media-clock ticks, not ns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct RtpTs(pub u32);

impl RtpTs {
    /// The underlying 32-bit tick count.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Big-endian octets, as the timestamp sits in an RTP header.
    pub fn to_be_bytes(self) -> [u8; 4] {
        self.0.to_be_bytes()
    }

    /// This timestamp advanced by `ticks` (wrapping), e.g. per-packet sample-count
    /// stepping within a frame.
    pub fn wrapping_add(self, ticks: u32) -> Self {
        Self(self.0.wrapping_add(ticks))
    }
}

impl From<u32> for RtpTs {
    fn from(ts: u32) -> Self {
        Self(ts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tai_ns_accessors() {
        let t = TaiNs::from(1_000);
        assert_eq!(t.get(), 1_000);
        assert_eq!(t.saturating_add_ns(500).get(), 1_500);
        assert_eq!(TaiNs::from(u64::MAX).saturating_add_ns(1), TaiNs(u64::MAX), "saturates");
    }

    #[test]
    fn ref_ns_accessors() {
        let r = RefNs::from(1_000_000_000);
        assert_eq!(r.get(), 1_000_000_000);
        assert_eq!(RefNs::default(), RefNs(0));
        // A reference and a TAI time are distinct types: they cannot be mixed up.
        assert_ne!(core::any::type_name::<RefNs>(), core::any::type_name::<TaiNs>());
    }

    #[test]
    fn rtp_ts_accessors() {
        let r = RtpTs::from(0xFFFF_FFFF);
        assert_eq!(r.get(), 0xFFFF_FFFF);
        assert_eq!(r.wrapping_add(2), RtpTs(1), "wraps at 2^32");
        assert_eq!(r.to_be_bytes(), [0xFF, 0xFF, 0xFF, 0xFF]);
    }
}
