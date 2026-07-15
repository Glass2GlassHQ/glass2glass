//! ST 2110-10 media clock: the mapping between a PTP/TAI time and an RTP
//! timestamp, the piece that ties RTP media transport to the pipeline's PTP
//! clock (M595).
//!
//! In SMPTE ST 2110 every essence stream carries an RTP timestamp derived from a
//! common media clock counting at a fixed rate from the PTP epoch: 90 kHz for
//! video, the sample rate for audio. Two receivers locked to the same
//! grandmaster therefore compute the *same* RTP timestamp for the same sampling
//! instant, which is what lets a video and an audio stream (or two cameras)
//! present frame-accurately together. This module is that conversion: PTP ns to a
//! 32-bit (wrapping) RTP timestamp and back. Pure `no_std` integer math, so it
//! runs on an embedded 2110 endpoint as well as a host.
//!
//! The reverse ([`tai_from_rtp`](MediaClock::tai_from_rtp)) resolves the 32-bit
//! wrap against a reference time (roughly "now" on the PTP clock), the standard
//! extended-timestamp technique, so a receiver recovers the full sampling time of
//! a packet whose timestamp has wrapped.

use crate::time::{RtpTs, TaiNs};

/// A media clock running at a fixed rate, converting PTP/TAI nanoseconds to and
/// from RTP timestamps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediaClock {
    rate_hz: u32,
}

impl MediaClock {
    /// The RTP clock rate for ST 2110-20 video (and the common 90 kHz video RTP
    /// clock generally).
    pub const VIDEO_RATE_HZ: u32 = 90_000;

    /// A media clock at `rate_hz` ticks per second (clamped to at least 1 to keep
    /// the arithmetic well-defined).
    pub fn new(rate_hz: u32) -> Self {
        Self { rate_hz: rate_hz.max(1) }
    }

    /// The 90 kHz video media clock (ST 2110-20).
    pub fn video() -> Self {
        Self::new(Self::VIDEO_RATE_HZ)
    }

    /// An audio media clock at the PCM sample rate (ST 2110-30), e.g. 48000.
    pub fn audio(sample_rate_hz: u32) -> Self {
        Self::new(sample_rate_hz)
    }

    /// The clock rate in Hz.
    pub fn rate_hz(&self) -> u32 {
        self.rate_hz
    }

    /// Full 64-bit media-clock tick count at a PTP/TAI time.
    pub fn ticks(&self, tai: TaiNs) -> u64 {
        // u128 so the ns*rate product (up to ~1.7e18 * 9e4) does not overflow.
        (u128::from(tai.get()) * u128::from(self.rate_hz) / 1_000_000_000) as u64
    }

    /// The 32-bit (wrapping) RTP timestamp for a PTP/TAI time, as it goes on the
    /// wire.
    pub fn rtp_timestamp(&self, tai: TaiNs) -> RtpTs {
        RtpTs(self.ticks(tai) as u32)
    }

    /// Nanoseconds spanned by `ticks` media-clock ticks (e.g. a packet's sample
    /// count for audio, a frame's duration for video). A duration, not an absolute
    /// time, so it stays a plain `u64`.
    pub fn ticks_to_ns(&self, ticks: u64) -> u64 {
        (u128::from(ticks) * 1_000_000_000 / u128::from(self.rate_hz)) as u64
    }

    /// Recover the full PTP/TAI time of a 32-bit RTP timestamp, resolving the wrap
    /// against `reference` (roughly the current PTP time): the result is the tick
    /// nearest the reference whose low 32 bits are `rtp`.
    pub fn tai_from_rtp(&self, rtp: RtpTs, reference: TaiNs) -> TaiNs {
        let ref_ticks = self.ticks(reference);
        // Nearest signed tick offset from the reference's low 32 bits: the u32
        // wrap-around difference reinterpreted as i32 picks the closest window.
        let diff = i64::from(rtp.get().wrapping_sub(ref_ticks as u32) as i32);
        let ticks = (ref_ticks as i64).saturating_add(diff).max(0) as u64;
        TaiNs(self.ticks_to_ns(ticks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A realistic TAI-scale instant (~2024 in ns since the PTP epoch).
    const TAI: u64 = 1_700_000_000_123_456_789;

    #[test]
    fn video_and_audio_rates() {
        assert_eq!(MediaClock::video().rate_hz(), 90_000);
        assert_eq!(MediaClock::audio(48_000).rate_hz(), 48_000);
        // A zero rate is clamped, never a divide-by-zero.
        assert_eq!(MediaClock::new(0).rate_hz(), 1);
    }

    #[test]
    fn rtp_timestamp_is_the_low_32_bits_of_the_tick_count() {
        let c = MediaClock::video();
        let ticks = c.ticks(TaiNs(TAI));
        assert_eq!(c.rtp_timestamp(TaiNs(TAI)).get(), ticks as u32);
        // 90 kHz over ~1.7e9 s is far past 2^32, so the wire timestamp has wrapped.
        assert!(ticks > u64::from(u32::MAX), "the full tick count exceeds 32 bits");
    }

    #[test]
    fn round_trips_within_one_tick() {
        // For both clocks, recovering the TAI time of a fresh timestamp (reference
        // near the same instant) is exact to within a tick period.
        for c in [MediaClock::video(), MediaClock::audio(48_000)] {
            let rtp = c.rtp_timestamp(TaiNs(TAI));
            let recovered = c.tai_from_rtp(rtp, TaiNs(TAI + 3_000_000)); // reference 3 ms later
            let tick_ns = c.ticks_to_ns(1);
            let diff = TAI.abs_diff(recovered.get());
            assert!(diff <= tick_ns + 1, "off by {diff} ns (> tick {tick_ns} ns) at {}Hz", c.rate_hz());
        }
    }

    #[test]
    fn resolves_the_32_bit_wrap_across_the_boundary() {
        let c = MediaClock::audio(48_000);
        // Pick an instant whose tick count sits just past a 2^32 boundary, and a
        // reference on the other side of it: the extended timestamp must still
        // recover the right window rather than jumping a whole wrap.
        let boundary_ticks: u64 = 0x1_0000_0000 + 100;
        let tai = c.ticks_to_ns(boundary_ticks);
        let rtp = c.rtp_timestamp(TaiNs(tai)); // low 32 bits ~ 100 (just past the wrap)
        assert!(rtp.get() < 1_000, "timestamp sits just past the 2^32 wrap, got {}", rtp.get());
        // Reference 50 ms earlier, whose ticks are just below the boundary.
        let reference = tai - 50_000_000;
        let recovered = c.tai_from_rtp(rtp, TaiNs(reference)).get();
        assert!(tai.abs_diff(recovered) <= c.ticks_to_ns(1) + 1, "wrap not resolved: {recovered} vs {tai}");
    }

    #[test]
    fn two_receivers_agree_on_the_timestamp() {
        // The whole point: the same sampling instant on the same clock rate maps
        // to the same wire timestamp regardless of who computes it.
        let a = MediaClock::video();
        let b = MediaClock::new(90_000);
        assert_eq!(a.rtp_timestamp(TaiNs(TAI)), b.rtp_timestamp(TaiNs(TAI)));
    }
}
