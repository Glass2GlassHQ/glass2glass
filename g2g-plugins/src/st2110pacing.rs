//! ST 2110-21 sender pacing: the traffic-shaping schedule that spreads a
//! frame's RTP packets across the frame period instead of bursting them, so the
//! network sees a smooth flow a -21-conformant receiver can absorb.
//!
//! Uncompressed -20 video is multi-Gbps; a naive sender that blasts a whole frame's
//! packets back to back produces a burst that overruns switch buffers and a
//! receiver's virtual-receive buffer (VRX). ST 2110-21 defines traffic profiles that
//! bound how bunched the packets may be. This module is the sans-IO scheduling core:
//! given a frame's packet count and period it yields each packet's target emission
//! offset (a [`Pacer`]), and it can check whether a set of actual emission offsets
//! stays within a burst tolerance ([`Pacer::conforms`]). A sink realizes the
//! schedule with a `PacedSender` thread; the timing math lives here so it is
//! deterministic and CI-testable without real waits.
//!
//! Two profiles: [`PacingProfile::Linear`] spreads packets evenly across the whole
//! frame period (the -21 narrow-linear / "2110TPNL" shape, no burst at the frame
//! boundary), and [`PacingProfile::Gapped`] spreads them across only the active
//! portion of the period (`active_ratio`, e.g. 1080/1125 for HD), leaving the
//! vertical-blanking interval idle (the default -20 "gapped" shape). This is the
//! pacing model, not a full VRX validator with the per-format `Cmax` / `TROFF`
//! constants from the SMPTE tables.

/// How a frame's packets are spread over its period.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacingProfile {
    /// Even spacing across the whole frame period (narrow-linear).
    Linear,
    /// Even spacing across the active portion only (`active_ratio`), leaving the
    /// blanking gap idle (narrow-gapped, the -20 default).
    Gapped,
}

/// Default active-portion ratio for the gapped profile: 1080 active lines of 1125
/// total (HD, SMPTE 274M). A caller with a different raster can override it.
const GAPPED_ACTIVE_NUM: u64 = 1080;
const GAPPED_ACTIVE_DEN: u64 = 1125;

/// Computes per-packet emission offsets for a frame under a [`PacingProfile`].
#[derive(Clone, Copy, Debug)]
pub struct Pacer {
    /// Spacing between consecutive packets, nanoseconds.
    interval_ns: u64,
    packet_count: usize,
}

impl Pacer {
    /// A pacer for `packet_count` packets over `frame_period_ns`, using the profile's
    /// spreading rule. The active-portion span for [`PacingProfile::Gapped`] uses the
    /// HD default ratio; see [`Self::with_active_ratio`] to override.
    pub fn new(profile: PacingProfile, packet_count: usize, frame_period_ns: u64) -> Self {
        Self::with_active_ratio(
            profile,
            packet_count,
            frame_period_ns,
            GAPPED_ACTIVE_NUM,
            GAPPED_ACTIVE_DEN,
        )
    }

    /// As [`Self::new`], but with an explicit gapped active ratio `num/den` (ignored
    /// for [`PacingProfile::Linear`]).
    pub fn with_active_ratio(
        profile: PacingProfile,
        packet_count: usize,
        frame_period_ns: u64,
        active_num: u64,
        active_den: u64,
    ) -> Self {
        // Spread over N-1 gaps: the last packet lands at the end of the span, the
        // first at offset 0, so N packets get N-1 equal intervals.
        let gaps = (packet_count.max(1) as u64).saturating_sub(1).max(1);
        let span = match profile {
            PacingProfile::Linear => frame_period_ns,
            PacingProfile::Gapped => {
                let den = active_den.max(1);
                // Multiply before dividing to avoid truncating the ratio (periods are
                // ~1e7 ns, so the product stays well within u64).
                frame_period_ns.saturating_mul(active_num.min(den)) / den
            }
        };
        Self {
            interval_ns: span / gaps,
            packet_count,
        }
    }

    /// Spacing between consecutive packets in nanoseconds.
    pub fn interval_ns(&self) -> u64 {
        self.interval_ns
    }

    /// Target emission offset (from the frame's send start) for packet `i`.
    pub fn offset_ns(&self, i: usize) -> u64 {
        (i as u64).saturating_mul(self.interval_ns)
    }

    /// Number of packets this pacer was built for.
    pub fn packet_count(&self) -> usize {
        self.packet_count
    }

    /// Whether a run of actual emission offsets conforms to the schedule within
    /// `tolerance_ns`: each packet must be emitted no earlier than its scheduled
    /// offset minus the tolerance (packets may be late, but not bunched earlier than
    /// the profile allows), and offsets must be non-decreasing. `actual.len()` must
    /// match the pacer's packet count.
    pub fn conforms(&self, actual: &[u64], tolerance_ns: u64) -> bool {
        if actual.len() != self.packet_count {
            return false;
        }
        let mut prev = 0u64;
        for (i, &a) in actual.iter().enumerate() {
            if a < prev {
                return false; // out of order
            }
            let scheduled = self.offset_ns(i);
            if a + tolerance_ns < scheduled {
                return false; // too early (bunched ahead of the profile)
            }
            prev = a;
        }
        true
    }
}

/// The outcome of a VRX evaluation over a run of actual packet emission offsets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VrxReport {
    /// Peak virtual-receive-buffer occupancy reached (packets).
    pub peak_occupancy: u64,
    /// The buffer depth the sender was checked against (packets).
    pub cmax: u64,
    /// A packet arrived after its scheduled read deadline (the receiver would starve).
    pub underflowed: bool,
    /// Whether the run conforms: peak within `cmax` and no late packet.
    pub conforms: bool,
    /// Index of the first packet that violated `cmax` or arrived late, if any.
    pub first_violation: Option<usize>,
}

/// Full ST 2110-21 virtual-receive-buffer (VRX) model for sender compliance.
///
/// A conformant receiver reads one packet every `drain_interval_ns` (the ideal read
/// schedule, equal to the matching [`Pacer`]'s interval), starting after an initial
/// `tr_offset_ns` head start (`TR_OFFSET`, one read interval by default, so the first
/// packet is not required at exactly `t = 0`). Feeding a run of actual packet emission
/// offsets through [`Self::evaluate`] tracks the buffer occupancy `sent - read` over
/// time: it rises when the sender leads the schedule (a burst) and falls as the
/// receiver drains. The sender conforms to a traffic profile when the peak occupancy
/// stays within that profile's `Cmax` (buffer depth, packets) and no packet arrives
/// after its scheduled read deadline (which would starve the receiver).
///
/// Unlike [`Pacer::conforms`] (a burst-tolerance approximation on the emission offsets
/// alone), this is the actual leaky-bucket buffer model. `Cmax` and `TR_OFFSET` are
/// the per-format -21 compliance parameters: pass the values from the SMPTE ST 2110-21
/// tables for the raster / rate under test, or derive the drain interval from a
/// [`Pacer`] with [`Self::for_pacer`] and supply the profile's `Cmax`.
#[derive(Clone, Copy, Debug)]
pub struct VrxValidator {
    drain_interval_ns: u64,
    tr_offset_ns: u64,
    cmax: u64,
}

impl VrxValidator {
    /// A validator draining one packet every `drain_interval_ns` after a `tr_offset_ns`
    /// head start, allowing a peak occupancy of `cmax` packets.
    pub fn new(drain_interval_ns: u64, tr_offset_ns: u64, cmax: u64) -> Self {
        Self {
            drain_interval_ns,
            tr_offset_ns,
            cmax,
        }
    }

    /// A validator whose drain schedule matches `pacer` (so a sender emitting exactly on
    /// the pacer's schedule conforms), with a one-interval `TR_OFFSET` head start and a
    /// `cmax` buffer depth from the format's -21 profile.
    pub fn for_pacer(pacer: &Pacer, cmax: u64) -> Self {
        let drain = pacer.interval_ns();
        Self::new(drain, drain, cmax)
    }

    /// The ideal inter-read interval (`TRS`) this validator drains at.
    pub fn drain_interval_ns(&self) -> u64 {
        self.drain_interval_ns
    }

    /// Evaluate a frame's actual packet emission offsets (from the send start,
    /// non-decreasing). Returns the peak buffer occupancy, whether any packet was late,
    /// and whether the run conforms to `cmax`.
    pub fn evaluate(&self, actual_offsets: &[u64]) -> VrxReport {
        let mut peak = 0u64;
        let mut underflowed = false;
        let mut first_violation = None;
        for (i, &a) in actual_offsets.iter().enumerate() {
            let sent = i as u64 + 1;
            // Reads the receiver has issued by time `a` (none before the head start).
            let reads = if a < self.tr_offset_ns || self.drain_interval_ns == 0 {
                0
            } else {
                (a - self.tr_offset_ns) / self.drain_interval_ns + 1
            };
            // Occupancy: the receiver cannot read a packet that has not arrived, so
            // reads are capped at packets sent; the surplus is what the buffer holds.
            let occ = sent - reads.min(sent);
            if occ > peak {
                peak = occ;
            }
            // Late: packet `i` missed its read deadline (TR_OFFSET + i * TRS), so the
            // receiver would have starved waiting for it.
            let deadline = self
                .tr_offset_ns
                .saturating_add((i as u64).saturating_mul(self.drain_interval_ns));
            let late = a > deadline;
            underflowed |= late;
            if (occ > self.cmax || late) && first_violation.is_none() {
                first_violation = Some(i);
            }
        }
        let conforms = peak <= self.cmax && !underflowed;
        VrxReport {
            peak_occupancy: peak,
            cmax: self.cmax,
            underflowed,
            conforms,
            first_violation,
        }
    }
}

// The socket-bound realization of a schedule. Needs std threads, so it is behind
// the `st2110` feature, the schedule math above stays sans-IO in the no_std baseline.
#[cfg(feature = "st2110")]
mod send {
    use super::Pacer;
    use std::io;
    use std::net::UdpSocket;
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};
    use std::vec::Vec;
    use tokio::sync::oneshot;

    // a sleep overshoots by the kernel timer slack, 50 us by default on Linux
    const SPIN_MARGIN: Duration = Duration::from_micros(200);
    const PACER_THREAD_NAME: &str = "st2110-pacer";
    const PACER_THREAD_GONE: &str = "st2110 pacing thread exited";

    struct PacedFrame {
        packets: Vec<Vec<u8>>,
        pacer: Option<Pacer>,
        sent: oneshot::Sender<io::Result<()>>,
    }

    // the thread spins out each deadline, a tokio timer only fires every millisecond
    #[derive(Debug)]
    pub struct PacedSender {
        frames: mpsc::Sender<PacedFrame>,
    }

    impl PacedSender {
        pub fn spawn(socket: UdpSocket) -> io::Result<Self> {
            let (frames, queued) = mpsc::channel::<PacedFrame>();
            let mut writer = DatagramWriter::new(socket);
            thread::Builder::new()
                .name(PACER_THREAD_NAME.into())
                .spawn(move || {
                    for frame in queued {
                        let result = send_frame(&mut writer, &frame.packets, frame.pacer.as_ref());
                        // the sink stopped waiting, nobody is left to report to
                        let _ = frame.sent.send(result);
                    }
                })?;
            Ok(Self { frames })
        }

        pub async fn send(&self, packets: Vec<Vec<u8>>, pacer: Option<Pacer>) -> io::Result<()> {
            let (sent, done) = oneshot::channel();
            self.frames
                .send(PacedFrame {
                    packets,
                    pacer,
                    sent,
                })
                .map_err(|_| io::Error::other(PACER_THREAD_GONE))?;
            done.await
                .map_err(|_| io::Error::other(PACER_THREAD_GONE))?
        }
    }

    fn send_frame(
        writer: &mut DatagramWriter,
        packets: &[Vec<u8>],
        pacer: Option<&Pacer>,
    ) -> io::Result<()> {
        let Some(pacer) = pacer else {
            return writer.send_all(packets);
        };
        let start = Instant::now();
        let mut sent = 0;
        while sent < packets.len() {
            let now = wait_until(start + Duration::from_nanos(pacer.offset_ns(sent)));
            let elapsed_ns = u64::try_from((now - start).as_nanos()).unwrap_or(u64::MAX);
            // a slow send catches up by writing every packet already due at once
            let mut due = sent + 1;
            while due < packets.len() && pacer.offset_ns(due) <= elapsed_ns {
                due += 1;
            }
            writer.send_all(&packets[sent..due])?;
            sent = due;
        }
        Ok(())
    }

    fn wait_until(deadline: Instant) -> Instant {
        loop {
            let now = Instant::now();
            let remaining = deadline.saturating_duration_since(now);
            if remaining.is_zero() {
                return now;
            }
            if remaining > SPIN_MARGIN {
                thread::sleep(remaining - SPIN_MARGIN);
            } else {
                std::hint::spin_loop();
            }
        }
    }

    // writes each run of equal-size packets as one UDP GSO datagram train
    #[derive(Debug)]
    struct DatagramWriter {
        socket: UdpSocket,
        segmentation: bool,
        // UDP_SEGMENT currently set on the socket, 0 when off
        #[cfg(target_os = "linux")]
        segment_size: usize,
    }

    impl DatagramWriter {
        fn new(socket: UdpSocket) -> Self {
            Self {
                socket,
                segmentation: cfg!(target_os = "linux"),
                #[cfg(target_os = "linux")]
                segment_size: 0,
            }
        }

        fn send_all(&mut self, packets: &[Vec<u8>]) -> io::Result<()> {
            let mut rest = packets;
            while !rest.is_empty() {
                let run = if self.segmentation {
                    segment_run(rest)
                } else {
                    1
                };
                self.send_run(&rest[..run])?;
                rest = &rest[run..];
            }
            Ok(())
        }

        #[cfg(not(target_os = "linux"))]
        fn send_run(&mut self, run: &[Vec<u8>]) -> io::Result<()> {
            self.send_each(run)
        }

        #[cfg(target_os = "linux")]
        fn send_run(&mut self, run: &[Vec<u8>]) -> io::Result<()> {
            if !self.segmentation {
                return self.send_each(run);
            }
            match self.send_segmented(run) {
                Err(error) if segmentation_unsupported(&error) => {
                    self.set_segment_size(0)?;
                    self.segmentation = false;
                    self.send_each(run)
                }
                result => result,
            }
        }

        fn send_each(&self, run: &[Vec<u8>]) -> io::Result<()> {
            run.iter()
                .try_for_each(|packet| self.socket.send(packet).map(drop))
        }

        #[cfg(target_os = "linux")]
        fn send_segmented(&mut self, run: &[Vec<u8>]) -> io::Result<()> {
            use std::os::fd::AsRawFd;
            self.set_segment_size(run[0].len())?;
            let mut iovecs = [libc::iovec {
                iov_base: core::ptr::null_mut(),
                iov_len: 0,
            }; MAX_SEGMENTS];
            for (iovec, packet) in iovecs.iter_mut().zip(run) {
                iovec.iov_base = packet.as_ptr().cast_mut().cast();
                iovec.iov_len = packet.len();
            }
            let count = libc::c_int::try_from(run.len()).map_err(io::Error::other)?;
            // SAFETY: the first `count` iovecs point into packets borrowed for this call, and writev only reads them
            let written = unsafe { libc::writev(self.socket.as_raw_fd(), iovecs.as_ptr(), count) };
            if written < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }

        #[cfg(target_os = "linux")]
        fn set_segment_size(&mut self, size: usize) -> io::Result<()> {
            use std::os::fd::AsRawFd;
            if size == self.segment_size {
                return Ok(());
            }
            let value = libc::c_int::try_from(size).map_err(io::Error::other)?;
            // SAFETY: the option value is a live c_int and the length passed is its size
            let result = unsafe {
                libc::setsockopt(
                    self.socket.as_raw_fd(),
                    libc::SOL_UDP,
                    libc::UDP_SEGMENT,
                    (&value as *const libc::c_int).cast(),
                    core::mem::size_of::<libc::c_int>() as libc::socklen_t,
                )
            };
            if result != 0 {
                return Err(io::Error::last_os_error());
            }
            self.segment_size = size;
            Ok(())
        }
    }

    // largest UDP payload of one IPv4 datagram
    const MAX_UDP_PAYLOAD: usize = 65_507;
    // UDP_MAX_SEGMENTS, the kernel's cap on segments per GSO write
    const MAX_SEGMENTS: usize = 64;

    // leading packets of the first one's size, plus one shorter packet as the last segment
    fn segment_run(packets: &[Vec<u8>]) -> usize {
        let size = packets[0].len();
        if size == 0 {
            return 1;
        }
        let limit = (MAX_UDP_PAYLOAD / size)
            .clamp(1, MAX_SEGMENTS)
            .min(packets.len());
        let full = packets[..limit]
            .iter()
            .take_while(|packet| packet.len() == size)
            .count();
        let shorter_tail = full < limit && packets[full].len() < size;
        full + usize::from(shorter_tail)
    }

    // ENOPROTOOPT before Linux 4.18, EIO without checksum offload, EINVAL past the MTU
    #[cfg(target_os = "linux")]
    fn segmentation_unsupported(error: &io::Error) -> bool {
        matches!(
            error.raw_os_error(),
            Some(libc::ENOPROTOOPT | libc::EIO | libc::EINVAL)
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::vec;

        const SEGMENT: usize = 1460;

        #[test]
        fn segment_run_takes_equal_packets_and_one_shorter_tail() {
            let packets = vec![
                vec![0; SEGMENT],
                vec![0; SEGMENT],
                vec![0; 620],
                vec![0; SEGMENT],
            ];
            assert_eq!(segment_run(&packets), 3);
        }

        #[test]
        fn segment_run_stops_before_a_longer_packet() {
            let packets = vec![vec![0; 1000], vec![0; SEGMENT]];
            assert_eq!(segment_run(&packets), 1);
        }

        #[test]
        fn segment_run_fits_one_udp_datagram() {
            let packets = vec![vec![0; SEGMENT]; MAX_SEGMENTS];
            assert_eq!(segment_run(&packets), MAX_UDP_PAYLOAD / SEGMENT);
        }
    }
}

#[cfg(feature = "st2110")]
pub use send::PacedSender;

#[cfg(test)]
mod tests {
    use super::*;

    const PERIOD_50_FPS_NS: u64 = 20_000_000;

    #[test]
    fn linear_spreads_evenly_across_the_period() {
        // 5 packets over a 20 ms (50 fps) period: 4 gaps of 5 ms; last lands at 20 ms.
        let period = PERIOD_50_FPS_NS;
        let p = Pacer::new(PacingProfile::Linear, 5, period);
        assert_eq!(p.interval_ns(), 5_000_000);
        assert_eq!(p.offset_ns(0), 0);
        assert_eq!(p.offset_ns(4), 20_000_000);
    }

    #[test]
    fn gapped_leaves_the_blanking_interval_idle() {
        // Gapped packs into the active 1080/1125 of the period, so the interval is
        // shorter than linear (the same packets finish before the frame boundary).
        let period = PERIOD_50_FPS_NS;
        let lin = Pacer::new(PacingProfile::Linear, 100, period);
        let gap = Pacer::new(PacingProfile::Gapped, 100, period);
        assert!(
            gap.interval_ns() < lin.interval_ns(),
            "gapped is denser than linear"
        );
        // Active span = 20 ms * 1080/1125 = 19.2 ms over 99 gaps.
        assert_eq!(gap.interval_ns(), (period * 1080 / 1125) / 99);
    }

    #[test]
    fn conforms_accepts_on_time_and_late_but_rejects_bunched() {
        // 5 packets over 20 ms = 4 gaps of 5 ms.
        let p = Pacer::new(PacingProfile::Linear, 5, PERIOD_50_FPS_NS);
        assert_eq!(p.interval_ns(), 5_000_000);
        // Exactly on schedule.
        assert!(p.conforms(&[0, 5_000_000, 10_000_000, 15_000_000, 20_000_000], 0));
        // A little late everywhere: still conformant (not bunched ahead).
        assert!(p.conforms(&[100_000, 5_100_000, 10_100_000, 15_100_000, 20_100_000], 0));
        // Bunched (all sent at the start): rejected beyond a small tolerance.
        assert!(!p.conforms(&[0, 0, 0, 0, 0], 1_000_000));
        // Wrong count is rejected.
        assert!(!p.conforms(&[0, 5_000_000], 0));
    }

    #[test]
    fn single_packet_frame_does_not_divide_by_zero() {
        let p = Pacer::new(PacingProfile::Linear, 1, PERIOD_50_FPS_NS);
        assert_eq!(p.offset_ns(0), 0);
        assert!(p.conforms(&[0], 0));
    }

    #[test]
    fn vrx_a_perfectly_paced_frame_conforms_with_a_tiny_buffer() {
        // A sender emitting exactly on the pacer's schedule keeps the VRX at the one
        // packet of head start, so it conforms even to a 1-packet buffer.
        let pacer = Pacer::new(PacingProfile::Linear, 100, PERIOD_50_FPS_NS);
        let offsets: alloc::vec::Vec<u64> = (0..100).map(|i| pacer.offset_ns(i)).collect();
        let vrx = VrxValidator::for_pacer(&pacer, 1);
        let report = vrx.evaluate(&offsets);
        assert_eq!(
            report.peak_occupancy, 1,
            "the schedule holds one packet of head start"
        );
        assert!(!report.underflowed, "no packet is late");
        assert!(report.conforms, "the pacer's own schedule passes its VRX");
        assert_eq!(report.first_violation, None);
    }

    #[test]
    fn vrx_a_burst_overruns_a_narrow_buffer() {
        // The whole frame sent at t=0 fills the buffer to the packet count, far past a
        // narrow buffer: no packet is late (they all arrive early), but the buffer
        // overruns, which is exactly the -21 non-conformance a burst causes.
        let pacer = Pacer::new(PacingProfile::Linear, 100, PERIOD_50_FPS_NS);
        let burst = alloc::vec![0u64; 100];
        let vrx = VrxValidator::for_pacer(&pacer, 8);
        let report = vrx.evaluate(&burst);
        assert_eq!(
            report.peak_occupancy, 100,
            "the whole frame piles into the buffer"
        );
        assert!(!report.underflowed, "early is not late");
        assert!(
            !report.conforms,
            "a burst overruns the narrow 8-packet buffer"
        );
        assert_eq!(
            report.first_violation,
            Some(8),
            "occupancy passes Cmax at the 9th packet"
        );
    }

    #[test]
    fn vrx_a_late_packet_starves_the_receiver() {
        // Packets on schedule except the last, delayed well past its read deadline: the
        // receiver would starve, so the run does not conform even with a huge buffer.
        let pacer = Pacer::new(PacingProfile::Linear, 5, PERIOD_50_FPS_NS); // 5 ms TRS
        let mut offsets: alloc::vec::Vec<u64> = (0..5).map(|i| pacer.offset_ns(i)).collect();
        *offsets.last_mut().unwrap() += 100_000_000; // 100 ms late
        let vrx = VrxValidator::for_pacer(&pacer, 1000);
        let report = vrx.evaluate(&offsets);
        assert!(report.underflowed, "the late packet starves the receiver");
        assert!(!report.conforms);
        assert_eq!(report.first_violation, Some(4));
    }
}
