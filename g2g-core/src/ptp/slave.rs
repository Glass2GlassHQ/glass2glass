//! SLAVE-mode PTP delay-request-response state machine (M594).
//!
//! Transport-agnostic: fed each received PTP message (raw bytes + the local time
//! it arrived) and the TX time of each Delay_Req it asks the transport to send,
//! it assembles the four timestamps of one synchronisation cycle and yields them
//! for the [`PtpServo`](super::PtpServo). A [`PtpClient`] provides the UDP
//! transport; this module is pure logic, so the whole slave path is CI-testable
//! by injecting message bytes with no sockets.
//!
//! One cycle (IEEE 1588 E2E, ordinary clock in SLAVE):
//! - `t1` = master's Sync TX time (from a one-step Sync's originTimestamp, or a
//!   two-step Sync's Follow_Up preciseOriginTimestamp), plus its correctionField(s),
//! - `t2` = our Sync RX time,
//! - `t3` = our Delay_Req TX time,
//! - `t4` = master's Delay_Req RX time (Delay_Resp receiveTimestamp), less its
//!   correctionField.
//!
//! A new Sync starts a fresh cycle (any in-flight Delay_Req is abandoned), so a
//! late Delay_Resp cannot cross a `t3` with a mismatched `t1`/`t2`.

use super::wire::{self, DelayResp, PtpHeader, PtpMessageType};

/// What the transport should do after a message is fed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlaveAction {
    /// Nothing to do.
    Idle,
    /// Send a Delay_Req with this sequence id, then report its TX timestamp via
    /// [`delay_req_sent`](PtpSlave::delay_req_sent).
    SendDelayReq(u16),
    /// A complete `(t1, t2, t3, t4)` cycle, ready for `PtpServo::sync_exchange`.
    Exchange { t1: u64, t2: u64, t3: u64, t4: u64 },
}

/// A SLAVE ordinary clock's per-cycle state.
#[derive(Clone, Debug)]
pub struct PtpSlave {
    domain: u8,
    our_clock_id: [u8; 8],
    our_port: u16,
    // Current cycle, cleared and restarted on each Sync.
    sync_seq: Option<u16>,
    two_step: bool,
    sync_correction_ns: i64,
    t1: Option<u64>,
    t2: Option<u64>,
    // Delay exchange within the current cycle.
    delay_req_seq: u16,
    dr_seq: Option<u16>,
    t3: Option<u64>,
}

impl PtpSlave {
    /// A slave on `domain` identifying itself with `clock_id` / `port` (echoed in
    /// its Delay_Req and matched against each Delay_Resp's requestingPortIdentity).
    pub fn new(domain: u8, clock_id: [u8; 8], port: u16) -> Self {
        Self {
            domain,
            our_clock_id: clock_id,
            our_port: port,
            sync_seq: None,
            two_step: false,
            sync_correction_ns: 0,
            t1: None,
            t2: None,
            delay_req_seq: 0,
            dr_seq: None,
            t3: None,
        }
    }

    /// Our clock identity (for the transport to build Delay_Req messages).
    pub fn clock_id(&self) -> [u8; 8] {
        self.our_clock_id
    }
    /// Our port number.
    pub fn port(&self) -> u16 {
        self.our_port
    }
    /// The domain this slave follows.
    pub fn domain(&self) -> u8 {
        self.domain
    }

    /// Feed a received PTP message (raw bytes) with the local time it arrived.
    /// Messages on another domain, our own multicast echoes (Delay_Req), Announce,
    /// and anything malformed are ignored.
    pub fn on_message(&mut self, msg: &[u8], local_rx_ns: u64) -> SlaveAction {
        let Some(h) = PtpHeader::parse(msg) else {
            return SlaveAction::Idle;
        };
        if h.domain != self.domain {
            return SlaveAction::Idle;
        }
        match h.message_type {
            PtpMessageType::Sync => self.on_sync(&h, msg, local_rx_ns),
            PtpMessageType::FollowUp => self.on_follow_up(&h, msg),
            PtpMessageType::DelayResp => self.on_delay_resp(&h, msg),
            _ => SlaveAction::Idle,
        }
    }

    fn on_sync(&mut self, h: &PtpHeader, msg: &[u8], local_rx_ns: u64) -> SlaveAction {
        // A Sync starts a fresh cycle: drop any in-flight delay exchange so a
        // stale Delay_Resp cannot pair a new t2 with an old t3.
        self.sync_seq = Some(h.sequence_id);
        self.two_step = h.two_step();
        self.sync_correction_ns = h.correction_ns;
        self.t2 = Some(local_rx_ns);
        self.t1 = None;
        self.dr_seq = None;
        self.t3 = None;

        if self.two_step {
            // Wait for the Follow_Up to carry the accurate t1.
            return SlaveAction::Idle;
        }
        // One-step: t1 is in the Sync itself.
        match wire::parse_sync_origin(msg) {
            Some(origin) => {
                self.t1 = Some(origin.saturating_add_signed(h.correction_ns));
                self.begin_delay_req()
            }
            None => SlaveAction::Idle,
        }
    }

    fn on_follow_up(&mut self, h: &PtpHeader, msg: &[u8]) -> SlaveAction {
        // Only for the current two-step cycle, and only once.
        if self.sync_seq != Some(h.sequence_id) || !self.two_step || self.t1.is_some() {
            return SlaveAction::Idle;
        }
        match wire::parse_follow_up_origin(msg) {
            Some(precise) => {
                // t1 carries both the Sync's and Follow_Up's correctionField.
                let corr = self.sync_correction_ns.saturating_add(h.correction_ns);
                self.t1 = Some(precise.saturating_add_signed(corr));
                self.begin_delay_req()
            }
            None => SlaveAction::Idle,
        }
    }

    fn on_delay_resp(&mut self, h: &PtpHeader, msg: &[u8]) -> SlaveAction {
        let Some(resp) = DelayResp::parse(msg) else {
            return SlaveAction::Idle;
        };
        // Must be the response to *our* outstanding Delay_Req.
        if self.dr_seq != Some(h.sequence_id)
            || resp.requesting_clock_id != self.our_clock_id
            || resp.requesting_port != self.our_port
        {
            return SlaveAction::Idle;
        }
        let (Some(t1), Some(t2), Some(t3)) = (self.t1, self.t2, self.t3) else {
            return SlaveAction::Idle;
        };
        // t4 less the Delay_Resp correctionField.
        let t4 = resp.receive_ts_ns.saturating_add_signed(-h.correction_ns);
        // Cycle complete; clear the delay half so a duplicate Resp does nothing.
        self.dr_seq = None;
        self.t3 = None;
        SlaveAction::Exchange { t1, t2, t3, t4 }
    }

    /// Both t1 and t2 are known: allocate a Delay_Req sequence and ask the
    /// transport to send it.
    fn begin_delay_req(&mut self) -> SlaveAction {
        self.delay_req_seq = self.delay_req_seq.wrapping_add(1);
        self.dr_seq = Some(self.delay_req_seq);
        self.t3 = None;
        SlaveAction::SendDelayReq(self.delay_req_seq)
    }

    /// Report the TX timestamp of the Delay_Req the transport just sent for `seq`.
    pub fn delay_req_sent(&mut self, seq: u16, t3_ns: u64) {
        if self.dr_seq == Some(seq) {
            self.t3 = Some(t3_ns);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::wire::{BODY_OFFSET, HEADER_LEN, PTP_VERSION, TIMESTAMP_LEN};
    use super::super::{PtpServo, PtpState};
    use super::*;
    use crate::time::{RefNs, TaiNs};
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicU64, Ordering};

    use crate::clock::PipelineClock;

    const US: [u8; 8] = [0xaa; 8];
    const US_PORT: u16 = 1;
    const MASTER: [u8; 8] = [0x11; 8];
    const EPOCH: i128 = 1_700_000_000_000_000_000;
    const DELAY: i128 = 100_000;
    const GAP: u64 = 1_000_000;

    #[derive(Debug, Default)]
    struct ManualClock(AtomicU64);
    impl ManualClock {
        fn set(&self, v: u64) {
            self.0.store(v, Ordering::Release);
        }
    }
    impl PipelineClock for ManualClock {
        fn now_ns(&self) -> u64 {
            self.0.load(Ordering::Acquire)
        }
    }

    fn header(buf: &mut [u8], mtype: u8, two_step: bool, src: [u8; 8], seq: u16) {
        buf[0] = mtype & 0x0f;
        buf[1] = PTP_VERSION;
        let len = buf.len() as u16;
        buf[2..4].copy_from_slice(&len.to_be_bytes());
        let flags: u16 = if two_step { 0x0200 } else { 0 };
        buf[6..8].copy_from_slice(&flags.to_be_bytes());
        buf[20..28].copy_from_slice(&src);
        buf[28..30].copy_from_slice(&1u16.to_be_bytes());
        buf[30..32].copy_from_slice(&seq.to_be_bytes());
    }

    fn put_ts(buf: &mut [u8], off: usize, ns: u64) {
        let secs = ns / 1_000_000_000;
        let nanos = (ns % 1_000_000_000) as u32;
        buf[off] = (secs >> 40) as u8;
        buf[off + 1] = (secs >> 32) as u8;
        buf[off + 2] = (secs >> 24) as u8;
        buf[off + 3] = (secs >> 16) as u8;
        buf[off + 4] = (secs >> 8) as u8;
        buf[off + 5] = secs as u8;
        buf[off + 6..off + 10].copy_from_slice(&nanos.to_be_bytes());
    }

    fn sync_msg(seq: u16, two_step: bool, origin_ns: u64) -> Vec<u8> {
        let mut b = vec![0u8; HEADER_LEN + TIMESTAMP_LEN];
        header(&mut b, 0x0, two_step, MASTER, seq);
        put_ts(&mut b, BODY_OFFSET, origin_ns);
        b
    }
    fn follow_up_msg(seq: u16, precise_ns: u64) -> Vec<u8> {
        let mut b = vec![0u8; HEADER_LEN + TIMESTAMP_LEN];
        header(&mut b, 0x8, false, MASTER, seq);
        put_ts(&mut b, BODY_OFFSET, precise_ns);
        b
    }
    fn delay_resp_msg(seq: u16, receive_ns: u64, req_id: [u8; 8], req_port: u16) -> Vec<u8> {
        let mut b = vec![0u8; HEADER_LEN + TIMESTAMP_LEN + 10];
        header(&mut b, 0x9, false, MASTER, seq);
        put_ts(&mut b, BODY_OFFSET, receive_ns);
        b[BODY_OFFSET + TIMESTAMP_LEN..BODY_OFFSET + TIMESTAMP_LEN + 8].copy_from_slice(&req_id);
        b[BODY_OFFSET + TIMESTAMP_LEN + 8..BODY_OFFSET + TIMESTAMP_LEN + 10]
            .copy_from_slice(&req_port.to_be_bytes());
        b
    }

    #[test]
    fn two_step_cycle_yields_the_quad() {
        let mut s = PtpSlave::new(0, US, US_PORT);
        // Sync received at t2 = 1000; two-step, so no t1 yet.
        assert_eq!(s.on_message(&sync_msg(5, true, 0), 1000), SlaveAction::Idle);
        // Follow_Up carries t1 = 900; both known -> send a Delay_Req.
        let SlaveAction::SendDelayReq(dr) = s.on_message(&follow_up_msg(5, 900), 0) else {
            panic!("expected SendDelayReq after Follow_Up");
        };
        // Transport sent it at t3 = 1100.
        s.delay_req_sent(dr, 1100);
        // Delay_Resp: master received our req at t4 = 1200.
        assert_eq!(
            s.on_message(&delay_resp_msg(dr, 1200, US, US_PORT), 0),
            SlaveAction::Exchange {
                t1: 900,
                t2: 1000,
                t3: 1100,
                t4: 1200
            }
        );
    }

    #[test]
    fn one_step_cycle_sends_delay_req_immediately() {
        let mut s = PtpSlave::new(0, US, US_PORT);
        // One-step Sync carries t1 = 900 in its own originTimestamp.
        let SlaveAction::SendDelayReq(dr) = s.on_message(&sync_msg(5, false, 900), 1000) else {
            panic!("one-step Sync should send a Delay_Req at once");
        };
        s.delay_req_sent(dr, 1100);
        assert_eq!(
            s.on_message(&delay_resp_msg(dr, 1200, US, US_PORT), 0),
            SlaveAction::Exchange {
                t1: 900,
                t2: 1000,
                t3: 1100,
                t4: 1200
            }
        );
    }

    #[test]
    fn ignores_other_domain_and_other_slaves_responses() {
        let mut s = PtpSlave::new(0, US, US_PORT);
        // A Sync on domain 1: our slave is domain 0.
        let mut other_domain = sync_msg(5, false, 900);
        other_domain[4] = 1;
        assert_eq!(s.on_message(&other_domain, 1000), SlaveAction::Idle);

        // Our own domain-0 cycle, but the Delay_Resp is addressed to another slave.
        let SlaveAction::SendDelayReq(dr) = s.on_message(&sync_msg(6, false, 900), 1000) else {
            panic!("expected a Delay_Req");
        };
        s.delay_req_sent(dr, 1100);
        assert_eq!(
            s.on_message(&delay_resp_msg(dr, 1200, [0x77; 8], 9), 0),
            SlaveAction::Idle,
            "a Delay_Resp for a different requester is ignored"
        );
    }

    #[test]
    fn full_slave_path_drives_the_servo_to_lock() {
        // End-to-end core: synthesise a two-step master's message stream, run it
        // through the parser + slave + servo (no sockets), and confirm lock.
        let clk = Arc::new(ManualClock::default());
        let mut servo = PtpServo::new(clk.clone());
        let mut slave = PtpSlave::new(0, US, US_PORT);

        let master = |local: u64| -> i128 { EPOCH + local as i128 };
        let mut local = 1_000_000_000u64;
        for seq in 0..24u16 {
            clk.set(local);
            // Sync arrives at t2 = local; t1 (sent DELAY earlier, in master time).
            let t1 = (master(local) - DELAY) as u64;
            assert_eq!(
                slave.on_message(&sync_msg(seq, true, 0), local),
                SlaveAction::Idle
            );
            let action = slave.on_message(&follow_up_msg(seq, t1), local);
            let SlaveAction::SendDelayReq(dr) = action else {
                panic!("expected SendDelayReq");
            };
            // We send the Delay_Req GAP later, at t3; master receives at t4.
            let t3 = local + GAP;
            clk.set(t3);
            slave.delay_req_sent(dr, t3);
            let t4 = (master(t3) + DELAY) as u64;
            match slave.on_message(&delay_resp_msg(dr, t4, US, US_PORT), t3) {
                SlaveAction::Exchange { t1, t2, t3, t4 } => {
                    servo.sync_exchange(TaiNs(t1), RefNs(t2), RefNs(t3), TaiNs(t4));
                }
                other => panic!("expected an Exchange, got {other:?}"),
            }
            local += 125_000_000;
        }

        assert_eq!(
            servo.state(),
            PtpState::Locked,
            "the full slave path locks the servo"
        );
        assert!(
            servo.error_ns().unsigned_abs() < 1_000,
            "sub-us servo error"
        );
        // Path delay recovered (~100 us; the tiny drift-over-gap term is < 1 us).
        assert!((servo.mean_path_delay_ns() - DELAY as i64).abs() < 1_000);
        clk.set(local);
        let est = servo.now_ns() as i128;
        assert!(
            (est - (EPOCH + local as i128)).abs() < 100_000,
            "TAI estimate within 100 us"
        );
    }
}
