//! In-process software PTP ordinary clock, SLAVE mode (M594): the UDP transport
//! around the `g2g-core` PTP servo + wire + slave state machine.
//!
//! This is the portable "own the primitive" path (no dependency on an OS PTP
//! daemon): g2g itself speaks PTP over UDP and disciplines a [`PtpClock`], so an
//! endpoint with no `linuxptp` (an embedded box, an appliance) can still lock to
//! a grandmaster. It joins the PTP multicast group on the event (319) and general
//! (320) ports, runs a receive loop per port on a worker thread, drives the
//! [`PtpSlave`] state machine, sends a Delay_Req whenever the slave asks, and
//! feeds each completed `(t1,t2,t3,t4)` cycle to the servo.
//!
//! Software timestamping: `t2` / `t3` are taken with `monotonic_ns()` at recv /
//! send, so accuracy is 10s-100s of us (fine for A/V lip-sync; hardware
//! timestamping for uncompressed ST 2110-20 is a later refinement).
//!
//! ## Running it
//!
//! PTP uses privileged ports (319 / 320), so this needs `CAP_NET_BIND_SERVICE`
//! (typically root) and a reachable grandmaster (`ptp4l -m` or a hardware GM) on
//! the network. It binds the ports exclusively, so it replaces `ptp4l` rather
//! than co-existing with it (co-running on one host would need `SO_REUSEPORT`, a
//! later enhancement). Not in scope: BMCA / Announce (it follows whatever master
//! sends Sync on its domain), peer-delay, unicast, hardware timestamping.

use core::sync::atomic::{AtomicBool, Ordering};

use std::io;
use std::net::{Ipv4Addr, UdpSocket};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use alloc::vec::Vec;

use g2g_core::metrics::monotonic_ns;
use g2g_core::ptp::wire;
use g2g_core::{
    ClockCandidate, MonotonicClock, PipelineClock, PtpClock, PtpSlave, PtpState, RefNs, SlaveAction,
    TaiNs,
};

/// The PTP default profile primary multicast address (224.0.1.129).
pub const PTP_PRIMARY: Ipv4Addr = Ipv4Addr::new(224, 0, 1, 129);
/// PTP event-message UDP port (Sync, Delay_Req).
pub const EVENT_PORT: u16 = 319;
/// PTP general-message UDP port (Follow_Up, Delay_Resp, Announce).
pub const GENERAL_PORT: u16 = 320;
/// Our port number within our clock (a SLAVE has one; 1 by convention).
const OUR_PORT: u16 = 1;
/// Receive buffer; PTP messages a slave consumes are well under this.
const RX_BUF: usize = 256;

/// A software PTP SLAVE ordinary clock disciplining a [`PtpClock`] from a
/// grandmaster on the network. Drop stops the workers.
pub struct PtpClient {
    clock: Arc<PtpClock>,
    stop: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
}

impl core::fmt::Debug for PtpClient {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PtpClient")
            .field("state", &self.state())
            .field("now_ns", &self.now_ns())
            .finish()
    }
}

impl PtpClient {
    /// Start a SLAVE on PTP domain 0.
    pub fn new() -> io::Result<Self> {
        Self::with_domain(0)
    }

    /// Start a SLAVE on the given PTP domain. Fails if the privileged ports
    /// cannot be bound (needs `CAP_NET_BIND_SERVICE` / root) or the multicast
    /// group cannot be joined.
    pub fn with_domain(domain: u8) -> io::Result<Self> {
        let reference: Arc<dyn PipelineClock + Send + Sync> = Arc::new(MonotonicClock);
        let clock = Arc::new(PtpClock::new(reference));
        let slave = Arc::new(Mutex::new(PtpSlave::new(domain, local_clock_id(), OUR_PORT)));
        let stop = Arc::new(AtomicBool::new(false));

        // Event socket both receives Sync and sends Delay_Req; general socket
        // receives Follow_Up / Delay_Resp. Delay_Req always goes out the event
        // socket, so both reader threads share it for sending.
        let event = Arc::new(bind_multicast(EVENT_PORT)?);
        let general = Arc::new(bind_multicast(GENERAL_PORT)?);

        let mut workers = Vec::with_capacity(2);
        for recv in [event.clone(), general.clone()] {
            let send = event.clone();
            let slave = slave.clone();
            let clock = clock.clone();
            let stop = stop.clone();
            workers.push(
                thread::Builder::new()
                    .name(alloc::string::String::from("g2g-ptpclient"))
                    .spawn(move || reader_loop(&recv, &send, &slave, &clock, &stop, domain))
                    .map_err(io::Error::other)?,
            );
        }

        Ok(Self { clock, stop, workers })
    }

    /// The disciplined clock, to share via an element's `provide_clock` or read.
    pub fn clock(&self) -> Arc<PtpClock> {
        self.clock.clone()
    }
    /// Election candidate at the `PtpGrandmaster` tier, offered only once locked.
    pub fn candidate(&self) -> Option<ClockCandidate> {
        self.clock.candidate()
    }
    /// Whether the slave has locked to the grandmaster.
    pub fn is_locked(&self) -> bool {
        self.clock.is_locked()
    }
    /// Current servo state.
    pub fn state(&self) -> PtpState {
        self.clock.state()
    }
    /// The grandmaster (TAI) time estimate now.
    pub fn now_ns(&self) -> u64 {
        self.clock.now_ns()
    }
    /// Last servo error (fit residual); the sync-quality metric.
    pub fn error_ns(&self) -> i64 {
        self.clock.error_ns()
    }
}

impl Drop for PtpClient {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Bind `0.0.0.0:port`, join the PTP multicast group, and set a read timeout so
/// the reader loop can poll the stop flag.
fn bind_multicast(port: u16) -> io::Result<UdpSocket> {
    let sock = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, port))?;
    sock.join_multicast_v4(&PTP_PRIMARY, &Ipv4Addr::UNSPECIFIED)?;
    sock.set_read_timeout(Some(Duration::from_millis(200)))?;
    Ok(sock)
}

/// Receive-and-dispatch loop for one socket. `recv` is the socket to read;
/// `send` is the event socket every Delay_Req goes out on.
fn reader_loop(
    recv: &UdpSocket,
    send: &UdpSocket,
    slave: &Mutex<PtpSlave>,
    clock: &PtpClock,
    stop: &AtomicBool,
    domain: u8,
) {
    let mut buf = [0u8; RX_BUF];
    while !stop.load(Ordering::Relaxed) {
        let n = match recv.recv_from(&mut buf) {
            Ok((n, _from)) => n,
            // Timeout / would-block: poll the stop flag and retry. Other errors
            // are transient here (a malformed datagram never reaches us as an
            // error); keep the loop alive.
            Err(_) => continue,
        };
        // Timestamp the receive as close to arrival as we can in software.
        let local_rx = monotonic_ns();
        let action = slave.lock().unwrap().on_message(&buf[..n], local_rx);
        match action {
            SlaveAction::SendDelayReq(seq) => {
                let (clock_id, port) = {
                    let s = slave.lock().unwrap();
                    (s.clock_id(), s.port())
                };
                let dr = wire::build_delay_req(domain, clock_id, port, seq);
                if send.send_to(&dr, (PTP_PRIMARY, EVENT_PORT)).is_ok() {
                    // t3: our Delay_Req TX time, sampled right after the send.
                    let t3 = monotonic_ns();
                    slave.lock().unwrap().delay_req_sent(seq, t3);
                }
            }
            SlaveAction::Exchange { t1, t2, t3, t4 } => {
                clock.sync_exchange(TaiNs(t1), RefNs(t2), RefNs(t3), TaiNs(t4));
            }
            SlaveAction::Idle => {}
        }
    }
}

/// A locally-administered clock identity for this process. Not a real
/// EUI-64/MAC-derived id (that needs an interface query); unique enough for a
/// SLAVE, which only needs the master to echo it back in Delay_Resp.
fn local_clock_id() -> [u8; 8] {
    let pid = std::process::id().to_be_bytes();
    [0xfe, 0xff, 0x00, pid[0], pid[1], pid[2], pid[3], 0x01]
}
