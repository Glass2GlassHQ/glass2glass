//! M998: `ptp4l` state query over the management socket.
//!
//! The thing under test is the management wire layout and the exchange around it,
//! so the peer here is a fake `ptp4l`: a Unix datagram socket on a temp path that
//! answers a GET the way `ptp4l` does, with responses laid out from the IEEE 1588
//! / linuxptp field offsets rather than from the parser being tested. That makes
//! the query provable in CI on a host with no `linuxptp` installed at all.
//!
//! Also covers the degraded path: with no daemon reachable, `PtpSystemClock`
//! reports the grandmaster state as unknown instead of claiming lock.
//!
//! Run: `cargo test -p g2g-plugins --features ptp --test m998_ptp4l_state_query`.
#![cfg(all(target_os = "linux", feature = "ptp"))]

use std::io;
use std::os::unix::net::UnixDatagram;
use std::path::{Path, PathBuf};
use std::thread::{self, sleep, JoinHandle};
use std::time::Duration;

use g2g_core::ptp::management::PortState;
use g2g_plugins::ptp4l::{self, Ptp4lStatus};
use g2g_plugins::ptpsystemclock::PtpSystemClock;

/// Management message field offsets (IEEE 1588-2008 clause 15), spelled out here
/// so the fake daemon does not borrow the parser's view of the layout.
const MESSAGE_TYPE_MANAGEMENT: u8 = 0x0d;
const PTP_VERSION_2019: u8 = 0x12;
const SEQUENCE_ID_OFFSET: usize = 30;
const MANAGEMENT_FLAGS_OFFSET: usize = 46;
const ACTION_GET: u8 = 0;
const ACTION_RESPONSE: u8 = 2;
const TLV_OFFSET: usize = 48;
const TLV_MANAGEMENT: u16 = 0x0001;
const MANAGEMENT_ID_OFFSET: usize = 52;
const DATA_OFFSET: usize = 54;
const PORT_DATA_SET: u16 = 0x2004;
const CURRENT_DATA_SET: u16 = 0x2001;
/// portDS: portIdentity(10) portState(1) ... versionNumber(1).
const PORT_DATA_SET_LEN: usize = 26;
/// currentDS: stepsRemoved(2) offsetFromMaster(8) meanPathDelay(8).
const CURRENT_DATA_SET_LEN: usize = 18;

/// What the fake daemon reports for one port.
struct FakePort {
    number: u16,
    state: u8,
}

/// The offset from master the fake daemon reports, in ns.
const FAKE_OFFSET_NS: i64 = -4321;
/// The mean path delay the fake daemon reports, in ns.
const FAKE_PATH_DELAY_NS: i64 = 12_345;
/// GETs one `query_status` sends (PORT_DATA_SET then CURRENT_DATA_SET).
const GETS_PER_QUERY: usize = 2;

/// A fake `ptp4l` management socket. Answers `GETS_PER_QUERY` requests, then the
/// thread ends and the socket is unlinked.
struct FakePtp4l {
    path: PathBuf,
    worker: Option<JoinHandle<()>>,
}

impl FakePtp4l {
    fn start(name: &str, ports: Vec<FakePort>) -> io::Result<Self> {
        let path =
            std::env::temp_dir().join(format!("g2g-fake-ptp4l.{}.{name}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let socket = UnixDatagram::bind(&path)?;
        socket.set_read_timeout(Some(Duration::from_secs(5)))?;
        let worker = thread::spawn(move || serve(&socket, &ports, GETS_PER_QUERY));
        Ok(Self {
            path,
            worker: Some(worker),
        })
    }
}

impl Drop for FakePtp4l {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Answer `requests` management GETs the way `ptp4l` does: one response per port
/// for the per-port portDS, a single response for the clock-wide currentDS.
fn serve(socket: &UnixDatagram, ports: &[FakePort], requests: usize) {
    let mut buf = [0u8; 512];
    for _ in 0..requests {
        let Ok((n, from)) = socket.recv_from(&mut buf) else {
            return;
        };
        let request = &buf[..n];
        assert_eq!(request[0] & 0x0f, MESSAGE_TYPE_MANAGEMENT, "not management");
        assert_eq!(
            request[MANAGEMENT_FLAGS_OFFSET] & 0x0f,
            ACTION_GET,
            "a status query only GETs"
        );
        assert_eq!(
            u16::from_be_bytes([request[TLV_OFFSET], request[TLV_OFFSET + 1]]),
            TLV_MANAGEMENT
        );
        assert_eq!(
            n, DATA_OFFSET,
            "a GET carries an empty management TLV, so 54 bytes"
        );
        let sequence_id =
            u16::from_be_bytes([request[SEQUENCE_ID_OFFSET], request[SEQUENCE_ID_OFFSET + 1]]);
        let management_id = u16::from_be_bytes([
            request[MANAGEMENT_ID_OFFSET],
            request[MANAGEMENT_ID_OFFSET + 1],
        ]);

        let mut replies = Vec::new();
        match management_id {
            PORT_DATA_SET => {
                for port in ports {
                    let mut datum = [0u8; PORT_DATA_SET_LEN];
                    datum[8..10].copy_from_slice(&port.number.to_be_bytes());
                    datum[10] = port.state;
                    replies.push(response(sequence_id, management_id, &datum));
                }
            }
            CURRENT_DATA_SET => {
                let mut datum = [0u8; CURRENT_DATA_SET_LEN];
                datum[0..2].copy_from_slice(&1u16.to_be_bytes());
                // Both fields are TimeInterval: nanoseconds scaled by 2^16.
                datum[2..10].copy_from_slice(&(FAKE_OFFSET_NS << 16).to_be_bytes());
                datum[10..18].copy_from_slice(&(FAKE_PATH_DELAY_NS << 16).to_be_bytes());
                replies.push(response(sequence_id, management_id, &datum));
            }
            other => panic!("unexpected managementId {other:#06x}"),
        }
        for reply in replies {
            socket
                .send_to_addr(&reply, &from)
                .expect("the client socket must be addressable");
        }
    }
}

/// Build a management RESPONSE carrying `datum`.
fn response(sequence_id: u16, management_id: u16, datum: &[u8]) -> Vec<u8> {
    let mut m = vec![0u8; DATA_OFFSET + datum.len()];
    m[0] = MESSAGE_TYPE_MANAGEMENT;
    m[1] = PTP_VERSION_2019;
    let length = m.len() as u16;
    m[2..4].copy_from_slice(&length.to_be_bytes());
    m[20..28].copy_from_slice(&[0x22; 8]); // sourcePortIdentity: the fake clock
    m[28..30].copy_from_slice(&1u16.to_be_bytes());
    m[SEQUENCE_ID_OFFSET..SEQUENCE_ID_OFFSET + 2].copy_from_slice(&sequence_id.to_be_bytes());
    m[MANAGEMENT_FLAGS_OFFSET] = ACTION_RESPONSE;
    m[TLV_OFFSET..TLV_OFFSET + 2].copy_from_slice(&TLV_MANAGEMENT.to_be_bytes());
    // lengthField covers the managementId plus the datum.
    let tlv_length = 2 + datum.len() as u16;
    m[TLV_OFFSET + 2..TLV_OFFSET + 4].copy_from_slice(&tlv_length.to_be_bytes());
    m[MANAGEMENT_ID_OFFSET..MANAGEMENT_ID_OFFSET + 2].copy_from_slice(&management_id.to_be_bytes());
    m[DATA_OFFSET..].copy_from_slice(datum);
    m
}

fn query(daemon: &FakePtp4l) -> Ptp4lStatus {
    ptp4l::query_status(&daemon.path).expect("the fake daemon answers")
}

/// A boundary clock following a grandmaster on one of its two ports.
#[test]
fn reads_a_slave_port_state_and_the_offset_from_master() {
    let daemon = FakePtp4l::start(
        "slave",
        vec![
            FakePort {
                number: 1,
                state: 9, // PS_SLAVE
            },
            FakePort {
                number: 2,
                state: 4, // PS_LISTENING
            },
        ],
    )
    .expect("bind the fake ptp4l socket");

    let status = query(&daemon);
    assert_eq!(
        status.port_states,
        vec![PortState::Slave, PortState::Listening],
        "one state per port that answered"
    );
    assert!(
        status.locked_to_grandmaster(),
        "a SLAVE port means real grandmaster lock"
    );
    assert_eq!(status.offset_from_master_ns(), Some(FAKE_OFFSET_NS));
    let current = status.current_data_set.expect("currentDS answered");
    assert_eq!(current.mean_path_delay_ns, FAKE_PATH_DELAY_NS);
    assert_eq!(current.steps_removed, 1);
}

/// A daemon that is running but not synced must not read as locked.
#[test]
fn a_listening_clock_is_not_locked_to_a_grandmaster() {
    let daemon = FakePtp4l::start(
        "listening",
        vec![FakePort {
            number: 1,
            state: 4, // PS_LISTENING
        }],
    )
    .expect("bind the fake ptp4l socket");

    let status = query(&daemon);
    assert_eq!(status.port_states, vec![PortState::Listening]);
    assert!(!status.locked_to_grandmaster());
}

/// No socket at the path: the query fails rather than inventing a state.
#[test]
fn a_missing_daemon_fails_the_query() {
    let missing = std::env::temp_dir().join("g2g-fake-ptp4l.does-not-exist");
    let _ = std::fs::remove_file(&missing);
    assert!(ptp4l::query_status(&missing).is_err());
}

/// `PtpSystemClock` reports the grandmaster state as unknown on a host with no
/// `ptp4l`, instead of reading a smooth `CLOCK_TAI` as proof of lock.
#[test]
fn ptp_system_clock_degrades_without_a_local_ptp4l() {
    let daemon_present = ptp4l::UDS_PATHS.iter().any(|p| Path::new(p).exists());
    let clock = PtpSystemClock::new();
    // The poller queries once at start; give it time to finish and publish.
    sleep(Duration::from_millis(800));

    match clock.grandmaster_locked() {
        None => assert!(
            !daemon_present,
            "a reachable ptp4l should have answered: {:?}",
            ptp4l::UDS_PATHS
        ),
        Some(locked) => {
            assert!(
                daemon_present,
                "no ptp4l socket exists, nothing could answer"
            );
            eprintln!(
                "m998: local ptp4l answered {:?}, grandmaster locked = {locked}",
                clock.ptp4l_status()
            );
        }
    }
}
