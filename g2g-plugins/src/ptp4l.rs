//! Query a local `ptp4l`'s sync state over its management socket (M998).
//!
//! `PtpSystemClock` reads `CLOCK_TAI`, which is always readable whether or not
//! `linuxptp` is actually locked to a grandmaster. This is the independent check:
//! the same PTP management GET that `pmc -u` sends, over the Unix datagram socket
//! `ptp4l` listens on, asking for each port's `PORT_DATA_SET` (a port in the
//! SLAVE state means the clock really is following a grandmaster) and the
//! clock-wide `CURRENT_DATA_SET` (how far it currently is from that master).
//!
//! The GET carries the wildcard target, so a boundary clock answers once per port
//! and a status holds every state that came back rather than one.
//!
//! Nothing here requires `ptp4l` to be running: with no daemon the send fails and
//! the caller learns the state is unknown. `ptp4l`'s read-only socket is
//! world-writable by default so an unprivileged process can query it; its
//! read-write socket is owner-only, so trying that one usually needs root.

use core::sync::atomic::{AtomicU32, Ordering};

use std::fs;
use std::io;
use std::os::unix::net::UnixDatagram;
use std::path::{Path, PathBuf};
use std::time::Duration;

use alloc::format;
use alloc::vec::Vec;

use g2g_core::ptp::management::{self, CurrentDataSet, ManagementResponse, PortDataSet, PortState};

use crate::ptpclient::local_clock_id;

/// Where `linuxptp` puts `ptp4l`'s management sockets, newest layout last: the
/// read-only socket (v4.0+) before the read-write one, since querying it needs no
/// privilege. `ptp4l`'s `uds_address` / `uds_ro_address` can move them anywhere,
/// hence [`query_status`] taking an explicit path.
pub const UDS_PATHS: [&str; 4] = [
    "/var/run/ptp4lro",
    "/var/run/ptp/ptp4lro",
    "/var/run/ptp4l",
    "/var/run/ptp/ptp4l",
];

/// How long to wait for `ptp4l` to answer a GET at all.
const RESPONSE_TIMEOUT: Duration = Duration::from_millis(150);
/// How long to wait for further responses once one arrived: the per-port answers
/// to a wildcard GET come back in one burst, so this only has to cover the gap.
const BURST_TIMEOUT: Duration = Duration::from_millis(10);
/// Cap on responses read per GET, so a chatty peer cannot spin the loop.
const MAX_RESPONSES: usize = 16;
/// Receive buffer; the data sets read here are tens of bytes.
const RX_BUF: usize = 512;
/// Our port number within our clock (a management client has one).
const OUR_PORT: u16 = 1;
/// Makes each query socket's name unique within this process.
static NEXT_QUERY_ID: AtomicU32 = AtomicU32::new(0);

/// What a local `ptp4l` reports about its own sync.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ptp4lStatus {
    /// One state per PTP port that answered.
    pub port_states: Vec<PortState>,
    /// The clock-wide currentDS, `None` if `ptp4l` did not answer that GET.
    pub current_data_set: Option<CurrentDataSet>,
}

impl Ptp4lStatus {
    /// Whether `ptp4l` is following a grandmaster: some port is in the SLAVE
    /// state. False while it is still listening / uncalibrated, and false when
    /// this host is itself the grandmaster (its ports are MASTER).
    pub fn locked_to_grandmaster(&self) -> bool {
        self.port_states.contains(&PortState::Slave)
    }

    /// Offset from the master in ns, when `ptp4l` reported a currentDS.
    pub fn offset_from_master_ns(&self) -> Option<i64> {
        self.current_data_set.map(|c| c.offset_from_master_ns)
    }
}

/// Query the `ptp4l` listening on `uds_path`. Fails if the socket is absent or
/// unwritable (no daemon, or its read-write socket without privilege).
pub fn query_status(uds_path: &Path) -> io::Result<Ptp4lStatus> {
    let client = QuerySocket::bind_near(uds_path)?;
    let socket = &client.socket;

    let mut port_states = Vec::new();
    get(socket, uds_path, 0, management::PORT_DATA_SET, |data| {
        if let Some(pds) = PortDataSet::parse(data) {
            port_states.push(pds.port_state);
        }
    })?;

    let mut current_data_set = None;
    get(socket, uds_path, 1, management::CURRENT_DATA_SET, |data| {
        current_data_set = current_data_set.or_else(|| CurrentDataSet::parse(data));
    })?;

    Ok(Ptp4lStatus {
        port_states,
        current_data_set,
    })
}

/// Query the first of [`UDS_PATHS`] that answers, or `None` when no local
/// `ptp4l` is reachable.
pub fn query_local_ptp4l() -> Option<Ptp4lStatus> {
    UDS_PATHS
        .iter()
        .map(Path::new)
        // Skipping absent paths keeps the common no-daemon case from binding a
        // socket per candidate just to have the send fail.
        .filter(|path| path.exists())
        .find_map(|path| query_status(path).ok())
}

/// The socket a query sends from, unlinked when the query ends.
#[derive(Debug)]
struct QuerySocket {
    socket: UnixDatagram,
    path: PathBuf,
}

impl QuerySocket {
    /// Bind a socket `ptp4l` can answer: it replies to whatever address the
    /// request came from, and the kernel leaves an unbound Unix datagram socket
    /// nameless, so the client needs a path of its own. Next to the daemon's own
    /// socket first, as `pmc` does, since that directory is certainly visible to
    /// it; the temp directory when that one is not writable (an unprivileged
    /// process querying the read-only socket under `/var/run`).
    fn bind_near(server: &Path) -> io::Result<Self> {
        let name = format!(
            "g2g-ptp4l-query.{}.{}",
            std::process::id(),
            NEXT_QUERY_ID.fetch_add(1, Ordering::Relaxed)
        );
        let server_directory = server.parent().unwrap_or(Path::new("/")).to_path_buf();
        let mut last_error = io::Error::other("nowhere to bind a query socket");
        for directory in [server_directory, std::env::temp_dir()] {
            let path = directory.join(&name);
            // A socket left behind by a killed process would hold the name.
            let _ = fs::remove_file(&path);
            match UnixDatagram::bind(&path) {
                Ok(socket) => return Ok(Self { socket, path }),
                Err(error) => last_error = error,
            }
        }
        Err(last_error)
    }
}

impl Drop for QuerySocket {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Send one management GET and hand every matching response datum to `on_datum`,
/// until the read timeout expires or [`MAX_RESPONSES`] arrive.
fn get(
    socket: &UnixDatagram,
    uds_path: &Path,
    sequence_id: u16,
    management_id: u16,
    mut on_datum: impl FnMut(&[u8]),
) -> io::Result<()> {
    let request = management::build_get(0, local_clock_id(), OUR_PORT, sequence_id, management_id);
    socket.set_read_timeout(Some(RESPONSE_TIMEOUT))?;
    socket.send_to(&request, uds_path)?;

    let mut buf = [0u8; RX_BUF];
    for received in 0..MAX_RESPONSES {
        // A timeout ends the burst of per-port responses; so does any other read
        // error, which for a datagram socket means there is nothing to wait for.
        let Ok(n) = socket.recv(&mut buf) else { break };
        if received == 0 {
            socket.set_read_timeout(Some(BURST_TIMEOUT))?;
        }
        let Some(response) = ManagementResponse::parse(&buf[..n]) else {
            continue;
        };
        if response.management_id == management_id && response.header.sequence_id == sequence_id {
            on_datum(response.data);
        }
    }
    Ok(())
}
