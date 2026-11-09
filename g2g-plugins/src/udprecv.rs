use alloc::string::String;
use alloc::vec::Vec;

use std::io::{self, IoSliceMut};
use std::net::{IpAddr, SocketAddr};

use g2g_core::bus::{BusHandle, BusMessage};
use quinn_udp::{RecvMeta, UdpSockRef, UdpSocketState};
use tokio::io::Interest;
use tokio::net::UdpSocket;

// largest payload an IPv4 UDP datagram can carry, so a buffer this size never truncates one
pub(crate) const MAX_DATAGRAM: usize = 65_507;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UdpReceiveConfig {
    pub(crate) batch_size: usize,
    pub(crate) mtu: usize,
    pub(crate) allow_gro: bool,
    pub(crate) source_filter: Vec<IpAddr>,
    pub(crate) source_filter_exclusive: bool,
    pub(crate) skip_first_bytes: usize,
}

impl Default for UdpReceiveConfig {
    fn default() -> Self {
        Self {
            // gst udpsrc2 batch-size default
            batch_size: 16,
            mtu: MAX_DATAGRAM,
            allow_gro: false,
            source_filter: Vec::new(),
            source_filter_exclusive: false,
            skip_first_bytes: 0,
        }
    }
}

impl UdpReceiveConfig {
    fn admits(&self, sender: IpAddr) -> bool {
        if self.source_filter.is_empty() {
            return true;
        }
        self.source_filter.contains(&sender) != self.source_filter_exclusive
    }
}

#[derive(Debug)]
pub(crate) struct IdleReport {
    pub(crate) timeout_ns: u64,
    pub(crate) bus: BusHandle,
    pub(crate) element: String,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Datagram<'a> {
    pub(crate) payload: &'a [u8],
    pub(crate) from: SocketAddr,
    pub(crate) arrival_ns: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct BatchCursor {
    buffer: usize,
    offset: usize,
}

#[derive(Debug, Clone, Copy)]
struct DatagramPosition {
    buffer: usize,
    start: usize,
    end: usize,
}

#[derive(Debug)]
pub(crate) struct UdpReceiver {
    socket: UdpSocket,
    state: UdpSocketState,
    config: UdpReceiveConfig,
    buffers: Vec<Vec<u8>>,
    metas: Vec<RecvMeta>,
    arrivals_ns: Vec<u64>,
    filled: usize,
    cursor: BatchCursor,
    idle_report: Option<IdleReport>,
    quiet_since_ns: u64,
}

impl UdpReceiver {
    pub(crate) fn new(socket: UdpSocket, config: UdpReceiveConfig) -> io::Result<Self> {
        let state = UdpSocketState::new(UdpSockRef::from(&socket))?;
        set_gro(&state, &socket, config.allow_gro)?;
        let segments = if config.allow_gro {
            state.gro_segments()
        } else {
            1
        };
        let buffer_len = config.mtu * segments;
        let batch_size = config.batch_size;
        Ok(Self {
            socket,
            state,
            config,
            buffers: (0..batch_size)
                .map(|_| alloc::vec![0u8; buffer_len])
                .collect(),
            metas: alloc::vec![RecvMeta::default(); batch_size],
            arrivals_ns: alloc::vec![0; batch_size],
            filled: 0,
            cursor: BatchCursor::default(),
            idle_report: None,
            quiet_since_ns: g2g_core::metrics::monotonic_ns(),
        })
    }

    #[cfg(feature = "udp-ingress")]
    pub(crate) fn with_idle_report(mut self, report: IdleReport) -> Self {
        self.idle_report = Some(report);
        self
    }

    pub(crate) fn socket(&self) -> &UdpSocket {
        &self.socket
    }

    pub(crate) async fn recv(&mut self) -> io::Result<Datagram<'_>> {
        loop {
            if let Some(position) = self.advance() {
                return Ok(self.datagram_at(position));
            }
            self.fill().await?;
        }
    }

    // the datagrams already read off the socket by the last batch, without waiting for more
    #[cfg(feature = "udp-ingress")]
    pub(crate) fn next_buffered(&mut self) -> Option<Datagram<'_>> {
        let position = self.advance()?;
        Some(self.datagram_at(position))
    }

    fn datagram_at(&self, position: DatagramPosition) -> Datagram<'_> {
        Datagram {
            payload: &self.buffers[position.buffer][position.start..position.end],
            from: self.metas[position.buffer].addr,
            arrival_ns: self.arrivals_ns[position.buffer],
        }
    }

    fn advance(&mut self) -> Option<DatagramPosition> {
        while self.cursor.buffer < self.filled {
            let meta = &self.metas[self.cursor.buffer];
            if self.cursor.offset >= meta.len || !self.config.admits(meta.addr.ip()) {
                self.cursor = BatchCursor {
                    buffer: self.cursor.buffer + 1,
                    offset: 0,
                };
                continue;
            }
            // a GRO buffer holds several datagrams at `stride` intervals, the last one may be shorter
            let start = self.cursor.offset;
            let end = (start + meta.stride.max(1)).min(meta.len);
            self.cursor.offset = end;
            let payload_start = start + self.config.skip_first_bytes;
            if payload_start > end {
                continue;
            }
            return Some(DatagramPosition {
                buffer: self.cursor.buffer,
                start: payload_start,
                end,
            });
        }
        None
    }

    // the quiet time lives in self, since the RTP loop drops a pending recv at every jitter or report deadline
    async fn fill(&mut self) -> io::Result<()> {
        let filled = loop {
            let Some(timeout_ns) = self.idle_report.as_ref().map(|report| report.timeout_ns) else {
                break self.read_batch().await?;
            };
            let wait_ns = self
                .quiet_since_ns
                .saturating_add(timeout_ns)
                .saturating_sub(g2g_core::metrics::monotonic_ns());
            let wait = core::time::Duration::from_nanos(wait_ns);
            if let Ok(read) = tokio::time::timeout(wait, self.read_batch()).await {
                break read?;
            }
            if let Some(report) = &self.idle_report {
                report.bus.try_post(BusMessage::ReceiveTimeout {
                    element: report.element.clone(),
                    timeout_ns,
                });
            }
            self.quiet_since_ns = g2g_core::metrics::monotonic_ns();
        };
        let clocks = ClockReadings::now();
        self.quiet_since_ns = clocks.monotonic_ns;
        for (arrival_ns, meta) in self.arrivals_ns.iter_mut().zip(&self.metas[..filled]) {
            *arrival_ns = clocks.arrival_ns(meta.timestamp);
        }
        self.filled = filled;
        self.cursor = BatchCursor::default();
        Ok(())
    }

    async fn read_batch(&mut self) -> io::Result<usize> {
        let Self {
            socket,
            state,
            buffers,
            metas,
            ..
        } = self;
        socket
            .async_io(Interest::READABLE, || {
                receive_batch(state, socket, buffers, metas)
            })
            .await
    }
}

// quinn-udp reads at most its own BATCH_SIZE per call, so a larger batch takes several
fn receive_batch(
    state: &UdpSocketState,
    socket: &UdpSocket,
    buffers: &mut [Vec<u8>],
    metas: &mut [RecvMeta],
) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buffers.len() {
        let mut slices: Vec<IoSliceMut<'_>> = buffers[filled..]
            .iter_mut()
            .map(|buffer| IoSliceMut::new(buffer))
            .collect();
        match state.recv(UdpSockRef::from(socket), &mut slices, &mut metas[filled..]) {
            Ok(count) => filled += count,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock && filled > 0 => break,
            Err(error) => return Err(error),
        }
    }
    Ok(filled)
}

#[derive(Debug, Clone, Copy)]
struct ClockReadings {
    monotonic_ns: u64,
    unix_ns: i64,
}

impl ClockReadings {
    fn now() -> Self {
        Self {
            monotonic_ns: g2g_core::metrics::monotonic_ns(),
            unix_ns: g2g_core::metrics::unix_time_ns(),
        }
    }

    // kernel receive timestamps are wall-clock time, so a packet's age moves it onto the monotonic clock
    fn arrival_ns(self, kernel_timestamp: Option<core::time::Duration>) -> u64 {
        let Some(timestamp) = kernel_timestamp else {
            return self.monotonic_ns;
        };
        let timestamp_ns = i64::try_from(timestamp.as_nanos()).unwrap_or(i64::MAX);
        let age_ns = u64::try_from(self.unix_ns.saturating_sub(timestamp_ns)).unwrap_or(0);
        self.monotonic_ns.saturating_sub(age_ns)
    }
}

// quinn-udp turns UDP_GRO on for every Linux socket it wraps
#[cfg(any(target_os = "linux", target_os = "android"))]
fn set_gro(_state: &UdpSocketState, socket: &UdpSocket, enabled: bool) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    if enabled {
        return Ok(());
    }
    let off: libc::c_int = 0;
    // SAFETY: `off` is a live c_int and the length passed is its size, the call only reads it.
    let result = unsafe {
        libc::setsockopt(
            socket.as_raw_fd(),
            libc::SOL_UDP,
            libc::UDP_GRO,
            (&raw const off).cast(),
            core::mem::size_of::<libc::c_int>() as libc::socklen_t,
        )
    };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

// quinn-udp leaves receive coalescing off on Windows
#[cfg(windows)]
fn set_gro(state: &UdpSocketState, socket: &UdpSocket, enabled: bool) -> io::Result<()> {
    state.set_gro(UdpSockRef::from(socket), enabled)
}

#[cfg(not(any(target_os = "linux", target_os = "android", windows)))]
fn set_gro(_state: &UdpSocketState, _socket: &UdpSocket, _enabled: bool) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECEIVE_DEADLINE: core::time::Duration = core::time::Duration::from_secs(5);

    fn receiver_on(address: &str, config: UdpReceiveConfig) -> UdpReceiver {
        let socket = std::net::UdpSocket::bind(address).unwrap();
        socket.set_nonblocking(true).unwrap();
        UdpReceiver::new(UdpSocket::from_std(socket).unwrap(), config).unwrap()
    }

    async fn next_payload(receiver: &mut UdpReceiver) -> Vec<u8> {
        tokio::time::timeout(RECEIVE_DEADLINE, receiver.recv())
            .await
            .expect("a datagram arrives")
            .unwrap()
            .payload
            .to_vec()
    }

    #[test]
    fn kernel_timestamps_move_onto_the_monotonic_clock_by_their_age() {
        let clocks = ClockReadings {
            monotonic_ns: 10_000_000_000,
            unix_ns: 1_700_000_000_000_000_000,
        };
        let age_ns = 5_000_000;
        let received = core::time::Duration::from_nanos((clocks.unix_ns - age_ns) as u64);
        assert_eq!(
            clocks.arrival_ns(Some(received)),
            clocks.monotonic_ns - age_ns as u64
        );
        assert_eq!(clocks.arrival_ns(None), clocks.monotonic_ns);
        let ahead = core::time::Duration::from_nanos((clocks.unix_ns + age_ns) as u64);
        assert_eq!(clocks.arrival_ns(Some(ahead)), clocks.monotonic_ns);
    }

    #[tokio::test]
    async fn skip_first_bytes_drops_each_header() {
        let header = b"hdr";
        let mut receiver = receiver_on(
            "127.0.0.1:0",
            UdpReceiveConfig {
                skip_first_bytes: header.len(),
                ..UdpReceiveConfig::default()
            },
        );
        let destination = receiver.socket().local_addr().unwrap();
        let sender = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        sender.send_to(b"hdrpayload", destination).unwrap();
        assert_eq!(next_payload(&mut receiver).await, b"payload");
    }

    // linux routes all of 127.0.0.0/8 to loopback, so a second sender address needs no setup
    #[cfg(target_os = "linux")]
    async fn filtered_payloads(exclusive: bool) -> [Vec<u8>; 2] {
        let kept_sender: IpAddr = "127.0.0.2".parse().unwrap();
        let mut receiver = receiver_on(
            "0.0.0.0:0",
            UdpReceiveConfig {
                source_filter: alloc::vec![kept_sender],
                source_filter_exclusive: exclusive,
                ..UdpReceiveConfig::default()
            },
        );
        let port = receiver.socket().local_addr().unwrap().port();
        let destination = SocketAddr::from(([127, 0, 0, 1], port));
        let first = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let second = std::net::UdpSocket::bind((kept_sender, 0)).unwrap();
        for round in [b"1", b"2"] {
            first.send_to(&[b'a', round[0]], destination).unwrap();
            second.send_to(&[b'b', round[0]], destination).unwrap();
        }
        [
            next_payload(&mut receiver).await,
            next_payload(&mut receiver).await,
        ]
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn source_filter_keeps_only_the_listed_senders() {
        assert_eq!(
            filtered_payloads(false).await,
            [b"b1".to_vec(), b"b2".to_vec()]
        );
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn exclusive_source_filter_drops_the_listed_senders() {
        assert_eq!(
            filtered_payloads(true).await,
            [b"a1".to_vec(), b"a2".to_vec()]
        );
    }

    // the RTP loop drops a pending recv at every jitter or report deadline, shorter than the timeout here
    #[cfg(feature = "udp-ingress")]
    #[tokio::test]
    async fn the_receive_timeout_survives_dropped_receives() {
        const TIMEOUT_NS: u64 = 60_000_000;
        const RECEIVE_SLICE: core::time::Duration = core::time::Duration::from_millis(20);
        const WATCHED_FOR: core::time::Duration = core::time::Duration::from_millis(200);
        let (bus, handle) = g2g_core::bus::Bus::new(8);
        let mut receiver = receiver_on("127.0.0.1:0", UdpReceiveConfig::default())
            .with_idle_report(IdleReport {
                timeout_ns: TIMEOUT_NS,
                bus: handle,
                element: String::from("udpsrc0"),
            });
        let watch_until = tokio::time::Instant::now() + WATCHED_FOR;
        while tokio::time::Instant::now() < watch_until {
            let _ = tokio::time::timeout(RECEIVE_SLICE, receiver.recv()).await;
        }
        assert_eq!(
            bus.try_recv(),
            Some(BusMessage::ReceiveTimeout {
                element: String::from("udpsrc0"),
                timeout_ns: TIMEOUT_NS,
            })
        );
    }

    const SEGMENT_LEN: usize = 100;
    const SEGMENTS: usize = 3;

    // linux loopback hands a GSO send to a UDP_GRO socket as one buffer instead of splitting it
    #[cfg(target_os = "linux")]
    fn send_one_gso_buffer(destination: SocketAddr) -> Vec<u8> {
        let sender = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let state = UdpSocketState::new(UdpSockRef::from(&sender)).unwrap();
        let contents: Vec<u8> = (0..SEGMENT_LEN * SEGMENTS).map(|i| i as u8).collect();
        let transmit = quinn_udp::Transmit {
            destination,
            ecn: None,
            contents: &contents,
            segment_size: Some(SEGMENT_LEN),
            src_ip: None,
        };
        state
            .try_send(UdpSockRef::from(&sender), &transmit)
            .unwrap();
        contents
    }

    #[cfg(target_os = "linux")]
    async fn received_segments(config: UdpReceiveConfig) -> (Vec<Vec<u8>>, Vec<u8>) {
        let mut receiver = receiver_on("127.0.0.1:0", config);
        let sent = send_one_gso_buffer(receiver.socket().local_addr().unwrap());
        let mut segments = alloc::vec![next_payload(&mut receiver).await];
        for _ in 1..SEGMENTS {
            segments.push(next_payload(&mut receiver).await);
        }
        (segments, sent)
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn a_gro_buffer_splits_at_the_stride() {
        let (segments, sent) = received_segments(UdpReceiveConfig {
            allow_gro: true,
            ..UdpReceiveConfig::default()
        })
        .await;
        let expected: Vec<Vec<u8>> = sent.chunks(SEGMENT_LEN).map(<[u8]>::to_vec).collect();
        assert_eq!(segments, expected);
    }

    // with GRO left on, the merged buffer would be cut to one mtu and the other segments lost
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn gro_off_receives_each_segment_into_an_mtu_buffer() {
        let (segments, sent) = received_segments(UdpReceiveConfig {
            mtu: SEGMENT_LEN,
            ..UdpReceiveConfig::default()
        })
        .await;
        let expected: Vec<Vec<u8>> = sent.chunks(SEGMENT_LEN).map(<[u8]>::to_vec).collect();
        assert_eq!(segments, expected);
    }
}
