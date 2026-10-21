// Helpers the RIST test binaries (`m1217_rist`, `m1734_rist_bonding`) share,
// included per binary with `mod rist_common;`.

use std::collections::BTreeSet;
use std::future::Future;
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::rtp::RtpHeader;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket,
};

use g2g_plugins::bytestream::{
    mpegts_caps, TS_DATAGRAM_PAYLOAD, TS_PACKETS_PER_DATAGRAM, TS_PACKET_SIZE,
};
use g2g_plugins::rist::{self, MAX_DATAGRAM, RIST_APP_NAME, SUBTYPE_RANGE_NACK};
use g2g_plugins::rtcp::{self, RtcpPacket};

/// Gap between two packets of a paced sender: wide enough that a scattered
/// loss is seen alone.
pub(crate) const SEND_INTERVAL: Duration = Duration::from_millis(2);

pub(crate) const GST_DEADLINE: Duration = Duration::from_secs(60);

/// How long a peer receiver (GStreamer, librist) gets to bind before g2g sends.
pub(crate) const GST_BIND_WAIT: Duration = Duration::from_millis(1000);

/// How often a relay thread wakes to check whether the test is over.
pub(crate) const RELAY_POLL: Duration = Duration::from_millis(20);

pub(crate) struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

pub(crate) fn output_path(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(name);
    let _ = std::fs::remove_file(&path);
    path
}

/// TS packets with the sync byte and a byte pattern that differs per packet, so
/// a lost or reordered one changes the received bytes.
pub(crate) fn synthetic_ts(rtp_packets: usize) -> Vec<u8> {
    const SYNC_BYTE: u8 = 0x47;
    let packet_count = rtp_packets * TS_PACKETS_PER_DATAGRAM;
    let mut stream = Vec::with_capacity(packet_count * TS_PACKET_SIZE);
    for packet in 0..packet_count {
        stream.push(SYNC_BYTE);
        stream.extend((1..TS_PACKET_SIZE).map(|offset| (packet * 31 + offset) as u8));
    }
    stream
}

/// A source that hands out one RTP payload's worth of TS per frame, spaced
/// [`SEND_INTERVAL`] apart like a live encoder.
#[derive(Debug)]
pub(crate) struct PacedTsSource {
    pub(crate) stream: Vec<u8>,
}

impl SourceLoop for PacedTsSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(mpegts_caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(mpegts_caps()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut emitted = 0u64;
            for chunk in self.stream.chunks(TS_DATAGRAM_PAYLOAD) {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        chunk.to_vec().into_boxed_slice(),
                    )),
                    timing: FrameTiming {
                        pts_ns: emitted * SEND_INTERVAL.as_nanos() as u64,
                        ..FrameTiming::default()
                    },
                    sequence: emitted,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
                emitted += 1;
                tokio::time::sleep(SEND_INTERVAL).await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(emitted)
        })
    }
}

/// Two adjacent loopback UDP sockets, the first on an even port: a RIST
/// media + RTCP pair.
pub(crate) fn bound_even_port_pair() -> (UdpSocket, UdpSocket) {
    loop {
        let media = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind an ephemeral port");
        let port = media.local_addr().expect("bound address").port();
        if !port.is_multiple_of(2) || port == u16::MAX {
            continue;
        }
        if let Ok(rtcp) = UdpSocket::bind((Ipv4Addr::LOCALHOST, port + 1)) {
            return (media, rtcp);
        }
    }
}

/// An even port with the odd one above it free, released for an element to bind.
pub(crate) fn free_even_port() -> u16 {
    let (media, _rtcp) = bound_even_port_pair();
    media.local_addr().expect("bound address").port()
}

pub(crate) fn loopback(port: u16) -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, port))
}

// A peer process, killed when the test drops it, a panicking one too: a peer
// left running keeps the test's output pipe open, so the run never ends.
pub(crate) struct PeerProcess(pub(crate) Child);

impl Drop for PeerProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

pub(crate) fn spawn_gst(description: &str) -> PeerProcess {
    PeerProcess(
        Command::new("gst-launch-1.0")
            .arg("-q")
            .args(description.split_whitespace())
            .spawn()
            .expect("gst-launch-1.0 is on PATH"),
    )
}

/// Neither GStreamer RIST element lets `gst-launch-1.0` exit on its own: the
/// bins drop EOS. Stop the peer once its output is complete.
pub(crate) fn stop_gst(peer: PeerProcess) {
    drop(peer);
}

/// Wait for a live gst receiver to have written `expected` bytes, then stop it.
pub(crate) fn collect_from_gst(mut child: PeerProcess, path: &Path, expected: usize) -> Vec<u8> {
    let deadline = Instant::now() + GST_DEADLINE;
    while Instant::now() < deadline {
        let written = std::fs::metadata(path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        if written >= expected {
            break;
        }
        assert!(
            child.0.try_wait().expect("poll the gst peer").is_none(),
            "the gst receiver exited early"
        );
        std::thread::sleep(RELAY_POLL);
    }
    stop_gst(child);
    std::fs::read(path).expect("read what gst received")
}

/// What the relay saw cross it.
#[derive(Debug, Default)]
pub(crate) struct RelayLog {
    /// The original packets in arrival order, dropped ones too.
    pub(crate) originals: Vec<Vec<u8>>,
    pub(crate) dropped: Vec<u16>,
    pub(crate) retransmitted: BTreeSet<u16>,
    pub(crate) retransmissions: usize,
    pub(crate) range_nacks: usize,
    pub(crate) generic_nacks: usize,
}

impl RelayLog {
    /// The payloads of the original packets: the byte stream the sender put on
    /// the wire.
    pub(crate) fn sent_payloads(&self) -> Vec<u8> {
        let mut payloads = Vec::new();
        for packet in &self.originals {
            let parsed = RtpHeader::parse(packet).expect("the relay keeps only RTP packets");
            let payload_end = parsed.payload_offset + parsed.payload_len;
            payloads.extend_from_slice(&packet[parsed.payload_offset..payload_end]);
        }
        payloads
    }
}

/// A UDP forwarder between a RIST sender and receiver. It drops the media
/// packets at the given indices once, passes every retransmission and every
/// RTCP packet, and records the NACK forms the receiver sent back.
pub(crate) struct LossyRelay {
    pub(crate) media_port: u16,
    log: Arc<Mutex<RelayLog>>,
    stop: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
}

impl LossyRelay {
    pub(crate) fn start(receiver_media: SocketAddr, dropped_indices: &'static [u16]) -> Self {
        let (media_in, rtcp_in) = bound_even_port_pair();
        let media_port = media_in.local_addr().expect("bound address").port();
        let media_out = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind relay output");
        let rtcp_out = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind relay output");
        for socket in [&media_in, &rtcp_in, &rtcp_out] {
            socket
                .set_read_timeout(Some(RELAY_POLL))
                .expect("read timeout");
        }
        let mut receiver_rtcp = receiver_media;
        receiver_rtcp.set_port(receiver_media.port() + 1);

        let log = Arc::new(Mutex::new(RelayLog::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let sender_rtcp: Arc<Mutex<Option<SocketAddr>>> = Arc::new(Mutex::new(None));
        let mut workers = Vec::new();

        let (worker_log, worker_stop) = (log.clone(), stop.clone());
        workers.push(std::thread::spawn(move || {
            let mut first_original: Option<u16> = None;
            let mut datagram = vec![0u8; MAX_DATAGRAM];
            while !worker_stop.load(Ordering::Relaxed) {
                let Ok((length, _)) = media_in.recv_from(&mut datagram) else {
                    continue;
                };
                let packet = &datagram[..length];
                if let Some(parsed) = RtpHeader::parse(packet) {
                    let header = parsed.header;
                    let mut log = worker_log.lock().unwrap();
                    if rist::is_retransmission(header.ssrc) {
                        log.retransmissions += 1;
                        log.retransmitted.insert(header.sequence);
                    } else {
                        log.originals.push(packet.to_vec());
                        let first = *first_original.get_or_insert(header.sequence);
                        let index = header.sequence.wrapping_sub(first);
                        if dropped_indices.contains(&index)
                            && !log.dropped.contains(&header.sequence)
                        {
                            log.dropped.push(header.sequence);
                            continue;
                        }
                    }
                }
                media_out
                    .send_to(packet, receiver_media)
                    .expect("relay media");
            }
        }));

        let rtcp_in_back = rtcp_in.try_clone().expect("clone relay socket");
        let rtcp_out_back = rtcp_out.try_clone().expect("clone relay socket");
        let (forward_stop, forward_sender) = (stop.clone(), sender_rtcp.clone());
        workers.push(std::thread::spawn(move || {
            let mut datagram = vec![0u8; MAX_DATAGRAM];
            while !forward_stop.load(Ordering::Relaxed) {
                let Ok((length, from)) = rtcp_in.recv_from(&mut datagram) else {
                    continue;
                };
                *forward_sender.lock().unwrap() = Some(from);
                rtcp_out
                    .send_to(&datagram[..length], receiver_rtcp)
                    .expect("relay sender RTCP");
            }
        }));

        let (back_log, back_stop) = (log.clone(), stop.clone());
        workers.push(std::thread::spawn(move || {
            let mut datagram = vec![0u8; MAX_DATAGRAM];
            while !back_stop.load(Ordering::Relaxed) {
                let Ok((length, _)) = rtcp_out_back.recv_from(&mut datagram) else {
                    continue;
                };
                let compound = &datagram[..length];
                {
                    let mut log = back_log.lock().unwrap();
                    for packet in rtcp::parse_compound(compound) {
                        match packet {
                            RtcpPacket::Nack { .. } => log.generic_nacks += 1,
                            RtcpPacket::App {
                                subtype: SUBTYPE_RANGE_NACK,
                                name: RIST_APP_NAME,
                                ..
                            } => log.range_nacks += 1,
                            _ => {}
                        }
                    }
                }
                let sender = *sender_rtcp.lock().unwrap();
                if let Some(sender) = sender {
                    rtcp_in_back
                        .send_to(compound, sender)
                        .expect("relay receiver RTCP");
                }
            }
        }));

        Self {
            media_port,
            log,
            stop,
            workers,
        }
    }

    pub(crate) fn finish(self) -> RelayLog {
        self.stop.store(true, Ordering::Relaxed);
        for worker in self.workers {
            worker.join().expect("relay thread");
        }
        Arc::try_unwrap(self.log)
            .expect("relay threads are done")
            .into_inner()
            .unwrap()
    }
}

/// Every dropped packet came back as a retransmission after the receiver
/// asked for it.
pub(crate) fn assert_recovered_through(log: &RelayLog, dropped: &[u16]) {
    assert_eq!(
        log.dropped.len(),
        dropped.len(),
        "the relay dropped the whole set"
    );
    for sequence in &log.dropped {
        assert!(
            log.retransmitted.contains(sequence),
            "dropped packet {sequence} was retransmitted (SSRC low bit 1)"
        );
    }
    assert!(
        log.range_nacks + log.generic_nacks > 0,
        "the receiver sent NACKs"
    );
}
