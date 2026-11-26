// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::net::{Ipv4Addr, SocketAddr, TcpListener as StdTcpListener};

use tokio::sync::{mpsc, Mutex};

use g2g_core::wire::encode_packet;
use g2g_core::{
    Caps, G2gError, NetworkFailure, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::remoteclient::{PacketClient, RemoteClient};
use crate::remotesource::{leading_caps, PacketTransport, RemoteSource, TransportFuture};
use crate::remotewire::map_wire;

pub(crate) const MAIN_STREAM: u8 = 0;
pub(crate) const TRACK_STREAM: u8 = 1;

const QUEUED_TRACK_PACKETS: usize = 4;
// packets that wait while the graph behind the track is stalled, 2.56 s of 10 ms audio frames
const RECEIVED_TRACK_PACKETS: usize = 256;
// gst rtpjitterbuffer's latency default, in milliseconds
const DEFAULT_LATENCY_MS: u64 = 200;
const NANOS_PER_MILLISECOND: u64 = 1_000_000;
// the fastest delivery in this window anchors release times, so they follow drift between the two clocks
const FASTEST_DELIVERY_WINDOW_NS: u64 = 2_000_000_000;
// tokio timers and a push downstream delay a release by a few milliseconds
const RELEASE_LATENESS_KEPT_NS: u64 = 20_000_000;

pub(crate) type SharedTrackOutgoing = Arc<Mutex<mpsc::Receiver<Vec<u8>>>>;

fn track_closed() -> G2gError {
    G2gError::Network(NetworkFailure::Stream)
}

// audio frames decode on their own, so a burst after a stall is dropped instead of delaying playback
fn queue_track_frame<T>(queue: &mpsc::Sender<T>, frame: T) -> Result<(), G2gError> {
    match queue.try_send(frame) {
        Ok(()) | Err(mpsc::error::TrySendError::Full(_)) => Ok(()),
        Err(mpsc::error::TrySendError::Closed(_)) => Err(track_closed()),
    }
}

pub type RemoteWtTrackSink = RemoteClient<TrackClient>;

pub(crate) fn track_sink() -> (RemoteWtTrackSink, SharedTrackOutgoing) {
    let (outgoing, transport) = mpsc::channel(QUEUED_TRACK_PACKETS);
    let sink = RemoteClient::from_transport(TrackClient { outgoing });
    (sink, Arc::new(Mutex::new(transport)))
}

#[derive(Debug)]
pub struct TrackClient {
    outgoing: mpsc::Sender<Vec<u8>>,
}

impl PacketClient for TrackClient {
    const NAME: &'static str = "Remote WebTransport track sink";
    const DESCRIPTION: &'static str =
        "Sends a second packet stream on the connection of a RemoteWtSink";
    const PROPERTIES: &'static [PropertySpec] = &[];

    fn is_connected(&self) -> bool {
        true
    }

    fn connect(&mut self) -> TransportFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }

    fn send<'a>(&'a mut self, packet: &'a PipelinePacket) -> TransportFuture<'a, ()> {
        Box::pin(async move {
            let body = encode_packet(packet).map_err(map_wire)?;
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                return queue_track_frame(&self.outgoing, body);
            }
            self.outgoing.send(body).await.map_err(|_| track_closed())
        })
    }

    fn reset(&mut self) {}

    fn close(&mut self) -> TransportFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }

    fn configure_connect(&mut self, _eager: bool) -> Result<(), G2gError> {
        Ok(())
    }

    fn set_transport_prop(
        &mut self,
        _name: &str,
        _value: &PropValue,
    ) -> Option<Result<(), PropError>> {
        None
    }

    fn get_transport_prop(&self, _name: &str) -> Option<PropValue> {
        None
    }
}

pub type RemoteWtTrackSrc = RemoteSource<TrackTransport>;

pub(crate) fn track_source() -> (RemoteWtTrackSrc, mpsc::Sender<TrackArrival>) {
    let (incoming, receiver) = mpsc::channel(RECEIVED_TRACK_PACKETS);
    // the track has no socket of its own, it arrives on the RemoteWtSrc connection
    let mut source: RemoteWtTrackSrc =
        RemoteSource::new(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)));
    source.transport_mut().incoming = Some(receiver);
    (source, incoming)
}

#[derive(Debug)]
pub(crate) struct TrackArrival {
    packet: PipelinePacket,
    arrived_ns: u64,
}

pub(crate) async fn deliver_track_packet(
    track: &mpsc::Sender<TrackArrival>,
    packet: PipelinePacket,
) -> Result<(), G2gError> {
    let is_frame = matches!(packet, PipelinePacket::DataFrame(_));
    let arrival = TrackArrival {
        packet,
        arrived_ns: g2g_core::metrics::monotonic_ns(),
    };
    if is_frame {
        return queue_track_frame(track, arrival);
    }
    track.send(arrival).await.map_err(|_| track_closed())
}

#[derive(Debug, Clone, Copy)]
struct Delivery {
    arrived_ns: u64,
    arrival_minus_pts_ns: i128,
}

#[derive(Debug)]
struct Waiting {
    packet: PipelinePacket,
    release_ns: u64,
}

#[derive(Debug)]
pub(crate) struct JitterQueue {
    latency_ns: u64,
    fastest_in_window: VecDeque<Delivery>,
    waiting: VecDeque<Waiting>,
}

impl JitterQueue {
    pub(crate) fn new(latency_ns: u64) -> Self {
        Self {
            latency_ns,
            fastest_in_window: VecDeque::new(),
            waiting: VecDeque::new(),
        }
    }

    fn fastest_arrival_minus_pts_ns(&mut self, delivery: Delivery) -> i128 {
        while self
            .fastest_in_window
            .back()
            .is_some_and(|kept| kept.arrival_minus_pts_ns >= delivery.arrival_minus_pts_ns)
        {
            self.fastest_in_window.pop_back();
        }
        self.fastest_in_window.push_back(delivery);
        let window_start_ns = delivery
            .arrived_ns
            .saturating_sub(FASTEST_DELIVERY_WINDOW_NS);
        while self
            .fastest_in_window
            .front()
            .is_some_and(|kept| kept.arrived_ns < window_start_ns)
        {
            self.fastest_in_window.pop_front();
        }
        self.fastest_in_window[0].arrival_minus_pts_ns
    }

    pub(crate) fn push(&mut self, arrival: TrackArrival) {
        let TrackArrival { packet, arrived_ns } = arrival;
        let PipelinePacket::DataFrame(frame) = &packet else {
            // a new sender's pts start over, and deliveries from the old sender put every new frame past its release time
            if matches!(packet, PipelinePacket::CapsChanged(_)) {
                self.fastest_in_window.clear();
            }
            self.waiting.push_back(Waiting {
                packet,
                release_ns: arrived_ns,
            });
            return;
        };
        let pts_ns = i128::from(frame.timing.pts_ns);
        let fastest_arrival_minus_pts_ns = self.fastest_arrival_minus_pts_ns(Delivery {
            arrived_ns,
            arrival_minus_pts_ns: i128::from(arrived_ns) - pts_ns,
        });
        let release_ns = pts_ns + fastest_arrival_minus_pts_ns + i128::from(self.latency_ns);
        if i128::from(arrived_ns) > release_ns {
            return;
        }
        self.waiting.push_back(Waiting {
            packet,
            release_ns: u64::try_from(release_ns)
                .expect("a release time is at or after its arrival"),
        });
    }

    pub(crate) fn next_release_ns(&self) -> Option<u64> {
        self.waiting.front().map(|waiting| waiting.release_ns)
    }

    pub(crate) fn pop_released(&mut self, now_ns: u64) -> Option<PipelinePacket> {
        loop {
            let waiting = self.waiting.front()?;
            if waiting.release_ns > now_ns {
                return None;
            }
            let missed_slot = matches!(waiting.packet, PipelinePacket::DataFrame(_))
                && now_ns - waiting.release_ns > RELEASE_LATENESS_KEPT_NS;
            let waiting = self.waiting.pop_front()?;
            if !missed_slot {
                return Some(waiting.packet);
            }
        }
    }
}

#[derive(Debug)]
pub struct TrackConnection {
    incoming: mpsc::Receiver<TrackArrival>,
    jitter: JitterQueue,
}

#[derive(Debug)]
pub struct TrackListener {
    incoming: Option<mpsc::Receiver<TrackArrival>>,
    latency_ns: u64,
}

#[derive(Debug)]
pub struct TrackTransport {
    incoming: Option<mpsc::Receiver<TrackArrival>>,
    latency_ns: u64,
}

impl Default for TrackTransport {
    fn default() -> Self {
        Self {
            incoming: None,
            latency_ns: DEFAULT_LATENCY_MS * NANOS_PER_MILLISECOND,
        }
    }
}

static TRACK_SOURCE_PROPERTIES: &[PropertySpec] = &[PropertySpec::new(
    "latency",
    PropKind::Uint,
    "milliseconds a frame may arrive after the fastest recent one before it is dropped",
)
.with_default("200")];

impl PacketTransport for TrackTransport {
    type Conn = TrackConnection;
    type Listener = TrackListener;
    const NAME: &'static str = "Remote WebTransport track source";
    const DESCRIPTION: &'static str =
        "Receives the second packet stream on the connection of a RemoteWtSrc";
    const PROPERTIES: &'static [PropertySpec] = TRACK_SOURCE_PROPERTIES;

    fn listen(
        &mut self,
        _bind: SocketAddr,
        _adopt: Option<StdTcpListener>,
    ) -> TransportFuture<'_, Self::Listener> {
        Box::pin(async move {
            Ok(TrackListener {
                incoming: self.incoming.take(),
                latency_ns: self.latency_ns,
            })
        })
    }

    fn listen_addr(_listener: &Self::Listener) -> Option<SocketAddr> {
        None
    }

    fn accept(listener: &mut Self::Listener) -> TransportFuture<'_, (Self::Conn, Caps)> {
        Box::pin(async move {
            let mut incoming = listener.incoming.take().ok_or(G2gError::NotConfigured)?;
            let caps = leading_caps(incoming.recv().await.map(|arrival| arrival.packet))?;
            let jitter = JitterQueue::new(listener.latency_ns);
            Ok((TrackConnection { incoming, jitter }, caps))
        })
    }

    fn recv(conn: &mut Self::Conn) -> TransportFuture<'_, Option<PipelinePacket>> {
        Box::pin(async move {
            loop {
                if let Some(packet) = conn.jitter.pop_released(g2g_core::metrics::monotonic_ns()) {
                    return Ok(Some(packet));
                }
                let Some(release_ns) = conn.jitter.next_release_ns() else {
                    match conn.incoming.recv().await {
                        Some(arrival) => conn.jitter.push(arrival),
                        None => return Ok(None),
                    }
                    continue;
                };
                let wait_ns = release_ns.saturating_sub(g2g_core::metrics::monotonic_ns());
                tokio::select! {
                    arrival = conn.incoming.recv() => match arrival {
                        Some(arrival) => conn.jitter.push(arrival),
                        // packets already held still play out before the end
                        None => tokio::time::sleep(core::time::Duration::from_nanos(wait_ns)).await,
                    },
                    () = tokio::time::sleep(core::time::Duration::from_nanos(wait_ns)) => {}
                }
            }
        })
    }

    fn set_transport_prop(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Option<Result<(), PropError>> {
        if name != "latency" {
            return None;
        }
        Some(match value.as_uint() {
            Some(milliseconds) => {
                self.latency_ns = milliseconds.saturating_mul(NANOS_PER_MILLISECOND);
                Ok(())
            }
            None => Err(PropError::Type),
        })
    }

    fn get_transport_prop(&self, name: &str) -> Option<PropValue> {
        (name == "latency").then_some(PropValue::Uint(self.latency_ns / NANOS_PER_MILLISECOND))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use g2g_core::frame::Frame;

    fn frame(sequence: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::from_system_bytes(
            vec![0u8; 4],
            Default::default(),
            sequence,
        ))
    }

    fn sequence(packet: Option<PipelinePacket>) -> Option<u64> {
        match packet {
            Some(PipelinePacket::DataFrame(frame)) => Some(frame.sequence),
            _ => None,
        }
    }

    #[tokio::test]
    async fn a_full_track_drops_frames_and_waits_for_room_for_control_packets() {
        let (track, mut received) = mpsc::channel(1);
        deliver_track_packet(&track, frame(0))
            .await
            .expect("queued");
        deliver_track_packet(&track, frame(1))
            .await
            .expect("dropped");
        assert_eq!(
            sequence(received.recv().await.map(|arrival| arrival.packet)),
            Some(0)
        );
        assert!(received.try_recv().is_err(), "the second frame was dropped");

        deliver_track_packet(&track, frame(2))
            .await
            .expect("queued");
        let eos =
            tokio::spawn(async move { deliver_track_packet(&track, PipelinePacket::Eos).await });
        assert_eq!(
            sequence(received.recv().await.map(|arrival| arrival.packet)),
            Some(2)
        );
        assert!(matches!(
            received.recv().await.map(|arrival| arrival.packet),
            Some(PipelinePacket::Eos)
        ));
        eos.await.expect("joined").expect("Eos waited for room");
    }

    const LATENCY_MS: u64 = 50;

    fn ms(milliseconds: u64) -> u64 {
        milliseconds * NANOS_PER_MILLISECOND
    }

    fn arrival_at(sequence: u64, pts_ms: u64, arrived_ms: u64) -> TrackArrival {
        let PipelinePacket::DataFrame(mut frame) = frame(sequence) else {
            unreachable!("frame builds a data frame");
        };
        frame.timing.pts_ns = ms(pts_ms);
        TrackArrival {
            packet: PipelinePacket::DataFrame(frame),
            arrived_ns: ms(arrived_ms),
        }
    }

    #[test]
    fn frames_later_than_the_latency_are_dropped_and_the_rest_keep_their_pts_spacing() {
        let mut jitter = JitterQueue::new(ms(LATENCY_MS));
        jitter.push(arrival_at(0, 0, 1000));
        jitter.push(arrival_at(1, 10, 1040));
        jitter.push(arrival_at(2, 20, 1080));
        jitter.push(arrival_at(3, 30, 1075));

        assert_eq!(
            jitter.pop_released(ms(1049)).map(|_| ()),
            None,
            "frame 0 waits for its slot"
        );
        assert_eq!(sequence(jitter.pop_released(ms(1050))), Some(0));
        assert_eq!(
            jitter.next_release_ns(),
            Some(ms(1060)),
            "30 ms late is within the latency"
        );
        assert_eq!(sequence(jitter.pop_released(ms(1060))), Some(1));
        assert_eq!(
            jitter.next_release_ns(),
            Some(ms(1080)),
            "frame 2 arrived 60 ms late and was dropped"
        );
        assert_eq!(sequence(jitter.pop_released(ms(1080))), Some(3));
        assert_eq!(jitter.next_release_ns(), None);
    }

    #[test]
    fn frames_a_stalled_reader_leaves_waiting_past_their_slot_are_dropped() {
        let mut jitter = JitterQueue::new(ms(LATENCY_MS));
        for sequence in 0..10 {
            jitter.push(arrival_at(sequence, sequence * 10, 1000 + sequence * 10));
        }
        // frame k is due at 1050 + 10k, so a reader back at 1145 finds frames 0 to 7 more than 20 ms past due
        assert_eq!(sequence(jitter.pop_released(ms(1145))), Some(8));
    }

    #[test]
    fn the_release_timeline_follows_a_delay_that_grows_for_longer_than_the_window() {
        let mut jitter = JitterQueue::new(ms(LATENCY_MS));
        jitter.push(arrival_at(0, 0, 1000));
        jitter.pop_released(ms(1050));
        // 100 ms slower than the first frame, which left the window 2 s ago
        jitter.push(arrival_at(1, 3000, 4100));
        assert_eq!(jitter.next_release_ns(), Some(ms(4150)));
    }

    #[test]
    fn new_caps_restart_the_timeline_for_a_sender_whose_pts_start_over() {
        let mut jitter = JitterQueue::new(ms(LATENCY_MS));
        jitter.push(arrival_at(0, 5000, 1000));
        jitter.push(TrackArrival {
            packet: PipelinePacket::CapsChanged(Caps::Klv),
            arrived_ns: ms(1005),
        });
        jitter.push(arrival_at(1, 0, 1010));

        assert_eq!(sequence(jitter.pop_released(ms(1050))), Some(0));
        assert!(matches!(
            jitter.pop_released(ms(1050)),
            Some(PipelinePacket::CapsChanged(_))
        ));
        assert_eq!(sequence(jitter.pop_released(ms(1060))), Some(1));
    }
}
