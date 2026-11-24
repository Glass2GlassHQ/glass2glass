// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::net::{Ipv4Addr, SocketAddr, TcpListener as StdTcpListener};

use tokio::sync::{mpsc, Mutex};

use g2g_core::wire::encode_packet;
use g2g_core::{
    Caps, G2gError, NetworkFailure, PipelinePacket, PropError, PropValue, PropertySpec,
};

use crate::remoteclient::{PacketClient, RemoteClient};
use crate::remotesource::{leading_caps, PacketTransport, RemoteSource, TransportFuture};
use crate::remotewire::map_wire;

pub(crate) const MAIN_STREAM: u8 = 0;
pub(crate) const TRACK_STREAM: u8 = 1;

const QUEUED_TRACK_PACKETS: usize = 4;
const RECEIVED_TRACK_PACKETS: usize = 2;

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

pub(crate) fn track_source() -> (RemoteWtTrackSrc, mpsc::Sender<PipelinePacket>) {
    let (incoming, receiver) = mpsc::channel(RECEIVED_TRACK_PACKETS);
    // the track has no socket of its own, it arrives on the RemoteWtSrc connection
    let mut source: RemoteWtTrackSrc =
        RemoteSource::new(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)));
    source.transport_mut().incoming = Some(receiver);
    (source, incoming)
}

pub(crate) async fn deliver_track_packet(
    track: &mpsc::Sender<PipelinePacket>,
    packet: PipelinePacket,
) -> Result<(), G2gError> {
    if matches!(packet, PipelinePacket::DataFrame(_)) {
        return queue_track_frame(track, packet);
    }
    track.send(packet).await.map_err(|_| track_closed())
}

#[derive(Debug, Default)]
pub struct TrackTransport {
    incoming: Option<mpsc::Receiver<PipelinePacket>>,
}

impl PacketTransport for TrackTransport {
    type Conn = mpsc::Receiver<PipelinePacket>;
    type Listener = Option<mpsc::Receiver<PipelinePacket>>;
    const NAME: &'static str = "Remote WebTransport track source";
    const DESCRIPTION: &'static str =
        "Receives the second packet stream on the connection of a RemoteWtSrc";
    const PROPERTIES: &'static [PropertySpec] = &[];

    fn listen(
        &mut self,
        _bind: SocketAddr,
        _adopt: Option<StdTcpListener>,
    ) -> TransportFuture<'_, Self::Listener> {
        Box::pin(async move { Ok(self.incoming.take()) })
    }

    fn listen_addr(_listener: &Self::Listener) -> Option<SocketAddr> {
        None
    }

    fn accept(listener: &mut Self::Listener) -> TransportFuture<'_, (Self::Conn, Caps)> {
        Box::pin(async move {
            let mut incoming = listener.take().ok_or(G2gError::NotConfigured)?;
            let caps = leading_caps(incoming.recv().await)?;
            Ok((incoming, caps))
        })
    }

    fn recv(conn: &mut Self::Conn) -> TransportFuture<'_, Option<PipelinePacket>> {
        Box::pin(async move { Ok(conn.recv().await) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::MemoryDomain;

    fn frame(sequence: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; 4].into_boxed_slice())),
            timing: Default::default(),
            sequence,
            meta: Default::default(),
        })
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
        assert_eq!(sequence(received.recv().await), Some(0));
        assert!(received.try_recv().is_err(), "the second frame was dropped");

        deliver_track_packet(&track, frame(2))
            .await
            .expect("queued");
        let eos =
            tokio::spawn(async move { deliver_track_packet(&track, PipelinePacket::Eos).await });
        assert_eq!(sequence(received.recv().await), Some(2));
        assert!(matches!(received.recv().await, Some(PipelinePacket::Eos)));
        eos.await.expect("joined").expect("Eos waited for room");
    }
}
