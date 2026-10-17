//! Live statistics of a WebRTC element's PeerConnection, the str0m
//! `PeerStats` / `MediaIngressStats` / `MediaEgressStats` events folded into one
//! snapshot. An element hands out a [`WebRtcStatsHandle`] before its run so the
//! application reads the snapshot while the session is live (the element itself
//! is borrowed by its run future), the analogue of gst webrtcsink's `stats`.

use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::Mutex;
use std::time::Duration;

use str0m::stats::{MediaEgressStats, MediaIngressStats, PeerStats};
use str0m::Event;

/// How often str0m emits the stats events the handle folds.
pub(crate) const STATS_INTERVAL: Duration = Duration::from_millis(500);

/// Whole-connection counters.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PeerSnapshot {
    /// Bytes received over the connection, RTP and RTCP.
    pub bytes_rx: u64,
    /// Bytes sent over the connection, RTP and RTCP.
    pub bytes_tx: u64,
    /// Round-trip time of the selected candidate pair, when measured.
    pub rtt: Option<Duration>,
    /// Loss fraction the peer reports for what we send.
    pub egress_loss: Option<f32>,
    /// Loss fraction we measure on what the peer sends.
    pub ingress_loss: Option<f32>,
    /// Current send-side bandwidth estimate, when congestion control runs.
    pub bwe_bps: Option<u64>,
}

/// One m-line's counters in one direction.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MediaSnapshot {
    /// The m-line, as `a=mid:` names it.
    pub mid: String,
    /// The simulcast layer, when the m-line carries layers.
    pub rid: Option<String>,
    pub bytes: u64,
    pub packets: u64,
    /// Keyframe requests (FIR) seen in this direction.
    pub firs: u64,
    /// Keyframe requests (PLI) seen in this direction.
    pub plis: u64,
    /// Retransmission requests seen in this direction.
    pub nacks: u64,
    /// Interarrival jitter in RTP clock ticks, receive side only.
    pub jitter: Option<u32>,
    /// Loss fraction over the last interval, receive side only.
    pub loss: Option<f32>,
    pub rtt: Option<Duration>,
}

/// Everything the handle has folded so far.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WebRtcStats {
    pub peer: Option<PeerSnapshot>,
    /// What the peer sends us, one entry per m-line (and layer).
    pub ingress: Vec<MediaSnapshot>,
    /// What we send the peer, one entry per m-line (and layer).
    pub egress: Vec<MediaSnapshot>,
}

impl WebRtcStats {
    /// Packets received across every ingress m-line.
    pub fn packets_received(&self) -> u64 {
        self.ingress.iter().map(|m| m.packets).sum()
    }

    /// Packets sent across every egress m-line.
    pub fn packets_sent(&self) -> u64 {
        self.egress.iter().map(|m| m.packets).sum()
    }
}

/// Shared, clonable view of an element's [`WebRtcStats`].
#[derive(Debug, Clone, Default)]
pub struct WebRtcStatsHandle(Arc<Mutex<WebRtcStats>>);

impl WebRtcStatsHandle {
    /// A copy of the latest snapshot.
    pub fn snapshot(&self) -> WebRtcStats {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Fold a str0m event into the snapshot. Non-stats events are ignored.
    pub(crate) fn record(&self, event: &Event) {
        match event {
            Event::PeerStats(p) => self.update(|s| s.peer = Some(peer_snapshot(p))),
            Event::MediaIngressStats(m) => {
                self.update(|s| upsert(&mut s.ingress, ingress_snapshot(m)))
            }
            Event::MediaEgressStats(m) => {
                self.update(|s| upsert(&mut s.egress, egress_snapshot(m)))
            }
            _ => {}
        }
    }

    fn update(&self, apply: impl FnOnce(&mut WebRtcStats)) {
        apply(&mut self.0.lock().unwrap_or_else(|e| e.into_inner()));
    }
}

fn upsert(list: &mut Vec<MediaSnapshot>, entry: MediaSnapshot) {
    match list
        .iter_mut()
        .find(|m| m.mid == entry.mid && m.rid == entry.rid)
    {
        Some(existing) => *existing = entry,
        None => list.push(entry),
    }
}

fn peer_snapshot(p: &PeerStats) -> PeerSnapshot {
    PeerSnapshot {
        bytes_rx: p.peer_bytes_rx,
        bytes_tx: p.peer_bytes_tx,
        rtt: p.rtt,
        egress_loss: p.egress_loss_fraction,
        ingress_loss: p.ingress_loss_fraction,
        bwe_bps: p.bwe_tx.map(|b| b.as_u64()),
    }
}

fn ingress_snapshot(m: &MediaIngressStats) -> MediaSnapshot {
    MediaSnapshot {
        mid: m.mid.to_string(),
        rid: m.rid.map(|r| r.to_string()),
        bytes: m.bytes,
        packets: m.packets,
        firs: m.firs,
        plis: m.plis,
        nacks: m.nacks,
        jitter: Some(m.jitter),
        loss: m.loss,
        rtt: m.rtt,
    }
}

fn egress_snapshot(m: &MediaEgressStats) -> MediaSnapshot {
    MediaSnapshot {
        mid: m.mid.to_string(),
        rid: m.rid.map(|r| r.to_string()),
        bytes: m.bytes,
        packets: m.packets,
        firs: m.firs,
        plis: m.plis,
        nacks: m.nacks,
        jitter: None,
        loss: None,
        rtt: m.rtt,
    }
}
