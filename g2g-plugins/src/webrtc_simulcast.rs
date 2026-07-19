//! Shared WebRTC send-simulcast machinery for grouped-pad publishing sessions:
//! groups N video layers under one m-line with N send rids, and routes a remote
//! per-`(mid, rid)` keyframe request to exactly the source feeding that layer.
//! str0m-generic on purpose: the same code serves any egress session (LiveKit
//! today via [`crate::livekitsink`], a WHIP grouped-pad session later). The rid
//! naming follows the LiveKit / browser convention so the SFU maps each rid to
//! the right spatial layer; the SFU-specific quality metadata lives in the sink.

use alloc::vec::Vec;

use str0m::media::{Mid, Rid, Simulcast, SimulcastLayer};

use g2g_core::ReverseChannel;

/// Canonical simulcast rids, lowest to highest resolution. Matches the LiveKit /
/// browser convention (`q` = low, `h` = mid, `f` = high), so an SFU that maps rid
/// to spatial layer by this table binds each layer to the right quality.
pub const SIMULCAST_RIDS: [&str; 3] = ["q", "h", "f"];

/// The rids for `n` layers in pad order, highest resolution first (pad 0 is the
/// top layer). E.g. `n = 2` yields `["h", "q"]`, `n = 3` yields `["f", "h", "q"]`.
/// Clamped to the available rids.
pub fn rids_high_to_low(n: usize) -> Vec<&'static str> {
    let n = n.min(SIMULCAST_RIDS.len());
    SIMULCAST_RIDS[..n].iter().rev().copied().collect()
}

/// One configured send layer: its rid plus the resolution read from the pad's
/// fixated caps (carried to the SFU as track metadata, not as SDP restrictions
/// for now).
#[derive(Debug, Clone, Copy)]
pub struct SendLayer {
    pub rid: &'static str,
    pub width: u32,
    pub height: u32,
}

/// Build the str0m send-simulcast description for `layers`, or `None` for fewer
/// than two layers (a single stream is not simulcast and takes no rid). The rids
/// carry no restriction attributes: resolution is announced out-of-band in the
/// SFU's track metadata for this milestone.
pub fn send_simulcast(layers: &[SendLayer]) -> Option<Simulcast> {
    if layers.len() < 2 {
        return None;
    }
    let mut sc = Simulcast::new();
    for l in layers {
        sc.add_send_layer(SimulcastLayer::new(l.rid));
    }
    Some(sc)
}

/// Routes a remote keyframe request to the reverse channel of the source feeding
/// the named layer. Each published `(mid, rid)` maps to one channel; a request
/// with a rid fires only that layer, never a sibling, and a rid-less request (a
/// single non-simulcast stream) matches the mid's rid-less entry.
#[derive(Debug, Default)]
pub struct KeyframeRoutes {
    routes: Vec<(Mid, Option<Rid>, ReverseChannel)>,
}

impl KeyframeRoutes {
    pub fn new() -> Self {
        Self { routes: Vec::new() }
    }

    /// Register the reverse channel of the source feeding `(mid, rid)`.
    pub fn push(&mut self, mid: Mid, rid: Option<Rid>, channel: ReverseChannel) {
        self.routes.push((mid, rid, channel));
    }

    /// Fire the keyframe request for exactly `(mid, rid)`. No-op if nothing
    /// matches (an unknown layer must not force a sibling's encoder).
    pub fn request_keyframe(&self, mid: Mid, rid: Option<Rid>) {
        for (m, r, channel) in &self.routes {
            if *m == mid && *r == rid {
                channel.request_keyframe();
                return;
            }
        }
    }
}

/// Nominal send bitrate for a layer, from its resolution at ~30 fps (the 0.1
/// bits-per-pixel rule of thumb: `w * h * 30 * 0.1`). Only used to budget the
/// aggregate BWE estimate across layers, not to configure any encoder.
pub fn nominal_bps(width: u32, height: u32) -> u64 {
    (width as u64) * (height as u64) * 3
}

/// How long starvation must persist before a layer is dropped.
const DROP_AFTER: core::time::Duration = core::time::Duration::from_secs(2);
/// How long headroom must persist before a dropped layer is restored.
const RESTORE_AFTER: core::time::Duration = core::time::Duration::from_secs(5);
/// Drop when the estimate is below this fraction of the active layers' budget.
const DROP_BELOW: f64 = 0.9;
/// Restore only when the estimate covers the would-be budget with this margin.
const RESTORE_ABOVE: f64 = 1.15;

/// Splits the aggregate BWE estimate across simulcast layers as whole-layer
/// on/off (str0m's estimate is per-connection; per-layer budgeting is the
/// caller's job by design). Under sustained starvation the TOP layer drops
/// first: it costs the most, and subscribers fall back to a lower layer, which
/// matches libwebrtc's behavior (the design doc said lowest-first; that saves
/// almost nothing and starves low-bandwidth subscribers instead). The lowest
/// layer never drops. Hysteresis on both edges so a noisy estimate cannot flap
/// a layer.
#[derive(Debug)]
pub struct LayerAllocator {
    /// Per layer: rid + nominal bps, ordered LOWEST first (drop order is from
    /// the back). `on` count tracks how many of the front layers are active.
    layers: Vec<(Rid, u64)>,
    active: usize,
    /// Estimates are clamped to this before budgeting (0 = no cap): the
    /// `max-send-bitrate` knob, and the deterministic test lever.
    cap: u64,
    starved_since: Option<std::time::Instant>,
    headroom_since: Option<std::time::Instant>,
}

impl LayerAllocator {
    /// `layers`: (rid, width, height) in any order; sorted lowest-resolution
    /// first internally. All layers start active.
    pub fn new(layers: &[SendLayer], cap: u64) -> Self {
        let mut l: Vec<(Rid, u64)> = layers
            .iter()
            .map(|s| (Rid::from(s.rid), nominal_bps(s.width, s.height)))
            .collect();
        l.sort_by_key(|(_, bps)| *bps);
        Self {
            active: l.len(),
            layers: l,
            cap,
            starved_since: None,
            headroom_since: None,
        }
    }

    fn budget(&self, count: usize) -> u64 {
        self.layers[..count].iter().map(|(_, b)| b).sum()
    }

    /// Feed one aggregate estimate. Returns `true` when the active layer set
    /// changed (the caller logs / signals sources on the edge).
    pub fn update(&mut self, now: std::time::Instant, estimate_bps: u64) -> bool {
        let est = if self.cap > 0 {
            estimate_bps.min(self.cap)
        } else {
            estimate_bps
        };
        let needed = self.budget(self.active);
        let starved = self.active > 1 && (est as f64) < (needed as f64) * DROP_BELOW;
        let would_restore = self.active < self.layers.len()
            && (est as f64) >= (self.budget(self.active + 1) as f64) * RESTORE_ABOVE;

        if starved {
            self.headroom_since = None;
            let since = *self.starved_since.get_or_insert(now);
            if now.duration_since(since) >= DROP_AFTER {
                self.active -= 1;
                self.starved_since = None;
                return true;
            }
        } else if would_restore {
            self.starved_since = None;
            let since = *self.headroom_since.get_or_insert(now);
            if now.duration_since(since) >= RESTORE_AFTER {
                self.active += 1;
                self.headroom_since = None;
                return true;
            }
        } else {
            self.starved_since = None;
            self.headroom_since = None;
        }
        false
    }

    /// Whether the layer with `rid` is currently sent.
    pub fn is_on(&self, rid: Rid) -> bool {
        self.layers[..self.active].iter().any(|(r, _)| *r == rid)
    }

    /// The initial BWE target: every layer's nominal budget, capped.
    pub fn initial_bps(&self) -> u64 {
        let all = self.budget(self.layers.len());
        if self.cap > 0 {
            all.min(self.cap)
        } else {
            all
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Instant;

    use str0m::crypto::from_feature_flags;
    use str0m::media::{Direction, MediaKind};
    use str0m::RtcConfig;

    #[test]
    fn rids_are_high_to_low_and_clamped() {
        assert_eq!(rids_high_to_low(2), alloc::vec!["h", "q"]);
        assert_eq!(rids_high_to_low(3), alloc::vec!["f", "h", "q"]);
        // More layers than rids is clamped, never panics.
        assert_eq!(rids_high_to_low(9), alloc::vec!["f", "h", "q"]);
        // A single layer is not simulcast.
        assert!(send_simulcast(&[SendLayer {
            rid: "q",
            width: 320,
            height: 240
        }])
        .is_none());
    }

    #[test]
    fn offer_sdp_carries_both_rids_and_one_simulcast() {
        // Build the offer the grouped-pad session builds: one send-only video
        // m-line grouping two layers, and assert str0m emits an `a=rid:<r> send`
        // per layer plus a single `a=simulcast:send`.
        let layers = [
            SendLayer {
                rid: "h",
                width: 640,
                height: 480,
            },
            SendLayer {
                rid: "q",
                width: 320,
                height: 240,
            },
        ];
        let mut rtc = RtcConfig::new()
            .set_crypto_provider(alloc::sync::Arc::new(from_feature_flags()))
            .clear_codecs()
            .enable_h264(true)
            .build(Instant::now());
        let mut api = rtc.sdp_api();
        api.add_media(
            MediaKind::Video,
            Direction::SendOnly,
            None,
            None,
            send_simulcast(&layers),
        );
        let (offer, _pending) = api.apply().expect("offer applies");
        let sdp = offer.to_sdp_string();

        assert!(sdp.contains("a=rid:h send"), "missing high rid:\n{sdp}");
        assert!(sdp.contains("a=rid:q send"), "missing low rid:\n{sdp}");
        assert_eq!(
            sdp.matches("a=simulcast:send").count(),
            1,
            "expected exactly one a=simulcast:send line:\n{sdp}"
        );
        assert!(
            sdp.contains("a=simulcast:send h;q") || sdp.contains("a=simulcast:send q;h"),
            "simulcast line must list both rids:\n{sdp}"
        );
    }

    #[test]
    fn allocator_drops_top_layer_after_sustained_starvation_and_restores() {
        use std::time::{Duration as D, Instant};
        let layers = [
            SendLayer {
                rid: "h",
                width: 640,
                height: 480,
            },
            SendLayer {
                rid: "q",
                width: 320,
                height: 240,
            },
        ];
        // Budget: 640*480*3 + 320*240*3 = 921600 + 230400 = 1152000 bps.
        let mut a = LayerAllocator::new(&layers, 0);
        assert_eq!(a.initial_bps(), 1_152_000);
        let t0 = Instant::now();
        let (h, q) = (Rid::from("h"), Rid::from("q"));

        // A single starved sample does not drop (hysteresis).
        assert!(!a.update(t0, 400_000));
        assert!(a.is_on(h) && a.is_on(q));
        // Recovery in between resets the window.
        assert!(!a.update(t0 + D::from_secs(1), 2_000_000));
        assert!(!a.update(t0 + D::from_secs(2), 400_000));
        assert!(!a.update(t0 + D::from_secs(3), 400_000));
        // Sustained starvation drops exactly the TOP layer.
        assert!(a.update(t0 + D::from_secs(5), 400_000));
        assert!(!a.is_on(h), "top layer drops first");
        assert!(a.is_on(q), "low layer stays");
        // The last layer never drops, however starved.
        assert!(!a.update(t0 + D::from_secs(6), 10_000));
        assert!(!a.update(t0 + D::from_secs(60), 10_000));
        assert!(a.is_on(q));

        // Restore needs sustained headroom over the FULL budget.
        assert!(!a.update(t0 + D::from_secs(61), 2_000_000));
        assert!(!a.update(t0 + D::from_secs(64), 2_000_000));
        assert!(a.update(t0 + D::from_secs(67), 2_000_000));
        assert!(a.is_on(h) && a.is_on(q));
    }

    #[test]
    fn allocator_cap_bounds_the_estimate() {
        use std::time::{Duration as D, Instant};
        let layers = [
            SendLayer {
                rid: "h",
                width: 640,
                height: 480,
            },
            SendLayer {
                rid: "q",
                width: 320,
                height: 240,
            },
        ];
        // Cap below the full budget: even a huge estimate reads as 400k, so the
        // top layer must drop after the window.
        let mut a = LayerAllocator::new(&layers, 400_000);
        assert_eq!(a.initial_bps(), 400_000);
        let t0 = Instant::now();
        assert!(!a.update(t0, 100_000_000));
        assert!(a.update(t0 + D::from_secs(3), 100_000_000));
        assert!(!a.is_on(Rid::from("h")));
        // And it can never restore: the capped estimate cannot clear the full
        // budget with margin.
        assert!(!a.update(t0 + D::from_secs(30), 100_000_000));
        assert!(!a.is_on(Rid::from("h")));
    }

    #[test]
    fn keyframe_request_fires_only_the_named_layer() {
        // Two layers on one mid; a PLI for "q" fires only "q"'s source, never "h".
        let mid = Mid::from("0");
        let rc_high = ReverseChannel::new();
        let rc_low = ReverseChannel::new();
        let mut routes = KeyframeRoutes::new();
        routes.push(mid, Some(Rid::from("h")), rc_high.clone());
        routes.push(mid, Some(Rid::from("q")), rc_low.clone());

        routes.request_keyframe(mid, Some(Rid::from("q")));
        assert!(
            matches!(rc_low.take(), Some(g2g_core::PushOutcome::Reconfigure(_))),
            "the low layer's source got the keyframe request"
        );
        assert!(rc_low.take().is_none(), "consumed once");
        assert!(
            rc_high.take().is_none(),
            "a PLI for the low layer must not fire the high layer"
        );

        // An unknown rid fires nothing.
        routes.request_keyframe(mid, Some(Rid::from("f")));
        assert!(rc_high.take().is_none());
        assert!(rc_low.take().is_none());
    }
}
