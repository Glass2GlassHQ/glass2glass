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
