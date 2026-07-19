//! Shared WebRTC send-simulcast machinery for grouped-pad publishing sessions:
//! groups N video layers under one m-line with N send rids, and routes a remote
//! per-`(mid, rid)` keyframe request to exactly the source feeding that layer.
//! str0m-generic on purpose: the same code serves any egress session (LiveKit
//! today via [`crate::livekitsink`], a WHIP grouped-pad session later). The rid
//! naming follows the LiveKit / browser convention so the SFU maps each rid to
//! the right spatial layer; the SFU-specific quality metadata lives in the sink.

use alloc::vec::Vec;

use str0m::media::{Mid, Rid, Simulcast, SimulcastLayer};

use g2g_core::{AudioFormat, Caps, Dim, G2gError, ReverseChannel, VideoCodec};

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

/// The most simulcast layers a sink groups on one m-line (the `f`/`h`/`q` rids).
pub const MAX_VIDEO_LAYERS: usize = 3;

/// Per-pad bookkeeping shared by the simulcast-capable session sinks
/// (`LiveKitSink`, `WebRtcSessionSink`): N video-layer pads (pad 0 = highest
/// resolution) plus an optional audio pad, each with its fixated track kind,
/// video geometry, and reverse-signal channel. The element owns the network
/// half; this owns the pad half.
#[derive(Debug)]
pub(crate) struct SimulcastPads {
    pub(crate) video_layers: usize,
    pub(crate) has_audio: bool,
    /// Track kind per input pad, set in `configure(input, caps)`.
    pub(crate) tracks: Vec<Option<crate::webrtcsink::Track>>,
    /// Fixated video resolution per input pad ((0,0) for audio).
    pub(crate) dims: Vec<(u32, u32)>,
    /// Per-input reverse-signal channel (PLI / per-layer bitrate).
    pub(crate) reverse: Vec<ReverseChannel>,
}

impl SimulcastPads {
    pub(crate) fn new() -> Self {
        let mut p = Self {
            video_layers: 1,
            has_audio: false,
            tracks: Vec::new(),
            dims: Vec::new(),
            reverse: Vec::new(),
        };
        p.rebuild();
        p
    }

    /// Resize the per-pad vectors for the current video-layer + audio count.
    pub(crate) fn rebuild(&mut self) {
        let n = self.input_count();
        self.tracks = alloc::vec![None; n];
        self.dims = alloc::vec![(0u32, 0u32); n];
        self.reverse = (0..n).map(|_| ReverseChannel::new()).collect();
    }

    pub(crate) fn set_audio(&mut self, on: bool) {
        self.has_audio = on;
        self.rebuild();
    }

    pub(crate) fn set_video_layers(&mut self, layers: usize) {
        self.video_layers = layers.clamp(1, MAX_VIDEO_LAYERS);
        self.rebuild();
    }

    /// Shape by bare pad count (the launch-registry path, M725): the track
    /// kinds are read from each pad's caps at configure time, so `n` linked
    /// pads simply become `n` slots (an audio-caps pad is the audio track, the
    /// video-caps pads are the layers in pad order).
    pub(crate) fn set_pad_count(&mut self, n: usize) {
        self.video_layers = n.max(1);
        self.has_audio = false;
        self.rebuild();
    }

    pub(crate) fn input_count(&self) -> usize {
        self.video_layers + self.has_audio as usize
    }

    /// Record one pad's fixated caps (track kind + geometry).
    pub(crate) fn configure(&mut self, input: usize, caps: &Caps) -> Result<(), G2gError> {
        let track = track_of(caps).ok_or(G2gError::CapsMismatch)?;
        *self.tracks.get_mut(input).ok_or(G2gError::CapsMismatch)? = Some(track);
        if let Some(d) = self.dims.get_mut(input) {
            *d = video_dims(caps);
        }
        Ok(())
    }

    /// True once every input pad has been configured.
    pub(crate) fn all_configured(&self) -> bool {
        self.tracks.iter().all(|t| t.is_some())
    }

    /// Input pad indices carrying video, in pad order (pad 0 = top layer).
    pub(crate) fn video_inputs(&self) -> Vec<usize> {
        self.tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == Some(crate::webrtcsink::Track::Video))
            .map(|(i, _)| i)
            .collect()
    }

    /// The rid-tagged simulcast layer list, highest resolution first (empty
    /// geometry when a pad's caps were not fixated to a size).
    pub(crate) fn layers(&self) -> Vec<SendLayer> {
        let video = self.video_inputs();
        let rids = rids_high_to_low(video.len());
        video
            .iter()
            .zip(rids)
            .map(|(&i, rid)| SendLayer {
                rid,
                width: self.dims[i].0,
                height: self.dims[i].1,
            })
            .collect()
    }

    /// The audio pad's index, if audio is configured.
    pub(crate) fn audio_input(&self) -> Option<usize> {
        self.tracks
            .iter()
            .position(|t| *t == Some(crate::webrtcsink::Track::Audio))
    }
}

impl Default for SimulcastPads {
    fn default() -> Self {
        Self::new()
    }
}

/// The fixated `(width, height)` of a video caps, `(0, 0)` for audio or an
/// unfixated dimension (the layer metadata then simply omits the resolution).
pub(crate) fn video_dims(caps: &Caps) -> (u32, u32) {
    if let Caps::CompressedVideo { width, height, .. } = caps {
        let w = if let Dim::Fixed(w) = width { *w } else { 0 };
        let h = if let Dim::Fixed(h) = height { *h } else { 0 };
        (w, h)
    } else {
        (0, 0)
    }
}

/// The track kind an input's caps select (H.264 video or Opus audio).
pub(crate) fn track_of(caps: &Caps) -> Option<crate::webrtcsink::Track> {
    match caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            ..
        } => Some(crate::webrtcsink::Track::Video),
        Caps::Audio {
            format: AudioFormat::Opus,
            ..
        } => Some(crate::webrtcsink::Track::Audio),
        _ => None,
    }
}

/// Build the str0m send-simulcast description for `layers`, or `None` for fewer
/// than two layers (a single stream is not simulcast and takes no rid). Each
/// rid carries its `max-width` / `max-height` restriction (RFC 8851) from the
/// pad's fixated caps, so an SFU that orders layers from the SDP (not from
/// out-of-band track metadata) still binds each rid to the right quality.
pub fn send_simulcast(layers: &[SendLayer]) -> Option<Simulcast> {
    if layers.len() < 2 {
        return None;
    }
    let mut sc = Simulcast::new();
    for l in layers {
        let layer = if l.width > 0 && l.height > 0 {
            SimulcastLayer::new_with_attributes(l.rid)
                .max_width(l.width)
                .max_height(l.height)
                .build()
        } else {
            SimulcastLayer::new(l.rid)
        };
        sc.add_send_layer(layer);
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
    pub(crate) fn new() -> Self {
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

    /// Per-layer bitrate targets for one aggregate estimate (M722): each
    /// active layer gets its nominal share scaled by the estimate over the
    /// active set's total (clamped to 0.25x..2x nominal so a transient
    /// estimate cannot starve or balloon an encoder), and a shed layer gets
    /// `0`, the idle hint (see `PushOutcome::Bitrate`).
    pub fn targets(&self, estimate_bps: u64) -> Vec<(Rid, u32)> {
        let est = if self.cap > 0 {
            estimate_bps.min(self.cap)
        } else {
            estimate_bps
        };
        let needed = self.budget(self.active).max(1);
        let scale = (est as f64 / needed as f64).clamp(0.25, 2.0);
        self.layers
            .iter()
            .enumerate()
            .map(|(i, (rid, nominal))| {
                let bps = if i < self.active {
                    (*nominal as f64 * scale) as u32
                } else {
                    0
                };
                (*rid, bps)
            })
            .collect()
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
    fn targets_share_the_estimate_and_idle_shed_layers() {
        let layers = [
            SendLayer {
                rid: "f",
                width: 640,
                height: 480,
            },
            SendLayer {
                rid: "h",
                width: 480,
                height: 360,
            },
            SendLayer {
                rid: "q",
                width: 320,
                height: 240,
            },
        ];
        let mut a = LayerAllocator::new(&layers, 0);
        // All active at exactly the nominal budget: every target ~= nominal.
        let full = a.initial_bps();
        let t = a.targets(full);
        assert!(
            t.iter().all(|(_, bps)| *bps > 0),
            "all layers funded: {t:?}"
        );
        // Starve long enough to shed the top layer: it gets the 0 idle hint,
        // the survivors share the estimate proportionally.
        let t0 = Instant::now();
        let low = full / 4;
        a.update(t0, low);
        a.update(t0 + DROP_AFTER, low);
        assert_eq!(a.active, 2, "top layer shed");
        let t = a.targets(low);
        let top = t.iter().find(|(r, _)| *r == Rid::from("f")).unwrap();
        assert_eq!(top.1, 0, "shed layer gets the idle hint");
        let kept: Vec<_> = t.iter().filter(|(_, bps)| *bps > 0).collect();
        assert_eq!(kept.len(), 2);
        // Scale clamps: a wildly low estimate cannot starve an active encoder
        // below a quarter of its nominal rate.
        let t = a.targets(1);
        let q = t.iter().find(|(r, _)| *r == Rid::from("q")).unwrap();
        assert!(q.1 > 0);
    }

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
        // Each rid carries its resolution restriction (RFC 8851) so the SFU can
        // order layers from the SDP alone.
        assert!(
            sdp.contains("max-width=640") && sdp.contains("max-height=480"),
            "high rid missing restrictions:\n{sdp}"
        );
        assert!(
            sdp.contains("max-width=320") && sdp.contains("max-height=240"),
            "low rid missing restrictions:\n{sdp}"
        );
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
