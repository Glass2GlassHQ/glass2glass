// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;

use core::time::Duration;
use std::time::Instant;

use str0m::media::{KeyframeRequestKind, MediaAdded, MediaData, MediaKind, Mid};
use str0m::Rtc;

use g2g_core::clock::{CaptureAnchor, ClockSync};
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{FrameMetaSet, FrameTiming, MemoryDomain};

use crate::webrtc_codec::{capture_unix_ns, WebRtcVideoCodec};

pub(crate) const PLI_INTERVAL: Duration = Duration::from_secs(1);
pub(crate) const UDP_RECV_BYTES: usize = 2000;

// one video m-line's wait for its first keyframe, asking the sender with a PLI every PLI_INTERVAL
#[derive(Debug)]
pub(crate) struct KeyframeGate {
    open: bool,
    last_pli: Instant,
}

impl KeyframeGate {
    pub(crate) fn new() -> Self {
        Self {
            open: false,
            last_pli: Instant::now(),
        }
    }

    pub(crate) fn is_open(&self) -> bool {
        self.open
    }

    // mid-GOP join: drop until the first keyframe
    pub(crate) fn admit(&mut self, keyframe: bool) -> bool {
        self.open |= keyframe;
        self.open
    }

    pub(crate) fn request_keyframe_if_due(&mut self, rtc: &mut Rtc, mid: Mid) -> bool {
        if self.open || self.last_pli.elapsed() < PLI_INTERVAL {
            return false;
        }
        self.last_pli = Instant::now();
        if let Some(stream) = rtc.direct_api().stream_rx_by_mid(mid, None) {
            stream.request_keyframe(KeyframeRequestKind::Pli);
            return true;
        }
        false
    }
}

// one PeerConnection's receive side: which mids carry video and audio, and the video keyframe gate
#[derive(Debug)]
pub(crate) struct MediaReceiver {
    video_codec: WebRtcVideoCodec,
    video_mid: Option<Mid>,
    audio_mid: Option<Mid>,
    video_gate: KeyframeGate,
}

#[derive(Debug)]
pub(crate) struct ReceivedUnit {
    pub(crate) kind: MediaKind,
    pub(crate) capture_unix_ns: Option<i64>,
    pub(crate) data: Vec<u8>,
    // true on the unit that opened the keyframe gate (callers log it)
    pub(crate) first_keyframe: bool,
    keyframe: bool,
}

impl MediaReceiver {
    pub(crate) fn new(video_codec: WebRtcVideoCodec) -> Self {
        Self {
            video_codec,
            video_mid: None,
            audio_mid: None,
            video_gate: KeyframeGate::new(),
        }
    }

    // str0m raises no MediaAdded for an m-line this side offered, so those mids come from add_media
    pub(crate) fn with_mids(
        video_codec: WebRtcVideoCodec,
        video_mid: Option<Mid>,
        audio_mid: Option<Mid>,
    ) -> Self {
        Self {
            video_mid,
            audio_mid,
            ..Self::new(video_codec)
        }
    }

    pub(crate) fn media_added(&mut self, added: &MediaAdded) {
        match added.kind {
            MediaKind::Video if self.video_mid.is_none() => self.video_mid = Some(added.mid),
            MediaKind::Audio if self.audio_mid.is_none() => self.audio_mid = Some(added.mid),
            _ => {}
        }
    }

    pub(crate) fn media_data(&mut self, data: &MediaData) -> Option<ReceivedUnit> {
        let capture_unix_ns = media_capture_unix_ns(data);
        if Some(data.mid) == self.video_mid {
            let first_keyframe = !self.video_gate.is_open();
            if !self
                .video_gate
                .admit(self.video_codec.is_keyframe(&data.data))
            {
                return None;
            }
            let converted = self.video_codec.pipeline_unit(&data.data);
            return Some(ReceivedUnit {
                kind: MediaKind::Video,
                capture_unix_ns,
                keyframe: self.video_codec.is_keyframe(&converted),
                data: converted,
                first_keyframe,
            });
        }
        if Some(data.mid) == self.audio_mid {
            return Some(ReceivedUnit {
                kind: MediaKind::Audio,
                capture_unix_ns,
                data: data.data.to_vec(),
                first_keyframe: false,
                keyframe: false,
            });
        }
        None
    }

    pub(crate) fn request_keyframe_if_due(&mut self, rtc: &mut Rtc) -> bool {
        let Some(mid) = self.video_mid else {
            return false;
        };
        self.video_gate.request_keyframe_if_due(rtc, mid)
    }
}

impl ReceivedUnit {
    pub(crate) fn into_frame(self, stamp: &mut ArrivalStamp, sequence: u64) -> Frame {
        stamp.frame(
            self.data,
            self.keyframe,
            sequence,
            capture_time_meta(self.capture_unix_ns),
        )
    }
}

#[derive(Debug)]
pub(crate) struct ArrivalStamp {
    clock_sync: Option<ClockSync>,
    anchor: CaptureAnchor,
    first_arrival_ns: Option<u64>,
}

impl ArrivalStamp {
    pub(crate) fn new(clock_sync: Option<ClockSync>) -> Self {
        Self {
            clock_sync,
            anchor: CaptureAnchor::new(),
            first_arrival_ns: None,
        }
    }

    pub(crate) fn frame(
        &mut self,
        data: Vec<u8>,
        keyframe: bool,
        sequence: u64,
        meta: FrameMetaSet,
    ) -> Frame {
        let arrival_ns = g2g_core::metrics::monotonic_ns();
        let elapsed_ns = arrival_ns - *self.first_arrival_ns.get_or_insert(arrival_ns);
        let pts_ns = match &self.clock_sync {
            Some(sync) => self.anchor.stamp(sync, elapsed_ns, 0),
            None => elapsed_ns,
        };
        Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            timing: FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns: 0,
                capture_ns: pts_ns,
                arrival_ns,
                keyframe,
                ..FrameTiming::default()
            },
            sequence,
            meta,
        }
    }
}

pub(crate) fn media_capture_unix_ns(data: &MediaData) -> Option<i64> {
    data.ext_vals
        .abs_capture_time
        .as_ref()
        .and_then(capture_unix_ns)
}

pub(crate) fn capture_time_meta(capture_unix_ns: Option<i64>) -> FrameMetaSet {
    #[cfg(feature = "metadata")]
    {
        let mut meta = FrameMetaSet::new();
        if let Some(unix_nanos) = capture_unix_ns {
            meta.attach(g2g_core::WallClockMeta { unix_nanos });
        }
        meta
    }
    #[cfg(not(feature = "metadata"))]
    {
        let _ = capture_unix_ns;
        FrameMetaSet::new()
    }
}
