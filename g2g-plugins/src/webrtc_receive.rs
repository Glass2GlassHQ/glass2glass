use alloc::vec::Vec;

use core::time::Duration;
use std::time::Instant;

use str0m::media::{KeyframeRequestKind, MediaAdded, MediaData, MediaKind, Mid};
use str0m::Rtc;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{FrameMetaSet, FrameTiming, MemoryDomain};

use crate::webrtc_codec::{capture_unix_ns, WebRtcVideoCodec};
use crate::webrtc_util::media_time_ns;

pub(crate) const PLI_INTERVAL: Duration = Duration::from_secs(1);
pub(crate) const UDP_RECV_BYTES: usize = 2000;

// one PeerConnection's receive side: which mids carry video and audio, the keyframe gate, the pli timer
#[derive(Debug)]
pub(crate) struct MediaReceiver {
    video_codec: WebRtcVideoCodec,
    video_mid: Option<Mid>,
    audio_mid: Option<Mid>,
    video_keyframed: bool,
    last_pli: Instant,
}

#[derive(Debug)]
pub(crate) struct ReceivedUnit {
    pub(crate) kind: MediaKind,
    pub(crate) pts_ns: u64,
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
            video_keyframed: false,
            last_pli: Instant::now(),
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
        let pts_ns = media_time_ns(data.time);
        let capture_unix_ns = media_capture_unix_ns(data);
        if Some(data.mid) == self.video_mid {
            let first_keyframe = !self.video_keyframed;
            if first_keyframe {
                // mid-GOP join: drop until the first keyframe
                if !self.video_codec.is_keyframe(&data.data) {
                    return None;
                }
                self.video_keyframed = true;
            }
            let converted = self.video_codec.pipeline_unit(&data.data);
            return Some(ReceivedUnit {
                kind: MediaKind::Video,
                pts_ns,
                capture_unix_ns,
                keyframe: self.video_codec.is_keyframe(&converted),
                data: converted,
                first_keyframe,
            });
        }
        if Some(data.mid) == self.audio_mid {
            return Some(ReceivedUnit {
                kind: MediaKind::Audio,
                pts_ns,
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
        if self.video_keyframed || self.last_pli.elapsed() < PLI_INTERVAL {
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

impl ReceivedUnit {
    pub(crate) fn into_frame(self, sequence: u64) -> Frame {
        Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(self.data.into_boxed_slice())),
            timing: FrameTiming {
                pts_ns: self.pts_ns,
                dts_ns: self.pts_ns,
                duration_ns: 0,
                capture_ns: self.pts_ns,
                arrival_ns: g2g_core::metrics::monotonic_ns(),
                keyframe: self.keyframe,
            },
            sequence,
            meta: capture_time_meta(self.capture_unix_ns),
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
