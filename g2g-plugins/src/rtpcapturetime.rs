use core::num::NonZeroU8;

use g2g_core::property::{PropError, PropKind, PropValue, PropertySpec};
use g2g_core::rtp::CaptureTimeExtensionIds;

const ABS_CAPTURE_TIME_ID: &str = "abs-capture-time-id";
const NTP_64_ID: &str = "ntp-64-id";
const EXTENSION_ID_OFF: u64 = 0;

pub(crate) const ABS_CAPTURE_TIME_ID_PROPERTY: PropertySpec = PropertySpec::new(
    ABS_CAPTURE_TIME_ID,
    PropKind::Uint,
    "RFC 8285 id carrying each frame's capture time as WebRTC abs-capture-time (0 = off)",
)
.with_default("0")
.with_range("0", "255");

pub(crate) const NTP_64_ID_PROPERTY: PropertySpec = PropertySpec::new(
    NTP_64_ID,
    PropKind::Uint,
    "RFC 8285 id carrying each frame's capture time as RFC 6051 ntp-64 (0 = off)",
)
.with_default("0")
.with_range("0", "255");

pub(crate) fn set_capture_time_prop(
    ids: &mut CaptureTimeExtensionIds,
    name: &str,
    value: &PropValue,
) -> Option<Result<(), PropError>> {
    let (slot, other) = match name {
        ABS_CAPTURE_TIME_ID => (&mut ids.abs_capture_time, ids.ntp_64),
        NTP_64_ID => (&mut ids.ntp_64, ids.abs_capture_time),
        _ => return None,
    };
    Some(set_extension_id(slot, other, value))
}

pub(crate) fn get_capture_time_prop(
    ids: &CaptureTimeExtensionIds,
    name: &str,
) -> Option<PropValue> {
    let id = match name {
        ABS_CAPTURE_TIME_ID => ids.abs_capture_time,
        NTP_64_ID => ids.ntp_64,
        _ => return None,
    };
    Some(PropValue::Uint(
        id.map_or(EXTENSION_ID_OFF, |id| u64::from(id.get())),
    ))
}

fn set_extension_id(
    slot: &mut Option<NonZeroU8>,
    other: Option<NonZeroU8>,
    value: &PropValue,
) -> Result<(), PropError> {
    let raw = value.as_uint().ok_or(PropError::Type)?;
    let id = NonZeroU8::new(u8::try_from(raw).map_err(|_| PropError::Value)?);
    // RFC 8285 ids name one extension each
    if id.is_some() && id == other {
        return Err(PropError::Value);
    }
    *slot = id;
    Ok(())
}

// stamps follow PTS so a bursting source still steps one frame period per frame
#[cfg(feature = "udp-egress")]
#[derive(Debug, Default)]
pub(crate) struct CaptureClock {
    pts_origin_unix_ns: Option<i64>,
}

#[cfg(feature = "udp-egress")]
impl CaptureClock {
    pub(crate) fn capture_ntp(&mut self, frame: &g2g_core::Frame) -> u64 {
        #[cfg(feature = "metadata")]
        if let Some(wall_clock) = frame.meta.get::<g2g_core::WallClockMeta>() {
            return g2g_core::ntp_from_unix_nanos(wall_clock.unix_nanos).unwrap_or(0);
        }
        let entered_unix_ns = entered_unix_ns(frame);
        let pts_ns = frame.timing.pts().and_then(|pts| i64::try_from(pts).ok());
        let capture_unix_ns = match pts_ns {
            Some(pts_ns) => {
                let origin = *self
                    .pts_origin_unix_ns
                    .get_or_insert(entered_unix_ns.saturating_sub(pts_ns));
                origin.saturating_add(pts_ns)
            }
            None => entered_unix_ns,
        };
        g2g_core::ntp_from_unix_nanos(capture_unix_ns).unwrap_or(0)
    }

    pub(crate) fn restart(&mut self) {
        self.pts_origin_unix_ns = None;
    }
}

// when the frame entered this pipeline, or now when its source did not say
#[cfg(feature = "udp-egress")]
fn entered_unix_ns(frame: &g2g_core::Frame) -> i64 {
    let now_unix_ns = g2g_core::metrics::unix_time_ns();
    if frame.timing.arrival_ns == 0 {
        return now_unix_ns;
    }
    let age_ns = g2g_core::metrics::monotonic_ns().saturating_sub(frame.timing.arrival_ns);
    now_unix_ns.saturating_sub(i64::try_from(age_ns).unwrap_or(i64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_set_read_back_and_refuse_a_clash() {
        let mut ids = CaptureTimeExtensionIds::default();
        let set = |ids: &mut CaptureTimeExtensionIds, name, id| {
            set_capture_time_prop(ids, name, &PropValue::Uint(id)).expect("a capture-time name")
        };
        assert_eq!(set(&mut ids, NTP_64_ID, 5), Ok(()));
        assert_eq!(set(&mut ids, ABS_CAPTURE_TIME_ID, 5), Err(PropError::Value));
        assert_eq!(
            set(&mut ids, ABS_CAPTURE_TIME_ID, 256),
            Err(PropError::Value)
        );
        assert_eq!(set(&mut ids, ABS_CAPTURE_TIME_ID, 200), Ok(()));
        assert_eq!(
            get_capture_time_prop(&ids, NTP_64_ID),
            Some(PropValue::Uint(5))
        );
        assert_eq!(set(&mut ids, NTP_64_ID, EXTENSION_ID_OFF), Ok(()));
        assert_eq!(ids.ntp_64, None);
        assert!(set_capture_time_prop(&mut ids, "port", &PropValue::Uint(1)).is_none());
    }

    #[cfg(feature = "udp-egress")]
    #[test]
    fn unstamped_frames_step_by_their_pts_however_fast_they_arrive() {
        const FRAME_PERIOD_NS: u64 = 33_333_333;
        let frame = |pts_ns| {
            g2g_core::Frame::new(
                g2g_core::MemoryDomain::System(g2g_core::SystemSlice::from_boxed(
                    alloc::vec![0u8; 1].into_boxed_slice(),
                )),
                g2g_core::FrameTiming {
                    pts_ns,
                    ..g2g_core::FrameTiming::default()
                },
                0,
            )
        };
        let unix = |ntp| g2g_core::unix_nanos_from_ntp(ntp).expect("a time");
        let mut clock = CaptureClock::default();
        let first = unix(clock.capture_ntp(&frame(0)));
        let third = unix(clock.capture_ntp(&frame(2 * FRAME_PERIOD_NS)));
        assert_eq!(third - first, 2 * FRAME_PERIOD_NS as i64);

        clock.restart();
        let after_seek = unix(clock.capture_ntp(&frame(0)));
        assert!(
            after_seek >= first,
            "a restart anchors the PTS origin again"
        );
    }
}
