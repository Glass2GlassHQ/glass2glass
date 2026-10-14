use g2g_core::time::ticks_to_ns;
use objc2_core_media::{CMTime, CMTimeFlags};

const NANOSECOND_TIMESCALE: i32 = 1_000_000_000;

pub(crate) fn cm_time(pts_ns: u64) -> CMTime {
    CMTime {
        value: pts_ns as i64,
        timescale: NANOSECOND_TIMESCALE,
        flags: CMTimeFlags::Valid,
        epoch: 0,
    }
}

// An unknown duration or DTS.
pub(crate) fn cm_time_invalid() -> CMTime {
    CMTime {
        value: 0,
        timescale: 0,
        flags: CMTimeFlags::empty(),
        epoch: 0,
    }
}

// 0 for an invalid time or a negative value.
pub(crate) fn cmtime_to_ns(time: CMTime) -> u64 {
    let Ok(timescale) = u32::try_from(time.timescale) else {
        return 0;
    };
    if timescale == 0 || !time.flags.contains(CMTimeFlags::Valid) {
        return 0;
    }
    ticks_to_ns(u64::try_from(time.value).unwrap_or(0), timescale)
}
