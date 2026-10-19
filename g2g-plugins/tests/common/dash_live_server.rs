use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn now_unix() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

/// Format `unix` seconds as an `xs:dateTime` in UTC, the `availabilityStartTime`
/// form ffmpeg writes, milliseconds included. The inverse of the parser's
/// `days_from_civil` (Howard Hinnant's `civil_from_days`). Keeping the
/// sub-second part matters: truncating it would move the anchor up to a second
/// earlier and could tip the window onto the next segment.
pub(crate) fn xs_datetime(unix: f64) -> String {
    let secs = unix.floor() as i64;
    let millis = ((unix - secs as f64) * 1000.0) as i64;
    let (days, tod) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        tod / 3600,
        (tod / 60) % 60,
        tod % 60
    )
}
