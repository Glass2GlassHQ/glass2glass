//! XML escaping and ISO-8601 UTC formatting shared by the XML-emitting
//! elements (`cotsink`, `onvif`). Always compiled, no_std + alloc.

use alloc::format;
use alloc::string::String;

/// XML-escape a string for an attribute value or element text, and drop the C0
/// control characters XML 1.0 does not admit in either form.
pub(crate) fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c if (c as u32) < 0x20 => {}
            c => out.push(c),
        }
    }
    out
}

/// civil_from_days (Howard Hinnant): days since 1970-01-01 -> (year, month,
/// day), so no chrono dependency is needed.
fn civil(secs: u64) -> (i64, i64, i64, u64, u64, u64) {
    let days = (secs / 86_400) as i64;
    let tod = secs % 86_400;
    let (hh, mm, ss) = (tod / 3_600, (tod % 3_600) / 60, tod % 60);

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d, hh, mm, ss)
}

/// Format a UNIX timestamp as an ISO-8601 UTC `xsd:dateTime` (no fractional
/// seconds, `Z` zone), the form ONVIF expects for `Created`.
#[cfg(feature = "onvif")]
pub(crate) fn iso8601_utc(secs: u64) -> String {
    let (year, m, d, hh, mm, ss) = civil(secs);
    format!("{year:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Format microseconds since the Unix epoch as
/// `CCYY-MM-DDThh:mm:ss.ssssssZ`, the W3C XML `xs:dateTime` profile pytak
/// writes for CoT `time` / `start` / `stale` (`W3C_XML_DATETIME =
/// "%Y-%m-%dT%H:%M:%S.%fZ"`, Python `%f` being 6 digits, always UTC with a
/// literal `Z`). Six digits is also the exact resolution of ST 0601 tag 2, so
/// no precision is lost.
pub(crate) fn iso8601_utc_us(unix_us: u64) -> String {
    let us = unix_us % 1_000_000;
    let (year, m, d, hh, mm, ss) = civil(unix_us / 1_000_000);
    format!("{year:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}.{us:06}Z")
}
