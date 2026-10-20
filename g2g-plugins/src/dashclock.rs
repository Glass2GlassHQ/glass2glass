use alloc::vec::Vec;

use g2g_core::log::LogSource;
use g2g_core::metrics::monotonic_ns;
use g2g_core::{g2g_debug, g2g_warn};

use crate::fetch::{get_bytes, get_text, net_err, MAX_MANIFEST_BYTES};
use crate::mpd::{parse_xs_datetime, Mpd, UtcTiming};
use crate::xmlutil::days_from_civil;

// gst dashdemux's SLOW_CLOCK_UPDATE_INTERVAL and FAST_CLOCK_UPDATE_INTERVAL
const POLL_AFTER_SUCCESS_NS: u64 = 30 * 60 * 1_000_000_000;
const POLL_AFTER_FAILURE_NS: u64 = 30 * 1_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UtcTimingScheme {
    // the manifest carries the server time itself
    Direct,
    Http(HttpTimeFormat),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HttpTimeFormat {
    DateHeader,
    IsoBody,
    NtpBody,
}

// the NTP and SNTP schemes need a UDP client and stay unsupported
const SCHEMES: [(&str, UtcTimingScheme); 10] = [
    ("urn:mpeg:dash:utc:direct:2014", UtcTimingScheme::Direct),
    (
        "urn:mpeg:dash:utc:http-head:2014",
        UtcTimingScheme::Http(HttpTimeFormat::DateHeader),
    ),
    (
        "urn:mpeg:dash:utc:http-xsdate:2014",
        UtcTimingScheme::Http(HttpTimeFormat::IsoBody),
    ),
    (
        "urn:mpeg:dash:utc:http-iso:2014",
        UtcTimingScheme::Http(HttpTimeFormat::IsoBody),
    ),
    (
        "urn:mpeg:dash:utc:http-ntp:2014",
        UtcTimingScheme::Http(HttpTimeFormat::NtpBody),
    ),
    ("urn:mpeg:dash:utc:direct:2012", UtcTimingScheme::Direct),
    (
        "urn:mpeg:dash:utc:http-head:2012",
        UtcTimingScheme::Http(HttpTimeFormat::DateHeader),
    ),
    (
        "urn:mpeg:dash:utc:http-xsdate:2012",
        UtcTimingScheme::Http(HttpTimeFormat::IsoBody),
    ),
    (
        "urn:mpeg:dash:utc:http-iso:2012",
        UtcTimingScheme::Http(HttpTimeFormat::IsoBody),
    ),
    (
        "urn:mpeg:dash:utc:http-ntp:2012",
        UtcTimingScheme::Http(HttpTimeFormat::NtpBody),
    ),
];

fn scheme(timing: &UtcTiming) -> Option<UtcTimingScheme> {
    SCHEMES
        .iter()
        .find(|(uri, _)| uri.eq_ignore_ascii_case(timing.scheme_id_uri.trim()))
        .map(|&(_, scheme)| scheme)
}

pub(crate) fn local_now_unix() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

// the local clock just before a request went out and just after its answer came back
#[derive(Debug, Clone, Copy)]
pub(crate) struct RequestSpan {
    pub(crate) sent_unix: f64,
    pub(crate) answered_unix: f64,
}

impl RequestSpan {
    // the server read its clock somewhere inside the span, so the midpoint is the best guess
    fn midpoint_unix(self) -> f64 {
        (self.sent_unix + self.answered_unix) / 2.0
    }
}

#[derive(Debug, Default)]
pub(crate) struct ServerClock {
    server_ahead_secs: f64,
    next_poll_ns: u64,
    next_source: usize,
}

impl LogSource for ServerClock {
    fn log_category(&self) -> &'static str {
        "dashsrc"
    }
}

impl ServerClock {
    pub(crate) fn now_unix(&self) -> f64 {
        local_now_unix() + self.server_ahead_secs
    }

    // gst picks the first UTCTiming whose scheme it supports and skips the rest
    pub(crate) async fn poll(
        &mut self,
        client: &reqwest::Client,
        mpd: &Mpd,
        manifest_request: RequestSpan,
    ) {
        let now_ns = monotonic_ns();
        if now_ns < self.next_poll_ns {
            return;
        }
        let Some((scheme, timing)) = mpd
            .utc_timings
            .iter()
            .find_map(|timing| scheme(timing).map(|scheme| (scheme, timing)))
        else {
            if let Some(timing) = mpd.utc_timings.first() {
                g2g_warn!(
                    self,
                    "no supported UTCTiming scheme (first is {}), using the local clock",
                    timing.scheme_id_uri
                );
            }
            self.next_poll_ns = now_ns.saturating_add(POLL_AFTER_FAILURE_NS);
            return;
        };
        let measured = match scheme {
            UtcTimingScheme::Direct => {
                parse_xs_datetime(&timing.value).map(|server| (server, manifest_request))
            }
            UtcTimingScheme::Http(format) => {
                let sources: Vec<&str> = timing.value.split_whitespace().collect();
                if sources.is_empty() {
                    None
                } else {
                    let url = sources[self.next_source % sources.len()];
                    self.next_source = self.next_source.wrapping_add(1);
                    fetch_server_unix(client, format, url).await
                }
            }
        };
        let Some((server_unix, request)) = measured else {
            g2g_warn!(
                self,
                "UTCTiming {} gave no usable time from {}",
                timing.scheme_id_uri,
                timing.value
            );
            self.next_poll_ns = now_ns.saturating_add(POLL_AFTER_FAILURE_NS);
            return;
        };
        self.server_ahead_secs = server_unix - request.midpoint_unix();
        g2g_debug!(
            self,
            "server clock is {:.3}s ahead of the local clock ({})",
            self.server_ahead_secs,
            timing.scheme_id_uri
        );
        self.next_poll_ns = now_ns.saturating_add(POLL_AFTER_SUCCESS_NS);
    }
}

async fn fetch_server_unix(
    client: &reqwest::Client,
    format: HttpTimeFormat,
    url: &str,
) -> Option<(f64, RequestSpan)> {
    let sent_unix = local_now_unix();
    let server_unix = match format {
        HttpTimeFormat::DateHeader => {
            let response = client
                .head(url)
                .send()
                .await
                .and_then(reqwest::Response::error_for_status)
                .map_err(net_err)
                .ok()?;
            let date = response.headers().get(reqwest::header::DATE)?;
            parse_http_date(date.to_str().ok()?)
        }
        HttpTimeFormat::NtpBody => {
            let body = get_bytes(client, url, NTP_TIMESTAMP_BYTES).await.ok()?;
            parse_ntp_timestamp(&body)
        }
        HttpTimeFormat::IsoBody => {
            parse_xs_datetime(&get_text(client, url, MAX_MANIFEST_BYTES).await.ok()?)
        }
    }?;
    let answered_unix = local_now_unix();
    Some((
        server_unix,
        RequestSpan {
            sent_unix,
            answered_unix,
        },
    ))
}

const NTP_TIMESTAMP_BYTES: usize = 8;

fn parse_ntp_timestamp(body: &[u8]) -> Option<f64> {
    let timestamp = u64::from_be_bytes(body.try_into().ok()?);
    let unix_nanos = g2g_core::time::unix_nanos_from_ntp(timestamp)?;
    Some(unix_nanos as f64 / 1e9)
}

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

// the IMF-fixdate form RFC 9110 requires of a Date header: `Sun, 06 Nov 1994 08:49:37 GMT`
fn parse_http_date(text: &str) -> Option<f64> {
    let (_weekday, rest) = text.trim().split_once(", ")?;
    let mut fields = rest.split(' ');
    let day: u32 = fields.next()?.parse().ok()?;
    let month_name = fields.next()?;
    let month = MONTHS.iter().position(|&m| m == month_name)?;
    let year: i64 = fields.next()?.parse().ok()?;
    let mut clock = fields.next()?.split(':');
    let hour: u32 = clock.next()?.parse().ok()?;
    let minute: u32 = clock.next()?.parse().ok()?;
    let second: u32 = clock.next()?.parse().ok()?;
    if fields.next()? != "GMT" || fields.next().is_some() || clock.next().is_some() {
        return None;
    }
    if !(1..=31).contains(&day) || hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    let month = u32::try_from(month + 1).ok()?;
    let days = days_from_civil(year, month, day);
    Some((days * 86_400 + i64::from(hour * 3_600 + minute * 60 + second)) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // RFC 9110's own example date and the unix second it names
    const HTTP_DATE: &str = "Sun, 06 Nov 1994 08:49:37 GMT";
    const HTTP_DATE_UNIX: f64 = 784_111_777.0;

    #[test]
    fn reads_an_http_date_header() {
        assert_eq!(parse_http_date(HTTP_DATE), Some(HTTP_DATE_UNIX));
        assert_eq!(parse_http_date("Sun, 06 Nov 1994 08:49:37 PST"), None);
        assert_eq!(parse_http_date("Sunday, 06-Nov-94 08:49:37 GMT"), None);
    }

    #[test]
    fn reads_an_http_ntp_body() {
        let ntp_seconds = HTTP_DATE_UNIX as u64 + g2g_core::time::NTP_TO_UNIX_EPOCH_SECS;
        let half_second_fraction = 1u64 << 31;
        let body = ((ntp_seconds << 32) | half_second_fraction).to_be_bytes();
        assert_eq!(parse_ntp_timestamp(&body), Some(HTTP_DATE_UNIX + 0.5));
        assert_eq!(parse_ntp_timestamp(&body[..4]), None);
    }
}
