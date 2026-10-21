use core::ops::{Range, RangeInclusive};
use core::time::Duration;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

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
    // one NTP client request over UDP, which also serves the SNTP scheme
    Ntp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HttpTimeFormat {
    DateHeader,
    IsoBody,
    NtpBody,
}

const SCHEMES: [(&str, UtcTimingScheme); 14] = [
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
    ("urn:mpeg:dash:utc:ntp:2014", UtcTimingScheme::Ntp),
    ("urn:mpeg:dash:utc:sntp:2014", UtcTimingScheme::Ntp),
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
    ("urn:mpeg:dash:utc:ntp:2012", UtcTimingScheme::Ntp),
    ("urn:mpeg:dash:utc:sntp:2012", UtcTimingScheme::Ntp),
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

    // tries the supported UTCTiming elements in manifest order until one gives a time
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
        let mut supported = mpd
            .utc_timings
            .iter()
            .filter_map(|timing| scheme(timing).map(|scheme| (scheme, timing)))
            .peekable();
        if supported.peek().is_none() {
            if let Some(timing) = mpd.utc_timings.first() {
                g2g_warn!(
                    self,
                    "no supported UTCTiming scheme (first is {}), using the local clock",
                    timing.scheme_id_uri
                );
            }
            self.next_poll_ns = now_ns.saturating_add(POLL_AFTER_FAILURE_NS);
            return;
        }
        for (scheme, timing) in supported {
            let Some((server_unix, request)) =
                self.measure(client, scheme, timing, manifest_request).await
            else {
                g2g_warn!(
                    self,
                    "UTCTiming {} gave no usable time from {}",
                    timing.scheme_id_uri,
                    timing.value
                );
                continue;
            };
            self.server_ahead_secs = server_unix - request.midpoint_unix();
            g2g_debug!(
                self,
                "server clock is {:.3}s ahead of the local clock ({})",
                self.server_ahead_secs,
                timing.scheme_id_uri
            );
            self.next_poll_ns = now_ns.saturating_add(POLL_AFTER_SUCCESS_NS);
            return;
        }
        self.next_poll_ns = now_ns.saturating_add(POLL_AFTER_FAILURE_NS);
    }

    async fn measure(
        &mut self,
        client: &reqwest::Client,
        scheme: UtcTimingScheme,
        timing: &UtcTiming,
        manifest_request: RequestSpan,
    ) -> Option<(f64, RequestSpan)> {
        match scheme {
            UtcTimingScheme::Direct => {
                parse_xs_datetime(&timing.value).map(|server| (server, manifest_request))
            }
            UtcTimingScheme::Http(format) => {
                let url = self.source_for_this_poll(&timing.value)?;
                fetch_server_unix(client, format, url).await
            }
            UtcTimingScheme::Ntp => {
                let server = self.source_for_this_poll(&timing.value)?;
                query_ntp_server(server).await
            }
        }
    }

    // a value can list several servers, which successive polls take in turn
    fn source_for_this_poll<'value>(&mut self, value: &'value str) -> Option<&'value str> {
        let sources: Vec<&str> = value.split_whitespace().collect();
        if sources.is_empty() {
            return None;
        }
        let source = sources[self.next_source % sources.len()];
        self.next_source = self.next_source.wrapping_add(1);
        Some(source)
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

// RFC 5905 packet layout, which SNTP (RFC 4330) shares
const NTP_PORT: u16 = 123;
const NTP_PACKET_BYTES: usize = 48;
const NTP_VERSION: u8 = 4;
const NTP_VERSION_SHIFT: u32 = 3;
const NTP_MODE_MASK: u8 = 0b111;
const NTP_MODE_CLIENT: u8 = 3;
const NTP_MODE_SERVER: u8 = 4;
const NTP_LEAP_SHIFT: u32 = 6;
const NTP_LEAP_UNSYNCHRONIZED: u8 = 3;
// stratum 0 is a kiss-of-death message, and past 15 the server is unsynchronized
const NTP_SYNCHRONIZED_STRATA: RangeInclusive<u8> = 1..=15;
const NTP_ORIGINATE_TIMESTAMP: Range<usize> = 24..32;
const NTP_TRANSMIT_TIMESTAMP: Range<usize> = 40..48;
// the playback loop waits on this, so an unreachable server must not stall it long
const NTP_REPLY_TIMEOUT: Duration = Duration::from_secs(1);

async fn query_ntp_server(server: &str) -> Option<(f64, RequestSpan)> {
    tokio::time::timeout(NTP_REPLY_TIMEOUT, exchange_ntp_packets(server))
        .await
        .ok()?
}

async fn exchange_ntp_packets(server: &str) -> Option<(f64, RequestSpan)> {
    let server = resolve_ntp_server(server).await?;
    let local: SocketAddr = if server.is_ipv4() {
        (Ipv4Addr::UNSPECIFIED, 0).into()
    } else {
        (Ipv6Addr::UNSPECIFIED, 0).into()
    };
    let socket = tokio::net::UdpSocket::bind(local).await.ok()?;
    socket.connect(server).await.ok()?;
    let sent_unix = local_now_unix();
    let request = ntp_client_request(sent_unix)?;
    socket.send(&request).await.ok()?;
    let mut reply = [0u8; NTP_PACKET_BYTES];
    let received = socket.recv(&mut reply).await.ok()?;
    let answered_unix = local_now_unix();
    let server_unix = parse_ntp_reply(&reply[..received], &request)?;
    Some((
        server_unix,
        RequestSpan {
            sent_unix,
            answered_unix,
        },
    ))
}

// `host`, `host:port` or an IP address, on port 123 unless one is given
async fn resolve_ntp_server(server: &str) -> Option<SocketAddr> {
    let names_port = server.contains(':') && server.parse::<IpAddr>().is_err();
    if names_port {
        tokio::net::lookup_host(server).await.ok()?.next()
    } else {
        tokio::net::lookup_host((server, NTP_PORT))
            .await
            .ok()?
            .next()
    }
}

fn ntp_client_request(sent_unix: f64) -> Option<[u8; NTP_PACKET_BYTES]> {
    let mut request = [0u8; NTP_PACKET_BYTES];
    request[0] = (NTP_VERSION << NTP_VERSION_SHIFT) | NTP_MODE_CLIENT;
    // the server echoes this as the originate timestamp, which ties its reply to this request
    let transmit = g2g_core::time::ntp_from_unix_nanos((sent_unix * 1e9) as i64)?;
    request[NTP_TRANSMIT_TIMESTAMP].copy_from_slice(&transmit.to_be_bytes());
    Some(request)
}

// the server's transmit timestamp, from a reply RFC 4330 section 5 accepts
fn parse_ntp_reply(reply: &[u8], request: &[u8; NTP_PACKET_BYTES]) -> Option<f64> {
    if reply.len() < NTP_PACKET_BYTES {
        return None;
    }
    let leap = reply[0] >> NTP_LEAP_SHIFT;
    let mode = reply[0] & NTP_MODE_MASK;
    let stratum = reply[1];
    let answers_request = reply[NTP_ORIGINATE_TIMESTAMP] == request[NTP_TRANSMIT_TIMESTAMP];
    if leap == NTP_LEAP_UNSYNCHRONIZED
        || mode != NTP_MODE_SERVER
        || !NTP_SYNCHRONIZED_STRATA.contains(&stratum)
        || !answers_request
    {
        return None;
    }
    parse_ntp_timestamp(&reply[NTP_TRANSMIT_TIMESTAMP])
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

    use alloc::string::String;

    use crate::mpd::parse;

    const NTP_SCHEME: &str = "urn:mpeg:dash:utc:ntp:2014";
    const DIRECT_SCHEME: &str = "urn:mpeg:dash:utc:direct:2014";
    // HTTP_DATE as an xs:dateTime
    const DIRECT_VALUE: &str = "1994-11-06T08:49:37Z";
    // the local clock reading the manifest request is pinned to, for the direct scheme
    const MANIFEST_REQUEST_UNIX: f64 = HTTP_DATE_UNIX - 10.0;
    const MANIFEST_REQUEST: RequestSpan = RequestSpan {
        sent_unix: MANIFEST_REQUEST_UNIX,
        answered_unix: MANIFEST_REQUEST_UNIX,
    };
    // the time the fake NTP server reports, far from both clocks so a match is no accident
    const NTP_SERVER_UNIX: f64 = HTTP_DATE_UNIX + 0.5;
    const SERVER_FIRST_BYTE: u8 = (NTP_VERSION << NTP_VERSION_SHIFT) | NTP_MODE_SERVER;
    const SERVER_STRATUM: u8 = 2;

    fn mpd_with_timings(timings: &[(&str, &str)]) -> Mpd {
        let elements: String = timings
            .iter()
            .map(|(scheme, value)| {
                alloc::format!("<UTCTiming schemeIdUri=\"{scheme}\" value=\"{value}\"/>")
            })
            .collect();
        parse(&alloc::format!(
            "<MPD type=\"dynamic\"><Period><AdaptationSet>\
               <SegmentTemplate media=\"$Number$.m4s\" duration=\"1\"/>\
               <Representation id=\"v\" bandwidth=\"1\"/>\
             </AdaptationSet></Period>{elements}</MPD>"
        ))
        .unwrap()
    }

    fn server_reply(request: &[u8], transmit_unix: f64) -> Vec<u8> {
        let mut reply = alloc::vec![0u8; NTP_PACKET_BYTES];
        reply[0] = SERVER_FIRST_BYTE;
        reply[1] = SERVER_STRATUM;
        reply[NTP_ORIGINATE_TIMESTAMP].copy_from_slice(&request[NTP_TRANSMIT_TIMESTAMP]);
        let transmit = g2g_core::time::ntp_from_unix_nanos((transmit_unix * 1e9) as i64).unwrap();
        reply[NTP_TRANSMIT_TIMESTAMP].copy_from_slice(&transmit.to_be_bytes());
        reply
    }

    // a local UDP server answering each request with `answer(request)`, returned as `host:port`
    async fn ntp_responder(answer: fn(&[u8]) -> Vec<u8>) -> String {
        let socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let address = socket.local_addr().unwrap();
        tokio::spawn(async move {
            let mut request = [0u8; NTP_PACKET_BYTES];
            loop {
                let (_, peer) = socket.recv_from(&mut request).await.unwrap();
                socket.send_to(&answer(&request), peer).await.unwrap();
            }
        });
        alloc::format!("{address}")
    }

    #[tokio::test]
    async fn an_ntp_reply_sets_the_offset_from_the_round_trip_midpoint() {
        let server = ntp_responder(|request| server_reply(request, NTP_SERVER_UNIX)).await;
        let mpd = mpd_with_timings(&[(NTP_SCHEME, server.as_str())]);
        let mut clock = ServerClock::default();
        let before = local_now_unix();
        clock
            .poll(&reqwest::Client::new(), &mpd, MANIFEST_REQUEST)
            .await;
        let after = local_now_unix();
        let midpoint = NTP_SERVER_UNIX - clock.server_ahead_secs;
        assert!(
            (before..=after).contains(&midpoint),
            "{midpoint} lies outside the request span {before}..={after}"
        );
    }

    #[tokio::test]
    async fn an_unanswered_ntp_request_falls_through_to_the_next_timing() {
        // bound but never read, so the request times out
        let silent = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let server = alloc::format!("{}", silent.local_addr().unwrap());
        let mpd = mpd_with_timings(&[(NTP_SCHEME, server.as_str()), (DIRECT_SCHEME, DIRECT_VALUE)]);
        let mut clock = ServerClock::default();
        clock
            .poll(&reqwest::Client::new(), &mpd, MANIFEST_REQUEST)
            .await;
        assert_eq!(
            clock.server_ahead_secs,
            HTTP_DATE_UNIX - MANIFEST_REQUEST_UNIX
        );
    }

    #[tokio::test]
    async fn a_malformed_ntp_reply_falls_through_to_the_next_timing() {
        let server = ntp_responder(|request| {
            let mut reply = server_reply(request, NTP_SERVER_UNIX);
            reply[0] = request[0];
            reply
        })
        .await;
        let mpd = mpd_with_timings(&[(NTP_SCHEME, server.as_str()), (DIRECT_SCHEME, DIRECT_VALUE)]);
        let mut clock = ServerClock::default();
        clock
            .poll(&reqwest::Client::new(), &mpd, MANIFEST_REQUEST)
            .await;
        assert_eq!(
            clock.server_ahead_secs,
            HTTP_DATE_UNIX - MANIFEST_REQUEST_UNIX
        );
    }

    #[test]
    fn ntp_and_sntp_schemes_share_the_udp_client() {
        for uri in [
            NTP_SCHEME,
            "urn:mpeg:dash:utc:sntp:2014",
            "urn:mpeg:dash:utc:ntp:2012",
            "urn:mpeg:dash:utc:sntp:2012",
        ] {
            let timing = UtcTiming {
                scheme_id_uri: String::from(uri),
                value: String::from("pool.ntp.org"),
            };
            assert_eq!(scheme(&timing), Some(UtcTimingScheme::Ntp), "{uri}");
        }
    }

    #[test]
    fn rejects_ntp_replies_rfc_4330_says_to_discard() {
        let request = ntp_client_request(HTTP_DATE_UNIX).unwrap();
        let valid = server_reply(&request, NTP_SERVER_UNIX);
        assert_eq!(parse_ntp_reply(&valid, &request), Some(NTP_SERVER_UNIX));

        assert_eq!(
            parse_ntp_reply(&valid[..NTP_PACKET_BYTES - 1], &request),
            None
        );
        let mut client_mode = valid.clone();
        client_mode[0] = request[0];
        assert_eq!(parse_ntp_reply(&client_mode, &request), None);
        let mut unsynchronized = valid.clone();
        unsynchronized[0] |= NTP_LEAP_UNSYNCHRONIZED << NTP_LEAP_SHIFT;
        assert_eq!(parse_ntp_reply(&unsynchronized, &request), None);
        let mut kiss_of_death = valid.clone();
        kiss_of_death[1] = 0;
        assert_eq!(parse_ntp_reply(&kiss_of_death, &request), None);
        let mut other_request = valid.clone();
        other_request[NTP_ORIGINATE_TIMESTAMP].fill(0);
        assert_eq!(parse_ntp_reply(&other_request, &request), None);
        let mut no_time = valid;
        no_time[NTP_TRANSMIT_TIMESTAMP].fill(0);
        assert_eq!(parse_ntp_reply(&no_time, &request), None);
    }

    // a host with a working clock agrees with the pool to well within this
    const LIVE_POOL_AGREEMENT_SECS: f64 = 60.0;

    #[tokio::test]
    #[ignore = "needs outbound UDP port 123"]
    async fn pool_ntp_org_answers_an_ntp_request() {
        let (server_unix, request) = query_ntp_server("pool.ntp.org").await.unwrap();
        assert!((server_unix - request.midpoint_unix()).abs() < LIVE_POOL_AGREEMENT_SECS);
    }
}
