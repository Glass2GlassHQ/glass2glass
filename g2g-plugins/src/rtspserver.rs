//! Sans-IO RTSP 1.0 server responder (RFC 2326). Pure `no_std + alloc`, no
//! sockets: feed a complete request with [`RtspResponder::handle_request`] and
//! get back the response bytes plus an [`RtspEvent`] telling the I/O layer what
//! to do (start streaming on `PLAY`, expect media on `RECORD`, tear down).
//!
//! Scope: the method set a player (DESCRIBE / SETUP / PLAY / PAUSE / TEARDOWN)
//! or a publisher (ANNOUNCE / SETUP / RECORD) drives, one session, H.264 over
//! RTP/AVP. Both transports are negotiated: unicast UDP (the `client_port` range,
//! `RtspEvent::Setup`) and TCP-interleaved (RFC 2326 §10.12, the `interleaved=`
//! channels, `RtspEvent::SetupInterleaved`); the ingest I/O layer
//! ([`RtspServerSrc`](crate::rtspserversrc)) serves both, while the serving sink
//! ([`RtspServerSink`](crate::rtspserversink)) is UDP-only for now.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// What the I/O layer should do after a request, beyond sending the response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtspEvent {
    /// Nothing beyond the response (OPTIONS / DESCRIBE / GET_PARAMETER / PAUSE).
    None,
    /// `SETUP` negotiated unicast UDP; stream RTP to this client RTP port.
    Setup { client_rtp_port: u16 },
    /// `SETUP` negotiated TCP-interleaved transport (RFC 2326 §10.12): RTP / RTCP
    /// ride the control TCP connection as `$`-framed binary on these channels,
    /// rather than on their own UDP ports. What `ffmpeg -rtsp_transport tcp` uses.
    SetupInterleaved { rtp_channel: u8, rtcp_channel: u8 },
    /// `PLAY`: begin streaming the served media to the SETUP'd client port.
    Play,
    /// `RECORD`: the client will now push media to the server port.
    Record,
    /// `TEARDOWN`: stop and release the session.
    Teardown,
}

/// RTSP session lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Init,
    Ready,
    Playing,
    Recording,
}

/// A parsed RTSP request: the request line plus the headers the responder needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtspRequest {
    pub method: String,
    pub uri: String,
    pub cseq: u32,
    pub transport: Option<String>,
    pub content_length: usize,
    pub body: Vec<u8>,
}

impl RtspRequest {
    /// Parse one complete request from `buf`, returning it and the number of
    /// bytes consumed. `None` if `buf` does not yet hold a full request (no
    /// header terminator, or the body has not fully arrived).
    pub fn parse(buf: &[u8]) -> Option<(RtspRequest, usize)> {
        let header_end = find_double_crlf(buf)?;
        let head = core::str::from_utf8(&buf[..header_end]).ok()?;
        let mut lines = head.split("\r\n");

        let request_line = lines.next()?;
        let mut parts = request_line.split(' ');
        let method = parts.next()?.to_string();
        let uri = parts.next()?.to_string();

        let mut cseq = 0u32;
        let mut transport = None;
        let mut content_length = 0usize;
        for line in lines {
            let Some((key, value)) = line.split_once(':') else { continue };
            let value = value.trim();
            // RTSP header names are case-insensitive.
            if key.eq_ignore_ascii_case("CSeq") {
                cseq = value.parse().unwrap_or(0);
            } else if key.eq_ignore_ascii_case("Transport") {
                transport = Some(value.to_string());
            } else if key.eq_ignore_ascii_case("Content-Length") {
                content_length = value.parse().unwrap_or(0);
            }
        }

        let body_start = header_end + 4; // past the "\r\n\r\n"
        // saturating so a crafted Content-Length can't overflow the offset math
        // into an out-of-bounds slice (a reachable panic / DoS).
        let body_end = body_start.saturating_add(content_length);
        if buf.len() < body_end {
            return None; // body not fully arrived
        }
        let body = buf[body_start..body_end].to_vec();
        Some((RtspRequest { method, uri, cseq, transport, content_length, body }, body_end))
    }
}

/// Build the SDP an RTSP server offers for one H.264 stream over RTP/AVP at the
/// given dynamic payload type (90 kHz clock). Geometry rides in-band in the SPS,
/// so no `sprop-parameter-sets` are emitted (Annex-B/SIMPLE clients tune in on
/// the next keyframe).
pub fn sdp_h264(payload_type: u8) -> String {
    format!(
        "v=0\r\n\
         o=- 0 0 IN IP4 0.0.0.0\r\n\
         s=g2g\r\n\
         c=IN IP4 0.0.0.0\r\n\
         t=0 0\r\n\
         m=video 0 RTP/AVP {pt}\r\n\
         a=rtpmap:{pt} H264/90000\r\n\
         a=control:streamid=0\r\n",
        pt = payload_type & 0x7F,
    )
}

/// Sans-IO RTSP server responder for one session.
#[derive(Debug)]
pub struct RtspResponder {
    sdp: String,
    state: State,
    session_id: String,
    server_rtp_port: u16,
    ssrc: u32,
    client_rtp_port: Option<u16>,
    /// `(rtp_channel, rtcp_channel)` once a TCP-interleaved SETUP has been handled.
    interleaved: Option<(u8, u8)>,
}

impl RtspResponder {
    /// `sdp` is served in `DESCRIBE`; `server_rtp_port` is the UDP port this
    /// server sends RTP from (advertised in the SETUP response); `ssrc` is the
    /// RTP synchronization source.
    pub fn new(sdp: impl Into<String>, server_rtp_port: u16, ssrc: u32) -> Self {
        Self {
            sdp: sdp.into(),
            state: State::Init,
            session_id: format!("{ssrc:08X}"),
            server_rtp_port,
            ssrc,
            client_rtp_port: None,
            interleaved: None,
        }
    }

    /// The negotiated client RTP port, once a UDP `SETUP` has been handled.
    pub fn client_rtp_port(&self) -> Option<u16> {
        self.client_rtp_port
    }

    /// The negotiated `(rtp_channel, rtcp_channel)`, once a TCP-interleaved
    /// `SETUP` has been handled (RFC 2326 §10.12).
    pub fn interleaved_channels(&self) -> Option<(u8, u8)> {
        self.interleaved
    }

    /// The session identifier assigned at `SETUP`.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Handle one parsed request: returns the response bytes to send and the
    /// action the I/O layer should take.
    pub fn handle_request(&mut self, req: &RtspRequest) -> (Vec<u8>, RtspEvent) {
        match req.method.as_str() {
            "OPTIONS" => (
                self.respond(
                    req.cseq,
                    "200 OK",
                    &[(
                        "Public",
                        "OPTIONS, DESCRIBE, SETUP, PLAY, PAUSE, TEARDOWN, ANNOUNCE, RECORD, GET_PARAMETER",
                    )],
                    b"",
                ),
                RtspEvent::None,
            ),
            "DESCRIBE" => {
                let base = format!("{};", req.uri);
                let sdp = self.sdp.clone();
                (
                    self.respond(
                        req.cseq,
                        "200 OK",
                        &[("Content-Type", "application/sdp"), ("Content-Base", &base)],
                        sdp.as_bytes(),
                    ),
                    RtspEvent::None,
                )
            }
            // A publisher describing the stream it is about to RECORD.
            "ANNOUNCE" => {
                if !req.body.is_empty() {
                    if let Ok(sdp) = core::str::from_utf8(&req.body) {
                        self.sdp = sdp.to_string();
                    }
                }
                (self.respond(req.cseq, "200 OK", &[], b""), RtspEvent::None)
            }
            "SETUP" => {
                self.state = State::Ready;
                let session = self.session_id.clone();
                // TCP-interleaved transport (RFC 2326 §10.12): RTP / RTCP ride the
                // control connection on the negotiated channels; no UDP ports.
                if let Some((rtp_ch, rtcp_ch)) =
                    req.transport.as_deref().and_then(parse_interleaved_channels)
                {
                    self.interleaved = Some((rtp_ch, rtcp_ch));
                    let transport = format!(
                        "RTP/AVP/TCP;unicast;interleaved={rtp_ch}-{rtcp_ch};ssrc={:08X}",
                        self.ssrc,
                    );
                    let resp = self.respond(
                        req.cseq,
                        "200 OK",
                        &[("Transport", &transport), ("Session", &session)],
                        b"",
                    );
                    return (
                        resp,
                        RtspEvent::SetupInterleaved { rtp_channel: rtp_ch, rtcp_channel: rtcp_ch },
                    );
                }
                self.client_rtp_port = req.transport.as_deref().and_then(parse_client_rtp_port);
                let transport = format!(
                    "RTP/AVP;unicast;client_port={}-{};server_port={}-{};ssrc={:08X}",
                    self.client_rtp_port.unwrap_or(0),
                    self.client_rtp_port.map(|p| p.saturating_add(1)).unwrap_or(0),
                    self.server_rtp_port,
                    self.server_rtp_port.saturating_add(1),
                    self.ssrc,
                );
                let resp = self.respond(
                    req.cseq,
                    "200 OK",
                    &[("Transport", &transport), ("Session", &session)],
                    b"",
                );
                match self.client_rtp_port {
                    Some(port) => (resp, RtspEvent::Setup { client_rtp_port: port }),
                    None => (resp, RtspEvent::None),
                }
            }
            "PLAY" => {
                self.state = State::Playing;
                let session = self.session_id.clone();
                let rtp_info = format!("url={};seq=0;rtptime=0", req.uri);
                (
                    self.respond(
                        req.cseq,
                        "200 OK",
                        &[("Session", &session), ("RTP-Info", &rtp_info)],
                        b"",
                    ),
                    RtspEvent::Play,
                )
            }
            "RECORD" => {
                self.state = State::Recording;
                let session = self.session_id.clone();
                (
                    self.respond(req.cseq, "200 OK", &[("Session", &session)], b""),
                    RtspEvent::Record,
                )
            }
            "PAUSE" => {
                self.state = State::Ready;
                let session = self.session_id.clone();
                (self.respond(req.cseq, "200 OK", &[("Session", &session)], b""), RtspEvent::None)
            }
            "TEARDOWN" => {
                self.state = State::Init;
                (self.respond(req.cseq, "200 OK", &[], b""), RtspEvent::Teardown)
            }
            // Common keepalive during PLAY.
            "GET_PARAMETER" | "SET_PARAMETER" => {
                (self.respond(req.cseq, "200 OK", &[], b""), RtspEvent::None)
            }
            _ => (self.respond(req.cseq, "501 Not Implemented", &[], b""), RtspEvent::None),
        }
    }

    /// Assemble an RTSP response: status line, echoed CSeq, the extra headers, a
    /// `Content-Length` when there is a body, then the blank line and body.
    fn respond(&self, cseq: u32, status: &str, headers: &[(&str, &str)], body: &[u8]) -> Vec<u8> {
        let mut out = String::new();
        out.push_str("RTSP/1.0 ");
        out.push_str(status);
        out.push_str("\r\n");
        out.push_str(&format!("CSeq: {cseq}\r\n"));
        out.push_str("Server: g2g\r\n");
        for (k, v) in headers {
            out.push_str(&format!("{k}: {v}\r\n"));
        }
        if !body.is_empty() {
            out.push_str(&format!("Content-Length: {}\r\n", body.len()));
        }
        out.push_str("\r\n");
        let mut bytes = out.into_bytes();
        bytes.extend_from_slice(body);
        bytes
    }
}

/// Find the index of the `\r\n\r\n` that ends the header block.
fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

/// Pull the first port of a `client_port=RTP-RTCP` pair out of a Transport
/// header (`RTP/AVP;unicast;client_port=5000-5001`).
fn parse_client_rtp_port(transport: &str) -> Option<u16> {
    let after = transport.split("client_port=").nth(1)?;
    let range = after.split(';').next()?;
    let first = range.split('-').next()?;
    first.trim().parse().ok()
}

/// Recognize a TCP-interleaved Transport (RFC 2326 §10.12) and pull its
/// `(rtp_channel, rtcp_channel)` out of `interleaved=N-M`
/// (`RTP/AVP/TCP;unicast;interleaved=0-1`). Requires the `TCP` lower-transport
/// token so a UDP Transport that (unusually) carried an `interleaved=` param is
/// not misread as interleaved. A single `interleaved=N` (no RTCP channel)
/// defaults the RTCP channel to `N + 1`.
fn parse_interleaved_channels(transport: &str) -> Option<(u8, u8)> {
    // The lower transport is the third `/`-token of the profile (RTP/AVP/TCP).
    let is_tcp = transport
        .split(';')
        .next()
        .map(|profile| profile.split('/').any(|t| t.eq_ignore_ascii_case("TCP")))
        .unwrap_or(false);
    if !is_tcp {
        return None;
    }
    let after = transport.split("interleaved=").nth(1)?;
    let range = after.split(';').next()?;
    let mut parts = range.split('-');
    let rtp_ch: u8 = parts.next()?.trim().parse().ok()?;
    let rtcp_ch: u8 = match parts.next() {
        Some(m) => m.trim().parse().ok()?,
        None => rtp_ch.saturating_add(1),
    };
    Some((rtp_ch, rtcp_ch))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(text: &str) -> RtspRequest {
        RtspRequest::parse(text.as_bytes()).expect("parses").0
    }

    fn responder() -> RtspResponder {
        RtspResponder::new(sdp_h264(96), 6000, 0x1234_5678)
    }

    #[test]
    fn parses_request_line_and_headers() {
        let r = request("SETUP rtsp://h/s/streamid=0 RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP;unicast;client_port=5000-5001\r\n\r\n");
        assert_eq!(r.method, "SETUP");
        assert_eq!(r.uri, "rtsp://h/s/streamid=0");
        assert_eq!(r.cseq, 3);
        assert_eq!(r.transport.as_deref(), Some("RTP/AVP;unicast;client_port=5000-5001"));
    }

    #[test]
    fn parse_waits_for_full_body() {
        // Content-Length 10 but no body bytes yet -> incomplete.
        let partial = "ANNOUNCE rtsp://h/s RTSP/1.0\r\nCSeq: 1\r\nContent-Length: 10\r\n\r\n";
        assert!(RtspRequest::parse(partial.as_bytes()).is_none());
        let full = "ANNOUNCE rtsp://h/s RTSP/1.0\r\nCSeq: 1\r\nContent-Length: 10\r\n\r\n0123456789";
        let (r, consumed) = RtspRequest::parse(full.as_bytes()).expect("complete");
        assert_eq!(r.body, b"0123456789");
        assert_eq!(consumed, full.len());
    }

    #[test]
    fn overflowing_content_length_does_not_panic() {
        // A Content-Length near usize::MAX must not overflow the body-offset math
        // into an out-of-bounds slice; it reads as a not-yet-complete body.
        let req = "ANNOUNCE rtsp://h/s RTSP/1.0\r\nCSeq: 1\r\nContent-Length: 18446744073709551615\r\n\r\nx";
        assert!(RtspRequest::parse(req.as_bytes()).is_none());
    }

    #[test]
    fn setup_with_max_client_port_does_not_overflow() {
        let mut s = responder();
        let req = request(
            "SETUP rtsp://h/s RTSP/1.0\r\nCSeq: 2\r\nTransport: RTP/AVP;unicast;client_port=65535-65535\r\n\r\n",
        );
        let (resp, _ev) = s.handle_request(&req);
        let text = core::str::from_utf8(&resp).unwrap();
        assert!(text.starts_with("RTSP/1.0 200 OK\r\n"));
        // The client RTCP port saturates instead of wrapping past u16.
        assert!(text.contains("client_port=65535-65535"), "{text}");
    }

    #[test]
    fn options_lists_methods_and_echoes_cseq() {
        let mut s = responder();
        let (resp, ev) = s.handle_request(&request("OPTIONS * RTSP/1.0\r\nCSeq: 1\r\n\r\n"));
        let text = core::str::from_utf8(&resp).unwrap();
        assert!(text.starts_with("RTSP/1.0 200 OK\r\n"));
        assert!(text.contains("CSeq: 1\r\n"));
        assert!(text.contains("Public: OPTIONS, DESCRIBE, SETUP, PLAY"));
        assert_eq!(ev, RtspEvent::None);
    }

    #[test]
    fn describe_returns_sdp_with_content_length() {
        let mut s = responder();
        let (resp, _) = s.handle_request(&request("DESCRIBE rtsp://h/s RTSP/1.0\r\nCSeq: 2\r\n\r\n"));
        let text = core::str::from_utf8(&resp).unwrap();
        assert!(text.contains("Content-Type: application/sdp\r\n"));
        assert!(text.contains("m=video 0 RTP/AVP 96\r\n"));
        assert!(text.contains("a=rtpmap:96 H264/90000\r\n"));
        // Content-Length must equal the SDP body length.
        let body = text.split("\r\n\r\n").nth(1).unwrap();
        assert!(text.contains(&format!("Content-Length: {}\r\n", body.len())));
    }

    #[test]
    fn full_play_handshake_negotiates_transport_and_starts() {
        let mut s = responder();
        let _ = s.handle_request(&request("OPTIONS * RTSP/1.0\r\nCSeq: 1\r\n\r\n"));
        let _ = s.handle_request(&request("DESCRIBE rtsp://h/s RTSP/1.0\r\nCSeq: 2\r\n\r\n"));

        let (setup, ev) = s.handle_request(&request(
            "SETUP rtsp://h/s/streamid=0 RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP;unicast;client_port=5000-5001\r\n\r\n",
        ));
        let text = core::str::from_utf8(&setup).unwrap();
        assert!(text.contains("server_port=6000-6001"), "advertises the server RTP port pair");
        assert!(text.contains("client_port=5000-5001"));
        assert!(text.contains("Session: 12345678\r\n"));
        assert_eq!(ev, RtspEvent::Setup { client_rtp_port: 5000 });
        assert_eq!(s.client_rtp_port(), Some(5000));

        let (_, ev) = s.handle_request(&request("PLAY rtsp://h/s RTSP/1.0\r\nCSeq: 4\r\nSession: 12345678\r\n\r\n"));
        assert_eq!(ev, RtspEvent::Play);

        let (_, ev) = s.handle_request(&request("TEARDOWN rtsp://h/s RTSP/1.0\r\nCSeq: 5\r\nSession: 12345678\r\n\r\n"));
        assert_eq!(ev, RtspEvent::Teardown);
    }

    #[test]
    fn announce_record_path_accepts_sdp_and_arms_receive() {
        let mut s = responder();
        let announce = "ANNOUNCE rtsp://h/s RTSP/1.0\r\nCSeq: 1\r\nContent-Type: application/sdp\r\nContent-Length: 10\r\n\r\nv=0\r\no=- 0";
        let (resp, ev) = s.handle_request(&request(announce));
        assert!(core::str::from_utf8(&resp).unwrap().starts_with("RTSP/1.0 200 OK"));
        assert_eq!(ev, RtspEvent::None);

        let (_, ev) = s.handle_request(&request(
            "SETUP rtsp://h/s RTSP/1.0\r\nCSeq: 2\r\nTransport: RTP/AVP;unicast;client_port=7000-7001\r\n\r\n",
        ));
        assert_eq!(ev, RtspEvent::Setup { client_rtp_port: 7000 });
        let (_, ev) = s.handle_request(&request("RECORD rtsp://h/s RTSP/1.0\r\nCSeq: 3\r\nSession: 12345678\r\n\r\n"));
        assert_eq!(ev, RtspEvent::Record);
    }

    #[test]
    fn setup_tcp_interleaved_negotiates_channels() {
        let mut s = responder();
        let (resp, ev) = s.handle_request(&request(
            "SETUP rtsp://h/s/streamid=0 RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n\r\n",
        ));
        let text = core::str::from_utf8(&resp).unwrap();
        assert!(text.starts_with("RTSP/1.0 200 OK\r\n"));
        assert!(text.contains("RTP/AVP/TCP;unicast;interleaved=0-1"), "{text}");
        assert_eq!(ev, RtspEvent::SetupInterleaved { rtp_channel: 0, rtcp_channel: 1 });
        assert_eq!(s.interleaved_channels(), Some((0, 1)));
        assert_eq!(s.client_rtp_port(), None, "no UDP client port for interleaved");
    }

    #[test]
    fn setup_interleaved_defaults_rtcp_channel() {
        // A single `interleaved=2` (no RTCP channel) defaults RTCP to 3.
        let mut s = responder();
        let (_, ev) = s.handle_request(&request(
            "SETUP rtsp://h/s RTSP/1.0\r\nCSeq: 3\r\nTransport: RTP/AVP/TCP;unicast;interleaved=2\r\n\r\n",
        ));
        assert_eq!(ev, RtspEvent::SetupInterleaved { rtp_channel: 2, rtcp_channel: 3 });
    }

    #[test]
    fn udp_setup_is_not_misread_as_interleaved() {
        // A UDP Transport (no TCP token) picks the UDP path even if some odd param
        // mentioned interleaving; the profile lower-transport is what decides.
        assert_eq!(parse_interleaved_channels("RTP/AVP;unicast;client_port=5000-5001"), None);
        assert_eq!(parse_interleaved_channels("RTP/AVP/TCP;unicast;interleaved=4-5"), Some((4, 5)));
    }

    #[test]
    fn unknown_method_is_not_implemented() {
        let mut s = responder();
        let (resp, _) = s.handle_request(&request("FROBNICATE * RTSP/1.0\r\nCSeq: 9\r\n\r\n"));
        assert!(core::str::from_utf8(&resp).unwrap().starts_with("RTSP/1.0 501 Not Implemented"));
    }
}
