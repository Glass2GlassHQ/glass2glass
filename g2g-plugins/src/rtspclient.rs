use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use md5::{Digest, Md5};

use g2g_core::{PropError, PropValue};

use crate::rtspserver::RtspMessage;
use crate::textencoding::{base64_encode, hex_encode};

/// An RTP lower transport a SETUP may request, the subset of gst's
/// `GstRTSPLowerTrans` g2g implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerTransport {
    Udp,
    Tcp,
}

impl LowerTransport {
    pub fn nick(self) -> &'static str {
        match self {
            LowerTransport::Udp => "udp",
            LowerTransport::Tcp => "tcp",
        }
    }

    /// Parse a `protocols=` flag nick.
    pub fn from_nick(nick: &str) -> Option<Self> {
        match nick {
            "udp" => Some(LowerTransport::Udp),
            "tcp" => Some(LowerTransport::Tcp),
            _ => None,
        }
    }
}

// a `protocols=` value as the transports to try in the order written
pub fn transports_from_flags(value: &PropValue) -> Result<Vec<LowerTransport>, PropError> {
    let mut order = Vec::new();
    for nick in value.as_flags().ok_or(PropError::Type)? {
        let transport = LowerTransport::from_nick(nick.as_str()).ok_or(PropError::Value)?;
        if !order.contains(&transport) {
            order.push(transport);
        }
    }
    if order.is_empty() {
        return Err(PropError::Value);
    }
    Ok(order)
}

pub fn transports_to_flags(transports: &[LowerTransport]) -> PropValue {
    PropValue::Flags(transports.iter().map(|t| t.nick().to_string()).collect())
}

pub const STATUS_OK: u16 = 200;
pub const STATUS_UNAUTHORIZED: u16 = 401;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtspClientError {
    MalformedResponse,
    UnexpectedCseq { sent: u32, received: u32 },
    Refused { method: String, status_line: String },
    CredentialsNeeded,
    UnsupportedChallenge,
    MissingSession,
    MissingTransport,
}

impl core::fmt::Display for RtspClientError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RtspClientError::MalformedResponse => write!(f, "malformed RTSP response"),
            RtspClientError::UnexpectedCseq { sent, received } => {
                write!(
                    f,
                    "response CSeq {received} does not answer request CSeq {sent}"
                )
            }
            RtspClientError::Refused {
                method,
                status_line,
            } => write!(f, "{method} refused: {status_line}"),
            RtspClientError::CredentialsNeeded => {
                write!(f, "server requires authentication and no user-id is set")
            }
            RtspClientError::UnsupportedChallenge => {
                write!(f, "server offers no Basic or MD5 Digest authentication")
            }
            RtspClientError::MissingSession => write!(f, "SETUP response has no Session"),
            RtspClientError::MissingTransport => {
                write!(f, "SETUP response has no usable Transport")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtspResponse {
    pub status: u16,
    pub status_line: String,
    pub cseq: u32,
    pub session: Option<String>,
    pub transport: Option<String>,
    pub www_authenticate: Vec<String>,
}

impl RtspResponse {
    // Ok(None) while the response has not fully arrived
    pub fn parse(buf: &[u8]) -> Result<Option<(Self, usize)>, RtspClientError> {
        let Some(message) = RtspMessage::parse(buf) else {
            return Ok(None);
        };
        let status = message
            .start_line
            .strip_prefix("RTSP/1.0 ")
            .and_then(|rest| rest.split(' ').next())
            .and_then(|code| code.parse().ok())
            .ok_or(RtspClientError::MalformedResponse)?;
        // the session id ends at the `;timeout=` parameter
        let session = message
            .header("Session")
            .and_then(|value| value.split(';').next())
            .map(|id| id.trim().to_string());
        let response = RtspResponse {
            status,
            status_line: message.start_line.to_string(),
            cseq: message.cseq(),
            session,
            transport: message.header("Transport").map(str::to_string),
            www_authenticate: message
                .headers_named("WWW-Authenticate")
                .map(str::to_string)
                .collect(),
        };
        Ok(Some((response, message.consumed)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Credentials {
    pub user: String,
    pub password: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Challenge {
    Basic,
    Digest {
        realm: String,
        nonce: String,
        opaque: Option<String>,
    },
}

impl Challenge {
    fn parse(header: &str) -> Option<Self> {
        let (scheme, params) = header.split_once(' ').unwrap_or((header, ""));
        if scheme.eq_ignore_ascii_case("Basic") {
            return Some(Challenge::Basic);
        }
        if !scheme.eq_ignore_ascii_case("Digest") {
            return None;
        }
        let md5 = auth_param(params, "algorithm").is_none_or(|a| a.eq_ignore_ascii_case("MD5"));
        if !md5 {
            return None;
        }
        Some(Challenge::Digest {
            realm: auth_param(params, "realm")?.to_string(),
            nonce: auth_param(params, "nonce")?.to_string(),
            opaque: auth_param(params, "opaque").map(str::to_string),
        })
    }

    fn authorization(&self, credentials: &Credentials, method: &str, uri: &str) -> String {
        match self {
            Challenge::Basic => {
                let pair = format!("{}:{}", credentials.user, credentials.password);
                format!("Basic {}", base64_encode(pair.as_bytes()))
            }
            // RFC 2617 §3.2.2 without qop: MD5(MD5(user:realm:password):nonce:MD5(method:uri))
            Challenge::Digest {
                realm,
                nonce,
                opaque,
            } => {
                let user_hash = md5_hex(&format!(
                    "{}:{realm}:{}",
                    credentials.user, credentials.password
                ));
                let request_hash = md5_hex(&format!("{method}:{uri}"));
                let response = md5_hex(&format!("{user_hash}:{nonce}:{request_hash}"));
                let mut header = format!(
                    "Digest username=\"{}\", realm=\"{realm}\", nonce=\"{nonce}\", uri=\"{uri}\", response=\"{response}\"",
                    credentials.user
                );
                if let Some(opaque) = opaque {
                    header.push_str(&format!(", opaque=\"{opaque}\""));
                }
                header
            }
        }
    }
}

fn md5_hex(text: &str) -> String {
    hex_encode(&Md5::digest(text.as_bytes()))
}

// one `key=value` or `key="value"` parameter of a WWW-Authenticate header
fn auth_param<'a>(params: &'a str, key: &str) -> Option<&'a str> {
    params.split(',').find_map(|param| {
        let (k, v) = param.split_once('=')?;
        k.trim()
            .eq_ignore_ascii_case(key)
            .then(|| v.trim().trim_matches('"'))
    })
}

// the request side of one publishing connection: CSeq numbering, the session id and the authorization answer
#[derive(Debug)]
pub struct RtspPublishRequests {
    user_agent: String,
    credentials: Option<Credentials>,
    challenge: Option<Challenge>,
    last_cseq: u32,
    session: Option<String>,
}

impl RtspPublishRequests {
    pub fn new(user_agent: String, credentials: Option<Credentials>) -> Self {
        Self {
            user_agent,
            credentials,
            challenge: None,
            last_cseq: 0,
            session: None,
        }
    }

    pub fn request(
        &mut self,
        method: &str,
        uri: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Vec<u8> {
        self.last_cseq += 1;
        let mut out = format!("{method} {uri} RTSP/1.0\r\nCSeq: {}\r\n", self.last_cseq);
        out.push_str(&format!("User-Agent: {}\r\n", self.user_agent));
        if let Some(session) = &self.session {
            out.push_str(&format!("Session: {session}\r\n"));
        }
        if let (Some(challenge), Some(credentials)) = (&self.challenge, &self.credentials) {
            out.push_str(&format!(
                "Authorization: {}\r\n",
                challenge.authorization(credentials, method, uri)
            ));
        }
        for (key, value) in headers {
            out.push_str(&format!("{key}: {value}\r\n"));
        }
        if !body.is_empty() {
            out.push_str(&format!("Content-Length: {}\r\n", body.len()));
        }
        out.push_str("\r\n");
        let mut bytes = out.into_bytes();
        bytes.extend_from_slice(body);
        bytes
    }

    pub fn check_cseq(&self, response: &RtspResponse) -> Result<(), RtspClientError> {
        if response.cseq != self.last_cseq {
            return Err(RtspClientError::UnexpectedCseq {
                sent: self.last_cseq,
                received: response.cseq,
            });
        }
        Ok(())
    }

    // takes the challenge of a 401 so the resent request carries an Authorization
    pub fn answer_challenge(&mut self, response: &RtspResponse) -> Result<(), RtspClientError> {
        if self.credentials.is_none() {
            return Err(RtspClientError::CredentialsNeeded);
        }
        let challenges: Vec<Challenge> = response
            .www_authenticate
            .iter()
            .filter_map(|header| Challenge::parse(header))
            .collect();
        let digest = challenges
            .iter()
            .find(|c| matches!(c, Challenge::Digest { .. }));
        let chosen = digest
            .or(challenges.first())
            .ok_or(RtspClientError::UnsupportedChallenge)?;
        self.challenge = Some(chosen.clone());
        Ok(())
    }

    pub fn accept_session(&mut self, response: &RtspResponse) -> Result<(), RtspClientError> {
        let session = response
            .session
            .clone()
            .ok_or(RtspClientError::MissingSession)?;
        self.session = Some(session);
        Ok(())
    }
}

pub fn record_transport_tcp(rtp_channel: u8) -> String {
    format!(
        "RTP/AVP/TCP;unicast;interleaved={rtp_channel}-{};mode=record",
        rtp_channel + 1
    )
}

pub fn record_transport_udp(client_rtp_port: u16) -> String {
    format!(
        "RTP/AVP;unicast;client_port={client_rtp_port}-{};mode=record",
        client_rtp_port + 1
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(text: &str) -> RtspResponse {
        RtspResponse::parse(text.as_bytes())
            .expect("well formed")
            .expect("complete")
            .0
    }

    fn credentials() -> Credentials {
        Credentials {
            user: "Mufasa".to_string(),
            password: "Circle Of Life".to_string(),
        }
    }

    #[test]
    fn response_parse_strips_the_session_timeout() {
        let r = response(
            "RTSP/1.0 200 OK\r\nCSeq: 3\r\nSession: c688f40b;timeout=55\r\nTransport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n\r\n",
        );
        assert_eq!(r.status, STATUS_OK);
        assert_eq!(r.cseq, 3);
        assert_eq!(r.session.as_deref(), Some("c688f40b"));
        assert_eq!(
            r.transport.as_deref(),
            Some("RTP/AVP/TCP;unicast;interleaved=0-1")
        );
    }

    #[test]
    fn response_parse_waits_for_the_body_and_rejects_a_request() {
        let partial = b"RTSP/1.0 200 OK\r\nCSeq: 1\r\nContent-Length: 4\r\n\r\nab";
        assert_eq!(RtspResponse::parse(partial), Ok(None));
        let request = b"OPTIONS rtsp://h/s RTSP/1.0\r\nCSeq: 1\r\n\r\n";
        assert_eq!(
            RtspResponse::parse(request),
            Err(RtspClientError::MalformedResponse)
        );
    }

    // RFC 2617 §3.5 example, with the RTSP method and the RFC's realm, nonce and URI
    #[test]
    fn digest_answer_matches_the_rfc_2617_construction() {
        let realm = "testrealm@host.com";
        let nonce = "dcd98b7102dd2f0e8b11d0f600bfb0c093";
        let uri = "/dir/index.html";
        let mut requests = RtspPublishRequests::new("g2g".to_string(), Some(credentials()));
        let challenge = response(&format!(
            "RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\nWWW-Authenticate: Basic realm=\"{realm}\"\r\nWWW-Authenticate: Digest realm=\"{realm}\", nonce=\"{nonce}\"\r\n\r\n"
        ));
        requests
            .answer_challenge(&challenge)
            .expect("digest offered");
        let request = String::from_utf8(requests.request("GET", uri, &[], b"")).unwrap();
        let user_hash = md5_hex(&format!("Mufasa:{realm}:Circle Of Life"));
        let request_hash = md5_hex(&format!("GET:{uri}"));
        let expected = md5_hex(&format!("{user_hash}:{nonce}:{request_hash}"));
        assert!(
            request.contains(&format!("response=\"{expected}\"")),
            "{request}"
        );
        assert!(request.contains("Authorization: Digest username=\"Mufasa\""));
    }

    #[test]
    fn md5_hex_matches_a_known_digest() {
        // RFC 1321 appendix A.5 test suite
        assert_eq!(md5_hex("abc"), "900150983cd24fb0d6963f7d28e17f72");
    }

    #[test]
    fn basic_answer_is_base64_of_user_and_password() {
        let mut requests = RtspPublishRequests::new("g2g".to_string(), Some(credentials()));
        let challenge = response(
            "RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\nWWW-Authenticate: Basic realm=\"x\"\r\n\r\n",
        );
        requests
            .answer_challenge(&challenge)
            .expect("basic offered");
        let request =
            String::from_utf8(requests.request("ANNOUNCE", "rtsp://h/s", &[], b"")).unwrap();
        // RFC 7617 §2 example credentials
        let expected = base64_encode(b"Mufasa:Circle Of Life");
        assert!(
            request.contains(&format!("Authorization: Basic {expected}\r\n")),
            "{request}"
        );
    }

    #[test]
    fn a_challenge_without_credentials_fails_loud() {
        let challenge = response(
            "RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\nWWW-Authenticate: Basic realm=\"x\"\r\n\r\n",
        );
        let mut anonymous = RtspPublishRequests::new("g2g".to_string(), None);
        assert_eq!(
            anonymous.answer_challenge(&challenge),
            Err(RtspClientError::CredentialsNeeded)
        );
    }

    #[test]
    fn requests_number_cseq_and_carry_the_session() {
        let mut requests = RtspPublishRequests::new("g2g".to_string(), None);
        let first = String::from_utf8(requests.request("OPTIONS", "rtsp://h/s", &[], b"")).unwrap();
        assert!(first.starts_with("OPTIONS rtsp://h/s RTSP/1.0\r\nCSeq: 1\r\n"));
        assert!(!first.contains("Session:"));
        let setup = response("RTSP/1.0 200 OK\r\nCSeq: 1\r\nSession: abc;timeout=60\r\n\r\n");
        requests.check_cseq(&setup).expect("answers CSeq 1");
        requests.accept_session(&setup).expect("session present");
        let record = String::from_utf8(requests.request("RECORD", "rtsp://h/s", &[], b"")).unwrap();
        assert!(
            record.contains("CSeq: 2\r\n") && record.contains("Session: abc\r\n"),
            "{record}"
        );
        assert_eq!(
            requests.check_cseq(&setup),
            Err(RtspClientError::UnexpectedCseq {
                sent: 2,
                received: 1
            })
        );
    }
}
