//! ONVIF camera discovery + RTSP stream-URI resolution.
//!
//! M311: an ONVIF camera does not stream over ONVIF; the ONVIF SOAP services
//! are a *control plane* that tells you the camera's RTSP URL (and could drive
//! PTZ, events, imaging). This element handles the two things needed to turn a
//! camera on the LAN into pixels:
//!
//! 1. **WS-Discovery** ([`discover`]): a single SOAP `Probe` multicast to
//!    `239.255.255.250:3702`; each camera answers with a `ProbeMatch` carrying
//!    its device-service `XAddrs`.
//! 2. **Stream-URI resolution** ([`resolve_stream_uri`]): three SOAP calls to
//!    the device, `GetCapabilities` (find the Media service) then
//!    `GetProfiles` (pick a media profile) then `GetStreamUri` (get the RTSP
//!    URL), authenticated with a WS-Security `UsernameToken` digest.
//!
//! [`OnvifSrc`] ties them together as a `SourceLoop`: it resolves the RTSP URI
//! during negotiation, then delegates to [`RtspSrc`](crate::rtspsrc::RtspSrc)
//! (threading the same credentials, since cameras gate the media stream behind
//! the device account). PTZ and event subscriptions are deliberately out of
//! scope for this milestone.
//!
//! The SOAP layer is hand-rolled (no `onvif`/`schema` crates): the request
//! bodies are fixed templates and the responses are read with `roxmltree`, so
//! the dependency footprint stays to reqwest + roxmltree + sha1 + base64, with
//! no git dependencies.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;
use std::string::{String, ToString};
use std::time::{SystemTime, UNIX_EPOCH};
use std::vec::Vec;
use std::{format, vec};

use alloc::boxed::Box;

use base64::Engine as _;
use sha1::{Digest, Sha1};

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, HardwareError,
    LatencyReport, OutputSink, PropError, PropKind, PropValue, PropertySpec,
};

use crate::rtspsrc::RtspSrc;

/// IANA WS-Discovery multicast group + port (SOAP-over-UDP).
const WS_DISCOVERY_ADDR: &str = "239.255.255.250:3702";

/// A camera found by [`discover`]. The `service_url` is the device-management
/// endpoint advertised in the `ProbeMatch` `XAddrs`; feed it to
/// [`resolve_stream_uri`] or [`OnvifSrc::new`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnvifDevice {
    /// First HTTP `XAddr` from the ProbeMatch (the devicemgmt service URL).
    pub service_url: String,
    /// The `<d:Types>` advertised (e.g. `dn:NetworkVideoTransmitter`), verbatim.
    pub types: String,
}

/// Probe the LAN for ONVIF cameras over WS-Discovery and collect their
/// device-service URLs. Sends one multicast `Probe` and gathers `ProbeMatch`
/// answers until `timeout` elapses (cameras reply once, but several may answer,
/// hence the drain rather than first-response-wins).
///
/// Returns the discovered devices (deduplicated by `service_url`); an empty
/// list means nothing answered in time, not an error. A bind/send failure
/// surfaces as [`G2gError::Hardware`].
pub async fn discover(timeout: Duration) -> Result<Vec<OnvifDevice>, G2gError> {
    let sock = tokio::net::UdpSocket::bind("0.0.0.0:0")
        .await
        .map_err(|_| hw())?;
    let probe = probe_message(&random_uuid());
    sock.send_to(probe.as_bytes(), WS_DISCOVERY_ADDR)
        .await
        .map_err(|_| hw())?;

    let mut devices: Vec<OnvifDevice> = Vec::new();
    let deadline = tokio::time::Instant::now() + timeout;
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tokio::time::timeout(remaining, sock.recv_from(&mut buf)).await {
            Ok(Ok((n, _from))) => {
                if let Some(dev) = parse_probe_match(&buf[..n]) {
                    if !devices.iter().any(|d| d.service_url == dev.service_url) {
                        devices.push(dev);
                    }
                }
            }
            // recv error or the per-iteration timeout firing: stop draining.
            Ok(Err(_)) | Err(_) => break,
        }
    }
    Ok(devices)
}

/// Resolve a camera's RTSP stream URL from its device-service URL.
///
/// Runs `GetCapabilities` (to locate the Media service `XAddr`), then
/// `GetProfiles` (to pick the first media profile token), then `GetStreamUri`
/// (to read the RTSP URL). When `user` is non-empty each request carries a
/// WS-Security `UsernameToken` digest; an empty `user` sends anonymous
/// requests (some cameras allow it). Returns the RTSP URL string.
pub async fn resolve_stream_uri(
    service_url: &str,
    user: &str,
    pass: &str,
) -> Result<String, G2gError> {
    let client = reqwest::Client::new();

    // 1. GetCapabilities(Media) -> media service XAddr.
    let caps_resp = soap_call(&client, service_url, user, pass, &get_capabilities_body()).await?;
    let media_url = parse_media_xaddr(&caps_resp).ok_or(G2gError::CapsMismatch)?;

    // 2. GetProfiles -> first profile token.
    let profiles_resp = soap_call(&client, &media_url, user, pass, &get_profiles_body()).await?;
    let token = parse_first_profile_token(&profiles_resp).ok_or(G2gError::CapsMismatch)?;

    // 3. GetStreamUri(token) -> RTSP URL.
    let uri_resp =
        soap_call(&client, &media_url, user, pass, &get_stream_uri_body(&token)).await?;
    parse_stream_uri(&uri_resp).ok_or(G2gError::CapsMismatch)
}

/// Source element: an ONVIF camera as an H.264 source. Resolves the RTSP URI
/// during negotiation, then delegates the whole RTSP/RTP loop to an inner
/// [`RtspSrc`] (credentials threaded through). Set the camera by
/// `device-service-url` plus `user` / `password`, either via the constructor or
/// the `gst-launch`-style properties.
#[allow(missing_debug_implementations)]
pub struct OnvifSrc {
    device_url: String,
    user: String,
    pass: String,
    /// Built lazily by [`ensure_resolved`](Self::ensure_resolved) the first
    /// time the runner negotiates: holds the resolved RTSP URI + credentials.
    inner: Option<RtspSrc>,
}

impl OnvifSrc {
    /// Create a source for the camera at `device_service_url` (the ONVIF
    /// devicemgmt endpoint, e.g. from [`discover`]). Credentials are set
    /// separately with [`with_credentials`](Self::with_credentials).
    pub fn new<S: Into<String>>(device_service_url: S) -> Self {
        Self {
            device_url: device_service_url.into(),
            user: String::new(),
            pass: String::new(),
            inner: None,
        }
    }

    /// Supply the ONVIF account. Used for the WS-Security SOAP digest and,
    /// because cameras reuse the same account for RTSP, threaded into the
    /// inner [`RtspSrc`] for the DESCRIBE/SETUP auth.
    pub fn with_credentials<U: Into<String>, P: Into<String>>(mut self, user: U, pass: P) -> Self {
        self.user = user.into();
        self.pass = pass.into();
        self
    }

    /// Resolve the RTSP URI (once) and build the inner `RtspSrc`. Idempotent:
    /// later calls (re-fixate retries, then `run`) reuse the built source.
    async fn ensure_resolved(&mut self) -> Result<(), G2gError> {
        if self.inner.is_some() {
            return Ok(());
        }
        if self.device_url.is_empty() {
            return Err(G2gError::NotConfigured);
        }
        let uri = resolve_stream_uri(&self.device_url, &self.user, &self.pass).await?;
        let mut src = RtspSrc::new(uri);
        if !self.user.is_empty() {
            src = src.with_credentials(self.user.clone(), self.pass.clone());
        }
        self.inner = Some(src);
        Ok(())
    }
}

impl SourceLoop for OnvifSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = Pin<Box<dyn Future<Output = Result<Caps, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        Box::pin(async move {
            self.ensure_resolved().await?;
            // `inner` is Some after a successful resolve.
            self.inner.as_mut().unwrap().intercept_caps().await
        })
    }

    async fn caps_constraint(&mut self) -> Result<CapsConstraint<'_>, G2gError> {
        self.ensure_resolved().await?;
        self.inner.as_mut().unwrap().caps_constraint().await
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // `intercept_caps` runs before `configure_pipeline`, so the inner
        // source exists by now; if not, negotiation never produced caps.
        self.inner
            .as_mut()
            .ok_or(G2gError::NotConfigured)?
            .configure_pipeline(absolute_caps)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            self.ensure_resolved().await?;
            self.inner.as_mut().unwrap().run(out).await
        })
    }

    fn latency(&self) -> LatencyReport {
        // Live camera: no on-demand production. Mirrors RtspSrc semantics.
        LatencyReport::live(0, None)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ONVIF source",
            "Source/Network",
            "Discovers an ONVIF camera's RTSP stream and receives H.264 via RtspSrc",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        ONVIF_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" | "device-service-url" => {
                self.device_url = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            "user" | "user-id" => {
                self.user = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            "password" => {
                self.pass = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" | "device-service-url" => Some(PropValue::Str(self.device_url.clone())),
            "user" | "user-id" => Some(PropValue::Str(self.user.clone())),
            // Password is write-only: never read it back.
            _ => None,
        }
    }
}

static ONVIF_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "ONVIF device-service URL (e.g. http://camera/onvif/device_service)",
    ),
    PropertySpec::new("user", PropKind::Str, "ONVIF account username"),
    PropertySpec::new("password", PropKind::Str, "ONVIF account password"),
];

// ---------------------------------------------------------------------------
// SOAP transport
// ---------------------------------------------------------------------------

fn hw() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// POST a SOAP 1.2 body to an ONVIF service and return the response text.
/// Wraps `body` in an envelope with a WS-Security header when `user` is set.
async fn soap_call(
    client: &reqwest::Client,
    url: &str,
    user: &str,
    pass: &str,
    body: &str,
) -> Result<String, G2gError> {
    let header = if user.is_empty() {
        String::new()
    } else {
        security_header(user, pass)
    };
    let envelope = soap_envelope(&header, body);
    let resp = client
        .post(url)
        .header("Content-Type", "application/soap+xml; charset=utf-8")
        .body(envelope)
        .send()
        .await
        .map_err(|_| hw())?;
    resp.text().await.map_err(|_| hw())
}

/// Wrap a header fragment + body fragment in a SOAP 1.2 envelope. The ONVIF
/// service namespaces (`tds`/`trt`/`tt`) are declared on the inner elements,
/// so the envelope only needs the SOAP namespace.
fn soap_envelope(header: &str, body: &str) -> String {
    format!(
        concat!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
            "<s:Envelope xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\">",
            "<s:Header>{}</s:Header>",
            "<s:Body>{}</s:Body>",
            "</s:Envelope>"
        ),
        header, body
    )
}

/// Build a WS-Security `UsernameToken` header with a password *digest*:
/// `Base64(SHA1(nonce ++ created ++ password))`, the standard ONVIF auth.
fn security_header(user: &str, pass: &str) -> String {
    let mut nonce = [0u8; 16];
    getrandom::getrandom(&mut nonce).expect("OS RNG for ONVIF WS-Security nonce");
    let created = iso8601_utc(now_unix_secs());
    let digest = password_digest(&nonce, &created, pass);
    let nonce_b64 = base64::engine::general_purpose::STANDARD.encode(nonce);
    format!(
        concat!(
            "<Security s:mustUnderstand=\"1\" ",
            "xmlns=\"http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd\">",
            "<UsernameToken>",
            "<Username>{user}</Username>",
            "<Password Type=\"http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest\">{digest}</Password>",
            "<Nonce EncodingType=\"http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary\">{nonce}</Nonce>",
            "<Created xmlns=\"http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd\">{created}</Created>",
            "</UsernameToken>",
            "</Security>"
        ),
        user = xml_escape(user),
        digest = digest,
        nonce = nonce_b64,
        created = created,
    )
}

/// `Base64(SHA1(nonce ++ created ++ password))` per WS-Security UsernameToken
/// Profile 1.0 (PasswordDigest).
fn password_digest(nonce: &[u8], created: &str, pass: &str) -> String {
    let mut h = Sha1::new();
    h.update(nonce);
    h.update(created.as_bytes());
    h.update(pass.as_bytes());
    base64::engine::general_purpose::STANDARD.encode(h.finalize())
}

// ---------------------------------------------------------------------------
// SOAP request bodies
// ---------------------------------------------------------------------------

fn probe_message(message_id: &str) -> String {
    format!(
        concat!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
            "<e:Envelope xmlns:e=\"http://www.w3.org/2003/05/soap-envelope\" ",
            "xmlns:w=\"http://schemas.xmlsoap.org/ws/2004/08/addressing\" ",
            "xmlns:d=\"http://schemas.xmlsoap.org/ws/2005/04/discovery\" ",
            "xmlns:dn=\"http://www.onvif.org/ver10/network/wsdl\">",
            "<e:Header>",
            "<w:MessageID>uuid:{id}</w:MessageID>",
            "<w:To e:mustUnderstand=\"1\">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>",
            "<w:Action e:mustUnderstand=\"1\">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>",
            "</e:Header>",
            "<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>",
            "</e:Envelope>"
        ),
        id = message_id
    )
}

fn get_capabilities_body() -> String {
    "<tds:GetCapabilities xmlns:tds=\"http://www.onvif.org/ver10/device/wsdl\">\
     <tds:Category>Media</tds:Category></tds:GetCapabilities>"
        .to_string()
}

fn get_profiles_body() -> String {
    "<trt:GetProfiles xmlns:trt=\"http://www.onvif.org/ver10/media/wsdl\"/>".to_string()
}

fn get_stream_uri_body(profile_token: &str) -> String {
    format!(
        concat!(
            "<trt:GetStreamUri xmlns:trt=\"http://www.onvif.org/ver10/media/wsdl\" ",
            "xmlns:tt=\"http://www.onvif.org/ver10/schema\">",
            "<trt:StreamSetup>",
            "<tt:Stream>RTP-Unicast</tt:Stream>",
            "<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport>",
            "</trt:StreamSetup>",
            "<trt:ProfileToken>{token}</trt:ProfileToken>",
            "</trt:GetStreamUri>"
        ),
        token = xml_escape(profile_token)
    )
}

// ---------------------------------------------------------------------------
// Response parsing (roxmltree, matched by local name to ignore namespaces)
// ---------------------------------------------------------------------------

/// First descendant element with the given local name, ignoring namespace.
fn find_local<'a, 'i>(
    root: roxmltree::Node<'a, 'i>,
    local: &str,
) -> Option<roxmltree::Node<'a, 'i>> {
    root.descendants()
        .find(|n| n.is_element() && n.tag_name().name() == local)
}

/// Pull the device-service URL out of a `ProbeMatch`: the first `http` entry in
/// the whitespace-separated `<d:XAddrs>` list.
fn parse_probe_match(bytes: &[u8]) -> Option<OnvifDevice> {
    let text = core::str::from_utf8(bytes).ok()?;
    let doc = roxmltree::Document::parse(text).ok()?;
    let xaddrs = find_local(doc.root(), "XAddrs")?.text()?;
    let service_url = xaddrs
        .split_whitespace()
        .find(|u| u.starts_with("http"))?
        .to_string();
    let types = find_local(doc.root(), "Types")
        .and_then(|n| n.text())
        .unwrap_or("")
        .to_string();
    Some(OnvifDevice { service_url, types })
}

/// The Media service `XAddr` from a `GetCapabilities` response.
fn parse_media_xaddr(xml: &str) -> Option<String> {
    let doc = roxmltree::Document::parse(xml).ok()?;
    let media = find_local(doc.root(), "Media")?;
    Some(find_local(media, "XAddr")?.text()?.to_string())
}

/// The first profile's `token` attribute from a `GetProfiles` response.
fn parse_first_profile_token(xml: &str) -> Option<String> {
    let doc = roxmltree::Document::parse(xml).ok()?;
    let profiles = find_local(doc.root(), "Profiles")?;
    profiles.attribute("token").map(|t| t.to_string())
}

/// The RTSP URL from a `GetStreamUri` response (`<tt:Uri>`).
fn parse_stream_uri(xml: &str) -> Option<String> {
    let doc = roxmltree::Document::parse(xml).ok()?;
    Some(find_local(doc.root(), "Uri")?.text()?.to_string())
}

// ---------------------------------------------------------------------------
// Small helpers (time, uuid, escaping)
// ---------------------------------------------------------------------------

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Format a UNIX timestamp as an ISO-8601 UTC `xsd:dateTime` (no fractional
/// seconds, `Z` zone), the form ONVIF expects for `Created`. Uses Howard
/// Hinnant's `civil_from_days` so it needs no chrono dependency.
fn iso8601_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let tod = secs % 86_400;
    let (hh, mm, ss) = (tod / 3_600, (tod % 3_600) / 60, tod % 60);

    // civil_from_days: days since 1970-01-01 -> (year, month, day).
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

    format!("{year:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Random UUID v4 (lowercase, hyphenated) for the WS-Discovery `MessageID`.
fn random_uuid() -> String {
    let mut b = [0u8; 16];
    getrandom::getrandom(&mut b).expect("OS RNG for ONVIF MessageID");
    b[6] = (b[6] & 0x0f) | 0x40; // version 4
    b[8] = (b[8] & 0x3f) | 0x80; // variant 1
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13],
        b[14], b[15]
    )
}

/// Minimal XML text/attribute escaping for the few values we interpolate
/// (username, profile token). ONVIF credentials rarely contain these, but a
/// password or token with `&`/`<` must not break the envelope.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso8601_formats_a_known_epoch() {
        // 2021-01-01T00:00:00Z = 1609459200.
        assert_eq!(iso8601_utc(1_609_459_200), "2021-01-01T00:00:00Z");
        // Same date + 12:34:56 (45296 s) exercises every time-of-day field.
        assert_eq!(iso8601_utc(1_609_459_200 + 45_296), "2021-01-01T12:34:56Z");
    }

    #[test]
    fn password_digest_matches_ws_security_vector() {
        // Deterministic inputs -> Base64(SHA1(nonce ++ created ++ pass)).
        let nonce = b"0123456789abcdef";
        let created = "2021-01-01T00:00:00Z";
        let digest = password_digest(nonce, created, "secret");
        // Recompute independently to pin the byte order (nonce, created, pass).
        let mut h = Sha1::new();
        h.update(nonce);
        h.update(created.as_bytes());
        h.update(b"secret");
        let expect = base64::engine::general_purpose::STANDARD.encode(h.finalize());
        assert_eq!(digest, expect);
        // A different password must change the digest.
        assert_ne!(digest, password_digest(nonce, created, "other"));
    }

    #[test]
    fn random_uuid_has_v4_layout() {
        let u = random_uuid();
        assert_eq!(u.len(), 36);
        let bytes: std::vec::Vec<&str> = u.split('-').collect();
        assert_eq!(bytes.len(), 5);
        // Version nibble is 4, variant nibble is 8/9/a/b.
        assert_eq!(&u[14..15], "4");
        assert!(matches!(&u[19..20], "8" | "9" | "a" | "b"));
    }

    #[test]
    fn parses_probe_match_xaddrs() {
        let xml = r#"<?xml version="1.0"?>
        <e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
                    xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery">
          <e:Body><d:ProbeMatches><d:ProbeMatch>
            <d:Types>dn:NetworkVideoTransmitter</d:Types>
            <d:XAddrs>http://192.168.1.50/onvif/device_service http://[fe80::1]/onvif/device_service</d:XAddrs>
          </d:ProbeMatch></d:ProbeMatches></e:Body>
        </e:Envelope>"#;
        let dev = parse_probe_match(xml.as_bytes()).expect("probe match parses");
        assert_eq!(dev.service_url, "http://192.168.1.50/onvif/device_service");
        assert_eq!(dev.types, "dn:NetworkVideoTransmitter");
    }

    #[test]
    fn parses_media_xaddr_from_capabilities() {
        let xml = r#"<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
          <s:Body><tds:GetCapabilitiesResponse xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
            <tds:Capabilities xmlns:tt="http://www.onvif.org/ver10/schema">
              <tt:Device><tt:XAddr>http://192.168.1.50/onvif/device_service</tt:XAddr></tt:Device>
              <tt:Media><tt:XAddr>http://192.168.1.50/onvif/Media</tt:XAddr></tt:Media>
            </tds:Capabilities>
          </tds:GetCapabilitiesResponse></s:Body>
        </s:Envelope>"#;
        assert_eq!(
            parse_media_xaddr(xml).as_deref(),
            Some("http://192.168.1.50/onvif/Media")
        );
    }

    #[test]
    fn parses_first_profile_token() {
        let xml = r#"<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
          <s:Body><trt:GetProfilesResponse xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
            <trt:Profiles token="MainStream" fixed="true"/>
            <trt:Profiles token="SubStream"/>
          </trt:GetProfilesResponse></s:Body>
        </s:Envelope>"#;
        assert_eq!(parse_first_profile_token(xml).as_deref(), Some("MainStream"));
    }

    #[test]
    fn parses_stream_uri() {
        let xml = r#"<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
          <s:Body><trt:GetStreamUriResponse xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
            <trt:MediaUri xmlns:tt="http://www.onvif.org/ver10/schema">
              <tt:Uri>rtsp://192.168.1.50:554/Streaming/Channels/101</tt:Uri>
              <tt:Timeout>PT60S</tt:Timeout>
            </trt:MediaUri>
          </trt:GetStreamUriResponse></s:Body>
        </s:Envelope>"#;
        assert_eq!(
            parse_stream_uri(xml).as_deref(),
            Some("rtsp://192.168.1.50:554/Streaming/Channels/101")
        );
    }

    #[test]
    fn envelope_includes_security_header_when_user_set() {
        let h = security_header("admin", "pw");
        assert!(h.contains("<Username>admin</Username>"));
        assert!(h.contains("PasswordDigest"));
        let env = soap_envelope(&h, &get_profiles_body());
        assert!(env.contains("<s:Header><Security"));
        assert!(env.contains("GetProfiles"));
    }

    #[test]
    fn xml_escape_handles_special_chars() {
        assert_eq!(xml_escape("a&b<c>\"'"), "a&amp;b&lt;c&gt;&quot;&apos;");
    }
}
