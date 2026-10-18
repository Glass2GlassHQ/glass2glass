use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::engine::general_purpose::{GeneralPurpose, GeneralPurposeConfig, STANDARD};
use base64::engine::DecodePaddingMode;
use base64::Engine as _;
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{Sink, SinkExt};
use hmac::{Hmac, Mac};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::{Error as WsError, Message};
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use str0m::{Candidate, Rtc};
use tokio::net::UdpSocket;

use g2g_core::{PropError, PropKind, PropValue, PropertySpec};

use crate::turn::TurnSet;
use crate::webrtc_util::{add_ice_candidates, IceTransportPolicy};

type HmacSha256 = Hmac<Sha256>;

const SERVICE: &str = "kinesisvideo";
const ALGORITHM: &str = "AWS4-HMAC-SHA256";
const SCOPE_TERMINATOR: &str = "aws4_request";
const SECRET_KEY_PREFIX: &str = "AWS4";
// the javascript sdk presigns for 299 s
const PRESIGNED_URL_EXPIRES_SECONDS: &str = "299";
const PRESIGNED_SIGNED_HEADERS: &str = "host";
const SECURE_WEBSOCKET_SCHEME: &str = "wss://";
const PLAIN_WEBSOCKET_SCHEME: &str = "ws://";
const SECURE_HTTP_SCHEME: &str = "https://";
const DESCRIBE_CHANNEL_PATH: &str = "/describeSignalingChannel";
const CHANNEL_ENDPOINT_PATH: &str = "/getSignalingChannelEndpoint";
const ICE_SERVER_CONFIG_PATH: &str = "/v1/get-ice-server-config";
const JSON_CONTENT_TYPE: &str = "application/json";
const WSS_PROTOCOL: &str = "WSS";
const HTTPS_PROTOCOL: &str = "HTTPS";
const STUN_HOST_PREFIX: &str = "stun.kinesisvideo.";
const STUN_HOST_SUFFIX: &str = ".amazonaws.com:443";
// the service rejects a longer messagePayload with a STATUS_RESPONSE
const MAX_MESSAGE_PAYLOAD_BYTES: usize = 10 * 1024;
const TURN_URI_SCHEMES: [&str; 2] = ["turn:", "turns:"];
// the turn client cannot speak dtls
const TURNS_OVER_UDP_PREFIX: &str = "turns:";
const UDP_TRANSPORT_SUFFIX: &str = "?transport=udp";
const CANDIDATE_LINE_PREFIX: &str = "a=candidate:";
const MID_LINE_PREFIX: &str = "a=mid:";
const SDP_ATTRIBUTE_PREFIX: &str = "a=";
const BUNDLED_MLINE_INDEX: u32 = 0;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
const HTTP_TIMEOUT: Duration = Duration::from_secs(20);
const MAX_CLIENT_ID_LENGTH: usize = 256;
const RESERVED_CLIENT_ID_PREFIX: &str = "aws_";

pub(crate) const OFFER_TYPE: &str = "offer";
pub(crate) const ANSWER_TYPE: &str = "answer";

pub(crate) const ENV_ACCESS_KEY: &str = "AWS_ACCESS_KEY_ID";
pub(crate) const ENV_SECRET_KEY: &str = "AWS_SECRET_ACCESS_KEY";
pub(crate) const ENV_SESSION_TOKEN: &str = "AWS_SESSION_TOKEN";
pub(crate) const ENV_REGION: &str = "AWS_REGION";
pub(crate) const ENV_DEFAULT_REGION: &str = "AWS_DEFAULT_REGION";

// other sdks may drop the base64 padding
const PAYLOAD_DECODER: GeneralPurpose = GeneralPurpose::new(
    &base64::alphabet::STANDARD,
    GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

pub(crate) type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;
pub(crate) type WsWrite = SplitSink<Ws, Message>;
pub(crate) type WsRead = SplitStream<Ws>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    Master,
    Viewer,
}

impl Role {
    fn name(self) -> &'static str {
        match self {
            Role::Master => "MASTER",
            Role::Viewer => "VIEWER",
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Credentials {
    access_key: String,
    secret_key: String,
    session_token: Option<String>,
}

impl core::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Credentials")
            .field("access_key", &self.access_key)
            .field("session_token", &self.session_token.is_some())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AmzTime {
    datetime: String,
}

impl AmzTime {
    pub(crate) fn from_unix_seconds(seconds: u64) -> Self {
        let (year, month, day, hour, minute, second) = crate::xmlutil::civil(seconds);
        Self {
            datetime: format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z"),
        }
    }

    pub(crate) fn now() -> Self {
        let since_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        Self::from_unix_seconds(since_epoch.as_secs())
    }

    fn date(&self) -> &str {
        &self.datetime[..8]
    }
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[usize::from(byte >> 4)] as char);
        out.push(DIGITS[usize::from(byte & 0x0f)] as char);
    }
    out
}

fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

fn hmac_sha256(key: &[u8], data: &str) -> [u8; 32] {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data.as_bytes());
    mac.finalize().into_bytes().into()
}

// sigv4 keeps only the rfc 3986 unreserved characters
fn uri_encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

fn canonical_query(parameters: &mut [(String, String)]) -> String {
    parameters.sort();
    parameters
        .iter()
        .map(|(name, value)| format!("{}={}", uri_encode(name), uri_encode(value)))
        .collect::<Vec<_>>()
        .join("&")
}

struct SigningScope<'a> {
    credentials: &'a Credentials,
    region: &'a str,
    service: &'a str,
    time: &'a AmzTime,
}

impl SigningScope<'_> {
    fn scope(&self) -> String {
        format!(
            "{}/{}/{}/{SCOPE_TERMINATOR}",
            self.time.date(),
            self.region,
            self.service
        )
    }

    fn credential(&self) -> String {
        format!("{}/{}", self.credentials.access_key, self.scope())
    }

    fn signature(&self, canonical_request: &str) -> String {
        let string_to_sign = [
            ALGORITHM,
            &self.time.datetime,
            &self.scope(),
            &sha256_hex(canonical_request.as_bytes()),
        ]
        .join("\n");
        let secret = format!("{SECRET_KEY_PREFIX}{}", self.credentials.secret_key);
        let date_key = hmac_sha256(secret.as_bytes(), self.time.date());
        let region_key = hmac_sha256(&date_key, self.region);
        let service_key = hmac_sha256(&region_key, self.service);
        let signing_key = hmac_sha256(&service_key, SCOPE_TERMINATOR);
        hex(&hmac_sha256(&signing_key, &string_to_sign))
    }

    fn request_headers(
        &self,
        method: &str,
        host: &str,
        path: &str,
        body: &[u8],
    ) -> Vec<(&'static str, String)> {
        let mut signed: Vec<(&'static str, String)> = alloc::vec![
            ("host", host.to_string()),
            ("x-amz-date", self.time.datetime.clone()),
        ];
        if let Some(token) = &self.credentials.session_token {
            signed.push(("x-amz-security-token", token.clone()));
        }
        let canonical_headers: String = signed
            .iter()
            .map(|(name, value)| format!("{name}:{}\n", value.trim()))
            .collect();
        let signed_names = signed
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join(";");
        let canonical_request = format!(
            "{method}\n{path}\n\n{canonical_headers}\n{signed_names}\n{}",
            sha256_hex(body)
        );
        let authorization = format!(
            "{ALGORITHM} Credential={}, SignedHeaders={signed_names}, Signature={}",
            self.credential(),
            self.signature(&canonical_request)
        );
        signed.retain(|(name, _)| *name != "host");
        signed.push(("authorization", authorization));
        signed
    }
}

pub(crate) fn presigned_url(
    credentials: &Credentials,
    region: &str,
    endpoint: &str,
    parameters: &[(&str, &str)],
    time: &AmzTime,
) -> Result<String, String> {
    let (scheme, rest) = [SECURE_WEBSOCKET_SCHEME, PLAIN_WEBSOCKET_SCHEME]
        .iter()
        .find_map(|scheme| endpoint.strip_prefix(scheme).map(|rest| (*scheme, rest)))
        .ok_or_else(|| format!("signalling endpoint {endpoint:?} is not a websocket url"))?;
    if rest.contains('?') {
        return Err(format!(
            "signalling endpoint {endpoint:?} already has a query"
        ));
    }
    let (host, path) = match rest.find('/') {
        Some(slash) => rest.split_at(slash),
        None => (rest, "/"),
    };
    let signing = SigningScope {
        credentials,
        region,
        service: SERVICE,
        time,
    };
    let mut query: Vec<(String, String)> = parameters
        .iter()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect();
    for (name, value) in [
        ("X-Amz-Algorithm", ALGORITHM.to_string()),
        ("X-Amz-Credential", signing.credential()),
        ("X-Amz-Date", time.datetime.clone()),
        ("X-Amz-Expires", PRESIGNED_URL_EXPIRES_SECONDS.to_string()),
        ("X-Amz-SignedHeaders", PRESIGNED_SIGNED_HEADERS.to_string()),
    ] {
        query.push((name.to_string(), value));
    }
    // kvs signalling wants the token inside the signed query
    if let Some(token) = &credentials.session_token {
        query.push(("X-Amz-Security-Token".to_string(), token.clone()));
    }
    let canonical_request = format!(
        "GET\n{path}\n{}\nhost:{host}\n\n{PRESIGNED_SIGNED_HEADERS}\n{}",
        canonical_query(&mut query),
        sha256_hex(b"")
    );
    query.push((
        "X-Amz-Signature".to_string(),
        signing.signature(&canonical_request),
    ));
    Ok(format!(
        "{scheme}{host}{path}?{}",
        canonical_query(&mut query)
    ))
}

// drops the signature and session token so the url can be logged
pub(crate) fn redacted_url(url: &str) -> &str {
    url.split('?').next().unwrap_or(url)
}

async fn post_signed(
    client: &reqwest::Client,
    credentials: &Credentials,
    region: &str,
    url: &str,
    body: &Value,
) -> Result<Value, String> {
    let parsed = reqwest::Url::parse(url).map_err(|e| format!("{url}: {e}"))?;
    let host_name = parsed.host_str().ok_or_else(|| format!("{url}: no host"))?;
    let host = match parsed.port() {
        Some(port) => format!("{host_name}:{port}"),
        None => host_name.to_string(),
    };
    let body = body.to_string();
    let signing = SigningScope {
        credentials,
        region,
        service: SERVICE,
        time: &AmzTime::now(),
    };
    let headers = signing.request_headers("POST", &host, parsed.path(), body.as_bytes());
    let mut request = client
        .post(parsed)
        .header(reqwest::header::CONTENT_TYPE, JSON_CONTENT_TYPE)
        .body(body);
    for (name, value) in headers {
        request = request.header(name, value);
    }
    let response = request.send().await.map_err(|e| format!("{url}: {e}"))?;
    let status = response.status();
    let text = response.text().await.map_err(|e| format!("{url}: {e}"))?;
    if !status.is_success() {
        return Err(format!("{url}: HTTP {status}: {text}"));
    }
    serde_json::from_str(&text).map_err(|e| format!("{url}: invalid json reply: {e}"))
}

fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(HTTP_TIMEOUT)
        .build()
        .map_err(|e| format!("http client: {e}"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SignalingEndpoints {
    pub(crate) websocket: String,
    pub(crate) https: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SignalingChannel {
    pub(crate) arn: String,
    pub(crate) endpoints: SignalingEndpoints,
}

fn endpoints_by_protocol(reply: &Value) -> Result<SignalingEndpoints, String> {
    let list = reply["ResourceEndpointList"]
        .as_array()
        .ok_or_else(|| format!("no ResourceEndpointList in {reply}"))?;
    let endpoint = |protocol: &str, scheme: &str| {
        list.iter()
            .filter(|entry| entry["Protocol"] == protocol)
            .find_map(|entry| entry["ResourceEndpoint"].as_str())
            .filter(|url| url.starts_with(scheme))
            .map(|url| url.trim_end_matches('/').to_string())
            .ok_or_else(|| format!("no {protocol} endpoint starting {scheme} in {reply}"))
    };
    Ok(SignalingEndpoints {
        websocket: endpoint(WSS_PROTOCOL, SECURE_WEBSOCKET_SCHEME)?,
        https: endpoint(HTTPS_PROTOCOL, SECURE_HTTP_SCHEME)?,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IceServer {
    uris: Vec<String>,
    username: String,
    password: String,
}

fn parse_ice_servers(reply: &Value) -> Vec<IceServer> {
    let Some(list) = reply["IceServerList"].as_array() else {
        return Vec::new();
    };
    list.iter()
        .filter_map(|server| {
            Some(IceServer {
                uris: server["Uris"]
                    .as_array()?
                    .iter()
                    .filter_map(|uri| uri.as_str().map(str::to_string))
                    .collect(),
                username: server["Username"].as_str()?.to_string(),
                password: server["Password"].as_str()?.to_string(),
            })
        })
        .collect()
}

pub(crate) fn turn_entries(servers: &[IceServer]) -> impl Iterator<Item = (&str, &str, &str)> {
    servers.iter().flat_map(|server| {
        server
            .uris
            .iter()
            .filter(|uri| {
                TURN_URI_SCHEMES
                    .iter()
                    .any(|scheme| uri.starts_with(scheme))
            })
            .filter(|uri| {
                !(uri.starts_with(TURNS_OVER_UDP_PREFIX) && uri.ends_with(UDP_TRANSPORT_SUFFIX))
            })
            .map(move |uri| {
                (
                    uri.as_str(),
                    server.username.as_str(),
                    server.password.as_str(),
                )
            })
    })
}

pub(crate) async fn ice_servers(
    credentials: &Credentials,
    region: &str,
    channel: &SignalingChannel,
) -> Result<Vec<IceServer>, String> {
    let url = format!("{}{ICE_SERVER_CONFIG_PATH}", channel.endpoints.https);
    let reply = post_signed(
        &http_client()?,
        credentials,
        region,
        &url,
        &json!({ "ChannelARN": channel.arn }),
    )
    .await?;
    Ok(parse_ice_servers(&reply))
}

pub(crate) fn default_stun_server(region: &str) -> String {
    format!("{STUN_HOST_PREFIX}{region}{STUN_HOST_SUFFIX}")
}

pub(crate) const CHANNEL_NAME_PROP: PropertySpec = PropertySpec::new(
    "channel-name",
    PropKind::Str,
    "signalling channel name, resolved to its ARN (channel-name or channel-arn is required)",
);
pub(crate) const CHANNEL_ARN_PROP: PropertySpec = PropertySpec::new(
    "channel-arn",
    PropKind::Str,
    "signalling channel ARN (takes precedence over channel-name)",
);
pub(crate) const REGION_PROP: PropertySpec = PropertySpec::new(
    "region",
    PropKind::Str,
    "AWS region (empty = AWS_REGION, then AWS_DEFAULT_REGION)",
);
pub(crate) const ACCESS_KEY_PROP: PropertySpec = PropertySpec::new(
    "access-key",
    PropKind::Str,
    "AWS access key id (empty = AWS_ACCESS_KEY_ID)",
);
pub(crate) const SECRET_KEY_PROP: PropertySpec = PropertySpec::new(
    "secret-key",
    PropKind::Str,
    "AWS secret access key (empty = AWS_SECRET_ACCESS_KEY)",
);
pub(crate) const SESSION_TOKEN_PROP: PropertySpec = PropertySpec::new(
    "session-token",
    PropKind::Str,
    "AWS session token for temporary credentials (empty = AWS_SESSION_TOKEN)",
);
pub(crate) const KVS_STUN_SERVER_PROP: PropertySpec = PropertySpec::new(
    "stun-server",
    PropKind::Str,
    "STUN server host:port for a server-reflexive candidate (empty = stun.kinesisvideo.<region>.amazonaws.com:443)",
);

#[derive(Clone, Default)]
pub(crate) struct KvsSettings {
    channel_name: Option<String>,
    channel_arn: Option<String>,
    region: Option<String>,
    access_key: Option<String>,
    secret_key: Option<String>,
    session_token: Option<String>,
    pub(crate) stun_server: Option<String>,
    pub(crate) transport_policy: IceTransportPolicy,
    endpoint_override: Option<SignalingEndpoints>,
}

impl core::fmt::Debug for KvsSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KvsSettings")
            .field("channel_name", &self.channel_name)
            .field("channel_arn", &self.channel_arn)
            .field("region", &self.region)
            .field("stun_server", &self.stun_server)
            .field("transport_policy", &self.transport_policy)
            .field("endpoint_override", &self.endpoint_override)
            .finish_non_exhaustive()
    }
}

fn region_is_valid(region: &str) -> bool {
    !region.is_empty()
        && region
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

pub(crate) fn client_id_is_valid(client_id: &str) -> bool {
    !client_id.is_empty()
        && client_id.len() <= MAX_CLIENT_ID_LENGTH
        && !client_id
            .to_ascii_lowercase()
            .starts_with(RESERVED_CLIENT_ID_PREFIX)
        && client_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
}

impl KvsSettings {
    pub(crate) fn set_channel_name(&mut self, name: impl Into<String>) {
        self.channel_name = Some(name.into());
    }

    pub(crate) fn set_channel_arn(&mut self, arn: impl Into<String>) {
        self.channel_arn = Some(arn.into());
    }

    pub(crate) fn set_region(&mut self, region: impl Into<String>) {
        self.region = Some(region.into());
    }

    pub(crate) fn set_credentials(
        &mut self,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
        session_token: Option<String>,
    ) {
        self.access_key = Some(access_key.into());
        self.secret_key = Some(secret_key.into());
        self.session_token = session_token;
    }

    pub(crate) fn set_endpoints(&mut self, endpoints: SignalingEndpoints) {
        self.endpoint_override = Some(endpoints);
    }

    pub(crate) fn set_property(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Option<Result<(), PropError>> {
        if name == "ice-transport-policy" {
            let Some(text) = value.as_str() else {
                return Some(Err(PropError::Type));
            };
            return Some(match IceTransportPolicy::parse(text) {
                Some(policy) => {
                    self.transport_policy = policy;
                    Ok(())
                }
                None => Err(PropError::Value),
            });
        }
        let field = match name {
            "channel-name" => &mut self.channel_name,
            "channel-arn" => &mut self.channel_arn,
            "region" => &mut self.region,
            "access-key" => &mut self.access_key,
            "secret-key" => &mut self.secret_key,
            "session-token" => &mut self.session_token,
            "stun-server" => &mut self.stun_server,
            _ => return None,
        };
        let Some(text) = value.as_str() else {
            return Some(Err(PropError::Type));
        };
        *field = (!text.is_empty()).then(|| text.to_string());
        Some(Ok(()))
    }

    pub(crate) fn get_property(&self, name: &str) -> Option<PropValue> {
        if name == "ice-transport-policy" {
            return Some(PropValue::Str(self.transport_policy.name().into()));
        }
        let field = match name {
            "channel-name" => &self.channel_name,
            "channel-arn" => &self.channel_arn,
            "region" => &self.region,
            "access-key" => &self.access_key,
            "secret-key" => &self.secret_key,
            "session-token" => &self.session_token,
            "stun-server" => &self.stun_server,
            _ => return None,
        };
        Some(PropValue::Str(field.clone().unwrap_or_default()))
    }

    // host and STUN candidates under `all`, the channel's TURN relays always
    pub(crate) async fn gather(
        &self,
        rtc: &mut Rtc,
        socket: &UdpSocket,
        region: &str,
        servers: &[IceServer],
    ) -> Result<TurnSet, String> {
        if self.transport_policy == IceTransportPolicy::All {
            let stun = self
                .stun_server
                .clone()
                .unwrap_or_else(|| default_stun_server(region));
            add_ice_candidates(rtc, socket, Some(&stun))
                .await
                .map_err(|_| "ice candidate gathering failed".to_string())?;
        }
        let turn = TurnSet::setup_each(rtc, socket, turn_entries(servers)).await;
        if self.transport_policy == IceTransportPolicy::Relay && turn.is_empty() {
            return Err("ice-transport-policy=relay but no TURN allocation succeeded".to_string());
        }
        Ok(turn)
    }

    pub(crate) fn credentials(&self) -> Result<Credentials, String> {
        self.credentials_from(|name| std::env::var(name).ok())
    }

    fn credentials_from(
        &self,
        environment: impl Fn(&str) -> Option<String>,
    ) -> Result<Credentials, String> {
        let lookup = |property: &Option<String>, variable: &str| {
            property
                .clone()
                .or_else(|| environment(variable).filter(|v| !v.is_empty()))
        };
        let access_key = lookup(&self.access_key, ENV_ACCESS_KEY)
            .ok_or_else(|| format!("no access-key and {ENV_ACCESS_KEY} is unset"))?;
        let secret_key = lookup(&self.secret_key, ENV_SECRET_KEY)
            .ok_or_else(|| format!("no secret-key and {ENV_SECRET_KEY} is unset"))?;
        Ok(Credentials {
            access_key,
            secret_key,
            session_token: lookup(&self.session_token, ENV_SESSION_TOKEN),
        })
    }

    pub(crate) fn region(&self) -> Result<String, String> {
        self.region_from(|name| std::env::var(name).ok())
    }

    fn region_from(&self, environment: impl Fn(&str) -> Option<String>) -> Result<String, String> {
        let region = self
            .region
            .clone()
            .or_else(|| environment(ENV_REGION).filter(|v| !v.is_empty()))
            .or_else(|| environment(ENV_DEFAULT_REGION).filter(|v| !v.is_empty()))
            .ok_or_else(|| {
                format!("no region and neither {ENV_REGION} nor {ENV_DEFAULT_REGION} is set")
            })?;
        if !region_is_valid(&region) {
            return Err(format!("region {region:?} is not an AWS region name"));
        }
        Ok(region)
    }

    pub(crate) async fn resolve(
        &self,
        role: Role,
        credentials: &Credentials,
        region: &str,
    ) -> Result<SignalingChannel, String> {
        if let Some(endpoints) = &self.endpoint_override {
            let arn = self
                .channel_arn
                .clone()
                .ok_or("channel-arn is required with fixed signalling endpoints")?;
            return Ok(SignalingChannel {
                arn,
                endpoints: endpoints.clone(),
            });
        }
        let client = http_client()?;
        let control_plane = format!("{SECURE_HTTP_SCHEME}{SERVICE}.{region}.amazonaws.com");
        let arn = match (&self.channel_arn, &self.channel_name) {
            (Some(arn), _) => arn.clone(),
            (None, Some(name)) => {
                let reply = post_signed(
                    &client,
                    credentials,
                    region,
                    &format!("{control_plane}{DESCRIBE_CHANNEL_PATH}"),
                    &json!({ "ChannelName": name }),
                )
                .await?;
                reply["ChannelInfo"]["ChannelARN"]
                    .as_str()
                    .ok_or_else(|| format!("no ChannelInfo.ChannelARN in {reply}"))?
                    .to_string()
            }
            (None, None) => return Err("set channel-name or channel-arn".to_string()),
        };
        let request = json!({
            "ChannelARN": arn,
            "SingleMasterChannelEndpointConfiguration": {
                "Protocols": [WSS_PROTOCOL, HTTPS_PROTOCOL],
                "Role": role.name(),
            },
        });
        let reply = post_signed(
            &client,
            credentials,
            region,
            &format!("{control_plane}{CHANNEL_ENDPOINT_PATH}"),
            &request,
        )
        .await?;
        Ok(SignalingChannel {
            arn,
            endpoints: endpoints_by_protocol(&reply)?,
        })
    }
}

pub(crate) fn connect_url(
    credentials: &Credentials,
    region: &str,
    channel: &SignalingChannel,
    client_id: Option<&str>,
) -> Result<String, String> {
    let mut parameters = alloc::vec![("X-Amz-ChannelARN", channel.arn.as_str())];
    if let Some(client_id) = client_id {
        parameters.push(("X-Amz-ClientId", client_id));
    }
    presigned_url(
        credentials,
        region,
        &channel.endpoints.websocket,
        &parameters,
        &AmzTime::now(),
    )
}

pub(crate) async fn connect(url: &str) -> Result<Ws, String> {
    let endpoint = redacted_url(url);
    let (ws, _) = tokio::time::timeout(CONNECT_TIMEOUT, connect_async(url))
        .await
        .map_err(|_| format!("connect to {endpoint} timed out"))?
        .map_err(|e| format!("connect to {endpoint} failed: {e}"))?;
    Ok(ws)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Action {
    SdpOffer,
    SdpAnswer,
    IceCandidate,
}

impl Action {
    fn name(self) -> &'static str {
        match self {
            Action::SdpOffer => "SDP_OFFER",
            Action::SdpAnswer => "SDP_ANSWER",
            Action::IceCandidate => "ICE_CANDIDATE",
        }
    }
}

pub(crate) fn sdp_payload(kind: &str, sdp: &str) -> Value {
    json!({ "type": kind, "sdp": sdp })
}

// RTCIceCandidateInit payloads for the candidates already in our sdp
pub(crate) fn candidate_payloads(sdp: &str) -> Vec<Value> {
    let mid = sdp
        .lines()
        .find_map(|line| line.trim_end().strip_prefix(MID_LINE_PREFIX))
        .unwrap_or_default();
    sdp.lines()
        .map(str::trim_end)
        .filter(|line| line.starts_with(CANDIDATE_LINE_PREFIX))
        .filter_map(|line| line.strip_prefix(SDP_ATTRIBUTE_PREFIX))
        .map(|candidate| {
            json!({ "candidate": candidate, "sdpMid": mid, "sdpMLineIndex": BUNDLED_MLINE_INDEX })
        })
        .collect()
}

pub(crate) fn encode_message(
    action: Action,
    recipient: Option<&str>,
    payload: &Value,
) -> Result<String, String> {
    let encoded = STANDARD.encode(payload.to_string());
    if encoded.len() > MAX_MESSAGE_PAYLOAD_BYTES {
        return Err(format!(
            "{} payload is {} bytes base64, over the service's {MAX_MESSAGE_PAYLOAD_BYTES}",
            action.name(),
            encoded.len()
        ));
    }
    let mut message = json!({
        "action": action.name(),
        "messagePayload": encoded,
    });
    if let Some(recipient) = recipient {
        message["recipientClientId"] = json!(recipient);
    }
    Ok(message.to_string())
}

pub(crate) async fn send_message<S>(
    ws: &mut S,
    action: Action,
    recipient: Option<&str>,
    payload: &Value,
) -> Result<(), String>
where
    S: Sink<Message, Error = WsError> + Unpin,
{
    ws.send(Message::Text(encode_message(action, recipient, payload)?))
        .await
        .map_err(|e| format!("websocket send: {e}"))
}

#[derive(Debug)]
pub(crate) enum SignalMessage {
    Offer {
        sender: String,
        sdp: String,
    },
    Answer {
        sdp: String,
    },
    Candidate {
        sender: Option<String>,
        candidate: Candidate,
    },
    GoAway,
    ReconnectIceServer,
    Status(String),
}

fn decode_payload(message: &Value) -> Result<Value, String> {
    let encoded = message["messagePayload"]
        .as_str()
        .ok_or("no messagePayload")?;
    let bytes = PAYLOAD_DECODER
        .decode(encoded)
        .map_err(|e| format!("messagePayload is not base64: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("messagePayload is not json: {e}"))
}

fn session_description(message: &Value, kind: &str) -> Result<String, String> {
    let payload = decode_payload(message)?;
    if payload["type"] != kind {
        return Err(format!("payload type is {}, not {kind}", payload["type"]));
    }
    payload["sdp"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| "payload has no sdp".to_string())
}

pub(crate) fn decode_message(text: &str) -> Result<SignalMessage, String> {
    let message: Value = serde_json::from_str(text).map_err(|e| format!("not json: {e}"))?;
    let sender = message["senderClientId"].as_str().map(str::to_string);
    match message["messageType"].as_str() {
        Some("SDP_OFFER") => Ok(SignalMessage::Offer {
            sender: sender.ok_or("SDP_OFFER without senderClientId")?,
            sdp: session_description(&message, OFFER_TYPE)?,
        }),
        Some("SDP_ANSWER") => Ok(SignalMessage::Answer {
            sdp: session_description(&message, ANSWER_TYPE)?,
        }),
        Some("ICE_CANDIDATE") => {
            let payload = decode_payload(&message)?;
            let line = payload["candidate"]
                .as_str()
                .ok_or("candidate payload has no candidate")?;
            let line = line.strip_prefix(SDP_ATTRIBUTE_PREFIX).unwrap_or(line);
            if line.is_empty() {
                return Err("end-of-candidates marker".to_string());
            }
            let candidate = Candidate::from_sdp_string(line)
                .map_err(|e| format!("candidate {line:?}: {e:?}"))?;
            Ok(SignalMessage::Candidate { sender, candidate })
        }
        Some("GO_AWAY") => Ok(SignalMessage::GoAway),
        Some("RECONNECT_ICE_SERVER") => Ok(SignalMessage::ReconnectIceServer),
        Some("STATUS_RESPONSE") => {
            let status = &message["statusResponse"];
            Ok(SignalMessage::Status(format!(
                "{} {}: {}",
                status["errorType"], status["statusCode"], status["description"]
            )))
        }
        other => Err(format!("unknown messageType {other:?}")),
    }
}

#[derive(Debug)]
pub(crate) enum Received {
    Signal(SignalMessage),
    Skipped(String),
    Control,
    Closed(String),
}

pub(crate) fn receive(frame: Option<Result<Message, WsError>>) -> Received {
    let text = match frame {
        None => return Received::Closed("websocket closed".to_string()),
        Some(Err(e)) => return Received::Closed(format!("websocket: {e}")),
        Some(Ok(Message::Close(_))) => return Received::Closed("websocket closed".to_string()),
        Some(Ok(Message::Text(text))) => text,
        Some(Ok(Message::Binary(bytes))) => match String::from_utf8(bytes) {
            Ok(text) => text,
            Err(_) => return Received::Skipped("binary frame is not utf-8".to_string()),
        },
        Some(Ok(_)) => return Received::Control,
    };
    // kvs answers every message we send with an empty frame
    if text.trim().is_empty() {
        return Received::Control;
    }
    match decode_message(&text) {
        Ok(message) => Received::Signal(message),
        Err(reason) => Received::Skipped(reason),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // the aws sigv4 test suite credentials and timestamp (get-vanilla)
    const EXAMPLE_ACCESS_KEY: &str = "AKIDEXAMPLE";
    const EXAMPLE_SECRET_KEY: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
    const EXAMPLE_UNIX_SECONDS: u64 = 1_440_938_160;
    const EXAMPLE_DATETIME: &str = "20150830T123600Z";
    const EXAMPLE_SESSION_TOKEN: &str = "session-token-example";
    const GET_VANILLA_SIGNATURE: &str =
        "5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31";
    // botocore 1.43 SigV4Auth / SigV4QueryAuth over the same inputs
    const DESCRIBE_SIGNATURE: &str =
        "49a3b7d56c0ca170f5f816836ee2c643ce560a90beceaa22aea31de63d76466a";
    const PRESIGNED_SIGNATURE: &str =
        "004ea61f40a672199cd6486d0ebb9ebd7ea85dde60c27db5f02103935d8ba755";
    const EXAMPLE_ARN: &str =
        "arn:aws:kinesisvideo:us-west-2:123456789012:channel/g2g-test/1234567890123";
    const EXAMPLE_WSS: &str = "wss://v-1234abcd.kinesisvideo.us-west-2.amazonaws.com";
    const EXAMPLE_CLIENT_ID: &str = "viewer-1";
    const EXAMPLE_REGION: &str = "us-west-2";
    const HOST_CANDIDATE: &str = "candidate:1 1 udp 2130706431 192.0.2.1 5000 typ host";

    fn example_credentials(session_token: Option<&str>) -> Credentials {
        Credentials {
            access_key: EXAMPLE_ACCESS_KEY.into(),
            secret_key: EXAMPLE_SECRET_KEY.into(),
            session_token: session_token.map(str::to_string),
        }
    }

    fn authorization(headers: &[(&str, String)]) -> String {
        headers
            .iter()
            .find(|(name, _)| *name == "authorization")
            .map(|(_, value)| value.clone())
            .expect("authorization header")
    }

    #[test]
    fn timestamps_use_the_basic_iso8601_form() {
        let time = AmzTime::from_unix_seconds(EXAMPLE_UNIX_SECONDS);
        assert_eq!(time.datetime, EXAMPLE_DATETIME);
        assert_eq!(time.date(), &EXAMPLE_DATETIME[..8]);
    }

    #[test]
    fn header_signing_matches_the_aws_get_vanilla_vector() {
        let credentials = example_credentials(None);
        let signing = SigningScope {
            credentials: &credentials,
            region: "us-east-1",
            service: "service",
            time: &AmzTime::from_unix_seconds(EXAMPLE_UNIX_SECONDS),
        };
        let headers = signing.request_headers("GET", "example.amazonaws.com", "/", b"");
        assert_eq!(
            authorization(&headers),
            format!("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, SignedHeaders=host;x-amz-date, Signature={GET_VANILLA_SIGNATURE}")
        );
        assert!(headers.contains(&("x-amz-date", EXAMPLE_DATETIME.to_string())));
        assert!(!headers.iter().any(|(name, _)| *name == "host"));
    }

    #[test]
    fn header_signing_covers_the_body_and_session_token() {
        let credentials = example_credentials(Some(EXAMPLE_SESSION_TOKEN));
        let signing = SigningScope {
            credentials: &credentials,
            region: EXAMPLE_REGION,
            service: SERVICE,
            time: &AmzTime::from_unix_seconds(EXAMPLE_UNIX_SECONDS),
        };
        let headers = signing.request_headers(
            "POST",
            "kinesisvideo.us-west-2.amazonaws.com",
            DESCRIBE_CHANNEL_PATH,
            br#"{"ChannelName":"g2g-test"}"#,
        );
        assert!(authorization(&headers).ends_with(&format!(
            "SignedHeaders=host;x-amz-date;x-amz-security-token, Signature={DESCRIBE_SIGNATURE}"
        )));
        assert!(headers.contains(&("x-amz-security-token", EXAMPLE_SESSION_TOKEN.to_string())));
    }

    #[test]
    fn presigned_url_matches_botocore_with_sorted_parameters() {
        let channel = SignalingChannel {
            arn: EXAMPLE_ARN.into(),
            endpoints: SignalingEndpoints {
                websocket: EXAMPLE_WSS.into(),
                https: String::new(),
            },
        };
        let url = presigned_url(
            &example_credentials(Some(EXAMPLE_SESSION_TOKEN)),
            EXAMPLE_REGION,
            &channel.endpoints.websocket,
            &[
                ("X-Amz-ChannelARN", EXAMPLE_ARN),
                ("X-Amz-ClientId", EXAMPLE_CLIENT_ID),
            ],
            &AmzTime::from_unix_seconds(EXAMPLE_UNIX_SECONDS),
        )
        .unwrap();
        let expected = format!(
            "{EXAMPLE_WSS}/?X-Amz-Algorithm=AWS4-HMAC-SHA256\
             &X-Amz-ChannelARN=arn%3Aaws%3Akinesisvideo%3Aus-west-2%3A123456789012%3Achannel%2Fg2g-test%2F1234567890123\
             &X-Amz-ClientId=viewer-1\
             &X-Amz-Credential=AKIDEXAMPLE%2F20150830%2Fus-west-2%2Fkinesisvideo%2Faws4_request\
             &X-Amz-Date=20150830T123600Z&X-Amz-Expires=299\
             &X-Amz-Security-Token=session-token-example\
             &X-Amz-Signature={PRESIGNED_SIGNATURE}&X-Amz-SignedHeaders=host"
        );
        assert_eq!(url, expected);
        assert_eq!(redacted_url(&url), format!("{EXAMPLE_WSS}/"));
    }

    #[test]
    fn presigning_rejects_non_websocket_endpoints() {
        let credentials = example_credentials(None);
        let time = AmzTime::from_unix_seconds(EXAMPLE_UNIX_SECONDS);
        for endpoint in ["https://example.com", "wss://example.com/?a=b"] {
            assert!(presigned_url(&credentials, EXAMPLE_REGION, endpoint, &[], &time).is_err());
        }
        let master = presigned_url(
            &credentials,
            EXAMPLE_REGION,
            "ws://127.0.0.1:9/",
            &[],
            &time,
        )
        .unwrap();
        assert!(master.starts_with("ws://127.0.0.1:9/?X-Amz-Algorithm="));
        assert!(!master.contains("X-Amz-Security-Token"));
    }

    #[test]
    fn envelopes_round_trip() {
        let offer =
            encode_message(Action::SdpOffer, None, &sdp_payload(OFFER_TYPE, "v=0\r\n")).unwrap();
        let sent: Value = serde_json::from_str(&offer).unwrap();
        assert_eq!(sent["action"], "SDP_OFFER");
        assert!(sent.get("recipientClientId").is_none());
        // what kvs delivers: the sender's id and the payload as sent
        let delivered = json!({
            "messageType": sent["action"],
            "senderClientId": EXAMPLE_CLIENT_ID,
            "messagePayload": sent["messagePayload"],
        });
        match decode_message(&delivered.to_string()) {
            Ok(SignalMessage::Offer { sender, sdp }) => {
                assert_eq!(sender, EXAMPLE_CLIENT_ID);
                assert_eq!(sdp, "v=0\r\n");
            }
            other => panic!("{other:?}"),
        }

        let candidate = json!({ "candidate": HOST_CANDIDATE, "sdpMid": "0", "sdpMLineIndex": 0 });
        let trickle =
            encode_message(Action::IceCandidate, Some(EXAMPLE_CLIENT_ID), &candidate).unwrap();
        let sent: Value = serde_json::from_str(&trickle).unwrap();
        assert_eq!(sent["action"], "ICE_CANDIDATE");
        assert_eq!(sent["recipientClientId"], EXAMPLE_CLIENT_ID);
        let delivered =
            json!({ "messageType": "ICE_CANDIDATE", "messagePayload": sent["messagePayload"] });
        match decode_message(&delivered.to_string()) {
            Ok(SignalMessage::Candidate { sender, candidate }) => {
                assert_eq!(sender, None);
                assert_eq!(candidate.to_sdp_string(), HOST_CANDIDATE);
            }
            other => panic!("{other:?}"),
        }

        let answer = encode_message(
            Action::SdpAnswer,
            Some(EXAMPLE_CLIENT_ID),
            &sdp_payload(ANSWER_TYPE, "v=0"),
        )
        .unwrap();
        let sent: Value = serde_json::from_str(&answer).unwrap();
        let delivered =
            json!({ "messageType": "SDP_ANSWER", "messagePayload": sent["messagePayload"] });
        assert!(matches!(
            decode_message(&delivered.to_string()),
            Ok(SignalMessage::Answer { sdp }) if sdp == "v=0"
        ));
    }

    #[test]
    fn malformed_envelopes_are_rejected() {
        let payload = |value: Value| STANDARD.encode(value.to_string());
        let unpadded = STANDARD.encode(json!({ "type": "answer", "sdp": "v=0" }).to_string());
        let unpadded = unpadded.trim_end_matches('=');
        assert!(matches!(
            decode_message(
                &json!({ "messageType": "SDP_ANSWER", "messagePayload": unpadded }).to_string()
            ),
            Ok(SignalMessage::Answer { .. })
        ));
        for bad in [
            "not json".to_string(),
            json!({ "messageType": "SDP_OFFER", "messagePayload": payload(json!({ "type": "offer", "sdp": "v=0" })) }).to_string(),
            json!({ "messageType": "SDP_OFFER", "senderClientId": "v", "messagePayload": "!!!" }).to_string(),
            json!({ "messageType": "SDP_OFFER", "senderClientId": "v", "messagePayload": payload(json!({ "type": "answer", "sdp": "v=0" })) }).to_string(),
            json!({ "messageType": "SDP_ANSWER", "messagePayload": payload(json!({ "type": "answer" })) }).to_string(),
            json!({ "messageType": "SDP_ANSWER", "messagePayload": STANDARD.encode("not json") }).to_string(),
            json!({ "messageType": "SDP_ANSWER" }).to_string(),
            json!({ "messageType": "ICE_CANDIDATE", "messagePayload": payload(json!({ "candidate": "" })) }).to_string(),
            json!({ "messageType": "ICE_CANDIDATE", "messagePayload": payload(json!({ "candidate": "candidate:garbage" })) }).to_string(),
            json!({ "messageType": "ICE_CANDIDATE", "messagePayload": payload(json!({ "sdpMid": "0" })) }).to_string(),
            json!({ "messageType": "SOMETHING_NEW", "messagePayload": payload(json!({})) }).to_string(),
            json!({ "action": "SDP_OFFER" }).to_string(),
        ] {
            assert!(decode_message(&bad).is_err(), "{bad}");
        }
        assert!(matches!(
            decode_message(&json!({ "messageType": "GO_AWAY" }).to_string()),
            Ok(SignalMessage::GoAway)
        ));
        let status = json!({ "messageType": "STATUS_RESPONSE", "statusResponse": {
            "correlationId": "1", "errorType": "InvalidArgumentException", "statusCode": "400", "description": "bad" } });
        assert!(matches!(
            decode_message(&status.to_string()),
            Ok(SignalMessage::Status(text)) if text.contains("InvalidArgumentException")
        ));
        assert!(matches!(
            receive(Some(Ok(Message::Binary(alloc::vec![0xff, 0xfe])))),
            Received::Skipped(_)
        ));
        assert!(matches!(receive(None), Received::Closed(_)));
        assert!(matches!(
            receive(Some(Ok(Message::Ping(Vec::new())))),
            Received::Control
        ));
        assert!(matches!(
            receive(Some(Ok(Message::Text(String::new())))),
            Received::Control
        ));
    }

    #[test]
    fn local_candidates_are_lifted_from_the_sdp() {
        let sdp = format!("v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=mid:0\r\na={HOST_CANDIDATE}\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=mid:1\r\n");
        let payloads = candidate_payloads(&sdp);
        assert_eq!(
            payloads,
            [json!({ "candidate": HOST_CANDIDATE, "sdpMid": "0", "sdpMLineIndex": 0 })]
        );
    }

    #[test]
    fn credentials_fall_back_to_the_environment() {
        let environment = |name: &str| match name {
            ENV_ACCESS_KEY => Some("env-access".to_string()),
            ENV_SECRET_KEY => Some("env-secret".to_string()),
            ENV_SESSION_TOKEN => Some(String::new()),
            ENV_DEFAULT_REGION => Some("eu-west-1".to_string()),
            _ => None,
        };
        let mut settings = KvsSettings::default();
        let credentials = settings.credentials_from(environment).unwrap();
        assert_eq!(credentials.access_key, "env-access");
        assert_eq!(credentials.secret_key, "env-secret");
        assert_eq!(credentials.session_token, None);
        assert_eq!(settings.region_from(environment).unwrap(), "eu-west-1");

        for (name, value) in [
            ("access-key", "property-access"),
            ("session-token", "property-token"),
            ("region", "us-east-2"),
        ] {
            settings
                .set_property(name, &PropValue::Str(value.into()))
                .unwrap()
                .unwrap();
        }
        let credentials = settings.credentials_from(environment).unwrap();
        assert_eq!(credentials.access_key, "property-access");
        assert_eq!(credentials.secret_key, "env-secret");
        assert_eq!(credentials.session_token.as_deref(), Some("property-token"));
        assert_eq!(settings.region_from(environment).unwrap(), "us-east-2");

        assert!(KvsSettings::default().credentials_from(|_| None).is_err());
        assert!(KvsSettings::default().region_from(|_| None).is_err());
        settings
            .set_property("region", &PropValue::Str("us-west-2.evil.example/".into()))
            .unwrap()
            .unwrap();
        assert!(settings.region_from(environment).is_err());
    }

    #[test]
    fn control_plane_replies_parse() {
        let reply = json!({ "ResourceEndpointList": [
            { "Protocol": "HTTPS", "ResourceEndpoint": "https://r-1.kinesisvideo.us-west-2.amazonaws.com" },
            { "Protocol": "WSS", "ResourceEndpoint": "wss://m-1.kinesisvideo.us-west-2.amazonaws.com/" },
        ] });
        assert_eq!(
            endpoints_by_protocol(&reply),
            Ok(SignalingEndpoints {
                websocket: "wss://m-1.kinesisvideo.us-west-2.amazonaws.com".into(),
                https: "https://r-1.kinesisvideo.us-west-2.amazonaws.com".into(),
            })
        );
        let downgraded = json!({ "ResourceEndpointList": [
            { "Protocol": "HTTPS", "ResourceEndpoint": "http://r-1.example" },
            { "Protocol": "WSS", "ResourceEndpoint": "ws://m-1.example" },
        ] });
        assert!(endpoints_by_protocol(&downgraded).is_err());
        assert!(endpoints_by_protocol(&json!({})).is_err());

        let ice = json!({ "IceServerList": [
            { "Uris": ["turn:1.example:443?transport=udp", "turns:1.example:443?transport=udp", "turns:1.example:443?transport=tcp"],
              "Username": "1700000000:djE6YXJu", "Password": "p1", "Ttl": 300 },
            { "Uris": "not a list", "Username": "u", "Password": "p" },
            { "Uris": ["stun:stun.example:443"], "Username": "u2", "Password": "p2" },
        ] });
        let servers = parse_ice_servers(&ice);
        assert_eq!(servers.len(), 2);
        let entries: Vec<_> = turn_entries(&servers).collect();
        assert_eq!(
            entries,
            [
                (
                    "turn:1.example:443?transport=udp",
                    "1700000000:djE6YXJu",
                    "p1"
                ),
                (
                    "turns:1.example:443?transport=tcp",
                    "1700000000:djE6YXJu",
                    "p1"
                ),
            ]
        );
        assert!(parse_ice_servers(&json!({ "IceServerList": 7 })).is_empty());
    }

    #[test]
    fn oversized_payloads_are_refused_before_sending() {
        let sdp = "a".repeat(MAX_MESSAGE_PAYLOAD_BYTES);
        let refused =
            encode_message(Action::SdpOffer, None, &sdp_payload(OFFER_TYPE, &sdp)).unwrap_err();
        assert!(refused.starts_with("SDP_OFFER payload is "), "{refused}");
        let small = sdp_payload(OFFER_TYPE, "v=0");
        assert!(encode_message(Action::SdpOffer, None, &small).is_ok());
    }

    #[test]
    fn the_stun_server_defaults_to_the_regional_kvs_host() {
        assert_eq!(
            default_stun_server("us-west-2"),
            "stun.kinesisvideo.us-west-2.amazonaws.com:443"
        );
        let mut settings = KvsSettings::default();
        assert_eq!(
            settings.get_property("ice-transport-policy"),
            Some(PropValue::Str("all".into()))
        );
        settings
            .set_property("ice-transport-policy", &PropValue::Str("relay".into()))
            .unwrap()
            .unwrap();
        assert_eq!(settings.transport_policy, IceTransportPolicy::Relay);
        assert_eq!(
            settings.set_property("ice-transport-policy", &PropValue::Str("host".into())),
            Some(Err(PropError::Value))
        );
    }

    #[test]
    fn client_ids_follow_the_service_pattern() {
        assert!(client_id_is_valid("viewer-1.a_b"));
        for bad in ["", "AWS_viewer", "aws_x", "has space", "slash/"] {
            assert!(!client_id_is_valid(bad), "{bad}");
        }
        assert!(!client_id_is_valid(&"a".repeat(MAX_CLIENT_ID_LENGTH + 1)));
    }
}
