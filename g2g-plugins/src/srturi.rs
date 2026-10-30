use alloc::string::{String, ToString};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use g2g_core::log::Target;
use g2g_core::{G2gError, NetworkFailure, PropError};

use crate::srtcrypto::AesKeySize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrtMode {
    Caller,
    Listener,
    Rendezvous,
}

impl SrtMode {
    pub const NAMES: &'static str = "caller | listener | rendezvous";

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "caller" => Some(SrtMode::Caller),
            "listener" => Some(SrtMode::Listener),
            "rendezvous" => Some(SrtMode::Rendezvous),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            SrtMode::Caller => "caller",
            SrtMode::Listener => "listener",
            SrtMode::Rendezvous => "rendezvous",
        }
    }
}

// an srt:// URI as gst's srtsrc and srtsink read it, with the libsrt query options g2g applies
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrtUri {
    pub host: Option<String>,
    pub port: Option<u16>,
    // caller when the URI names a host and listener when not, unless the query names one
    pub mode: SrtMode,
    pub local_address: Option<IpAddr>,
    pub local_port: Option<u16>,
    pub latency_ms: Option<u16>,
    pub passphrase: Option<String>,
    pub key_size: Option<AesKeySize>,
    pub stream_id: Option<String>,
    pub loss_max_ttl: Option<u32>,
}

// the peer a caller calls or a rendezvous meets, by IP or by a name resolved on each connect
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrtPeer {
    pub host: String,
    pub port: u16,
}

impl SrtPeer {
    pub async fn resolve(&self) -> Result<SocketAddr, G2gError> {
        let mut addresses = tokio::net::lookup_host((self.host.as_str(), self.port))
            .await
            .map_err(|error| {
                g2g_core::g2g_debug!(Target::category("srt"), "{self} does not resolve: {error}");
                G2gError::Network(NetworkFailure::Connect)
            })?;
        addresses
            .next()
            .ok_or(G2gError::Network(NetworkFailure::Connect))
    }
}

impl From<SocketAddr> for SrtPeer {
    fn from(address: SocketAddr) -> Self {
        Self {
            host: address.ip().to_string(),
            port: address.port(),
        }
    }
}

impl core::fmt::Display for SrtPeer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.host.parse::<Ipv6Addr>() {
            Ok(ip) => write!(f, "[{ip}]:{}", self.port),
            Err(_) => write!(f, "{}:{}", self.host, self.port),
        }
    }
}

impl SrtUri {
    // the address to bind, and the peer to call or meet, which a listener has none of
    pub fn endpoints(&self) -> Result<(SocketAddr, Option<SrtPeer>), PropError> {
        let remote = || -> Result<SrtPeer, PropError> {
            Ok(SrtPeer {
                host: self.host.clone().ok_or(PropError::Value)?,
                port: self.port.ok_or(PropError::Value)?,
            })
        };
        // an unspecified address takes the family of the peer once it resolves
        let local_ip = self
            .local_address
            .unwrap_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED));
        match self.mode {
            SrtMode::Listener => {
                let port = self.local_port.or(self.port).ok_or(PropError::Value)?;
                // a listener binds an address, never a name
                let host = self
                    .host
                    .as_deref()
                    .map(parse_value::<IpAddr>)
                    .transpose()?;
                let ip = self.local_address.or(host).unwrap_or(local_ip);
                Ok((SocketAddr::new(ip, port), None))
            }
            SrtMode::Caller => {
                let bind = SocketAddr::new(local_ip, self.local_port.unwrap_or(0));
                Ok((bind, Some(remote()?)))
            }
            // both sides send to each other, from localport or else the port they send to
            SrtMode::Rendezvous => {
                let remote = remote()?;
                let port = self.local_port.unwrap_or(remote.port);
                Ok((SocketAddr::new(local_ip, port), Some(remote)))
            }
        }
    }
}

// an unspecified bind address takes the family of the peer it sends to
pub fn bind_toward(bind: SocketAddr, peer: SocketAddr) -> SocketAddr {
    if !bind.ip().is_unspecified() {
        return bind;
    }
    let ip = match peer.ip() {
        IpAddr::V4(_) => IpAddr::V4(Ipv4Addr::UNSPECIFIED),
        IpAddr::V6(_) => IpAddr::V6(Ipv6Addr::UNSPECIFIED),
    };
    SocketAddr::new(ip, bind.port())
}

const MAX_HOST_NAME_BYTES: usize = 253;

// an IP or a DNS name, so a typo fails the property set rather than every call
pub fn parse_host(text: &str) -> Result<String, PropError> {
    let name_bytes = |b: u8| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_');
    let is_name =
        !text.is_empty() && text.len() <= MAX_HOST_NAME_BYTES && text.bytes().all(name_bytes);
    if is_name || text.parse::<IpAddr>().is_ok() {
        return Ok(text.to_string());
    }
    Err(PropError::Value)
}

const SCHEME: &str = "srt://";

pub fn parse_srt_uri(uri: &str) -> Result<SrtUri, PropError> {
    let rest = uri.strip_prefix(SCHEME).ok_or(PropError::Value)?;
    let (authority, query) = rest.split_once('?').unwrap_or((rest, ""));
    let (host, port) = parse_authority(authority.trim_end_matches('/'))?;
    let mut parsed = SrtUri {
        mode: if host.is_some() {
            SrtMode::Caller
        } else {
            SrtMode::Listener
        },
        host,
        port,
        local_address: None,
        local_port: None,
        latency_ms: None,
        passphrase: None,
        key_size: None,
        stream_id: None,
        loss_max_ttl: None,
    };
    for pair in query.split('&').filter(|pair| !pair.is_empty()) {
        let (key, value) = pair.split_once('=').ok_or(PropError::Value)?;
        let value = percent_decoded(value)?;
        match key {
            "mode" => parsed.mode = SrtMode::from_name(&value).ok_or(PropError::Value)?,
            "localaddress" => parsed.local_address = Some(parse_value(&value)?),
            "localport" => parsed.local_port = Some(parse_value(&value)?),
            "latency" => parsed.latency_ms = Some(parse_value(&value)?),
            "passphrase" => parsed.passphrase = Some(value),
            // 0 leaves libsrt's default, AES-128
            "pbkeylen" => {
                parsed.key_size = match parse_value::<u64>(&value)? {
                    0 => None,
                    bytes => Some(AesKeySize::from_bytes(bytes).ok_or(PropError::Value)?),
                }
            }
            "streamid" => parsed.stream_id = Some(value),
            "lossmaxttl" => parsed.loss_max_ttl = Some(parse_value(&value)?),
            // libsrt takes about 30 more options, none of which g2g applies
            _ => return Err(PropError::Value),
        }
    }
    Ok(parsed)
}

// `[v6]:port`, `host:port`, `:port`, or a bare host
fn parse_authority(authority: &str) -> Result<(Option<String>, Option<u16>), PropError> {
    let (host, port) = match authority.strip_prefix('[') {
        Some(bracketed) => {
            let (host, after) = bracketed.split_once(']').ok_or(PropError::Value)?;
            parse_value::<Ipv6Addr>(host)?;
            let port = match after {
                "" => None,
                _ => Some(after.strip_prefix(':').ok_or(PropError::Value)?),
            };
            (host, port)
        }
        None => match authority.rsplit_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (authority, None),
        },
    };
    let host = (!host.is_empty()).then(|| parse_host(host)).transpose()?;
    let port = port.map(parse_value::<u16>).transpose()?;
    Ok((host, port))
}

fn parse_value<T: core::str::FromStr>(text: &str) -> Result<T, PropError> {
    text.parse().map_err(|_| PropError::Value)
}

fn percent_decoded(text: &str) -> Result<String, PropError> {
    let bytes = crate::dataurisrc::percent_decode(text).ok_or(PropError::Value)?;
    String::from_utf8(bytes).map_err(|_| PropError::Value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_host_makes_a_caller_and_no_host_a_listener() {
        let caller = parse_srt_uri("srt://192.0.2.7:9000").unwrap();
        assert_eq!(caller.mode, SrtMode::Caller);
        assert_eq!(caller.host.as_deref(), Some("192.0.2.7"));
        assert_eq!(caller.port, Some(9000));
        let listener = parse_srt_uri("srt://:9000").unwrap();
        assert_eq!(listener.mode, SrtMode::Listener);
        assert_eq!(listener.host, None);
        assert_eq!(listener.port, Some(9000));
    }

    #[test]
    fn the_query_sets_the_mode_and_the_options() {
        let uri = parse_srt_uri(
            "srt://[2001:db8::1]:7001?mode=rendezvous&localport=7002&latency=200\
             &passphrase=a%26b%3Dc&pbkeylen=24&streamid=live%2Fcam&lossmaxttl=40",
        )
        .unwrap();
        assert_eq!(uri.mode, SrtMode::Rendezvous);
        assert_eq!(uri.host.as_deref(), Some("2001:db8::1"));
        assert_eq!(uri.port, Some(7001));
        assert_eq!(uri.local_port, Some(7002));
        assert_eq!(uri.latency_ms, Some(200));
        assert_eq!(uri.passphrase.as_deref(), Some("a&b=c"));
        assert_eq!(uri.key_size, Some(AesKeySize::Aes192));
        assert_eq!(uri.stream_id.as_deref(), Some("live/cam"));
        assert_eq!(uri.loss_max_ttl, Some(40));
    }

    #[test]
    fn an_option_g2g_does_not_apply_is_refused() {
        assert_eq!(
            parse_srt_uri("srt://:9000?sndbuf=1000000"),
            Err(PropError::Value)
        );
        assert_eq!(
            parse_srt_uri("srt://:9000?pbkeylen=20"),
            Err(PropError::Value)
        );
        assert_eq!(parse_srt_uri("udp://:9000"), Err(PropError::Value));
    }

    #[test]
    fn a_host_name_is_kept_for_each_connect_and_a_malformed_one_refused() {
        let uri = parse_srt_uri("srt://srt.example.com:9000").unwrap();
        assert_eq!(uri.host.as_deref(), Some("srt.example.com"));
        let (_, remote) = uri.endpoints().unwrap();
        assert_eq!(remote.unwrap().to_string(), "srt.example.com:9000");
        for refused in [
            "srt://srt example:9000",
            "srt://user@host:9000",
            "srt://[srt.example.com]:9000",
            // a listener binds an address
            "srt://srt.example.com:9000?mode=listener",
        ] {
            let endpoints = parse_srt_uri(refused).and_then(|uri| uri.endpoints());
            assert_eq!(endpoints, Err(PropError::Value), "{refused}");
        }
    }

    #[test]
    fn an_unspecified_bind_takes_the_peer_family() {
        let any_v4: SocketAddr = "0.0.0.0:7000".parse().unwrap();
        let peer_v6: SocketAddr = "[::1]:9000".parse().unwrap();
        assert_eq!(bind_toward(any_v4, peer_v6), "[::]:7000".parse().unwrap());
        let fixed: SocketAddr = "192.0.2.1:7000".parse().unwrap();
        assert_eq!(bind_toward(fixed, peer_v6), fixed);
    }
}
