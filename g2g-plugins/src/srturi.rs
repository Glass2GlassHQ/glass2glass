use alloc::string::String;
use std::net::IpAddr;

use g2g_core::PropError;

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
    pub host: Option<IpAddr>,
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

const SCHEME: &str = "srt://";

pub fn parse_srt_uri(uri: &str) -> Result<SrtUri, PropError> {
    let rest = uri.strip_prefix(SCHEME).ok_or(PropError::Value)?;
    let (authority, query) = rest.split_once('?').unwrap_or((rest, ""));
    let (host, port) = parse_authority(authority.trim_end_matches('/'))?;
    let mut parsed = SrtUri {
        host,
        port,
        mode: if host.is_some() {
            SrtMode::Caller
        } else {
            SrtMode::Listener
        },
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

// `[v6]:port`, `v4:port`, `:port`, or a bare host
fn parse_authority(authority: &str) -> Result<(Option<IpAddr>, Option<u16>), PropError> {
    let (host, port) = match authority.strip_prefix('[') {
        Some(bracketed) => {
            let (host, after) = bracketed.split_once(']').ok_or(PropError::Value)?;
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
    // TODO: resolve host names, gst's srt elements take them
    let host = (!host.is_empty())
        .then(|| parse_value::<IpAddr>(host))
        .transpose()?;
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
        assert_eq!(caller.host, Some("192.0.2.7".parse().unwrap()));
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
        assert_eq!(uri.host, Some("2001:db8::1".parse().unwrap()));
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
}
