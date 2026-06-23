//! Shared helpers for the str0m-based WebRTC elements (`WebRtcSink` WHIP egress
//! and `WebRtcWhepSrc` WHEP ingest): ICE host-candidate IP selection and the
//! HTTP SDP exchange. WHIP and WHEP are the same wire move - an
//! `application/sdp` POST of the local offer that returns the remote answer SDP.

use alloc::format;
use alloc::string::String;

use std::net::{IpAddr, Ipv4Addr, UdpSocket as StdUdpSocket};

use g2g_core::{G2gError, HardwareError};

/// Pick a route-local host IP for an ICE host candidate. Connecting a UDP socket
/// sends no packet; the OS just resolves the source address for the route to a
/// public address. Falls back to loopback when offline.
pub(crate) fn select_host_ip() -> IpAddr {
    if let Ok(s) = StdUdpSocket::bind(("0.0.0.0", 0)) {
        if s.connect(("8.8.8.8", 80)).is_ok() {
            if let Ok(addr) = s.local_addr() {
                return addr.ip();
            }
        }
    }
    IpAddr::V4(Ipv4Addr::LOCALHOST)
}

/// POST an SDP offer to a WHIP/WHEP endpoint (`application/sdp`) and return the
/// answer SDP from the response body.
pub(crate) async fn post_sdp(
    url: &str,
    bearer: Option<&str>,
    offer_sdp: String,
) -> Result<String, G2gError> {
    let client = reqwest::Client::new();
    let mut req = client.post(url).header("Content-Type", "application/sdp").body(offer_sdp);
    if let Some(token) = bearer {
        req = req.header("Authorization", format!("Bearer {token}"));
    }
    let resp = req.send().await.map_err(|_| G2gError::Hardware(HardwareError::Other))?;
    if !resp.status().is_success() {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    resp.text().await.map_err(|_| G2gError::Hardware(HardwareError::Other))
}
