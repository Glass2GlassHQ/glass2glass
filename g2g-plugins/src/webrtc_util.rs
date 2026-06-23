//! Shared helpers for the str0m-based WebRTC elements (`WebRtcSink` WHIP egress
//! and `WebRtcWhepSrc` WHEP ingest): ICE host-candidate IP selection and the
//! HTTP SDP exchange. WHIP and WHEP are the same wire move - an
//! `application/sdp` POST of the local offer that returns the remote answer SDP.

use alloc::format;
use alloc::string::String;

use core::time::Duration;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket as StdUdpSocket};

use tokio::net::UdpSocket;

use str0m::{Candidate, Rtc};

use g2g_core::{G2gError, HardwareError};

/// STUN magic cookie (RFC 5389).
const STUN_MAGIC: u32 = 0x2112_A442;

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

/// Add this socket's ICE candidates to `rtc`: always the host candidate, plus a
/// server-reflexive (public) candidate discovered through `stun_server` when one
/// is configured. The reflexive candidate is what a cloud SFU across NAT can
/// actually reach; host-only works on a LAN. STUN failures degrade gracefully to
/// host-only (the run continues), so an unreachable STUN server only costs a
/// short timeout. `stun_server` is `host:port` (resolved here).
pub(crate) async fn add_ice_candidates(
    rtc: &mut Rtc,
    socket: &UdpSocket,
    stun_server: Option<&str>,
) -> Result<(), G2gError> {
    let local = socket.local_addr().map_err(|_| G2gError::Hardware(HardwareError::Other))?;
    if let Ok(host) = Candidate::host(local, "udp") {
        rtc.add_local_candidate(host);
    }
    if let Some(server) = stun_server {
        if let Some(stun_addr) = tokio::net::lookup_host(server).await.ok().and_then(|mut a| a.next())
        {
            if let Some(srflx) = gather_srflx(socket, stun_addr).await {
                if let Ok(c) = Candidate::server_reflexive(srflx, local, "udp") {
                    rtc.add_local_candidate(c);
                }
            }
        }
    }
    Ok(())
}

/// Discover this socket's server-reflexive (public) address via a STUN Binding
/// Request to `stun_server` (RFC 5389). Sends on the same socket str0m will use
/// for ICE, so the mapped address matches that NAT binding. Returns `None` on
/// timeout / parse failure. IPv4 only in v1. Done once during setup, before the
/// str0m loop owns the socket, so there is no read contention.
async fn gather_srflx(socket: &UdpSocket, stun_server: SocketAddr) -> Option<SocketAddr> {
    // Binding Request: type 0x0001, length 0, magic cookie, 12-byte txn id.
    // The txn id need not be cryptographic for a one-shot public binding; vary
    // it by the local + server ports so concurrent sockets don't collide.
    let local_port = socket.local_addr().ok()?.port();
    let mut txn = [0u8; 12];
    txn[0..2].copy_from_slice(&local_port.to_be_bytes());
    txn[2..4].copy_from_slice(&stun_server.port().to_be_bytes());
    txn[4..8].copy_from_slice(&STUN_MAGIC.to_be_bytes());

    let mut req = [0u8; 20];
    req[0..2].copy_from_slice(&0x0001u16.to_be_bytes()); // Binding Request
    // bytes 2..4 = length 0
    req[4..8].copy_from_slice(&STUN_MAGIC.to_be_bytes());
    req[8..20].copy_from_slice(&txn);

    socket.send_to(&req, stun_server).await.ok()?;

    let mut buf = [0u8; 512];
    let (n, _from) = tokio::time::timeout(Duration::from_secs(2), socket.recv_from(&mut buf))
        .await
        .ok()?
        .ok()?;
    parse_xor_mapped_address(&buf[..n], &txn)
}

/// Parse a STUN Binding Success Response for the (XOR-)MAPPED-ADDRESS, verifying
/// the transaction id matches our request. IPv4 only.
fn parse_xor_mapped_address(msg: &[u8], txn: &[u8; 12]) -> Option<SocketAddr> {
    if msg.len() < 20 {
        return None;
    }
    // 0x0101 = Binding Success Response; the txn id (bytes 8..20) is echoed.
    if u16::from_be_bytes([msg[0], msg[1]]) != 0x0101 || &msg[8..20] != txn {
        return None;
    }
    let magic = STUN_MAGIC.to_be_bytes();
    let mut i = 20;
    while i + 4 <= msg.len() {
        let atype = u16::from_be_bytes([msg[i], msg[i + 1]]);
        let alen = u16::from_be_bytes([msg[i + 2], msg[i + 3]]) as usize;
        let val_start = i + 4;
        if val_start + alen > msg.len() {
            break;
        }
        let val = &msg[val_start..val_start + alen];
        // 0x0020 = XOR-MAPPED-ADDRESS, 0x0001 = MAPPED-ADDRESS; family 0x01 = IPv4.
        if (atype == 0x0020 || atype == 0x0001) && val.len() >= 8 && val[1] == 0x01 {
            let mut port = u16::from_be_bytes([val[2], val[3]]);
            let mut ip = [val[4], val[5], val[6], val[7]];
            if atype == 0x0020 {
                port ^= (STUN_MAGIC >> 16) as u16;
                for k in 0..4 {
                    ip[k] ^= magic[k];
                }
            }
            return Some(SocketAddr::from((Ipv4Addr::from(ip), port)));
        }
        // Attributes are 4-byte aligned.
        i = val_start + alen + ((4 - (alen % 4)) % 4);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_xor_mapped_address() {
        let txn = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        // Build a Binding Success Response with one XOR-MAPPED-ADDRESS attribute
        // encoding 203.0.113.7:51234.
        let ip = [203u8, 0, 113, 7];
        let port: u16 = 51234;
        let magic = STUN_MAGIC.to_be_bytes();
        let mut xip = ip;
        for k in 0..4 {
            xip[k] ^= magic[k];
        }
        let xport = port ^ (STUN_MAGIC >> 16) as u16;
        let mut msg = alloc::vec::Vec::new();
        msg.extend_from_slice(&0x0101u16.to_be_bytes()); // success response
        msg.extend_from_slice(&12u16.to_be_bytes()); // attr section length
        msg.extend_from_slice(&STUN_MAGIC.to_be_bytes());
        msg.extend_from_slice(&txn);
        msg.extend_from_slice(&0x0020u16.to_be_bytes()); // XOR-MAPPED-ADDRESS
        msg.extend_from_slice(&8u16.to_be_bytes()); // attr length
        msg.push(0); // reserved
        msg.push(0x01); // family IPv4
        msg.extend_from_slice(&xport.to_be_bytes());
        msg.extend_from_slice(&xip);

        let addr = parse_xor_mapped_address(&msg, &txn).expect("parses");
        assert_eq!(addr, SocketAddr::from((Ipv4Addr::new(203, 0, 113, 7), 51234)));
    }

    #[test]
    fn rejects_wrong_txn_or_type() {
        let txn = [0u8; 12];
        let mut msg = alloc::vec::Vec::new();
        msg.extend_from_slice(&0x0001u16.to_be_bytes()); // request, not response
        msg.extend_from_slice(&0u16.to_be_bytes());
        msg.extend_from_slice(&STUN_MAGIC.to_be_bytes());
        msg.extend_from_slice(&txn);
        assert!(parse_xor_mapped_address(&msg, &txn).is_none());
    }
}
