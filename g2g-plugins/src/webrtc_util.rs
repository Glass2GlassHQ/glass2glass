//! Shared helpers for the str0m-based WebRTC elements (`WebRtcSink` WHIP egress
//! and `WebRtcWhepSrc` WHEP ingest): ICE host-candidate IP selection, the HTTP
//! SDP exchange, and trickle ICE (RFC 9725). WHIP and WHEP share the same wire
//! move: an `application/sdp` POST of the local offer that returns the remote
//! answer SDP plus a `Location` resource URL. The offer POSTs immediately with
//! the host candidate only; the server-reflexive (STUN) and relay (TURN)
//! candidates are gathered afterwards and trickled to the resource via `PATCH`
//! (`application/trickle-ice-sdpfrag`). On a clean end the resource is torn down
//! with `DELETE`.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use core::time::Duration;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket as StdUdpSocket};
use std::time::Instant;

use tokio::net::UdpSocket;

use str0m::net::{Protocol, Receive, Transmit};
use str0m::{Candidate, IceCreds, Input, Rtc};

use crate::turn::TurnSet;
use g2g_core::{G2gError, HardwareError};

/// STUN magic cookie (RFC 5389).
const STUN_MAGIC: u32 = 0x2112_A442;

/// The WHIP/WHEP resource created by the SDP POST, plus the answer body. The
/// resource URL (from the `Location` header, resolved absolute) and entity tag
/// (`ETag`) are what later `PATCH` (trickle / ICE restart) and `DELETE`
/// (teardown) target; both are optional (a minimal server may omit them).
#[derive(Debug, Default, Clone)]
pub(crate) struct WhipSession {
    pub answer: String,
    pub resource: Option<String>,
    pub etag: Option<String>,
}

/// TURN relay config passed to the trickle gather: `None` server = no relay.
/// `server` may be a comma-separated list (each entry a `host:port`, an RFC
/// 7065 `turn:` / `turns:` URI, or `turn://user:pass@host:port` with embedded
/// credentials overriding `user` / `pass`), one allocation per entry.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TurnConfig<'a> {
    pub server: Option<&'a str>,
    pub user: &'a str,
    pub pass: &'a str,
}

/// The ICE parameters (`a=ice-ufrag` / `a=ice-pwd`), the first `m=` line, and the
/// first `a=mid` parsed from our own offer SDP. Trickle and ICE-restart sdpfrags
/// carry these so the server can associate the candidates with the session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IceParams {
    pub ufrag: String,
    pub pwd: String,
    pub mline: String,
    pub mid: String,
}

/// Outcome of a trickle / ICE-restart `PATCH`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TricklePatch {
    /// 2xx: the server accepted the candidates / restart.
    Accepted,
    /// 405 / 501: the server does not implement trickle / restart. Degrade
    /// gracefully (the connection keeps whatever candidates it already has).
    Unsupported,
    /// Any other status, or a transport error. Non-fatal for trickle (host
    /// direct may still work); a signal to fall back for restart.
    Failed,
}

/// After the disconnect threshold before an ICE restart is attempted.
pub(crate) const ICE_RESTART_TIMEOUT: Duration = Duration::from_secs(3);

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
    // No IPv4 route: an IPv6-only host still gets a usable interface address
    // (the STUN / TURN codecs are family-agnostic, M718).
    if let Ok(s) = StdUdpSocket::bind(("::", 0)) {
        if s.connect(("2001:4860:4860::8888", 80)).is_ok() {
            if let Ok(addr) = s.local_addr() {
                return addr.ip();
            }
        }
    }
    IpAddr::V4(Ipv4Addr::LOCALHOST)
}

/// POST an SDP offer to a WHIP/WHEP endpoint (`application/sdp`) and return the
/// answer SDP plus the created resource (`Location` header, resolved absolute)
/// and its `ETag`, for later trickle / DELETE. Keeps the localhost-IPv4 retry
/// fallback.
pub(crate) async fn post_sdp(
    url: &str,
    bearer: Option<&str>,
    offer_sdp: String,
) -> Result<WhipSession, G2gError> {
    let client = reqwest::Client::new();
    let resp = match send_sdp(&client, url, bearer, offer_sdp.clone()).await {
        Ok(resp) => resp,
        Err(e) => {
            log_sdp_post_error(url, &e);
            if let Some(fallback_url) = localhost_ipv4_url(url) {
                std::eprintln!("retrying WHIP/WHEP SDP POST with {fallback_url}");
                send_sdp(&client, &fallback_url, bearer, offer_sdp)
                    .await
                    .map_err(|e| {
                        log_sdp_post_error(&fallback_url, &e);
                        G2gError::Hardware(HardwareError::Other)
                    })?
            } else {
                return Err(G2gError::Hardware(HardwareError::Other));
            }
        }
    };
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        std::eprintln!("webrtc sdp POST failed for {url}: HTTP {status}: {body}");
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    // Resolve the resource URL before consuming the response for its body:
    // `Location` may be relative and is resolved against the request URL.
    let resource = resp
        .headers()
        .get(reqwest::header::LOCATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|loc| resp.url().join(loc).ok())
        .map(|u| u.to_string());
    let etag = resp
        .headers()
        .get(reqwest::header::ETAG)
        .and_then(|v| v.to_str().ok())
        .map(String::from);
    let answer = resp.text().await.map_err(|e| {
        std::eprintln!("webrtc sdp response read failed for {url}: {e}");
        G2gError::Hardware(HardwareError::Other)
    })?;
    Ok(WhipSession {
        answer,
        resource,
        etag,
    })
}

async fn send_sdp(
    client: &reqwest::Client,
    url: &str,
    bearer: Option<&str>,
    offer_sdp: String,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut req = client
        .post(url)
        .header("Content-Type", "application/sdp")
        .body(offer_sdp);
    if let Some(token) = bearer {
        req = req.header("Authorization", format!("Bearer {token}"));
    }
    req.send().await
}

/// Parse the ICE params (`a=ice-ufrag`, `a=ice-pwd`), the first `m=` line, and
/// the first `a=mid` from our own offer SDP. Used to build a trickle / restart
/// sdpfrag that names the session's credentials and media. `None` if any of the
/// four is absent (a well-formed str0m offer always has them).
pub(crate) fn parse_ice_params(offer_sdp: &str) -> Option<IceParams> {
    let mut ufrag = None;
    let mut pwd = None;
    let mut mline = None;
    let mut mid = None;
    for line in offer_sdp.lines() {
        let line = line.trim_end();
        if let Some(v) = line.strip_prefix("a=ice-ufrag:") {
            ufrag.get_or_insert_with(|| v.to_string());
        } else if let Some(v) = line.strip_prefix("a=ice-pwd:") {
            pwd.get_or_insert_with(|| v.to_string());
        } else if let Some(v) = line.strip_prefix("m=") {
            mline.get_or_insert_with(|| alloc::format!("m={v}"));
        } else if let Some(v) = line.strip_prefix("a=mid:") {
            mid.get_or_insert_with(|| v.to_string());
        }
    }
    Some(IceParams {
        ufrag: ufrag?,
        pwd: pwd?,
        mline: mline?,
        mid: mid?,
    })
}

/// Build a `application/trickle-ice-sdpfrag` body (RFC 8840 / 9725): the
/// session ICE credentials, one `m=` section with its `a=mid`, and the
/// `a=candidate` lines. `candidates` are raw str0m candidate strings
/// (`candidate:...`), prefixed with `a=` here. `mid` is passed explicitly
/// because some servers (mediamtx/pion) reject non-numeric mids, so the caller
/// may retry with the m-line index in place of the real mid token.
pub(crate) fn build_sdpfrag(params: &IceParams, mid: &str, candidates: &[String]) -> String {
    let mut s = String::new();
    s.push_str(&format!("a=ice-ufrag:{}\r\n", params.ufrag));
    s.push_str(&format!("a=ice-pwd:{}\r\n", params.pwd));
    s.push_str(&format!("{}\r\n", params.mline));
    s.push_str(&format!("a=mid:{mid}\r\n"));
    for c in candidates {
        s.push_str(&format!("a={c}\r\n"));
    }
    s
}

/// PATCH an sdpfrag built from `params` + `candidates`, first with the real mid
/// token (RFC 8840), then once more with the m-line index (`"0"`, we always frag
/// the first m-line) when the server hard-rejects it: mediamtx parses the frag
/// mid with `ParseUint` because browser / pion mids are numeric, while str0m
/// mids are random tokens. With BUNDLE there is one transport, so the index
/// addresses the same candidate set. Returns the outcome plus any success body
/// (an ICE-restart answer frag).
pub(crate) async fn patch_frag_with_mid_fallback(
    resource: &str,
    bearer: Option<&str>,
    etag: Option<&str>,
    params: &IceParams,
    candidates: &[String],
) -> (TricklePatch, Option<String>) {
    let frag = build_sdpfrag(params, &params.mid, candidates);
    let (outcome, body) = patch_sdpfrag(resource, bearer, etag, frag).await;
    if outcome != TricklePatch::Failed || params.mid == "0" {
        return (outcome, body);
    }
    std::eprintln!("webrtc trickle PATCH retrying with m-line index mid");
    let frag = build_sdpfrag(params, "0", candidates);
    patch_sdpfrag(resource, bearer, etag, frag).await
}

/// PATCH an sdpfrag to the WHIP/WHEP resource (RFC 9725 trickle / ICE restart):
/// `If-Match` uses the stored `ETag` when present, else `"*"`. Maps the response
/// status to a [`TricklePatch`]; a 405/501 means the server has no trickle
/// support and the caller degrades gracefully. On success the response body is
/// returned too: a restart PATCH answers 200 with the server's new frag.
pub(crate) async fn patch_sdpfrag(
    resource: &str,
    bearer: Option<&str>,
    etag: Option<&str>,
    frag: String,
) -> (TricklePatch, Option<String>) {
    let client = reqwest::Client::new();
    let mut req = client
        .patch(resource)
        .header("Content-Type", "application/trickle-ice-sdpfrag")
        .header("If-Match", etag.unwrap_or("*"))
        .body(frag);
    if let Some(token) = bearer {
        req = req.header("Authorization", format!("Bearer {token}"));
    }
    match req.send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            if resp.status().is_success() {
                let body = resp.text().await.ok().filter(|b| !b.is_empty());
                (TricklePatch::Accepted, body)
            } else if status == 405 || status == 501 {
                std::eprintln!("webrtc trickle PATCH unsupported by {resource}: HTTP {status}");
                (TricklePatch::Unsupported, None)
            } else {
                let body = resp.text().await.unwrap_or_default();
                std::eprintln!("webrtc trickle PATCH to {resource} failed: HTTP {status}: {body}");
                (TricklePatch::Failed, None)
            }
        }
        Err(e) => {
            std::eprintln!("webrtc trickle PATCH to {resource} errored: {e}");
            (TricklePatch::Failed, None)
        }
    }
}

/// DELETE the WHIP/WHEP resource on a clean end (RFC 9725 teardown). Best
/// effort: logs the outcome and never fails the caller.
pub(crate) async fn delete_resource(resource: &str, bearer: Option<&str>) {
    let client = reqwest::Client::new();
    let mut req = client.delete(resource);
    if let Some(token) = bearer {
        req = req.header("Authorization", format!("Bearer {token}"));
    }
    match req.send().await {
        Ok(resp) => std::eprintln!("webrtc DELETE {resource}: HTTP {}", resp.status().as_u16()),
        Err(e) => std::eprintln!("webrtc DELETE {resource} errored: {e}"),
    }
}

/// Add just the ICE host candidate to `rtc`, before the offer is built, so the
/// initial offer already carries a directly reachable candidate (trickle sends
/// the reflexive / relay ones after the POST).
pub(crate) fn add_host_candidate(rtc: &mut Rtc, socket: &UdpSocket) -> Result<(), G2gError> {
    let local = socket
        .local_addr()
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
    if let Ok(host) = Candidate::host(local, "udp") {
        rtc.add_local_candidate(host);
    }
    Ok(())
}

/// Gather the reflexive (STUN) and relay (TURN) ICE candidates on `socket`
/// *after* the offer has been POSTed, add them to `rtc` as local candidates, and
/// trickle them to `resource` via PATCH. Returns the live TURN client (or
/// `None`). Runs before the str0m loop takes the socket, so the STUN reply / TURN
/// handshake read directly with no contention; the server's answer candidates
/// arrive during this window, cutting connect latency. Degrades gracefully at
/// every step: no STUN/TURN, no resource, or a server that rejects the PATCH all
/// continue with whatever candidates the connection already has.
pub(crate) async fn trickle_candidates(
    rtc: &mut Rtc,
    socket: &UdpSocket,
    offer_sdp: &str,
    session: &WhipSession,
    bearer: Option<&str>,
    stun_server: Option<&str>,
    turn_cfg: TurnConfig<'_>,
) -> TurnSet {
    let Ok(local) = socket.local_addr() else {
        return TurnSet::empty();
    };
    let mut lines: Vec<String> = Vec::new();

    if let Some(server) = stun_server {
        if let Some(stun_addr) = tokio::net::lookup_host(server)
            .await
            .ok()
            .and_then(|mut a| a.next())
        {
            if let Some(srflx) = gather_srflx(socket, stun_addr).await {
                if let Ok(c) = Candidate::server_reflexive(srflx, local, "udp") {
                    lines.push(c.to_sdp_string());
                    rtc.add_local_candidate(c);
                }
            }
        }
    }

    // TURN allocate (adds each relayed candidate to `rtc` inside setup);
    // rebuild the same candidate strings for the trickle frag.
    let turn = match turn_cfg.server {
        Some(servers) => TurnSet::setup(rtc, socket, servers, turn_cfg.user, turn_cfg.pass).await,
        None => TurnSet::empty(),
    };
    for relay in turn.relay_addrs() {
        if let Ok(c) = Candidate::relayed(relay, local, "udp") {
            lines.push(c.to_sdp_string());
        }
    }

    if !lines.is_empty() {
        if let (Some(resource), Some(params)) =
            (session.resource.as_deref(), parse_ice_params(offer_sdp))
        {
            patch_frag_with_mid_fallback(
                resource,
                bearer,
                session.etag.as_deref(),
                &params,
                &lines,
            )
            .await;
        }
    }
    turn
}

/// Attempt an ICE restart against the WHIP/WHEP resource (RFC 9725): mint fresh
/// ICE credentials on `rtc`, build a new offer keeping the existing local
/// candidates, and PATCH it as an sdpfrag carrying the new ufrag/pwd +
/// candidates. A restarting server answers 200 with its own frag (new remote
/// credentials + candidates), which must be applied or every connectivity check
/// fails its message integrity: the creds go in via the direct API and the
/// candidates via `add_remote_candidate`. On 405/501/error the caller falls
/// back (re-POST a fresh session).
pub(crate) async fn ice_restart(
    rtc: &mut Rtc,
    resource: &str,
    bearer: Option<&str>,
    etag: Option<&str>,
) -> TricklePatch {
    let (offer_sdp, ufrag, pwd) = {
        let mut api = rtc.sdp_api();
        let creds = api.ice_restart(true);
        match api.apply() {
            Some((offer, _pending)) => (offer.to_sdp_string(), creds.ufrag, creds.pass),
            None => return TricklePatch::Failed,
        }
    };
    let Some(mut params) = parse_ice_params(&offer_sdp) else {
        return TricklePatch::Failed;
    };
    // The restart offer carries the fresh credentials; prefer str0m's returned
    // creds (authoritative) over the parse.
    params.ufrag = ufrag;
    params.pwd = pwd;
    let candidates: Vec<String> = offer_sdp
        .lines()
        .filter_map(|l| l.trim_end().strip_prefix("a=").map(String::from))
        .filter(|l| l.starts_with("candidate:"))
        .collect();
    let (outcome, body) =
        patch_frag_with_mid_fallback(resource, bearer, etag, &params, &candidates).await;
    if outcome == TricklePatch::Accepted {
        if let Some(frag) = body {
            apply_restart_answer_frag(rtc, &frag);
        }
    }
    outcome
}

/// Parse a restart PATCH's 200 answer frag into the server's new ICE
/// credentials and its candidates. Lines that fail to parse are skipped;
/// peer-reflexive discovery still converges if the server checks first.
fn parse_restart_answer_frag(frag: &str) -> (Option<IceCreds>, Vec<Candidate>) {
    let mut ufrag = None;
    let mut pwd = None;
    let mut candidates = Vec::new();
    for line in frag.lines() {
        let line = line.trim_end();
        if let Some(v) = line.strip_prefix("a=ice-ufrag:") {
            ufrag.get_or_insert_with(|| v.to_string());
        } else if let Some(v) = line.strip_prefix("a=ice-pwd:") {
            pwd.get_or_insert_with(|| v.to_string());
        } else if let Some(c) = line.strip_prefix("a=candidate:") {
            if let Ok(cand) = Candidate::from_sdp_string(&format!("candidate:{c}")) {
                candidates.push(cand);
            }
        }
    }
    let creds = match (ufrag, pwd) {
        (Some(ufrag), Some(pass)) => Some(IceCreds { ufrag, pass }),
        _ => None,
    };
    (creds, candidates)
}

/// Apply a restart PATCH's 200 answer frag: the server's new ICE credentials
/// (required, or our checks are signed with the stale password) and its
/// candidate lines.
fn apply_restart_answer_frag(rtc: &mut Rtc, frag: &str) {
    let (creds, candidates) = parse_restart_answer_frag(frag);
    for cand in candidates {
        rtc.add_remote_candidate(cand);
    }
    if let Some(creds) = creds {
        rtc.direct_api().set_remote_ice_credentials(creds);
    }
}

fn localhost_ipv4_url(url: &str) -> Option<String> {
    if url.starts_with("http://localhost:") || url.starts_with("https://localhost:") {
        Some(url.replacen("://localhost:", "://127.0.0.1:", 1))
    } else {
        None
    }
}

fn log_sdp_post_error(url: &str, e: &reqwest::Error) {
    std::eprintln!("webrtc sdp POST failed for {url}: {e:?}");
    let mut source = std::error::Error::source(e);
    while let Some(e) = source {
        std::eprintln!("  caused by: {e}");
        source = e.source();
    }
}

/// Add this socket's ICE candidates to `rtc` up front: the host candidate plus a
/// server-reflexive (STUN) candidate when `stun_server` is set. Used by the P2P
/// duplex path, which exchanges SDP directly (no WHIP/WHEP resource) and so
/// cannot trickle: all candidates must be in the offer. The WHIP/WHEP elements
/// use [`add_host_candidate`] + [`trickle_candidates`] instead.
pub(crate) async fn add_ice_candidates(
    rtc: &mut Rtc,
    socket: &UdpSocket,
    stun_server: Option<&str>,
) -> Result<(), G2gError> {
    let local = socket
        .local_addr()
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
    if let Ok(host) = Candidate::host(local, "udp") {
        rtc.add_local_candidate(host);
    }
    if let Some(server) = stun_server {
        if let Some(stun_addr) = tokio::net::lookup_host(server)
            .await
            .ok()
            .and_then(|mut a| a.next())
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

/// Drive `rtc` briefly after the media ends so packets still queued in str0m's
/// pacer reach the wire before the socket drops (graceful EOS flush, M726).
/// Bounded to a short window: the media is done, this only flushes the tail.
pub(crate) async fn drain_pacer(rtc: &mut Rtc, socket: &UdpSocket, turn: &mut TurnSet) {
    use str0m::Output;
    let deadline = Instant::now() + Duration::from_millis(250);
    loop {
        let now = Instant::now();
        if now >= deadline {
            return;
        }
        match rtc.poll_output() {
            Ok(Output::Transmit(t)) => send_transmit(socket, turn, &t).await,
            Ok(Output::Timeout(t)) => {
                let wake = t.min(deadline);
                let wait = wake.saturating_duration_since(Instant::now());
                if !wait.is_zero() {
                    tokio::time::sleep(wait).await;
                }
                let _ = rtc.handle_input(Input::Timeout(Instant::now()));
            }
            Ok(Output::Event(_)) => {}
            Err(_) => return,
        }
    }
}

/// Send one str0m `Transmit`: a relay-sourced datagram goes through TURN (a Send
/// indication to the server, after ensuring a permission for the peer); a direct
/// host/srflx datagram goes straight out. Shared by every webrtc element's poll
/// loop.
pub(crate) async fn send_transmit(socket: &UdpSocket, turn: &mut TurnSet, t: &Transmit) {
    match turn.for_relay(t.source) {
        Some(tc) => {
            let _ = tc.ensure_permission(socket, t.destination).await;
            let wrapped = tc.wrap_send(t.destination, &t.contents);
            let _ = socket.send_to(&wrapped, tc.server_addr()).await;
        }
        None => {
            let _ = socket.send_to(&t.contents, t.destination).await;
        }
    }
}

/// Feed one received UDP datagram into `rtc`. A datagram from the TURN server is
/// unwrapped to its (peer, payload) Data indication and fed as if it arrived on
/// the relay candidate (control responses unwrap to `None` and are dropped); any
/// other datagram is fed directly from `source` to `local`. Returns `false` when
/// str0m rejects the input, so a caller that tears down on error can stop; the
/// callers that ignore transient input errors discard the result.
pub(crate) fn feed_datagram(
    rtc: &mut Rtc,
    turn: &mut TurnSet,
    local: SocketAddr,
    datagram: &[u8],
    source: SocketAddr,
) -> bool {
    if let Some(tc) = turn.for_server(source) {
        if let Some((peer, payload)) = tc.parse_data(datagram) {
            let relay = tc.relay_addr();
            if let Ok(contents) = payload.as_slice().try_into() {
                let input = Input::Receive(
                    Instant::now(),
                    Receive {
                        proto: Protocol::Udp,
                        source: peer,
                        destination: relay,
                        contents,
                    },
                );
                return rtc.handle_input(input).is_ok();
            }
        }
    } else if let Ok(contents) = datagram.try_into() {
        let input = Input::Receive(
            Instant::now(),
            Receive {
                proto: Protocol::Udp,
                source,
                destination: local,
                contents,
            },
        );
        return rtc.handle_input(input).is_ok();
    }
    true
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

    // Loop until our Binding response arrives or the deadline passes: because the
    // offer is already POSTed, the peer's ICE connectivity checks land on this
    // socket too, so skip any datagram that is not our txn's response (a dropped
    // check is retransmitted once the run loop takes over).
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut buf = [0u8; 512];
    loop {
        let remaining = deadline.checked_duration_since(Instant::now())?;
        let (n, _from) = tokio::time::timeout(remaining, socket.recv_from(&mut buf))
            .await
            .ok()?
            .ok()?;
        if let Some(addr) = parse_xor_mapped_address(&buf[..n], &txn) {
            return Some(addr);
        }
    }
}

/// Parse a STUN Binding Success Response for the (XOR-)MAPPED-ADDRESS, verifying
/// the transaction id matches our request. IPv4 or IPv6 (the latter XORed with
/// cookie + transaction id, M718).
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
        // 0x0020 = XOR-MAPPED-ADDRESS, 0x0001 = MAPPED-ADDRESS.
        if (atype == 0x0020 || atype == 0x0001) && val.len() >= 8 {
            let xored = atype == 0x0020;
            let mut port = u16::from_be_bytes([val[2], val[3]]);
            if xored {
                port ^= (STUN_MAGIC >> 16) as u16;
            }
            match val[1] {
                0x01 => {
                    let mut ip = [val[4], val[5], val[6], val[7]];
                    if xored {
                        for k in 0..4 {
                            ip[k] ^= magic[k];
                        }
                    }
                    return Some(SocketAddr::from((Ipv4Addr::from(ip), port)));
                }
                0x02 if val.len() >= 20 => {
                    let mut ip = [0u8; 16];
                    ip.copy_from_slice(&val[4..20]);
                    if xored {
                        for k in 0..16 {
                            ip[k] ^= if k < 4 { magic[k] } else { txn[k - 4] };
                        }
                    }
                    return Some(SocketAddr::from((std::net::Ipv6Addr::from(ip), port)));
                }
                _ => {}
            }
        }
        // Attributes are 4-byte aligned.
        i = val_start + alen + ((4 - (alen % 4)) % 4);
    }
    None
}

#[cfg(fuzzing)]
pub fn fuzz_parse(data: &[u8]) {
    let _ = parse_xor_mapped_address(data, &[0u8; 12]);
    let text = String::from_utf8_lossy(data);
    let _ = parse_ice_params(&text);
    let _ = parse_restart_answer_frag(&text);
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
        assert_eq!(
            addr,
            SocketAddr::from((Ipv4Addr::new(203, 0, 113, 7), 51234))
        );
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

    #[test]
    fn parses_ice_params_from_offer() {
        // A trimmed str0m-style offer: session line, an m= section, and the
        // media-level ICE credentials + mid.
        let offer = "v=0\r\n\
            o=- 1 1 IN IP4 0.0.0.0\r\n\
            m=video 9 UDP/TLS/RTP/SAVPF 96\r\n\
            a=ice-ufrag:UF01\r\n\
            a=ice-pwd:PW0123456789\r\n\
            a=mid:0\r\n\
            a=candidate:1 1 udp 2113 10.0.0.2 5000 typ host\r\n";
        let p = parse_ice_params(offer).expect("parses");
        assert_eq!(p.ufrag, "UF01");
        assert_eq!(p.pwd, "PW0123456789");
        assert_eq!(p.mline, "m=video 9 UDP/TLS/RTP/SAVPF 96");
        assert_eq!(p.mid, "0");
        // A missing field yields None.
        assert!(parse_ice_params("v=0\r\nm=video 9 x 96\r\n").is_none());
    }

    /// A restart answer frag as mediamtx actually returns it (captured live,
    /// 2026-07-19: media-level creds, per-candidate `ufrag` suffix, IPv6 rows,
    /// `a=end-of-candidates`): the parser must extract the new credentials and
    /// every parseable candidate.
    #[test]
    fn parses_restart_answer_frag() {
        let frag = "a=ice-options:trickle ice2\r\n\
             m=video 9 UDP/TLS/RTP/SAVPF 127 108\r\n\
             a=mid:pGh\r\n\
             a=ice-ufrag:ZSfcZbhuWoqBrXDe\r\n\
             a=ice-pwd:mgnIlmoZjcVsSCpHttvVXhCVwAdWPedw\r\n\
             a=candidate:2878742611 1 udp 2130706431 127.0.0.1 8189 typ host ufrag ZSfcZbhuWoqBrXDe\r\n\
             a=candidate:2846313311 1 udp 2130706431 192.168.2.21 8189 typ host ufrag ZSfcZbhuWoqBrXDe\r\n\
             a=candidate:1682108319 1 udp 2130706431 fd7a:115c:a1e0::363a:3e3b 8189 typ host ufrag ZSfcZbhuWoqBrXDe\r\n\
             a=end-of-candidates\r\n\
             m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n\
             a=mid:crP\r\n\
             a=ice-ufrag:ZSfcZbhuWoqBrXDe\r\n";
        let (creds, candidates) = parse_restart_answer_frag(frag);
        let creds = creds.expect("creds parsed");
        assert_eq!(creds.ufrag, "ZSfcZbhuWoqBrXDe");
        assert_eq!(creds.pass, "mgnIlmoZjcVsSCpHttvVXhCVwAdWPedw");
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0].addr().to_string(), "127.0.0.1:8189");
    }

    #[test]
    fn builds_trickle_sdpfrag() {
        let params = IceParams {
            ufrag: "UF01".into(),
            pwd: "PW0123456789".into(),
            mline: "m=video 9 UDP/TLS/RTP/SAVPF 96".into(),
            mid: "0".into(),
        };
        let cands = alloc::vec![
            "candidate:1 1 udp 2113 10.0.0.2 5000 typ host".to_string(),
            "candidate:2 1 udp 1694 203.0.113.7 6000 typ srflx raddr 10.0.0.2 rport 5000"
                .to_string(),
        ];
        let frag = build_sdpfrag(&params, &params.mid, &cands);
        assert_eq!(
            frag,
            "a=ice-ufrag:UF01\r\n\
             a=ice-pwd:PW0123456789\r\n\
             m=video 9 UDP/TLS/RTP/SAVPF 96\r\n\
             a=mid:0\r\n\
             a=candidate:1 1 udp 2113 10.0.0.2 5000 typ host\r\n\
             a=candidate:2 1 udp 1694 203.0.113.7 6000 typ srflx raddr 10.0.0.2 rport 5000\r\n"
        );
        // A non-numeric real mid can be swapped for the m-line index (mediamtx
        // interop, see patch_frag_with_mid_fallback).
        let frag = build_sdpfrag(&params, "0", &cands);
        assert!(frag.contains("a=mid:0\r\n"));
    }

    /// The SDP POST must capture the resource URL (relative `Location` resolved
    /// against the request URL) and the `ETag`. A hand-rolled one-shot TCP
    /// responder stands in for the WHIP server (no new HTTP-server dependency).
    #[tokio::test]
    async fn post_sdp_captures_location_and_etag() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            // Drain the request (headers + body) enough that reqwest's write
            // completes; a single read is sufficient for the small offer.
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf).await.unwrap();
            let body = "v=0\r\no=- 2 2 IN IP4 0.0.0.0\r\n";
            let resp = alloc::format!(
                "HTTP/1.1 201 Created\r\n\
                 Content-Type: application/sdp\r\n\
                 Location: /whip/resource/abc123\r\n\
                 ETag: \"etag-xyz\"\r\n\
                 Content-Length: {}\r\n\
                 \r\n{}",
                body.len(),
                body
            );
            stream.write_all(resp.as_bytes()).await.unwrap();
            stream.flush().await.unwrap();
        });

        let url = alloc::format!("http://127.0.0.1:{port}/whip");
        let session = post_sdp(&url, None, "v=0\r\n".to_string())
            .await
            .expect("post ok");
        server.await.unwrap();

        assert_eq!(
            session.resource.as_deref(),
            Some(alloc::format!("http://127.0.0.1:{port}/whip/resource/abc123").as_str())
        );
        assert_eq!(session.etag.as_deref(), Some("\"etag-xyz\""));
        assert_eq!(session.answer, "v=0\r\no=- 2 2 IN IP4 0.0.0.0\r\n");
    }
}
