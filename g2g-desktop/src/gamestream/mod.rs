// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

mod control;
mod http;
mod identity;
mod input;
mod pairing;
mod rtsp;
mod stream;
mod tls;

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::rc::Rc;
use std::time::Duration;

use g2g_core::log::Target;
use mdns_sd::{ServiceDaemon, ServiceInfo};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::CertificateDer;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{oneshot, Notify};
use tokio::task::{spawn_local, JoinHandle};
use tokio_rustls::TlsAcceptor;

use http::{read_request, write_reply, Reply, Request};
use identity::{HostIdentity, PairedClients};
use pairing::Pairings;
use rtsp::{RtspReply, RtspRequest, StreamSettings};
pub use stream::{CaptureFactory, InjectorFactory, ServedDesktop};
use stream::{AUDIO_PORT, VIDEO_PORT};

pub const HTTP_PORT: u16 = 47989;
pub const HTTPS_PORT: u16 = 47984;
const RTSP_PORT: u16 = 48010;
const LOG_CATEGORY: &str = "g2g-desktop-gamestream";
const SERVICE_TYPE: &str = "_nvstream._tcp.local.";
// moonlight takes a negative fourth number as a Sunshine host, which selects SHA-256 pairing and RTSP over TCP
const APP_VERSION: &str = "7.1.431.-1";
const GFE_VERSION: &str = "3.23.0.74";
// ServerCodecModeSupport bits SCM_H264 (0x1) and SCM_HEVC (0x100)
const CODEC_MODES: u32 = 0x101;
// moonlight skips HEVC when this is 0
const MAX_LUMA_PIXELS_HEVC: u64 = 1_869_449_984;
const DESKTOP_APP_ID: u32 = 1;
const DESKTOP_APP_TITLE: &str = "Desktop";
const PIN_WAIT: Duration = Duration::from_secs(5 * 60);
const PIN_DIGITS: usize = 4;
const SALT_BYTES: usize = 16;
const SESSION_KEY_BYTES: usize = 16;
const SESSION_STOP_WAIT: Duration = Duration::from_secs(2);

#[derive(Debug)]
pub enum GamestreamError {
    Io(io::Error),
    Tls(rustls::Error),
    Certificate(rcgen::Error),
    Mdns(mdns_sd::Error),
    NoConfigDirectory,
    BadHostCertificate,
    BadHostKey,
    KeyGeneration,
    Signing,
    PinRefused(String),
}

impl fmt::Display for GamestreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Tls(e) => write!(f, "TLS: {e}"),
            Self::Certificate(e) => write!(f, "host certificate: {e}"),
            Self::Mdns(e) => write!(f, "mDNS: {e}"),
            Self::NoConfigDirectory => f.write_str("no config directory for the host identity"),
            Self::BadHostCertificate => f.write_str("the stored host certificate does not parse"),
            Self::BadHostKey => f.write_str("the stored host key is not an RSA PKCS#8 key"),
            Self::KeyGeneration => f.write_str("could not generate the host key"),
            Self::Signing => f.write_str("could not sign with the host key"),
            Self::PinRefused(message) => write!(f, "the host refused the PIN: {message}"),
        }
    }
}

impl std::error::Error for GamestreamError {}

impl From<io::Error> for GamestreamError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<rustls::Error> for GamestreamError {
    fn from(e: rustls::Error) -> Self {
        Self::Tls(e)
    }
}

impl From<rcgen::Error> for GamestreamError {
    fn from(e: rcgen::Error) -> Self {
        Self::Certificate(e)
    }
}

impl From<mdns_sd::Error> for GamestreamError {
    fn from(e: mdns_sd::Error) -> Self {
        Self::Mdns(e)
    }
}

#[derive(Debug)]
struct Session {
    client: IpAddr,
    settings: Option<StreamSettings>,
    play: Option<oneshot::Sender<StreamSettings>>,
    stop: Rc<Notify>,
    task: JoinHandle<()>,
}

struct Host {
    name: String,
    identity: HostIdentity,
    paired: PairedClients,
    pairings: Pairings,
    desktop: ServedDesktop,
    session: RefCell<Option<Session>>,
}

impl fmt::Debug for Host {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Host")
            .field("name", &self.name)
            .field("session", &self.session)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy)]
struct Connection<'a> {
    local: SocketAddr,
    peer: SocketAddr,
    // only the HTTPS port carries a client certificate
    client_certificate: Option<&'a [u8]>,
    https: bool,
}

impl Host {
    fn load(directory: &Path, desktop: ServedDesktop) -> Result<Self, GamestreamError> {
        Ok(Self {
            name: gethostname::gethostname().to_string_lossy().into_owned(),
            identity: HostIdentity::load_or_create(directory)?,
            paired: PairedClients::load(directory)?,
            pairings: Pairings::default(),
            desktop,
            session: RefCell::new(None),
        })
    }

    fn is_paired(&self, connection: Connection<'_>) -> bool {
        connection
            .client_certificate
            .is_some_and(|certificate| self.paired.contains(certificate))
    }

    async fn route(self: &Rc<Self>, request: Request, connection: Connection<'_>) -> Reply {
        match (
            request.method.as_str(),
            request.path.as_str(),
            connection.https,
        ) {
            ("GET", "/serverinfo", _) => self.server_info(connection),
            ("GET", "/pair", _) => self.pair(&request.query, connection).await,
            ("GET", "/applist" | "/launch" | "/resume" | "/cancel", true)
                if !self.is_paired(connection) =>
            {
                Reply::xml_error(401, "client is not paired")
            }
            ("GET", "/applist", true) => app_list(),
            ("GET", "/launch", true) => {
                self.launch(&request.query, connection, "gamesession").await
            }
            // moonlight's /resume carries a fresh rikey
            ("GET", "/resume", true) => self.launch(&request.query, connection, "resume").await,
            ("GET", "/cancel", true) => {
                self.end_session().await;
                Reply::xml("<cancel>1</cancel>")
            }
            ("POST", "/submit-pin", false) => self.submit_pin(&request.body, connection),
            _ => Reply::not_found(),
        }
    }

    fn server_info(&self, connection: Connection<'_>) -> Reply {
        let pair_status = u8::from(self.is_paired(connection));
        let (current_game, state) = match self.session.borrow().is_some() {
            true => (DESKTOP_APP_ID, "G2G_SERVER_BUSY"),
            false => (0, "G2G_SERVER_FREE"),
        };
        Reply::xml(&format!(
            "<hostname>{}</hostname><appversion>{APP_VERSION}</appversion>\
             <GfeVersion>{GFE_VERSION}</GfeVersion><uniqueid>{}</uniqueid>\
             <HttpsPort>{HTTPS_PORT}</HttpsPort><ExternalPort>{HTTP_PORT}</ExternalPort>\
             <mac>00:00:00:00:00:00</mac><MaxLumaPixelsHEVC>{MAX_LUMA_PIXELS_HEVC}</MaxLumaPixelsHEVC>\
             <LocalIP>{}</LocalIP><ServerCodecModeSupport>{CODEC_MODES}</ServerCodecModeSupport>\
             <SupportedDisplayMode></SupportedDisplayMode><PairStatus>{pair_status}</PairStatus>\
             <currentgame>{current_game}</currentgame><state>{state}</state>",
            http::escape_xml(&self.name),
            self.identity.unique_id,
            connection.local.ip(),
        ))
    }

    async fn pair(&self, query: &HashMap<String, String>, connection: Connection<'_>) -> Reply {
        let Some(unique_id) = query.get("uniqueid") else {
            return Reply::xml_error(400, "pairing needs a uniqueid");
        };
        let step = |name: &str| query.get(name).map(hex::decode);
        let reply = match query.get("phrase").map(String::as_str) {
            Some("getservercert") => return self.get_server_cert(unique_id, query).await,
            Some("pairchallenge") if self.is_paired(connection) => {
                return Reply::xml("<paired>1</paired>")
            }
            Some("pairchallenge") => return Reply::xml("<paired>0</paired>"),
            Some(other) => {
                return Reply::xml_error(400, &format!("unknown pairing phrase {other}"))
            }
            None => match (
                step("clientchallenge"),
                step("serverchallengeresp"),
                step("clientpairingsecret"),
            ) {
                (Some(Ok(challenge)), _, _) => self
                    .pairings
                    .client_challenge(unique_id, &challenge, &self.identity)
                    .map(|response| {
                        format!(
                            "<challengeresponse>{}</challengeresponse>",
                            hex::encode(response)
                        )
                    }),
                (_, Some(Ok(response)), _) => self
                    .pairings
                    .server_challenge_response(unique_id, &response, &self.identity)
                    .map(|secret| {
                        format!("<pairingsecret>{}</pairingsecret>", hex::encode(secret))
                    }),
                (_, _, Some(Ok(secret))) => self
                    .pairings
                    .client_pairing_secret(unique_id, &secret)
                    .and_then(|certificate| {
                        self.paired
                            .add(&certificate)
                            .map_err(|_| pairing::PairingError::BadClientCertificate)
                    })
                    .map(|()| {
                        eprintln!("g2g-desktop host: paired a Moonlight client");
                        String::new()
                    }),
                _ => return Reply::xml_error(400, "pairing step is missing or not hex"),
            },
        };
        match reply {
            Ok(inner) => Reply::xml(&format!("<paired>1</paired>{inner}")),
            Err(e) => {
                g2g_core::g2g_warn!(Target::category(LOG_CATEGORY), "pairing failed: {e}");
                self.pairings.cancel(unique_id);
                Reply::xml_error(400, &e.to_string())
            }
        }
    }

    async fn get_server_cert(&self, unique_id: &str, query: &HashMap<String, String>) -> Reply {
        let salt = query
            .get("salt")
            .and_then(|salt| hex::decode(salt).ok())
            .and_then(|salt| <[u8; SALT_BYTES]>::try_from(salt).ok());
        let certificate = query
            .get("clientcert")
            .and_then(|pem| hex::decode(pem).ok())
            .and_then(|pem| CertificateDer::from_pem_slice(&pem).ok());
        let (Some(salt), Some(certificate)) = (salt, certificate) else {
            return Reply::xml_error(
                400,
                "getservercert needs a 16-byte salt and a PEM clientcert",
            );
        };
        let device = query.get("devicename").map_or("a device", String::as_str);
        let pin_entered = self.pairings.begin(unique_id, certificate.to_vec(), salt);
        eprintln!(
            "g2g-desktop host: {device} asks to pair, run `g2g-desktop pin <PIN>` with the PIN Moonlight shows"
        );
        let entered = tokio::time::timeout(PIN_WAIT, pin_entered.notified()).await;
        if entered.is_err() || !self.pairings.has_key(unique_id) {
            self.pairings.cancel(unique_id);
            return Reply::xml_error(400, "no PIN was entered");
        }
        Reply::xml(&format!(
            "<paired>1</paired><plaincert>{}</plaincert>",
            hex::encode(&self.identity.certificate_pem)
        ))
    }

    async fn end_session(&self) {
        let session = self.session.borrow_mut().take();
        let Some(mut session) = session else {
            return;
        };
        session.stop.notify_one();
        if tokio::time::timeout(SESSION_STOP_WAIT, &mut session.task)
            .await
            .is_err()
        {
            session.task.abort();
            // the aborted task frees the stream ports only once it is dropped
            let _ = session.task.await;
        }
    }

    async fn launch(
        self: &Rc<Self>,
        query: &HashMap<String, String>,
        connection: Connection<'_>,
        reply_tag: &str,
    ) -> Reply {
        let key = query
            .get("rikey")
            .and_then(|key| hex::decode(key).ok())
            .and_then(|key| <[u8; SESSION_KEY_BYTES]>::try_from(key).ok());
        let Some(key) = key else {
            return Reply::xml_error(400, "launch needs a 16-byte hexadecimal rikey");
        };
        // moonlight prints rikeyid as a signed int, which the IV counter reads as unsigned
        let key_id = query.get("rikeyid").and_then(|id| {
            id.parse::<i32>()
                .map(|id| id as u32)
                .or_else(|_| id.parse::<u32>())
                .ok()
        });
        let Some(key_id) = key_id else {
            return Reply::xml_error(400, "launch needs a 32-bit rikeyid");
        };
        let key = stream::SessionKey { key, key_id };
        self.end_session().await;
        let sockets = match stream::bind().await {
            Ok(sockets) => sockets,
            Err(e) => {
                return Reply::xml_error(503, &format!("could not open the stream ports: {e}"))
            }
        };
        let (play, played) = oneshot::channel();
        let host = Rc::clone(self);
        let client = connection.peer.ip();
        let stop = Rc::new(Notify::new());
        let session_stop = Rc::clone(&stop);
        let task = spawn_local(async move {
            match stream::run(sockets, client, key, played, &session_stop, &host.desktop).await {
                Ok(()) => eprintln!("g2g-desktop host: Moonlight session ended"),
                Err(e) => eprintln!("g2g-desktop host: Moonlight session failed: {e}"),
            }
            host.session.borrow_mut().take();
        });
        *self.session.borrow_mut() = Some(Session {
            client,
            settings: None,
            play: Some(play),
            stop,
            task,
        });
        eprintln!("g2g-desktop host: Moonlight at {client} opened a session");
        Reply::xml(&format!(
            "<{reply_tag}>1</{reply_tag}><sessionUrl0>rtsp://{}:{RTSP_PORT}</sessionUrl0>",
            connection.local.ip()
        ))
    }

    fn answer_rtsp(&self, request: &RtspRequest) -> RtspReply {
        let mut session = self.session.borrow_mut();
        let Some(session) = session.as_mut() else {
            return RtspReply::not_found();
        };
        match request.method.as_str() {
            "OPTIONS" => RtspReply::ok(),
            "DESCRIBE" => RtspReply::description(),
            "SETUP" => match request.target.split('/').next() {
                Some("streamid=video") => RtspReply::setup(VIDEO_PORT),
                Some("streamid=audio") => RtspReply::setup(AUDIO_PORT),
                Some("streamid=control") => RtspReply::setup(control::CONTROL_PORT),
                _ => RtspReply::bad_request(),
            },
            "ANNOUNCE" => match rtsp::parse_announce(&request.body) {
                Ok(settings) => {
                    session.settings = Some(settings);
                    RtspReply::ok()
                }
                Err(e) => {
                    g2g_core::g2g_warn!(Target::category(LOG_CATEGORY), "{e}");
                    RtspReply::bad_request()
                }
            },
            "PLAY" => match (session.settings, session.play.take()) {
                (Some(settings), Some(play)) => {
                    // the receiver is gone only when the session already ended
                    let _ = play.send(settings);
                    RtspReply::ok()
                }
                _ => RtspReply::bad_request(),
            },
            _ => RtspReply::bad_request(),
        }
    }

    fn submit_pin(&self, body: &[u8], connection: Connection<'_>) -> Reply {
        if !connection.peer.ip().is_loopback() {
            return Reply::text(403, "a PIN is accepted from this machine only");
        }
        let form: HashMap<String, String> = form_urlencoded::parse(body).into_owned().collect();
        let Some(pin) = form
            .get("pin")
            .filter(|pin| pin.len() == PIN_DIGITS && pin.bytes().all(|b| b.is_ascii_digit()))
        else {
            return Reply::text(400, "the PIN is four digits");
        };
        match self
            .pairings
            .enter_pin(pin, form.get("uniqueid").map(String::as_str))
        {
            Ok(()) => Reply::text(200, "PIN accepted"),
            Err(e @ pairing::PinError::SeveralWaiting) => Reply::text(409, &e.to_string()),
            Err(e) => Reply::text(404, &e.to_string()),
        }
    }
}

fn app_list() -> Reply {
    Reply::xml(&format!(
        "<App><IsHdrSupported>0</IsHdrSupported><AppTitle>{DESKTOP_APP_TITLE}</AppTitle>\
         <ID>{DESKTOP_APP_ID}</ID></App>"
    ))
}

async fn answer<S: AsyncRead + AsyncWrite + Unpin>(
    host: &Rc<Host>,
    stream: &mut S,
    connection: Connection<'_>,
) -> io::Result<()> {
    let request = read_request(stream).await?;
    let reply = host.route(request, connection).await;
    write_reply(stream, reply).await
}

async fn answer_http(host: Rc<Host>, mut stream: TcpStream) -> io::Result<()> {
    let connection = Connection {
        local: stream.local_addr()?,
        peer: stream.peer_addr()?,
        client_certificate: None,
        https: false,
    };
    answer(&host, &mut stream, connection).await
}

async fn answer_https(host: Rc<Host>, acceptor: TlsAcceptor, stream: TcpStream) -> io::Result<()> {
    let local = stream.local_addr()?;
    let peer = stream.peer_addr()?;
    let mut stream = acceptor.accept(stream).await?;
    let client_certificate = stream
        .get_ref()
        .1
        .peer_certificates()
        .and_then(|chain| chain.first())
        .map(|certificate| certificate.to_vec());
    let connection = Connection {
        local,
        peer,
        client_certificate: client_certificate.as_deref(),
        https: true,
    };
    answer(&host, &mut stream, connection).await
}

async fn answer_rtsp(host: Rc<Host>, mut stream: TcpStream) -> io::Result<()> {
    let peer = stream.peer_addr()?;
    let from_client = host
        .session
        .borrow()
        .as_ref()
        .is_some_and(|session| session.client == peer.ip());
    if !from_client {
        return Ok(());
    }
    let request = rtsp::read_request(&mut stream).await?;
    let reply = host.answer_rtsp(&request);
    rtsp::write_reply(&mut stream, &request, reply).await
}

fn advertise(name: &str) -> Result<ServiceDaemon, GamestreamError> {
    let daemon = ServiceDaemon::new()?;
    let service = ServiceInfo::new(
        SERVICE_TYPE,
        name,
        &format!("{name}.local."),
        (),
        HTTP_PORT,
        None::<HashMap<String, String>>,
    )?
    .enable_addr_auto();
    daemon.register(service)?;
    Ok(daemon)
}

fn log_failure(port: u16, result: io::Result<()>) {
    if let Err(e) = result {
        g2g_core::g2g_warn!(
            Target::category(LOG_CATEGORY),
            "request on port {port} failed: {e}"
        );
    }
}

pub async fn serve(desktop: ServedDesktop) -> Result<(), GamestreamError> {
    let directory = dirs::config_dir()
        .ok_or(GamestreamError::NoConfigDirectory)?
        .join("g2g-desktop")
        .join("gamestream");
    let host = Rc::new(Host::load(&directory, desktop)?);
    let acceptor = TlsAcceptor::from(tls::server_config(&host.identity)?);
    let http = TcpListener::bind((Ipv4Addr::UNSPECIFIED, HTTP_PORT)).await?;
    let https = TcpListener::bind((Ipv4Addr::UNSPECIFIED, HTTPS_PORT)).await?;
    let rtsp = TcpListener::bind((Ipv4Addr::UNSPECIFIED, RTSP_PORT)).await?;
    let _mdns = advertise(&host.name)?;
    eprintln!(
        "g2g-desktop host: Moonlight host {} on TCP ports {HTTP_PORT} and {HTTPS_PORT}",
        host.name
    );
    loop {
        tokio::select! {
            accepted = http.accept() => {
                let (stream, _) = accepted?;
                let task = answer_http(Rc::clone(&host), stream);
                spawn_local(async move { log_failure(HTTP_PORT, task.await) });
            }
            accepted = https.accept() => {
                let (stream, _) = accepted?;
                let task = answer_https(Rc::clone(&host), acceptor.clone(), stream);
                spawn_local(async move { log_failure(HTTPS_PORT, task.await) });
            }
            accepted = rtsp.accept() => {
                let (stream, _) = accepted?;
                let task = answer_rtsp(Rc::clone(&host), stream);
                spawn_local(async move { log_failure(RTSP_PORT, task.await) });
            }
        }
    }
}

pub fn submit_pin(pin: &str) -> Result<(), GamestreamError> {
    let body = form_urlencoded::Serializer::new(String::new())
        .append_pair("pin", pin)
        .finish();
    let mut stream =
        std::net::TcpStream::connect(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), HTTP_PORT))?;
    write!(
        stream,
        "POST /submit-pin HTTP/1.1\r\nHost: localhost\r\n\
         Content-Type: application/x-www-form-urlencoded\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    )?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    let (head, message) = response.split_once("\r\n\r\n").unwrap_or((&response, ""));
    match head.starts_with("HTTP/1.1 200") {
        true => Ok(()),
        false => Err(GamestreamError::PinRefused(message.trim().to_string())),
    }
}
