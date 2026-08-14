//! Multi-track MoQ Transport subscriber (`moqtsessionsrc`, M914, `moqt`
//! feature): subscribes to several tracks of one broadcast over a single MoQT
//! session and emits each on its own output pad.
//!
//! ```text
//! moqtsessionsrc name=s location=https://relay:4443/ namespace=live/cam tracks=1.m4s,2.m4s
//!   s. ! fmp4demux ! ...
//!   s. ! fmp4demux ! ...
//! ```
//!
//! The multi-track counterpart of [`MoqtSrc`](crate::moqtsrc::MoqtSrc), which
//! keeps the one-pad surface. Both sit on the same session driver
//! ([`subscriber`](crate::moqt::subscriber)), so a track behaves the same
//! whichever element asked for it: catalog discovery, the joining FETCH
//! catch-up, and the publisher-initiated PUBLISH path all apply here too.
//!
//! Pads are filled from the `tracks` property, in order, and from the catalog's
//! track order for any pad it does not name. Every pad's stream opens with the
//! broadcast's init segment: an fMP4 init segment describes every track of the
//! broadcast, and the publisher carries one for all of them, so each demuxer
//! downstream sees a whole fMP4 stream of its own track's fragments.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::log::LogSource;
use g2g_core::{
    ByteStreamEncoding, Caps, G2gError, MultiOutputSink, MultiOutputSource, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec,
};

use crate::moqt::parse_versions;
use crate::moqt::subscriber::{byte_frame, connect, session_err, Pumped, SubscriberConfig};

/// Subscribes to several tracks of one MoQ Transport broadcast and emits each
/// on its own output pad.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::moqtsessionsrc::MoqtSessionSrc;
///
/// let source = MoqtSessionSrc::new("https://127.0.0.1:4443/", "g2g")
///     .with_outputs(2)
///     .with_tracks("video,audio");
/// ```
#[derive(Debug)]
pub struct MoqtSessionSrc {
    cfg: SubscriberConfig,
    /// Track names, comma-separated, one per output pad in order. Empty names
    /// take the catalog's tracks in order.
    tracks: String,
    outputs: usize,
    /// `u64::MAX` runs until the broadcast ends; otherwise stop after this many
    /// frames across all pads and emit EOS.
    num_buffers: u64,

    /// The track each pad played, once the run has started.
    selected: Vec<String>,
    objects_received: u64,
}

impl Default for MoqtSessionSrc {
    fn default() -> Self {
        Self::new("https://127.0.0.1:4443/", "g2g")
    }
}

impl MoqtSessionSrc {
    /// Subscribe to `namespace` (a `/`-separated path) on the relay at
    /// `location`.
    pub fn new(location: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            cfg: SubscriberConfig {
                location: location.into(),
                namespace: namespace.into(),
                ..SubscriberConfig::default()
            },
            tracks: String::new(),
            outputs: 0,
            num_buffers: u64::MAX,
            selected: Vec::new(),
            objects_received: 0,
        }
    }

    /// Number of output pads. The launch path passes the branch count; a caller
    /// building the element itself says how many tracks it wants.
    pub fn with_outputs(mut self, outputs: usize) -> Self {
        self.outputs = outputs;
        self
    }

    /// Name the track on each pad, in order (`"video.m4s,audio.m4s"`).
    pub fn with_tracks(mut self, tracks: impl Into<String>) -> Self {
        self.tracks = tracks.into();
        self.cfg.wanted_tracks = self.named_tracks();
        self
    }

    /// Accept only relay certificates whose SHA-256 digest is listed (hex,
    /// comma-separated) instead of requiring a system root.
    pub fn with_server_certificate_hashes(mut self, hashes: impl Into<String>) -> Self {
        self.cfg.cert_hashes = hashes.into();
        self
    }

    /// Stop after this many frames across all pads (init segments included).
    /// 0 emits EOS on every pad without subscribing.
    pub fn with_num_buffers(mut self, n: u64) -> Self {
        self.num_buffers = n;
        self
    }

    /// The track each pad played, in pad order.
    pub fn selected_tracks(&self) -> &[String] {
        &self.selected
    }

    /// Frames the last run handed downstream, init segments included.
    pub fn objects_received(&self) -> u64 {
        self.objects_received
    }

    fn named_tracks(&self) -> Vec<String> {
        self.tracks
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(String::from)
            .collect()
    }

    fn output_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        }
    }
}

impl LogSource for MoqtSessionSrc {
    fn log_category(&self) -> &'static str {
        "moqtsessionsrc"
    }
}

impl MultiOutputSource for MoqtSessionSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn output_count(&self) -> usize {
        self.outputs.max(self.named_tracks().len()).max(1)
    }

    fn output_caps(&self, output: usize) -> Result<Caps, G2gError> {
        if output >= self.output_count() {
            return Err(G2gError::CapsMismatch);
        }
        Ok(Self::output_caps())
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let pads = self.output_count();
            if crate::numbuffers::finished_at_zero_limit_multi(self.num_buffers, pads, out).await? {
                return Ok(0);
            }
            let mut driver = connect(&self.cfg).await?;
            let listed = driver.read_catalog(&self.cfg).await?;

            // The `tracks` property names the pads it covers; the rest take the
            // catalog's tracks in order, skipping the ones already named.
            let mut names = self.named_tracks();
            names.truncate(pads);
            for track in &listed {
                if names.len() >= pads {
                    break;
                }
                if !names.contains(&track.name) {
                    names.push(track.name.clone());
                }
            }
            self.selected = names.clone();

            // Every track of a broadcast shares one init segment, so the first
            // named track's catalog entry decides where it comes from.
            let init_track = listed
                .iter()
                .find(|t| names.first().is_some_and(|name| *name == t.name))
                .map(|t| t.init_track.clone())
                .filter(|init| !init.is_empty())
                .unwrap_or_else(|| self.cfg.init_track.clone());
            let Some(init) = driver.read_init(&init_track).await? else {
                driver.shutdown().await;
                return Err(session_err());
            };

            let mut subs = Vec::new();
            for name in &names {
                subs.push(
                    driver
                        .subscribe_media(name, self.cfg.catchup_groups)
                        .await?,
                );
            }

            let limit = self.num_buffers;
            let mut emitted = 0u64;
            for port in 0..pads {
                out.push_to(port, PipelinePacket::CapsChanged(Self::output_caps()))
                    .await?;
                // A pad with no track, or one the limit already ran out on,
                // still gets its caps and its EOS below, so the branch is never
                // left hanging.
                if port < subs.len() && emitted < limit {
                    out.push_to(port, PipelinePacket::DataFrame(byte_frame(init.clone(), 0)))
                        .await?;
                    emitted += 1;
                }
            }

            'run: loop {
                let mut moved = false;
                for (port, at) in subs.iter().copied().enumerate() {
                    while emitted < limit {
                        let Some(payload) = driver.state().subs[at].next_payload() else {
                            break;
                        };
                        out.push_to(
                            port,
                            PipelinePacket::DataFrame(byte_frame(payload, emitted)),
                        )
                        .await?;
                        emitted += 1;
                        moved = true;
                    }
                    if emitted >= limit {
                        break 'run;
                    }
                }
                if subs.iter().all(|at| driver.state().subs[*at].drained()) {
                    break;
                }
                if !moved {
                    match driver.pump().await? {
                        Pumped::Applied => {}
                        // A silent relay is the end of the broadcast.
                        Pumped::Ended | Pumped::TimedOut => break,
                    }
                }
            }

            driver.shutdown().await;
            self.objects_received = emitted;
            for port in 0..pads {
                out.push_to(port, PipelinePacket::Eos).await?;
            }
            Ok(emitted)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        MOQTSESSIONSRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let string = |v: &PropValue| v.as_str().map(ToString::to_string).ok_or(PropError::Type);
        let uint = |v: &PropValue| v.as_uint().ok_or(PropError::Type);
        match name {
            "location" => self.cfg.location = string(&value)?,
            "namespace" => self.cfg.namespace = string(&value)?,
            "tracks" => {
                self.tracks = string(&value)?;
                // The named tracks are also the ones a publisher-initiated
                // PUBLISH may establish here.
                self.cfg.wanted_tracks = self.named_tracks();
            }
            "init-track-name" => self.cfg.init_track = string(&value)?,
            "catalog-track-name" => self.cfg.catalog_track = string(&value)?,
            "server-certificate-hashes" => self.cfg.cert_hashes = string(&value)?,
            "catalog" => self.cfg.use_catalog = value.as_bool().ok_or(PropError::Type)?,
            "max-request-id" => self.cfg.max_request_id = uint(&value)?,
            "versions" => {
                let list = string(&value)?;
                parse_versions(&list).map_err(|_| PropError::Value)?;
                self.cfg.versions = list;
            }
            "max-groups" => self.cfg.max_groups = uint(&value)?,
            "max-buffer-bytes" => self.cfg.max_buffer_bytes = uint(&value)?,
            "max-object-size" => self.cfg.max_object_bytes = uint(&value)?,
            "catchup-groups" => self.cfg.catchup_groups = uint(&value)?,
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.num_buffers, &value)?,
            "timeout" => self.cfg.timeout_ms = uint(&value)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.cfg.location.clone())),
            "namespace" => Some(PropValue::Str(self.cfg.namespace.clone())),
            "tracks" => Some(PropValue::Str(self.tracks.clone())),
            "init-track-name" => Some(PropValue::Str(self.cfg.init_track.clone())),
            "catalog-track-name" => Some(PropValue::Str(self.cfg.catalog_track.clone())),
            "server-certificate-hashes" => Some(PropValue::Str(self.cfg.cert_hashes.clone())),
            "catalog" => Some(PropValue::Bool(self.cfg.use_catalog)),
            "max-request-id" => Some(PropValue::Uint(self.cfg.max_request_id)),
            "versions" => Some(PropValue::Str(self.cfg.versions.clone())),
            "max-groups" => Some(PropValue::Uint(self.cfg.max_groups)),
            "max-buffer-bytes" => Some(PropValue::Uint(self.cfg.max_buffer_bytes)),
            "max-object-size" => Some(PropValue::Uint(self.cfg.max_object_bytes)),
            "catchup-groups" => Some(PropValue::Uint(self.cfg.catchup_groups)),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.num_buffers)),
            "timeout" => Some(PropValue::Uint(self.cfg.timeout_ms)),
            _ => None,
        }
    }
}

static MOQTSESSIONSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "WebTransport URL of the MoQT relay (e.g. https://host:4443/)",
    )
    .with_default("https://127.0.0.1:4443/"),
    PropertySpec::new(
        "namespace",
        PropKind::Str,
        "broadcast namespace to subscribe to, a /-separated path",
    )
    .with_default("g2g"),
    PropertySpec::new(
        "tracks",
        PropKind::Str,
        "media tracks to play, comma-separated, one per output pad in order; pads it does not name take the catalog's tracks in order",
    ),
    PropertySpec::new(
        "init-track-name",
        PropKind::Str,
        "track carrying the ftyp+moov init segment, when the catalog names none",
    )
    .with_default("0.mp4"),
    PropertySpec::new(
        "catalog-track-name",
        PropKind::Str,
        "track carrying the JSON catalog",
    )
    .with_default(".catalog"),
    PropertySpec::new(
        "catalog",
        PropKind::Bool,
        "read the catalog track to discover the tracks",
    )
    .with_default("true"),
    PropertySpec::new(
        "max-request-id",
        PropKind::Uint,
        "MAX_REQUEST_ID advertised to the relay in CLIENT_SETUP (draft-16 sessions only)",
    )
    .with_default("100"),
    PropertySpec::new(
        "versions",
        PropKind::Str,
        "MoQ Transport draft versions offered on CONNECT, comma-separated in preference order; the server's pick decides",
    )
    .with_default("18,16"),
    PropertySpec::new(
        "max-groups",
        PropKind::Uint,
        "groups held per track while reordering; the oldest is dropped past this",
    )
    .with_default("8"),
    PropertySpec::new(
        "max-buffer-bytes",
        PropKind::Uint,
        "bytes held per track while reordering; the oldest group is dropped past this",
    )
    .with_default("33554432"),
    PropertySpec::new(
        "max-object-size",
        PropKind::Uint,
        "largest single object accepted from the relay, in bytes",
    )
    .with_default("16777216"),
    PropertySpec::new(
        "catchup-groups",
        PropKind::Uint,
        "groups before the live edge to FETCH on join and emit before the live objects (0 = start live)",
    )
    .with_default("0"),
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "frames to emit across all pads then EOS, init segments included (-1 = until the broadcast ends)",
    )
    .with_default("-1")
    .with_range("-1", "9223372036854775807"),
    PropertySpec::new(
        "timeout",
        PropKind::Uint,
        "give up if nothing arrives for this many ms; before the first frame that fails the run, after it ends the stream (0 = wait forever)",
    )
    .with_default("15000"),
    PropertySpec::new(
        "server-certificate-hashes",
        PropKind::Str,
        "accept only relay certificates with these SHA-256 digests (hex, comma-separated); empty = system roots",
    ),
];
