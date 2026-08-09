//! MoQ Transport subscriber source (`moqtsrc`, M903, `moqt` feature): subscribes
//! to a broadcast on an IETF MoQT relay over WebTransport and emits the
//! fragmented-MP4 byte stream it carries.
//!
//! ```text
//! moqtsrc location=https://relay:4443/ namespace=live/cam ! fmp4demux ! ...
//! ```
//!
//! It is the inverse of [`MoqtSink`](crate::moqtsink), and plays either side's
//! broadcast: Cloudflare's `moq-pub` and `moqtsink` publish the same track
//! layout.
//!
//! - the `.catalog` track names the media tracks and the init track. Without it
//!   (`catalog=false`, or a publisher that omits it) the defaults are the
//!   reference ones: `0.mp4` for the init track, `{track_id}.m4s` for media.
//! - the init track's single object is the `ftyp`+`moov`, emitted first so the
//!   demuxer downstream sees a whole fMP4 stream.
//! - each media object is one `moof`+`mdat` fragment, emitted in group and
//!   object order (see [`reassembly`](crate::moqt::reassembly) for the ordering
//!   policy and its bounds).
//!
//! `catchup-groups` asks for that many groups before the live edge with a
//! joining FETCH and emits them ahead of the live objects, so playback starts
//! with a buffer rather than at the edge. A track the publisher offers with
//! PUBLISH, rather than waiting to be asked, establishes the same subscription
//! (§9.13); one for a track this run does not want is refused.
//!
//! The session machinery is shared with the multi-track
//! [`MoqtSessionSrc`](crate::moqtsessionsrc::MoqtSessionSrc); see
//! [`subscriber`](crate::moqt::subscriber).
//!
//! The stream ends on the publisher's PUBLISH_DONE for the media subscription,
//! on the session closing, or on `num-buffers` / `timeout`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::log::LogSource;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::moqt::catalog::CatalogTrack;
use crate::moqt::parse_versions;
use crate::moqt::subscriber::{
    byte_frame, connect, select_track, session_err, Pumped, SubscriberConfig,
};

/// Subscribes to a MoQ Transport broadcast and emits its fMP4 byte stream.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::moqtsrc::MoqtSrc;
///
/// let src = MoqtSrc::new("https://127.0.0.1:4443/", "g2g")
///     .with_track_name("video")
///     .with_catchup_groups(2);
/// ```
#[derive(Debug)]
pub struct MoqtSrc {
    cfg: SubscriberConfig,
    /// The media track to play; empty takes the catalog's first.
    track_name: String,
    num_buffers: u64,

    configured: bool,
    /// Media track the catalog (or the fallback) selected, for tests and logs.
    selected_track: String,
    objects_received: u64,
    catchup_objects: u64,
    groups_dropped: u64,
    objects_dropped: u64,
}

impl Default for MoqtSrc {
    fn default() -> Self {
        Self::new("https://127.0.0.1:4443/", "g2g")
    }
}

impl MoqtSrc {
    /// Subscribe to `namespace` (a `/`-separated path) on the relay at
    /// `location`.
    pub fn new(location: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            cfg: SubscriberConfig {
                location: location.into(),
                namespace: namespace.into(),
                ..SubscriberConfig::default()
            },
            track_name: String::new(),
            num_buffers: 0,
            configured: false,
            selected_track: String::new(),
            objects_received: 0,
            catchup_objects: 0,
            groups_dropped: 0,
            objects_dropped: 0,
        }
    }

    /// Accept only relay certificates whose SHA-256 digest is listed (hex,
    /// comma-separated) instead of requiring a system root.
    pub fn with_server_certificate_hashes(mut self, hashes: impl Into<String>) -> Self {
        self.cfg.cert_hashes = hashes.into();
        self
    }

    /// Subscribe to this media track by name instead of the catalog's first.
    pub fn with_track_name(mut self, name: impl Into<String>) -> Self {
        self.track_name = name.into();
        self.cfg.wanted_tracks = Vec::from([self.track_name.clone()]);
        self
    }

    /// Ask the publisher for this many groups before the live edge with a
    /// joining FETCH, and emit them before the live objects. Zero (the default)
    /// starts at the live edge.
    pub fn with_catchup_groups(mut self, groups: u64) -> Self {
        self.cfg.catchup_groups = groups;
        self
    }

    /// Objects a catch-up FETCH delivered on the last run.
    pub fn catchup_objects(&self) -> u64 {
        self.catchup_objects
    }

    /// Stop after `n` frames (the init segment counts as one).
    pub fn with_num_buffers(mut self, n: u64) -> Self {
        self.num_buffers = n;
        self
    }

    /// The media track actually subscribed to, once the stream has started.
    pub fn selected_track(&self) -> &str {
        &self.selected_track
    }

    /// Frames the last run handed downstream, init segment included.
    pub fn objects_received(&self) -> u64 {
        self.objects_received
    }

    /// Groups the last run's ordering policy abandoned: a mid-group join, or a
    /// group that never completed inside the buffering bounds.
    pub fn groups_dropped(&self) -> u64 {
        self.groups_dropped
    }

    /// Objects the last run threw away: lost (a hole in a group that ended,
    /// which is what a dropped datagram leaves), late, or duplicated.
    pub fn objects_dropped(&self) -> u64 {
        self.objects_dropped
    }

    fn output_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        }
    }
}

impl LogSource for MoqtSrc {
    fn log_category(&self) -> &'static str {
        "moqtsrc"
    }
}

impl SourceLoop for MoqtSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(Self::output_caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(
            Self::output_caps(),
        ))))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if *absolute_caps != Self::output_caps() {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let mut driver = connect(&self.cfg).await?;

            // The catalog names the tracks. Without it, fall back to the
            // reference layout, which is also what `moq-sub` does.
            let listed = driver.read_catalog(&self.cfg).await?;
            let selected = select_track(&self.track_name, &listed).unwrap_or(CatalogTrack {
                name: String::from("1.m4s"),
                init_track: String::new(),
            });
            let init_track = if selected.init_track.is_empty() {
                self.cfg.init_track.clone()
            } else {
                selected.init_track.clone()
            };

            let Some(init) = driver.read_init(&init_track).await? else {
                driver.shutdown().await;
                return Err(session_err());
            };
            let media = driver
                .subscribe_media(&selected.name, self.cfg.catchup_groups)
                .await?;

            out.push(PipelinePacket::CapsChanged(Self::output_caps()))
                .await?;
            out.push(PipelinePacket::DataFrame(byte_frame(init, 0)))
                .await?;
            let mut emitted = 1u64;

            let limit = self.num_buffers;
            loop {
                while let Some(payload) = driver.state().subs[media].next_payload() {
                    out.push(PipelinePacket::DataFrame(byte_frame(payload, emitted)))
                        .await?;
                    emitted += 1;
                    if limit != 0 && emitted >= limit {
                        break;
                    }
                }
                if (limit != 0 && emitted >= limit) || driver.state().subs[media].drained() {
                    break;
                }
                match driver.pump().await? {
                    Pumped::Applied => {}
                    // A silent relay is the end of the stream: the publisher is
                    // gone, and this is what bounds a live subscription that
                    // nothing else stops.
                    Pumped::Ended | Pumped::TimedOut => break,
                }
            }

            let stats = driver.state().subs[media].reassembler.stats();
            self.catchup_objects = driver.state().subs[media].fetch_objects;
            driver.shutdown().await;
            self.selected_track = selected.name;
            self.objects_received = emitted;
            self.groups_dropped = stats.groups_dropped;
            self.objects_dropped = stats.objects_dropped;
            out.push(PipelinePacket::Eos).await?;
            Ok(emitted)
        })
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "MoQ Transport source",
            "Source/Network",
            "Subscribes to an IETF MoQ Transport broadcast over WebTransport and emits its fragmented-MP4 stream",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        MOQTSRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let string = |v: &PropValue| v.as_str().map(ToString::to_string).ok_or(PropError::Type);
        let uint = |v: &PropValue| v.as_uint().ok_or(PropError::Type);
        match name {
            "location" => self.cfg.location = string(&value)?,
            "namespace" => self.cfg.namespace = string(&value)?,
            "track-name" => {
                self.track_name = string(&value)?;
                // The named track is also the only one a publisher-initiated
                // PUBLISH may establish here.
                self.cfg.wanted_tracks = if self.track_name.is_empty() {
                    Vec::new()
                } else {
                    Vec::from([self.track_name.clone()])
                };
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
            "num-buffers" => self.num_buffers = uint(&value)?,
            "timeout" => self.cfg.timeout_ms = uint(&value)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.cfg.location.clone())),
            "namespace" => Some(PropValue::Str(self.cfg.namespace.clone())),
            "track-name" => Some(PropValue::Str(self.track_name.clone())),
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
            "num-buffers" => Some(PropValue::Uint(self.num_buffers)),
            "timeout" => Some(PropValue::Uint(self.cfg.timeout_ms)),
            _ => None,
        }
    }
}

static MOQTSRC_PROPS: &[PropertySpec] = &[
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
        "track-name",
        PropKind::Str,
        "media track to play; empty takes the catalog's first track",
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
        "groups held while reordering; the oldest is dropped past this",
    )
    .with_default("8"),
    PropertySpec::new(
        "max-buffer-bytes",
        PropKind::Uint,
        "bytes held while reordering; the oldest group is dropped past this",
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
        PropKind::Uint,
        "stop after this many frames, init segment included (0 = unlimited)",
    )
    .with_default("0"),
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

impl PadTemplates for MoqtSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(MoqtSrc::output_caps()))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_selection_prefers_the_property_then_the_catalog() {
        let listed = Vec::from([
            CatalogTrack {
                name: String::from("1.m4s"),
                init_track: String::from("0.mp4"),
            },
            CatalogTrack {
                name: String::from("2.m4s"),
                init_track: String::from("0.mp4"),
            },
        ]);
        assert_eq!(
            select_track("", &listed).map(|t| t.name).as_deref(),
            Some("1.m4s")
        );
        assert_eq!(
            select_track("2.m4s", &listed)
                .map(|t| t.init_track)
                .as_deref(),
            Some("0.mp4"),
            "a named track keeps the catalog's init track"
        );
        // A name the catalog does not list is still subscribed to: the
        // publisher may serve a track it never advertised.
        assert_eq!(
            select_track("9.m4s", &listed).map(|t| t.name).as_deref(),
            Some("9.m4s")
        );
        assert!(select_track("", &[]).is_none());
    }
}
