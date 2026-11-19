// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;

use g2g_core::runtime::{GraphNode, PadKind, PadRequest, ParseError, PlaybinSinks, Registry};
use g2g_core::{ByteStreamEncoding, Caps, Graph, StreamType, TagList};

use super::{
    build_av_fanout, byte_source, cc_request, mkv_playbin_tracks, mp4_playbin_tracks,
    ps_playbin_tracks, ts_playbin_tracks,
};

/// Which track of a media kind to play. Track indices start at zero within
/// each kind and use the same numbering as the track catalog.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PlaybinTrackChoice {
    #[default]
    Auto,
    Disabled,
    Track(usize),
}

/// Track choices for a file-backed playbin. Auto takes the first track of each
/// kind, with embedded subtitles shown when a video track is selected.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PlaybinTrackSelection {
    pub video: PlaybinTrackChoice,
    pub audio: PlaybinTrackChoice,
    pub subtitle: PlaybinTrackChoice,
}

/// A track in the file's playback catalog.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaybinTrack {
    pub kind: StreamType,
    pub index: usize,
    pub caps: Caps,
    pub tags: TagList,
    pub selected: bool,
}

pub(super) struct TrackPicker {
    choice: PlaybinTrackSelection,
    tracks: Vec<PlaybinTrack>,
}

impl TrackPicker {
    fn choice(&self, kind: StreamType) -> PlaybinTrackChoice {
        match kind {
            StreamType::Video => self.choice.video,
            StreamType::Audio => self.choice.audio,
            StreamType::Text => self.choice.subtitle,
            _ => PlaybinTrackChoice::Disabled,
        }
    }

    pub(super) fn push(&mut self, caps: &Caps, tags: TagList, eligible: bool) -> bool {
        let kind = StreamType::of(caps);
        let index = self
            .tracks
            .iter()
            .filter(|track| track.kind == kind)
            .count();
        let wanted = match self.choice(kind) {
            PlaybinTrackChoice::Auto => index == 0,
            PlaybinTrackChoice::Disabled => false,
            PlaybinTrackChoice::Track(wanted) => index == wanted,
        };
        let selected = eligible && wanted && (kind != StreamType::Text || self.has_video());
        self.tracks.push(PlaybinTrack {
            kind,
            index,
            caps: caps.clone(),
            tags,
            selected,
        });
        selected
    }

    pub(super) fn has_video(&self) -> bool {
        self.tracks
            .iter()
            .any(|track| track.selected && track.kind == StreamType::Video)
    }

    pub(super) fn uses_uri_captions(&self) -> bool {
        self.choice.subtitle == PlaybinTrackChoice::Auto
    }

    pub(super) fn validate(&self, source: &str) -> Result<(), ParseError> {
        for kind in [StreamType::Video, StreamType::Audio, StreamType::Text] {
            if let PlaybinTrackChoice::Track(index) = self.choice(kind) {
                if !self
                    .tracks
                    .iter()
                    .any(|track| track.kind == kind && track.index == index && track.selected)
                {
                    return Err(ParseError::UnknownOutputPad {
                        element: source.to_string(),
                        pad: format!("{}_{index}", kind.name()),
                    });
                }
            }
        }
        if self.tracks.iter().any(|track| track.selected) {
            Ok(())
        } else {
            Err(ParseError::NoStreamDecoder {
                location: source.to_string(),
                codecs: Vec::new(),
            })
        }
    }
}

pub(super) fn matroska_tags(demux: &crate::matroska::MatroskaDemuxer, number: u64) -> TagList {
    let find = |tags: &[(u64, TagList)]| {
        tags.iter()
            .find(|(track, _)| *track == number)
            .map(|(_, tags)| tags.clone())
            .unwrap_or_default()
    };
    g2g_core::resolve_tags(&find(demux.track_entry_tags()), &find(demux.track_tags()))
}

/// Build a local container's playbin with one selected track per kind and
/// return its catalog. A source without a selectable file demuxer returns
/// `None` so the caller can use the ordinary playbin path.
pub fn file_track_playbin(
    reg: &Registry,
    sinks: &PlaybinSinks<'_>,
    uri: &str,
    selection: PlaybinTrackSelection,
) -> Result<Option<(Graph<GraphNode>, Vec<PlaybinTrack>)>, ParseError> {
    let (base_uri, _) = cc_request(uri);
    let Some(path) = super::file_uri_path(&base_uri) else {
        return Ok(None);
    };
    let Some(caps) = super::read_sniff_window(path)
        .as_deref()
        .and_then(crate::typefind::sniff_caps)
    else {
        return Ok(None);
    };
    let Caps::ByteStream { encoding } = caps else {
        return Ok(None);
    };
    let mut picker = TrackPicker {
        choice: selection,
        tracks: Vec::new(),
    };
    let graph = match encoding {
        ByteStreamEncoding::Matroska => mkv_playbin_tracks(reg, sinks, uri, Some(&mut picker))?,
        ByteStreamEncoding::Mp4 | ByteStreamEncoding::IsoBmff => {
            mp4_playbin_tracks(reg, sinks, uri, Some(&mut picker))?
        }
        ByteStreamEncoding::MpegTs => ts_playbin_tracks(reg, sinks, uri, Some(&mut picker))?,
        ByteStreamEncoding::MpegPs => ps_playbin_tracks(reg, sinks, uri, Some(&mut picker))?,
        _ => select_demux_ports(reg, sinks, path, encoding, &mut picker)?,
    };
    Ok(graph.map(|graph| (graph, picker.tracks)))
}

fn select_demux_ports(
    reg: &Registry,
    sinks: &PlaybinSinks<'_>,
    path: &str,
    encoding: ByteStreamEncoding,
    picker: &mut TrackPicker,
) -> Result<Option<Graph<GraphNode>>, ParseError> {
    let caps = Caps::ByteStream { encoding };
    let Some((all_demux, mut ports)) = reg
        .decodebin_select_hooks()
        .iter()
        .find_map(|hook| hook(path, &caps))
    else {
        return Ok(None);
    };
    let all_ports = ports.len();
    let mut pads = Vec::new();
    ports.retain(|port| {
        if !picker.push(&port.caps, TagList::new(), true) {
            return false;
        }
        let track = picker.tracks.last().expect("just added");
        let kind = match track.kind {
            StreamType::Video => PadKind::Video,
            StreamType::Audio => PadKind::Audio,
            _ => PadKind::Text,
        };
        pads.push(PadRequest {
            kind,
            index: track.index,
        });
        true
    });
    picker.validate(path)?;
    let demux = if ports.len() == all_ports {
        all_demux
    } else {
        reg.container_streams(path, &caps)
            .and_then(|streams| streams.streams.first().map(|stream| stream.demux))
            .and_then(|name| {
                reg.demux_select_hooks()
                    .iter()
                    .find_map(|hook| hook(name, path, &pads))
            })
            .ok_or_else(|| ParseError::NoStreamDecoder {
                location: path.to_string(),
                codecs: Vec::new(),
            })?
    };
    build_av_fanout(reg, sinks, byte_source(path, encoding), demux, &ports).map(Some)
}
