// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::path::Path;

use g2g_plugins::subparse::Cue;
use g2g_plugins::textoverlay::{read_cue_file, TextOverlay};

// mpv's two subtitle tracks, sid drawn at the bottom and secondary-sid at the top
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleSlot {
    Subtitle,
    Caption,
}

const SLOTS: [SubtitleSlot; 2] = [SubtitleSlot::Subtitle, SubtitleSlot::Caption];
const TOP_LINE_PERCENT: u8 = 0;

impl SubtitleSlot {
    fn index(self) -> usize {
        match self {
            SubtitleSlot::Subtitle => 0,
            SubtitleSlot::Caption => 1,
        }
    }

    fn place(self, mut cues: Vec<Cue>) -> Vec<Cue> {
        if self == SubtitleSlot::Caption {
            for cue in cues.iter_mut().filter(|cue| cue.settings.line.is_none()) {
                cue.settings.line = Some(TOP_LINE_PERCENT);
            }
        }
        cues
    }
}

#[derive(Debug, Default)]
pub(crate) struct Subtitles {
    tracks: [Option<TextOverlay>; 2],
    hidden: [bool; 2],
    // changes with every file or visibility change, so render contexts draw the cues again
    generation: u64,
}

// the cues each slot shows at one time, with the subtitle settings they were picked under
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShowingCues {
    generation: u64,
    cues: [Vec<usize>; 2],
}

impl ShowingCues {
    pub(crate) fn is_empty(&self) -> bool {
        self.cues.iter().all(Vec::is_empty)
    }
}

impl Subtitles {
    pub(crate) fn load(&mut self, slot: SubtitleSlot, path: Option<&Path>) -> std::io::Result<()> {
        let track = match path {
            Some(path) => {
                let cues = read_cue_file(&path.to_string_lossy())?;
                Some(TextOverlay::new().with_cues(slot.place(cues)))
            }
            None => None,
        };
        self.tracks[slot.index()] = track;
        self.generation += 1;
        Ok(())
    }

    pub(crate) fn drop_files(&mut self) {
        self.tracks = Default::default();
        self.generation += 1;
    }

    pub(crate) fn set_visible(&mut self, slot: SubtitleSlot, visible: bool) {
        self.hidden[slot.index()] = !visible;
        self.generation += 1;
    }

    fn shown_track(&mut self, slot: SubtitleSlot) -> Option<&mut TextOverlay> {
        if self.hidden[slot.index()] {
            return None;
        }
        self.tracks[slot.index()].as_mut()
    }

    pub(crate) fn showing_at(&mut self, pts_ns: u64) -> ShowingCues {
        let cues = SLOTS.map(|slot| {
            self.shown_track(slot)
                .map(|track| track.showing_at(pts_ns))
                .unwrap_or_default()
        });
        ShowingCues {
            generation: self.generation,
            cues,
        }
    }

    pub(crate) fn render(&mut self, rgba: &mut [u8], size: (u32, u32), pts_ns: u64) {
        for slot in SLOTS {
            if let Some(track) = self.shown_track(slot) {
                track.render_onto(rgba, size, pts_ns);
            }
        }
    }
}
