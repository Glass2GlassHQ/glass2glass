// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! AMR parser element (`amrparse`): a single-channel AMR-NB or AMR-WB storage
//! file (RFC 4867 section 5, `#!AMR\n` or `#!AMR-WB\n` then frames) in,
//! one frame per buffer out, as gst's `amrparse` does.
//!
//! AMR travels as the `AudioFormat::Tagged` caps the MP4 demuxers give an
//! `samr` / `sawb` track, so the libavcodec decoder that opens on that track's
//! sample entry opens on the one this element sends ahead of the first frame.
//! Input that does not open on the storage magic, such as an MP4 track already
//! one sample per buffer behind its sample entry, passes through unchanged.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{ElementSeekRoute, SeekController};
use g2g_core::time::ticks_to_ns;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CodecTag, ConfigureOutcome, ElementMetadata,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

use crate::audioframe::{locate_frame, Located, SyncFrameHeader};
use crate::compressed_audio_passthrough::push_admitted;
use crate::demuxseek::DemuxSeek;
use crate::frame_clock::FrameClock;
use crate::typefind::{AMR_NB_MAGIC, AMR_NB_TAG, AMR_WB_MAGIC, AMR_WB_TAG};

// libavformat's amr.c sizes, the TOC byte included, indexed by the TOC's frame type
const NARROWBAND_FRAME_LENS: [u8; 16] = [13, 14, 16, 18, 20, 21, 27, 32, 6, 1, 1, 1, 1, 1, 1, 1];
const WIDEBAND_FRAME_LENS: [u8; 16] = [18, 24, 33, 37, 41, 47, 51, 59, 61, 6, 1, 1, 1, 1, 1, 1];

// a storage file frame's TOC has its follow-on bit and two padding bits clear, gst's sync test
const TOC_RESERVED_BITS: u8 = 0x83;

// every AMR frame is 20 ms
const NARROWBAND_SAMPLES_PER_FRAME: u64 = 160;
const WIDEBAND_SAMPLES_PER_FRAME: u64 = 320;
const NARROWBAND_RATE_HZ: u32 = 8_000;
const WIDEBAND_RATE_HZ: u32 = 16_000;

fn frame_len(toc: u8, lens: &[u8; 16]) -> Option<usize> {
    (toc & TOC_RESERVED_BITS == 0).then(|| usize::from(lens[usize::from(toc >> 3 & 0x0F)]))
}

#[derive(Debug)]
struct NarrowbandToc(usize);

impl SyncFrameHeader for NarrowbandToc {
    const HEADER_LEN: usize = 1;

    fn parse(buf: &[u8]) -> Option<Self> {
        frame_len(*buf.first()?, &NARROWBAND_FRAME_LENS).map(Self)
    }

    fn coded_bytes(&self) -> usize {
        self.0
    }

    fn confirms_sync(&self, _next: &Self) -> bool {
        true
    }
}

#[derive(Debug)]
struct WidebandToc(usize);

impl SyncFrameHeader for WidebandToc {
    const HEADER_LEN: usize = 1;

    fn parse(buf: &[u8]) -> Option<Self> {
        frame_len(*buf.first()?, &WIDEBAND_FRAME_LENS).map(Self)
    }

    fn coded_bytes(&self) -> usize {
        self.0
    }

    fn confirms_sync(&self, _next: &Self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AmrBand {
    Narrow,
    Wide,
}

impl AmrBand {
    fn of(caps: &Caps) -> Option<AmrBand> {
        match caps {
            Caps::Audio {
                format: AudioFormat::Tagged(tag),
                ..
            } if *tag == AMR_NB_TAG => Some(AmrBand::Narrow),
            Caps::Audio {
                format: AudioFormat::Tagged(tag),
                ..
            } if *tag == AMR_WB_TAG => Some(AmrBand::Wide),
            _ => None,
        }
    }

    fn tag(self) -> CodecTag {
        match self {
            AmrBand::Narrow => AMR_NB_TAG,
            AmrBand::Wide => AMR_WB_TAG,
        }
    }

    fn magic(self) -> &'static [u8] {
        match self {
            AmrBand::Narrow => AMR_NB_MAGIC,
            AmrBand::Wide => AMR_WB_MAGIC,
        }
    }

    fn sample_rate(self) -> u32 {
        match self {
            AmrBand::Narrow => NARROWBAND_RATE_HZ,
            AmrBand::Wide => WIDEBAND_RATE_HZ,
        }
    }

    fn samples_per_frame(self) -> u64 {
        match self {
            AmrBand::Narrow => NARROWBAND_SAMPLES_PER_FRAME,
            AmrBand::Wide => WIDEBAND_SAMPLES_PER_FRAME,
        }
    }

    fn caps(self) -> Caps {
        Caps::Audio {
            format: AudioFormat::Tagged(self.tag()),
            channels: 1,
            sample_rate: self.sample_rate(),
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    // the sample entry an MP4 track of this band carries, which the tagged decoder opens on
    fn sample_entry(self) -> Vec<u8> {
        crate::mp4box::audio_sample_entry(&self.tag().tag.to_le_bytes(), 1, self.sample_rate(), &[])
    }

    fn locate(self, buf: &[u8], eos: bool) -> Located {
        match self {
            AmrBand::Narrow => locate_frame::<NarrowbandToc>(buf, eos),
            AmrBand::Wide => locate_frame::<WidebandToc>(buf, eos),
        }
    }
}

const PARSED_FORMATS: &[AudioFormat] = &[
    AudioFormat::Tagged(AMR_NB_TAG),
    AudioFormat::Tagged(AMR_WB_TAG),
];

#[derive(Debug, Default)]
pub struct AmrParse {
    band: Option<AmrBand>,
    // set once the bytes after a start or flush show whether they open on the storage magic
    storage: Option<bool>,
    undecided: Vec<Frame>,
    caps_sent: bool,
    buf: Vec<u8>,
    buf_offset: u64,
    clock: FrameClock,
    // a seek re-reads the file from its start and drops the frames before the target
    seek: DemuxSeek<Frame>,
    sequence: u64,
    log_name: LogName,
}

impl AmrParse {
    pub fn new() -> Self {
        Self::default()
    }

    fn configured_band(&self) -> Result<AmrBand, G2gError> {
        self.band.ok_or(G2gError::NotConfigured)
    }

    async fn send_caps(&mut self, band: AmrBand, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if !self.caps_sent {
            out.push(PipelinePacket::CapsChanged(band.caps())).await?;
            self.caps_sent = true;
        }
        Ok(())
    }

    async fn open_storage(
        &mut self,
        band: AmrBand,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let entry = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(
                band.sample_entry().into_boxed_slice(),
            )),
            FrameTiming::default(),
            self.sequence,
        );
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(entry)).await?;
        Ok(())
    }

    async fn sniff(
        &mut self,
        band: AmrBand,
        frame: Frame,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let slice = frame
            .domain
            .require_system_slice(short_type_name::<Self>())?;
        self.buf.extend_from_slice(slice);
        self.undecided.push(frame);
        let magic = band.magic();
        if self.buf.len() < magic.len() && magic.starts_with(&self.buf) {
            return Ok(());
        }
        self.decide(band, out).await
    }

    async fn decide(&mut self, band: AmrBand, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let magic = band.magic();
        let storage = self.buf.starts_with(magic);
        self.storage = Some(storage);
        let undecided = core::mem::take(&mut self.undecided);
        if !storage {
            self.buf.clear();
            for frame in undecided {
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            return Ok(());
        }
        self.open_storage(band, out).await?;
        self.buf.drain(..magic.len());
        self.buf_offset = magic.len() as u64;
        self.drain(false, out).await
    }

    async fn drain(&mut self, eos: bool, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let band = self.configured_band()?;
        loop {
            let (start, len) = match band.locate(&self.buf, eos) {
                Located::Frame { start, len } => (start, len),
                Located::NeedMore { junk } => {
                    self.buf.drain(..junk);
                    self.buf_offset += junk as u64;
                    return Ok(());
                }
            };
            self.buf.drain(..start);
            self.buf_offset += start as u64;
            let start_offset = self.buf_offset;
            let data: Vec<u8> = self.buf.drain(..len).collect();
            self.buf_offset += len as u64;
            self.emit(band, data, start_offset, out).await?;
        }
    }

    async fn emit(
        &mut self,
        band: AmrBand,
        data: Vec<u8>,
        start_offset: u64,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let samples = band.samples_per_frame();
        let pts_ns = self
            .clock
            .frame_pts(start_offset, samples, band.sample_rate());
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns: ticks_to_ns(samples, band.sample_rate()),
                keyframe: true,
                ..FrameTiming::default()
            },
            self.sequence,
        );
        self.sequence += 1;
        // AMR frames decode on their own, so no frame ahead of the target is kept
        push_admitted(self.seek.admit_with_preroll(pts_ns, 0, frame), out).await
    }
}

impl AsyncElement for AmrParse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn seek_route(&self) -> Option<ElementSeekRoute> {
        Some(ElementSeekRoute::UpstreamBytes)
    }

    fn set_seek(&mut self, app: SeekController, upstream: SeekController) {
        self.seek.with(app, upstream);
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        crate::compressed_audio_passthrough::require_format(PARSED_FORMATS, upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(crate::compressed_audio_passthrough::passthrough_caps(
            PARSED_FORMATS,
        ))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.band = Some(AmrBand::of(absolute_caps).ok_or(G2gError::CapsMismatch)?);
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AMR parser",
            "Codec/Parser/Audio",
            "Splits an AMR-NB or AMR-WB storage file into frames",
            crate::AARON_BOXER,
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let band = self.configured_band()?;
            self.seek.hold_at_segment_end().await;
            self.seek.poll_request();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if self.seek.dropping_input() {
                        return Ok(());
                    }
                    self.send_caps(band, out).await?;
                    match self.storage {
                        None => return self.sniff(band, frame, out).await,
                        Some(false) => {
                            out.push(PipelinePacket::DataFrame(frame)).await?;
                            return Ok(());
                        }
                        Some(true) => {}
                    }
                    let slice = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    if let Some(pts) = frame.timing.pts().filter(|pts| *pts != 0) {
                        self.clock.note_chunk(
                            self.buf_offset,
                            self.buf_offset + self.buf.len() as u64,
                            pts,
                        );
                    }
                    self.buf.extend_from_slice(slice);
                    self.drain(false, out).await?;
                }
                PipelinePacket::Eos => {
                    if self
                        .seek
                        .eos_before_flush(self.seek.dropping_input())
                        .await?
                    {
                        return Ok(());
                    }
                    if self.storage.is_none() && !self.undecided.is_empty() {
                        self.decide(band, out).await?;
                    }
                    if self.storage == Some(true) {
                        self.drain(true, out).await?;
                    }
                    self.seek.end_and_wait_for_seek(out).await?;
                }
                PipelinePacket::Flush => {
                    let forward = self.seek.on_flush();
                    self.buf.clear();
                    self.buf_offset = 0;
                    self.clock.reset();
                    self.storage = None;
                    self.undecided.clear();
                    if forward {
                        out.push(PipelinePacket::Flush).await?;
                    }
                }
                // AMR's rate and channel count are fixed by its band, which these caps state
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }
}

impl LogSource for AmrParse {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

impl PadTemplates for AmrParse {
    fn pad_templates() -> Vec<PadTemplate> {
        crate::compressed_audio_passthrough::pad_templates(PARSED_FORMATS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{data_pts_ns, first_caps, run, run_packets, CollectSink};
    use crate::typefind::elementary_audio_caps;

    // tools/amr-fixtures.sh, frame counts as ffprobe reports them
    const NARROWBAND_FIXTURE: &[u8] = include_bytes!("../tests/fixtures/amr_nb_440hz_11frames.amr");
    const NARROWBAND_FRAMES: usize = 11;
    const WIDEBAND_FIXTURE: &[u8] = include_bytes!("../tests/fixtures/amr_wb_440hz_10frames.amr");
    const WIDEBAND_FRAMES: usize = 10;
    const FRAME_NS: u64 = 20_000_000;

    fn data_frames(sink: &CollectSink) -> Vec<Vec<u8>> {
        sink.packets
            .iter()
            .filter_map(|packet| match packet {
                PipelinePacket::DataFrame(frame) => {
                    frame.domain.as_system_slice().map(<[u8]>::to_vec)
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_storage_file_comes_out_one_frame_per_buffer_behind_a_sample_entry() {
        for (fixture, tag, frames) in [
            (NARROWBAND_FIXTURE, AMR_NB_TAG, NARROWBAND_FRAMES),
            (WIDEBAND_FIXTURE, AMR_WB_TAG, WIDEBAND_FRAMES),
        ] {
            let band = AmrBand::of(&elementary_audio_caps(AudioFormat::Tagged(tag))).unwrap();
            // lands inside the magic and inside frames
            let chunks: Vec<&[u8]> = fixture.chunks(7).collect();
            let sink = run(
                &mut AmrParse::new(),
                &elementary_audio_caps(AudioFormat::Tagged(tag)),
                &chunks,
            );
            assert_eq!(first_caps(&sink.packets), Some(band.caps()));
            let buffers = data_frames(&sink);
            let (entry, audio) = buffers.split_first().unwrap();
            let description = crate::fmp4::SoundDescription::parse(
                &entry[crate::mp4box::BOX_HEADER_LEN..],
                tag.tag,
            )
            .unwrap();
            assert_eq!(
                (description.channels, description.sample_rate),
                (1, band.sample_rate())
            );
            assert_eq!(audio.len(), frames);
            assert_eq!(audio.concat(), &fixture[band.magic().len()..]);
            let audio_pts = &data_pts_ns(&sink.packets)[1..];
            assert!(audio_pts
                .iter()
                .enumerate()
                .all(|(index, pts)| *pts == index as u64 * FRAME_NS));
        }
    }

    #[test]
    fn input_without_the_storage_magic_passes_through() {
        let caps = elementary_audio_caps(AudioFormat::Tagged(AMR_NB_TAG));
        let entry = AmrBand::Narrow.sample_entry();
        let frame = &NARROWBAND_FIXTURE[AMR_NB_MAGIC.len()..][..13];
        let sink = run_packets(
            &mut AmrParse::new(),
            &caps,
            [&entry[..], frame]
                .map(|bytes| PipelinePacket::DataFrame(crate::testutil::frame(bytes.to_vec()))),
        );
        assert_eq!(data_frames(&sink), [entry.clone(), frame.to_vec()]);
    }

    #[test]
    fn typefind_names_both_bands() {
        for (fixture, tag) in [
            (NARROWBAND_FIXTURE, AMR_NB_TAG),
            (WIDEBAND_FIXTURE, AMR_WB_TAG),
        ] {
            assert_eq!(
                crate::typefind::sniff_caps(fixture),
                Some(elementary_audio_caps(AudioFormat::Tagged(tag)))
            );
        }
    }
}
