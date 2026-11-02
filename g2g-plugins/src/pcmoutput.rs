use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, Caps, CapsSet, ChannelLayout, FrameTiming, G2gError, OutputSink, PipelinePacket,
    Segment, ANY_CHANNELS, ANY_SAMPLE_RATE,
};

use crate::pcmclip::clip_pcm_to_segment;

#[derive(Debug)]
pub(crate) struct DecodedPcm {
    pub(crate) pcm: Vec<u8>,
    pub(crate) channels: u8,
    pub(crate) sample_rate: u32,
    pub(crate) channel_layout: ChannelLayout,
}

impl DecodedPcm {
    pub(crate) fn bytes_per_sample_frame(&self) -> usize {
        usize::from(self.channels) * 2
    }

    fn caps(&self) -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: self.channels,
            sample_rate: self.sample_rate,
            channel_layout: self.channel_layout,
        }
    }
}

pub(crate) fn f32_to_s16(sample: f32) -> i16 {
    // swresample converts float to s16 this way: lrintf at 1<<15 scale, clipped
    (sample * 32768.0)
        .round_ties_even()
        .clamp(-32768.0, 32767.0) as i16
}

pub(crate) fn pcm_s16_decoder_output() -> CapsSet {
    // fixate cannot settle a lone ANY_SAMPLE_RATE
    let pcm = |sample_rate| Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: ANY_CHANNELS,
        sample_rate,
        channel_layout: ChannelLayout::UNSPECIFIED,
    };
    CapsSet::from_alternatives(alloc::vec![pcm(48_000), pcm(ANY_SAMPLE_RATE)])
}

#[derive(Debug, Default)]
pub(crate) struct PcmOutput {
    segment: Segment,
    announced_caps: Option<Caps>,
    sequence: u64,
}

impl PcmOutput {
    pub(crate) fn frames_pushed(&self) -> u64 {
        self.sequence
    }

    #[cfg(any(
        all(any(target_os = "linux", windows), feature = "ffmpeg"),
        feature = "aac-decode",
        feature = "mpeg-audio-decode"
    ))]
    pub(crate) fn forget_caps(&mut self) {
        self.announced_caps = None;
    }

    pub(crate) async fn forward_segment(
        &mut self,
        segment: Segment,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        self.segment = segment;
        out.push(PipelinePacket::Segment(segment)).await?;
        Ok(())
    }

    pub(crate) async fn forward_caps(
        &mut self,
        caps: Caps,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        // the runner hands a decoder its own solved output caps as an incoming CapsChanged
        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
        self.announced_caps = Some(caps);
        Ok(())
    }

    pub(crate) async fn push(
        &mut self,
        timing: FrameTiming,
        decoded: DecodedPcm,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let caps = decoded.caps();
        let bytes_per_sample_frame = decoded.bytes_per_sample_frame();
        let Some((timing, pcm)) = clip_pcm_to_segment(
            &self.segment,
            timing,
            decoded.pcm,
            bytes_per_sample_frame,
            decoded.sample_rate,
        )?
        else {
            return Ok(());
        };
        if self.announced_caps.as_ref() != Some(&caps) {
            out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
            self.announced_caps = Some(caps);
        }
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(pcm.into_boxed_slice())),
            timing,
            self.sequence,
        );
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    #[cfg(any(
        all(any(target_os = "linux", windows), feature = "ffmpeg"),
        feature = "aac-decode",
        feature = "mpeg-audio-decode"
    ))]
    pub(crate) async fn push_consecutive(
        &mut self,
        timing: FrameTiming,
        decoded: Vec<DecodedPcm>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let mut pts_ns = timing.pts_ns;
        for pcm in decoded {
            let samples = pcm
                .pcm
                .len()
                .checked_div(pcm.bytes_per_sample_frame())
                .ok_or(G2gError::CapsMismatch)?;
            let frame_timing = FrameTiming { pts_ns, ..timing };
            if pts_ns != FrameTiming::PTS_NONE {
                pts_ns = pts_ns
                    .saturating_add(g2g_core::time::ticks_to_ns(samples as u64, pcm.sample_rate));
            }
            self.push(frame_timing, pcm, out).await?;
        }
        Ok(())
    }
}
