//! Synthetic video source. Emits a deterministic gradient pattern at a
//! fixed framerate in the system memory domain. CPU-only.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, Rate, VideoFormat,
};

#[derive(Debug)]
pub struct VideoTestSrc {
    width: u32,
    height: u32,
    framerate_q16: u32,
    target_frames: u64,
    configured: bool,
}

impl VideoTestSrc {
    /// `framerate` is in nominal fps; stored internally as Q16 fixed-point.
    pub fn new(width: u32, height: u32, framerate: u32, target_frames: u64) -> Self {
        Self {
            width,
            height,
            framerate_q16: framerate << 16,
            target_frames,
            configured: false,
        }
    }

    fn caps(&self) -> Caps {
        Caps::Video {
            format: VideoFormat::Rgba8,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(self.framerate_q16),
        }
    }
}

impl SourceLoop for VideoTestSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.caps())
    }

    fn configure_pipeline(
        &mut self,
        _absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(
        &'a mut self,
        out: &'a mut dyn OutputSink,
    ) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }

            let fps_q16 = u64::from(self.framerate_q16);
            let pts_step_ns: u64 = (1_000_000_000u64 << 16)
                .checked_div(fps_q16)
                .unwrap_or(0);

            let bytes_per_frame = (self.width as usize)
                .checked_mul(self.height as usize)
                .and_then(|n| n.checked_mul(4))
                .ok_or(G2gError::CapsMismatch)?;

            let caps = self.caps();

            for seq in 0..self.target_frames {
                let mut buf = vec![0u8; bytes_per_frame].into_boxed_slice();
                for (i, b) in buf.iter_mut().enumerate() {
                    *b = ((i as u64).wrapping_add(seq) & 0xFF) as u8;
                }

                let pts = seq * pts_step_ns;
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(buf)),
                    caps: caps.clone(),
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns: pts_step_ns,
                        capture_ns: pts,
                    },
                    sequence: seq,
                };

                out.push(PipelinePacket::DataFrame(frame)).await?;
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(self.target_frames)
        })
    }
}
