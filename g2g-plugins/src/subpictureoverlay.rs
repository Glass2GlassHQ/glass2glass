//! Bitmap-subtitle overlay (M1005): a video pad and a decoded-subpicture pad in,
//! the video with the cue on screen blended over it out.
//!
//! The subpicture decoders ([`vobsubdec`](crate::vobsubdec),
//! [`dvbsubdec`](crate::dvbsubdec), [`pgsdec`](crate::pgsdec)) paint each cue onto
//! a full-frame transparent RGBA canvas, plus a fully transparent one at the cue's
//! hide time. This element is what puts those canvases on the picture: it holds
//! the last canvas whose PTS the video has reached and source-over blends it onto
//! every frame, so a cue stays up between canvases and the clearing canvas takes
//! it down. It is a [`MultiInputElement`] opting into the runner's
//! `input_pts_ordered` merge, like [`TextOverlayN`](crate::textoverlay::TextOverlayN),
//! so a canvas lands just before the first video frame it covers.
//!
//! RGBA8 in and out on both pads, CPU (`videoconvert` on either side for another
//! format). A canvas whose geometry differs from the video is scaled bilinear to
//! fit, so a PAL-sized DVD subpicture composites onto a scaled picture.
//!
//! ```text
//! matroskademux name=d  d.video_0 ! avdec_h264 ! videoconvert ! o.video
//!                       d.text_0 ! vobsubdec ! o.text
//!                       subpictureoverlay name=o ! videoconvert ! autovideosink
//! ```

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError, MemoryDomain,
    MultiInputElement, OutputSink, PipelinePacket, Rate, RawVideoFormat,
};

use crate::compositor::blend_over_scaled;
use crate::paint::blend_px;
use crate::textoverlay::TextOverlay;

/// One decoded subpicture cue: the canvas pixels and the geometry to read them
/// at. `pixels` is exactly `width * height * 4` bytes.
#[derive(Debug)]
struct Canvas {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl Canvas {
    /// A canvas with at least one pixel to draw, else `None`. The clearing canvas
    /// a decoder emits at a cue's hide time has none, and dropping it here is
    /// what takes the cue off the picture.
    fn visible(width: u32, height: u32, pixels: Vec<u8>) -> Option<Self> {
        if !pixels.chunks_exact(4).any(|px| px[3] != 0) {
            return None;
        }
        Some(Self {
            width,
            height,
            pixels,
        })
    }

    /// Source-over this canvas onto an RGBA8 video frame of `width` x `height`
    /// (at least `width * height * 4` bytes), scaling bilinear when the two
    /// geometries differ.
    fn blend_onto(&self, video: &mut [u8], width: u32, height: u32) {
        if (self.width, self.height) != (width, height) {
            blend_over_scaled(
                video,
                width as usize,
                height as usize,
                &self.pixels,
                self.width as usize,
                self.height as usize,
                0,
                0,
                width as usize,
                height as usize,
                255,
            );
            return;
        }
        // Same geometry: blend pixel for pixel, skipping the transparent run a
        // subpicture canvas is mostly made of.
        for (i, px) in self.pixels.chunks_exact(4).enumerate() {
            if px[3] != 0 {
                blend_px(video, i * 4, [px[0], px[1], px[2], px[3]], 255);
            }
        }
    }
}

/// Blends decoded subpicture cues onto video.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::subpictureoverlay::SubPictureOverlay;
///
/// let overlay = SubPictureOverlay::new();
/// assert_eq!(overlay.drawn_count(), 0);
/// ```
#[derive(Debug, Default)]
pub struct SubPictureOverlay {
    /// The negotiated video caps, captured at `configure(VIDEO)`; the merged
    /// output (it `output_follows_input` the video pad).
    video_caps: Option<Caps>,
    width: u32,
    height: u32,
    /// Geometry the canvases arrive at, from the subpicture pad's caps: a canvas
    /// frame is bare pixels with nothing in it to say how wide it is.
    canvas_width: u32,
    canvas_height: u32,
    /// Canvases whose show time the video has not reached yet, in arrival order.
    /// `None` for a clearing canvas. The PTS merge keeps this at a single entry.
    pending: Vec<(u64, Option<Canvas>)>,
    /// The cue on screen, `None` between cues.
    shown: Option<Canvas>,
    drawn: u64,
}

impl SubPictureOverlay {
    /// Input pad indices: video on 0, the subpicture canvases on 1.
    const VIDEO: usize = 0;
    const SUBPICTURE: usize = 1;

    /// An overlay with nothing on screen. Both geometries are set at negotiation
    /// and follow any mid-stream `CapsChanged`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Count of video frames processed (whether or not a cue was showing).
    pub fn drawn_count(&self) -> u64 {
        self.drawn
    }

    /// Show every canvas whose PTS the video has reached, in order, so the last
    /// one due is the one on screen.
    fn show_due(&mut self, pts_ns: u64) {
        while self.pending.first().is_some_and(|(pts, _)| *pts <= pts_ns) {
            self.shown = self.pending.remove(0).1;
        }
    }

    /// Take a canvas frame's pixels at the geometry the pad's caps declare.
    /// A frame too short for that geometry is dropped: it cannot be read, and the
    /// cue it carried is one cue, not the stream.
    fn take_canvas(&self, frame: &g2g_core::frame::Frame) -> Option<Option<Canvas>> {
        let slice = frame.domain.as_system_slice()?;
        let need = self.canvas_width as usize * self.canvas_height as usize * 4;
        if need == 0 || slice.len() < need {
            return None;
        }
        Some(Canvas::visible(
            self.canvas_width,
            self.canvas_height,
            slice[..need].to_vec(),
        ))
    }

    /// RGBA8 at any geometry, what both pads take.
    fn rgba_any() -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
        }
    }
}

impl MultiInputElement for SubPictureOverlay {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Subpicture overlay",
            "Filter/Editor/Video",
            "Blends decoded bitmap-subtitle canvases onto video by PTS",
            "g2g",
        )
    }

    fn input_count(&self) -> usize {
        2
    }

    /// Merge the two pads by PTS, so a canvas lands before the first video frame
    /// it covers (correct subtitle timing).
    fn input_pts_ordered(&self) -> bool {
        true
    }

    /// The merged output is the video pad's stream (the same pixels with the cue
    /// painted on), so the solver derives the output caps from pad 0.
    fn output_follows_input(&self) -> Option<usize> {
        Some(Self::VIDEO)
    }

    /// Named request pads: `video` -> the video pad (0), `text` / `subtitle` ->
    /// the subpicture pad (1), so a launch line can wire the two branches in
    /// either order. A demuxer surfaces a subpicture track as a text pad, which
    /// is why the canvases arrive on that name.
    fn input_pad_index(
        &self,
        req: &g2g_core::runtime::PadRequest,
        _ordinal: usize,
    ) -> Option<usize> {
        match req.kind {
            g2g_core::runtime::PadKind::Video => Some(Self::VIDEO),
            g2g_core::runtime::PadKind::Text => Some(Self::SUBPICTURE),
            _ => None,
        }
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if TextOverlay::accepts(upstream_caps) {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    /// Both pads accept RGBA8 at any geometry: the video sets the output shape and
    /// a canvas is scaled onto it, so the two need not agree.
    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(Self::rgba_any()))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        let (w, h) = TextOverlay::dims(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        match input {
            Self::VIDEO => {
                self.width = w;
                self.height = h;
                self.video_caps = Some(absolute_caps.clone());
            }
            Self::SUBPICTURE => {
                self.canvas_width = w;
                self.canvas_height = h;
            }
            _ => return Err(G2gError::CapsMismatch),
        }
        Ok(ConfigureOutcome::Accepted)
    }

    /// The merged output is the video stream. Negotiation derives it from the
    /// video pad (`output_follows_input`); this is the runtime mirror, valid once
    /// that pad is configured.
    fn output_caps(&self) -> Result<Caps, G2gError> {
        self.video_caps.clone().ok_or(G2gError::NotConfigured)
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match input {
                Self::VIDEO => match packet {
                    PipelinePacket::DataFrame(mut frame) => {
                        self.show_due(frame.timing.pts_ns);
                        if let Some(canvas) = &self.shown {
                            let MemoryDomain::System(slice) = &mut frame.domain else {
                                return Err(G2gError::UnsupportedDomain);
                            };
                            let need = self.width as usize * self.height as usize * 4;
                            let buf = slice.as_mut_slice();
                            if buf.len() < need {
                                return Err(G2gError::CapsMismatch);
                            }
                            canvas.blend_onto(&mut buf[..need], self.width, self.height);
                        }
                        self.drawn += 1;
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                    }
                    PipelinePacket::CapsChanged(caps) => {
                        if let Some((w, h)) = TextOverlay::dims(&caps) {
                            self.width = w;
                            self.height = h;
                        }
                        out.push(PipelinePacket::CapsChanged(caps)).await?;
                    }
                    // The runner emits the merged Eos; don't double it.
                    PipelinePacket::Eos => {}
                    other => {
                        out.push(other).await?;
                    }
                },
                // Subpicture pad: hold each canvas until the video reaches it.
                // Nothing here is forwarded, the video pad's stream is the output.
                Self::SUBPICTURE => match packet {
                    PipelinePacket::DataFrame(frame) => {
                        if let Some(canvas) = self.take_canvas(&frame) {
                            self.pending.push((frame.timing.pts_ns, canvas));
                        }
                    }
                    PipelinePacket::CapsChanged(caps) => {
                        if let Some((w, h)) = TextOverlay::dims(&caps) {
                            self.canvas_width = w;
                            self.canvas_height = h;
                        }
                    }
                    // A flush / seek on the subtitle stream clears the picture.
                    PipelinePacket::Flush => {
                        self.pending.clear();
                        self.shown = None;
                    }
                    _ => {}
                },
                _ => {}
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::frame::{Frame, FrameTiming};
    use g2g_core::memory::SystemSlice;
    use g2g_core::PushOutcome;

    const W: u32 = 8;
    const H: u32 = 4;

    #[derive(Default)]
    struct CollectSink {
        packets: Vec<PipelinePacket>,
    }
    impl OutputSink for CollectSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");

            self.packets.push(packet);
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    fn rgba(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(25 << 16),
            interlace: g2g_core::Interlace::Any,
        }
    }

    fn frame(pixels: Vec<u8>, pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(pixels.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                ..FrameTiming::default()
            },
            0,
        ))
    }

    /// A canvas whose only drawn pixel is an opaque red one at (1, 0).
    fn one_dot(w: u32, h: u32) -> Vec<u8> {
        let mut px = alloc::vec![0u8; (w * h * 4) as usize];
        px[4..8].copy_from_slice(&[255, 0, 0, 255]);
        px
    }

    fn configured() -> SubPictureOverlay {
        let mut overlay = SubPictureOverlay::new();
        overlay
            .configure_pipeline(SubPictureOverlay::VIDEO, &rgba(W, H))
            .unwrap();
        overlay
            .configure_pipeline(SubPictureOverlay::SUBPICTURE, &rgba(W, H))
            .unwrap();
        overlay
    }

    fn out_pixels(sink: &CollectSink) -> Vec<&[u8]> {
        sink.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => f.domain.as_system_slice(),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn a_canvas_paints_only_from_its_own_pts() {
        let mut overlay = configured();
        let mut sink = CollectSink::default();
        let black = || alloc::vec![0u8, 0, 0, 255].repeat((W * H) as usize);

        overlay
            .process(SubPictureOverlay::VIDEO, frame(black(), 0), &mut sink)
            .await
            .unwrap();
        overlay
            .process(
                SubPictureOverlay::SUBPICTURE,
                frame(one_dot(W, H), 1_000),
                &mut sink,
            )
            .await
            .unwrap();
        overlay
            .process(SubPictureOverlay::VIDEO, frame(black(), 2_000), &mut sink)
            .await
            .unwrap();

        let frames = out_pixels(&sink);
        assert_eq!(frames.len(), 2, "only the video pad reaches the output");
        assert_eq!(&frames[0][4..8], [0, 0, 0, 255], "before the cue's PTS");
        assert_eq!(&frames[1][4..8], [255, 0, 0, 255], "the cue is painted");
        assert_eq!(&frames[1][0..4], [0, 0, 0, 255], "and only where it draws");
    }

    #[tokio::test]
    async fn a_clearing_canvas_takes_the_cue_down() {
        let mut overlay = configured();
        let mut sink = CollectSink::default();
        let black = || alloc::vec![0u8, 0, 0, 255].repeat((W * H) as usize);
        let clear = alloc::vec![0u8; (W * H * 4) as usize];

        overlay
            .process(
                SubPictureOverlay::SUBPICTURE,
                frame(one_dot(W, H), 0),
                &mut sink,
            )
            .await
            .unwrap();
        overlay
            .process(SubPictureOverlay::VIDEO, frame(black(), 1_000), &mut sink)
            .await
            .unwrap();
        overlay
            .process(
                SubPictureOverlay::SUBPICTURE,
                frame(clear, 2_000),
                &mut sink,
            )
            .await
            .unwrap();
        overlay
            .process(SubPictureOverlay::VIDEO, frame(black(), 3_000), &mut sink)
            .await
            .unwrap();

        let frames = out_pixels(&sink);
        assert_eq!(&frames[0][4..8], [255, 0, 0, 255], "cue showing");
        assert_eq!(&frames[1][4..8], [0, 0, 0, 255], "cue cleared");
    }

    #[tokio::test]
    async fn a_smaller_canvas_scales_onto_the_video() {
        let mut overlay = SubPictureOverlay::new();
        overlay
            .configure_pipeline(SubPictureOverlay::VIDEO, &rgba(4, 4))
            .unwrap();
        overlay
            .configure_pipeline(SubPictureOverlay::SUBPICTURE, &rgba(2, 2))
            .unwrap();
        let mut sink = CollectSink::default();

        // Opaque green over the whole 2x2 canvas: every video pixel is covered.
        let canvas = alloc::vec![0u8, 255, 0, 255].repeat(4);
        overlay
            .process(SubPictureOverlay::SUBPICTURE, frame(canvas, 0), &mut sink)
            .await
            .unwrap();
        overlay
            .process(
                SubPictureOverlay::VIDEO,
                frame(alloc::vec![0u8, 0, 0, 255].repeat(16), 0),
                &mut sink,
            )
            .await
            .unwrap();

        let frames = out_pixels(&sink);
        for px in frames[0].chunks_exact(4) {
            assert_eq!(px, [0, 255, 0, 255], "the canvas covers the scaled frame");
        }
    }

    #[test]
    fn rejects_a_non_rgba_pad() {
        let mut overlay = SubPictureOverlay::new();
        assert!(overlay
            .configure_pipeline(SubPictureOverlay::VIDEO, &Caps::Klv)
            .is_err());
    }
}
