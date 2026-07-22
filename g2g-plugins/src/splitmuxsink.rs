//! Splitting muxer sink (`splitmuxsink`). Muxes an H.264 / H.265 elementary
//! stream into a series of self-contained container files, starting a new file at
//! a keyframe once the current one reaches `max-size-time` or `max-size-bytes`.
//! The g2g analog of GStreamer's `splitmuxsink`. std-gated.
//!
//! The `muxer` property picks the child container: `mp4` (default), `matroska`, or
//! `mpegts`. It owns one muxer per segment and writes its output to the current
//! file through an internal byte-capturing sink; rotating finalizes the current
//! muxer (so its `moov`/cues/tail is written) and opens a fresh one, so every file
//! is independently playable. With both limits `0` (the default) it never splits
//! and behaves like `<muxer> ! filesink`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use std::fs::File;
use std::io::{BufWriter, Write};

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, PushOutcome, Rate, VideoCodec,
};

use crate::filesink::io_err;

// A type alias (not a `use` of the trait) so the trait's methods do not land in
// scope: `SplitMuxSink` implements `AsyncElement`, and the blanket
// `DynAsyncElement` impl would otherwise make `self.process` / `configure_pipeline`
// ambiguous. Method calls on the boxed muxer use the fully-qualified path.
type BoxedMuxer = Box<dyn g2g_core::element::DynAsyncElement>;

/// The container the child muxer writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MuxerKind {
    Mp4,
    Matroska,
    MpegTs,
}

impl MuxerKind {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "mp4" | "mp4mux" | "qtmux" => Some(Self::Mp4),
            "matroska" | "matroskamux" | "mkv" | "webm" => Some(Self::Matroska),
            "mpegts" | "mpegtsmux" | "ts" => Some(Self::MpegTs),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Matroska => "matroska",
            Self::MpegTs => "mpegts",
        }
    }

    /// A fresh child muxer instance for this container.
    fn make(self) -> BoxedMuxer {
        match self {
            Self::Mp4 => Box::new(crate::mp4mux::Mp4Mux::new()),
            Self::Matroska => Box::new(crate::mkvmux::MkvMux::new()),
            Self::MpegTs => Box::new(crate::tsmux::TsMux::new()),
        }
    }
}

/// An `OutputSink` that writes the muxer's output bytes to one segment file and
/// counts them (the split decision reads that count).
#[derive(Debug)]
struct SegmentSink {
    writer: BufWriter<File>,
    bytes: u64,
}

impl SegmentSink {
    fn create(path: &str) -> Result<Self, G2gError> {
        let file = File::create(path).map_err(io_err)?;
        Ok(Self {
            writer: BufWriter::new(file),
            bytes: 0,
        })
    }

    fn flush(&mut self) -> Result<(), G2gError> {
        self.writer.flush().map_err(io_err)
    }
}

impl OutputSink for SegmentSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = &packet {
                if let Some(b) = frame.domain.as_system_slice() {
                    self.writer.write_all(b).map_err(io_err)?;
                    self.bytes += b.len() as u64;
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

pub struct SplitMuxSink {
    location: String,
    muxer: MuxerKind,
    max_size_time_ns: u64,
    max_size_bytes: u64,
    index: u64,
    caps: Option<Caps>,
    mux: Option<BoxedMuxer>,
    seg: Option<SegmentSink>,
    segment_start_ns: u64,
    started: bool,
    files_written: u64,
}

// `dyn DynAsyncElement` is not Debug, so implement it by hand (like StreamDemux).
impl core::fmt::Debug for SplitMuxSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SplitMuxSink")
            .field("location", &self.location)
            .field("muxer", &self.muxer)
            .field("max_size_time_ns", &self.max_size_time_ns)
            .field("max_size_bytes", &self.max_size_bytes)
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl SplitMuxSink {
    /// `location` is a printf-style pattern with one integer field, e.g.
    /// `clip%03d.mp4`; without a field the index is appended. Defaults to the MP4
    /// child muxer; set `muxer` for matroska / mpegts.
    pub fn new(location: impl Into<String>) -> Self {
        Self {
            location: location.into(),
            muxer: MuxerKind::Mp4,
            max_size_time_ns: 0,
            max_size_bytes: 0,
            index: 0,
            caps: None,
            mux: None,
            seg: None,
            segment_start_ns: 0,
            started: false,
            files_written: 0,
        }
    }

    pub fn with_max_size_time(mut self, ns: u64) -> Self {
        self.max_size_time_ns = ns;
        self
    }

    pub fn files_written(&self) -> u64 {
        self.files_written
    }

    fn accept_input(&self, caps: &Caps) -> Result<(), G2gError> {
        match caps {
            Caps::CompressedVideo {
                codec: VideoCodec::H264 | VideoCodec::H265,
                ..
            } => Ok(()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Open a fresh segment: a new file and a new muxer configured with the caps.
    fn open_segment(&mut self) -> Result<(), G2gError> {
        let caps = self.caps.clone().ok_or(G2gError::NotConfigured)?;
        let path = crate::multifilesink::expand(&self.location, self.index);
        let mut mux = self.muxer.make();
        g2g_core::element::DynAsyncElement::configure_pipeline(mux.as_mut(), &caps)?;
        self.mux = Some(mux);
        self.seg = Some(SegmentSink::create(&path)?);
        self.index += 1;
        self.files_written += 1;
        Ok(())
    }

    /// Finalize the current segment: feed EOS so the muxer writes its tail, then
    /// flush the file.
    async fn finalize_current(&mut self) -> Result<(), G2gError> {
        if let (Some(mux), Some(seg)) = (self.mux.as_mut(), self.seg.as_mut()) {
            g2g_core::element::DynAsyncElement::process(mux.as_mut(), PipelinePacket::Eos, seg)
                .await?;
            seg.flush()?;
        }
        self.mux = None;
        self.seg = None;
        Ok(())
    }

    fn split_due(&self, pts_ns: u64) -> bool {
        let by_time = self.max_size_time_ns > 0
            && pts_ns.saturating_sub(self.segment_start_ns) >= self.max_size_time_ns;
        let by_bytes = self.max_size_bytes > 0
            && self.seg.as_ref().map(|s| s.bytes).unwrap_or(0) >= self.max_size_bytes;
        by_time || by_bytes
    }
}

impl AsyncElement for SplitMuxSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_input(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        let video = |codec| Caps::CompressedVideo {
            codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        CapsConstraint::Accepts(CapsSet::from_alternatives(Vec::from([
            video(VideoCodec::H264),
            video(VideoCodec::H265),
        ])))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.accept_input(absolute_caps)?;
        self.caps = Some(absolute_caps.clone());
        // The first segment opens lazily on the first frame so segment_start_ns
        // tracks the real first PTS.
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let pts = frame.timing.pts_ns;
                    if self.started && frame.timing.keyframe && self.split_due(pts) {
                        self.finalize_current().await?;
                        self.open_segment()?;
                        self.segment_start_ns = pts;
                    }
                    if !self.started {
                        self.open_segment()?;
                        self.segment_start_ns = pts;
                        self.started = true;
                    }
                    let mux = self.mux.as_mut().ok_or(G2gError::NotConfigured)?;
                    let seg = self.seg.as_mut().ok_or(G2gError::NotConfigured)?;
                    g2g_core::element::DynAsyncElement::process(
                        mux.as_mut(),
                        PipelinePacket::DataFrame(frame),
                        seg,
                    )
                    .await?;
                }
                PipelinePacket::Eos => {
                    self.finalize_current().await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    self.accept_input(&c)?;
                    self.caps = Some(c);
                }
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SPLITMUXSINK_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Split-muxer sink",
            "Sink/File",
            "Muxes to a series of MP4 files",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => self.location = value.as_str().ok_or(PropError::Type)?.into(),
            "muxer" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.muxer = MuxerKind::from_str(s).ok_or(PropError::Value)?;
            }
            "max-size-time" => self.max_size_time_ns = value.as_uint().ok_or(PropError::Type)?,
            "max-size-bytes" => self.max_size_bytes = value.as_uint().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.location.clone())),
            "muxer" => Some(PropValue::Str(self.muxer.as_str().into())),
            "max-size-time" => Some(PropValue::Uint(self.max_size_time_ns)),
            "max-size-bytes" => Some(PropValue::Uint(self.max_size_bytes)),
            _ => None,
        }
    }
}

static SPLITMUXSINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "printf-style file pattern, e.g. clip%03d.mp4",
    ),
    PropertySpec::new(
        "muxer",
        PropKind::Str,
        "child container: mp4 | matroska | mpegts",
    ),
    PropertySpec::new(
        "max-size-time",
        PropKind::Uint,
        "max segment duration in ns (0 = no split)",
    ),
    PropertySpec::new(
        "max-size-bytes",
        PropKind::Uint,
        "max segment size in bytes (0 = no split)",
    ),
];

impl PadTemplates for SplitMuxSink {
    fn pad_templates() -> Vec<PadTemplate> {
        let video = |codec| Caps::CompressedVideo {
            codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([PadTemplate::sink(CapsSet::from_alternatives(Vec::from([
            video(VideoCodec::H264),
            video(VideoCodec::H265),
        ])))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use g2g_core::{Frame, FrameTiming, MemoryDomain, SystemSlice};

    struct NullSink;
    impl OutputSink for NullSink {
        fn push<'a>(
            &'a mut self,
            _packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async { Ok(PushOutcome::Accepted) })
        }
    }

    /// A minimal Annex-B access unit: SPS + PPS + IDR (keyframe) or a single P
    /// slice, enough for `Mp4Mux` to build its moov and accept subsequent AUs.
    fn au(nals: &[&[u8]]) -> Vec<u8> {
        let mut v = Vec::new();
        for n in nals {
            v.extend_from_slice(&[0, 0, 0, 1]);
            v.extend_from_slice(n);
        }
        v
    }

    fn h264(w: u32, h: u32) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        }
    }

    fn keyframe(pts_ns: u64) -> PipelinePacket {
        // SPS (type 7), PPS (type 8), IDR (type 5).
        let bytes = au(&[
            &[0x67, 0x42, 0x00, 0x0a, 0x8b, 0x95],
            &[0x68, 0xce, 0x3c, 0x80],
            &[0x65, 0x88, 0x80],
        ]);
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming {
                pts_ns,
                keyframe: true,
                duration_ns: 40_000_000,
                ..FrameTiming::default()
            },
            sequence: 0,
            meta: Default::default(),
        })
    }

    #[tokio::test]
    async fn splits_a_new_file_at_each_keyframe_past_the_time_limit() {
        let dir = std::env::temp_dir();
        let pat = dir.join("g2g_smux_%03d.mp4").to_string_lossy().into_owned();
        for i in 0..4 {
            let _ = std::fs::remove_file(crate::multifilesink::expand(&pat, i));
        }
        let mut sink = SplitMuxSink::new(&pat).with_max_size_time(50_000_000); // 50ms
        sink.configure_pipeline(&h264(320, 240)).unwrap();
        let mut out = NullSink;
        // Keyframes at 0, 60ms, 120ms: each past the 50ms limit -> new file each.
        sink.process(keyframe(0), &mut out).await.unwrap();
        sink.process(keyframe(60_000_000), &mut out).await.unwrap();
        sink.process(keyframe(120_000_000), &mut out).await.unwrap();
        sink.process(PipelinePacket::Eos, &mut out).await.unwrap();
        assert_eq!(
            sink.files_written(),
            3,
            "one file per keyframe past the limit"
        );
        for i in 0..3 {
            let path = crate::multifilesink::expand(&pat, i);
            let meta = std::fs::metadata(&path).expect("segment file exists");
            assert!(meta.len() > 0, "segment {i} has data");
            let _ = std::fs::remove_file(&path);
        }
    }

    async fn split_into_two_with_muxer(muxer: &str, ext: &str) {
        let dir = std::env::temp_dir();
        let pat = dir
            .join(format!("g2g_smux_{muxer}_%03d.{ext}"))
            .to_string_lossy()
            .into_owned();
        for i in 0..3 {
            let _ = std::fs::remove_file(crate::multifilesink::expand(&pat, i));
        }
        let mut sink = SplitMuxSink::new(&pat);
        sink.set_property("muxer", PropValue::Str(muxer.into()))
            .unwrap();
        sink.set_property("max-size-time", PropValue::Uint(50_000_000))
            .unwrap();
        sink.configure_pipeline(&h264(320, 240)).unwrap();
        let mut out = NullSink;
        sink.process(keyframe(0), &mut out).await.unwrap();
        sink.process(keyframe(60_000_000), &mut out).await.unwrap();
        sink.process(PipelinePacket::Eos, &mut out).await.unwrap();
        assert_eq!(
            sink.files_written(),
            2,
            "{muxer}: one file per keyframe past the limit"
        );
        for i in 0..2 {
            let path = crate::multifilesink::expand(&pat, i);
            let meta =
                std::fs::metadata(&path).unwrap_or_else(|_| panic!("{muxer} segment {i} exists"));
            assert!(meta.len() > 0, "{muxer} segment {i} has data");
            let _ = std::fs::remove_file(&path);
        }
    }

    #[tokio::test]
    async fn mpegts_muxer_splits() {
        split_into_two_with_muxer("mpegts", "ts").await;
    }

    #[tokio::test]
    async fn matroska_muxer_splits() {
        split_into_two_with_muxer("matroska", "mkv").await;
    }

    #[tokio::test]
    async fn no_limit_writes_a_single_file() {
        let dir = std::env::temp_dir();
        let pat = dir
            .join("g2g_smux_single_%03d.mp4")
            .to_string_lossy()
            .into_owned();
        let _ = std::fs::remove_file(crate::multifilesink::expand(&pat, 0));
        let mut sink = SplitMuxSink::new(&pat);
        sink.configure_pipeline(&h264(320, 240)).unwrap();
        let mut out = NullSink;
        sink.process(keyframe(0), &mut out).await.unwrap();
        sink.process(keyframe(40_000_000), &mut out).await.unwrap();
        sink.process(PipelinePacket::Eos, &mut out).await.unwrap();
        assert_eq!(sink.files_written(), 1, "no split limit -> one file");
        let _ = std::fs::remove_file(crate::multifilesink::expand(&pat, 0));
    }
}
