//! A third-party g2g source element, registered and run via the text launcher.
//!
//! `cargo run -p g2g-plugins --features std --example third_party_source`
//!
//! The source half of [`third_party_element`](third_party_element.rs): implement
//! `SourceLoop`, add `PadTemplates` + `metadata` (so `g2g-inspect` sees it),
//! expose a `register(&mut Registry)`, then use the source by name in a
//! `gst-launch` line. A source owns its loop, so `run` decides when the stream
//! ends and pushes the final `Eos` itself, which is the one packet a transform
//! must never forward.

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph, Registry, SourceFactory, SourceLoop};
use g2g_core::{
    Caps, CapsSet, Colorimetry, ConfigureOutcome, Dim, ElementMetadata, G2gError, Interlace,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, Rate, RawVideoFormat,
};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const FRAMERATE: u32 = 30;
/// gst's `num-buffers` default: emit until something else ends the run.
const UNBOUNDED: i64 = -1;

/// Emits solid RGBA frames whose grey level rises one step per frame. Swap the
/// body of the `run` loop for real capture or synthesis.
#[derive(Debug)]
struct RampSrc {
    num_buffers: i64,
    configured: bool,
}

impl Default for RampSrc {
    fn default() -> Self {
        Self {
            num_buffers: UNBOUNDED,
            configured: false,
        }
    }
}

impl RampSrc {
    /// The one shape this source produces. Every field is fixed, so the caps
    /// solver has nothing left to fixate (`Dim::Any` would fail fixation).
    fn caps() -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Fixed(FRAMERATE << 16),
            interlace: Interlace::Any,
            colorimetry: Colorimetry::UNKNOWN,
        }
    }
}

impl SourceLoop for RampSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    // A source that has to do I/O to learn its caps (an RTSP DESCRIBE, a device
    // probe) returns a real future here; a synthetic one is already `Ready`.
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(Self::caps()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let frame_duration_ns = 1_000_000_000 / u64::from(FRAMERATE);
            let mut emitted = 0u64;
            while self.num_buffers < 0 || emitted < self.num_buffers as u64 {
                let grey = (emitted % 256) as u8;
                let pixels = vec![grey; (WIDTH * HEIGHT * 4) as usize].into_boxed_slice();
                let pts_ns = emitted * frame_duration_ns;
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(pixels)),
                    timing: FrameTiming {
                        pts_ns,
                        dts_ns: pts_ns,
                        duration_ns: frame_duration_ns,
                        capture_ns: pts_ns,
                        arrival_ns: g2g_core::metrics::monotonic_ns(),
                        keyframe: true,
                    },
                    sequence: emitted,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
                emitted += 1;
            }
            // A source must emit the final Eos itself before returning Ok.
            out.push(PipelinePacket::Eos).await?;
            Ok(emitted)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        RAMPSRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "num-buffers" => self.num_buffers = value.as_int().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "num-buffers" => Some(PropValue::Int(self.num_buffers)),
            _ => None,
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Grey ramp source",
            "Source/Video",
            "Emits solid RGBA frames whose grey level rises one step per frame.",
            "third-party",
        )
    }
}

static RAMPSRC_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "num-buffers",
    PropKind::Int,
    "frames to emit before EOS, -1 for unbounded",
)
.with_default("-1")];

impl PadTemplates for RampSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(RampSrc::caps()))])
    }
}

/// A source is registered with `register_source` (a transform uses
/// `register_launch`, a muxer `register_muxer`). The declared caps are what the
/// `decodebin` auto-plug search sees, so they match what `run` actually emits.
fn register(registry: &mut Registry) {
    registry.register_source(SourceFactory::new("rampsrc", RampSrc::caps(), || {
        Box::<RampSrc>::default()
    }));
}

fn main() {
    let mut registry = default_registry();
    register(&mut registry);

    let line = "rampsrc num-buffers=5 ! fakesink";
    let graph = parse_launch(&registry, line).expect("pipeline parses");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    let stats = rt
        .block_on(run_graph(graph, &WallClock::new(), 4))
        .expect("pipeline runs");
    println!("ran `{line}`");
    println!(
        "frames emitted: {}, consumed: {}",
        stats.frames_emitted, stats.frames_consumed
    );
}
