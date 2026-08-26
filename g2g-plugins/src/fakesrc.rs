//! Fake source: emits `num-buffers` byte buffers of `sizemax` bytes, filled as
//! `filltype` says. The gst `fakesrc` analog, for driving a graph without a
//! file, a device or a network.
//!
//! The buffers are a raw byte stream typed exactly as an untyped `filesrc`
//! types one, so `fakesrc ! typefind` and `filesrc ! typefind` behave alike.
//!
//! Two deliberate departures from gst `fakesrc`: `sizetype` is not a property,
//! every buffer is `sizemax` bytes (gst's `sizetype=empty` default makes
//! `fakesrc ! anything` push zero-byte buffers), and `filltype=nothing` writes
//! zeros, since Rust has no uninitialised buffer to hand out.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

/// gst `fakesrc`'s `sizemax` default.
const DEFAULT_BUFFER_SIZE: usize = 4096;

/// Seed of the `filltype=random` generator, fixed so two runs of the same
/// pipeline produce the same bytes.
const RANDOM_SEED: u32 = 0x2545_f491;

/// How each emitted buffer is filled.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FillType {
    /// gst leaves the malloced bytes as they are; Rust hands out no
    /// uninitialised memory, so this writes zeros. gst's default.
    #[default]
    Nothing,
    /// Zeros.
    Zero,
    /// A deterministic xorshift sequence, seeded from [`RANDOM_SEED`].
    Random,
    /// A byte counter `0x00 -> 0xff`, restarting at each buffer.
    Pattern,
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::fakesrc::FakeSrc;
///
/// // gst-launch equivalent: fakesrc num-buffers=20 sizemax=4096 filltype=pattern
/// let source = FakeSrc::new();
/// ```
#[derive(Debug)]
pub struct FakeSrc {
    /// Buffers to emit before EOS; `u64::MAX` is unlimited.
    target_buffers: u64,
    buffer_size: usize,
    fill: FillType,
    configured: bool,
}

impl Default for FakeSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeSrc {
    pub fn new() -> Self {
        Self {
            target_buffers: u64::MAX,
            buffer_size: DEFAULT_BUFFER_SIZE,
            fill: FillType::Nothing,
            configured: false,
        }
    }

    /// The type an untyped `filesrc` gives a raw byte stream, so the two sources
    /// are interchangeable ahead of a `typefind` or a demuxer.
    fn caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        }
    }
}

impl SourceLoop for FakeSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(Self::caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(Self::caps()))))
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
            let mut random_state = RANDOM_SEED;
            let mut sequence = 0u64;
            while sequence < self.target_buffers {
                let mut buf = vec![0u8; self.buffer_size].into_boxed_slice();
                fill_buffer(self.fill, &mut buf, &mut random_state);
                // Source-side wall-clock stamp so a sink can record
                // glass-to-glass latency, as filesrc does. Std-gated because
                // `monotonic_ns` is; a no_std build leaves it zero and the sink
                // skips the recording.
                #[cfg(feature = "std")]
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                #[cfg(not(feature = "std"))]
                let arrival_ns: u64 = 0;
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(buf)),
                    FrameTiming {
                        arrival_ns,
                        ..FrameTiming::default()
                    },
                    sequence,
                );
                sequence += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        FAKESRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Fake source",
            "Source",
            "Produces buffers of a chosen size and fill, for testing",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.target_buffers, &value),
            "sizemax" => {
                let bytes = value.as_uint().ok_or(PropError::Type)?;
                if bytes > MAX_BUFFER_SIZE {
                    return Err(PropError::Value);
                }
                self.buffer_size = bytes as usize;
                Ok(())
            }
            "filltype" => {
                let fill = value.as_str().ok_or(PropError::Type)?;
                self.fill = fill_from_str(fill).ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.target_buffers)),
            "sizemax" => Some(PropValue::Uint(self.buffer_size as u64)),
            "filltype" => Some(PropValue::Str(fill_to_str(self.fill).into())),
            _ => None,
        }
    }
}

/// Largest `sizemax` accepted, gst `fakesrc`'s own upper bound (its `sizemax`
/// is a signed 32-bit int).
const MAX_BUFFER_SIZE: u64 = i32::MAX as u64;

/// `FakeSrc`'s settable properties, named and defaulted as gst `fakesrc`.
static FAKESRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "buffers to emit then EOS (-1 = forever)",
    )
    .with_range("-1", "9223372036854775807")
    .with_default("-1"),
    PropertySpec::new("sizemax", PropKind::Uint, "bytes per emitted buffer")
        .with_range("0", "2147483647")
        .with_default("4096"),
    PropertySpec::new("filltype", PropKind::Str, "how each buffer is filled")
        .with_enum_values("nothing | zero | random | pattern")
        .with_default("nothing"),
];

/// Parse a `filltype` property string. The names are gst's enum nicks.
fn fill_from_str(s: &str) -> Option<FillType> {
    match s {
        "nothing" => Some(FillType::Nothing),
        "zero" => Some(FillType::Zero),
        "random" => Some(FillType::Random),
        "pattern" => Some(FillType::Pattern),
        _ => None,
    }
}

/// The `filltype` property string for a [`FillType`].
fn fill_to_str(fill: FillType) -> &'static str {
    match fill {
        FillType::Nothing => "nothing",
        FillType::Zero => "zero",
        FillType::Random => "random",
        FillType::Pattern => "pattern",
    }
}

/// Write `fill` over an already-zeroed buffer, advancing `random_state` so a
/// `random` run does not repeat one buffer's bytes.
fn fill_buffer(fill: FillType, buf: &mut [u8], random_state: &mut u32) {
    match fill {
        FillType::Nothing | FillType::Zero => {}
        FillType::Random => {
            for byte in buf.iter_mut() {
                *byte = next_random(random_state) as u8;
            }
        }
        FillType::Pattern => {
            for (index, byte) in buf.iter_mut().enumerate() {
                *byte = index as u8;
            }
        }
    }
}

/// Marsaglia xorshift32. Deterministic and dependency-free, which is all a fill
/// pattern needs; it is not a source of randomness for anything else.
fn next_random(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}
