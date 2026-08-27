//! Random buffer size: re-chunks a byte stream into buffers of a random length
//! between `min` and `max`, the gst `rndbuffersize` analog. It exists to prove
//! a parser or demuxer downstream does not depend on where its input happens to
//! be cut.
//!
//! The sizes come from a seeded xorshift, so a `seed` replays the same cut
//! points. Only [`Caps::ByteStream`] input makes sense here: anything else
//! carries one frame per buffer, and cutting a frame in half destroys it.
//!
//! Each output buffer takes the pts of the input buffer being consumed when it
//! was cut. The tail left over at `Eos` goes out as one last buffer, shorter
//! than `min` when that is all the stream had.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::random::{next_random, XORSHIFT_BASE_STATE};
use crate::rechunk::Rechunker;

/// gst `rndbuffersize`'s `min` default.
const DEFAULT_MIN_BYTES: u64 = 1;
/// gst `rndbuffersize`'s `max` default.
const DEFAULT_MAX_BYTES: u64 = 8192;
/// Largest size accepted, gst `rndbuffersize`'s own bound (its `min` / `max`
/// are signed 32-bit ints).
const MAX_SIZE_BYTES: u64 = i32::MAX as u64;

/// # Example
///
/// ```no_run
/// use g2g_plugins::rndbuffersize::RndBufferSize;
///
/// // gst-launch equivalent: rndbuffersize min=100 max=500 seed=7
/// let element = RndBufferSize::new();
/// ```
#[derive(Debug)]
pub struct RndBufferSize {
    min: u64,
    max: u64,
    seed: u32,
    /// The generator, restarted from `seed` at each configure.
    state: u32,
    chunks: Rechunker,
    configured: bool,
}

impl Default for RndBufferSize {
    fn default() -> Self {
        Self::new()
    }
}

impl RndBufferSize {
    pub fn new() -> Self {
        Self {
            min: DEFAULT_MIN_BYTES,
            max: DEFAULT_MAX_BYTES,
            seed: 0,
            state: XORSHIFT_BASE_STATE,
            chunks: Rechunker::new(),
            configured: false,
        }
    }

    /// Buffers emitted so far.
    pub fn emitted(&self) -> u64 {
        self.chunks.emitted()
    }

    /// Cut every whole chunk the pending bytes hold and push it, at lengths
    /// drawn uniformly from `[min, max]`.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let (min, span, state) = (self.min, self.max - self.min + 1, &mut self.state);
        self.chunks
            .drain(
                || (min + u64::from(next_random(state)) % span) as usize,
                out,
            )
            .await
    }
}

impl AsyncElement for RndBufferSize {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "RndBufferSize",
            "Generic",
            "Re-chunks a byte stream into randomly sized buffers",
            "g2g",
        )
    }

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::ByteStream { .. } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// The encoding is untouched, so input and output are the same caps; the
    /// byte-stream requirement is enforced in `configure_pipeline`.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(absolute_caps, Caps::ByteStream { .. }) {
            return Err(G2gError::CapsMismatch);
        }
        if self.min > self.max {
            return Err(G2gError::CapsMismatch);
        }
        self.state = XORSHIFT_BASE_STATE ^ self.seed;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let bytes = frame
                        .domain
                        .as_system_slice()
                        .ok_or(G2gError::UnsupportedDomain)?;
                    self.chunks.accept(bytes, frame.timing.pts_ns);
                    self.drain(out).await?;
                }
                PipelinePacket::Flush => {
                    self.chunks.clear();
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos => {
                    let tail = self.chunks.take_pending();
                    if !tail.is_empty() {
                        self.chunks.emit(tail, out).await?;
                    }
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        RNDBUFFERSIZE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "min" => {
                self.min = size_bytes(&value)?;
                Ok(())
            }
            "max" => {
                let bytes = size_bytes(&value)?;
                if bytes == 0 {
                    return Err(PropError::Value);
                }
                self.max = bytes;
                Ok(())
            }
            "seed" => {
                let seed = value.as_uint().ok_or(PropError::Type)?;
                self.seed = u32::try_from(seed).map_err(|_| PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "min" => Some(PropValue::Uint(self.min)),
            "max" => Some(PropValue::Uint(self.max)),
            "seed" => Some(PropValue::Uint(u64::from(self.seed))),
            _ => None,
        }
    }
}

/// A `min` / `max` property value, bounded as gst bounds them.
fn size_bytes(value: &PropValue) -> Result<u64, PropError> {
    let bytes = value.as_uint().ok_or(PropError::Type)?;
    if bytes > MAX_SIZE_BYTES {
        return Err(PropError::Value);
    }
    Ok(bytes)
}

/// `RndBufferSize`'s settable properties, named and defaulted as gst
/// `rndbuffersize`.
static RNDBUFFERSIZE_PROPS: &[PropertySpec] = &[
    PropertySpec::new("min", PropKind::Uint, "smallest buffer emitted, in bytes")
        .with_range("0", "2147483647")
        .with_default("1"),
    PropertySpec::new("max", PropKind::Uint, "largest buffer emitted, in bytes")
        .with_range("1", "2147483647")
        .with_default("8192"),
    PropertySpec::new("seed", PropKind::Uint, "seed of the size generator")
        .with_range("0", "4294967295")
        .with_default("0"),
];
