//! Chop my data: re-chunks a byte stream into buffers whose length is a random
//! multiple of `step-size` inside `[min-size, max-size]`, the gst `chopmydata`
//! analog. Like [`rndbuffersize`](crate::rndbuffersize) it exists to prove a
//! parser downstream does not depend on where its input happens to be cut; the
//! step keeps every cut on a alignment boundary, which is what a parser reading
//! fixed-width units cares about.
//!
//! At `Eos` the bytes held back go out as whole `min-size` buffers and whatever
//! is left below that is dropped, as gst does. gst leaves those tail buffers
//! unstamped; here they keep the pts of the input they were cut from, like every
//! other chunk.
//!
//! gst draws its sizes from an unseeded generator and exposes no `seed`, so this
//! has none either: the sequence starts from the crate's fixed base state and
//! replays the same cut points every run.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::random::{next_random, XORSHIFT_BASE_STATE};
use crate::rechunk::Rechunker;

/// gst `chopmydata`'s `min-size` default.
const DEFAULT_MIN_SIZE: i64 = 1;
/// gst `chopmydata`'s `max-size` default.
const DEFAULT_MAX_SIZE: i64 = 4096;
/// gst `chopmydata`'s `step-size` default.
const DEFAULT_STEP_SIZE: i64 = 1;
/// Smallest and largest size accepted, gst's own bounds (its three sizes are
/// signed 32-bit ints, and none of them may be zero).
const MIN_SIZE: i64 = 1;
const MAX_SIZE: i64 = i32::MAX as i64;

/// # Example
///
/// ```no_run
/// use g2g_plugins::chopmydata::ChopMyData;
///
/// // gst-launch equivalent: chopmydata min-size=64 max-size=256 step-size=64
/// let element = ChopMyData::new();
/// ```
#[derive(Debug)]
pub struct ChopMyData {
    min_size: i64,
    max_size: i64,
    step_size: i64,
    /// The size generator, restarted at each configure so a run replays.
    state: u32,
    chunks: Rechunker,
    configured: bool,
}

impl Default for ChopMyData {
    fn default() -> Self {
        Self::new()
    }
}

impl ChopMyData {
    pub fn new() -> Self {
        Self {
            min_size: DEFAULT_MIN_SIZE,
            max_size: DEFAULT_MAX_SIZE,
            step_size: DEFAULT_STEP_SIZE,
            state: XORSHIFT_BASE_STATE,
            chunks: Rechunker::new(),
            configured: false,
        }
    }

    /// Buffers emitted so far.
    pub fn emitted(&self) -> u64 {
        self.chunks.emitted()
    }

    /// The half-open range of step counts a chunk length is drawn from, gst's
    /// `[ceil(min/step), (max+step)/step)`. An empty range (`min` above `max`)
    /// pins every chunk at `begin` steps, as gst's does.
    fn step_range(&self) -> (u64, u64) {
        let step = self.step_size as u64;
        let begin = (self.min_size as u64).div_ceil(step);
        let end = (self.max_size as u64 + step) / step;
        (begin, end)
    }

    /// Cut every whole chunk the pending bytes hold and push it.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let (begin, end) = self.step_range();
        let step = self.step_size as u64;
        let state = &mut self.state;
        self.chunks
            .drain(
                || {
                    let steps = if begin >= end {
                        begin
                    } else {
                        begin + u64::from(next_random(state)) % (end - begin)
                    };
                    (steps * step) as usize
                },
                out,
            )
            .await
    }
}

impl AsyncElement for ChopMyData {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ChopMyData",
            "Generic",
            "Re-chunks a byte stream into step-aligned random buffer sizes",
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
        self.state = XORSHIFT_BASE_STATE;
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
                    // gst empties its adapter at `min-size` and drops the
                    // remainder rather than pushing a short buffer.
                    self.chunks.clear_target();
                    let min = self.min_size as usize;
                    self.chunks.drain(|| min, out).await?;
                    self.chunks.clear();
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        CHOPMYDATA_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let size = size_bytes(&value)?;
        match name {
            "min-size" => self.min_size = size,
            "max-size" => self.max_size = size,
            "step-size" => self.step_size = size,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "min-size" => Some(PropValue::Int(self.min_size)),
            "max-size" => Some(PropValue::Int(self.max_size)),
            "step-size" => Some(PropValue::Int(self.step_size)),
            _ => None,
        }
    }
}

/// One of the three size properties, bounded as gst bounds them.
fn size_bytes(value: &PropValue) -> Result<i64, PropError> {
    let bytes = value.as_int().ok_or(PropError::Type)?;
    if !(MIN_SIZE..=MAX_SIZE).contains(&bytes) {
        return Err(PropError::Value);
    }
    Ok(bytes)
}

/// `ChopMyData`'s settable properties, named and defaulted as gst `chopmydata`.
static CHOPMYDATA_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "min-size",
        PropKind::Int,
        "smallest buffer emitted, in bytes",
    )
    .with_range("1", "2147483647")
    .with_default("1"),
    PropertySpec::new(
        "max-size",
        PropKind::Int,
        "largest buffer emitted, in bytes",
    )
    .with_range("1", "2147483647")
    .with_default("4096"),
    PropertySpec::new(
        "step-size",
        PropKind::Int,
        "every buffer length is a multiple of this, in bytes",
    )
    .with_range("1", "2147483647")
    .with_default("1"),
];
