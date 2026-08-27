//! Break my data: overwrites bytes at random as they pass, the gst
//! `breakmydata` analog. Put it ahead of a parser or a decoder to prove that
//! corrupt input fails the parse instead of panicking.
//!
//! Every byte past the first `skip` of the stream draws once from a seeded
//! xorshift; a draw at or below `probability` replaces that byte with `set-to`,
//! or with a second draw when `set-to` is `-1`. `skip` counts from the start of
//! the stream, not per buffer, so the first `skip` bytes (a container header,
//! say) always arrive intact.
//!
//! Timing and sequence are untouched: only the payload changes, so a corrupted
//! stream stays the same shape as the one it came from.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, MemoryDomain,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::random::{next_random, next_unit, XORSHIFT_BASE_STATE};

/// `set-to = -1` draws a random replacement byte instead of a fixed one.
const SET_TO_RANDOM: i64 = -1;
/// gst's `set-to` bound: a byte value, or [`SET_TO_RANDOM`].
const SET_TO_MAX: i64 = u8::MAX as i64;
/// gst's `skip` / `seed` bound (both are 32-bit unsigned).
const MAX_U32: u64 = u32::MAX as u64;

/// # Example
///
/// ```no_run
/// use g2g_plugins::breakmydata::BreakMyData;
///
/// // gst-launch equivalent: breakmydata probability=0.01 skip=64 seed=7
/// let element = BreakMyData::new();
/// ```
#[derive(Debug)]
pub struct BreakMyData {
    probability: f64,
    seed: u32,
    set_to: i64,
    skip: u64,
    /// The generator, restarted from `seed` at each configure. gst builds its
    /// own on the READY to PAUSED transition, so a seed set mid-run does not
    /// take effect until the pipeline restarts either.
    state: u32,
    /// Bytes seen so far, counted against `skip` across the whole stream.
    skipped: u64,
    /// Bytes actually overwritten, for a test to check the seed replays.
    corrupted: u64,
    configured: bool,
}

impl Default for BreakMyData {
    fn default() -> Self {
        Self::new()
    }
}

impl BreakMyData {
    pub fn new() -> Self {
        Self {
            probability: 0.0,
            seed: 0,
            set_to: SET_TO_RANDOM,
            skip: 0,
            state: XORSHIFT_BASE_STATE,
            skipped: 0,
            corrupted: 0,
            configured: false,
        }
    }

    /// Bytes overwritten so far.
    pub fn corrupted(&self) -> u64 {
        self.corrupted
    }

    /// Overwrite `bytes` in place, from wherever the stream-wide skip debt ends.
    fn corrupt(&mut self, bytes: &mut [u8]) {
        let start = self
            .skip
            .saturating_sub(self.skipped)
            .min(bytes.len() as u64) as usize;
        for byte in &mut bytes[start..] {
            if next_unit(&mut self.state) <= self.probability {
                *byte = match self.set_to {
                    SET_TO_RANDOM => (next_random(&mut self.state) % 256) as u8,
                    set => set as u8,
                };
                self.corrupted += 1;
            }
        }
        self.skipped = self.skipped.saturating_add(bytes.len() as u64);
    }
}

impl AsyncElement for BreakMyData {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "BreakMyData",
            "Generic",
            "Overwrites bytes at random to exercise error handling downstream",
            "g2g",
        )
    }

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Corruption changes bytes, not shape, so input and output caps are equal
    /// whatever the media type is.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.state = XORSHIFT_BASE_STATE ^ self.seed;
        self.skipped = 0;
        self.corrupted = 0;
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
                    let mut bytes: Vec<u8> = frame
                        .domain
                        .as_system_slice()
                        .ok_or(G2gError::UnsupportedDomain)?
                        .to_vec();
                    self.corrupt(&mut bytes);
                    let broken = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                        frame.timing,
                        frame.sequence,
                    );
                    out.push(PipelinePacket::DataFrame(broken)).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        BREAKMYDATA_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "probability" => {
                let p = value.as_double().ok_or(PropError::Type)?;
                if !(0.0..=1.0).contains(&p) {
                    return Err(PropError::Value);
                }
                self.probability = p;
                Ok(())
            }
            "seed" => {
                let seed = value.as_uint().ok_or(PropError::Type)?;
                self.seed = u32::try_from(seed).map_err(|_| PropError::Value)?;
                Ok(())
            }
            "set-to" => {
                let set = value.as_int().ok_or(PropError::Type)?;
                if !(SET_TO_RANDOM..=SET_TO_MAX).contains(&set) {
                    return Err(PropError::Value);
                }
                self.set_to = set;
                Ok(())
            }
            "skip" => {
                let skip = value.as_uint().ok_or(PropError::Type)?;
                if skip > MAX_U32 {
                    return Err(PropError::Value);
                }
                self.skip = skip;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "probability" => Some(PropValue::Double(self.probability)),
            "seed" => Some(PropValue::Uint(u64::from(self.seed))),
            "set-to" => Some(PropValue::Int(self.set_to)),
            "skip" => Some(PropValue::Uint(self.skip)),
            _ => None,
        }
    }
}

/// `BreakMyData`'s settable properties, named and defaulted as gst
/// `breakmydata`.
static BREAKMYDATA_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "probability",
        PropKind::Double,
        "chance each byte is overwritten",
    )
    .with_range("0", "1")
    .with_default("0"),
    PropertySpec::new("seed", PropKind::Uint, "seed of the corruption generator")
        .with_range("0", "4294967295")
        .with_default("0"),
    PropertySpec::new(
        "set-to",
        PropKind::Int,
        "value written over a broken byte, -1 for a random one",
    )
    .with_range("-1", "255")
    .with_default("-1"),
    PropertySpec::new(
        "skip",
        PropKind::Uint,
        "bytes left intact at the start of the stream",
    )
    .with_range("0", "4294967295")
    .with_default("0"),
];
