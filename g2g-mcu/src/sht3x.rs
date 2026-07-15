//! I2C environmental-sensor source: a real driver for the Sensirion SHT3x
//! (SHT30 / SHT31 / SHT35 temperature + humidity), over the portable
//! `embedded-hal` 1.0 [`I2c`] seam. It broadens the peripheral catalog from the
//! capture / display / audio elements into the I2C sensor / control plane a real
//! product also needs.
//!
//! The driver logic is real and datasheet-anchored, only the bus is mocked in
//! tests: the single-shot measurement command (clock-stretching, high
//! repeatability, `0x2C06`), the CRC-8 the sensor appends to each 16-bit word
//! (polynomial `0x31`, init `0xFF`, the datasheet's checked example `0xBEEF ->
//! 0x92`), and the fixed-point conversion from the datasheet's transfer
//! functions (`T = -45 + 175 * S/2^16`, `RH = 100 * S/2^16`) to milli-units.
//! A word whose CRC does not match is rejected as a bus-integrity fault rather
//! than trusted (the parser-discipline rule applied to a sensor).

use embedded_hal::i2c::I2c;
use g2g_core::error::{G2gError, HardwareError};
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticSource;

use crate::lend::lend_slot;

/// A bus / sensor-integrity fault (an I2C error, or a CRC that does not match
/// the sensor's data, so the reading cannot be trusted).
fn peripheral() -> G2gError {
    G2gError::Hardware(HardwareError::Peripheral)
}

/// The SHT3x default I2C address (ADDR pin low). The alternate (ADDR high) is
/// `0x45`; a board picks one at construction.
pub const SHT3X_ADDR_DEFAULT: u8 = 0x44;

/// One reading's payload byte length: temperature then humidity, each a
/// little-endian `i32` in milli-units (m°C and m%RH).
pub const SHT3X_READING_BYTES: usize = 8;

/// Single-shot measurement, clock stretching enabled, high repeatability
/// (datasheet command `0x2C06`): the sensor holds SCL until the conversion is
/// done, so one `write_read` returns the result with no host-side delay.
const CMD_SINGLE_SHOT_HIGH: [u8; 2] = [0x2C, 0x06];

/// CRC-8 over `data` as the SHT3x specifies: polynomial `0x31` (representing
/// `x^8 + x^5 + x^4 + x^0`), initialization `0xFF`, no reflection, no final XOR.
/// The datasheet's worked example is `crc8(&[0xBE, 0xEF]) == 0x92`.
pub fn crc8(data: &[u8]) -> u8 {
    let mut crc: u8 = 0xFF;
    for &byte in data {
        crc ^= byte;
        for _ in 0..8 {
            if crc & 0x80 != 0 {
                crc = (crc << 1) ^ 0x31;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// Datasheet temperature transfer function in milli-degrees Celsius:
/// `T = -45 + 175 * raw / (2^16 - 1)`. The multiply is widened to `i64` because
/// `175000 * 65535` overflows `i32`; the result (-45000..=130000) fits `i32`.
pub const fn raw_to_millicelsius(raw: u16) -> i32 {
    (-45_000_i64 + (175_000_i64 * raw as i64) / 65_535) as i32
}

/// Datasheet humidity transfer function in milli-percent RH:
/// `RH = 100 * raw / (2^16 - 1)`. Widened to `i64` for the same reason; the
/// result (0..=100000) fits `i32`.
pub const fn raw_to_milli_rh(raw: u16) -> i32 {
    ((100_000_i64 * raw as i64) / 65_535) as i32
}

/// A heap-free SHT3x sensor [`StaticSource`]: each `next` issues one measurement,
/// validates both CRC-8 check bytes, converts to milli-units, and lends an
/// [`SHT3X_READING_BYTES`]-byte reading downstream (`i32` m°C then `i32` m%RH,
/// little-endian) from a [`StaticLendRing`].
pub struct Sht3xSrc<'r, I2C, const N: usize, const BYTES: usize> {
    i2c: I2C,
    addr: u8,
    ring: &'r StaticLendRing<N, BYTES>,
    frame_interval_ns: u64,
    remaining: Option<u32>,
    seq: u64,
}

impl<I2C: I2c, const N: usize, const BYTES: usize> Sht3xSrc<'static, I2C, N, BYTES> {
    /// A sensor source over a `'static` ring (the MCU idiom), which makes the
    /// zero-copy lend sound by construction. `frame_interval_ns` is the nominal
    /// sample period used to derive PTS.
    pub fn new(
        i2c: I2C,
        addr: u8,
        ring: &'static StaticLendRing<N, BYTES>,
        frame_interval_ns: u64,
    ) -> Self {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(i2c, addr, ring, frame_interval_ns) }
    }
}

impl<'r, I2C: I2c, const N: usize, const BYTES: usize> Sht3xSrc<'r, I2C, N, BYTES> {
    /// A sensor source over a borrowed ring.
    ///
    /// # Safety
    /// The ring must outlive every frame this source publishes (the
    /// [`RingSlot::publish`](g2g_core::staticpool::RingSlot::publish) contract).
    pub unsafe fn with_ring(
        i2c: I2C,
        addr: u8,
        ring: &'r StaticLendRing<N, BYTES>,
        frame_interval_ns: u64,
    ) -> Self {
        Self { i2c, addr, ring, frame_interval_ns, remaining: None, seq: 0 }
    }

    /// End the stream after `frames` readings (a sensor is polled endlessly by
    /// default; proofs and tests bound it).
    pub fn with_frame_limit(mut self, frames: u32) -> Self {
        self.remaining = Some(frames);
        self
    }

    /// Release the I2C bus (e.g. to share it with another device).
    pub fn free(self) -> I2C {
        self.i2c
    }
}

impl<I2C: I2c, const N: usize, const BYTES: usize> StaticSource for Sht3xSrc<'_, I2C, N, BYTES> {
    async fn next(&mut self) -> Result<Option<Frame>, G2gError> {
        if self.remaining == Some(0) {
            return Ok(None);
        }
        // Read the 6-byte result: [T_msb, T_lsb, T_crc, RH_msb, RH_lsb, RH_crc].
        let mut buf = [0u8; 6];
        self.i2c.write_read(self.addr, &CMD_SINGLE_SHOT_HIGH, &mut buf).map_err(|_| peripheral())?;
        // Validate both CRC-8 check bytes before trusting the reading.
        let (Some(t_word), Some(t_crc), Some(rh_word), Some(rh_crc)) =
            (buf.get(0..2), buf.get(2).copied(), buf.get(3..5), buf.get(5).copied())
        else {
            return Err(peripheral());
        };
        if crc8(t_word) != t_crc || crc8(rh_word) != rh_crc {
            return Err(peripheral());
        }
        let raw_t = u16::from_be_bytes([t_word[0], t_word[1]]);
        let raw_rh = u16::from_be_bytes([rh_word[0], rh_word[1]]);
        let t_mc = raw_to_millicelsius(raw_t);
        let rh_mpct = raw_to_milli_rh(raw_rh);

        let pts_ns = self.seq.saturating_mul(self.frame_interval_ns);
        // SAFETY: the constructor established the ring-outlives-frames contract
        // (`new`: 'static; `with_ring`: caller's contract).
        let frame = unsafe {
            lend_slot(
                self.ring,
                FrameTiming { pts_ns, ..FrameTiming::default() },
                self.seq,
                SHT3X_READING_BYTES,
                |dst| {
                    if let Some(t) = dst.get_mut(0..4) {
                        t.copy_from_slice(&t_mc.to_le_bytes());
                    }
                    if let Some(rh) = dst.get_mut(4..8) {
                        rh.copy_from_slice(&rh_mpct.to_le_bytes());
                    }
                },
            )?
        };
        if let Some(remaining) = &mut self.remaining {
            *remaining -= 1;
        }
        self.seq += 1;
        Ok(Some(frame))
    }
}

impl<I2C: I2c, const N: usize, const BYTES: usize> g2g_core::supervise::Recover
    for Sht3xSrc<'_, I2C, N, BYTES>
{
    /// The default no-op recover: a soft-reset command sequence could be issued
    /// here, but the single-shot read is self-contained (each `next` is a fresh
    /// transaction), so there is no latched state to clear.
    async fn recover(&mut self) -> Result<(), G2gError> {
        Ok(())
    }
}

impl<I2C, const N: usize, const BYTES: usize> core::fmt::Debug for Sht3xSrc<'_, I2C, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Sht3xSrc")
            .field("addr", &self.addr)
            .field("slots", &N)
            .field("slot_bytes", &BYTES)
            .field("seq", &self.seq)
            .finish_non_exhaustive()
    }
}
