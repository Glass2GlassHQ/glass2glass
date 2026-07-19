//! SPI TFT display sink over the MIPI-DCS command set (ST7789 / ILI9341
//! family), generic over `embedded-hal` 1.0 [`SpiDevice`] + a data/command
//! [`OutputPin`], the standard 4-wire panel wiring. The element owns the real
//! driver logic: the init sequence, window addressing, and streaming RGBA ->
//! RGB565 conversion; a board supplies only its HAL's SPI device and GPIO.
//!
//! The pixel path is heap-free and panic-free by construction (fixed stack
//! chunk buffer, no slice indexing, checked geometry math), so the element
//! belongs to the same no-alloc subset the `g2g-noalloc` proofs cover, and a
//! frame of any size streams through a 128-byte buffer: an MCU never holds a
//! second copy of the framebuffer.
//!
//! For a panel too large to hold even one RGBA frame in a ring (a 240x240
//! module is 230 KB), [`SpiDisplaySink::with_stripe`] switches to banded
//! streaming: each frame is one `width x rows` horizontal band, written to the
//! next vertical sub-window, so the pipeline ring holds a single band and a
//! full refresh completes every `height / rows` frames.

use embedded_hal::delay::DelayNs;
use embedded_hal::digital::OutputPin;
use embedded_hal::spi::SpiDevice;
use g2g_core::error::{G2gError, HardwareError};
use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomain;
use g2g_core::StaticSink;

// MIPI-DCS opcodes shared by the ST7789 / ILI9341 controller family.
const SWRESET: u8 = 0x01;
const SLPOUT: u8 = 0x11;
const NORON: u8 = 0x13;
const INVOFF: u8 = 0x20;
const INVON: u8 = 0x21;
const DISPON: u8 = 0x29;
const CASET: u8 = 0x2A;
const RASET: u8 = 0x2B;
const RAMWR: u8 = 0x2C;
const MADCTL: u8 = 0x36;
const COLMOD: u8 = 0x3A;

/// COLMOD parameter: 16-bit RGB565 over the serial interface.
const COLMOD_RGB565: u8 = 0x55;

/// Pixels converted per SPI write burst; the whole element needs only this
/// 2-bytes-per-pixel stack buffer regardless of frame size.
const CHUNK_PX: usize = 64;

fn peripheral() -> G2gError {
    G2gError::Hardware(HardwareError::Peripheral)
}

/// A heap-free [`StaticSink`] presenting RGBA8888 frames on an SPI TFT panel
/// (ST7789 / ILI9341 family), converting to RGB565 big-endian on the fly.
///
/// Construct with [`st7789`](Self::st7789) or [`ili9341`](Self::ili9341) for
/// panel-appropriate defaults, call [`init`](Self::init) once (it needs a
/// [`DelayNs`] for the datasheet reset/wake delays), then feed it frames whose
/// payload is exactly `width * height * 4` RGBA bytes; anything else is a
/// [`G2gError::CapsMismatch`]. The SPI transfers are blocking (`embedded-hal`
/// 1.0's `SpiDevice`); an async-SPI variant can follow the same seam.
pub struct SpiDisplaySink<SPI, DC> {
    spi: SPI,
    dc: DC,
    width: u16,
    height: u16,
    x_offset: u16,
    y_offset: u16,
    madctl: u8,
    invert: bool,
    initialized: bool,
    /// Rows per incoming frame in banded streaming mode, or 0 for whole-frame
    /// mode. When non-zero each `consume` blits one `width x stripe_rows` band
    /// to the next vertical sub-window, so a panel is refreshed without ever
    /// holding a full framebuffer (240x240 RGBA = 230 KB, too big for an MCU
    /// ring; a 240x16 band is 15 KB). `stripe_rows` must divide `height`.
    stripe_rows: u16,
    /// Row of the next band to write (banded mode), wrapping at `height` so a
    /// full refresh completes every `height / stripe_rows` frames.
    y_cursor: u16,
}

impl<SPI: SpiDevice, DC: OutputPin> SpiDisplaySink<SPI, DC> {
    /// An ST7789 panel (`width` x `height` visible pixels). ST7789 modules run
    /// with display inversion on (the panel is normally-inverted IPS glass).
    pub fn st7789(spi: SPI, dc: DC, width: u16, height: u16) -> Self {
        Self {
            spi,
            dc,
            width,
            height,
            x_offset: 0,
            y_offset: 0,
            madctl: 0x00,
            invert: true,
            initialized: false,
            stripe_rows: 0,
            y_cursor: 0,
        }
    }

    /// An ILI9341 panel (`width` x `height` visible pixels), no inversion.
    pub fn ili9341(spi: SPI, dc: DC, width: u16, height: u16) -> Self {
        Self {
            invert: false,
            ..Self::st7789(spi, dc, width, height)
        }
    }

    /// Panel-window offset for glass smaller than the controller RAM (e.g. a
    /// 240x240 ST7789 module sitting at (0, 80) of the 240x320 RAM).
    pub fn with_offset(mut self, x: u16, y: u16) -> Self {
        self.x_offset = x;
        self.y_offset = y;
        self
    }

    /// Raw `MADCTL` value (orientation / RGB-BGR order), for rotated mounts.
    pub fn with_madctl(mut self, madctl: u8) -> Self {
        self.madctl = madctl;
        self
    }

    /// Stream the panel in horizontal bands of `rows` rows instead of one
    /// whole-frame blit: each `consume` takes a `width x rows` RGBA frame and
    /// writes it to the next vertical sub-window, advancing a cursor that wraps
    /// at `height`. This is the MCU-scale path for a large panel: the pipeline
    /// ring holds one band (`width * rows * 4` bytes), never the full
    /// framebuffer. `rows` should divide `height` evenly; a band that would
    /// run past the last row is rejected as [`G2gError::CapsMismatch`]. `rows`
    /// of 0 restores whole-frame mode.
    pub fn with_stripe(mut self, rows: u16) -> Self {
        self.stripe_rows = rows;
        self.y_cursor = 0;
        self
    }

    /// Wake and configure the panel: software reset, sleep-out, RGB565 pixel
    /// format, orientation, inversion, normal mode, display on. The delays are
    /// the ST7789/ILI9341 datasheet minima (120 ms out of reset / sleep).
    pub fn init(&mut self, delay: &mut impl DelayNs) -> Result<(), G2gError> {
        self.command(SWRESET, &[])?;
        delay.delay_ms(150);
        self.command(SLPOUT, &[])?;
        delay.delay_ms(120);
        self.command(COLMOD, &[COLMOD_RGB565])?;
        self.command(MADCTL, &[self.madctl])?;
        self.command(if self.invert { INVON } else { INVOFF }, &[])?;
        self.command(NORON, &[])?;
        delay.delay_ms(10);
        self.command(DISPON, &[])?;
        delay.delay_ms(100);
        self.initialized = true;
        Ok(())
    }

    /// Release the SPI device and D/C pin (e.g. to hand the bus elsewhere).
    pub fn free(self) -> (SPI, DC) {
        (self.spi, self.dc)
    }

    /// One DCS command: opcode with D/C low, then `params` with D/C high.
    fn command(&mut self, cmd: u8, params: &[u8]) -> Result<(), G2gError> {
        self.dc.set_low().map_err(|_| peripheral())?;
        self.spi.write(&[cmd]).map_err(|_| peripheral())?;
        if !params.is_empty() {
            self.dc.set_high().map_err(|_| peripheral())?;
            self.spi.write(params).map_err(|_| peripheral())?;
        }
        Ok(())
    }

    /// Pixel data continuing a `RAMWR`, D/C high.
    fn data(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
        self.dc.set_high().map_err(|_| peripheral())?;
        self.spi.write(bytes).map_err(|_| peripheral())?;
        Ok(())
    }

    /// Address a `width x rows` window starting `row0` rows below the panel's
    /// y offset (offsets applied) and open `RAMWR`. Whole-frame mode passes
    /// `(0, height)`; banded mode passes `(cursor, stripe_rows)`.
    fn set_window(&mut self, row0: u16, rows: u16) -> Result<(), G2gError> {
        // Inclusive end columns/rows; saturating so degenerate geometry cannot
        // overflow-panic (a mismatching frame is rejected before this anyway).
        let x0 = self.x_offset;
        let y0 = self.y_offset.saturating_add(row0);
        let x1 = x0.saturating_add(self.width).saturating_sub(1);
        let y1 = y0.saturating_add(rows).saturating_sub(1);
        let [x0h, x0l] = x0.to_be_bytes();
        let [x1h, x1l] = x1.to_be_bytes();
        let [y0h, y0l] = y0.to_be_bytes();
        let [y1h, y1l] = y1.to_be_bytes();
        self.command(CASET, &[x0h, x0l, x1h, x1l])?;
        self.command(RASET, &[y0h, y0l, y1h, y1l])?;
        self.command(RAMWR, &[])
    }

    /// Stream `rgba` (4 bytes per pixel) as big-endian RGB565, converting
    /// through the fixed [`CHUNK_PX`] stack buffer.
    fn write_pixels(&mut self, rgba: &[u8]) -> Result<(), G2gError> {
        let mut buf = [0u8; CHUNK_PX * 2];
        for block in rgba.chunks(CHUNK_PX * 4) {
            let mut used = 0usize;
            for (dst, src) in buf.chunks_exact_mut(2).zip(block.chunks_exact(4)) {
                // Slice patterns, not indexing: no bounds-check panic path may
                // enter the no-alloc subset (chunk sizes make these infallible).
                let (&[r, g, b, _], [hi, lo]) = (src, dst) else {
                    continue;
                };
                let v = ((r as u16 & 0xF8) << 8) | ((g as u16 & 0xFC) << 3) | (b as u16 >> 3);
                [*hi, *lo] = v.to_be_bytes();
                used = used.saturating_add(2);
            }
            let Some(out) = buf.get(..used) else { continue };
            self.data(out)?;
        }
        Ok(())
    }
}

impl<SPI: SpiDevice, DC: OutputPin> StaticSink for SpiDisplaySink<SPI, DC> {
    /// Blit one frame. In whole-frame mode the payload must be
    /// `width * height * 4` RGBA bytes; in banded mode (`with_stripe`) it must
    /// be `width * stripe_rows * 4` and is written to the next vertical band,
    /// advancing the cursor (wrapping at `height`). The payload lives in
    /// `MemoryDomain::System` (e.g. a `StaticLendRing` lend).
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
        if !self.initialized {
            return Err(G2gError::NotConfigured);
        }
        let MemoryDomain::System(slice) = &frame.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        let rgba = slice.as_slice();
        // Rows this frame carries, and where they land: the whole panel from
        // row 0, or one stripe at the running cursor.
        let (row0, rows) = if self.stripe_rows == 0 {
            (0, self.height)
        } else {
            (self.y_cursor, self.stripe_rows)
        };
        // A band that would run past the last row means the stripe does not
        // tile the panel; reject rather than address off-panel RAM.
        if row0.saturating_add(rows) > self.height {
            return Err(G2gError::CapsMismatch);
        }
        let expected = (self.width as usize)
            .checked_mul(rows as usize)
            .and_then(|px| px.checked_mul(4));
        if expected != Some(rgba.len()) {
            return Err(G2gError::CapsMismatch);
        }
        self.set_window(row0, rows)?;
        self.write_pixels(rgba)?;
        // Advance to the next band; wrap once the panel is full so the next
        // frame starts a fresh refresh from the top.
        if self.stripe_rows != 0 {
            self.y_cursor = self.y_cursor.saturating_add(rows);
            if self.y_cursor >= self.height {
                self.y_cursor = 0;
            }
        }
        Ok(())
    }
}

impl<SPI, DC> core::fmt::Debug for SpiDisplaySink<SPI, DC> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SpiDisplaySink")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("x_offset", &self.x_offset)
            .field("y_offset", &self.y_offset)
            .field("madctl", &self.madctl)
            .field("invert", &self.invert)
            .field("initialized", &self.initialized)
            .field("stripe_rows", &self.stripe_rows)
            .field("y_cursor", &self.y_cursor)
            .finish()
    }
}
