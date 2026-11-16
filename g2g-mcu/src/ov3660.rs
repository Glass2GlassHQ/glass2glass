// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
// Copyright 2015-2026 Espressif Systems (Shanghai) PTE LTD (esp32-camera)
//
// SPDX-License-Identifier: MPL-2.0 AND Apache-2.0

use embedded_hal::delay::DelayNs;
use embedded_hal::i2c::I2c;
use g2g_core::error::G2gError;

use crate::peripheral;

pub const OV3660_ADDRESS: u8 = 0x3c;
pub const OV3660_96X96_GRAYSCALE_BYTES: usize = 96 * 96;

const PRODUCT_ID: u16 = 0x300a;
const SYSTEM_CONTROL0: u16 = 0x3008;
const X_ADDRESS_START: u16 = 0x3800;
const X_ADDRESS_END: u16 = 0x3804;
const X_OUTPUT_SIZE: u16 = 0x3808;
const X_TOTAL_SIZE: u16 = 0x380c;
const X_OFFSET: u16 = 0x3810;
const ISP_CONTROL_01: u16 = 0x5001;
const TIMING_TC_REG20: u16 = 0x3820;
const TIMING_TC_REG21: u16 = 0x3821;
const X_INCREMENT: u16 = 0x3814;
const Y_INCREMENT: u16 = 0x3815;
const PLL_CONTROL0: u16 = 0x303a;
const PLL_CONTROL1: u16 = 0x303b;
const PLL_CONTROL2: u16 = 0x303c;
const PLL_CONTROL3: u16 = 0x303d;
const PCLK_RATIO: u16 = 0x3824;
const VFIFO_CONTROL0C: u16 = 0x460c;
const DELAY_ENTRY: u16 = 0xffff;
const OV3660_PRODUCT_ID: u16 = 0x3660;
const RESET_SETTLE_MILLISECONDS: u32 = 100;
const SOFT_RESET: u8 = 0x82;
const WIDTH: u16 = 96;
const HEIGHT: u16 = 96;

// esp32-camera's 1:1 window: start, end and total size in sensor pixels.
const WINDOW_START: (u16, u16) = (256, 0);
const WINDOW_END: (u16, u16) = (1823, 1547);
const WINDOW_TOTAL: (u16, u16) = (2044, 1564);
// 96x96 is under half the window, so esp32-camera bins 2x2 and scales.
const BINNED_TOTAL_Y: u16 = WINDOW_TOTAL.1 / 2 + 1;
const BINNED_OFFSET: (u16, u16) = (8, 2);
const ISP_SCALE_ENABLE: u8 = 0x20;
const TIMING_BINNING: u8 = 0x01;
const TIMING_4514_BINNING: u8 = 0xaa;
const TIMING_4520_BINNING: u8 = 0x0b;
const SKIP_ODD_3_EVEN_1: u8 = 0x31;
// esp32-camera's PLL below QVGA: multiplier 8, system divider 1, PCLK divider 8, manual PCLK.
const PLL_MULTIPLIER: u8 = 8;
const PLL_SYSTEM_DIVIDER: u8 = 0x10 | 1;
const PCLK_DIVIDER: u8 = 8;
const VFIFO_MANUAL_PCLK: u8 = 0x22;
// set_ae_level(0): target 55 with a +-8% stable band and a fast band twice as wide.
const AE_TARGET: u16 = 55;

const DEFAULTS: &[[u16; 2]] = &[
    [0x3008, 0x82],
    [0xffff, 0x0a],
    [0x3103, 0x13],
    [0x3008, 0x42],
    [0x3017, 0xff],
    [0x3018, 0xff],
    [0x302c, 0xc3],
    [0x4740, 0x21],
    [0x3611, 0x01],
    [0x3612, 0x2d],
    [0x3032, 0x00],
    [0x3614, 0x80],
    [0x3618, 0x00],
    [0x3619, 0x75],
    [0x3622, 0x80],
    [0x3623, 0x00],
    [0x3624, 0x03],
    [0x3630, 0x52],
    [0x3632, 0x07],
    [0x3633, 0xd2],
    [0x3704, 0x80],
    [0x3708, 0x66],
    [0x3709, 0x12],
    [0x370b, 0x12],
    [0x3717, 0x00],
    [0x371b, 0x60],
    [0x371c, 0x00],
    [0x3901, 0x13],
    [0x3600, 0x08],
    [0x3620, 0x43],
    [0x3702, 0x20],
    [0x3739, 0x48],
    [0x3730, 0x20],
    [0x370c, 0x0c],
    [0x3a18, 0x00],
    [0x3a19, 0xf8],
    [0x3000, 0x10],
    [0x3004, 0xef],
    [0x6700, 0x05],
    [0x6701, 0x19],
    [0x6702, 0xfd],
    [0x6703, 0xd1],
    [0x6704, 0xff],
    [0x6705, 0xff],
    [0x3c01, 0x80],
    [0x3c00, 0x04],
    [0x3a08, 0x00],
    [0x3a09, 0x62],
    [0x3a0e, 0x08],
    [0x3a0a, 0x00],
    [0x3a0b, 0x52],
    [0x3a0d, 0x09],
    [0x3a00, 0x3a],
    [0x3a14, 0x09],
    [0x3a15, 0x30],
    [0x3a02, 0x09],
    [0x3a03, 0x30],
    [0x440e, 0x08],
    [0x4520, 0x0b],
    [0x460b, 0x37],
    [0x4713, 0x02],
    [0x471c, 0xd0],
    [0x5086, 0x00],
    [0x5002, 0x00],
    [0x501f, 0x00],
    [0x3008, 0x02],
    [0x5180, 0xff],
    [0x5181, 0xf2],
    [0x5182, 0x00],
    [0x5183, 0x14],
    [0x5184, 0x25],
    [0x5185, 0x24],
    [0x5186, 0x16],
    [0x5187, 0x16],
    [0x5188, 0x16],
    [0x5189, 0x68],
    [0x518a, 0x60],
    [0x518b, 0xe0],
    [0x518c, 0xb2],
    [0x518d, 0x42],
    [0x518e, 0x35],
    [0x518f, 0x56],
    [0x5190, 0x56],
    [0x5191, 0xf8],
    [0x5192, 0x04],
    [0x5193, 0x70],
    [0x5194, 0xf0],
    [0x5195, 0xf0],
    [0x5196, 0x03],
    [0x5197, 0x01],
    [0x5198, 0x04],
    [0x5199, 0x12],
    [0x519a, 0x04],
    [0x519b, 0x00],
    [0x519c, 0x06],
    [0x519d, 0x82],
    [0x519e, 0x38],
    [0x5381, 0x1d],
    [0x5382, 0x60],
    [0x5383, 0x03],
    [0x5384, 0x0c],
    [0x5385, 0x78],
    [0x5386, 0x84],
    [0x5387, 0x7d],
    [0x5388, 0x6b],
    [0x5389, 0x12],
    [0x538a, 0x01],
    [0x538b, 0x98],
    [0x5480, 0x01],
    [0x5000, 0xa7],
    [0x5800, 0x0c],
    [0x5801, 0x09],
    [0x5802, 0x0c],
    [0x5803, 0x0c],
    [0x5804, 0x0d],
    [0x5805, 0x17],
    [0x5806, 0x06],
    [0x5807, 0x05],
    [0x5808, 0x04],
    [0x5809, 0x06],
    [0x580a, 0x09],
    [0x580b, 0x0e],
    [0x580c, 0x05],
    [0x580d, 0x01],
    [0x580e, 0x01],
    [0x580f, 0x01],
    [0x5810, 0x05],
    [0x5811, 0x0d],
    [0x5812, 0x05],
    [0x5813, 0x01],
    [0x5814, 0x01],
    [0x5815, 0x01],
    [0x5816, 0x05],
    [0x5817, 0x0d],
    [0x5818, 0x08],
    [0x5819, 0x06],
    [0x581a, 0x05],
    [0x581b, 0x07],
    [0x581c, 0x0b],
    [0x581d, 0x0d],
    [0x581e, 0x12],
    [0x581f, 0x0d],
    [0x5820, 0x0e],
    [0x5821, 0x10],
    [0x5822, 0x10],
    [0x5823, 0x1e],
    [0x5824, 0x53],
    [0x5825, 0x15],
    [0x5826, 0x05],
    [0x5827, 0x14],
    [0x5828, 0x54],
    [0x5829, 0x25],
    [0x582a, 0x33],
    [0x582b, 0x33],
    [0x582c, 0x34],
    [0x582d, 0x16],
    [0x582e, 0x24],
    [0x582f, 0x41],
    [0x5830, 0x50],
    [0x5831, 0x42],
    [0x5832, 0x15],
    [0x5833, 0x25],
    [0x5834, 0x34],
    [0x5835, 0x33],
    [0x5836, 0x24],
    [0x5837, 0x26],
    [0x5838, 0x54],
    [0x5839, 0x25],
    [0x583a, 0x15],
    [0x583b, 0x25],
    [0x583c, 0x53],
    [0x583d, 0xcf],
    [0x3a0f, 0x30],
    [0x3a10, 0x28],
    [0x3a1b, 0x30],
    [0x3a1e, 0x28],
    [0x3a11, 0x60],
    [0x3a1f, 0x14],
    [0x5302, 0x28],
    [0x5303, 0x20],
    [0x5306, 0x1c],
    [0x5307, 0x28],
    [0x4002, 0xc5],
    [0x4003, 0x81],
    [0x4005, 0x12],
    [0x5688, 0x11],
    [0x5689, 0x11],
    [0x568a, 0x11],
    [0x568b, 0x11],
    [0x568c, 0x11],
    [0x568d, 0x11],
    [0x568e, 0x11],
    [0x568f, 0x11],
    [0x5580, 0x06],
    [0x5588, 0x00],
    [0x5583, 0x40],
    [0x5584, 0x2c],
    [0x5001, 0x83],
];
const GRAYSCALE: &[[u16; 2]] = &[[0x501f, 0x00], [0x4300, 0x10]];

pub struct Ov3660<I2C> {
    i2c: I2C,
}

impl<I2C: I2c> Ov3660<I2C> {
    pub fn new(i2c: I2C) -> Self {
        Self { i2c }
    }

    pub fn free(self) -> I2C {
        self.i2c
    }

    fn write(&mut self, register: u16, value: u8) -> Result<(), G2gError> {
        let [high, low] = register.to_be_bytes();
        self.i2c
            .write(OV3660_ADDRESS, &[high, low, value])
            .map_err(|_| peripheral())
    }

    fn read(&mut self, register: u16) -> Result<u8, G2gError> {
        let mut value = [0u8];
        self.i2c
            .write_read(OV3660_ADDRESS, &register.to_be_bytes(), &mut value)
            .map_err(|_| peripheral())?;
        Ok(value[0])
    }

    fn write_pair(&mut self, register: u16, (x, y): (u16, u16)) -> Result<(), G2gError> {
        for (offset, value) in [(0, x), (2, y)] {
            let [high, low] = value.to_be_bytes();
            self.write(register + offset, high)?;
            self.write(register + offset + 1, low)?;
        }
        Ok(())
    }

    fn write_table(
        &mut self,
        table: &[[u16; 2]],
        delay: &mut impl DelayNs,
    ) -> Result<(), G2gError> {
        for &[register, value] in table {
            if register == DELAY_ENTRY {
                delay.delay_ms(value as u32);
            } else {
                self.write(register, value as u8)?;
            }
        }
        Ok(())
    }

    // The register sequence esp32-camera's esp_camera_init runs for 96x96 grayscale.
    pub fn configure_grayscale_96x96(&mut self, delay: &mut impl DelayNs) -> Result<(), G2gError> {
        let product_id = u16::from_be_bytes([self.read(PRODUCT_ID)?, self.read(PRODUCT_ID + 1)?]);
        if product_id != OV3660_PRODUCT_ID {
            return Err(peripheral());
        }

        self.write(SYSTEM_CONTROL0, SOFT_RESET)?;
        delay.delay_ms(RESET_SETTLE_MILLISECONDS);
        self.write_table(DEFAULTS, delay)?;
        let (low, high) = (AE_TARGET * 23 / 25, AE_TARGET * 27 / 25);
        for (register, value) in [
            (0x3a0f, high),
            (0x3a10, low),
            (0x3a1b, high),
            (0x3a1e, low),
            (0x3a11, (high << 1).min(0xff)),
            (0x3a1f, low >> 1),
        ] {
            self.write(register, value as u8)?;
        }
        delay.delay_ms(RESET_SETTLE_MILLISECONDS);

        self.write_pair(X_ADDRESS_START, WINDOW_START)?;
        self.write_pair(X_ADDRESS_END, WINDOW_END)?;
        self.write_pair(X_OUTPUT_SIZE, (WIDTH, HEIGHT))?;
        self.write_pair(X_TOTAL_SIZE, (WINDOW_TOTAL.0, BINNED_TOTAL_Y))?;
        self.write_pair(X_OFFSET, BINNED_OFFSET)?;
        let isp = self.read(ISP_CONTROL_01)?;
        self.write(ISP_CONTROL_01, isp | ISP_SCALE_ENABLE)?;
        self.write(TIMING_TC_REG20, TIMING_BINNING)?;
        self.write(TIMING_TC_REG21, TIMING_BINNING)?;
        self.write(0x4514, TIMING_4514_BINNING)?;
        self.write(0x4520, TIMING_4520_BINNING)?;
        self.write(X_INCREMENT, SKIP_ODD_3_EVEN_1)?;
        self.write(Y_INCREMENT, SKIP_ODD_3_EVEN_1)?;
        self.write(PLL_CONTROL0, 0x00)?;
        self.write(PLL_CONTROL1, PLL_MULTIPLIER)?;
        self.write(PLL_CONTROL2, PLL_SYSTEM_DIVIDER)?;
        self.write(PLL_CONTROL3, 0x00)?;
        self.write(PCLK_RATIO, PCLK_DIVIDER)?;
        self.write(VFIFO_CONTROL0C, VFIFO_MANUAL_PCLK)?;

        self.write_table(GRAYSCALE, delay)
    }
}

impl<I2C> core::fmt::Debug for Ov3660<I2C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Ov3660").finish_non_exhaustive()
    }
}
