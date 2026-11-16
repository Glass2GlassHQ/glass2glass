// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use esp_hal::delay::Delay;
use esp_hal::dma::DmaRxStreamBuf;
use esp_hal::dma_rx_stream_buffer;
use esp_hal::i2c::master::{Config as I2cConfig, I2c};
use esp_hal::lcd_cam::cam::{Camera, CameraTransfer, Config as CameraConfig};
use esp_hal::lcd_cam::LcdCam;
use esp_hal::peripherals::{
    DMA_CH1, GPIO10, GPIO11, GPIO12, GPIO13, GPIO14, GPIO15, GPIO16, GPIO17, GPIO18, GPIO38,
    GPIO39, GPIO40, GPIO47, GPIO48, I2C0, LCD_CAM,
};
use esp_hal::time::Rate;
use g2g_core::error::G2gError;
use g2g_mcu::{FrameGrabber, Ov3660};

use crate::peripheral;
use crate::person::IMAGE_BYTES;

const MASTER_CLOCK_MHZ: u32 = 20;
const SCCB_KHZ: u32 = 100;
// three frames of the OV3660's 10243 bytes per 96x96 frame, which covers both
// models blocking back to back between drains
const DMA_BYTES: usize = 32 * DMA_CHUNK_BYTES;
// each frame starts a new DMA descriptor
const DMA_CHUNK_BYTES: usize = 1024;

pub struct CameraPins {
    // the XIAO ESP32S3 Sense connector, from arduino-esp32's camera_pins.h
    pub xclk: GPIO10<'static>,
    pub sda: GPIO40<'static>,
    pub scl: GPIO39<'static>,
    pub d0: GPIO15<'static>,
    pub d1: GPIO17<'static>,
    pub d2: GPIO18<'static>,
    pub d3: GPIO16<'static>,
    pub d4: GPIO14<'static>,
    pub d5: GPIO12<'static>,
    pub d6: GPIO11<'static>,
    pub d7: GPIO48<'static>,
    pub vsync: GPIO38<'static>,
    pub href: GPIO47<'static>,
    pub pclk: GPIO13<'static>,
}

pub struct CameraGrabber {
    transfer: Option<CameraTransfer<'static, DmaRxStreamBuf>>,
    // the OV3660 sends blank rows after the first 96 of each frame
    frame: &'static mut [u8; IMAGE_BYTES],
    frame_bytes: usize,
    // the frame being assembled began right after an end of frame
    aligned: bool,
}

pub fn open(
    lcd_cam: LCD_CAM<'static>,
    dma: DMA_CH1<'static>,
    i2c0: I2C0<'static>,
    pins: CameraPins,
    frame: &'static mut [u8; IMAGE_BYTES],
) -> Result<CameraGrabber, G2gError> {
    let lcd_cam = LcdCam::new(lcd_cam);
    let config = CameraConfig::default().with_frequency(Rate::from_mhz(MASTER_CLOCK_MHZ));
    let camera = Camera::new(lcd_cam.cam, dma, config)
        .map_err(|_| peripheral())?
        .with_master_clock(pins.xclk)
        .with_pixel_clock(pins.pclk)
        .with_vsync(pins.vsync)
        .with_h_enable(pins.href)
        .with_data0(pins.d0)
        .with_data1(pins.d1)
        .with_data2(pins.d2)
        .with_data3(pins.d3)
        .with_data4(pins.d4)
        .with_data5(pins.d5)
        .with_data6(pins.d6)
        .with_data7(pins.d7);
    let i2c = I2c::new(
        i2c0,
        I2cConfig::default().with_frequency(Rate::from_khz(SCCB_KHZ)),
    )
    .map_err(|_| peripheral())?
    .with_sda(pins.sda)
    .with_scl(pins.scl);
    Ov3660::new(i2c).configure_grayscale_96x96(&mut Delay::new())?;
    let ring = dma_rx_stream_buffer!(DMA_BYTES, DMA_CHUNK_BYTES);
    let transfer = camera.receive(ring).map_err(|_| peripheral())?;
    Ok(CameraGrabber {
        transfer: Some(transfer),
        frame,
        frame_bytes: 0,
        aligned: false,
    })
}

impl CameraGrabber {
    fn drain(&mut self, out: &mut [u8]) -> Result<bool, G2gError> {
        let transfer = self.transfer.as_mut().ok_or_else(peripheral)?;
        let mut whole_frame = false;
        loop {
            let (data, end_of_frame) = transfer.peek_until_eof();
            let length = data.len();
            if length == 0 && !end_of_frame {
                return Ok(whole_frame);
            }
            if let Some(room) = self.frame.get_mut(self.frame_bytes..) {
                let take = room.len().min(length);
                room[..take].copy_from_slice(&data[..take]);
            }
            self.frame_bytes += length;
            transfer.consume(length);
            if end_of_frame {
                if self.aligned && self.frame_bytes >= IMAGE_BYTES {
                    let destination = out.get_mut(..IMAGE_BYTES).ok_or(G2gError::CapsMismatch)?;
                    destination.copy_from_slice(self.frame.as_slice());
                    whole_frame = true;
                }
                self.aligned = true;
                self.frame_bytes = 0;
            }
        }
    }
}

impl FrameGrabber for CameraGrabber {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        loop {
            if self.drain(buf)? {
                return Ok(IMAGE_BYTES);
            }
            // the camera stops when every descriptor holds unread data
            if self
                .transfer
                .as_ref()
                .is_some_and(|transfer| transfer.is_done())
            {
                let (camera, ring) = self.transfer.take().ok_or_else(peripheral)?.stop();
                self.transfer = Some(camera.receive(ring).map_err(|_| peripheral())?);
                self.aligned = false;
                self.frame_bytes = 0;
            }
            embassy_futures::yield_now().await;
        }
    }
}
