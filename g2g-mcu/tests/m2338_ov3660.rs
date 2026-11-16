// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use embedded_hal::delay::DelayNs;
use embedded_hal::i2c::{ErrorKind, ErrorType, I2c, Operation};
use g2g_core::error::G2gError;
use g2g_mcu::{Ov3660, OV3660_ADDRESS};

// esp32-camera's OV3660 driver run against the same register model, recorded by tools/ov3660-fixture.sh.
const ESP32_CAMERA_TRANSACTIONS: &str = include_str!("fixtures/ov3660/grayscale_96x96.txt");
const PRODUCT_ID: usize = 0x300a;
const OV3660_PRODUCT_ID: [u8; 2] = [0x36, 0x60];

struct RecordingSensor {
    registers: Vec<u8>,
    log: Vec<String>,
}

impl RecordingSensor {
    fn with_product_id(product_id: [u8; 2]) -> Self {
        let mut registers = vec![0u8; 0x10000];
        registers[PRODUCT_ID..PRODUCT_ID + 2].copy_from_slice(&product_id);
        Self {
            registers,
            log: Vec::new(),
        }
    }
}

impl ErrorType for RecordingSensor {
    type Error = ErrorKind;
}

impl I2c for RecordingSensor {
    fn transaction(
        &mut self,
        address: u8,
        operations: &mut [Operation<'_>],
    ) -> Result<(), ErrorKind> {
        assert_eq!(address, OV3660_ADDRESS);
        let mut register = None;
        for operation in operations {
            match operation {
                Operation::Write(&[high, low, value]) => {
                    let reg = u16::from_be_bytes([high, low]);
                    self.log.push(format!("write 0x{reg:04x} 0x{value:02x}"));
                    self.registers[reg as usize] = value;
                }
                Operation::Write(&[high, low]) => register = Some(u16::from_be_bytes([high, low])),
                Operation::Read([value]) => {
                    let reg = register.expect("register written before the read");
                    self.log.push(format!("read 0x{reg:04x}"));
                    *value = self.registers[reg as usize];
                }
                other => panic!("unexpected SCCB transaction {other:?}"),
            }
        }
        Ok(())
    }
}

struct NoDelay;

impl DelayNs for NoDelay {
    fn delay_ns(&mut self, _ns: u32) {}
}

#[test]
fn ov3660_setup_matches_esp32_camera_transaction_for_transaction() {
    let mut sensor = Ov3660::new(RecordingSensor::with_product_id(OV3660_PRODUCT_ID));
    sensor.configure_grayscale_96x96(&mut NoDelay).unwrap();
    let log = sensor.free().log;
    let expected: Vec<&str> = ESP32_CAMERA_TRANSACTIONS.lines().collect();
    assert_eq!(log, expected);
}

#[test]
fn ov3660_setup_refuses_another_sensor() {
    let mut sensor = Ov3660::new(RecordingSensor::with_product_id([0x56, 0x40]));
    assert!(matches!(
        sensor.configure_grayscale_96x96(&mut NoDelay),
        Err(G2gError::Hardware(_))
    ));
}
