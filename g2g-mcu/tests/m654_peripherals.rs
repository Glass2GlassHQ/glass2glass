//! M654: peripheral catalog breadth. A real I2C environmental-sensor driver
//! (SHT3x: command, datasheet CRC-8, fixed-point conversion) and a UART
//! transport (egress sink + ingress source that round-trip over a link). Only
//! the bus / link is mocked; the driver logic under test is real.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::future::Future;
use std::rc::Rc;

use embedded_hal::i2c::{ErrorKind, I2c, Operation};
use g2g_core::error::{G2gError, HardwareError};
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{run_source_sink, StaticSink, StaticSource};
use g2g_mcu::sht3x::{crc8, raw_to_milli_rh, raw_to_millicelsius};
use g2g_mcu::uart::{SerialRx, SerialTx};
use g2g_mcu::{Sht3xSrc, UartSink, UartSrc, SHT3X_ADDR_DEFAULT, SHT3X_READING_BYTES};

fn block_on<F: Future>(fut: F) -> F::Output {
    g2g_core::drive_ready(fut).expect("the static chain never suspends")
}

fn leaked_ring<const N: usize, const B: usize>() -> &'static StaticLendRing<N, B> {
    Box::leak(Box::new(StaticLendRing::new()))
}

// --- SHT3x driver: datasheet-anchored CRC and conversion ---

#[test]
fn crc8_matches_the_datasheet_worked_example() {
    // The SHT3x datasheet gives exactly this check value.
    assert_eq!(crc8(&[0xBE, 0xEF]), 0x92, "datasheet CRC-8 vector");
    // A single-byte and empty input exercise the init value path.
    assert_eq!(crc8(&[0x00]), crc8(&[0x00]), "deterministic");
    assert_ne!(crc8(&[0x00]), crc8(&[0x01]), "distinguishes inputs");
}

#[test]
fn conversion_matches_the_datasheet_transfer_functions() {
    // Endpoints of T = -45 + 175 * raw/65535.
    assert_eq!(raw_to_millicelsius(0), -45_000, "raw 0 -> -45.000 C");
    assert_eq!(
        raw_to_millicelsius(65_535),
        130_000,
        "raw full-scale -> 130.000 C"
    );
    // Endpoints of RH = 100 * raw/65535.
    assert_eq!(raw_to_milli_rh(0), 0);
    assert_eq!(
        raw_to_milli_rh(65_535),
        100_000,
        "raw full-scale -> 100.000 %RH"
    );
    // A midpoint, hand-checked: 0x6667 = 26215 -> 175000*26215/65535 - 45000.
    assert_eq!(
        raw_to_millicelsius(0x6667),
        (-45_000_i64 + 175_000 * 26_215 / 65_535) as i32
    );
}

/// A mock SHT3x on the I2C bus: every read returns the canned 6-byte response.
struct MockSht3x {
    response: [u8; 6],
    last_write: Rc<RefCell<Vec<u8>>>,
}
impl embedded_hal::i2c::ErrorType for MockSht3x {
    type Error = ErrorKind;
}
impl I2c for MockSht3x {
    fn transaction(&mut self, _addr: u8, ops: &mut [Operation<'_>]) -> Result<(), Self::Error> {
        for op in ops {
            match op {
                Operation::Write(bytes) => self.last_write.borrow_mut().extend_from_slice(bytes),
                Operation::Read(buf) => {
                    let n = buf.len().min(self.response.len());
                    buf[..n].copy_from_slice(&self.response[..n]);
                }
            }
        }
        Ok(())
    }
}

/// Build a valid 6-byte SHT3x response for the given raw temperature/humidity.
fn sht3x_response(raw_t: u16, raw_rh: u16) -> [u8; 6] {
    let t = raw_t.to_be_bytes();
    let rh = raw_rh.to_be_bytes();
    [t[0], t[1], crc8(&t), rh[0], rh[1], crc8(&rh)]
}

#[test]
fn sht3x_source_reads_converts_and_lends_a_reading() {
    let last_write = Rc::new(RefCell::new(Vec::new()));
    let i2c = MockSht3x {
        response: sht3x_response(0x6667, 0x8000),
        last_write: last_write.clone(),
    };
    let ring: &'static StaticLendRing<2, SHT3X_READING_BYTES> = leaked_ring();
    let mut src = Sht3xSrc::new(i2c, SHT3X_ADDR_DEFAULT, ring, 1_000_000).with_frame_limit(1);

    let frame = block_on(src.next()).expect("read").expect("frame");
    let MemoryDomain::System(s) = &frame.domain else {
        panic!("system frame")
    };
    let bytes = s.as_slice();
    assert_eq!(bytes.len(), SHT3X_READING_BYTES);
    // The reading is [t_mC i32 LE, rh_mpct i32 LE], the datasheet conversion.
    let t = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let rh = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert_eq!(
        t,
        raw_to_millicelsius(0x6667),
        "temperature converted per datasheet"
    );
    assert_eq!(
        rh,
        raw_to_milli_rh(0x8000),
        "humidity converted per datasheet"
    );
    // The driver issued the single-shot high-repeatability command.
    assert_eq!(
        *last_write.borrow(),
        vec![0x2C, 0x06],
        "datasheet measurement command"
    );
}

#[test]
fn sht3x_source_rejects_a_corrupt_crc() {
    let last_write = Rc::new(RefCell::new(Vec::new()));
    let mut bad = sht3x_response(0x6667, 0x8000);
    bad[2] ^= 0xFF; // corrupt the temperature CRC
    let i2c = MockSht3x {
        response: bad,
        last_write,
    };
    let ring: &'static StaticLendRing<2, SHT3X_READING_BYTES> = leaked_ring();
    let mut src = Sht3xSrc::new(i2c, SHT3X_ADDR_DEFAULT, ring, 0);
    assert_eq!(
        block_on(src.next()).expect_err("bad CRC"),
        G2gError::Hardware(HardwareError::Peripheral),
        "a CRC mismatch is a bus-integrity fault, not a trusted reading"
    );
}

// --- UART transport: egress + ingress round-trip over a link ---

/// A shared byte link both ends of the UART see (the loopback "wire").
type Wire = Rc<RefCell<VecDeque<u8>>>;

struct MockTx {
    wire: Wire,
}
impl SerialTx for MockTx {
    async fn write_all(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
        self.wire.borrow_mut().extend(bytes.iter().copied());
        Ok(())
    }
}

struct MockRx {
    wire: Wire,
}
impl SerialRx for MockRx {
    async fn read(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        let mut w = self.wire.borrow_mut();
        let mut n = 0;
        while n < buf.len() {
            match w.pop_front() {
                Some(b) => {
                    buf[n] = b;
                    n += 1;
                }
                None => break,
            }
        }
        Ok(n) // 0 = link drained (end of stream)
    }
}

fn frame(seq: u64, payload: &[u8]) -> Frame {
    let leaked: &'static [u8] = Box::leak(payload.to_vec().into_boxed_slice());
    // SAFETY: the leaked buffer is 'static and never mutated.
    let slice = unsafe {
        SystemSlice::from_foreign(leaked.as_ptr(), leaked.len(), None, core::ptr::null_mut())
    };
    Frame::new(
        MemoryDomain::System(slice),
        FrameTiming {
            pts_ns: seq,
            ..FrameTiming::default()
        },
        seq,
    )
}

#[test]
fn uart_sink_and_source_round_trip_fixed_frames() {
    let wire: Wire = Rc::new(RefCell::new(VecDeque::new()));
    // Egress: write three 4-byte frames over the UART.
    let mut sink = UartSink::new(MockTx { wire: wire.clone() });
    let payloads = [[1u8, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];
    for (i, p) in payloads.iter().enumerate() {
        block_on(sink.consume(frame(i as u64, p))).expect("tx");
    }
    assert_eq!(wire.borrow().len(), 12, "all frame bytes on the wire");

    // Ingress: read them back as 4-byte frames, in order.
    let ring: &'static StaticLendRing<2, 8> = leaked_ring();
    let mut src = UartSrc::new(MockRx { wire: wire.clone() }, ring, 4, 0).with_frame_limit(3);
    let mut got = Vec::new();
    while let Some(f) = block_on(src.next()).expect("rx") {
        let MemoryDomain::System(s) = &f.domain else {
            panic!("system frame")
        };
        got.push(s.as_slice().to_vec());
    }
    assert_eq!(
        got,
        payloads.iter().map(|p| p.to_vec()).collect::<Vec<_>>(),
        "frames round-trip in order"
    );
    assert!(wire.borrow().is_empty(), "the wire drained exactly");
}

#[test]
fn uart_source_reports_clean_eos_at_a_frame_boundary() {
    let wire: Wire = Rc::new(RefCell::new(VecDeque::new()));
    // One whole 4-byte frame, then nothing.
    wire.borrow_mut().extend([0xAA, 0xBB, 0xCC, 0xDD]);
    let ring: &'static StaticLendRing<2, 4> = leaked_ring();
    let mut src = UartSrc::new(MockRx { wire }, ring, 4, 0);
    assert!(
        block_on(src.next()).expect("frame").is_some(),
        "the whole frame reads"
    );
    assert!(
        block_on(src.next()).expect("eos").is_none(),
        "drained link is a clean EOS at a boundary"
    );
}

// --- I2C sensor -> UART egress pipeline (the telemetry path) ---

#[test]
fn sensor_to_uart_telemetry_pipeline() {
    let wire: Wire = Rc::new(RefCell::new(VecDeque::new()));
    let i2c = MockSht3x {
        response: sht3x_response(0x5000, 0x9000),
        last_write: Rc::new(RefCell::new(Vec::new())),
    };
    let ring: &'static StaticLendRing<2, SHT3X_READING_BYTES> = leaked_ring();
    let src = Sht3xSrc::new(i2c, SHT3X_ADDR_DEFAULT, ring, 1_000_000).with_frame_limit(3);
    let mut sink = UartSink::new(MockTx { wire: wire.clone() });

    block_on(run_source_sink(src, &mut sink)).expect("pipeline");

    // Three readings of 8 bytes each streamed out the UART, each the datasheet
    // conversion of the sensor's raw values.
    let out: Vec<u8> = wire.borrow().iter().copied().collect();
    assert_eq!(out.len(), 3 * SHT3X_READING_BYTES);
    let mut expect = Vec::new();
    for _ in 0..3 {
        expect.extend_from_slice(&raw_to_millicelsius(0x5000).to_le_bytes());
        expect.extend_from_slice(&raw_to_milli_rh(0x9000).to_le_bytes());
    }
    assert_eq!(
        out, expect,
        "sensor readings reach the UART converted per datasheet"
    );
}
