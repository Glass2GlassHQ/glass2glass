//! M631: `PcmSink` decodes S16LE interleaved frames and streams them to a
//! mock `PcmWriter`; the tests assert exact sample reconstruction (including
//! negative values and chunk boundaries), framing validation before any
//! peripheral traffic, and failure propagation. The mock stands in for the
//! I2S/SAI peripheral only; the element under test is real.

use std::future::Future;

use g2g_core::error::{G2gError, HardwareError};
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::MemoryDomain;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticSink;
use g2g_mcu::{PcmSink, PcmWriter};

fn block_on<F: Future>(fut: F) -> F::Output {
    g2g_core::drive_ready(fut).expect("the static chain never suspends")
}

#[derive(Default)]
struct MockWriter {
    blocks: Vec<Vec<i16>>,
    fail: bool,
}

impl PcmWriter for &mut MockWriter {
    async fn write(&mut self, samples: &[i16]) -> Result<(), G2gError> {
        if self.fail {
            return Err(G2gError::Hardware(HardwareError::Peripheral));
        }
        self.blocks.push(samples.to_vec());
        Ok(())
    }
}

/// Lend `pcm` (S16LE bytes) out of a ring as a frame, as a capture path would.
fn frame_of<const B: usize>(ring: &StaticLendRing<1, B>, pcm: &[u8]) -> Frame {
    let mut slot = ring.acquire().expect("free slot");
    slot.buf_mut()[..pcm.len()].copy_from_slice(pcm);
    // SAFETY: every test keeps the ring alive past the frame.
    let slice = unsafe { slot.publish(pcm.len()) };
    Frame::new(MemoryDomain::System(slice), FrameTiming::default(), 0)
}

fn le_bytes(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

#[test]
fn decodes_interleaved_s16le_exactly() {
    let mut writer = MockWriter::default();
    let mut sink = PcmSink::new(&mut writer, 2);
    let samples: Vec<i16> = vec![0, 1, -1, i16::MAX, i16::MIN, 12345, -12345, 42];
    let ring: StaticLendRing<1, 16> = StaticLendRing::new();
    block_on(sink.consume(frame_of(&ring, &le_bytes(&samples)))).expect("render");
    let rendered: Vec<i16> = writer.blocks.concat();
    assert_eq!(rendered, samples, "bit-exact sample reconstruction");
}

#[test]
fn long_frames_stream_through_the_fixed_chunk() {
    let mut writer = MockWriter::default();
    let mut sink = PcmSink::new(&mut writer, 1);
    // 100 samples: crosses the 64-sample chunk, so two writes (64 + 36).
    let samples: Vec<i16> = (0..100).map(|i| i * 3 - 150).collect();
    let ring: StaticLendRing<1, 200> = StaticLendRing::new();
    block_on(sink.consume(frame_of(&ring, &le_bytes(&samples)))).expect("render");
    assert_eq!(writer.blocks.len(), 2, "chunked writes");
    assert_eq!(writer.blocks[0].len(), 64);
    assert_eq!(writer.blocks[1].len(), 36);
    assert_eq!(
        writer.blocks.concat(),
        samples,
        "order preserved across chunks"
    );
}

#[test]
fn torn_sample_frames_are_caps_mismatch_before_any_write() {
    let mut writer = MockWriter::default();
    let mut sink = PcmSink::new(&mut writer, 2);
    // 6 bytes = 3 samples: not a whole stereo (4-byte) sample frame.
    let ring: StaticLendRing<1, 8> = StaticLendRing::new();
    let err = block_on(sink.consume(frame_of(&ring, &[1, 0, 2, 0, 3, 0])));
    assert_eq!(err.expect_err("torn framing"), G2gError::CapsMismatch);
    assert!(writer.blocks.is_empty(), "nothing reached the peripheral");
}

#[test]
fn writer_failure_propagates() {
    let mut writer = MockWriter {
        fail: true,
        ..Default::default()
    };
    let mut sink = PcmSink::new(&mut writer, 1);
    let ring: StaticLendRing<1, 4> = StaticLendRing::new();
    let err = block_on(sink.consume(frame_of(&ring, &[0, 0, 1, 0])));
    assert_eq!(
        err.expect_err("peripheral failure"),
        G2gError::Hardware(HardwareError::Peripheral)
    );
}
