//! Shared NDK MediaCodec plumbing for the Android encode / decode elements
//! (`mediacodecdec`, `mediacodecenc`): the buffer-flag constants and the
//! dequeue-input-with-retries skeleton, which were duplicated byte-for-byte.
//! The output side differs (decode renders to an ImageReader Surface, encode
//! reads ByteBuffers), so it stays in each element.

use core::time::Duration;

use ndk::media::media_codec::{DequeuedInputBufferResult, MediaCodec};

use g2g_core::{G2gError, HardwareError};

/// `AMEDIACODEC_BUFFER_FLAG_KEY_FRAME`: the access unit is an IDR.
pub(crate) const BUFFER_FLAG_KEY_FRAME: u32 = 1;
/// `AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG`: the buffer is codec-specific data
/// (Annex-B parameter sets), not a displayable frame.
pub(crate) const BUFFER_FLAG_CODEC_CONFIG: u32 = 2;
/// `AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM`: mark the final (empty) input buffer.
pub(crate) const BUFFER_FLAG_END_OF_STREAM: u32 = 4;

/// Bounded output polls so an EOS drain waits for the codec to flush without
/// spinning forever if it never raises the end-of-stream flag.
pub(crate) const MAX_OUTPUT_POLLS: u32 = 256;

/// Bounded retries when the codec has no free input buffer yet, so a stuck codec
/// surfaces as an error rather than spinning forever.
const MAX_INPUT_RETRIES: u32 = 100;

/// Hand `data` to a free input buffer on `codec` with the given microsecond pts
/// and flags. Retries a bounded number of times while the codec reports no free
/// buffer (it frees them as it drains), then errors rather than spinning forever.
pub(crate) fn queue_input(
    codec: &MediaCodec,
    data: &[u8],
    pts_us: u64,
    flags: u32,
) -> Result<(), G2gError> {
    for _ in 0..MAX_INPUT_RETRIES {
        match codec
            .dequeue_input_buffer(Duration::from_millis(10))
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?
        {
            DequeuedInputBufferResult::Buffer(mut input) => {
                let dst = input.buffer_mut();
                if dst.len() < data.len() {
                    // A single access unit larger than an input buffer would need
                    // splitting across buffers; not handled in v1.
                    return Err(G2gError::Hardware(HardwareError::Other));
                }
                for (d, &s) in dst.iter_mut().zip(data) {
                    d.write(s);
                }
                codec
                    .queue_input_buffer(input, 0, data.len(), pts_us, flags)
                    .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                return Ok(());
            }
            DequeuedInputBufferResult::TryAgainLater => continue,
        }
    }
    Err(G2gError::Hardware(HardwareError::Other))
}
