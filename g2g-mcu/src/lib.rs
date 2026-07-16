//! MCU peripheral elements and MCU-fit codecs: heap-free static elements
//! ([`g2g_core::staticelem`]) written against portable `embedded-hal` 1.0
//! trait seams instead of chip registers, so each element's real logic
//! (command sequences, format conversion, framing) runs and is asserted on a
//! host with mock peripherals, and a board port is only the thin HAL adapter
//! the vendor HAL already provides. Codecs ([`g711`]) are pure fixed-point
//! math validated bit-exact against reference peers on the host ([`g711`], [`adpcm`]).
//!
//! Everything here is `no_std` with no `alloc`: the elements stay linkable on
//! targets with no allocator, extending the `g2g-noalloc` guarantee (zero
//! allocator symbols, zero panic machinery) to real peripheral elements.
//!
//! [`g2g_core::staticelem`]: g2g_core::staticelem

#![no_std]

pub mod adpcm;
pub mod cffi;
pub mod g711;
pub mod grabber;
pub mod hwh264;
pub mod hwjpeg;
pub mod jitter;
mod lend;
pub mod mixer;
pub mod pcm;
pub mod resample;
pub mod resample_tables;
pub mod rtp;
pub mod rtprecv;
pub mod sht3x;
pub mod spidisplay;
pub mod uart;
pub mod videoconvert;
pub mod watchdog;

pub use adpcm::{AdpcmDec, AdpcmEnc, ImaState};
pub use cffi::{CFrameGrabber, CH264Encoder, CPacketSender, CaptureFn, EncodeFn, SendFn};
pub use g711::{G711Dec, G711Enc, Law};
pub use grabber::{FrameGrabber, GrabberSrc};
pub use hwh264::{H264EncodeInfo, H264Encoder, HwH264Enc};
pub use hwjpeg::{HwJpegDec, JpegDecoder, JpegImageInfo, JpegSubsampling};
pub use jitter::JitterBuffer;
pub use mixer::Mixer;
pub use pcm::{PcmConvert, PcmSink, PcmWriter};
pub use resample::{Resampler, SampleRate};
pub use rtp::{PacketSender, RtpSink};
pub use rtprecv::{PacketReceiver, RtpSrc};
pub use sht3x::{Sht3xSrc, SHT3X_ADDR_DEFAULT, SHT3X_READING_BYTES};
pub use spidisplay::SpiDisplaySink;
pub use uart::{SerialRx, SerialTx, UartSink, UartSrc};
pub use videoconvert::{yuyv_len, YuyvToI420};
pub use watchdog::{SupervisorWatchdog, WatchdogTimer};
