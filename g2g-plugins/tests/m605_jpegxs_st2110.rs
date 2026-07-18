//! M605: the full ST 2110-22 JPEG XS path end to end, on real SVT-JPEG-XS and real
//! UDP: raw I422p10 -> `SvtJpegXsEnc` -> `St2110JxsSink` -> UDP loopback ->
//! `St2110JxsSrc` -> `SvtJpegXsDec` -> raw. Proves the compressed mezzanine essence
//! survives encode, RFC 9134 packetization, the wire, reassembly, and decode.
//!
//! Needs both `jpegxs` (SVT-JPEG-XS) and `st2110` (UDP transport); CI-excluded like
//! the other codec/network features.
#![cfg(all(feature = "jpegxs", feature = "st2110"))]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{block_on, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ClockSync, Dim, G2gError, MemoryDomain, MonotonicClock, OutputSink,
    PipelinePacket, PropValue, PushOutcome, Rate, RawVideoFormat,
};

use g2g_plugins::st2110jxsrtp::{St2110JxsSink, St2110JxsSrc};
use g2g_plugins::svtjpegxs::{SvtJpegXsDec, SvtJpegXsEnc};

/// Captures DataFrame payloads (and the last raw caps) an element emits.
#[derive(Default)]
struct Capture {
    frames: Vec<Vec<u8>>,
}
impl OutputSink for Capture {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    self.frames.push(s.as_slice().to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn data_frame(bytes: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: g2g_core::FrameTiming {
            pts_ns,
            ..Default::default()
        },
        sequence: 0,
        meta: Default::default(),
    })
}

/// A smooth I422p10 ramp (10-bit samples in 16-bit LE): compresses cleanly.
fn i422p10_ramp(w: usize, h: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w * h * 4];
    let y_bytes = w * h * 2;
    let c_bytes = (w / 2) * h * 2;
    let mut plane = |off: usize, pw: usize, ph: usize, bias: u16| {
        for y in 0..ph {
            for x in 0..pw {
                let v = ((x * 1023 / pw.max(1)) as u16).wrapping_add(bias) & 0x03FF;
                let o = off + (y * pw + x) * 2;
                buf[o..o + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    };
    plane(0, w, h, 0);
    plane(y_bytes, w / 2, h, 128);
    plane(y_bytes + c_bytes, w / 2, h, 256);
    buf
}

#[test]
fn raw_jpegxs_st2110_22_roundtrip() {
    let (w, h) = (128usize, 64usize);
    let raw_caps = Caps::RawVideo {
        format: RawVideoFormat::I422p10,
        width: Dim::Fixed(w as u32),
        height: Dim::Fixed(h as u32),
        framerate: Rate::Fixed(60 << 16),
    };
    let jxs_caps = Caps::CompressedVideo {
        codec: g2g_core::VideoCodec::JpegXs,
        width: Dim::Fixed(w as u32),
        height: Dim::Fixed(h as u32),
        framerate: Rate::Fixed(60 << 16),
    };
    let input = i422p10_ramp(w, h);

    // 1. Encode raw -> JPEG XS codestream (high bpp: near-lossless).
    let mut enc = SvtJpegXsEnc::new();
    enc.set_property("bpp", PropValue::Fraction(10, 1)).unwrap();
    enc.configure_pipeline(&raw_caps)
        .expect("encoder configures");
    let mut enc_out = Capture::default();
    block_on(enc.process(data_frame(input.clone(), 0), &mut enc_out)).expect("encode");
    let codestream = enc_out.frames.remove(0);

    // 2. Bind the -22 source on an ephemeral port.
    let mut src = St2110JxsSrc::new();
    src.set_property("address", PropValue::Str("127.0.0.1".into()))
        .unwrap();
    src.set_property("port", PropValue::Uint(0)).unwrap();
    src.set_property("width", PropValue::Uint(w as u64))
        .unwrap();
    src.set_property("height", PropValue::Uint(h as u64))
        .unwrap();
    src.configure_pipeline(&jxs_caps).expect("src binds");
    let port = src.local_port().expect("bound");

    // 3. Send the codestream over the -22 sink (RFC 9134 over UDP).
    let mut sink = St2110JxsSink::new();
    sink.set_property("host", PropValue::Str("127.0.0.1".into()))
        .unwrap();
    sink.set_property("port", PropValue::Uint(u64::from(port)))
        .unwrap();
    sink.set_property("max-packet", PropValue::Uint(400))
        .unwrap(); // split across packets
    sink.configure_pipeline(&jxs_caps).expect("sink configures");
    let clock: Arc<dyn g2g_core::PipelineClock + Send + Sync> = Arc::new(MonotonicClock);
    sink.set_clock_sync(ClockSync::new(clock, 1_700_000_000_000_000_000));
    let mut null = Capture::default();
    block_on(sink.process(data_frame(codestream, 0), &mut null)).expect("sink sends");

    // 4. Reassemble on the source.
    let mut rx = Capture::default();
    let n = block_on(src.run(&mut rx)).expect("src runs");
    assert_eq!(n, 1, "one codestream reassembled off the wire");
    let received = rx.frames.remove(0);

    // 5. Decode back to raw.
    let mut dec = SvtJpegXsDec::new();
    let mut dec_out = Capture::default();
    block_on(dec.process(data_frame(received, 0), &mut dec_out)).expect("decode");
    let decoded = dec_out.frames.remove(0);
    assert_eq!(decoded.len(), input.len(), "same-size planar frame back");

    // Near-lossless end to end: mean 10-bit sample error is small.
    let n = input.len() / 2;
    let mut sum = 0u64;
    for i in 0..n {
        let a = u16::from_le_bytes([input[2 * i], input[2 * i + 1]]) & 0x03FF;
        let b = u16::from_le_bytes([decoded[2 * i], decoded[2 * i + 1]]) & 0x03FF;
        sum += u64::from(a.abs_diff(b));
    }
    let mean = sum as f64 / n as f64;
    assert!(
        mean < 8.0,
        "mean 10-bit error {mean} too high across the full -22 path"
    );
}
