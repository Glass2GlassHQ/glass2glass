//! M1140: a same-format `VideoConvert` hands its input buffer downstream
//! instead of copying it.
//!
//! The proof is allocation identity: the emitted frame's system bytes start at
//! the same address as the frame that went in. The padded (M977) and oversized
//! inputs are the control: neither can be forwarded, so both still land in a
//! fresh tight buffer.

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Colorimetry, Dim, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::videoconvert::VideoConvert;

const WIDTH: u32 = 32;
const HEIGHT: u32 = 16;
/// Row padding for the strided control, wide enough that a tight read of the
/// buffer would land on the wrong pixels.
const PAD_BYTES: usize = 24;

#[derive(Default)]
struct CollectSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn raw(format: RawVideoFormat) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
    }
}

fn frame_of(bytes: Vec<u8>) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    }
}

fn system_bytes(frame: &Frame) -> &[u8] {
    frame.domain.as_system_slice().expect("system memory out")
}

/// Run one frame through a `format`-targeted convert configured for `input`.
async fn convert_one(input: RawVideoFormat, target: RawVideoFormat, frame: Frame) -> Vec<Frame> {
    let mut convert = VideoConvert::new(target);
    convert
        .configure_pipeline(&raw(input))
        .expect("convertible input caps");
    let mut sink = CollectSink::default();
    convert
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .expect("the convert runs");
    sink.packets
        .into_iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect()
}

/// The I420 frame size at the test geometry, taken from the element's own output
/// on a real conversion rather than written out here.
async fn i420_frame_size() -> usize {
    let rgba = frame_of(vec![0u8; WIDTH as usize * HEIGHT as usize * 4]);
    let out = convert_one(RawVideoFormat::Rgba8, RawVideoFormat::I420, rgba).await;
    system_bytes(&out[0]).len()
}

/// Bytes that differ everywhere, so a wrong slice cannot pass by luck.
fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

#[tokio::test]
async fn same_format_forwards_the_input_allocation() {
    let size = i420_frame_size().await;
    let bytes = pattern(size);
    let frame = frame_of(bytes.clone());
    let input_ptr = system_bytes(&frame).as_ptr();

    let out = convert_one(RawVideoFormat::I420, RawVideoFormat::I420, frame).await;
    assert_eq!(out.len(), 1, "one frame out");
    assert_eq!(
        system_bytes(&out[0]).as_ptr(),
        input_ptr,
        "the same allocation went downstream, not a copy of it"
    );
    assert_eq!(system_bytes(&out[0]), &bytes[..], "pixels unchanged");
}

#[tokio::test]
async fn same_format_still_announces_its_output_caps() {
    let size = i420_frame_size().await;
    let mut convert = VideoConvert::new(RawVideoFormat::I420);
    convert
        .configure_pipeline(&raw(RawVideoFormat::I420))
        .unwrap();
    let mut sink = CollectSink::default();
    convert
        .process(
            PipelinePacket::DataFrame(frame_of(pattern(size))),
            &mut sink,
        )
        .await
        .unwrap();
    let announced = sink.packets.iter().any(|p| {
        matches!(
            p,
            PipelinePacket::CapsChanged(Caps::RawVideo {
                format: RawVideoFormat::I420,
                width: Dim::Fixed(WIDTH),
                height: Dim::Fixed(HEIGHT),
                ..
            })
        )
    });
    assert!(
        announced,
        "forwarding the buffer must not skip the output caps, got {:?}",
        sink.packets
    );
}

#[tokio::test]
async fn an_oversized_input_is_still_trimmed_into_its_own_buffer() {
    // Trailing bytes past the frame are not part of the picture, so the buffer
    // cannot go downstream as it is.
    let size = i420_frame_size().await;
    let bytes = pattern(size + PAD_BYTES);
    let frame = frame_of(bytes.clone());
    let input_ptr = system_bytes(&frame).as_ptr();

    let out = convert_one(RawVideoFormat::I420, RawVideoFormat::I420, frame).await;
    let got = system_bytes(&out[0]);
    assert_ne!(got.as_ptr(), input_ptr, "a trimmed copy, not the input");
    assert_eq!(got, &bytes[..size], "the frame, without the trailing bytes");
}

/// M977: a producer's padded rows must never be forwarded as if they were tight.
#[cfg(feature = "metadata")]
#[tokio::test]
async fn padded_rows_are_packed_out_rather_than_forwarded() {
    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    let tight = w * 4;
    let stride = tight + PAD_BYTES;
    let padded = pattern(stride * h);
    let mut frame = frame_of(padded.clone());
    frame
        .meta
        .attach(g2g_core::meta::PlaneLayout::single(stride));
    let input_ptr = system_bytes(&frame).as_ptr();

    let out = convert_one(RawVideoFormat::Rgba8, RawVideoFormat::Rgba8, frame).await;
    let got = system_bytes(&out[0]);
    assert_ne!(got.as_ptr(), input_ptr, "the padding had to be removed");
    let expected: Vec<u8> = (0..h)
        .flat_map(|y| padded[y * stride..y * stride + tight].to_vec())
        .collect();
    assert_eq!(got, &expected[..], "the rows, depadded");
}
