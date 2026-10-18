//! M1580 interlaced motion JPEG: an AVI chunk holding two half-height JPEG
//! fields goes through `jpegparse` as one access unit at the frame height, and
//! `MjpegDec` weaves the two fields into one interleaved frame.

#![cfg(all(feature = "mjpeg-encode", feature = "mjpeg"))]

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, Interlace, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::mjpegdec::MjpegDec;
use g2g_plugins::mjpegenc::MjpegEnc;
use g2g_plugins::stillparse::JpegParse;

const FIELD_WIDTH: u32 = 32;
const FIELD_HEIGHT: u32 = 16;
const FRAME_HEIGHT: u32 = FIELD_HEIGHT * 2;
const FRAMERATE: Rate = Rate::Fixed(25 << 16);
const FIRST_FIELD_RGBA: [u8; 4] = [200, 40, 40, 255];
const SECOND_FIELD_RGBA: [u8; 4] = [40, 40, 200, 255];

#[derive(Default)]
struct CaptureSink {
    caps: Vec<Caps>,
    frames: Vec<Vec<u8>>,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        match packet_slot.take().expect("poll_push without a packet") {
            PipelinePacket::CapsChanged(caps) => self.caps.push(caps),
            PipelinePacket::DataFrame(frame) => self.frames.push(
                frame
                    .domain
                    .as_system_slice()
                    .expect("system memory")
                    .to_vec(),
            ),
            _ => {}
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn frame(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

fn mjpeg_caps(width: Dim, height: Dim) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::Mjpeg,
        width,
        height,
        framerate: FRAMERATE,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

async fn encode_field(rgba: [u8; 4]) -> Vec<u8> {
    let mut encoder = MjpegEnc::new();
    encoder
        .configure_pipeline(&Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(FIELD_WIDTH),
            height: Dim::Fixed(FIELD_HEIGHT),
            framerate: FRAMERATE,
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        })
        .expect("RGBA input");
    let mut sink = CaptureSink::default();
    encoder
        .process(
            frame(rgba.repeat((FIELD_WIDTH * FIELD_HEIGHT) as usize)),
            &mut sink,
        )
        .await
        .expect("the field encodes");
    sink.frames.remove(0)
}

async fn decode(caps: &Caps, format: RawVideoFormat, access_unit: Vec<u8>) -> CaptureSink {
    let mut decoder = MjpegDec::new().with_output_format(format);
    decoder.configure_pipeline(caps).expect("MJPEG input");
    let mut sink = CaptureSink::default();
    decoder
        .process(frame(access_unit), &mut sink)
        .await
        .expect("the access unit decodes");
    sink
}

// each plane's rows, split from a tightly packed frame of `format`
fn plane_rows(format: RawVideoFormat, pixels: &[u8], height: u32) -> Vec<Vec<&[u8]>> {
    (0..format.plane_count())
        .map(|plane| {
            let start = format.plane_offset(plane, FIELD_WIDTH, height).unwrap() as usize;
            let bytes = format.plane_bytes(plane, FIELD_WIDTH, height).unwrap() as usize;
            let stride = format.plane_stride(plane, FIELD_WIDTH).unwrap() as usize;
            pixels[start..start + bytes].chunks_exact(stride).collect()
        })
        .collect()
}

#[tokio::test]
async fn two_jpeg_fields_weave_into_one_interleaved_frame() {
    let first_field = encode_field(FIRST_FIELD_RGBA).await;
    let second_field = encode_field(SECOND_FIELD_RGBA).await;
    let avi_chunk = [first_field.clone(), second_field.clone()].concat();

    let mut parser = JpegParse::new();
    parser
        .configure_pipeline(&mjpeg_caps(
            Dim::Fixed(FIELD_WIDTH),
            Dim::Fixed(FRAME_HEIGHT),
        ))
        .expect("MJPEG input");
    let mut parsed = CaptureSink::default();
    parser
        .process(frame(avi_chunk.clone()), &mut parsed)
        .await
        .expect("the chunk parses");
    assert_eq!(
        parsed.frames,
        vec![avi_chunk],
        "both fields in one access unit"
    );
    let parsed_caps = parsed.caps.pop().expect("jpegparse declares its caps");
    assert_eq!(
        parsed_caps,
        mjpeg_caps(Dim::Fixed(FIELD_WIDTH), Dim::Fixed(FRAME_HEIGHT))
    );

    for format in [RawVideoFormat::Rgba8, RawVideoFormat::I420] {
        let any_size = mjpeg_caps(Dim::Any, Dim::Any);
        let first = decode(&any_size, format, first_field.clone()).await;
        let second = decode(&any_size, format, second_field.clone()).await;
        assert_ne!(first.frames, second.frames, "the fields must differ");

        let mut woven = decode(&parsed_caps, format, parsed.frames[0].clone()).await;
        let Some(Caps::RawVideo {
            height, interlace, ..
        }) = woven.caps.pop()
        else {
            panic!("MjpegDec declares raw video caps");
        };
        assert_eq!(height, Dim::Fixed(FRAME_HEIGHT), "{format:?}");
        assert_eq!(interlace, Interlace::Interleaved, "{format:?}");

        let woven_planes = plane_rows(format, &woven.frames[0], FRAME_HEIGHT);
        let first_planes = plane_rows(format, &first.frames[0], FIELD_HEIGHT);
        let second_planes = plane_rows(format, &second.frames[0], FIELD_HEIGHT);
        for (plane, rows) in woven_planes.iter().enumerate() {
            for (row, bytes) in rows.iter().enumerate() {
                let field = if row % 2 == 0 {
                    &first_planes
                } else {
                    &second_planes
                };
                assert_eq!(
                    *bytes,
                    field[plane][row / 2],
                    "{format:?} plane {plane} row {row} comes from field {}",
                    row % 2
                );
            }
        }
    }
}
