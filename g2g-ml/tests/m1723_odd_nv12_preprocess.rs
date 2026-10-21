#![cfg(feature = "wgpu")]

use g2g_core::element::PushOutcome;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AsyncElement, Caps, Dim, G2gError, OutputSink, Rate, RawVideoFormat};
use g2g_ml::wgpupreprocess::{gpu_available, nv12_to_rgb_tensor, WgpuPreprocess};

const LUMA_PLANE: usize = 0;
const CHROMA_PLANE: usize = 1;
const TOLERANCE: f32 = 1e-4;

static GPU_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.packets.push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn nv12_caps(width: u32, height: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

// Every luma byte differs, and every Cb,Cr pair differs from its neighbours, so
// reading a chroma row at the luma stride shifts the colour of some pixel.
fn tight_nv12(width: u32, height: u32) -> Vec<u8> {
    let format = RawVideoFormat::Nv12;
    let size = format.unpadded_frame_bytes(width, height).unwrap() as usize;
    let luma_bytes = format.plane_bytes(LUMA_PLANE, width, height).unwrap() as usize;
    let chroma_offset = format.plane_offset(CHROMA_PLANE, width, height).unwrap() as usize;
    let chroma_stride = format.plane_stride(CHROMA_PLANE, width).unwrap() as usize;
    let chroma_rows = format.plane_rows(CHROMA_PLANE, height).unwrap() as usize;
    let mut frame = vec![0u8; size];
    for (index, byte) in frame[..luma_bytes].iter_mut().enumerate() {
        *byte = (20 + 5 * index) as u8;
    }
    for row in 0..chroma_rows {
        for pair in 0..chroma_stride / 2 {
            let start = chroma_offset + row * chroma_stride + pair * 2;
            let block = row * chroma_stride / 2 + pair;
            frame[start] = (60 + 17 * block) as u8;
            frame[start + 1] = (200 - 13 * block) as u8;
        }
    }
    frame
}

async fn preprocess(width: u32, height: u32, nv12: Vec<u8>) -> Vec<f32> {
    let mut element = WgpuPreprocess::new();
    element
        .configure_pipeline(&nv12_caps(width, height))
        .expect("an odd NV12 size configures");
    let frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(nv12.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    };
    let mut out = Collect::default();
    element
        .process(PipelinePacket::DataFrame(frame), &mut out)
        .await
        .expect("preprocess an odd NV12 frame");
    let tensor = out
        .packets
        .iter()
        .find_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
        .expect("one tensor frame");
    let bytes = tensor
        .domain
        .as_system_slice()
        .expect("system-memory tensor");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

#[tokio::test]
async fn odd_nv12_matches_the_cpu_reference() {
    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    for (width, height) in [(5, 3), (7, 5), (5, 4), (6, 3)] {
        let nv12 = tight_nv12(width, height);
        let expected = nv12_to_rgb_tensor(&nv12, width as usize, height as usize);
        let got = preprocess(width, height, nv12).await;
        assert_eq!(got.len(), expected.len(), "{width}x{height} tensor length");
        for (index, (gpu, cpu)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (gpu - cpu).abs() < TOLERANCE,
                "{width}x{height} element {index}: gpu {gpu} vs cpu reference {cpu}"
            );
        }
    }
}
