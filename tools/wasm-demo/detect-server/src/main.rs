//! Native detection server for the browser demo's Architecture B: server-side
//! inference, thin browser client.
//!
//! The browser's generic `WsWireTransform` element (M555, the distributed-graph
//! primitive) ships each decoded RGBA frame here over a WebSocket, serialized by
//! the g2g-core wire codec; this server runs the REAL native g2g detection chain
//! (`OrtInference` -> native ONNX Runtime -> `DetectionPostprocess`, the exact
//! chain `g2g-ml/tests/yolo_detect.rs` and the `detect_overlay` example use),
//! attaches the detections to the frame as `AnalyticsMeta`, and sends the frame
//! back over the same wire codec. The browser emits it and its overlay draws the
//! boxes. The point: the graph (`decode -> detect -> overlay -> canvas`) is
//! identical to the in-browser Architecture A; only the `detect` element is the
//! generic remote transform pointing here. Inference moved across the network by
//! swapping one element, over the media-agnostic primitive (no bespoke protocol).
//!
//! Wire protocol (the g2g-core wire codec, one binary WebSocket message per
//! packet): the client sends a leading `CapsChanged` (RGBA + dims) then one
//! `DataFrame` per frame; the server replies one `DataFrame` per frame, the same
//! frame with an `AnalyticsMeta` (the detections) attached in band. FIFO, one
//! frame in flight.
//!
//! Standalone dev tool (not a g2g element). Usage:
//!   cargo run --release -- [bind=127.0.0.1:8090] [model=../models/yolov8n.onnx]

use futures_util::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message;

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::wire::{decode_packet, encode_packet};
use g2g_core::{AnalyticsMeta, Caps, Dim, G2gError, TensorDType, TensorLayout, TensorShape};
use g2g_ml::detect::DetectionPostprocess;
use g2g_ml::ortinfer::OrtInference;

const DEFAULT_ADDR: &str = "127.0.0.1:8090";
const DEFAULT_MODEL: &str = "../models/yolov8n.onnx";
/// YOLOv8 model input side (square).
const SIZE: u32 = 640;
/// COCO YOLOv8n output `[1, 4 + 80, 8400]`: 4 box + 80 class channels, 8400 anchors.
const CHANNELS: u32 = 84;
const ANCHORS: u32 = 8400;

/// Capturing sink that keeps the first `DataFrame`'s system bytes (the inference
/// output tensor). Mirrors `yolo_detect.rs`'s `OneFrame`.
#[derive(Default)]
struct OneFrame {
    bytes: Option<Vec<u8>>,
}
impl OutputSink for OneFrame {
    fn push<'a>(&'a mut self, packet: PipelinePacket) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if self.bytes.is_none() {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.bytes = Some(s.as_slice().to_vec());
                    }
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Capturing sink that reads the detections off the frame's `AnalyticsMeta`.
#[derive(Default)]
struct MetaSink {
    dets: Vec<g2g_core::ObjectDetection>,
}
impl OutputSink for MetaSink {
    fn push<'a>(&'a mut self, packet: PipelinePacket) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = &packet {
                if let Some(a) = f.meta.get::<g2g_core::AnalyticsMeta>() {
                    self.dets = a.detections().copied().collect();
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn tensor_caps(shape: &[u32]) -> Caps {
    Caps::Tensor { dtype: TensorDType::F32, shape: TensorShape::from_slice(shape).unwrap(), layout: TensorLayout::Nchw }
}

fn frame_from(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    })
}

/// Resize the RGBA frame whole-to-whole (nearest) into `[1,3,640,640]` NCHW f32
/// (RGB planar, normalized to `[0,1]`), the standard YOLOv8 input. The native
/// twin of the browser `WebOrtDetect::preprocess`; whole-to-whole (no letterbox)
/// so the decoded `[0,1]` boxes map back onto the source frame.
fn preprocess(rgba: &[u8], w: u32, h: u32) -> Vec<f32> {
    let (sw, sh) = (SIZE as usize, SIZE as usize);
    let (w, h) = (w as usize, h as usize);
    let plane = sw * sh;
    let mut out = vec![0f32; 3 * plane];
    for y in 0..sh {
        let sy = y * h / sh;
        for x in 0..sw {
            let sx = x * w / sw;
            let si = (sy * w + sx) * 4;
            let di = y * sw + x;
            out[di] = rgba[si] as f32 / 255.0; // R
            out[plane + di] = rgba[si + 1] as f32 / 255.0; // G
            out[2 * plane + di] = rgba[si + 2] as f32 / 255.0; // B
        }
    }
    out
}

/// The native detection chain, held per connection. `OrtInference` owns an ONNX
/// Runtime session (not `Send`), so the server runs single-threaded and serves
/// one connection at a time (a demo, one browser client).
struct Detector {
    infer: OrtInference,
    decode: DetectionPostprocess,
}

impl Detector {
    fn new(model_path: &str) -> Result<Self, G2gError> {
        let mut infer = OrtInference::from_file(model_path)?.with_tensor_input();
        infer.configure_pipeline(&tensor_caps(&[1, 3, SIZE, SIZE]))?;
        let mut decode = DetectionPostprocess::new(0.25, 0.45).with_input_size(SIZE, SIZE);
        decode.configure_pipeline(&tensor_caps(&[1, CHANNELS, ANCHORS]))?;
        Ok(Self { infer, decode })
    }

    /// Run one RGBA frame through inference + decode, returning the detections.
    async fn detect(&mut self, rgba: &[u8], w: u32, h: u32) -> Result<Vec<g2g_core::ObjectDetection>, G2gError> {
        let input = preprocess(rgba, w, h);
        // f32 -> LE bytes (the tensor DataFrame the elements consume).
        let mut input_bytes = Vec::with_capacity(input.len() * 4);
        for v in &input {
            input_bytes.extend_from_slice(&v.to_le_bytes());
        }
        // Stage 1: real YOLO, image tensor -> [1,84,8400] raw detections.
        let mut raw = OneFrame::default();
        self.infer.process(frame_from(input_bytes), &mut raw).await?;
        let raw_bytes = raw.bytes.ok_or(G2gError::CapsMismatch)?;
        // Stage 2: anchor decode + per-class NMS -> AnalyticsMeta detections.
        let mut sink = MetaSink::default();
        self.decode.process(frame_from(raw_bytes), &mut sink).await?;
        Ok(sink.dets)
    }
}

/// The fixed value of a `Dim`, else 0 (an open / unknown dimension).
fn fixed(d: &Dim) -> u32 {
    match d {
        Dim::Fixed(v) => *v,
        _ => 0,
    }
}

async fn serve(tcp: TcpStream, model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut ws = tokio_tungstenite::accept_async(tcp).await?;
    let mut detector = Detector::new(model_path).map_err(|e| format!("detector init: {e:?}"))?;
    // Frame geometry from the client's CapsChanged (updated on any refinement).
    let (mut w, mut h) = (0u32, 0u32);
    let mut frames = 0u64;
    while let Some(msg) = ws.next().await {
        let Message::Binary(bytes) = msg? else {
            continue; // ignore text / ping / close-adjacent frames
        };
        match decode_packet(&bytes).map_err(|e| format!("wire decode: {e:?}"))? {
            PipelinePacket::CapsChanged(Caps::RawVideo { width, height, .. }) => {
                w = fixed(&width);
                h = fixed(&height);
            }
            PipelinePacket::CapsChanged(_) => {}
            PipelinePacket::DataFrame(mut frame) => {
                // Copy the RGBA out (ending the immutable borrow) so we can attach
                // metadata below; validate the length against the announced
                // geometry (never trust the wire) before running inference.
                let rgba: Option<Vec<u8>> = match &frame.domain {
                    MemoryDomain::System(s) => {
                        let need = (w as usize).saturating_mul(h as usize).saturating_mul(4);
                        let bytes = s.as_slice();
                        if w > 0 && h > 0 && bytes.len() >= need {
                            Some(bytes[..need].to_vec())
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                if let Some(rgba) = rgba {
                    match detector.detect(&rgba, w, h).await {
                        Ok(dets) => {
                            frames += 1;
                            if frames <= 3 || frames % 30 == 0 {
                                println!("frame {frames}: {w}x{h} -> {} detections", dets.len());
                            }
                            let mut analytics = AnalyticsMeta::new();
                            for d in dets {
                                analytics.add_detection(d);
                            }
                            frame.meta.attach(analytics);
                        }
                        Err(e) => eprintln!("inference error: {e:?}"),
                    }
                }
                // Reply with the same frame (detections attached in band). The
                // client emits it one-for-one, so this keeps the stream FIFO.
                let body = encode_packet(&PipelinePacket::DataFrame(frame))
                    .map_err(|e| format!("wire encode: {e:?}"))?;
                ws.send(Message::Binary(body)).await?;
            }
            PipelinePacket::Eos => break,
            _ => {} // Segment / Flush: not sent by the transform
        }
    }
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let mut args = std::env::args().skip(1);
    let addr = args.next().unwrap_or_else(|| DEFAULT_ADDR.to_string());
    let model = args.next().unwrap_or_else(|| DEFAULT_MODEL.to_string());

    // Fail loud at startup if the model is missing / invalid, rather than per frame.
    if let Err(e) = Detector::new(&model) {
        panic!("load detection model {model}: {e:?}");
    }
    let listener = TcpListener::bind(&addr).await.unwrap_or_else(|e| panic!("bind {addr}: {e}"));
    println!("detect-server: model {model}, serving ws://{addr} (native OrtInference + DetectionPostprocess)");

    loop {
        match listener.accept().await {
            Ok((tcp, peer)) => {
                println!("client connected: {peer}");
                // One client at a time: the ONNX session is single-threaded.
                if let Err(e) = serve(tcp, &model).await {
                    println!("client {peer} done: {e}");
                }
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
}
