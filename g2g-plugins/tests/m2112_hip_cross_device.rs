// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "hip"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use std::sync::Arc;

use collect_sink::CollectSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{GpuDevice, GpuVendor, MemoryDomain, OwnedHipBuffer};
use g2g_core::runtime::{block_on, parse_launch, run_graph, GraphNode};
use g2g_core::{
    AsyncElement, Caps, Colorimetry, Dim, Graph, Interlace, PipelineClock, PropValue, Rate,
    RawVideoFormat,
};
use g2g_plugins::appsink::register_appsink_pull;
use g2g_plugins::appsrc::register_appsrc;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::hip::{HipDevice, HipEvent, HipMemory, HipStream};
use g2g_plugins::hipconvertscale::HipConvertScale;
use g2g_plugins::hipdownload::HipDownload;
use g2g_plugins::hipupload::HipUpload;
use g2g_plugins::registry::default_registry;
use g2g_plugins::videotestsrc::VideoTestSrc;

const WIDTH: u32 = 16;
const HEIGHT: u32 = 8;

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Any,
        interlace: Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    }
}

fn both_gpus() -> Option<[Arc<HipDevice>; 2]> {
    let open = |vendor| match HipDevice::open(vendor, 0) {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("skipping: no {vendor:?} HIP device 0: {error:?}");
            None
        }
    };
    Some([open(GpuVendor::Nvidia)?, open(GpuVendor::Amd)?])
}

fn rgba_frame() -> Vec<u8> {
    (0..WIDTH * HEIGHT)
        .flat_map(|i| [(i * 3) as u8, (i * 7) as u8, (255 - i) as u8, 255])
        .collect()
}

fn upload(device: &Arc<HipDevice>, bytes: &[u8]) -> OwnedHipBuffer {
    let stream = HipStream::new(device).unwrap();
    let memory = HipMemory::allocate(device, bytes.len()).unwrap();
    stream.upload_blocking(&memory, bytes).unwrap();
    let ready = HipEvent::new(device).unwrap();
    stream.record(&ready).unwrap();
    memory.into_buffer(ready, bytes.len()).unwrap()
}

fn download(device: &Arc<HipDevice>, buffer: &OwnedHipBuffer) -> Vec<u8> {
    let mut out = vec![0u8; buffer.len];
    HipStream::new(device)
        .unwrap()
        .download_blocking(buffer, &mut out)
        .unwrap();
    out
}

#[test]
fn hipupload_moves_a_frame_from_one_gpu_onto_the_other() {
    let Some(gpus) = both_gpus() else {
        return;
    };
    for (from, to) in [(&gpus[0], &gpus[1]), (&gpus[1], &gpus[0])] {
        let target = GpuDevice {
            vendor: to.vendor(),
            ordinal: to.ordinal(),
        };
        let mut element = HipUpload::on_device(target);
        element.configure_pipeline(&caps()).unwrap();
        let frame = Frame::new(
            MemoryDomain::Hip(upload(from, &rgba_frame())),
            FrameTiming::default(),
            0,
        );
        let mut sink = CollectSink::default();
        block_on(element.process(PipelinePacket::DataFrame(frame), &mut sink)).unwrap();
        let moved = sink
            .packets
            .into_iter()
            .find_map(|packet| match packet {
                PipelinePacket::DataFrame(Frame {
                    domain: MemoryDomain::Hip(buffer),
                    ..
                }) => Some(buffer),
                _ => None,
            })
            .expect("hipupload emits a Hip frame");
        assert_eq!(
            (moved.vendor, moved.device_ordinal),
            (target.vendor, target.ordinal)
        );
        assert_eq!(download(to, &moved), rgba_frame(), "{:?}", target.vendor);
    }
}

#[derive(Debug)]
struct ZeroClock;

impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[test]
fn a_launch_line_copies_frames_onto_the_gpu_hipconvertscale_names() {
    if both_gpus().is_none() {
        return;
    }
    let feed = register_appsrc("m2112_in");
    assert!(feed.push(&rgba_frame(), 0));
    feed.end_of_stream();
    let pull = register_appsink_pull("m2112_out");
    // hipconvertscale refuses frames from another device, so this runs only with a copy spliced in
    let line = format!(
        "appsrc channel=m2112_in caps=video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},framerate=30/1 \
         ! hipupload vendor=nvidia ! hipconvertscale vendor=amd ! video/x-raw,format=BGRA \
         ! hipdownload ! appsink channel=m2112_out"
    );
    let graph = parse_launch(&default_registry(), &line).expect("parses");
    block_on(run_graph(graph, &ZeroClock, 4)).expect("runs");
    let g2g_plugins::appsink::Pull::Frame(frame) = pull.try_pull() else {
        panic!("no frame reached the appsink");
    };
    let swapped: Vec<u8> = rgba_frame()
        .chunks(4)
        .flat_map(|pixel| [pixel[2], pixel[1], pixel[0], pixel[3]])
        .collect();
    assert_eq!(
        frame.domain.as_system_slice().expect("a system frame"),
        swapped
    );
}

#[test]
fn a_graph_built_in_code_gets_the_same_copy_from_the_registry() {
    if both_gpus().is_none() {
        return;
    }
    const FRAMES: u64 = 3;
    let mut convert = HipConvertScale::new();
    convert
        .set_property("vendor", PropValue::Str("amd".into()))
        .unwrap();
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(VideoTestSrc::new(
        WIDTH, HEIGHT, 30, FRAMES,
    )));
    let upload = graph.add_transform(GraphNode::element(HipUpload::on_device(GpuDevice {
        vendor: GpuVendor::Nvidia,
        ordinal: 0,
    })));
    let convert = graph.add_transform(GraphNode::element(convert));
    let download = graph.add_transform(GraphNode::element(HipDownload::new()));
    let sink = graph.add_sink(GraphNode::element(FakeSink::new()));
    for (from, to) in [
        (source, upload),
        (upload, convert),
        (convert, download),
        (download, sink),
    ] {
        graph.link(from, to).unwrap();
    }
    let graph = default_registry().splice_domain_converters(graph);
    assert_eq!(graph.node_count(), 6, "a copy onto the AMD GPU was spliced");
    let stats = block_on(run_graph(graph, &ZeroClock, 4)).expect("runs");
    assert_eq!(stats.frames_consumed, FRAMES);
}

#[cfg(feature = "nvdec")]
#[test]
fn nvdec_frames_convert_on_the_amd_gpu_as_on_the_nvidia_one() {
    if both_gpus().is_none() {
        return;
    }
    let decoded_on = |vendor: &str| {
        let channel = format!("m2112_nvdec_{vendor}");
        let pull = register_appsink_pull(&channel);
        let line = format!(
            "filesrc location={}/tests/fixtures/h264_640x480.h264 ! h264parse ! nvdec \
             ! hipconvertscale vendor={vendor} ! video/x-raw,format=RGBA \
             ! hipdownload ! appsink channel={channel}",
            env!("CARGO_MANIFEST_DIR")
        );
        let graph = parse_launch(&default_registry(), &line).expect("parses");
        // appsink stops the graph once its pull queue is full, so frames are pulled while it runs
        let puller = std::thread::spawn(move || {
            let mut frames = Vec::new();
            while let Some(frame) = block_on(pull.pull()) {
                frames.push(
                    frame
                        .domain
                        .as_system_slice()
                        .expect("a system frame")
                        .to_vec(),
                );
            }
            frames
        });
        block_on(run_graph(graph, &ZeroClock, 4)).expect("runs");
        puller.join().expect("the pull thread finishes")
    };
    let on_amd = decoded_on("amd");
    let on_nvidia = decoded_on("nvidia");
    assert!(!on_amd.is_empty(), "no frame reached the appsink");
    assert_eq!(on_amd.len(), on_nvidia.len(), "frame count");
    for (index, (amd, nvidia)) in on_amd.iter().zip(&on_nvidia).enumerate() {
        let worst = amd.iter().zip(nvidia).map(|(a, b)| a.abs_diff(*b)).max();
        // the two GPUs round the same float math differently in the last bit
        assert!(worst <= Some(1), "frame {index}: differs by {worst:?}");
    }
}
