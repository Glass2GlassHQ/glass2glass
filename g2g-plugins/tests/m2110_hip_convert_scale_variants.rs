#![cfg(all(target_os = "linux", feature = "hip"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use std::sync::Arc;

use collect_sink::CollectSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{HipVendor, MemoryDomain, SystemSlice};
use g2g_core::meta::Orientation;
use g2g_core::runtime::block_on;
use g2g_core::{
    AsyncElement, Caps, Colorimetry, Dim, Interlace, PipelineClock, Rate, RawVideoFormat,
};
use g2g_plugins::hip::{HipDevice, HipEvent, HipMemory, HipStream};
use g2g_plugins::hipconvertscale::HipConverter;
use g2g_plugins::videoflip::VideoFlip;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 6;
const VENDORS: [HipVendor; 2] = [HipVendor::Amd, HipVendor::Nvidia];
const DIRECTIONS: [(Orientation, &str); 8] = [
    (Orientation::Identity, "identity"),
    (Orientation::Rotate90Cw, "90r"),
    (Orientation::Rotate180, "180"),
    (Orientation::Rotate90Ccw, "90l"),
    (Orientation::HorizontalMirror, "horiz"),
    (Orientation::VerticalMirror, "vert"),
    (Orientation::Transpose, "ul-lr"),
    (Orientation::Transverse, "ur-ll"),
];

fn caps(format: RawVideoFormat, width: u32, height: u32) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Any,
        interlace: Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
    }
}

fn device(vendor: HipVendor) -> Option<Arc<HipDevice>> {
    match HipDevice::open(vendor, 0) {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("skipping: no {vendor:?} HIP device 0: {error:?}");
            None
        }
    }
}

fn no_device() -> bool {
    VENDORS.iter().all(|&vendor| device(vendor).is_none())
}

fn rgba_frame() -> Vec<u8> {
    (0..WIDTH * HEIGHT)
        .flat_map(|i| {
            let (x, y) = (i % WIDTH, i / WIDTH);
            [(x * 30) as u8, (y * 40) as u8, (x + 10 * y) as u8, 255]
        })
        .collect()
}

fn output_size(direction: Orientation) -> (u32, u32) {
    match direction.swaps_dims() {
        true => (HEIGHT, WIDTH),
        false => (WIDTH, HEIGHT),
    }
}

fn cpu_flip(direction: Orientation) -> Vec<u8> {
    let mut flip = VideoFlip::new(direction);
    flip.configure_pipeline(&caps(RawVideoFormat::Rgba8, WIDTH, HEIGHT))
        .unwrap();
    let mut sink = CollectSink::default();
    let input = MemoryDomain::System(SystemSlice::from_boxed(rgba_frame().into_boxed_slice()));
    let frame = Frame::new(input, FrameTiming::default(), 0);
    block_on(flip.process(PipelinePacket::DataFrame(frame), &mut sink)).unwrap();
    sink.packets
        .into_iter()
        .find_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => frame.domain.as_system_slice().map(<[u8]>::to_vec),
            _ => None,
        })
        .expect("VideoFlip emits a frame")
}

fn gpu_flip(device: &Arc<HipDevice>, direction: Orientation) -> Vec<u8> {
    let (width, height) = output_size(direction);
    let converter = HipConverter::new(
        device,
        &caps(RawVideoFormat::Rgba8, WIDTH, HEIGHT),
        &caps(RawVideoFormat::Rgba8, width, height),
        false,
        direction,
    )
    .unwrap();
    let stream = HipStream::new(device).unwrap();
    let bytes = rgba_frame();
    let input = HipMemory::allocate(device, bytes.len()).unwrap();
    stream.upload_blocking(&input, &bytes).unwrap();
    let uploaded = HipEvent::new(device).unwrap();
    stream.record(&uploaded).unwrap();
    let input = input.into_buffer(uploaded, bytes.len()).unwrap();
    let len = converter.output_len().unwrap();
    let mut output = HipMemory::allocate(device, len).unwrap();
    converter.convert(&stream, &input, &mut output).unwrap();
    let done = HipEvent::new(device).unwrap();
    stream.record(&done).unwrap();
    let output = output.into_buffer(done, len).unwrap();
    let mut out = vec![0u8; len];
    stream.download_blocking(&output, &mut out).unwrap();
    out
}

#[test]
fn every_video_direction_matches_videoflip() {
    for vendor in VENDORS {
        let Some(device) = device(vendor) else {
            continue;
        };
        for (direction, name) in DIRECTIONS {
            assert_eq!(
                gpu_flip(&device, direction),
                cpu_flip(direction),
                "{vendor:?} {name}"
            );
        }
    }
}

#[derive(Debug)]
struct ZeroClock;

impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

// `middle` sits between an RGBA appsrc and an appsink, the result is the first frame's bytes
fn run_line(channel: &str, middle: &str) -> Result<Vec<u8>, String> {
    let input = format!("{channel}_in");
    let output = format!("{channel}_out");
    let feed = g2g_plugins::appsrc::register_appsrc(&input);
    assert!(feed.push(&rgba_frame(), 0));
    feed.end_of_stream();
    let pull = g2g_plugins::appsink::register_appsink_pull(&output);
    let line = format!(
        "appsrc channel={input} caps=video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},framerate=30/1 \
         ! {middle} ! hipdownload ! appsink channel={output}"
    );
    let registry = g2g_plugins::registry::default_registry();
    let graph = g2g_core::runtime::parse_launch(&registry, &line).map_err(|e| format!("{e:?}"))?;
    block_on(g2g_core::runtime::run_graph(graph, &ZeroClock, 4)).map_err(|e| format!("{e:?}"))?;
    let g2g_plugins::appsink::Pull::Frame(frame) = pull.try_pull() else {
        return Err(String::from("no frame reached the appsink"));
    };
    Ok(frame
        .domain
        .as_system_slice()
        .expect("a system frame")
        .to_vec())
}

#[test]
fn a_launch_line_rotation_swaps_the_frame_size() {
    if no_device() {
        return;
    }
    let rotated = run_line("m2110_rotate", "hipconvertscale video-direction=90r").unwrap();
    assert_eq!(rotated, cpu_flip(Orientation::Rotate90Cw));
}

#[test]
fn hipscale_resizes_and_hipconvert_converts() {
    if no_device() {
        return;
    }
    let scaled = run_line("m2110_scale", "hipscale ! video/x-raw,width=4,height=3").unwrap();
    assert_eq!(scaled.len(), 4 * 3 * 4);
    let converted = run_line("m2110_convert", "hipconvert ! video/x-raw,format=BGRA").unwrap();
    let swapped: Vec<u8> = rgba_frame()
        .chunks(4)
        .flat_map(|pixel| [pixel[2], pixel[1], pixel[0], pixel[3]])
        .collect();
    assert_eq!(converted, swapped);
}

#[test]
fn hipscale_cannot_convert_and_hipconvert_cannot_resize() {
    if no_device() {
        return;
    }
    assert!(run_line("m2110_scale_format", "hipscale ! video/x-raw,format=BGRA").is_err());
    assert!(run_line(
        "m2110_convert_size",
        "hipconvert ! video/x-raw,width=4,height=3"
    )
    .is_err());
}
