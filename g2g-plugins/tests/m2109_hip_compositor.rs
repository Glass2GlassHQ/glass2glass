#![cfg(all(target_os = "linux", feature = "hip"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use std::sync::Arc;

use collect_sink::CollectSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{GpuVendor, MemoryDomain, OwnedHipBuffer, SystemSlice};
use g2g_core::meta::Orientation;
use g2g_core::runtime::block_on;
use g2g_core::{
    AsyncElement, Caps, Colorimetry, Dim, Interlace, MultiInputElement, PipelineClock, Rate,
    RawVideoFormat,
};
use g2g_plugins::compositor::{Compositor, CompositorOperator, CompositorPad, SizingPolicy};
use g2g_plugins::hip::{HipDevice, HipEvent, HipMemory, HipStream};
use g2g_plugins::hipcompositor::HipCompositor;
use g2g_plugins::hipconvertscale::HipConverter;
use g2g_plugins::videoconvert::VideoConvert;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const OVERLAY_WIDTH: u32 = 32;
const OVERLAY_HEIGHT: u32 = 24;
// the CPU compositor blends and scales in 8-bit integers, the kernel in floats
const TOLERANCE: u8 = 2;
const VENDORS: [GpuVendor; 2] = [GpuVendor::Amd, GpuVendor::Nvidia];

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

fn device(vendor: GpuVendor) -> Option<Arc<HipDevice>> {
    match HipDevice::open(vendor, 0) {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("skipping: no {vendor:?} HIP device 0: {error:?}");
            None
        }
    }
}

fn background_frame() -> Vec<u8> {
    (0..WIDTH * HEIGHT)
        .flat_map(|i| {
            let (x, y) = (i % WIDTH, i / WIDTH);
            [(x * 4) as u8, (y * 5) as u8, (x + y) as u8, 255]
        })
        .collect()
}

fn translucent_overlay() -> Vec<u8> {
    (0..OVERLAY_WIDTH * OVERLAY_HEIGHT)
        .flat_map(|i| {
            let (x, y) = (i % OVERLAY_WIDTH, i / OVERLAY_WIDTH);
            [
                200,
                (x * 8) as u8,
                (y * 10) as u8,
                (64 + x * 4 + y * 2) as u8,
            ]
        })
        .collect()
}

fn nv12_frame() -> Vec<u8> {
    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    let luma = (0..w * h).map(|i| (16 + (i % w * 3 + i / w * 5) % 220) as u8);
    let chroma = (0..w * h / 2).map(|i| {
        let (pair, row) = (i % w / 2, i / w);
        match i % 2 {
            0 => (64 + pair + row) as u8,
            _ => (192 - pair - row) as u8,
        }
    });
    luma.chain(chroma).collect()
}

fn pads() -> Vec<CompositorPad> {
    Vec::from([
        CompositorPad::at(0, 0),
        CompositorPad::at(-4, 10)
            .with_size(48, 30)
            .with_alpha(160)
            .with_zorder(1),
    ])
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

// feeds the overlay first so the compositor is primed when the background frame lands
fn composite<E: MultiInputElement>(
    element: &mut E,
    inputs: [(Caps, MemoryDomain); 2],
) -> MemoryDomain {
    for (input, (caps, _)) in inputs.iter().enumerate() {
        element.configure_pipeline(input, caps).unwrap();
    }
    let mut sink = CollectSink::default();
    for (input, (_, domain)) in inputs.into_iter().enumerate().rev() {
        let frame = Frame::new(domain, FrameTiming::default(), 0);
        block_on(element.process(input, PipelinePacket::DataFrame(frame), &mut sink)).unwrap();
    }
    sink.packets
        .into_iter()
        .find_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame.domain),
            _ => None,
        })
        .expect("the compositor emits a frame")
}

fn system(bytes: Vec<u8>) -> MemoryDomain {
    MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice()))
}

fn hip_bytes(device: &Arc<HipDevice>, domain: MemoryDomain) -> Vec<u8> {
    let MemoryDomain::Hip(buffer) = domain else {
        panic!("hipcompositor emits Hip frames");
    };
    download(device, &buffer)
}

fn cpu_composite(pads: Vec<CompositorPad>, background: Vec<u8>, overlay: Vec<u8>) -> Vec<u8> {
    let mut cpu = Compositor::new(WIDTH, HEIGHT, pads);
    let out = composite(
        &mut cpu,
        [
            (
                caps(RawVideoFormat::Rgba8, WIDTH, HEIGHT),
                system(background),
            ),
            (
                caps(RawVideoFormat::Rgba8, OVERLAY_WIDTH, OVERLAY_HEIGHT),
                system(overlay),
            ),
        ],
    );
    out.as_system_slice().expect("a system frame").to_vec()
}

fn cpu_convert(input: &Caps, bytes: &[u8], output: RawVideoFormat) -> Vec<u8> {
    let Caps::RawVideo { width, height, .. } = input else {
        panic!("raw video input");
    };
    let mut convert = VideoConvert::new(output);
    convert.configure_pipeline(input).unwrap();
    let (Dim::Fixed(width), Dim::Fixed(height)) = (width, height) else {
        panic!("fixed geometry");
    };
    convert
        .configure_output(&caps(output, *width, *height))
        .unwrap();
    let mut sink = CollectSink::default();
    let frame = Frame::new(system(bytes.to_vec()), FrameTiming::default(), 0);
    block_on(convert.process(PipelinePacket::DataFrame(frame), &mut sink)).unwrap();
    sink.packets
        .into_iter()
        .find_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => frame.domain.as_system_slice().map(<[u8]>::to_vec),
            _ => None,
        })
        .expect("VideoConvert emits a frame")
}

fn assert_close(gpu: &[u8], cpu: &[u8], what: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{what}: length");
    let worst = gpu
        .iter()
        .zip(cpu)
        .enumerate()
        .max_by_key(|(_, (a, b))| a.abs_diff(**b));
    if let Some((index, (a, b))) = worst {
        assert!(
            a.abs_diff(*b) <= TOLERANCE,
            "{what}: byte {index} is {a} on the GPU and {b} on the CPU"
        );
    }
}

#[test]
fn a_scaled_translucent_overlay_matches_the_cpu_compositor() {
    for vendor in VENDORS {
        let Some(device) = device(vendor) else {
            continue;
        };
        let mut hip = HipCompositor::new(WIDTH, HEIGHT, pads());
        let out = composite(
            &mut hip,
            [
                (
                    caps(RawVideoFormat::Rgba8, WIDTH, HEIGHT),
                    MemoryDomain::Hip(upload(&device, &background_frame())),
                ),
                (
                    caps(RawVideoFormat::Rgba8, OVERLAY_WIDTH, OVERLAY_HEIGHT),
                    MemoryDomain::Hip(upload(&device, &translucent_overlay())),
                ),
            ],
        );
        assert_close(
            &hip_bytes(&device, out),
            &cpu_composite(pads(), background_frame(), translucent_overlay()),
            &format!("{vendor:?}"),
        );
    }
}

#[test]
fn the_source_operator_and_keep_aspect_ratio_match_the_cpu_compositor() {
    let pads = || {
        Vec::from([
            CompositorPad::at(0, 0),
            CompositorPad::at(-4, 10)
                .with_size(30, 30)
                .with_alpha(160)
                .with_zorder(1)
                .with_operator(CompositorOperator::Source)
                .with_sizing_policy(SizingPolicy::KeepAspectRatio),
        ])
    };
    for vendor in VENDORS {
        let Some(device) = device(vendor) else {
            continue;
        };
        let mut hip = HipCompositor::new(WIDTH, HEIGHT, pads());
        let out = composite(
            &mut hip,
            [
                (
                    caps(RawVideoFormat::Rgba8, WIDTH, HEIGHT),
                    MemoryDomain::Hip(upload(&device, &background_frame())),
                ),
                (
                    caps(RawVideoFormat::Rgba8, OVERLAY_WIDTH, OVERLAY_HEIGHT),
                    MemoryDomain::Hip(upload(&device, &translucent_overlay())),
                ),
            ],
        );
        assert_close(
            &hip_bytes(&device, out),
            &cpu_composite(pads(), background_frame(), translucent_overlay()),
            &format!("{vendor:?}"),
        );
    }
}

#[test]
fn a_yuv_background_converts_as_it_composites() {
    for vendor in VENDORS {
        let Some(device) = device(vendor) else {
            continue;
        };
        let nv12 = caps(RawVideoFormat::Nv12, WIDTH, HEIGHT);
        let mut hip = HipCompositor::new(WIDTH, HEIGHT, pads());
        let out = composite(
            &mut hip,
            [
                (
                    nv12.clone(),
                    MemoryDomain::Hip(upload(&device, &nv12_frame())),
                ),
                (
                    caps(RawVideoFormat::Rgba8, OVERLAY_WIDTH, OVERLAY_HEIGHT),
                    MemoryDomain::Hip(upload(&device, &translucent_overlay())),
                ),
            ],
        );
        let background = cpu_convert(&nv12, &nv12_frame(), RawVideoFormat::Rgba8);
        assert_close(
            &hip_bytes(&device, out),
            &cpu_composite(pads(), background, translucent_overlay()),
            &format!("{vendor:?}"),
        );
    }
}

#[test]
fn the_output_format_is_converted_from_the_canvas() {
    for vendor in VENDORS {
        let Some(device) = device(vendor) else {
            continue;
        };
        let inputs = || {
            [
                (
                    caps(RawVideoFormat::Rgba8, WIDTH, HEIGHT),
                    MemoryDomain::Hip(upload(&device, &background_frame())),
                ),
                (
                    caps(RawVideoFormat::Rgba8, OVERLAY_WIDTH, OVERLAY_HEIGHT),
                    MemoryDomain::Hip(upload(&device, &translucent_overlay())),
                ),
            ]
        };
        let rgba = composite(&mut HipCompositor::new(WIDTH, HEIGHT, pads()), inputs());
        let nv12 = composite(
            &mut HipCompositor::new(WIDTH, HEIGHT, pads()).with_format(RawVideoFormat::Nv12),
            inputs(),
        );
        let MemoryDomain::Hip(rgba) = rgba else {
            panic!("hipcompositor emits Hip frames");
        };
        let converter = HipConverter::new(
            &device,
            &caps(RawVideoFormat::Rgba8, WIDTH, HEIGHT),
            &caps(RawVideoFormat::Nv12, WIDTH, HEIGHT),
            false,
            Orientation::Identity,
        )
        .unwrap();
        let stream = HipStream::new(&device).unwrap();
        let len = converter.output_len().unwrap();
        let mut converted = HipMemory::allocate(&device, len).unwrap();
        converter.convert(&stream, &rgba, &mut converted).unwrap();
        let done = HipEvent::new(&device).unwrap();
        stream.record(&done).unwrap();
        let expected = download(&device, &converted.into_buffer(done, len).unwrap());
        assert_close(
            &hip_bytes(&device, nv12),
            &expected,
            &format!("{vendor:?} NV12"),
        );
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
fn a_launch_line_places_an_overlay_on_the_canvas() {
    if VENDORS.iter().all(|&vendor| device(vendor).is_none()) {
        return;
    }
    let (red, blue) = ([255u8, 0, 0, 255], [0u8, 0, 255, 255]);
    let solid = |color: [u8; 4], pixels: u32| color.repeat(pixels as usize);
    let background = g2g_plugins::appsrc::register_appsrc("m2109_background");
    assert!(background.push(&solid(red, WIDTH * HEIGHT), 0));
    background.end_of_stream();
    let overlay = g2g_plugins::appsrc::register_appsrc("m2109_overlay");
    assert!(overlay.push(&solid(blue, 16 * 16), 0));
    overlay.end_of_stream();
    let pull = g2g_plugins::appsink::register_appsink_pull("m2109_out");
    let line = format!(
        "appsrc channel=m2109_background caps=video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},framerate=30/1 \
         ! hipupload ! c. \
         appsrc channel=m2109_overlay caps=video/x-raw,format=RGBA,width=16,height=16,framerate=30/1 \
         ! hipupload ! c. \
         hipcompositor name=c width={WIDTH} height={HEIGHT} sink1-xpos=8 sink1-ypos=8 sink1-zorder=1 \
         ! hipdownload ! appsink channel=m2109_out"
    );
    let registry = g2g_plugins::registry::default_registry();
    let graph = g2g_core::runtime::parse_launch(&registry, &line).expect("parses");
    block_on(g2g_core::runtime::run_graph(graph, &ZeroClock, 4)).expect("runs");
    let g2g_plugins::appsink::Pull::Frame(frame) = pull.try_pull() else {
        panic!("no frame reached the appsink");
    };
    let pixels = frame.domain.as_system_slice().expect("a system frame");
    let at = |x: u32, y: u32| &pixels[((y * WIDTH + x) * 4) as usize..][..4];
    assert_eq!(at(0, 0), red);
    assert_eq!(at(10, 10), blue);
    assert_eq!(at(23, 23), blue);
    assert_eq!(at(24, 24), red);
}
