#![cfg(all(target_os = "linux", feature = "hip"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use std::sync::Arc;

use collect_sink::CollectSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{HipVendor, MemoryDomain, SystemSlice};
use g2g_core::runtime::block_on;
use g2g_core::{
    AsyncElement, Caps, Colorimetry, Dim, Interlace, PipelineClock, PropValue, Rate, RawVideoFormat,
};
use g2g_plugins::hip::{HipDevice, HipEvent, HipMemory, HipStream};
use g2g_plugins::hipconvertscale::{HipConvertScale, HipConverter};
use g2g_plugins::videoconvert::VideoConvert;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
// VideoConvert rounds through 8-bit fixed-point coefficients, the kernel through floats
const CPU_TOLERANCE: u8 = 2;
const F16_TOLERANCE: f32 = 1e-3;
const VENDORS: [HipVendor; 2] = [HipVendor::Amd, HipVendor::Nvidia];

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

fn nv12_frame() -> Vec<u8> {
    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    let luma = (0..w * h).map(|i| (16 + (i % w * 3 + i / w * 5) % 220) as u8);
    // VideoConvert repeats chroma over each 2x2 block where the kernel interpolates between samples
    let chroma = (0..w * h / 2).map(|i| {
        let (pair, row) = (i % w / 2, i / w);
        let (u, v) = (64 + pair + row, 192 - pair - row);
        if i % 2 == 0 {
            u as u8
        } else {
            v as u8
        }
    });
    luma.chain(chroma).collect()
}

fn yuy2_frame() -> Vec<u8> {
    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    (0..w * h * 2)
        .map(|i| {
            let (x, y) = (i % (w * 2) / 2, i / (w * 2));
            let pair = x / 2;
            // byte order Y0 U Y1 V
            match i % 4 {
                1 => (64 + pair + y) as u8,
                3 => (192 - pair - y) as u8,
                _ => (16 + (x * 3 + y * 5) % 220) as u8,
            }
        })
        .collect()
}

fn rgba_frame() -> Vec<u8> {
    (0..WIDTH * HEIGHT * 4)
        .map(|i| (i * 13 % 256) as u8)
        .collect()
}

fn gpu_convert(
    device: &Arc<HipDevice>,
    input: &Caps,
    bytes: &[u8],
    output: &Caps,
    add_borders: bool,
) -> Vec<u8> {
    let stream = HipStream::new(device).unwrap();
    let memory = HipMemory::allocate(device, bytes.len()).unwrap();
    stream.upload_blocking(&memory, bytes).unwrap();
    let ready = HipEvent::new(device).unwrap();
    stream.record(&ready).unwrap();
    let buffer = memory.into_buffer(ready, bytes.len()).unwrap();
    let converter = HipConverter::new(device, input, output, add_borders).unwrap();
    let len = converter.output_len().unwrap();
    let mut converted = HipMemory::allocate(device, len).unwrap();
    converter.convert(&stream, &buffer, &mut converted).unwrap();
    let done = HipEvent::new(device).unwrap();
    stream.record(&done).unwrap();
    let result = converted.into_buffer(done, len).unwrap();
    let mut out = vec![0u8; len];
    stream.download_blocking(&result, &mut out).unwrap();
    out
}

fn cpu_convert(input: &Caps, bytes: &[u8], output: &Caps) -> Vec<u8> {
    let Caps::RawVideo { format, .. } = output else {
        panic!("raw video output");
    };
    let mut convert = VideoConvert::new(*format);
    convert.configure_pipeline(input).unwrap();
    convert.configure_output(output).unwrap();
    let mut sink = CollectSink::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
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
            a.abs_diff(*b) <= CPU_TOLERANCE,
            "{what}: byte {index} is {a} on the GPU and {b} on the CPU"
        );
    }
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn halves(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| {
            let half = u16::from_le_bytes(*b);
            let exponent = i32::from((half >> 10) & 0x1f);
            let mantissa = f32::from(half & 0x3ff);
            match exponent {
                0 => mantissa * 2f32.powi(-24),
                _ => (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
            }
        })
        .collect()
}

#[test]
fn yuv_and_rgb_match_the_cpu_converter() {
    for vendor in VENDORS {
        let Some(device) = device(vendor) else {
            continue;
        };
        let nv12 = caps(RawVideoFormat::Nv12, WIDTH, HEIGHT);
        let rgba = caps(RawVideoFormat::Rgba8, WIDTH, HEIGHT);
        let (nv12_bytes, rgba_bytes) = (nv12_frame(), rgba_frame());
        assert_close(
            &gpu_convert(&device, &nv12, &nv12_bytes, &rgba, false),
            &cpu_convert(&nv12, &nv12_bytes, &rgba),
            &format!("{vendor:?} NV12 to RGBA"),
        );
        assert_close(
            &gpu_convert(&device, &rgba, &rgba_bytes, &nv12, false),
            &cpu_convert(&rgba, &rgba_bytes, &nv12),
            &format!("{vendor:?} RGBA to NV12"),
        );
    }
}

#[test]
fn yuy2_matches_the_cpu_converter() {
    for vendor in VENDORS {
        let Some(device) = device(vendor) else {
            continue;
        };
        let yuy2 = caps(RawVideoFormat::Yuyv, WIDTH, HEIGHT);
        let rgba = caps(RawVideoFormat::Rgba8, WIDTH, HEIGHT);
        let (yuy2_bytes, rgba_bytes) = (yuy2_frame(), rgba_frame());
        assert_close(
            &gpu_convert(&device, &yuy2, &yuy2_bytes, &rgba, false),
            &cpu_convert(&yuy2, &yuy2_bytes, &rgba),
            &format!("{vendor:?} YUY2 to RGBA"),
        );
        assert_close(
            &gpu_convert(&device, &rgba, &rgba_bytes, &yuy2, false),
            &cpu_convert(&rgba, &rgba_bytes, &yuy2),
            &format!("{vendor:?} RGBA to YUY2"),
        );
    }
}

#[test]
fn planar_and_float_rgb_hold_the_same_samples() {
    let pixels = (WIDTH * HEIGHT) as usize;
    for vendor in VENDORS {
        let Some(device) = device(vendor) else {
            continue;
        };
        let rgba = caps(RawVideoFormat::Rgba8, WIDTH, HEIGHT);
        let bytes = rgba_frame();
        let channel = |c: usize| bytes.iter().skip(c).step_by(4).copied();
        let to = |format| gpu_convert(&device, &rgba, &bytes, &caps(format, WIDTH, HEIGHT), false);

        let rgbp = to(RawVideoFormat::Rgbp8);
        let expected: Vec<u8> = (0..3).flat_map(channel).collect();
        assert!(rgbp == expected, "{vendor:?}: RGBP planes are R, G, B");
        let bgrp = to(RawVideoFormat::Bgrp8);
        let expected: Vec<u8> = [2, 1, 0].into_iter().flat_map(channel).collect();
        assert!(bgrp == expected, "{vendor:?}: BGRP planes are B, G, R");

        let rgbp_f32 = floats(&to(RawVideoFormat::RgbpF32));
        let expected: Vec<f32> = (0..3)
            .flat_map(channel)
            .map(|code| f32::from(code) / 255.0)
            .collect();
        assert_eq!(rgbp_f32.len(), 3 * pixels);
        assert!(
            rgbp_f32
                .iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() < 1e-6),
            "{vendor:?}: RGBP_F32LE is the 8-bit code over 255"
        );
        let rgbp_f16 = halves(&to(RawVideoFormat::RgbpF16));
        assert!(
            rgbp_f16
                .iter()
                .zip(&rgbp_f32)
                .all(|(a, b)| (a - b).abs() < F16_TOLERANCE),
            "{vendor:?}: RGBP_F16LE follows RGBP_F32LE"
        );
        let rgba_f32 = floats(&to(RawVideoFormat::RgbaF32));
        let expected: Vec<f32> = bytes.iter().map(|&code| f32::from(code) / 255.0).collect();
        assert!(
            rgba_f32
                .iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() < 1e-6),
            "{vendor:?}: RGBA_F32LE keeps every channel, alpha included"
        );
        let rgb_f16 = halves(&to(RawVideoFormat::RgbF16));
        assert_eq!(
            rgb_f16.len(),
            3 * pixels,
            "{vendor:?}: RGB_F16LE has three channels"
        );

        let rgbp_f32_caps = caps(RawVideoFormat::RgbpF32, WIDTH, HEIGHT);
        let rgbp_f32_bytes = to(RawVideoFormat::RgbpF32);
        let back = gpu_convert(&device, &rgbp_f32_caps, &rgbp_f32_bytes, &rgba, false);
        let opaque: Vec<u8> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], u8::MAX])
            .collect();
        assert!(
            back == opaque,
            "{vendor:?}: RGBP_F32LE converts back to the same RGB, opaque"
        );
    }
}

#[test]
fn p010_carries_the_nv12_codes() {
    for vendor in VENDORS {
        let Some(device) = device(vendor) else {
            continue;
        };
        let nv12_bytes = nv12_frame();
        let p010_bytes: Vec<u8> = nv12_bytes
            .iter()
            .flat_map(|&code| (u16::from(code) << 8).to_le_bytes())
            .collect();
        let converted = gpu_convert(
            &device,
            &caps(RawVideoFormat::P010, WIDTH, HEIGHT),
            &p010_bytes,
            &caps(RawVideoFormat::Nv12, WIDTH, HEIGHT),
            false,
        );
        assert!(
            converted == nv12_bytes,
            "{vendor:?}: P010 to NV12 keeps every code"
        );
    }
}

#[test]
fn halving_a_gradient_averages_pixel_pairs() {
    for vendor in VENDORS {
        let Some(device) = device(vendor) else {
            continue;
        };
        let gradient: Vec<u8> = (0..WIDTH * HEIGHT)
            .flat_map(|i| {
                let value = (i % WIDTH * 4) as u8;
                [value, value, value, 255]
            })
            .collect();
        let scaled = gpu_convert(
            &device,
            &caps(RawVideoFormat::Rgba8, WIDTH, HEIGHT),
            &gradient,
            &caps(RawVideoFormat::Rgba8, WIDTH / 2, HEIGHT / 2),
            false,
        );
        for (i, pixel) in scaled.as_chunks::<4>().0.iter().enumerate() {
            let x = i as u32 % (WIDTH / 2);
            let expected = (8 * x + 2) as u8;
            assert_eq!(pixel[0], expected, "{vendor:?}: column {x}");
        }
    }
}

#[test]
fn add_borders_letterboxes_and_its_absence_stretches() {
    for vendor in VENDORS {
        let Some(device) = device(vendor) else {
            continue;
        };
        let white = vec![255u8; (WIDTH * HEIGHT / 2 * 4) as usize];
        let wide = caps(RawVideoFormat::Rgba8, WIDTH, HEIGHT / 2);
        let square = caps(RawVideoFormat::Rgba8, WIDTH, WIDTH);
        let row_value = |bytes: &[u8], y: u32| bytes[(y * WIDTH * 4) as usize];
        let boxed = gpu_convert(&device, &wide, &white, &square, true);
        let band = (WIDTH - HEIGHT / 2) / 2;
        assert_eq!(row_value(&boxed, 0), 0, "{vendor:?}: top border");
        assert_eq!(
            row_value(&boxed, band - 1),
            0,
            "{vendor:?}: last border row"
        );
        assert_eq!(
            row_value(&boxed, band),
            255,
            "{vendor:?}: first picture row"
        );
        assert_eq!(
            row_value(&boxed, WIDTH - band),
            0,
            "{vendor:?}: bottom border"
        );
        let stretched = gpu_convert(&device, &wide, &white, &square, false);
        assert!(
            stretched.iter().all(|&byte| byte == 255),
            "{vendor:?}: stretched"
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
fn a_launch_line_converts_to_model_input() {
    if VENDORS.iter().all(|&vendor| device(vendor).is_none()) {
        return;
    }
    let (out_width, out_height) = (32, 32);
    let feed = g2g_plugins::appsrc::register_appsrc("m2105_in");
    assert!(feed.push(&nv12_frame(), 0));
    feed.end_of_stream();
    let pull = g2g_plugins::appsink::register_appsink_pull("m2105_out");
    let line = format!(
        "appsrc channel=m2105_in caps=video/x-raw,format=NV12,width={WIDTH},height={HEIGHT},framerate=30/1 \
         ! hipconvertscale ! video/x-raw,format=RGBP_F32LE,width={out_width},height={out_height} \
         ! hipdownload ! appsink channel=m2105_out"
    );
    let registry = g2g_plugins::registry::default_registry();
    let graph = g2g_core::runtime::parse_launch(&registry, &line).expect("parses");
    block_on(g2g_core::runtime::run_graph(graph, &ZeroClock, 4)).expect("runs");
    let g2g_plugins::appsink::Pull::Frame(frame) = pull.try_pull() else {
        panic!("no frame reached the appsink");
    };
    let samples = floats(frame.domain.as_system_slice().expect("a system frame"));
    assert_eq!(samples.len(), 3 * out_width * out_height);
    assert!(samples.iter().all(|sample| (0.0..=1.0).contains(sample)));
    assert!(samples.iter().any(|&sample| sample > 0.0), "not all black");
}

#[test]
fn add_borders_is_a_property() {
    let mut element = HipConvertScale::new();
    assert_eq!(
        element.get_property("add-borders"),
        Some(PropValue::Bool(true))
    );
    element
        .set_property("add-borders", PropValue::Bool(false))
        .unwrap();
    assert_eq!(
        element.get_property("add-borders"),
        Some(PropValue::Bool(false))
    );
}
