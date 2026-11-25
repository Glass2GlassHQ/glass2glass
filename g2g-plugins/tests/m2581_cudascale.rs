// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "cuda"))]

use g2g_core::runtime::block_on;
use g2g_core::{AsyncElement, Caps, Dim, G2gError, PipelineClock, Rate, RawVideoFormat};
use g2g_plugins::cuda::CudaUpload;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 4;

#[derive(Debug)]
struct ZeroClock;

impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn no_cuda_device() -> bool {
    let caps = Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    };
    match CudaUpload::new().configure_pipeline(&caps) {
        Err(error @ G2gError::Hardware(_)) => {
            eprintln!("skipping: no CUDA device: {error:?}");
            true
        }
        _ => false,
    }
}

// luma steps 10 per column and 30 per row, chroma U steps 10 per column and 50 per row
fn nv12_frame() -> Vec<u8> {
    let luma = (0..HEIGHT).flat_map(|y| (0..WIDTH).map(move |x| (10 * x + 30 * y) as u8));
    let chroma = (0..HEIGHT / 2)
        .flat_map(|y| (0..WIDTH / 2).flat_map(move |x| [(10 * x + 50 * y) as u8, 128]));
    luma.chain(chroma).collect()
}

#[test]
fn cudascale_halves_an_nv12_frame_by_averaging_each_two_by_two_block() {
    if no_cuda_device() {
        return;
    }
    let feed = g2g_plugins::appsrc::register_appsrc("m2581_in");
    assert!(feed.push(&nv12_frame(), 0));
    feed.end_of_stream();
    let pull = g2g_plugins::appsink::register_appsink_pull("m2581_out");
    let line = format!(
        "appsrc channel=m2581_in caps=video/x-raw,format=NV12,width={WIDTH},height={HEIGHT},framerate=30/1 \
         ! cudaupload ! cudascale ! video/x-raw,width={},height={} ! cudadownload ! appsink channel=m2581_out",
        WIDTH / 2,
        HEIGHT / 2
    );
    let registry = g2g_plugins::registry::default_registry();
    let graph = g2g_core::runtime::parse_launch(&registry, &line).unwrap();
    block_on(g2g_core::runtime::run_graph(graph, &ZeroClock, 4)).unwrap();
    let g2g_plugins::appsink::Pull::Frame(frame) = pull.try_pull() else {
        panic!("no frame reached the appsink");
    };
    let scaled = frame.domain.as_system_slice().expect("a system frame");
    let luma = [20, 40, 60, 80, 80, 100, 120, 140];
    let chroma = [30, 128, 50, 128];
    assert_eq!(scaled, [luma.as_slice(), chroma.as_slice()].concat());
}
