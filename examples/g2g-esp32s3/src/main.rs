// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![no_std]
#![no_main]

mod camera;
mod keyword;
mod network;
mod person;
mod tflm;

use core::mem::MaybeUninit;
use core::pin::pin;

use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::psram::{Psram, PsramConfig};
use esp_hal::time::Instant;
use esp_println::println;
use g2g_core::error::{G2gError, HardwareError};
use g2g_mcu::CTensorModel;
use noalloc_pipeline::keyword::CLIP_BYTES;

// espflash refuses images without the esp-idf application descriptor.
esp_bootloader_esp_idf::esp_app_desc!();

pub(crate) fn peripheral() -> G2gError {
    G2gError::Hardware(HardwareError::Peripheral)
}

fn report(name: &str, run: impl FnOnce() -> u64, expected: u64) -> bool {
    let start = Instant::now();
    let got = run();
    let micros = start.elapsed().as_micros();
    let matches = got == expected;
    let verdict = if matches { "OK" } else { "MISMATCH" };
    println!("{verdict} {name}: {got} (expected {expected}), {micros} us");
    matches
}

struct PsramBuffers {
    speech_arena: &'static mut [MaybeUninit<u8>],
    person_arena: &'static mut [MaybeUninit<u8>],
    microphone_window: &'static mut [u8; CLIP_BYTES],
    camera_frame: &'static mut [u8; person::IMAGE_BYTES],
}

fn psram_buffers(psram: &Psram) -> PsramBuffers {
    let (start, bytes) = psram.raw_parts();
    assert!(
        bytes >= keyword::ARENA_BYTES + person::ARENA_BYTES + CLIP_BYTES + person::IMAGE_BYTES,
        "PSRAM holds {bytes} bytes"
    );
    // SAFETY: Psram maps `bytes` bytes at `start` for the life of the program, and only this function hands them out.
    let memory = unsafe { core::slice::from_raw_parts_mut(start.cast::<MaybeUninit<u8>>(), bytes) };
    let (speech_arena, rest) = memory.split_at_mut(keyword::ARENA_BYTES);
    let (person_arena, rest) = rest.split_at_mut(person::ARENA_BYTES);
    let (window, rest) = rest.split_at_mut(CLIP_BYTES);
    let frame = &mut rest[..person::IMAGE_BYTES];
    for byte in window.iter_mut().chain(frame.iter_mut()) {
        byte.write(0);
    }
    // SAFETY: every byte of both buffers was just written.
    let (microphone_window, camera_frame) = unsafe {
        (
            &mut *window.as_mut_ptr().cast::<[u8; CLIP_BYTES]>(),
            &mut *frame.as_mut_ptr().cast::<[u8; person::IMAGE_BYTES]>(),
        )
    };
    PsramBuffers {
        speech_arena,
        person_arena,
        microphone_window,
        camera_frame,
    }
}

// without LTO, rustc gives main one stack frame for everything inlined into it
#[inline(never)]
fn run_checks(speech: &mut CTensorModel, person_model: &mut CTensorModel) -> bool {
    let display = report(
        "display",
        noalloc_pipeline::run,
        noalloc_pipeline::EXPECTED_CHECKSUM,
    );
    let audio = report(
        "audio",
        noalloc_pipeline::audio::run_audio,
        noalloc_pipeline::audio::AUDIO_EXPECTED_CHECKSUM,
    );
    let features = report(
        "keyword features",
        || noalloc_pipeline::keyword::run_keyword().unwrap_or(0),
        noalloc_pipeline::keyword::EXPECTED_FEATURES_DIGEST,
    );
    let speech_scores = keyword::check_model(speech);
    let person_scores = person::check_samples(person_model);
    display && audio && features && speech_scores && person_scores
}

#[inline(never)]
fn run_live(
    speech: &mut CTensorModel,
    person_model: &mut CTensorModel,
    microphone: keyword::Microphone,
    camera: camera::CameraGrabber,
    network: network::Network,
) {
    let network::Network {
        mut controller,
        stack,
        mut runner,
    } = network;
    let networking = pin!(runner.run());
    let listening = pin!(keyword::listen(speech, microphone));
    let watching = pin!(async {
        network::connect(&mut controller, stack).await?;
        person::watch(person_model, camera, network::udp_sender(stack)?).await
    });
    let graphs = embassy_futures::join::join(listening, watching);
    let embassy_futures::select::Either::Second((keyword_stopped, person_stopped)) =
        embassy_futures::block_on(embassy_futures::select::select(networking, graphs));
    println!(
        "keyword graph stopped: {keyword_stopped:?}, person graph stopped: {person_stopped:?}"
    );
}

#[esp_hal::main]
fn main() -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
    println!("g2g on esp32s3");

    let psram = Psram::new(peripherals.PSRAM, PsramConfig::default());
    let buffers = psram_buffers(&psram);
    let mut speech = keyword::load_model(buffers.speech_arena).expect("speech model");
    let mut person_model = person::load_model(buffers.person_arena).expect("person_detect model");
    let network = network::start(
        peripherals.WIFI,
        peripherals.TIMG0,
        peripherals.FROM_CPU_INTR0,
    )
    .expect("wifi");
    let verdict = if run_checks(&mut speech, &mut person_model) {
        "PASS"
    } else {
        "FAIL"
    };
    println!("{verdict}");

    let pins = camera::CameraPins {
        xclk: peripherals.GPIO10,
        sda: peripherals.GPIO40,
        scl: peripherals.GPIO39,
        d0: peripherals.GPIO15,
        d1: peripherals.GPIO17,
        d2: peripherals.GPIO18,
        d3: peripherals.GPIO16,
        d4: peripherals.GPIO14,
        d5: peripherals.GPIO12,
        d6: peripherals.GPIO11,
        d7: peripherals.GPIO48,
        vsync: peripherals.GPIO38,
        href: peripherals.GPIO47,
        pclk: peripherals.GPIO13,
    };
    let camera = camera::open(
        peripherals.LCD_CAM,
        peripherals.DMA_CH1,
        peripherals.I2C0,
        pins,
        buffers.camera_frame,
    )
    .expect("OV3660 camera");
    let microphone = keyword::Microphone {
        i2s0: peripherals.I2S0,
        dma: peripherals.DMA_CH0,
        clock: peripherals.GPIO42,
        data: peripherals.GPIO41,
        window: buffers.microphone_window,
    };
    run_live(&mut speech, &mut person_model, microphone, camera, network);
    loop {}
}
