// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![no_std]
#![no_main]

extern crate alloc;

mod camera;
mod keyword;
mod network;
mod person;
mod tflm;

use alloc::boxed::Box;
use core::mem::MaybeUninit;
use core::ptr;

use embassy_executor::raw::TaskStorage;
use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::psram::{Psram, PsramConfig};
use esp_hal::system::Stack;
use esp_hal::time::Instant;
use esp_println::println;
use esp_radio::wifi::WifiController;
use esp_rtos::embassy::Executor;
use g2g_core::error::{G2gError, HardwareError};
use noalloc_pipeline::keyword::CLIP_BYTES;
use static_cell::StaticCell;
use tflm::TflmModel;

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

// holds core 1's task futures, the keyword task's alone about 61 KB, plus the frames their polls call into
const SECOND_CORE_STACK_BYTES: usize = 128 * 1024;

struct PsramBuffers {
    second_core_stack: &'static mut Stack<SECOND_CORE_STACK_BYTES>,
    speech_arena: &'static mut [MaybeUninit<u8>],
    person_arena: &'static mut [MaybeUninit<u8>],
    microphone_window: &'static mut [u8; CLIP_BYTES],
    camera_frame: &'static mut [u8; person::IMAGE_BYTES],
    handoff_frames: [&'static mut [u8; person::LABELLED_FRAME_BYTES]; 2],
}

fn psram_buffers(psram: &Psram) -> PsramBuffers {
    let (start, bytes) = psram.raw_parts();
    assert!(
        bytes
            >= SECOND_CORE_STACK_BYTES
                + keyword::ARENA_BYTES
                + person::ARENA_BYTES
                + CLIP_BYTES
                + person::IMAGE_BYTES
                + 2 * person::LABELLED_FRAME_BYTES,
        "PSRAM holds {bytes} bytes"
    );
    // SAFETY: Psram maps `bytes` bytes at `start` for the life of the program, and only this function hands them out.
    let memory = unsafe { core::slice::from_raw_parts_mut(start.cast::<MaybeUninit<u8>>(), bytes) };
    let (stack, rest) = memory.split_at_mut(SECOND_CORE_STACK_BYTES);
    assert!(stack
        .as_ptr()
        .cast::<Stack<SECOND_CORE_STACK_BYTES>>()
        .is_aligned());
    // SAFETY: the slice is aligned for Stack and as long as it, and Stack is uninitialized memory.
    let second_core_stack =
        unsafe { &mut *stack.as_mut_ptr().cast::<Stack<SECOND_CORE_STACK_BYTES>>() };
    let (speech_arena, rest) = rest.split_at_mut(keyword::ARENA_BYTES);
    let (person_arena, rest) = rest.split_at_mut(person::ARENA_BYTES);
    let (window, rest) = rest.split_at_mut(CLIP_BYTES);
    let (frame, rest) = rest.split_at_mut(person::IMAGE_BYTES);
    let (first_handoff, rest) = rest.split_at_mut(person::LABELLED_FRAME_BYTES);
    let second_handoff = &mut rest[..person::LABELLED_FRAME_BYTES];
    for byte in window
        .iter_mut()
        .chain(frame.iter_mut())
        .chain(first_handoff.iter_mut())
        .chain(second_handoff.iter_mut())
    {
        byte.write(0);
    }
    // SAFETY: every byte of the four buffers was just written.
    let (microphone_window, camera_frame, first_handoff, second_handoff) = unsafe {
        (
            &mut *window.as_mut_ptr().cast::<[u8; CLIP_BYTES]>(),
            &mut *frame.as_mut_ptr().cast::<[u8; person::IMAGE_BYTES]>(),
            &mut *first_handoff
                .as_mut_ptr()
                .cast::<[u8; person::LABELLED_FRAME_BYTES]>(),
            &mut *second_handoff
                .as_mut_ptr()
                .cast::<[u8; person::LABELLED_FRAME_BYTES]>(),
        )
    };
    PsramBuffers {
        second_core_stack,
        speech_arena,
        person_arena,
        microphone_window,
        camera_frame,
        handoff_frames: [first_handoff, second_handoff],
    }
}

// without LTO, rustc gives main one stack frame for everything inlined into it
#[inline(never)]
fn run_checks(speech: &mut TflmModel, person_model: &mut TflmModel) -> bool {
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

fn run_second_core(
    speech: &'static mut TflmModel,
    microphone: keyword::Microphone,
    mut controller: WifiController<'static>,
    handoff: &'static person::FrameHandoff,
) -> ! {
    let (stack, mut runner) = network::ip_stack();
    // embassy-executor task storage for the keyword graph is 61312 bytes
    esp_alloc::heap_allocator!(size: 64 * 1024);
    // esp32s3 compare-and-set is not atomic on PSRAM, and core 0 wakes these tasks
    let network_task = &*Box::leak(Box::new(TaskStorage::new()));
    let stream_task = &*Box::leak(Box::new(TaskStorage::new()));
    let listen_task = &*Box::leak(Box::new(TaskStorage::new()));
    static SECOND_CORE_EXECUTOR: StaticCell<Executor> = StaticCell::new();
    SECOND_CORE_EXECUTOR.init(Executor::new()).run(|spawner| {
        let networking = network_task.spawn(|| async move { runner.run().await });
        spawner.spawn(networking.expect("network task"));
        let streaming = stream_task.spawn(|| async move {
            let stopped = async {
                network::connect(&mut controller, stack).await?;
                person::stream(handoff, network::udp_sender(stack)?).await
            }
            .await;
            println!("camera stream stopped: {stopped:?}");
        });
        spawner.spawn(streaming.expect("stream task"));
        let listening = listen_task.spawn(|| async move {
            let stopped = keyword::listen(speech, microphone).await;
            println!("keyword graph stopped: {stopped:?}");
        });
        spawner.spawn(listening.expect("keyword task"));
    })
}

// without LTO, rustc gives main one stack frame for everything inlined into it
#[inline(never)]
fn run_live(
    person_model: &'static mut TflmModel,
    camera: camera::CameraGrabber,
    handoff: &'static person::FrameHandoff,
) -> ! {
    let watch_task = TaskStorage::new();
    // SAFETY: Executor::run below never returns, so this frame outlives the task.
    let watch_task = unsafe { &*ptr::from_ref(&watch_task) };
    static EXECUTOR: StaticCell<Executor> = StaticCell::new();
    EXECUTOR.init(Executor::new()).run(|spawner| {
        let watching = watch_task.spawn(|| async move {
            let stopped = person::watch(person_model, camera, handoff).await;
            println!("person graph stopped: {stopped:?}");
        });
        spawner.spawn(watching.expect("person task"));
    })
}

#[esp_hal::main]
fn main() -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
    println!("g2g on esp32s3");

    let psram = Psram::new(peripherals.PSRAM, PsramConfig::default());
    let buffers = psram_buffers(&psram);
    let mut speech = keyword::load_model(buffers.speech_arena).expect("speech model");
    let mut person_model = person::load_model(buffers.person_arena).expect("person_detect model");
    let controller = network::start(
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
    static SPEECH: StaticCell<TflmModel> = StaticCell::new();
    static PERSON_MODEL: StaticCell<TflmModel> = StaticCell::new();
    static HANDOFF: StaticCell<person::FrameHandoff> = StaticCell::new();
    let speech = SPEECH.init(speech);
    let handoff: &'static person::FrameHandoff =
        HANDOFF.init(person::FrameHandoff::new(buffers.handoff_frames));
    esp_rtos::start_second_core(
        peripherals.CPU_CTRL,
        peripherals.FROM_CPU_INTR1,
        buffers.second_core_stack,
        move || run_second_core(speech, microphone, controller, handoff),
    );
    run_live(PERSON_MODEL.init(person_model), camera, handoff)
}
