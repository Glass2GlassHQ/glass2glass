// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::time::Instant;
use esp_println::println;

// espflash refuses images without the esp-idf application descriptor.
esp_bootloader_esp_idf::esp_app_desc!();

fn report(name: &str, run: impl FnOnce() -> u64, expected: u64) -> bool {
    let start = Instant::now();
    let got = run();
    let micros = start.elapsed().as_micros();
    let matches = got == expected;
    let verdict = if matches { "OK" } else { "MISMATCH" };
    println!("{verdict} {name}: {got} (expected {expected}), {micros} us");
    matches
}

#[esp_hal::main]
fn main() -> ! {
    esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
    println!("g2g on esp32s3");

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
    let keyword = report(
        "keyword",
        || noalloc_pipeline::keyword::run_keyword().map_or(0, |run| run.digest()),
        noalloc_pipeline::keyword::EXPECTED_RUN.digest(),
    );
    let verdict = if display && audio && keyword {
        "PASS"
    } else {
        "FAIL"
    };
    println!("{verdict}");
    loop {}
}
