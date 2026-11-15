// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![no_std]
#![no_main]

use core::cell::Cell;

use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::dma::DmaRxStreamBuf;
use esp_hal::dma_rx_stream_buffer;
use esp_hal::i2s::master::{I2s, I2sRxDmaTransfer, PdmConfig, PdmRxConfig, PdmSlotMode};
use esp_hal::time::{Instant, Rate};
use esp_hal::Blocking;
use esp_println::println;
use g2g_core::error::{G2gError, HardwareError};
use g2g_mcu::FrameGrabber;
use noalloc_pipeline::keyword::{run_keyword_graph, CLIP_BYTES, KEYWORDS, LABELS};
use static_cell::StaticCell;

// espflash refuses images without the esp-idf application descriptor.
esp_bootloader_esp_idf::esp_app_desc!();

const SAMPLE_RATE_HZ: u32 = 16_000;
const SAMPLE_BYTES: usize = 2;
const STRIDE_MILLISECONDS: usize = 500;
const STRIDE_BYTES: usize = SAMPLE_RATE_HZ as usize * STRIDE_MILLISECONDS / 1000 * SAMPLE_BYTES;
const STRIDE_NS: u64 = STRIDE_MILLISECONDS as u64 * 1_000_000;
// esp-hal DMA descriptors carry at most 4095 bytes.
const DMA_CHUNK_BYTES: usize = 4000;
const DMA_BYTES: usize = CLIP_BYTES;

fn report(name: &str, run: impl FnOnce() -> u64, expected: u64) -> bool {
    let start = Instant::now();
    let got = run();
    let micros = start.elapsed().as_micros();
    let matches = got == expected;
    let verdict = if matches { "OK" } else { "MISMATCH" };
    println!("{verdict} {name}: {got} (expected {expected}), {micros} us");
    matches
}

// Each capture slides a one-second window forward by STRIDE_BYTES of new audio.
struct MicWindow<'d, 'a> {
    transfer: I2sRxDmaTransfer<'d, Blocking, DmaRxStreamBuf>,
    window: &'static mut [u8; CLIP_BYTES],
    filled: bool,
    peak: &'a Cell<i16>,
}

impl MicWindow<'_, '_> {
    fn read_exact(&mut self, from: usize) -> Result<(), G2gError> {
        let mut at = from;
        while at < CLIP_BYTES {
            // esp-hal stops the DMA when every descriptor is full and unread.
            if self.transfer.is_done() {
                return Err(G2gError::Hardware(HardwareError::Peripheral));
            }
            at += self.transfer.pop(&mut self.window[at..]);
        }
        Ok(())
    }
}

impl FrameGrabber for MicWindow<'_, '_> {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        if self.filled {
            self.window.copy_within(STRIDE_BYTES.., 0);
            self.read_exact(CLIP_BYTES - STRIDE_BYTES)?;
        } else {
            self.read_exact(0)?;
            self.filled = true;
        }
        let Some(slot) = buf.get_mut(..CLIP_BYTES) else {
            return Err(G2gError::CapsMismatch);
        };
        slot.copy_from_slice(self.window.as_slice());
        let peak = self
            .window
            .chunks_exact(SAMPLE_BYTES)
            .map(|sample| i16::from_le_bytes([sample[0], sample[1]]).saturating_abs())
            .max()
            .unwrap_or(0);
        self.peak.set(peak);
        Ok(CLIP_BYTES)
    }
}

fn print_scores(scores: [f32; KEYWORDS], peak: i16) {
    let mut best = 0;
    for index in 1..KEYWORDS {
        if scores[index] > scores[best] {
            best = index;
        }
    }
    println!(
        "{:<8} silence {:.2} unknown {:.2} yes {:.2} no {:.2} peak {peak}",
        LABELS[best], scores[0], scores[1], scores[2], scores[3]
    );
}

// without LTO, the keyword graph inlined into main overflows the 287 KB main stack.
#[inline(never)]
fn listen(peripherals: esp_hal::peripherals::Peripherals) {
    static WINDOW: StaticCell<[u8; CLIP_BYTES]> = StaticCell::new();
    let pdm = PdmConfig::rx_only(PdmRxConfig::new_pcm_default(
        Rate::from_hz(SAMPLE_RATE_HZ),
        PdmSlotMode::Mono,
    ));
    let i2s = I2s::new_pdm(peripherals.I2S0, peripherals.DMA_CH0, pdm).expect("i2s pdm config");
    // XIAO ESP32S3 Sense: the PDM microphone clock is GPIO42 and its data is GPIO41.
    let i2s_rx = i2s
        .i2s_rx
        .with_clk(peripherals.GPIO42)
        .with_din(peripherals.GPIO41)
        .build();
    let transfer = i2s_rx
        .read(dma_rx_stream_buffer!(DMA_BYTES, DMA_CHUNK_BYTES))
        .map_err(|(error, _, _)| error)
        .expect("i2s dma start");

    let peak = Cell::new(0);
    let microphone = MicWindow {
        transfer,
        window: WINDOW.init([0; CLIP_BYTES]),
        filled: false,
        peak: &peak,
    };
    println!("listening: say yes or no");
    let result = g2g_core::drive_ready(run_keyword_graph(microphone, None, STRIDE_NS, |scores| {
        print_scores(scores, peak.get())
    }));
    println!("keyword graph stopped: {result:?}");
}

#[esp_hal::main]
fn main() -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
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

    listen(peripherals);
    loop {}
}
