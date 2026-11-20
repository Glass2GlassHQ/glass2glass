// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use noalloc_pipeline::EXPECTED_CHECKSUM;

// ipc-shmem-2 in the board's device tree, readable from Linux through /dev/mem
const SHARED_REPORT_ADDRESS: usize = 0x200C_6000;
const REPORT_MAGIC: u32 = u32::from_le_bytes(*b"g2g0");

#[repr(C)]
struct SharedReport {
    magic: u32,
    runs: u32,
    wrong_checksums: u32,
    last_checksum_low: u32,
    last_checksum_high: u32,
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[entry]
fn main() -> ! {
    let report = SHARED_REPORT_ADDRESS as *mut SharedReport;
    let mut runs: u32 = 0;
    let mut wrong_checksums: u32 = 0;
    loop {
        let checksum = noalloc_pipeline::run();
        runs = runs.wrapping_add(1);
        if checksum != EXPECTED_CHECKSUM {
            wrong_checksums = wrong_checksums.wrapping_add(1);
        }
        // SAFETY: the M0+ is the only writer of ipc-shmem-2, a fixed 8 KiB region the
        // device tree reserves for it, and the report is word-aligned and fits in it.
        unsafe {
            report.write_volatile(SharedReport {
                magic: REPORT_MAGIC,
                runs,
                wrong_checksums,
                last_checksum_low: checksum as u32,
                last_checksum_high: (checksum >> 32) as u32,
            });
        }
    }
}
