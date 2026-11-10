// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "hip"))]

use std::sync::Arc;

use g2g_core::memory::{GpuVendor, MemoryDomain, MemoryDomainKind};
use g2g_plugins::hip::{HipDevice, HipEvent, HipMemory, HipStream};

fn open_device(vendor: GpuVendor) -> Option<Arc<HipDevice>> {
    match HipDevice::open(vendor, 0) {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("skipping: no {vendor:?} HIP device 0: {error:?}");
            None
        }
    }
}

fn bytes_reach_a_second_stream_through_a_shared_buffer(vendor: GpuVendor) {
    let Some(device) = open_device(vendor) else {
        return;
    };
    let pattern: Vec<u8> = (0..1 << 20)
        .map(|index: u32| (index * 7 % 251) as u8)
        .collect();
    let producer = HipStream::new(&device).unwrap();
    let memory = HipMemory::allocate(&device, pattern.len() + 4096).unwrap();
    producer.upload_blocking(&memory, &pattern).unwrap();
    let ready = HipEvent::new(&device).unwrap();
    producer.record(&ready).unwrap();
    let domain = MemoryDomain::Hip(memory.into_buffer(ready, pattern.len()).unwrap());
    assert_eq!(domain.kind(), MemoryDomainKind::Hip);

    let shared = domain.share();
    drop(domain);
    drop(producer);
    let MemoryDomain::Hip(buffer) = &shared else {
        panic!("share kept the domain, got {:?}", shared.kind());
    };
    assert_eq!((buffer.vendor, buffer.len), (vendor, pattern.len()));
    let consumer = HipStream::new(&device).unwrap();
    let mut read_back = vec![0u8; pattern.len()];
    consumer.download_blocking(buffer, &mut read_back).unwrap();
    assert!(
        read_back == pattern,
        "{vendor:?}: bytes changed between streams"
    );
}

// more memory in total than any test GPU has, so a buffer that is never freed runs out
const RELEASE_ALLOCATION: usize = 256 << 20;
const RELEASE_ROUNDS: usize = 200;

fn dropped_buffers_are_freed(vendor: GpuVendor) {
    let Some(device) = open_device(vendor) else {
        return;
    };
    let stream = HipStream::new(&device).unwrap();
    for round in 0..RELEASE_ROUNDS {
        let memory = HipMemory::allocate(&device, RELEASE_ALLOCATION)
            .unwrap_or_else(|error| panic!("{vendor:?} round {round}: {error:?}"));
        let ready = HipEvent::new(&device).unwrap();
        stream.record(&ready).unwrap();
        let domain = MemoryDomain::Hip(memory.into_buffer(ready, RELEASE_ALLOCATION).unwrap());
        drop(domain.share());
    }
}

#[test]
fn amd_bytes_reach_a_second_stream_through_a_shared_buffer() {
    bytes_reach_a_second_stream_through_a_shared_buffer(GpuVendor::Amd);
}

#[test]
fn nvidia_bytes_reach_a_second_stream_through_a_shared_buffer() {
    bytes_reach_a_second_stream_through_a_shared_buffer(GpuVendor::Nvidia);
}

#[test]
fn amd_dropped_buffers_are_freed() {
    dropped_buffers_are_freed(GpuVendor::Amd);
}

#[test]
fn nvidia_dropped_buffers_are_freed() {
    dropped_buffers_are_freed(GpuVendor::Nvidia);
}
