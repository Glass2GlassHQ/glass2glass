#![cfg(all(target_os = "linux", feature = "hip"))]

use std::sync::Arc;

use g2g_core::memory::{HipVendor, OwnedHipBuffer};
use g2g_core::{G2gError, HardwareError};
use g2g_plugins::hip::{
    HipDevice, HipEvent, HipKernelModule, HipMemory, HipStream, KernelArgument,
};

const SCALE_SOURCE: &str = r#"
extern "C" __global__ void scale_plus_one(const float* input, float* output, unsigned int count, float scale) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input[index] * scale + 1.0f;
    }
}
"#;
const COUNT: u32 = 1 << 20;
const BLOCK: u32 = 256;
// a power of two keeps every product exact, so fused multiply-add on the GPU rounds the same as the CPU
const SCALE: f32 = 0.5;

fn open_device(vendor: HipVendor) -> Option<Arc<HipDevice>> {
    match HipDevice::open(vendor, 0) {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("skipping: no {vendor:?} HIP device 0: {error:?}");
            None
        }
    }
}

fn compile(
    device: &Arc<HipDevice>,
    source: &str,
) -> Option<Result<Arc<HipKernelModule>, G2gError>> {
    match HipKernelModule::compile(device, source, &[]) {
        Err(G2gError::Hardware(HardwareError::LibraryNotFound(library))) => {
            eprintln!("skipping: {library} is not installed");
            None
        }
        result => Some(result),
    }
}

fn ready_buffer(stream: &HipStream, memory: HipMemory, len: usize) -> OwnedHipBuffer {
    let ready = HipEvent::new(stream.device()).unwrap();
    stream.record(&ready).unwrap();
    memory.into_buffer(ready, len).unwrap()
}

fn kernel_output_matches_the_cpu(vendor: HipVendor) {
    let Some(device) = open_device(vendor) else {
        return;
    };
    let Some(module) = compile(&device, SCALE_SOURCE) else {
        return;
    };
    let kernel = module.unwrap().kernel("scale_plus_one").unwrap();
    let input: Vec<f32> = (0..COUNT).map(|index| index as f32).collect();
    let input_bytes: Vec<u8> = input.iter().flat_map(|value| value.to_ne_bytes()).collect();
    let stream = HipStream::new(&device).unwrap();
    let input_memory = HipMemory::allocate(&device, input_bytes.len()).unwrap();
    stream.upload_blocking(&input_memory, &input_bytes).unwrap();
    let input_buffer = ready_buffer(&stream, input_memory, input_bytes.len());

    let mut output_memory = HipMemory::allocate(&device, input_bytes.len()).unwrap();
    stream
        .launch(
            &kernel,
            [COUNT.div_ceil(BLOCK), 1, 1],
            [BLOCK, 1, 1],
            &[
                KernelArgument::Read(&input_buffer),
                KernelArgument::Write(&mut output_memory),
                KernelArgument::U32(COUNT),
                KernelArgument::F32(SCALE),
            ],
        )
        .unwrap();
    let output_buffer = ready_buffer(&stream, output_memory, input_bytes.len());
    let mut output_bytes = vec![0u8; input_bytes.len()];
    HipStream::new(&device)
        .unwrap()
        .download_blocking(&output_buffer, &mut output_bytes)
        .unwrap();

    let output: Vec<f32> = output_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_ne_bytes(*bytes))
        .collect();
    let expected: Vec<f32> = input.iter().map(|value| value * SCALE + 1.0).collect();
    assert!(
        output == expected,
        "{vendor:?}: kernel output differs from the CPU"
    );
}

fn compile_errors_are_reported(vendor: HipVendor) {
    let Some(device) = open_device(vendor) else {
        return;
    };
    let Some(result) = compile(&device, r#"extern "C" __global__ void broken( {"#) else {
        return;
    };
    assert!(
        matches!(
            result,
            Err(G2gError::Hardware(HardwareError::KernelCompilation(_)))
        ),
        "{vendor:?}: {result:?}"
    );
}

#[test]
fn amd_kernel_output_matches_the_cpu() {
    kernel_output_matches_the_cpu(HipVendor::Amd);
}

#[test]
fn nvidia_kernel_output_matches_the_cpu() {
    kernel_output_matches_the_cpu(HipVendor::Nvidia);
}

#[test]
fn amd_compile_errors_are_reported() {
    compile_errors_are_reported(HipVendor::Amd);
}

#[test]
fn nvidia_compile_errors_are_reported() {
    compile_errors_are_reported(HipVendor::Nvidia);
}
