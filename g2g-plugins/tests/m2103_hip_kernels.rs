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

const DELAYED_COPY_SOURCE: &str = r#"
extern "C" __global__ void delayed_copy(const unsigned int* input, unsigned int* output, unsigned int count, unsigned int delay_cycles) {
    long long start = clock64();
    while (clock64() - start < delay_cycles) {
    }
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input[index];
    }
}
"#;
// about 100 ms at the shader clocks of the RTX 3060 and the Radeon 680M
const DELAY_CYCLES: u32 = 200_000_000;
// few enough threads to run in at most two waves on the Radeon 680M
const DELAYED_COUNT: u32 = 1 << 12;

fn read_buffers_stay_allocated_while_the_kernel_runs(vendor: HipVendor) {
    let Some(device) = open_device(vendor) else {
        return;
    };
    let Some(module) = compile(&device, DELAYED_COPY_SOURCE) else {
        return;
    };
    let kernel = module.unwrap().kernel("delayed_copy").unwrap();
    let pattern: Vec<u8> = (0..DELAYED_COUNT)
        .flat_map(|index| (index * 7 + 1).to_ne_bytes())
        .collect();
    let stream = HipStream::new(&device).unwrap();
    let input_memory = HipMemory::allocate(&device, pattern.len()).unwrap();
    stream.upload_blocking(&input_memory, &pattern).unwrap();
    let input_buffer = ready_buffer(&stream, input_memory, pattern.len());
    let mut output_memory = HipMemory::allocate(&device, pattern.len()).unwrap();
    stream
        .launch(
            &kernel,
            [DELAYED_COUNT.div_ceil(BLOCK), 1, 1],
            [BLOCK, 1, 1],
            &[
                KernelArgument::Read(&input_buffer),
                KernelArgument::Write(&mut output_memory),
                KernelArgument::U32(DELAYED_COUNT),
                KernelArgument::U32(DELAY_CYCLES),
            ],
        )
        .unwrap();

    let input_pointer = input_buffer.device_pointer;
    drop(input_buffer);
    // gives the release thread time to free the input if nothing holds it
    std::thread::sleep(std::time::Duration::from_millis(10));
    let overwrite = HipStream::new(&device).unwrap();
    let reused = HipMemory::allocate(&device, pattern.len()).unwrap();
    // an allocation still held cannot be handed out again, and on the NVIDIA driver a freed one is
    assert_ne!(
        reused.device_pointer(),
        input_pointer,
        "{vendor:?}: the input was freed while the kernel ran"
    );
    overwrite
        .upload_blocking(&reused, &vec![0u8; pattern.len()])
        .unwrap();

    let output_buffer = ready_buffer(&stream, output_memory, pattern.len());
    let mut output = vec![0u8; pattern.len()];
    overwrite
        .download_blocking(&output_buffer, &mut output)
        .unwrap();
    assert!(
        output == pattern,
        "{vendor:?}: the kernel read its input after it was freed"
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

#[test]
fn amd_read_buffers_stay_allocated_while_the_kernel_runs() {
    read_buffers_stay_allocated_while_the_kernel_runs(HipVendor::Amd);
}

#[test]
fn nvidia_read_buffers_stay_allocated_while_the_kernel_runs() {
    read_buffers_stay_allocated_while_the_kernel_runs(HipVendor::Nvidia);
}
