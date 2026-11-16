// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::ffi::{c_char, c_void, CStr};
use core::mem::MaybeUninit;

use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::mutex::Mutex;
use esp_hal::time::Instant;
use esp_println::println;
use g2g_core::error::{G2gError, HardwareError};
use g2g_mcu::{CTensorModel, TensorFormat, TensorModel};

#[repr(C, align(16))]
pub struct Aligned<T: ?Sized>(pub T);

extern "C" {
    fn tflm_load(model: *const u8, arena: *mut u8, arena_bytes: usize) -> *mut c_void;
    fn tflm_arena_used(model: *mut c_void) -> usize;
    fn tflm_infer(
        model: *mut c_void,
        input: *const u8,
        input_bytes: usize,
        output: *mut u8,
        output_bytes: usize,
    ) -> i32;
}

#[no_mangle]
extern "C" fn tflm_log(text: *const c_char) {
    // SAFETY: glue.cc and esp-nn pass NUL-terminated strings.
    let text = unsafe { CStr::from_ptr(text) };
    println!("tflm: {}", text.to_str().unwrap_or("(not utf-8)"));
}

#[no_mangle]
extern "C" fn esp_timer_get_time() -> i64 {
    Instant::now().duration_since_epoch().as_micros() as i64
}

#[no_mangle]
extern "C" fn abort() -> ! {
    panic!("TFLite Micro called abort");
}

// esp-nn keeps the conv, depthwise conv and softmax scratch pointers in C globals
static INVOKE: Mutex<CriticalSectionRawMutex, ()> = Mutex::new(());

pub struct TflmModel(CTensorModel);

// SAFETY: a TFLite Micro interpreter has no tie to the core that loaded it, and INVOKE keeps two from running esp-nn at once.
unsafe impl Send for TflmModel {}

impl TensorModel for TflmModel {
    fn input(&self) -> TensorFormat {
        self.0.input()
    }

    fn output(&self) -> TensorFormat {
        self.0.output()
    }

    async fn run(&mut self, input: &[u8], output: &mut [u8]) -> Result<(), G2gError> {
        let _invoking = INVOKE.lock().await;
        self.0.run(input, output).await
    }
}

pub fn load(
    name: &str,
    model: &'static [u8],
    arena: &'static mut [MaybeUninit<u8>],
    input: TensorFormat,
    output: TensorFormat,
) -> Result<TflmModel, G2gError> {
    // SAFETY: the model and the arena are 'static, and TFLite Micro writes arena bytes before reading them.
    let interpreter = unsafe { tflm_load(model.as_ptr(), arena.as_mut_ptr().cast(), arena.len()) };
    if interpreter.is_null() {
        return Err(G2gError::Hardware(HardwareError::Peripheral));
    }
    // SAFETY: `interpreter` came from tflm_load.
    let used = unsafe { tflm_arena_used(interpreter) };
    println!("{name}: arena {used} of {} bytes", arena.len());
    // SAFETY: tflm_infer checks both lengths against the model's tensors before touching them.
    Ok(TflmModel(unsafe {
        CTensorModel::new(tflm_infer, interpreter, input, output)
    }))
}
