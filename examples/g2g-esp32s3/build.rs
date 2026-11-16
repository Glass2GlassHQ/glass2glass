// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::fs;
use std::path::{Path, PathBuf};

const CXX: &str = "xtensa-esp32s3-elf-g++";
const CC: &str = "xtensa-esp32s3-elf-gcc";
const AR: &str = "xtensa-esp32s3-elf-ar";

fn files_with_extension(dir: &Path, extension: &str) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("{}: {error}", dir.display()))
        .map(|entry| entry.expect("directory entry").path())
        .filter(|path| path.extension().is_some_and(|found| found == extension))
        .collect();
    files.sort();
    files
}

fn main() {
    let manifest =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let tflm = manifest.join("third_party/esp-tflite-micro");
    let nn = manifest.join("third_party/esp-nn");
    assert!(
        tflm.is_dir() && nn.is_dir(),
        "{} is missing, run tools/esp-tflm-fetch.sh",
        manifest.join("third_party").display()
    );
    let glue = manifest.join("tflm");
    println!("cargo:rerun-if-changed={}", glue.display());
    println!("cargo:rustc-env=ESP_TFLM_DIR={}", tflm.display());

    let kernels = tflm.join("tensorflow/lite/micro/kernels");
    let esp_nn_kernels = files_with_extension(&kernels.join("esp_nn"), "cc");
    let replaced: Vec<_> = esp_nn_kernels
        .iter()
        .filter_map(|path| path.file_name())
        .collect();
    let mut sources: Vec<PathBuf> = fs::read_to_string(glue.join("micro_sources.txt"))
        .expect("micro_sources.txt")
        .lines()
        .map(|line| tflm.join(line))
        .collect();
    sources.extend(
        files_with_extension(&kernels, "cc")
            .into_iter()
            .filter(|path| {
                !path
                    .file_name()
                    .is_some_and(|name| replaced.contains(&name))
            }),
    );
    sources.extend(esp_nn_kernels.iter().cloned());
    sources.push(glue.join("glue.cc"));
    sources.push(glue.join("bare_metal.cc"));

    let shim = glue.join("shim");
    let sdkconfig = shim.join("sdkconfig.h");
    cc::Build::new()
        .cpp(true)
        .compiler(CXX)
        .archiver(AR)
        .cpp_link_stdlib(None)
        .files(&sources)
        .include(&tflm)
        .include(tflm.join("third_party/flatbuffers/include"))
        .include(tflm.join("third_party/kissfft"))
        .include(&shim)
        .include(nn.join("include"))
        .include(nn.join("src/common"))
        .flag("-include")
        .flag(sdkconfig.to_str().expect("utf-8 path"))
        .flag("-O3")
        .flag("-mlongcalls")
        .flag("-std=gnu++17")
        .flag("-fno-rtti")
        .flag("-fno-exceptions")
        .flag("-fno-threadsafe-statics")
        .flag("-fno-unwind-tables")
        .flag("-Wno-unused-parameter")
        .define("TF_LITE_STATIC_MEMORY", None)
        .define("TF_LITE_DISABLE_X86_NEON", None)
        .define("FLATBUFFERS_LOCALE_INDEPENDENT", "0")
        .define("ESP_NN", None)
        .warnings(false)
        .compile("tflm");

    let nn_sources: Vec<PathBuf> = walk(&nn.join("src"))
        .into_iter()
        .filter(|path| {
            let name = path.to_string_lossy();
            !name.contains("riscv") && (name.ends_with(".c") || name.ends_with(".S"))
        })
        .collect();
    cc::Build::new()
        .compiler(CC)
        .archiver(AR)
        .files(&nn_sources)
        .include(&shim)
        .include(nn.join("include"))
        .include(nn.join("src/common"))
        .flag("-include")
        .flag(sdkconfig.to_str().expect("utf-8 path"))
        .flag("-O2")
        .flag("-fno-unroll-loops")
        .flag("-mlongcalls")
        .warnings(false)
        .compile("esp_nn");

    // esp-radio's linker script PROVIDE of gettimeofday loses to newlib's copy in -lc
    println!("cargo:rustc-link-arg=-Wl,--defsym=gettimeofday=__esp_radio_gettimeofday");
    println!("cargo:rustc-link-arg=-Wl,--start-group");
    println!("cargo:rustc-link-arg=-lm");
    println!("cargo:rustc-link-arg=-lc");
    println!("cargo:rustc-link-arg=-lgcc");
    println!("cargo:rustc-link-arg=-Wl,--end-group");
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for entry in fs::read_dir(dir).unwrap_or_else(|error| panic!("{}: {error}", dir.display())) {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            found.extend(walk(&path));
        } else {
            found.push(path);
        }
    }
    found.sort();
    found
}
