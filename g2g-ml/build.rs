const LIBRARY_DIRECTORY_VARIABLE: &str = "EXECUTORCH_RS_EXECUTORCH_LIB_DIR";

// the `executorch` crate links the runtime but not the operator kernels a program calls
fn main() {
    if std::env::var_os("CARGO_FEATURE_EXECUTORCH").is_none() {
        return;
    }
    println!("cargo::rerun-if-env-changed={LIBRARY_DIRECTORY_VARIABLE}");
    match std::env::var(LIBRARY_DIRECTORY_VARIABLE) {
        Ok(library_directory) => {
            println!("cargo::rustc-link-search=native={library_directory}/kernels/portable/")
        }
        Err(_) => println!(
            "cargo::warning={LIBRARY_DIRECTORY_VARIABLE} is not set, build the ExecuTorch libraries with tools/executorch-build.sh"
        ),
    }
    println!("cargo::rustc-link-lib=static:+whole-archive=portable_kernels");
    println!("cargo::rustc-link-lib=static:+whole-archive=portable_ops_lib");
}
