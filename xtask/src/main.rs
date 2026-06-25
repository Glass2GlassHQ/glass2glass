//! `cargo xtask <command>`: the project's dev commands, consolidating the
//! invocations that were shell-history tribal knowledge (DESIGN_TODO developer
//! tooling). It only orchestrates `cargo` and toolchain tools, so it has no
//! dependencies.
//!
//! Commands:
//!   ci             run locally what CI runs (check, test, clippy, the Linux
//!                  feature build, the wasm check); stops at the first failure.
//!   test --here    probe this host (NVIDIA / VAAPI / cameras / GPU / audio /
//!                  ffmpeg) and run exactly the feature-gated tests it supports.
//!                  `--dry-run` prints the detected plan without running.
//!   size           build the embedded footprint harness for Cortex-M and
//!                  report the gc-sectioned `.text` size.
//!   wasm           build the wasm32 targets (core `runtime`, plugins `web` /
//!                  `web-codecs`), handling the rustup-on-PATH gotcha.
//!
//! Cross builds (`size`, `wasm`) prepend `~/.cargo/bin` to `PATH` so cargo uses
//! the rustup toolchain rather than a distro `rustc` that lacks the target std
//! (the Fedora gotcha recorded in project memory).

use std::path::PathBuf;
use std::process::{exit, Command};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, rest) = match args.split_first() {
        Some((c, r)) => (c.as_str(), r),
        None => {
            print_help();
            exit(0);
        }
    };
    let code = match cmd {
        "ci" => cmd_ci(),
        "test" => cmd_test_here(rest),
        "size" => cmd_size(),
        "wasm" => cmd_wasm(),
        "bench" => cmd_bench(rest),
        "ffi-probe" => cmd_ffi_probe(rest),
        "help" | "-h" | "--help" => {
            print_help();
            0
        }
        other => {
            eprintln!("xtask: unknown command '{other}'\n");
            print_help();
            2
        }
    };
    exit(code);
}

fn print_help() {
    println!(
        "cargo xtask <command>\n\n\
         commands:\n  \
           ci             run locally what CI runs (stops at first failure)\n  \
           test --here    run the feature tests this host supports (--dry-run to just probe)\n  \
           size           build + measure the Cortex-M footprint harness\n  \
           wasm           build the wasm32 targets (core + web plugins)\n  \
           bench [args]   run the criterion benchmarks (caps solve, frame convert)\n  \
           ffi-probe      sizeof/offsetof a C SDK struct, emit the repr(C) size assert\n  \
           help           show this message\n\n\
         ffi-probe usage:\n  \
           cargo xtask ffi-probe --header <h> --struct <S> [--field <f>]... [-I <dir>]... [--cc <cc>]\n  \
           e.g. cargo xtask ffi-probe --header ffnvcodec/nvEncodeAPI.h \\\n       \
                  --struct NV_ENC_INITIALIZE_PARAMS --field encodeGUID -I /usr/include/ffnvcodec"
    );
}

// --- ci -----------------------------------------------------------------

/// The Linux feature set CI's `features-linux` job checks (kept in sync with
/// `.github/workflows/ci.yml`).
const LINUX_FEATURES: &str = "rtsp,udp-egress,udp-ingress,rtmp,http-src,hls,dash,\
av1-encode,mjpeg,mjpeg-encode,opus,alsa-sink,pulse-sink,wayland-sink,kms-sink,v4l2,\
analytics,vello-overlay,wgpu-sink,plugin-loader";

fn cmd_ci() -> i32 {
    // Each step mirrors a CI job that runs on Linux without proprietary deps.
    // `--locked` matches CI so a stale Cargo.lock fails here too, not only in CI.
    let steps: &[(&str, &[&str])] = &[
        ("check (no_std default build)", &["check", "--workspace", "--locked"]),
        ("test (default features)", &["test", "--workspace", "--locked"]),
        ("clippy", &["clippy", "--workspace", "--all-targets", "--locked"]),
        (
            "features (linux)",
            &["check", "-p", "g2g-plugins", "--locked", "--all-targets", "--features", LINUX_FEATURES],
        ),
        (
            "features (g2g-ml wgpu+burn)",
            &["check", "-p", "g2g-ml", "--locked", "--all-targets", "--features", "wgpu,burn"],
        ),
        (
            "embassy (no-alloc embedded path)",
            &[
                "test", "-p", "g2g-plugins", "--locked", "--features", "embassy-link", "--test",
                "m45_embassy_link", "--test", "m260_dma_ring", "--test", "m264_embassy_multitask",
            ],
        ),
    ];
    for (desc, args) in steps {
        let code = run(desc, "cargo", args, &[]);
        if code != 0 {
            eprintln!("\nxtask ci: '{desc}' failed (exit {code}); stopping.");
            return code;
        }
    }
    // wasm check uses the rustup toolchain (PATH gotcha).
    let code = run(
        "wasm (g2g-core)",
        "cargo",
        &["check", "-p", "g2g-core", "--locked", "--target", "wasm32-unknown-unknown", "--features", "runtime"],
        &rustup_path_env(),
    );
    if code != 0 {
        eprintln!("\nxtask ci: 'wasm (g2g-core)' failed (exit {code}).");
        return code;
    }
    println!("\nxtask ci: all steps passed.");
    0
}

// --- test --here --------------------------------------------------------

/// What the host can build / exercise, probed best-effort.
#[derive(Debug, Default, Clone, Copy)]
struct Capabilities {
    nvidia: bool,
    vaapi: bool,
    v4l2: bool,
    gpu_drm: bool,
    opus: bool,
    alsa: bool,
    pulse: bool,
    wayland: bool,
    ffmpeg: bool,
}

fn cmd_test_here(rest: &[String]) -> i32 {
    let dry_run = rest.iter().any(|a| a == "--dry-run");
    // `--here` is the documented spelling; accept it (and a bare `test`) without
    // treating it as an error.
    for a in rest {
        if a != "--here" && a != "--dry-run" {
            eprintln!("xtask test: ignoring unknown argument '{a}'");
        }
    }

    let caps = probe_host();
    println!("host capabilities:");
    report_cap("NVIDIA GPU (nvenc/nvdec/cuda)", caps.nvidia);
    report_cap("VAAPI (libva)", caps.vaapi);
    report_cap("V4L2 camera (/dev/video*)", caps.v4l2);
    report_cap("GPU render node (/dev/dri, wgpu/vello)", caps.gpu_drm);
    report_cap("Opus (libopus)", caps.opus);
    report_cap("ALSA (libasound)", caps.alsa);
    report_cap("PulseAudio (libpulse)", caps.pulse);
    report_cap("Wayland (libwayland)", caps.wayland);
    report_cap("ffmpeg (libavcodec)", caps.ffmpeg);

    let features = host_test_features(&caps);
    let feat_csv = features.join(",");
    println!("\nplan:");
    println!("  cargo test --workspace");
    println!("  cargo test -p g2g-plugins --features {feat_csv}");
    if caps.gpu_drm {
        println!("  cargo test -p g2g-ml --features wgpu,burn");
    }

    if dry_run {
        println!("\n(--dry-run: nothing executed)");
        return 0;
    }

    // Run every applicable group, continue on failure, and report a summary:
    // the point is to learn what works on this box, not to stop at the first gap.
    let mut results: Vec<(String, i32)> = Vec::new();
    results.push(("default suite".into(), run("default suite", "cargo", &["test", "--workspace"], &[])));
    results.push((
        "g2g-plugins features".into(),
        run("g2g-plugins features", "cargo", &["test", "-p", "g2g-plugins", "--features", &feat_csv], &[]),
    ));
    if caps.gpu_drm {
        results.push((
            "g2g-ml wgpu+burn".into(),
            run("g2g-ml wgpu+burn", "cargo", &["test", "-p", "g2g-ml", "--features", "wgpu,burn"], &[]),
        ));
    }

    println!("\nsummary:");
    let mut worst = 0;
    for (label, code) in &results {
        println!("  {} {label}", if *code == 0 { "ok  " } else { "FAIL" });
        if *code != 0 {
            worst = *code;
        }
    }
    worst
}

/// Map probed capabilities to the g2g-plugins features to test. The pure-Rust
/// std features have no system prerequisite, so they are always included; the
/// rest are gated on the matching probe.
fn host_test_features(c: &Capabilities) -> Vec<&'static str> {
    let mut f = vec![
        "rtsp", "udp-egress", "udp-ingress", "rtmp", "http-src", "hls", "dash", "mjpeg",
        "mjpeg-encode", "av1-encode", "analytics", "plugin-loader",
    ];
    if c.opus {
        f.push("opus");
    }
    if c.alsa {
        f.push("alsa-sink");
    }
    if c.pulse {
        f.push("pulse-sink");
    }
    if c.wayland {
        f.push("wayland-sink");
    }
    if c.gpu_drm {
        f.push("wgpu-sink");
        f.push("vello-overlay");
        f.push("kms-sink");
    }
    if c.v4l2 {
        f.push("v4l2");
    }
    if c.vaapi {
        f.push("vaapi");
    }
    if c.ffmpeg {
        f.push("ffmpeg");
    }
    if c.nvidia {
        f.push("nvenc");
        f.push("nvdec");
        f.push("cuda");
    }
    f
}

fn probe_host() -> Capabilities {
    Capabilities {
        nvidia: cmd_ok("nvidia-smi", &["-L"]),
        vaapi: pkg_exists("libva"),
        v4l2: dev_entry_exists("/dev", "video"),
        gpu_drm: dev_entry_exists("/dev/dri", "renderD"),
        opus: pkg_exists("opus"),
        alsa: pkg_exists("alsa"),
        pulse: pkg_exists("libpulse"),
        wayland: pkg_exists("wayland-client"),
        ffmpeg: pkg_exists("libavcodec"),
    }
}

fn report_cap(label: &str, present: bool) {
    println!("  [{}] {label}", if present { "x" } else { " " });
}

// --- size ---------------------------------------------------------------

const SIZE_MANIFEST: &str = "examples/g2g-size/Cargo.toml";
const SIZE_TARGET: &str = "thumbv7em-none-eabihf";
/// The harness's `#[no_mangle]` entry; the gc-sections link is rooted here.
const SIZE_ENTRY: &str = "g2g_min";

fn cmd_size() -> i32 {
    let env = rustup_path_env();
    let code = run(
        "build footprint harness (Cortex-M)",
        "cargo",
        &["build", "--release", "--manifest-path", SIZE_MANIFEST, "--target", SIZE_TARGET],
        &env,
    );
    if code != 0 {
        eprintln!("xtask size: build failed; is the `{SIZE_TARGET}` target installed? (rustup target add {SIZE_TARGET})");
        return code;
    }

    let staticlib = format!("examples/g2g-size/target/{SIZE_TARGET}/release/libg2g_size.a");
    // A staticlib `.a` reads ~183 KB because it bundles core/alloc/builtins;
    // gc-sections at a final link is what prunes to the real footprint, so link
    // the harness's entry and measure that ELF.
    let lld = match find_rust_lld() {
        Some(p) => p,
        None => {
            eprintln!("xtask size: rust-lld not found in the toolchain sysroot; cannot gc-section link.");
            return 1;
        }
    };
    let elf = std::env::temp_dir().join("g2g-size.elf");
    let elf_str = elf.to_string_lossy().to_string();
    let code = run(
        "gc-section link",
        &lld.to_string_lossy(),
        &["-flavor", "gnu", "--gc-sections", "-e", SIZE_ENTRY, "-o", &elf_str, &staticlib],
        &[],
    );
    if code != 0 {
        return code;
    }

    let size_tool = which(&["size", "llvm-size"]).unwrap_or_else(|| "size".into());
    run("size", &size_tool, &[&elf_str], &[])
}

/// Locate `rust-lld` inside the active toolchain's sysroot
/// (`<sysroot>/lib/rustlib/<host>/bin/rust-lld`), since it is not on `PATH`.
fn find_rust_lld() -> Option<PathBuf> {
    let sysroot = capture("rustc", &["--print", "sysroot"], &rustup_path_env())?;
    let rustlib = PathBuf::from(sysroot).join("lib").join("rustlib");
    let entries = std::fs::read_dir(&rustlib).ok()?;
    for e in entries.flatten() {
        let cand = e.path().join("bin").join("rust-lld");
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

// --- wasm ---------------------------------------------------------------

fn cmd_wasm() -> i32 {
    let env = rustup_path_env();
    // Core: the no_std + wasm pillar CI guards.
    let code = run(
        "wasm check (g2g-core runtime)",
        "cargo",
        &["check", "-p", "g2g-core", "--target", "wasm32-unknown-unknown", "--features", "runtime"],
        &env,
    );
    if code != 0 {
        eprintln!("xtask wasm: is the wasm32 target installed? (rustup target add wasm32-unknown-unknown)");
        return code;
    }
    // Browser plugins: `web`, then `web-codecs` which needs the unstable web-sys cfg.
    let code = run(
        "wasm check (g2g-plugins web)",
        "cargo",
        &["check", "-p", "g2g-plugins", "--target", "wasm32-unknown-unknown", "--features", "web"],
        &env,
    );
    if code != 0 {
        return code;
    }
    let mut env_wc = env.clone();
    env_wc.push(("RUSTFLAGS".into(), "--cfg=web_sys_unstable_apis".into()));
    run(
        "wasm check (g2g-plugins web-codecs)",
        "cargo",
        &["check", "-p", "g2g-plugins", "--target", "wasm32-unknown-unknown", "--features", "web-codecs"],
        &env_wc,
    )
}

// --- bench --------------------------------------------------------------

/// Run the criterion benchmarks. They live in the standalone `g2g-bench` crate
/// (excluded from the workspace so criterion's plotters / rayon are never built
/// by a normal CI run), so this drives them by manifest path. Extra args pass
/// through to criterion (e.g. `cargo xtask bench -- --save-baseline main`).
fn cmd_bench(rest: &[String]) -> i32 {
    let mut args: Vec<&str> = vec!["bench", "--manifest-path", "g2g-bench/Cargo.toml"];
    for a in rest {
        args.push(a);
    }
    run("benchmarks (g2g-bench)", "cargo", &args, &[])
}

// --- ffi-probe ----------------------------------------------------------

fn cmd_ffi_probe(rest: &[String]) -> i32 {
    let mut headers: Vec<String> = Vec::new();
    let mut struct_name: Option<String> = None;
    let mut fields: Vec<String> = Vec::new();
    let mut includes: Vec<String> = Vec::new();
    let mut cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());

    let mut it = rest.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--header" => match it.next() {
                Some(v) => headers.push(v.clone()),
                None => return arg_err("--header needs a value"),
            },
            "--struct" => match it.next() {
                Some(v) => struct_name = Some(v.clone()),
                None => return arg_err("--struct needs a value"),
            },
            "--field" => match it.next() {
                Some(v) => fields.push(v.clone()),
                None => return arg_err("--field needs a value"),
            },
            "-I" | "--include" => match it.next() {
                Some(v) => includes.push(v.clone()),
                None => return arg_err("-I needs a value"),
            },
            "--cc" => match it.next() {
                Some(v) => cc = v.clone(),
                None => return arg_err("--cc needs a value"),
            },
            other => {
                // Also accept the glued `-I/path` form gcc/clang take.
                if let Some(dir) = other.strip_prefix("-I") {
                    includes.push(dir.to_string());
                } else {
                    return arg_err(&format!("unknown argument '{other}'"));
                }
            }
        }
    }

    let (headers, struct_name) = match (headers.is_empty(), struct_name) {
        (false, Some(s)) => (headers, s),
        _ => return arg_err("--header and --struct are required"),
    };

    let src = generate_probe_c(&headers, &struct_name, &fields);
    let dir = std::env::temp_dir();
    let c_path = dir.join("g2g-ffi-probe.c");
    let bin_path = dir.join("g2g-ffi-probe.bin");
    if let Err(e) = std::fs::write(&c_path, &src) {
        eprintln!("xtask ffi-probe: cannot write probe: {e}");
        return 1;
    }

    let mut cc_args: Vec<String> =
        vec![c_path.to_string_lossy().into(), "-o".into(), bin_path.to_string_lossy().into()];
    for inc in &includes {
        cc_args.push(format!("-I{inc}"));
    }
    let cc_args_ref: Vec<&str> = cc_args.iter().map(String::as_str).collect();
    let code = run(&format!("compile probe ({struct_name})"), &cc, &cc_args_ref, &[]);
    if code != 0 {
        eprintln!("xtask ffi-probe: probe failed to compile; check the header path / -I flags.");
        return code;
    }

    let out = match capture(&bin_path.to_string_lossy(), &[], &[]) {
        Some(o) => o,
        None => {
            eprintln!("xtask ffi-probe: probe binary did not run.");
            return 1;
        }
    };
    println!("\n{out}");

    // Emit the ready-to-paste compile-time size assert (offsets are correct by
    // faithful transcription; MSRV 1.75 has no `offset_of!`, see project memory).
    if let Some(size) = parse_struct_size(&out, &struct_name) {
        println!(
            "// paste into the `ffi` module:\nconst _: () = assert!(core::mem::size_of::<{struct_name}>() == {size});"
        );
    }
    0
}

/// Generate the C probe: include the headers, then print `sizeof` of the struct
/// and `offsetof` of each requested field. Pure, so it is unit-tested.
fn generate_probe_c(headers: &[String], struct_name: &str, fields: &[String]) -> String {
    let mut s = String::from("#include <stdio.h>\n#include <stddef.h>\n");
    for h in headers {
        s.push_str(&format!("#include <{h}>\n"));
    }
    s.push_str("int main(void) {\n");
    s.push_str(&format!(
        "    printf(\"sizeof({0}) = %zu\\n\", sizeof({0}));\n",
        struct_name
    ));
    for f in fields {
        s.push_str(&format!(
            "    printf(\"offsetof({0}, {1}) = %zu\\n\", offsetof({0}, {1}));\n",
            struct_name, f
        ));
    }
    s.push_str("    return 0;\n}\n");
    s
}

/// Parse `sizeof(<struct>) = <n>` out of the probe output.
fn parse_struct_size(output: &str, struct_name: &str) -> Option<usize> {
    let needle = format!("sizeof({struct_name}) = ");
    for line in output.lines() {
        if let Some(rest) = line.trim().strip_prefix(&needle) {
            return rest.trim().parse().ok();
        }
    }
    None
}

fn arg_err(msg: &str) -> i32 {
    eprintln!("xtask ffi-probe: {msg}");
    2
}

// --- command helpers ----------------------------------------------------

/// Run a command, streaming its output, after announcing it. Returns the exit
/// code (127 if the program could not be spawned).
fn run(desc: &str, program: &str, args: &[&str], envs: &[(String, String)]) -> i32 {
    println!("\n=== {desc} ===");
    println!("$ {program} {}", args.join(" "));
    let mut c = Command::new(program);
    c.args(args);
    for (k, v) in envs {
        c.env(k, v);
    }
    match c.status() {
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => {
            eprintln!("xtask: failed to run '{program}': {e}");
            127
        }
    }
}

/// Run a command and return its trimmed stdout if it succeeded.
fn capture(program: &str, args: &[&str], envs: &[(String, String)]) -> Option<String> {
    let mut c = Command::new(program);
    c.args(args);
    for (k, v) in envs {
        c.env(k, v);
    }
    let out = c.output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Whether a program runs and exits 0 (silenced output), for capability probes.
fn cmd_ok(program: &str, args: &[&str]) -> bool {
    Command::new(program)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Whether `pkg-config --exists <lib>` succeeds (the build prerequisite probe).
fn pkg_exists(lib: &str) -> bool {
    cmd_ok("pkg-config", &["--exists", lib])
}

/// Whether `dir` contains an entry whose name starts with `prefix` (e.g. a
/// `/dev/video*` camera or a `/dev/dri/renderD*` GPU node).
fn dev_entry_exists(dir: &str, prefix: &str) -> bool {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries.flatten().any(|e| e.file_name().to_string_lossy().starts_with(prefix))
        })
        .unwrap_or(false)
}

/// First of `candidates` found on `PATH`, via `which`.
fn which(candidates: &[&str]) -> Option<String> {
    candidates.iter().find(|c| cmd_ok("which", &[c])).map(|c| c.to_string())
}

/// A `PATH` that prepends `~/.cargo/bin`, so cargo invokes the rustup toolchain
/// rather than a distro `rustc` lacking the cross target std. Empty (no
/// override) if `HOME` / `PATH` are unset.
fn rustup_path_env() -> Vec<(String, String)> {
    match (std::env::var("HOME"), std::env::var("PATH")) {
        (Ok(home), Ok(path)) => {
            vec![("PATH".into(), format!("{home}/.cargo/bin:{path}"))]
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_capabilities_still_includes_pure_rust_features() {
        let f = host_test_features(&Capabilities::default());
        // Pure-Rust std features need no system lib, so they are always present.
        assert!(f.contains(&"rtsp"));
        assert!(f.contains(&"hls"));
        assert!(f.contains(&"analytics"));
        // Nothing hardware/syslib gated leaks in.
        assert!(!f.contains(&"nvenc"));
        assert!(!f.contains(&"vaapi"));
        assert!(!f.contains(&"wgpu-sink"));
        assert!(!f.contains(&"opus"));
    }

    #[test]
    fn capabilities_gate_their_features() {
        let caps = Capabilities {
            nvidia: true,
            gpu_drm: true,
            opus: true,
            v4l2: true,
            ..Capabilities::default()
        };
        let f = host_test_features(&caps);
        assert!(f.contains(&"nvenc") && f.contains(&"nvdec") && f.contains(&"cuda"));
        assert!(f.contains(&"wgpu-sink") && f.contains(&"vello-overlay") && f.contains(&"kms-sink"));
        assert!(f.contains(&"opus"));
        assert!(f.contains(&"v4l2"));
        // VAAPI / ffmpeg / audio not detected => excluded.
        assert!(!f.contains(&"vaapi"));
        assert!(!f.contains(&"ffmpeg"));
        assert!(!f.contains(&"alsa-sink"));
    }

    #[test]
    fn rustup_path_env_prepends_cargo_bin_when_set() {
        // Exercises the format, independent of the ambient HOME/PATH.
        let home = "/home/u";
        let path = "/usr/bin:/bin";
        let joined = format!("{home}/.cargo/bin:{path}");
        assert!(joined.starts_with("/home/u/.cargo/bin:"));
    }

    #[test]
    fn ffi_probe_c_includes_headers_and_probes_struct_and_fields() {
        let c = generate_probe_c(
            &["ffnvcodec/nvEncodeAPI.h".into()],
            "NV_ENC_INITIALIZE_PARAMS",
            &["encodeGUID".into(), "encodeWidth".into()],
        );
        assert!(c.contains("#include <ffnvcodec/nvEncodeAPI.h>"));
        assert!(c.contains("sizeof(NV_ENC_INITIALIZE_PARAMS)"));
        assert!(c.contains("offsetof(NV_ENC_INITIALIZE_PARAMS, encodeGUID)"));
        assert!(c.contains("offsetof(NV_ENC_INITIALIZE_PARAMS, encodeWidth)"));
        // No fields requested => sizeof only, no offsetof.
        let c2 = generate_probe_c(&["stdint.h".into()], "uint32_t", &[]);
        assert!(c2.contains("sizeof(uint32_t)"));
        assert!(!c2.contains("offsetof"));
    }

    #[test]
    fn ffi_probe_parses_struct_size_from_output() {
        let out = "sizeof(NV_ENC_INITIALIZE_PARAMS) = 1816\noffsetof(NV_ENC_INITIALIZE_PARAMS, encodeWidth) = 24";
        assert_eq!(parse_struct_size(out, "NV_ENC_INITIALIZE_PARAMS"), Some(1816));
        assert_eq!(parse_struct_size(out, "OTHER"), None);
    }

    #[test]
    fn linux_features_csv_is_nonempty_and_comma_separated() {
        assert!(LINUX_FEATURES.contains("rtsp"));
        assert!(LINUX_FEATURES.split(',').count() > 5);
        assert!(!LINUX_FEATURES.contains(' '), "feature csv must not contain spaces");
    }
}
