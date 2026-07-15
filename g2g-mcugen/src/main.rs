//! `g2g-mcugen <graph.yaml> [-o out.rs]`: compile a declarative MCU graph
//! (audio or video / display) into a monomorphized static MCU pipeline and
//! report its ring-memory budget. With no `-o`, the generated Rust goes to
//! stdout and the budget to stderr.

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "-o" | "--output" => match args.next() {
                Some(p) => output = Some(p),
                None => return usage("-o needs a path"),
            },
            "-h" | "--help" => {
                eprintln!("usage: g2g-mcugen <graph.yaml> [-o out.rs]");
                return ExitCode::SUCCESS;
            }
            other if input.is_none() => input = Some(other.to_string()),
            other => return usage(&format!("unexpected argument `{other}`")),
        }
    }
    let Some(input) = input else {
        return usage("no input graph given");
    };

    let text = match std::fs::read_to_string(&input) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("g2g-mcugen: cannot read {input}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let compiled = match g2g_mcugen::compile_str(&text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("g2g-mcugen: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Budget report to stderr (so `-o -` style piping stays clean on stdout).
    eprintln!("== ring-memory budget ==");
    for (name, bytes) in &compiled.rings {
        eprintln!("  {name:<24} {bytes:>6} bytes");
    }
    eprintln!("  {:<24} {:>6} bytes total", "RING_BYTES_TOTAL", compiled.ring_bytes_total);
    let entry_params = compiled.grabber_params.iter().chain(&compiled.sink_params).cloned().collect::<Vec<_>>();
    eprintln!("  entry: {}({})", compiled.entry, entry_params.join(", "));

    match &output {
        Some(path) => {
            if let Err(e) = std::fs::write(path, &compiled.source) {
                eprintln!("g2g-mcugen: cannot write {path}: {e}");
                return ExitCode::FAILURE;
            }
            eprintln!("wrote {path}");
        }
        None => print!("{}", compiled.source),
    }
    ExitCode::SUCCESS
}

fn usage(msg: &str) -> ExitCode {
    eprintln!("g2g-mcugen: {msg}");
    eprintln!("usage: g2g-mcugen <graph.yaml> [-o out.rs]");
    ExitCode::FAILURE
}
