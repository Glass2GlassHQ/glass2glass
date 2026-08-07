//! `g2g-device-monitor`: the `gst-device-monitor-1.0` analog. Lists the devices
//! every compiled-in provider can see, and optionally follows hotplug.
//!
//! Usage:
//!   g2g-device-monitor                    # probe once, print every device
//!   g2g-device-monitor Video/Source       # only devices in that class
//!   g2g-device-monitor --follow [class]   # keep printing hotplug events
//!   g2g-device-monitor --json [class]     # machine-readable one-shot dump
//!
//! Backed by [`g2g_plugins::devicemon::default_device_monitor`] (M939). Which
//! providers exist depends on the build features, so run it with a broad set:
//! `--features v4l2,alsa-src,pipewire,wgpu-sink` on Linux,
//! `--features mf-video-src,wasapi-src,wasapi-sink,wgpu-sink` on Windows,
//! `--features avfoundation,coreaudio,wgpu-sink` on macOS. A provider that
//! fails to probe (no PipeWire daemon, no GPU) is reported on stderr and does
//! not fail the run.

use std::process;

use g2g_core::runtime::{Device, DeviceEvent};
use g2g_plugins::devicemon::default_device_monitor;

const USAGE: &str = "usage: g2g-device-monitor [--follow] [--json] [<classes>]\n  \
                     <classes>  filter, e.g. Video/Source, Audio/Sink, Compute/GPU\n  \
                     --follow   keep watching for hotplug (Ctrl-C to stop)\n  \
                     --json     one-shot machine-readable dump";

/// Parsed command line: at most one class filter plus the mode flags.
struct Args {
    classes: Option<String>,
    follow: bool,
    json: bool,
}

fn parse_args(raw: impl Iterator<Item = String>) -> Args {
    let mut args = Args {
        classes: None,
        follow: false,
        json: false,
    };
    for arg in raw {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                process::exit(0);
            }
            "--follow" | "-f" => args.follow = true,
            "--json" => args.json = true,
            other if other.starts_with('-') => {
                eprintln!("g2g-device-monitor: unknown option `{other}`\n{USAGE}");
                process::exit(2);
            }
            other if args.classes.is_none() => args.classes = Some(other.to_string()),
            other => {
                eprintln!("g2g-device-monitor: unexpected argument `{other}`\n{USAGE}");
                process::exit(2);
            }
        }
    }
    if args.json && args.follow {
        eprintln!("g2g-device-monitor: --json and --follow are mutually exclusive\n{USAGE}");
        process::exit(2);
    }
    args
}

/// One device as a `gst-device-monitor`-style block.
fn print_device(device: &Device) {
    println!("Device found:\n");
    println!("    name  : {}", device.display_name);
    println!("    class : {}", device.klass);
    println!("    id    : {}", device.persistent_id);
    let alternatives = device.caps.alternatives();
    match alternatives.split_first() {
        None => println!("    caps  : (none)"),
        Some((first, rest)) => {
            println!("    caps  : {first:?}");
            for caps in rest {
                println!("            {caps:?}");
            }
        }
    }
    if device.detail.is_empty() {
        println!("    properties: (none)");
    } else {
        println!("    properties:");
        for (key, value) in &device.detail {
            println!("            {key} = {value}");
        }
    }
    if !device.element.is_empty() {
        println!("    gst-launch-1.0-style: {}", device.launch_fragment());
    }
    println!();
}

/// `{"devices": [...], "errors": [...]}` for the one-shot dump.
#[cfg(feature = "tooling-json")]
fn dump_json(classes: Option<&str>) {
    use serde_json::{json, Map, Value};

    let mut monitor = default_device_monitor();
    if let Some(classes) = classes {
        monitor.add_filter(classes, None);
    }
    let outcome = monitor.probe();

    let pairs = |list: &[(String, String)]| {
        let mut map = Map::new();
        for (k, v) in list {
            map.insert(k.clone(), Value::String(v.clone()));
        }
        Value::Object(map)
    };

    let devices: Vec<Value> = outcome
        .devices
        .iter()
        .map(|d| {
            json!({
                "provider": d.provider,
                "klass": d.klass,
                "display-name": d.display_name,
                "persistent-id": d.persistent_id,
                "element": d.element,
                "launch": if d.element.is_empty() { String::new() } else { d.launch_fragment() },
                "props": pairs(&d.props),
                "detail": pairs(&d.detail),
                "caps": d.caps.alternatives().iter().map(|c| format!("{c:?}")).collect::<Vec<_>>(),
            })
        })
        .collect();
    let errors: Vec<Value> = outcome
        .errors
        .iter()
        .map(|(provider, err)| json!({ "provider": provider, "error": format!("{err:?}") }))
        .collect();

    let out = json!({ "devices": devices, "errors": errors });
    println!(
        "{}",
        serde_json::to_string_pretty(&out).expect("serialize devices")
    );
}

#[cfg(not(feature = "tooling-json"))]
fn dump_json(_classes: Option<&str>) {
    eprintln!(
        "g2g-device-monitor: --json needs the `tooling-json` build feature \
         (rebuild with --features tooling-json)"
    );
    process::exit(1);
}

/// Probe once and print; the shared prelude of both the one-shot and the
/// `--follow` run.
fn print_probe(classes: Option<&str>) {
    let mut monitor = default_device_monitor();
    if let Some(classes) = classes {
        monitor.add_filter(classes, None);
    }
    let outcome = monitor.probe();
    for device in &outcome.devices {
        print_device(device);
    }
    for (provider, err) in &outcome.errors {
        eprintln!("{provider}: {err:?}");
    }
}

/// Start the monitor and print events until every watch has ended (Ctrl-C in
/// practice: the process dies with its terminal).
fn follow(classes: Option<&str>) {
    let mut monitor = default_device_monitor();
    if let Some(classes) = classes {
        monitor.add_filter(classes, None);
    }
    let running = monitor.start();
    for (provider, err) in &running.watch_errors {
        eprintln!("{provider}: watch failed, polling instead: {err:?}");
    }
    while let Some(event) = g2g_core::runtime::block_on(running.recv()) {
        match event {
            DeviceEvent::Added(device) => {
                println!("ADDED:");
                print_device(&device);
            }
            DeviceEvent::Changed(device) => {
                println!("CHANGED:");
                print_device(&device);
            }
            DeviceEvent::Removed {
                provider,
                persistent_id,
            } => {
                println!("REMOVED: {provider}/{persistent_id}");
            }
        }
    }
}

fn main() {
    let args = parse_args(std::env::args().skip(1));
    if args.json {
        dump_json(args.classes.as_deref());
    } else if args.follow {
        follow(args.classes.as_deref());
    } else {
        print_probe(args.classes.as_deref());
    }
}
