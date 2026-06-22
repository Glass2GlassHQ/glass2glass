//! `g2g-inspect`: the `gst-inspect` analog. Introspects the standard element
//! registry (the same one the text-launch parser uses).
//!
//! Usage:
//!   g2g-inspect              # list every registerable element
//!   g2g-inspect <element>    # dump one element's role, properties, pad templates
//!   g2g-inspect --all        # dump every element in full (the index, in detail)
//!   g2g-inspect --gst <name> # what a GStreamer element name maps to in g2g
//!
//! Backed by [`g2g_plugins::registry::default_registry`] and
//! [`g2g_core::runtime::Registry::inspect`] (M105/M107). Requires the `std`
//! feature (the registry is std-only).

use std::process;

use g2g_plugins::gst_compat::{gst_equivalent, GstEquivalent};
use g2g_plugins::registry::default_registry;

fn main() {
    let reg = default_registry();
    let mut args = std::env::args().skip(1);
    match args.next() {
        // No element named: list them all, `name: Long-name` per line, the
        // `gst-inspect` index.
        None => {
            for line in reg.element_listing() {
                println!("{line}");
            }
        }
        // `--all` / `-a`: the full dump for every registered element, separated
        // by a rule, so the whole catalog can be read or grepped at once.
        Some(flag) if flag == "--all" || flag == "-a" => {
            let names = reg.element_names();
            let total = names.len();
            for (i, name) in names.into_iter().enumerate() {
                if let Some(dump) = reg.inspect(name) {
                    print!("{dump}");
                    if i + 1 < total {
                        println!("\n{}\n", "-".repeat(60));
                    }
                }
            }
        }
        // `--gst <name>`: map a GStreamer element name to its g2g equivalent,
        // for porting a pipeline element by element.
        Some(flag) if flag == "--gst" => {
            let Some(gst_name) = args.next() else {
                eprintln!("usage: g2g-inspect --gst <gstreamer-element-name>");
                process::exit(2);
            };
            match gst_equivalent(&reg, &gst_name) {
                GstEquivalent::Available => {
                    println!("{gst_name}: available in g2g under the same name");
                }
                GstEquivalent::Renamed(g) => {
                    println!("{gst_name}: g2g calls it `{g}` (run `g2g-inspect {g}` for details)");
                }
                GstEquivalent::Unsupported(hint) => {
                    println!("{gst_name}: no g2g element. {hint}");
                }
                GstEquivalent::Unknown => {
                    println!("{gst_name}: unknown to g2g; no known equivalent. List elements with `g2g-inspect`.");
                    process::exit(1);
                }
            }
        }
        Some(name) => match reg.inspect(&name) {
            Some(dump) => print!("{dump}"),
            None => {
                eprintln!("No such element: {name}");
                process::exit(1);
            }
        },
    }
}
