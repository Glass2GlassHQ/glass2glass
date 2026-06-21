//! `g2g-inspect`: the `gst-inspect` analog. Introspects the standard element
//! registry (the same one the text-launch parser uses).
//!
//! Usage:
//!   g2g-inspect              # list every registerable element
//!   g2g-inspect <element>    # dump one element's role, properties, pad templates
//!
//! Backed by [`g2g_plugins::registry::default_registry`] and
//! [`g2g_core::runtime::Registry::inspect`] (M105/M107). Requires the `std`
//! feature (the registry is std-only).

use std::process;

use g2g_plugins::registry::default_registry;

fn main() {
    let reg = default_registry();
    let mut args = std::env::args().skip(1);
    match args.next() {
        // No element named: list them all, one per line, the `gst-inspect` index.
        None => {
            for name in reg.element_names() {
                println!("{name}");
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
