//! Whether a Wayland compositor is reachable, the check the `auto*sink` aliases
//! use to skip a display sink that is compiled in but has nothing to draw on.
//!
//! Every Linux display sink here reaches the compositor the same way, through
//! `Connection::connect_to_env`, so one connect answers for all of them.

use alloc::format;
use alloc::string::String;

use smithay_client_toolkit::reexports::client::Connection;

/// Whether a Wayland compositor accepts a connection right now, and why not
/// when it does not.
///
/// A sink can be built into the binary and still have nothing to present on: a
/// headless build machine, an ssh session, a service with no seat.
pub(crate) fn compositor_reachable() -> Result<(), String> {
    let Err(err) = Connection::connect_to_env() else {
        return Ok(());
    };
    Err(match std::env::var("WAYLAND_DISPLAY") {
        Ok(display) => format!("no Wayland compositor on {display}: {err}"),
        Err(_) => String::from("WAYLAND_DISPLAY is unset"),
    })
}
