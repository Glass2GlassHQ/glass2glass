//! Whether a Wayland compositor is reachable, the check the `auto*sink` aliases
//! use to skip a display sink that is compiled in but has nothing to draw on.
//!
//! Every Linux display sink here reaches the compositor the same way, through
//! `Connection::connect_to_env`, so one connect answers for all of them.

use smithay_client_toolkit::reexports::client::Connection;

/// Whether a Wayland compositor accepts a connection right now.
///
/// A sink can be built into the binary and still have nothing to present on: a
/// headless build machine, an ssh session, a service with no seat.
pub(crate) fn compositor_reachable() -> bool {
    Connection::connect_to_env().is_ok()
}
