//! Shared `host`/`address` + `port` runtime-property dispatch for the network
//! source/sink elements. Each of these elements holds a `SocketAddr` (a bind
//! address on a source, a destination on a sink) and exposes its IP as a string
//! property (`host` on sinks, `address` on sources) plus `port` as a uint. The
//! set/get logic (parse a string as `IpAddr`, bounds-check a uint against
//! `u16::MAX`) is identical everywhere, so it lives here once rather than being
//! copied into every element's `set_property` / `get_property`.

use std::net::{IpAddr, SocketAddr};

use alloc::string::ToString;

use g2g_core::property::{PropError, PropValue};

/// Handle the address/port half of an element's `set_property`. `addr_key` is the
/// element's string property name for the IP (`"host"` on sinks, `"address"` on
/// sources). Returns `Some(result)` when `name` is `addr_key` or `"port"` (so the
/// caller returns it), or `None` when the name is some other property (so the
/// caller's own `match` handles it).
pub(crate) fn set_addr_prop(
    addr: &mut SocketAddr,
    addr_key: &str,
    name: &str,
    value: &PropValue,
) -> Option<Result<(), PropError>> {
    if name == addr_key {
        return Some(set_ip(addr, value));
    }
    if name == "port" {
        return Some(set_port(addr, value));
    }
    None
}

/// Handle the address/port half of an element's `get_property`. Returns the IP as
/// a string for `addr_key`, the port as a uint for `"port"`, and `None` otherwise.
pub(crate) fn get_addr_prop(addr: &SocketAddr, addr_key: &str, name: &str) -> Option<PropValue> {
    if name == addr_key {
        return Some(PropValue::Str(addr.ip().to_string()));
    }
    if name == "port" {
        return Some(PropValue::Uint(addr.port() as u64));
    }
    None
}

fn set_ip(addr: &mut SocketAddr, value: &PropValue) -> Result<(), PropError> {
    let ip = value
        .as_str()
        .ok_or(PropError::Type)?
        .parse::<IpAddr>()
        .map_err(|_| PropError::Value)?;
    addr.set_ip(ip);
    Ok(())
}

fn set_port(addr: &mut SocketAddr, value: &PropValue) -> Result<(), PropError> {
    let p = value.as_uint().ok_or(PropError::Type)?;
    if p > u16::MAX as u64 {
        return Err(PropError::Value);
    }
    addr.set_port(p as u16);
    Ok(())
}
