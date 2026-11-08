use alloc::string::String;
use alloc::vec::Vec;

use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use socket2::{Domain, InterfaceIndexOrAddress, Protocol, Socket, Type};

#[derive(Debug, Clone, Copy)]
pub(crate) struct UdpBind<'a> {
    pub(crate) local: SocketAddr,
    pub(crate) ifaces: &'a [String],
    pub(crate) multicast: Option<MulticastJoin<'a>>,
    pub(crate) reuse: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MulticastJoin<'a> {
    pub(crate) group: IpAddr,
    pub(crate) sources: &'a [IpAddr],
}

pub(crate) fn bind_udp_socket(bind: UdpBind<'_>) -> io::Result<Socket> {
    let local = bind.local;
    let socket = Socket::new(Domain::for_address(local), Type::DGRAM, Some(Protocol::UDP))?;
    // gst udpsrc2 skips reuse on an ephemeral unicast port, which another socket could then share
    let ephemeral_unicast = local.port() == 0 && bind.multicast.is_none();
    if bind.reuse && !ephemeral_unicast {
        socket.set_reuse_address(true)?;
        #[cfg(not(windows))]
        socket.set_reuse_port(true)?;
    }
    let interface_indices = bind
        .ifaces
        .iter()
        .map(|iface| interface_index(iface))
        .collect::<io::Result<Vec<u32>>>()?;
    // gst binds to the device only when one interface is named, SO_BINDTODEVICE takes just one
    #[cfg(any(target_os = "linux", target_os = "android"))]
    if let [iface] = bind.ifaces {
        socket.bind_device(Some(iface.as_bytes()))?;
    }
    socket.bind(&local.into())?;
    let Some(join) = bind.multicast else {
        return Ok(socket);
    };
    // interface 0 lets the kernel pick
    let join_on = if interface_indices.is_empty() {
        alloc::vec![0]
    } else {
        interface_indices
    };
    for &index in &join_on {
        join_group(&socket, join, index)?;
    }
    if join.group.is_ipv6() {
        socket.set_multicast_if_v6(join_on[0])?;
    }
    Ok(socket)
}

fn join_group(socket: &Socket, join: MulticastJoin<'_>, interface_index: u32) -> io::Result<()> {
    // linux delivers every group joined on the port by any socket unless IP_MULTICAST_ALL is off
    #[cfg(target_os = "linux")]
    match join.group {
        IpAddr::V4(_) => socket.set_multicast_all_v4(false)?,
        IpAddr::V6(_) => socket.set_multicast_all_v6(false)?,
    }
    let sources: Vec<IpAddr> = join
        .sources
        .iter()
        .copied()
        .filter(|source| source.is_ipv4() == join.group.is_ipv4())
        .collect();
    if !sources.is_empty() && source_specific_join_supported(join.group) {
        for source in sources {
            join_source_group(socket, join.group, source, interface_index)?;
        }
        return Ok(());
    }
    match join.group {
        IpAddr::V4(group) if interface_index == 0 => {
            socket.join_multicast_v4(&group, &Ipv4Addr::UNSPECIFIED)
        }
        IpAddr::V4(group) => {
            socket.join_multicast_v4_n(&group, &InterfaceIndexOrAddress::Index(interface_index))
        }
        IpAddr::V6(group) => socket.join_multicast_v6(&group, interface_index),
    }
}

fn source_specific_join_supported(group: IpAddr) -> bool {
    cfg!(any(target_os = "linux", target_os = "android")) || group.is_ipv4()
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn join_source_group(
    socket: &Socket,
    group: IpAddr,
    source: IpAddr,
    interface_index: u32,
) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    let level = match group {
        IpAddr::V4(_) => libc::IPPROTO_IP,
        IpAddr::V6(_) => libc::IPPROTO_IPV6,
    };
    // SAFETY: group_source_req is plain integers and byte arrays, so all zero bytes is a valid value.
    let mut request: libc::group_source_req = unsafe { core::mem::zeroed() };
    request.gsr_interface = interface_index;
    copy_into_storage(&mut request.gsr_group, group);
    copy_into_storage(&mut request.gsr_source, source);
    // SAFETY: `request` is a live, fully initialised group_source_req and the length passed is its size.
    let result = unsafe {
        libc::setsockopt(
            socket.as_raw_fd(),
            level,
            libc::MCAST_JOIN_SOURCE_GROUP,
            (&raw const request).cast(),
            core::mem::size_of::<libc::group_source_req>() as libc::socklen_t,
        )
    };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn copy_into_storage(storage: &mut libc::sockaddr_storage, ip: IpAddr) {
    let address = socket2::SockAddr::from(SocketAddr::new(ip, 0));
    // SAFETY: socket2 stores the address in a sockaddr_storage, so `len` bytes are readable at
    // `as_ptr` and fit in the destination storage, and the two never overlap.
    unsafe {
        core::ptr::copy_nonoverlapping(
            address.as_ptr().cast::<u8>(),
            (storage as *mut libc::sockaddr_storage).cast::<u8>(),
            address.len() as usize,
        );
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn join_source_group(
    socket: &Socket,
    group: IpAddr,
    source: IpAddr,
    interface_index: u32,
) -> io::Result<()> {
    // socket2 names the interface of a source-specific join by address, not index
    match (group, source, interface_index) {
        (IpAddr::V4(group), IpAddr::V4(source), 0) => {
            socket.join_ssm_v4(&source, &group, &Ipv4Addr::UNSPECIFIED)
        }
        _ => Err(io::Error::from(io::ErrorKind::Unsupported)),
    }
}

#[cfg(unix)]
fn interface_index(name: &str) -> io::Result<u32> {
    let name =
        std::ffi::CString::new(name).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    // SAFETY: `name` is a NUL-terminated string that outlives the call, which only reads it.
    let index = unsafe { libc::if_nametoindex(name.as_ptr()) };
    if index == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(index)
}

#[cfg(not(unix))]
fn interface_index(_name: &str) -> io::Result<u32> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "multicast-iface needs if_nametoindex",
    ))
}

#[cfg(any(feature = "rist", feature = "udp-ingress"))]
pub(crate) fn unspecified_address(like: SocketAddr, port: u16) -> SocketAddr {
    if like.is_ipv6() {
        SocketAddr::from((std::net::Ipv6Addr::UNSPECIFIED, port))
    } else {
        SocketAddr::from((Ipv4Addr::UNSPECIFIED, port))
    }
}
