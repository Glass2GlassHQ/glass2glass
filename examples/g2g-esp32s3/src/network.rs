// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::net::SocketAddrV4;

use embassy_net::udp::{PacketMetadata, UdpSocket};
use embassy_net::{IpEndpoint, Runner, Stack, StackResources};
use esp_hal::peripherals::{FROM_CPU_INTR0, TIMG0, WIFI};
use esp_hal::rng::Rng;
use esp_hal::timer::timg::TimerGroup;
use esp_println::println;
use esp_radio::wifi::sta::StationConfig;
use esp_radio::wifi::{AuthenticationMethodConfig, Config, Interface, WifiController};
use g2g_core::error::{G2gError, NetworkFailure};
use g2g_core::rtp::RTP_HEADER_LEN;
use g2g_mcu::PacketSender;
use static_cell::StaticCell;

const SSID: &str = env!("G2G_WIFI_SSID");
const PASSWORD: &str = env!("G2G_WIFI_PASSWORD");
const RTP_DESTINATION: &str = env!("G2G_RTP_DESTINATION");

// fits a 1500-byte MTU after the IP and UDP headers
pub const RTP_PACKET_BYTES: usize = 1400;
const SEND_QUEUE_PACKETS: usize = 8;
const DHCP_AND_RTP_SOCKETS: usize = 2;

pub fn start(
    wifi: WIFI<'static>,
    timer_group: TIMG0<'static>,
    interrupt: FROM_CPU_INTR0<'static>,
) -> Result<WifiController<'static>, G2gError> {
    // esp-radio peaks at 47 KB of heap on this board, measured with esp-alloc's heap stats
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 72 * 1024);
    esp_rtos::start(TimerGroup::new(timer_group).timer0, interrupt);
    let station = StationConfig::default()
        .with_ssid(SSID.try_into().map_err(|_| connect_failure())?)
        .with_authentication(AuthenticationMethodConfig::Wpa2Personal(
            PASSWORD.try_into().map_err(|_| connect_failure())?,
        ));
    let mut controller =
        WifiController::new(wifi, Default::default()).map_err(|_| connect_failure())?;
    controller
        .set_config(&Config::Station(station))
        .map_err(|_| connect_failure())?;
    Ok(controller)
}

pub fn ip_stack() -> (Stack<'static>, Runner<'static, Interface>) {
    static RESOURCES: StaticCell<StackResources<DHCP_AND_RTP_SOCKETS>> = StaticCell::new();
    let rng = Rng::new();
    let seed = u64::from(rng.random()) << 32 | u64::from(rng.random());
    embassy_net::new(
        Interface::station(),
        embassy_net::Config::dhcpv4(Default::default()),
        RESOURCES.init(StackResources::new()),
        seed,
    )
}

fn connect_failure() -> G2gError {
    G2gError::Network(NetworkFailure::Connect)
}

pub async fn connect(
    controller: &mut WifiController<'static>,
    stack: Stack<'static>,
) -> Result<(), G2gError> {
    println!("joining {SSID}");
    controller
        .connect_async()
        .await
        .map_err(|_| connect_failure())?;
    stack.wait_config_up().await;
    let address = stack.config_v4().ok_or_else(connect_failure)?.address;
    println!("address {address}, sending RTP to {RTP_DESTINATION}");
    Ok(())
}

pub struct UdpSender {
    socket: UdpSocket<'static>,
    destination: IpEndpoint,
    datagram: [u8; RTP_PACKET_BYTES],
}

pub fn udp_sender(stack: Stack<'static>) -> Result<UdpSender, G2gError> {
    const SEND_BYTES: usize = SEND_QUEUE_PACKETS * RTP_PACKET_BYTES;
    static SEND_METADATA: StaticCell<[PacketMetadata; SEND_QUEUE_PACKETS]> = StaticCell::new();
    static SEND_BUFFER: StaticCell<[u8; SEND_BYTES]> = StaticCell::new();
    static RECEIVE_METADATA: StaticCell<[PacketMetadata; 1]> = StaticCell::new();
    static RECEIVE_BUFFER: StaticCell<[u8; 1]> = StaticCell::new();
    let destination: SocketAddrV4 = RTP_DESTINATION.parse().map_err(|_| connect_failure())?;
    let mut socket = UdpSocket::new(
        stack,
        RECEIVE_METADATA.init([PacketMetadata::EMPTY; 1]),
        RECEIVE_BUFFER.init([0; 1]),
        SEND_METADATA.init([PacketMetadata::EMPTY; SEND_QUEUE_PACKETS]),
        SEND_BUFFER.init([0; SEND_BYTES]),
    );
    socket.bind(0).map_err(|_| connect_failure())?;
    Ok(UdpSender {
        socket,
        destination: destination.into(),
        datagram: [0; RTP_PACKET_BYTES],
    })
}

impl PacketSender for UdpSender {
    async fn send(
        &mut self,
        header: &[u8; RTP_HEADER_LEN],
        payload: &[u8],
    ) -> Result<(), G2gError> {
        let length = RTP_HEADER_LEN + payload.len();
        let datagram = self
            .datagram
            .get_mut(..length)
            .ok_or(G2gError::CapsMismatch)?;
        datagram[..RTP_HEADER_LEN].copy_from_slice(header);
        datagram[RTP_HEADER_LEN..].copy_from_slice(payload);
        // embassy-net's send_to_with panics when smoltcp 0.13.1 runs its closure and still reports a full queue
        self.socket
            .send_to(datagram, self.destination)
            .await
            .map_err(|_| G2gError::Network(NetworkFailure::Stream))
    }
}
