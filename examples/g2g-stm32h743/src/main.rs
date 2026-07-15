//! STM32H743 (Cortex-M7) on-device harness: run the heap-free flagship audio
//! graph (`capture -> convert -> resample -> mix -> encode -> RTP`, the same
//! `noalloc_pipeline::audio` pipeline every no_std proof covers) and egress its
//! RTP over the H743's on-chip Ethernet through a **pure-Rust** smoltcp /
//! embassy-net stack, on real silicon. No C in the network path, unlike the
//! ESP32-P4's C6/esp-idf WiFi.
//!
//! STATUS: this **compiles** for `thumbv7em-none-eabihf` (verified with the
//! pinned embassy versions in Cargo.toml); it is kept out of CI only for
//! embassy's build weight. What still needs the board is *runtime* config, not
//! compilation: the RCC/clock `Config` (`Default` compiles but will not clock
//! the Ethernet MAC), the RMII pin map, and the RTP destination, all marked
//! `VERIFY` below. The **g2g-specific** part, [`EmbassyNetSender`] and the
//! `run_audio_with` wiring, is what makes the point: our `PacketSender` seam
//! maps one-to-one onto an embassy-net `UdpSocket`, so `RtpSink` egress over a
//! pure-Rust stack is just this ~10-line bridge.

#![no_std]
#![no_main]

use embassy_executor::Spawner;
use embassy_net::udp::{PacketMetadata, UdpSocket};
use embassy_net::{IpEndpoint, Ipv4Address, StackResources};
use embassy_stm32::eth::generic_smi::GenericSMI;
use embassy_stm32::eth::{Ethernet, PacketQueue};
use embassy_stm32::{bind_interrupts, eth, peripherals};
use g2g_core::error::{G2gError, HardwareError};
use g2g_core::rtp::RTP_HEADER_LEN;
use g2g_mcu::rtp::PacketSender;
use noalloc_pipeline::audio::run_audio_with;
use static_cell::StaticCell;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

bind_interrupts!(struct Irqs {
    ETH => eth::InterruptHandler;
});

/// The concrete Ethernet device type (RMII generic-SMI PHY). Kept as an alias
/// so the net-runner task signature stays readable. VERIFY against the embassy
/// version's `Ethernet` type parameters.
type Device = Ethernet<'static, peripherals::ETH, GenericSMI>;

/// A [`PacketSender`] over an embassy-net UDP socket: the entire bridge from
/// g2g's RTP egress seam to a pure-Rust network stack. `RtpSink` hands us the
/// 12-byte RTP header and the payload as separate slices (scatter-gather);
/// embassy-net wants one datagram, so we concatenate into a stack buffer (no
/// heap) and `send_to` the destination.
struct EmbassyNetSender<'s> {
    socket: UdpSocket<'s>,
    dest: IpEndpoint,
    scratch: [u8; 1500],
}

impl PacketSender for EmbassyNetSender<'_> {
    async fn send(&mut self, header: &[u8; RTP_HEADER_LEN], payload: &[u8]) -> Result<(), G2gError> {
        let n = RTP_HEADER_LEN + payload.len();
        let Some(buf) = self.scratch.get_mut(..n) else {
            // A payload larger than one MTU datagram: reject, do not fragment
            // (the same discipline RtpSink applies to the C-seam sender).
            return Err(G2gError::CapsMismatch);
        };
        let (h, p) = buf.split_at_mut(RTP_HEADER_LEN);
        h.copy_from_slice(header);
        p.copy_from_slice(payload);
        self.socket
            .send_to(buf, self.dest)
            .await
            .map_err(|_| G2gError::Hardware(HardwareError::Peripheral))
    }
}

/// The embassy-net stack runner (drives smoltcp: polls the device, handles
/// DHCP/ARP). VERIFY the `Runner` type against the pinned embassy-net version.
#[embassy_executor::task]
async fn net_task(mut runner: embassy_net::Runner<'static, Device>) -> ! {
    runner.run().await
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    // --- Clock config -------------------------------------------------------
    // VERIFY: the H743 needs a specific PLL/RCC setup for the 400+ MHz core and
    // the Ethernet clock. Copy the exact `Config` from embassy's stm32h7
    // ethernet example for your board; `Default` will not clock Ethernet.
    let p = embassy_stm32::init(embassy_stm32::Config::default());

    // --- Ethernet (RMII, generic SMI PHY) -----------------------------------
    // VERIFY: the RMII pin map is the Nucleo-H743ZI2 wiring; confirm against the
    // board schematic. Order per the embassy `Ethernet::new` signature.
    static PACKETS: StaticCell<PacketQueue<4, 4>> = StaticCell::new();
    let mac = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];
    let device = Ethernet::new(
        PACKETS.init(PacketQueue::new()),
        p.ETH,
        Irqs,
        p.PA1,  // REF_CLK
        p.PA2,  // MDIO
        p.PC1,  // MDC
        p.PA7,  // CRS_DV
        p.PC4,  // RXD0
        p.PC5,  // RXD1
        p.PG13, // TXD0
        p.PB13, // TXD1
        p.PG11, // TX_EN
        GenericSMI::new(0),
        mac,
    );

    // --- Pure-Rust IP stack (smoltcp via embassy-net), DHCP -----------------
    let config = embassy_net::Config::dhcpv4(Default::default());
    static RESOURCES: StaticCell<StackResources<4>> = StaticCell::new();
    // A fixed seed is fine for a demo; use the RNG peripheral for production.
    let seed = 0x0123_4567_89ab_cdef;
    let (stack, runner) =
        embassy_net::new(device, config, RESOURCES.init(StackResources::new()), seed);
    spawner.spawn(net_task(runner)).unwrap();
    stack.wait_config_up().await; // block until DHCP has an address

    // --- UDP socket for RTP egress ------------------------------------------
    let mut rx_meta = [PacketMetadata::EMPTY; 4];
    let mut rx_buf = [0u8; 1500];
    let mut tx_meta = [PacketMetadata::EMPTY; 4];
    let mut tx_buf = [0u8; 1500];
    let mut socket =
        UdpSocket::new(stack, &mut rx_meta, &mut rx_buf, &mut tx_meta, &mut tx_buf);
    socket.bind(5004).unwrap();

    // --- Drive the flagship audio graph, egressing RTP over Ethernet --------
    // VERIFY: the RTP destination (your receiver's IP:port, e.g. an ffmpeg
    // `rtp://` listener). `run_audio_with` runs the full static pipeline and
    // calls `EmbassyNetSender::send` once per 10 ms frame.
    let sender = EmbassyNetSender {
        socket,
        dest: IpEndpoint::new(Ipv4Address::new(192, 168, 1, 100).into(), 5004),
        scratch: [0u8; 1500],
    };
    let _ = run_audio_with(sender).await;

    loop {}
}
