//! M1188 - the Linux audio sinks read a dma-buf in place.
//!
//! `AlsaSink` / `PulseSink` / `PipeWireSink` hand a host pointer to their
//! device, which used to mean system memory only: a dma-buf producer
//! (`localdmabufsrc`, `appsrc`, a capture device) paid a download through the
//! allocation cascade. They now accept the dma-buf domain and map it for
//! reading instead.
//!
//! The buffer here is a real dma-buf: a sealed memfd exported through
//! `/dev/udmabuf`, so the mapping runs against the kernel's own exporter rather
//! than a stand-in. Self-skips where that device is not available (the CI
//! runners), since nothing else makes a dma-buf without a GPU.
#![cfg(all(target_os = "linux", feature = "alsa-sink"))]

use g2g_core::memory::{MemoryDomain, OwnedDmaBuf};
use g2g_core::{
    AsyncElement, Caps, Frame, FrameTiming, G2gError, OutputSink, PipelinePacket, PushOutcome,
};
use g2g_plugins::alsasink::AlsaSink;
use g2g_plugins::dmabufmap::{frame_bytes, DmaBufReadMap};

/// One page, the smallest a udmabuf export takes.
const BUFFER_BYTES: usize = 4096;
/// A pattern with no run of equal bytes, so a mapping at the wrong offset
/// cannot match by accident.
fn pattern() -> Vec<u8> {
    (0..BUFFER_BYTES).map(|i| (i % 251) as u8).collect()
}

/// `UDMABUF_CREATE`: `_IOW('u', 0x42, struct udmabuf_create)`.
const UDMABUF_CREATE: libc::c_ulong = 0x4018_7542;

#[repr(C)]
struct UdmabufCreate {
    memfd: u32,
    flags: u32,
    offset: u64,
    size: u64,
}

/// Export `bytes` as a real dma-buf through `/dev/udmabuf`. `None` when the
/// device is absent or not readable by this user.
fn udmabuf_of(bytes: &[u8]) -> Option<OwnedDmaBuf> {
    let name = c"g2g-m1188";
    // SAFETY: opening a device path and creating a memfd are plain syscalls
    // whose arguments are valid for the call.
    let (device, memfd) = unsafe {
        let device = libc::open(c"/dev/udmabuf".as_ptr(), libc::O_RDWR);
        if device < 0 {
            return None;
        }
        let memfd = libc::memfd_create(name.as_ptr(), libc::MFD_ALLOW_SEALING);
        if memfd < 0 {
            libc::close(device);
            return None;
        }
        (device, memfd)
    };
    // SAFETY: `memfd` is the fresh descriptor above; the write covers exactly
    // the bytes just sized for, and the seal is what udmabuf requires.
    let created = unsafe {
        let sized = libc::ftruncate(memfd, bytes.len() as libc::off_t) == 0
            && libc::write(memfd, bytes.as_ptr().cast(), bytes.len()) == bytes.len() as isize
            && libc::fcntl(memfd, libc::F_ADD_SEALS, libc::F_SEAL_SHRINK) == 0;
        if !sized {
            libc::close(memfd);
            libc::close(device);
            return None;
        }
        let create = UdmabufCreate {
            memfd: memfd as u32,
            flags: 0,
            offset: 0,
            size: bytes.len() as u64,
        };
        let fd = libc::ioctl(device, UDMABUF_CREATE, &create as *const UdmabufCreate);
        libc::close(memfd);
        libc::close(device);
        fd
    };
    if created < 0 {
        return None;
    }
    // SAFETY: `created` is a fresh dma-buf fd this test solely owns, handed to
    // `OwnedDmaBuf` which closes it on drop.
    Some(unsafe { OwnedDmaBuf::from_raw(created, 0, 0) })
}

/// A dma-buf frame, as a producer exporting per-frame buffers makes one.
fn dmabuf_frame(dmabuf: OwnedDmaBuf) -> Frame {
    Frame::new(MemoryDomain::DmaBuf(dmabuf), FrameTiming::default(), 0)
}

#[test]
fn a_dmabuf_maps_to_the_bytes_its_exporter_wrote() {
    let bytes = pattern();
    let Some(dmabuf) = udmabuf_of(&bytes) else {
        eprintln!("SKIP: /dev/udmabuf is not available, so no dma-buf to map");
        return;
    };
    let map = DmaBufReadMap::read(&dmabuf).expect("the dma-buf maps for reading");
    assert_eq!(
        map.as_slice(),
        &bytes[..],
        "the mapping is the exported page"
    );
}

/// The payload reader the sinks share: a dma-buf frame reads without a copy
/// into system memory first, and a system frame still reads as before.
#[test]
fn the_sink_payload_reader_takes_either_domain() {
    let bytes = pattern();
    let system = Frame::new(
        MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
            bytes.clone().into_boxed_slice(),
        )),
        FrameTiming::default(),
        0,
    );
    assert_eq!(
        frame_bytes(&system, "test")
            .expect("system frames read")
            .as_slice(),
        &bytes[..]
    );

    let Some(dmabuf) = udmabuf_of(&bytes) else {
        eprintln!("SKIP: /dev/udmabuf is not available, so no dma-buf to map");
        return;
    };
    let frame = dmabuf_frame(dmabuf);
    assert_eq!(
        frame_bytes(&frame, "test")
            .expect("dma-buf frames read")
            .as_slice(),
        &bytes[..]
    );
}

/// An offset dma-buf carries its payload from that offset on, the shape a
/// producer packing several frames into one buffer exports.
#[test]
fn an_offset_dmabuf_reads_from_its_offset() {
    const OFFSET: u32 = 1024;
    let bytes = pattern();
    let Some(dmabuf) = udmabuf_of(&bytes) else {
        eprintln!("SKIP: /dev/udmabuf is not available, so no dma-buf to map");
        return;
    };
    // Re-export at an offset: the fd is the same buffer, the frame names where
    // its payload starts.
    // SAFETY: `dmabuf` owns an open dma-buf fd for the whole call.
    let raw = unsafe { libc::dup(dmabuf.as_raw()) };
    assert!(raw >= 0, "dup the dma-buf fd");
    // SAFETY: `raw` is the fresh duplicate above, owned from here on.
    let owned = unsafe { OwnedDmaBuf::from_raw(raw, 0, OFFSET) };
    let map = DmaBufReadMap::read(&owned).expect("the dma-buf maps at its offset");
    assert_eq!(map.as_slice(), &bytes[OFFSET as usize..]);
}

/// The whole path, against a real device: a dma-buf frame goes to `alsasink`
/// and reaches the card through libasound, mapped rather than copied in. Plays
/// to ALSA's `null` PCM, a real device that makes no sound, so the test runs
/// anywhere libasound does; self-skips when the device or `/dev/udmabuf` is
/// missing.
#[tokio::test]
async fn a_dmabuf_frame_plays_through_alsasink() {
    /// One period of S16LE stereo at 48 kHz, the shape the sink is configured
    /// for.
    const SAMPLE_RATE: u32 = 48_000;
    const CHANNELS: u8 = 2;
    const FRAMES: usize = 480;
    const BYTES: usize = FRAMES * CHANNELS as usize * 2;

    // A quiet ramp: real samples, so a mis-mapped buffer is not silence either
    // way.
    let mut samples = Vec::with_capacity(BUFFER_BYTES);
    for i in 0..BYTES / 2 {
        samples.extend_from_slice(&((i as i16 % 2048) - 1024).to_le_bytes());
    }
    samples.resize(BUFFER_BYTES, 0);

    let Some(dmabuf) = udmabuf_of(&samples) else {
        eprintln!("SKIP: /dev/udmabuf is not available, so no dma-buf to play");
        return;
    };

    let mut sink = AlsaSink::with_device("null");
    let caps = Caps::Audio {
        format: g2g_core::AudioFormat::PcmS16Le,
        channels: CHANNELS,
        sample_rate: SAMPLE_RATE,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    };
    if AsyncElement::configure_pipeline(&mut sink, &caps).is_err() {
        eprintln!("SKIP: no ALSA null device on this host");
        return;
    }

    let mut discard = DiscardOut;
    AsyncElement::process(
        &mut sink,
        PipelinePacket::DataFrame(dmabuf_frame(dmabuf)),
        &mut discard,
    )
    .await
    .expect("the dma-buf frame reached the device");
    AsyncElement::process(&mut sink, PipelinePacket::Eos, &mut discard)
        .await
        .expect("eos drains the device");
    assert!(
        sink.frames_rendered() > 0,
        "the card was handed the mapped buffer"
    );
}

/// A sink pushes nothing on, so its output goes nowhere.
struct DiscardOut;

impl OutputSink for DiscardOut {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// The sink advertises both domains, which is what keeps the allocation cascade
/// from demanding a download off a dma-buf producer.
#[test]
fn the_sink_accepts_the_dmabuf_domain() {
    let sink = AlsaSink::new();
    let domains = AsyncElement::input_domains(&sink);
    assert!(domains.contains(g2g_core::memory::MemoryDomainKind::DmaBuf));
    assert!(domains.contains(g2g_core::memory::MemoryDomainKind::System));
}
