//! M1039: the fan-in / fan-out halves declare their memory domains too, the
//! multi-pad counterpart of `m1035_input_domains_declared`. A muxer or demux
//! that reads host memory has to say so, otherwise the allocation cascade lets a
//! GPU producer feeding it stay on the device.

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::{MultiInputElement, MultiOutputElement};

const SYSTEM_ONLY: DomainSet = DomainSet::only(MemoryDomainKind::System);

#[test]
fn fan_in_muxer_takes_system_frames() {
    assert_eq!(
        g2g_plugins::tsmuxn::TsMux::new(2).input_domains(),
        SYSTEM_ONLY
    );
    assert_eq!(
        g2g_plugins::mp4muxn::Mp4MuxN::new(2).input_domains(),
        SYSTEM_ONLY
    );
}

#[test]
fn fan_out_demuxer_takes_system_frames() {
    let demux = g2g_plugins::tsdemux::TsDemuxN::new(vec![g2g_plugins::tsdemux::TsStream::H264]);
    assert_eq!(MultiOutputElement::input_domains(&demux), SYSTEM_ONLY);
}

/// The GPU compositor samples a delivered texture where it lies, so unlike its
/// CPU sibling it must not narrow to system memory.
#[cfg(feature = "wgpu-sink")]
#[test]
fn gpu_compositor_takes_a_texture_or_system_frames() {
    let compositor = g2g_plugins::wgpucompositor::WgpuCompositor::new(
        64,
        64,
        vec![g2g_plugins::compositor::CompositorPad::at(0, 0)],
    );
    assert_eq!(
        compositor.input_domains(),
        DomainSet::only(MemoryDomainKind::WgpuTexture).with(MemoryDomainKind::System)
    );
}
