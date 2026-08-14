//! M1035: a CPU element that reads its input through `require_system_slice`
//! declares `input_domains() == {System}`, so the allocation cascade turns a GPU
//! producer upstream into a download instead of letting the pipeline fail with
//! `UnsupportedDomain` at the first frame. The domain-transparent elements must
//! keep `DomainSet::ALL`: narrowing the default globally would force a pointless
//! download through every pass-through in the graph.

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::AsyncElement;

const SYSTEM_ONLY: DomainSet = DomainSet::only(MemoryDomainKind::System);

#[test]
fn muxer_takes_system_frames() {
    assert_eq!(
        g2g_plugins::tsmux::TsMux::new().input_domains(),
        SYSTEM_ONLY
    );
    assert_eq!(
        g2g_plugins::mkvmux::MkvMux::new().input_domains(),
        SYSTEM_ONLY
    );
}

#[test]
fn demuxer_takes_system_frames() {
    assert_eq!(
        g2g_plugins::tsdemux::TsDemux::new().input_domains(),
        SYSTEM_ONLY
    );
}

#[test]
fn audio_transform_takes_system_frames() {
    let convert = g2g_plugins::audioconvert::AudioConvert::new(g2g_core::AudioFormat::PcmS16Le, 2);
    assert_eq!(convert.input_domains(), SYSTEM_ONLY);
    assert_eq!(
        g2g_plugins::wavenc::WavEnc::new().input_domains(),
        SYSTEM_ONLY
    );
}

#[test]
fn video_transform_takes_system_frames() {
    assert_eq!(
        g2g_plugins::videoscale::VideoScale::new(320, 240).input_domains(),
        SYSTEM_ONLY
    );
}

#[cfg(feature = "std")]
#[test]
fn file_sink_takes_system_frames() {
    assert_eq!(
        g2g_plugins::filesink::FileSink::new("out.bin").input_domains(),
        SYSTEM_ONLY
    );
}

/// The guard against an accidental flip of the trait default: a pass-through
/// element imposes no domain requirement, so a GPU frame crosses it untouched.
#[test]
fn passthrough_stays_domain_transparent() {
    assert_eq!(
        g2g_plugins::identity::IdentityTransform::new().input_domains(),
        DomainSet::ALL
    );
}
