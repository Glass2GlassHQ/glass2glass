//! A pre-populated element [`Registry`] (M107), so a `gst-launch` text pipeline
//! and `gst-inspect` work out of the box without the caller hand-registering
//! every element.
//!
//! [`default_registry`] registers the standard `no_std`-baseline elements under
//! their conventional names: the test sources, the video and audio transform
//! chains, and the `fakesink` / `filesink` sinks. Each is default-constructed and
//! then configured by the parser from its `key=value` properties (M104/M106).
//!
//! `std`-only (the `Registry` is). Feature-gated capture / decode / display
//! elements (`v4l2src`, `ffmpeg`, `waylandsink`, ...) are not registered here yet;
//! a caller adds them to the returned registry with `register_source` /
//! `register_launch` as their features are enabled. `filesrc` is also omitted: it
//! needs output caps at construction (a raw byte stream has no self-describing
//! type), which the property bag cannot yet supply.

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::runtime::{LaunchFactory, Registry, SourceFactory};
use g2g_core::{AudioFormat, Caps, Dim, Rate, RawVideoFormat};

use crate::audioconvert::AudioConvert;
use crate::audioresample::AudioResample;
use crate::audiotestsrc::AudioTestSrc;
use crate::fakesink::FakeSink;
use crate::filesink::FileSink;
use crate::h264parse::H264Parse;
use crate::identity::IdentityTransform;
use crate::videoconvert::VideoConvert;
use crate::videocrop::VideoCrop;
use crate::videoflip::{FlipMethod, VideoFlip};
use crate::videorate::VideoRate;
use crate::videoscale::VideoScale;
use crate::videotestsrc::VideoTestSrc;

/// A [`Registry`] pre-populated with the standard elements, ready for
/// [`parse_launch`](g2g_core::runtime::parse_launch) and
/// [`inspect`](g2g_core::runtime::Registry::inspect).
///
/// ```text
/// videotestsrc num-buffers=10 ! videoconvert format=nv12 ! videoscale width=320 height=240 ! fakesink
/// audiotestsrc num-buffers=5 freq=440 ! audioconvert channels=1 ! audioresample samplerate=16000 ! fakesink
/// ```
pub fn default_registry() -> Registry {
    let mut reg = Registry::new();

    // Sources. The output caps are the autoplug `decodebin` input; the parser
    // only calls the constructor and applies properties.
    reg.register_source(SourceFactory::new(
        "videotestsrc",
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        },
        || Box::new(VideoTestSrc::new(320, 240, 30, 0)),
    ));
    reg.register_source(SourceFactory::new(
        "audiotestsrc",
        Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: 48_000 },
        || Box::new(AudioTestSrc::new(48_000, 2, 440, 0)),
    ));

    // Video transforms.
    reg.register_launch(LaunchFactory::of::<VideoConvert>("videoconvert", || {
        Box::new(VideoConvert::new(RawVideoFormat::Rgba8))
    }));
    reg.register_launch(LaunchFactory::of::<VideoScale>("videoscale", || {
        Box::new(VideoScale::new(0, 0))
    }));
    reg.register_launch(LaunchFactory::of::<VideoCrop>("videocrop", || {
        Box::new(VideoCrop::new(0, 0, 0, 0))
    }));
    reg.register_launch(LaunchFactory::of::<VideoFlip>("videoflip", || {
        Box::new(VideoFlip::new(FlipMethod::HorizontalMirror))
    }));
    // VideoRate / IdentityTransform have no pad templates declared.
    reg.register_launch(LaunchFactory::new("videorate", Vec::new(), || {
        Box::new(VideoRate::new(30.0))
    }));

    // Audio transforms.
    reg.register_launch(LaunchFactory::of::<AudioConvert>("audioconvert", || {
        Box::new(AudioConvert::new(AudioFormat::PcmS16Le, 2))
    }));
    reg.register_launch(LaunchFactory::of::<AudioResample>("audioresample", || {
        Box::new(AudioResample::new(48_000))
    }));

    // Parsers + passthrough.
    reg.register_launch(LaunchFactory::of::<H264Parse>("h264parse", || Box::new(H264Parse::new())));
    reg.register_launch(LaunchFactory::new("identity", Vec::new(), || {
        Box::new(IdentityTransform::new())
    }));

    // Sinks.
    reg.register_launch(LaunchFactory::of::<FakeSink>("fakesink", || Box::new(FakeSink::new())));
    reg.register_launch(LaunchFactory::of::<FileSink>("filesink", || Box::new(FileSink::new(""))));

    reg
}
