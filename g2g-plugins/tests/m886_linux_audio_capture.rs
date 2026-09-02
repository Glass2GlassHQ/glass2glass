//! M886: Linux audio capture sources (`alsasrc` / `pulsesrc`), the non-PipeWire
//! mic paths and the input mirrors of `alsasink` / `pulsesink`.
//!
//! Three things per element: the property surface round-trips (both halves
//! `parse_launch` needs), a launch line naming the element parses, and a capture
//! smoke drives the real device path (`configure_pipeline`, then `run` pulls a
//! few periods) and asserts the pushed buffers carry the negotiated geometry.
//! A host with no sound server / no capture device skips the smoke: `run` fails
//! loud with a hardware error, which the test reads as "no device".
//!
//! Each element is behind its own cargo feature, so run with the ones built:
//! `cargo test -p g2g-plugins --features alsa-src,pulse-src
//!  --test m886_linux_audio_capture`. Validated on this Fedora / PipeWire host
//! (pipewire-alsa + pipewire-pulse), where the default source exists even with
//! no microphone attached.
#![cfg(any(feature = "alsa-src", feature = "pulse-src"))]

use g2g_core::runtime::{block_on, SourceLoop};
use g2g_core::{AudioFormat, Caps, G2gError, OutputSink, PipelinePacket, PropValue, PushOutcome};

/// Periods to pull in a smoke run. At the default 10 ms period that is ~50 ms of
/// audio: enough for the device to have started and delivered real buffers.
const PERIODS: u64 = 5;

/// Collects what a source pushes, so the smoke can assert on buffer geometry
/// rather than only on a return count.
#[derive(Default)]
struct Collect {
    buffers: Vec<Vec<u8>>,
    eos: bool,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .as_system_slice()
                        .expect("capture pushes system memory");
                    self.buffers.push(slice.to_vec());
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Negotiate, configure and run `src`, then assert every pushed buffer holds a
/// whole number of sample frames of the negotiated shape. `Ok(None)` when the
/// device / server is unreachable (the skip path).
fn capture<S: SourceLoop>(src: &mut S, channels: u8, format: AudioFormat) -> Option<usize> {
    let caps = block_on(src.intercept_caps()).expect("caps from the requested config");
    assert_eq!(
        caps,
        Caps::Audio {
            format,
            channels,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    );
    src.configure_pipeline(&caps).expect("configure accepts");

    let mut out = Collect::default();
    let pushed = match block_on(src.run(&mut out)) {
        Ok(n) => n,
        Err(G2gError::Hardware(_)) => return None,
        Err(e) => panic!("capture failed: {e:?}"),
    };

    assert_eq!(pushed, PERIODS, "ran to the requested period count");
    assert_eq!(out.buffers.len() as u64, PERIODS);
    assert!(out.eos, "a bounded capture ends with EOS");

    let frame_bytes = match format {
        AudioFormat::PcmS16Le => 2,
        AudioFormat::PcmF32Le => 4,
        other => panic!("no size for {other:?}"),
    } * channels as usize;
    let mut frames = 0usize;
    for (i, buf) in out.buffers.iter().enumerate() {
        assert!(!buf.is_empty(), "buffer {i} is empty");
        assert_eq!(
            buf.len() % frame_bytes,
            0,
            "buffer {i} ({} bytes) is a whole number of {frame_bytes}-byte frames",
            buf.len()
        );
        frames += buf.len() / frame_bytes;
    }
    assert!(frames > 0, "captured no sample frames");
    Some(frames)
}

#[cfg(feature = "alsa-src")]
mod alsa {
    use super::*;
    use g2g_plugins::alsasrc::AlsaSrc;

    #[test]
    fn properties_round_trip() {
        let mut s = AlsaSrc::new();
        for name in [
            "device",
            "format",
            "samplerate",
            "channels",
            "buffer-time",
            "latency-time",
            "num-buffers",
        ] {
            assert!(
                s.properties().iter().any(|p| p.name == name),
                "alsasrc declares `{name}`"
            );
        }

        s.set_property("device", PropValue::Str("hw:1,0".into()))
            .unwrap();
        s.set_property("format", PropValue::Str("f32le".into()))
            .unwrap();
        s.set_property("samplerate", PropValue::Uint(16_000))
            .unwrap();
        s.set_property("channels", PropValue::Uint(1)).unwrap();
        s.set_property("buffer-time", PropValue::Uint(100_000))
            .unwrap();
        s.set_property("latency-time", PropValue::Uint(25_000))
            .unwrap();
        s.set_property("num-buffers", PropValue::Int(7)).unwrap();

        assert_eq!(
            s.get_property("device"),
            Some(PropValue::Str("hw:1,0".into()))
        );
        assert_eq!(
            s.get_property("format"),
            Some(PropValue::Str("F32LE".into()))
        );
        assert_eq!(s.get_property("samplerate"), Some(PropValue::Uint(16_000)));
        assert_eq!(s.get_property("channels"), Some(PropValue::Uint(1)));
        assert_eq!(
            s.get_property("buffer-time"),
            Some(PropValue::Uint(100_000))
        );
        assert_eq!(
            s.get_property("latency-time"),
            Some(PropValue::Uint(25_000))
        );
        assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(7)));

        // The properties reach the caps the element negotiates on.
        assert_eq!(
            block_on(s.intercept_caps()),
            Ok(Caps::Audio {
                format: AudioFormat::PcmF32Le,
                channels: 1,
                sample_rate: 16_000,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            })
        );

        // -1 is "forever", the gst basesrc spelling.
        s.set_property("num-buffers", PropValue::Int(-1)).unwrap();
        assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(-1)));
        // A format no ALSA device opens is rejected, not stored.
        assert!(s
            .set_property("format", PropValue::Str("opus".into()))
            .is_err());
        assert!(s.set_property("device", PropValue::Uint(1)).is_err());
    }

    /// Default device, default S16LE stereo 48 kHz: the plain mic path.
    #[test]
    fn captures_from_the_default_device() {
        let mut src = AlsaSrc::new().with_num_buffers(PERIODS);
        match capture(&mut src, 2, AudioFormat::PcmS16Le) {
            Some(frames) => eprintln!("m886 alsasrc default: captured {frames} frames"),
            None => eprintln!("skip m886 alsasrc: no reachable ALSA capture device"),
        }
    }

    /// A non-default shape (mono float, 20 ms periods) exercises the format and
    /// period plumbing rather than only the built-in defaults.
    #[test]
    fn captures_mono_float_with_a_longer_period() {
        let mut src = AlsaSrc::new()
            .with_format(AudioFormat::PcmF32Le)
            .with_channels(1)
            .with_latency_time(20_000)
            .with_num_buffers(PERIODS);
        match capture(&mut src, 1, AudioFormat::PcmF32Le) {
            Some(frames) => eprintln!("m886 alsasrc mono/f32: captured {frames} frames"),
            None => eprintln!("skip m886 alsasrc mono/f32: no reachable ALSA capture device"),
        }
    }

    /// A device that cannot be opened fails loud instead of pushing silence.
    #[test]
    fn a_bad_device_name_fails_loud() {
        let mut src = AlsaSrc::new()
            .with_device("g2g-no-such-alsa-device")
            .with_num_buffers(1);
        let caps = block_on(src.intercept_caps()).unwrap();
        src.configure_pipeline(&caps).unwrap();
        let mut out = Collect::default();
        assert!(
            matches!(
                block_on(src.run(&mut out)),
                Err(G2gError::Hardware(g2g_core::HardwareError::Alsa(_)))
            ),
            "an unopenable device is a hardware error"
        );
        assert!(out.buffers.is_empty(), "pushed nothing");
    }
}

#[cfg(feature = "pulse-src")]
mod pulse {
    use super::*;
    use g2g_plugins::pulsesrc::PulseSrc;

    #[test]
    fn properties_round_trip() {
        let mut s = PulseSrc::new();
        for name in [
            "server",
            "device",
            "client-name",
            "format",
            "samplerate",
            "channels",
            "buffer-time",
            "latency-time",
            "num-buffers",
        ] {
            assert!(
                s.properties().iter().any(|p| p.name == name),
                "pulsesrc declares `{name}`"
            );
        }

        s.set_property("server", PropValue::Str("tcp:localhost:4713".into()))
            .unwrap();
        s.set_property("device", PropValue::Str("alsa_output.monitor".into()))
            .unwrap();
        s.set_property("client-name", PropValue::Str("probe".into()))
            .unwrap();
        s.set_property("format", PropValue::Str("s32le".into()))
            .unwrap();
        s.set_property("samplerate", PropValue::Uint(44_100))
            .unwrap();
        s.set_property("channels", PropValue::Uint(1)).unwrap();
        s.set_property("buffer-time", PropValue::Uint(150_000))
            .unwrap();
        s.set_property("latency-time", PropValue::Uint(30_000))
            .unwrap();
        s.set_property("num-buffers", PropValue::Int(9)).unwrap();

        assert_eq!(
            s.get_property("server"),
            Some(PropValue::Str("tcp:localhost:4713".into()))
        );
        assert_eq!(
            s.get_property("device"),
            Some(PropValue::Str("alsa_output.monitor".into()))
        );
        assert_eq!(
            s.get_property("client-name"),
            Some(PropValue::Str("probe".into()))
        );
        assert_eq!(
            s.get_property("format"),
            Some(PropValue::Str("S32LE".into()))
        );
        assert_eq!(s.get_property("samplerate"), Some(PropValue::Uint(44_100)));
        assert_eq!(s.get_property("channels"), Some(PropValue::Uint(1)));
        assert_eq!(
            s.get_property("buffer-time"),
            Some(PropValue::Uint(150_000))
        );
        assert_eq!(
            s.get_property("latency-time"),
            Some(PropValue::Uint(30_000))
        );
        assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(9)));

        assert_eq!(
            block_on(s.intercept_caps()),
            Ok(Caps::Audio {
                format: AudioFormat::PcmS32Le,
                channels: 1,
                sample_rate: 44_100,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            })
        );

        s.set_property("num-buffers", PropValue::Int(-1)).unwrap();
        assert_eq!(s.get_property("num-buffers"), Some(PropValue::Int(-1)));
        assert!(s
            .set_property("format", PropValue::Str("aac".into()))
            .is_err());
        assert!(s.set_property("server", PropValue::Uint(1)).is_err());
    }

    /// Default server + default source: on a desktop that is the mic, or the
    /// output monitor when no mic is attached.
    #[test]
    fn captures_from_the_default_source() {
        let mut src = PulseSrc::new().with_num_buffers(PERIODS);
        match capture(&mut src, 2, AudioFormat::PcmS16Le) {
            Some(frames) => eprintln!("m886 pulsesrc default: captured {frames} frames"),
            None => eprintln!("skip m886 pulsesrc: no reachable PulseAudio server"),
        }
    }

    #[test]
    fn captures_mono_float_with_a_longer_fragment() {
        let mut src = PulseSrc::new()
            .with_format(AudioFormat::PcmF32Le)
            .with_channels(1)
            .with_latency_time(20_000)
            .with_num_buffers(PERIODS);
        match capture(&mut src, 1, AudioFormat::PcmF32Le) {
            Some(frames) => eprintln!("m886 pulsesrc mono/f32: captured {frames} frames"),
            None => eprintln!("skip m886 pulsesrc mono/f32: no reachable PulseAudio server"),
        }
    }

    /// An unreachable server fails loud rather than pushing silence. (A bogus
    /// *source* name is not this test: a server is free to fall back to its
    /// default source, and pipewire-pulse does.)
    #[test]
    fn an_unreachable_server_fails_loud() {
        let mut src = PulseSrc::new()
            .with_server("tcp:127.0.0.1:1")
            .with_num_buffers(1);
        let caps = block_on(src.intercept_caps()).unwrap();
        src.configure_pipeline(&caps).unwrap();
        let mut out = Collect::default();
        assert!(
            matches!(
                block_on(src.run(&mut out)),
                Err(G2gError::Hardware(g2g_core::HardwareError::PulseAudio(_)))
            ),
            "an unreachable server is a hardware error"
        );
        assert!(out.buffers.is_empty(), "pushed nothing");
    }
}

/// Running before `configure_pipeline` is an error, not a silent no-op, and it
/// touches no device on the way to saying so. Device-free, so this runs in CI.
#[test]
fn run_before_configure_is_an_error() {
    let mut out = Collect::default();

    #[cfg(feature = "alsa-src")]
    assert_eq!(
        block_on(g2g_plugins::alsasrc::AlsaSrc::new().run(&mut out)),
        Err(G2gError::NotConfigured)
    );

    #[cfg(feature = "pulse-src")]
    assert_eq!(
        block_on(g2g_plugins::pulsesrc::PulseSrc::new().run(&mut out)),
        Err(G2gError::NotConfigured)
    );

    assert!(out.buffers.is_empty());
}

/// Both capture elements are launch-registered: `parse_launch` resolves the
/// name and applies its properties by kind. Device-free, so this runs in CI.
#[test]
fn launch_lines_parse() {
    use g2g_core::runtime::parse_launch;
    use g2g_plugins::registry::default_registry;

    let reg = default_registry();

    #[cfg(feature = "alsa-src")]
    {
        parse_launch(
            &reg,
            "alsasrc device=default samplerate=48000 channels=2 latency-time=20000 num-buffers=3 \
             ! fakesink",
        )
        .expect("alsasrc launch line parses");
        // An undeclared property is an error, not silently dropped.
        assert!(parse_launch(&reg, "alsasrc nosuchprop=1 ! fakesink").is_err());
        // A format the element cannot open is rejected at parse time.
        assert!(parse_launch(&reg, "alsasrc format=opus ! fakesink").is_err());
    }

    #[cfg(feature = "pulse-src")]
    {
        parse_launch(
            &reg,
            "pulsesrc client-name=g2g-test format=f32le channels=1 buffer-time=100000 \
             num-buffers=3 ! fakesink",
        )
        .expect("pulsesrc launch line parses");
        assert!(parse_launch(&reg, "pulsesrc nosuchprop=1 ! fakesink").is_err());
    }
}
