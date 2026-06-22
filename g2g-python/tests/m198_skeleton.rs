//! M198: PyTransform skeleton, the always-compiled (no `python` feature)
//! surface. Caps negotiation, the bridged property bag, and the lifecycle
//! guard all work without an interpreter; the per-frame Python call is covered
//! separately under `--features python` (needs libpython).

use core::future::Future;
use core::pin::Pin;

use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, OutputSink, PipelinePacket, PropValue,
    PushOutcome, Rate, RawVideoFormat,
};
use g2g_python::PyTransform;

fn rgba(w: u32, h: u32, fps: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(fps),
    }
}

/// Collects everything pushed downstream, so a test can assert what flowed.
#[derive(Default)]
struct CollectSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        self.packets.push(packet);
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

#[test]
fn negotiates_concrete_geometry_against_any_accept() {
    let el = PyTransform::new("action", "ActionTransform");
    let upstream = rgba(640, 480, 30);
    // Default accept is RGBA at Any dims/rate; the intersection fixes the
    // concrete upstream geometry (never leaves Any to trip fixate).
    assert_eq!(el.intercept_caps(&upstream).unwrap(), upstream);
}

#[test]
fn rejects_a_format_outside_the_accepted_set() {
    let el = PyTransform::new("action", "ActionTransform");
    let nv12 = Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30),
    };
    assert!(el.intercept_caps(&nv12).is_err());
}

#[test]
fn with_accept_hosts_a_different_format() {
    let nv12 = Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    };
    let el = PyTransform::new("action", "ActionTransform").with_accept(nv12);
    let upstream = Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(1280),
        height: Dim::Fixed(720),
        framerate: Rate::Fixed(25),
    };
    assert_eq!(el.intercept_caps(&upstream).unwrap(), upstream);
}

#[test]
fn configure_accepts_fixed_caps() {
    let mut el = PyTransform::new("action", "ActionTransform");
    // No `python` feature: configure negotiates and arms the element without
    // instantiating an interpreter.
    let outcome = el.configure_pipeline(&rgba(320, 240, 15)).unwrap();
    assert!(matches!(outcome, ConfigureOutcome::Accepted));
}

#[test]
fn properties_round_trip() {
    let mut el = PyTransform::new("action", "ActionTransform").with_draw_label(true);
    assert_eq!(el.get_property("draw-label"), Some(PropValue::Bool(true)));
    assert_eq!(el.get_property("module"), Some(PropValue::Str("action".into())));

    el.set_property("class", PropValue::Str("OtherTransform".into())).unwrap();
    assert_eq!(el.get_property("class"), Some(PropValue::Str("OtherTransform".into())));

    assert!(el.set_property("nope", PropValue::Bool(true)).is_err());
}

#[test]
fn process_before_configure_is_rejected() {
    let mut el = PyTransform::new("action", "ActionTransform");
    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let res = rt.block_on(el.process(PipelinePacket::Eos, &mut sink));
    assert_eq!(res, Err(G2gError::NotConfigured));
}

#[test]
fn eos_after_configure_drains_to_nothing() {
    let mut el = PyTransform::new("action", "ActionTransform");
    el.configure_pipeline(&rgba(320, 240, 15)).unwrap();
    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    rt.block_on(el.process(PipelinePacket::Eos, &mut sink)).unwrap();
    // Stateless host: EOS buffers nothing, so nothing is pushed.
    assert!(sink.packets.is_empty());
}
