//! M198 step 4d: `PySource`, a Python frame source hosted as a g2g `SourceLoop`.
//!
//! Negotiation + launch registration are always testable; the live produce loop
//! runs under the `python` feature.

use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, Dim, Rate, RawVideoFormat};
use g2g_python::PySource;

fn rgba(w: u32, h: u32, fps: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(fps),
    }
}

#[test]
fn advertises_its_fixed_caps() {
    let mut src = PySource::new("counter", "CounterSource").with_caps(rgba(2, 1, 25));
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    // intercept_caps is synchronous (a Ready future) but typed async.
    let caps = rt.block_on(src.intercept_caps()).unwrap();
    assert_eq!(caps, rgba(2, 1, 25));
}

#[test]
fn properties_drive_the_output_caps() {
    let mut src = PySource::new("counter", "CounterSource");
    use g2g_core::PropValue;
    src.set_property("width", PropValue::Uint(640)).unwrap();
    src.set_property("height", PropValue::Uint(480)).unwrap();
    src.set_property("format", PropValue::Str("NV12".into()))
        .unwrap();
    src.set_property("num-buffers", PropValue::Int(5)).unwrap();
    assert_eq!(src.get_property("width"), Some(PropValue::Uint(640)));
    assert_eq!(
        src.get_property("format"),
        Some(PropValue::Str("NV12".into()))
    );
    assert_eq!(src.get_property("num-buffers"), Some(PropValue::Int(5)));
    assert!(src.set_property("bogus", PropValue::Int(1)).is_err());
}

#[test]
fn pysrc_parses_as_a_launch_source() {
    let mut reg = g2g_plugins::registry::default_registry();
    g2g_python::register(&mut reg);
    let line = "pysrc module=counter class=CounterSource num-buffers=3 ! fakesink";
    let graph = g2g_core::runtime::parse_launch(&reg, line);
    assert!(
        graph.is_ok(),
        "pysrc should parse as a launch source: {:?}",
        graph.err()
    );
}

#[cfg(feature = "python")]
#[test]
fn produces_frames_until_python_signals_eos() {
    use core::future::Future;
    use core::pin::Pin;
    use g2g_core::{Frame, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome};

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

    std::env::set_var(
        "PYTHONPATH",
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
    );

    // Fixture CounterSource ends after 3 frames; no num-buffers cap, so EOS
    // comes from Python's `g2g_produce` returning False.
    let mut src = PySource::new("echo_element", "CounterSource").with_caps(rgba(2, 1, 30));
    src.configure_pipeline(&rgba(2, 1, 30)).unwrap();

    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let count = rt.block_on(src.run(&mut sink)).unwrap();

    assert_eq!(count, 3);
    assert_eq!(src.emitted_count(), 3);
    // 3 DataFrames carrying byte0 = 0,1,2, then a terminal Eos.
    let bytes: Vec<u8> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(Frame {
                domain: MemoryDomain::System(s),
                ..
            }) => Some(s.as_slice()[0]),
            _ => None,
        })
        .collect();
    assert_eq!(bytes, vec![0, 1, 2]);
    assert!(matches!(sink.packets.last(), Some(PipelinePacket::Eos)));
}
