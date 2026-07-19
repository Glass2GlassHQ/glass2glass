//! M739: ScreenCaptureKit on the macOS CI runner. Screen recording is TCC
//! permission-gated (and the lookup needs a window server), so this probes
//! like the camera test: a denied / absent capture surfaces as the structured
//! hardware error; with permission granted (a real Mac) it captures the
//! display for real.
#![cfg(all(target_os = "macos", feature = "screencapture"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::PipelinePacket;
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, Dim, G2gError, OutputSink, PushOutcome, RawVideoFormat};
use g2g_plugins::sck::ScreenCaptureSrc;

#[derive(Default)]
struct Collect {
    frames: Vec<MemoryDomain>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                self.frames.push(f.domain);
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[tokio::test(flavor = "current_thread")]
async fn captures_display_or_reports_denied() {
    let mut src = ScreenCaptureSrc::new(3);
    let caps = match src.intercept_caps().await {
        Err(G2gError::Hardware(_)) => {
            // Denied screen-recording permission or no display: the structured
            // probe result, not a hang or a panic.
            eprintln!("skipping screen capture: permission denied or no display");
            return;
        }
        other => other.expect("caps"),
    };
    let Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        ..
    } = caps
    else {
        panic!("display caps must be fixed NV12, got {caps:?}");
    };
    eprintln!("display reports {w}x{h}");
    src.configure_pipeline(&caps).expect("configure");

    let mut out = Collect::default();
    match src.run(&mut out).await {
        Ok(n) => {
            assert_eq!(n, 3, "captured the requested frames");
            for d in &out.frames {
                let MemoryDomain::System(s) = d else {
                    panic!("packed mode emits System frames");
                };
                assert_eq!(
                    s.as_slice().len(),
                    (w * h * 3 / 2) as usize,
                    "tight NV12 at the display geometry"
                );
            }
            eprintln!("captured {n} display frames");
        }
        Err(G2gError::Hardware(_)) => {
            // The stream opened but delivered nothing (an asynchronously
            // denied start): surfaced within the liveness deadline, no hang.
            eprintln!("skipping screen capture: stream delivered no frames");
        }
        Err(e) => panic!("screen capture failed unexpectedly: {e:?}"),
    }
}
