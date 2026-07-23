//! M738: AVFoundation capture on the macOS CI runner. The runner has no
//! camera (and capture is TCC permission-gated), so the video test probes the
//! graceful denial path; the audio test captures for real if the runner
//! exposes an input device (the Core Audio run showed it does) and probes
//! otherwise.
#![cfg(all(target_os = "macos", feature = "avfoundation"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::PipelinePacket;
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::SourceLoop;
use g2g_core::{G2gError, OutputSink, PushOutcome};
use g2g_plugins::avf::{AvfAudioSrc, AvfVideoSrc};

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
async fn camera_captures_or_reports_no_device() {
    let mut src = AvfVideoSrc::new(3);
    let caps = src.intercept_caps().await.expect("caps");
    match src.configure_pipeline(&caps) {
        Err(G2gError::Hardware(_)) => {
            // No camera / permission denied: the structured probe result.
            eprintln!("skipping camera capture: no device or permission");
            return;
        }
        other => {
            other.expect("configure");
        }
    }
    let mut out = Collect::default();
    match src.run(&mut out).await {
        Ok(n) => {
            assert_eq!(n, 3, "captured the requested frames");
            for d in &out.frames {
                let Some(s) = d.as_system_slice() else {
                    panic!("packed mode emits System frames");
                };
                assert_eq!(s.len(), 640 * 480 * 3 / 2, "tight VGA NV12");
            }
            eprintln!("captured {n} camera frames");
        }
        Err(G2gError::Hardware(_)) => {
            eprintln!("skipping camera capture: device delivered no data");
        }
        Err(e) => panic!("camera capture failed unexpectedly: {e:?}"),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn camera_cv_output_or_reports_no_device() {
    let mut src = AvfVideoSrc::new(2).with_cv_output();
    let caps = src.intercept_caps().await.expect("caps");
    if matches!(src.configure_pipeline(&caps), Err(G2gError::Hardware(_))) {
        eprintln!("skipping zero-copy camera capture: no device or permission");
        return;
    }
    let mut out = Collect::default();
    match src.run(&mut out).await {
        Ok(n) => {
            assert_eq!(n, 2);
            for d in &out.frames {
                let MemoryDomain::CvPixelBuffer(buf) = d else {
                    panic!("cv-output emits CvPixelBuffer frames, got {d:?}");
                };
                assert_eq!((buf.width, buf.height), (640, 480));
            }
            eprintln!("captured {n} zero-copy camera frames");
        }
        Err(G2gError::Hardware(_)) => {
            eprintln!("skipping zero-copy camera capture: device delivered no data");
        }
        Err(e) => panic!("camera capture failed unexpectedly: {e:?}"),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn microphone_captures_or_reports_no_device() {
    let mut src = AvfAudioSrc::new(48_000, 2, 3);
    let caps = src.intercept_caps().await.expect("caps");
    match src.configure_pipeline(&caps) {
        Err(G2gError::Hardware(_)) => {
            eprintln!("skipping mic capture: no device or permission");
            return;
        }
        other => {
            other.expect("configure");
        }
    }
    let mut out = Collect::default();
    match src.run(&mut out).await {
        Ok(n) => {
            assert_eq!(n, 3, "captured the requested buffers");
            assert!(
                out.frames
                    .iter()
                    .all(|d| matches!(d, MemoryDomain::System(_))),
                "PCM buffers are System bytes"
            );
            eprintln!("captured {n} mic buffers");
        }
        Err(G2gError::Hardware(_)) => {
            eprintln!("skipping mic capture: device delivered no data");
        }
        Err(e) => panic!("mic capture failed unexpectedly: {e:?}"),
    }
}
