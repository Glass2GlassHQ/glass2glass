//! M322: end-to-end host of the real gst-python-ml `objectdetector` under
//! `GSTML_BACKEND=g2g`.
//!
//! `PyTransform` imports the actual `objectdetector` element from a gst-python-ml
//! checkout, forwards `engine-name` / `model-name` / `device`, and drives one
//! real video frame through `g2g_process`. The hosted element loads its ONNX
//! YOLO model and runs inference; this asserts the detections come back as
//! `AnalyticsMeta` on the output frame.
//!
//! Host-only: needs a gst-python-ml checkout with its `.venv` (torch /
//! onnxruntime / opencv), the `yolo11m.onnx` model, `data/people.mp4`, and a
//! CUDA GPU (override the device with `G2G_PYML_DEVICE`). It is skipped unless
//! `G2G_PYML_DIR` points at that checkout, so CI (which has none of this) is a
//! no-op. Run it here with:
//!
//! ```sh
//! PYO3_PYTHON=$HOME/src/gst-python-ml/.venv/bin/python \
//! G2G_PYML_DIR=$HOME/src/gst-python-ml \
//!   cargo test -p g2g-python --features analytics --test m322_objectdetector_e2e -- --nocapture
//! ```
#![cfg(feature = "analytics")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;
use std::process::Command;

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AnalyticsMeta, AsyncElement, Caps, Dim, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PropValue, PushOutcome, Rate, RawVideoFormat,
};
use g2g_python::PyTransform;

const W: u32 = 640;
const H: u32 = 640;

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

/// Decode one frame of `data/people.mp4` to a raw `W*H*4` RGBA blob using the
/// checkout's venv python (opencv). Returns the bytes, or `None` if the helper
/// could not run (so the test skips rather than fails on a partial environment).
fn decode_people_frame_rgba(pyml: &str) -> Option<Vec<u8>> {
    let venv_py = PathBuf::from(pyml).join(".venv/bin/python");
    let out_path = std::env::temp_dir().join("g2g_m322_people_640_rgba.raw");
    let script = format!(
        r#"
import cv2, numpy as np
cap = cv2.VideoCapture("{pyml}/data/people.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ok, bgr = cap.read(); cap.release()
assert ok, "failed to read frame"
rgb = cv2.cvtColor(cv2.resize(bgr, ({W}, {H})), cv2.COLOR_BGR2RGB)
rgba = np.dstack([rgb, np.full(({H}, {W}), 255, np.uint8)])
np.ascontiguousarray(rgba).tofile("{out}")
"#,
        pyml = pyml,
        W = W,
        H = H,
        out = out_path.display(),
    );
    let status = Command::new(&venv_py)
        .arg("-c")
        .arg(&script)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&out_path).ok()?;
    (bytes.len() == (W * H * 4) as usize).then_some(bytes)
}

#[test]
fn hosted_objectdetector_loads_model_and_detects() {
    let Ok(pyml) = std::env::var("G2G_PYML_DIR") else {
        eprintln!("skip: set G2G_PYML_DIR to a gst-python-ml checkout to run this host-only test");
        return;
    };
    let device = std::env::var("G2G_PYML_DEVICE").unwrap_or_else(|_| "cuda:0".into());

    // The interpreter must see the plugin package plus the venv / user site dirs
    // (torch, onnxruntime, opencv). Set before the first GIL acquisition so it is
    // on sys.path at interpreter init.
    let pv = "python3.14";
    let pythonpath = [
        format!("{pyml}/plugins/python"),
        format!("{pyml}/.venv/lib/{pv}/site-packages"),
        format!("{pyml}/.venv/lib64/{pv}/site-packages"),
        format!(
            "{}/.local/lib/{pv}/site-packages",
            std::env::var("HOME").unwrap_or_default()
        ),
    ]
    .join(":");
    std::env::set_var("PYTHONPATH", pythonpath);

    let Some(rgba) = decode_people_frame_rgba(&pyml) else {
        eprintln!("skip: could not decode data/people.mp4 via the checkout venv");
        return;
    };

    let mut el = PyTransform::new("objectdetector", "ObjectDetector");
    el.set_property("engine-name", PropValue::Str("onnx".into()))
        .unwrap();
    el.set_property("model-name", PropValue::Str(format!("{pyml}/yolo11m.onnx")))
        .unwrap();
    el.set_property("device", PropValue::Str(device)).unwrap();

    let caps = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30),
    };
    el.configure_pipeline(&caps).unwrap();

    let frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(rgba.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: 0,
            dts_ns: 0,
            duration_ns: 0,
            capture_ns: 0,
            arrival_ns: 0,
            keyframe: true,
        },
        sequence: 0,
        meta: Default::default(),
    };

    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(el.process(PipelinePacket::DataFrame(frame), &mut sink))
        .expect("hosted objectdetector should run inference without error");

    let PipelinePacket::DataFrame(frame) = &sink.packets[0] else {
        panic!("expected a DataFrame downstream");
    };
    let analytics = frame
        .meta
        .get::<AnalyticsMeta>()
        .expect("the detector should attach detections as AnalyticsMeta");
    let dets: Vec<_> = analytics.detections().collect();
    eprintln!(
        "hosted objectdetector produced {} detections on the people frame",
        dets.len()
    );
    assert!(
        !dets.is_empty(),
        "expected at least one detection on a frame full of people"
    );
    // label 0 is COCO "person"; a people clip must contain at least one.
    assert!(
        dets.iter().any(|d| d.label == 0),
        "expected at least one person (label 0) detection"
    );
}
