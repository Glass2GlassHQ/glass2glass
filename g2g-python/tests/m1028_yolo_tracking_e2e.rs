//! End-to-end host of the real gst-python-ml `yolo` element with `track=true`
//! under `GSTML_BACKEND=g2g`, the tracking counterpart of the M322 detector test.
//!
//! Detections reach the frame through `MetaSink::add_object` alone, which M322
//! already covers. Tracking is the path that also needs `add_tracking` and
//! `relate`: the Python `tasks/yolo.py` stages a tracking record per detection
//! and relates the two by handle. This drives two consecutive frames (a tracker
//! needs more than one to keep an identity) and asserts the host materialized
//! those into `Tracking` nodes wired to their detections by a `Tracks` relation,
//! which is what `analyticsoverlay show-track=true` draws.
//!
//! Host-only: needs a gst-python-ml checkout with its `.venv` (torch /
//! ultralytics / opencv), the `yolo11m.pt` model, `data/soccer_tracking.mp4`,
//! and a CUDA GPU (override the device with `G2G_PYML_DEVICE`). It is skipped
//! unless `G2G_PYML_DIR` points at that checkout, so CI (which has none of this)
//! is a no-op. Run it here with:
//!
//! ```sh
//! PYO3_PYTHON=$HOME/src/gst-python-ml/.venv/bin/python \
//! G2G_PYML_DIR=$HOME/src/gst-python-ml \
//!   cargo test -p g2g-python --features analytics --test m1028_yolo_tracking_e2e -- --nocapture
//! ```
#![cfg(feature = "analytics")]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AnalyticsMeta, AnalyticsNode, AsyncElement, Caps, Dim, Frame, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelinePacket, PropValue, PushOutcome, Rate, RawVideoFormat,
    RelationKind,
};
use g2g_python::PyTransform;

const W: u32 = 640;
const H: u32 = 640;

#[derive(Default)]
struct CollectSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");

        self.packets.push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// How many consecutive frames to drive: one to open the tracks, one for the
/// tracker to carry them forward.
const FRAMES: usize = 2;

/// Decode the first [`FRAMES`] frames of `data/soccer_tracking.mp4` to raw
/// `W*H*4` RGBA blobs using the checkout's venv python (opencv). Returns `None`
/// if the helper could not run, so the test skips rather than fails on a partial
/// environment.
fn decode_tracking_frames_rgba(pyml: &str) -> Option<Vec<Vec<u8>>> {
    let venv_py = PathBuf::from(pyml).join(".venv/bin/python");
    let out_stem = std::env::temp_dir().join("g2g_m1028_soccer_640_rgba");
    let script = format!(
        r#"
import cv2, numpy as np
cap = cv2.VideoCapture("{pyml}/data/soccer_tracking.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
for i in range({frames}):
    ok, bgr = cap.read()
    assert ok, "failed to read frame"
    rgb = cv2.cvtColor(cv2.resize(bgr, ({W}, {H})), cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, np.full(({H}, {W}), 255, np.uint8)])
    np.ascontiguousarray(rgba).tofile(f"{out}_{{i}}.raw")
cap.release()
"#,
        pyml = pyml,
        frames = FRAMES,
        W = W,
        H = H,
        out = out_stem.display(),
    );
    let status = Command::new(&venv_py)
        .arg("-c")
        .arg(&script)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let mut frames = Vec::with_capacity(FRAMES);
    for i in 0..FRAMES {
        let bytes = std::fs::read(format!("{}_{i}.raw", out_stem.display())).ok()?;
        if bytes.len() != (W * H * 4) as usize {
            return None;
        }
        frames.push(bytes);
    }
    Some(frames)
}

#[test]
fn hosted_yolo_tracking_lands_as_related_tracking_nodes() {
    let Ok(pyml) = std::env::var("G2G_PYML_DIR") else {
        eprintln!("skip: set G2G_PYML_DIR to a gst-python-ml checkout to run this host-only test");
        return;
    };
    let device = std::env::var("G2G_PYML_DEVICE").unwrap_or_else(|_| "cuda:0".into());

    // The interpreter must see the plugin package plus the venv / user site dirs
    // (torch, ultralytics, opencv). Set before the first GIL acquisition so it is
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

    let Some(frames) = decode_tracking_frames_rgba(&pyml) else {
        eprintln!("skip: could not decode data/soccer_tracking.mp4 via the checkout venv");
        return;
    };

    // `yolo` makes engine-name read-only (it registers its own engine), so only
    // the model, device and tracking flag are forwarded.
    let mut el = PyTransform::new("yolo", "YOLOTransform");
    // The YOLO engine appends the `.pt` extension itself, so this is the stem.
    el.set_property("model-name", PropValue::Str(format!("{pyml}/yolo11m")))
        .unwrap();
    el.set_property("device", PropValue::Str(device)).unwrap();
    el.set_property("track", PropValue::Bool(true)).unwrap();

    let caps = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30),
        interlace: g2g_core::Interlace::Any,
    };
    el.configure_pipeline(&caps).unwrap();

    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for (sequence, rgba) in frames.into_iter().enumerate() {
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(rgba.into_boxed_slice())),
            timing: FrameTiming {
                pts_ns: sequence as u64 * 33_000_000,
                dts_ns: 0,
                duration_ns: 33_000_000,
                capture_ns: 0,
                arrival_ns: 0,
                keyframe: sequence == 0,
            },
            sequence: sequence as u64,
            meta: Default::default(),
        };
        rt.block_on(el.process(PipelinePacket::DataFrame(frame), &mut sink))
            .expect("hosted yolo should run inference without error");
    }

    // The last frame is the one a tracker has had a chance to carry ids into.
    let PipelinePacket::DataFrame(frame) = sink.packets.last().expect("a frame downstream") else {
        panic!("expected a DataFrame downstream");
    };
    let analytics = frame
        .meta
        .get::<AnalyticsMeta>()
        .expect("the detector should attach results as AnalyticsMeta");

    let detections = analytics.detections().count();
    let tracking_ids: Vec<u64> = analytics
        .nodes
        .iter()
        .filter_map(|node| match node {
            AnalyticsNode::Tracking(tracking) => Some(tracking.object_id),
            _ => None,
        })
        .collect();
    let tracks_relations = analytics
        .relations
        .iter()
        .filter(|r| r.kind == RelationKind::Tracks)
        .count();
    eprintln!(
        "hosted yolo produced {detections} detections, {} tracking ids, {tracks_relations} Tracks relations",
        tracking_ids.len()
    );

    assert!(
        detections > 0,
        "expected at least one detection on a soccer frame"
    );
    assert!(
        !tracking_ids.is_empty(),
        "track=true should stage tracking records through MetaSink::add_tracking"
    );
    assert!(
        tracks_relations > 0,
        "each tracking record should be related to its detection via MetaSink::relate"
    );
    // Every Tracks edge must point from a detection node to a tracking node, or
    // the overlay's caption lookup finds nothing.
    for relation in analytics
        .relations
        .iter()
        .filter(|r| r.kind == RelationKind::Tracks)
    {
        assert!(
            matches!(
                analytics.nodes.get(relation.from),
                Some(AnalyticsNode::Detection(_))
            ),
            "a Tracks relation must start at a detection node"
        );
        assert!(
            matches!(
                analytics.nodes.get(relation.to),
                Some(AnalyticsNode::Tracking(_))
            ),
            "a Tracks relation must end at a tracking node"
        );
    }
}
