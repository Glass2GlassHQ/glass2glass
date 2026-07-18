//! Diagnostic: print the camera's libcamera-advertised pixel formats and sizes.
//! Not an assertion test, a probe to understand a device's real capabilities.
//! `cargo test -p g2g-plugins --features libcamera --test libcamera_enum -- --ignored --nocapture`

#![cfg(all(target_os = "linux", feature = "libcamera"))]

use libcamera::camera_manager::CameraManager;
use libcamera::stream::StreamRole;

#[test]
#[ignore = "needs a real camera; diagnostic only"]
fn enumerate_formats() {
    let mgr = CameraManager::new().unwrap();
    let cameras = mgr.cameras();
    let idx = std::env::var("G2G_LIBCAMERA_INDEX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let cam = cameras.get(idx).expect("camera present");
    let cam = cam.acquire().expect("acquire");
    let cfgs = cam
        .generate_configuration(&[StreamRole::ViewFinder])
        .unwrap();
    let cfg = cfgs.get(0).unwrap();
    let formats = cfg.formats();
    for pf in formats.pixel_formats().into_iter() {
        let sizes = formats.sizes(pf);
        println!("format {pf:?}:");
        for s in sizes {
            println!("    {}x{}", s.width, s.height);
        }
    }
    println!("camera id: {}", cam.id());
    // ControlId names are private in the crate; the ControlInfoMap Debug impl
    // does resolve them, so print the whole map plus per-entry ids.
    println!("supported controls: {:#?}", cam.controls());
    for (key, info) in cam.controls().into_iter() {
        println!("    id {key}: {info:?}");
    }
}
