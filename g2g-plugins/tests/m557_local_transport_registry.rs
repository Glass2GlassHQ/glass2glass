//! M556/M557: the local zero-copy transport pairs are registered in the default
//! registry (so `gst-inspect` / `parse_launch` see them) and expose their
//! `location` (Unix socket path) property. Feature-gated: each block only runs
//! when its transport feature is on (both are Linux + local, CI-excluded), so the
//! test is a no-op in the default build.

#![cfg(target_os = "linux")]

#[cfg(any(feature = "local-ipc", feature = "local-dmabuf"))]
use g2g_core::PropValue;
#[cfg(any(feature = "local-ipc", feature = "local-dmabuf"))]
use g2g_plugins::registry::default_registry;

#[cfg(feature = "local-ipc")]
#[test]
fn local_cuda_pair_registered_with_location() {
    let reg = default_registry();

    let mut src = reg
        .make_source("localcudasrc")
        .expect("localcudasrc registered");
    src.set_property("location", PropValue::Str("/tmp/g2g-test-cuda.sock".into()))
        .unwrap();
    assert_eq!(
        src.get_property("location"),
        Some(PropValue::Str("/tmp/g2g-test-cuda.sock".into()))
    );

    let mut sink = reg
        .make_element("localcudasink")
        .expect("localcudasink registered");
    sink.set_property("location", PropValue::Str("/tmp/g2g-test-cuda.sock".into()))
        .unwrap();
    assert_eq!(
        sink.get_property("location"),
        Some(PropValue::Str("/tmp/g2g-test-cuda.sock".into()))
    );

    // The sink is a launch element, not a source (and vice versa).
    assert!(reg.make_source("localcudasink").is_none());
    assert!(reg.make_element("localcudasrc").is_none());
}

#[cfg(feature = "local-dmabuf")]
#[test]
fn dmabuf_pair_registered_with_location() {
    let reg = default_registry();

    let mut src = reg.make_source("dmabufsrc").expect("dmabufsrc registered");
    src.set_property("location", PropValue::Str("/tmp/g2g-test-dma.sock".into()))
        .unwrap();
    assert_eq!(
        src.get_property("location"),
        Some(PropValue::Str("/tmp/g2g-test-dma.sock".into()))
    );

    let mut sink = reg
        .make_element("dmabufsink")
        .expect("dmabufsink registered");
    sink.set_property("location", PropValue::Str("/tmp/g2g-test-dma.sock".into()))
        .unwrap();
    assert_eq!(
        sink.get_property("location"),
        Some(PropValue::Str("/tmp/g2g-test-dma.sock".into()))
    );

    assert!(reg.make_source("dmabufsink").is_none());
    assert!(reg.make_element("dmabufsrc").is_none());
}
