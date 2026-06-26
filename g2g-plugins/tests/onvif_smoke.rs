//! M311: smoke tests for the ONVIF source element.
//!
//! The element-wiring tests run always (no network): property round-trip and
//! the unconfigured-source error path. The live tests are `#[ignore]` and need
//! a real camera on the LAN. Override the target via env vars:
//!
//! ```sh
//! # WS-Discovery (cameras must answer on the LAN multicast group):
//! cargo test -p g2g-plugins --features onvif -- --ignored discover_lan
//!
//! # Stream-URI resolution against a known camera:
//! G2G_ONVIF_URL=http://192.168.1.50/onvif/device_service \
//! G2G_ONVIF_USER=admin G2G_ONVIF_PASS=secret \
//!     cargo test -p g2g-plugins --features onvif -- --ignored resolve_uri
//! ```

#![cfg(feature = "onvif")]

use core::time::Duration;

use g2g_core::runtime::SourceLoop as _;
use g2g_core::{ConfigureOutcome, G2gError, PropValue};
use g2g_plugins::onvif::{discover, resolve_stream_uri, OnvifSrc};

#[test]
fn properties_round_trip() {
    let mut src = OnvifSrc::new("");
    src.set_property(
        "location",
        PropValue::Str("http://cam/onvif/device_service".into()),
    )
    .unwrap();
    src.set_property("user", PropValue::Str("admin".into())).unwrap();
    src.set_property("password", PropValue::Str("secret".into())).unwrap();

    assert_eq!(
        src.get_property("location"),
        Some(PropValue::Str("http://cam/onvif/device_service".into()))
    );
    assert_eq!(src.get_property("user"), Some(PropValue::Str("admin".into())));
    // Password is write-only.
    assert_eq!(src.get_property("password"), None);

    assert_eq!(
        src.set_property("nope", PropValue::Str("x".into())),
        Err(g2g_core::PropError::Unknown)
    );
}

#[tokio::test]
async fn unconfigured_source_errors_cleanly() {
    // No device URL set: negotiation must fail with NotConfigured rather than
    // attempting any I/O or panicking.
    let mut src = OnvifSrc::new("");
    let err = src.intercept_caps().await.unwrap_err();
    assert_eq!(err, G2gError::NotConfigured);
    // configure_pipeline before a successful resolve also reports it (no inner).
    let caps = g2g_core::Caps::CompressedVideo {
        codec: g2g_core::VideoCodec::H264,
        width: g2g_core::Dim::Fixed(1920),
        height: g2g_core::Dim::Fixed(1080),
        framerate: g2g_core::Rate::Any,
    };
    match src.configure_pipeline(&caps) {
        Err(G2gError::NotConfigured) => {}
        other => panic!("expected NotConfigured, got {other:?}"),
    }
    let _ = ConfigureOutcome::Accepted; // keep the import meaningful
}

#[tokio::test]
#[ignore = "needs ONVIF cameras answering WS-Discovery on the LAN"]
async fn discover_lan() {
    let devices = discover(Duration::from_secs(3)).await.expect("discover");
    assert!(
        !devices.is_empty(),
        "no ONVIF cameras answered; check the LAN / firewall"
    );
    for d in &devices {
        assert!(d.service_url.starts_with("http"), "bad XAddr: {d:?}");
        std::println!("found ONVIF device: {}", d.service_url);
    }
}

#[tokio::test]
#[ignore = "needs a real camera; set G2G_ONVIF_URL/USER/PASS"]
async fn resolve_uri() {
    let url = std::env::var("G2G_ONVIF_URL").expect("set G2G_ONVIF_URL");
    let user = std::env::var("G2G_ONVIF_USER").unwrap_or_default();
    let pass = std::env::var("G2G_ONVIF_PASS").unwrap_or_default();
    let uri = resolve_stream_uri(&url, &user, &pass)
        .await
        .expect("resolve_stream_uri");
    assert!(uri.starts_with("rtsp://"), "expected RTSP URL, got {uri}");
    std::println!("resolved RTSP URI: {uri}");
}
