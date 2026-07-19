//! M311: smoke tests for the ONVIF source element.
//!
//! The offline tests run always: property round-trip, the unconfigured-source
//! error path, and `resolve_uri_against_mock_server` (drives the full
//! GetCapabilities -> GetProfiles -> GetStreamUri sequence over real loopback
//! sockets against an in-process SOAP mock). The live tests are `#[ignore]` and
//! need a real camera on the LAN. Override the target via env vars:
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
    src.set_property("user", PropValue::Str("admin".into()))
        .unwrap();
    src.set_property("password", PropValue::Str("secret".into()))
        .unwrap();

    assert_eq!(
        src.get_property("location"),
        Some(PropValue::Str("http://cam/onvif/device_service".into()))
    );
    assert_eq!(
        src.get_property("user"),
        Some(PropValue::Str("admin".into()))
    );
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

/// Drive `resolve_stream_uri` against an in-process mock ONVIF device: a tiny
/// loopback HTTP server returns canned SOAP keyed on the request's action, so
/// the real reqwest client + WS-Security digest + response parsing all run over
/// real sockets through the full three-call sequence (GetCapabilities ->
/// GetProfiles -> GetStreamUri). No camera, no external tooling; runs in CI.
#[tokio::test]
async fn resolve_uri_against_mock_server() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let port = listener.local_addr().unwrap().port();
    // Every request body the mock received, so the test can assert the client
    // sent the right actions *and* an authenticated header.
    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let cap = captured.clone();
    std::thread::spawn(move || {
        // The GetCapabilities reply points the Media service back at this same
        // mock, so calls 2 and 3 land here too.
        let media_resp = std::format!(
            r#"<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
              <s:Body><tds:GetCapabilitiesResponse xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
                <tds:Capabilities xmlns:tt="http://www.onvif.org/ver10/schema">
                  <tt:Media><tt:XAddr>http://127.0.0.1:{port}/onvif/Media</tt:XAddr></tt:Media>
                </tds:Capabilities>
              </tds:GetCapabilitiesResponse></s:Body></s:Envelope>"#
        );
        let profiles_resp = r#"<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
          <s:Body><trt:GetProfilesResponse xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
            <trt:Profiles token="MainStream"/></trt:GetProfilesResponse></s:Body></s:Envelope>"#;
        let stream_resp = r#"<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
          <s:Body><trt:GetStreamUriResponse xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
            <trt:MediaUri xmlns:tt="http://www.onvif.org/ver10/schema">
              <tt:Uri>rtsp://192.168.1.50:554/Streaming/Channels/101</tt:Uri>
            </trt:MediaUri></trt:GetStreamUriResponse></s:Body></s:Envelope>"#;

        for stream in listener.incoming() {
            let mut stream = match stream {
                Ok(s) => s,
                Err(_) => break,
            };
            // Read headers + Content-Length-delimited body.
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            loop {
                let n = stream.read(&mut tmp).unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&buf[..pos]).to_lowercase();
                    let cl = headers
                        .split("content-length:")
                        .nth(1)
                        .and_then(|s| s.split("\r\n").next())
                        .and_then(|s| s.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    if buf.len() >= pos + 4 + cl {
                        break;
                    }
                }
            }
            let req = String::from_utf8_lossy(&buf).to_string();
            let body = if req.contains("GetCapabilities") {
                media_resp.clone()
            } else if req.contains("GetProfiles") {
                profiles_resp.to_string()
            } else {
                stream_resp.to_string()
            };
            cap.lock().unwrap().push(req);
            let resp = std::format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/soap+xml; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
        }
    });

    let device_url = std::format!("http://127.0.0.1:{port}/onvif/device_service");
    let uri = resolve_stream_uri(&device_url, "admin", "secret")
        .await
        .expect("resolve against mock");
    assert_eq!(uri, "rtsp://192.168.1.50:554/Streaming/Channels/101");

    let reqs = captured.lock().unwrap();
    assert_eq!(
        reqs.len(),
        3,
        "expected GetCapabilities, GetProfiles, GetStreamUri"
    );
    assert!(reqs.iter().any(|r| r.contains("GetCapabilities")));
    assert!(reqs.iter().any(|r| r.contains("GetProfiles")));
    assert!(reqs.iter().any(|r| r.contains("GetStreamUri")));
    // Every call carried a WS-Security UsernameToken digest (auth was sent).
    assert!(
        reqs.iter()
            .all(|r| r.contains("PasswordDigest") && r.contains("<Username>admin</Username>")),
        "each SOAP call must carry the WS-Security digest"
    );
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
