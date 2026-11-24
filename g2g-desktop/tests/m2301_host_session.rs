// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "session"))]

use core::future::Future;
use core::pin::Pin;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use g2g_core::runtime::{parse_element, parse_source, run_simple_pipeline, GraphNodeRef};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, OutputSink, PipelinePacket,
};
use g2g_desktop::host::{
    run_session, HostCapture, HostEncoder, HostError, HostListener, HostSettings,
};
use g2g_desktop::injection::ScreenSize;
use g2g_desktop::packet::{CursorImage, Event, PacketReader, PacketWriter, Side};
use g2g_plugins::clock::WallClock;
use g2g_plugins::cursorshape::{CursorShape, CURSOR_BYTES_PER_PIXEL};
use g2g_plugins::datachannel::DataChannelEnds;
use g2g_plugins::registry::default_registry;
use g2g_plugins::remotewtio::generate_certificate;
use g2g_plugins::remotewtsrc::RemoteWtSrc;
use tokio::sync::{mpsc, watch};

const TOKEN: &str = "correct horse battery staple";
const WIDTH: u16 = 320;
const HEIGHT: u16 = 240;
const FRAMES: usize = 90;
const BITRATE: u32 = 1_000_000;
const RUN_TIMEOUT: Duration = Duration::from_secs(30);
const LINK_CAPACITY: usize = 2;
const HID_A: u16 = 0x04;
const CURSOR_SIDE: u32 = 4;
const HOST_CLIPBOARD: &str = "copied on the host";
const CLIENT_CLIPBOARD: &str = "copied on the client";

struct CountingSink {
    frames: Arc<AtomicU64>,
}

impl AsyncElement for CountingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, caps: &Caps) -> Result<Caps, G2gError> {
        Ok(caps.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                self.frames.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        })
    }
}

fn host_listener(settings: &HostSettings) -> (HostListener, String) {
    let certificate = generate_certificate().unwrap();
    let cert_hash = certificate.sha256_hex.clone();
    let listener = HostListener::bind(
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        certificate,
        settings,
    )
    .unwrap();
    (listener, cert_hash)
}

fn client_source(
    listener: &HostListener,
    cert_hash: &str,
    token: &str,
) -> (RemoteWtSrc, DataChannelEnds) {
    let location = format!("quic://{}", listener.local_addr());
    let mut source = RemoteWtSrc::new(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)))
        .with_caller(location, cert_hash);
    let channel = source.open_data_channel();
    channel.outgoing.send(token.as_bytes().to_vec()).unwrap();
    (source, channel)
}

fn capture(registry_source: &str) -> HostCapture {
    let registry = default_registry();
    HostCapture {
        nodes: vec![
            GraphNodeRef::Source(parse_source(&registry, registry_source).unwrap()),
            GraphNodeRef::Element(parse_element(&registry, "clocksync").unwrap()),
        ],
        encoder: HostEncoder::Software,
        cursor_shapes: None,
        clipboard_texts: None,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn client_sees_host_video_cursor_and_clipboard_and_host_receives_client_input() {
    let settings = HostSettings {
        token: TOKEN.into(),
        bitrate: BITRATE,
        audio: false,
        send_window: None,
        video_mode: None,
    };
    let (listener, cert_hash) = host_listener(&settings);
    let source = format!("videotestsrc num-buffers={FRAMES} width={WIDTH} height={HEIGHT}");
    let cursor_shape = CursorShape {
        width: CURSOR_SIDE,
        height: CURSOR_SIDE,
        hotspot_x: 1,
        hotspot_y: 2,
        rgba: (0..CURSOR_SIDE * CURSOR_SIDE * CURSOR_BYTES_PER_PIXEL as u32)
            .map(|i| i as u8)
            .collect(),
    };
    let (cursor_sender, cursor_shapes) = watch::channel(None);
    cursor_sender.send_replace(Some(cursor_shape.clone()));
    let (clipboard_sender, clipboard_texts) = watch::channel(None);
    clipboard_sender.send_replace(Some(String::from(HOST_CLIPBOARD)));
    let capture = HostCapture {
        cursor_shapes: Some(cursor_shapes),
        clipboard_texts: Some(clipboard_texts),
        ..capture(&source)
    };
    let (client, mut input) = client_source(&listener, &cert_hash, TOKEN);
    let (event_sender, mut host_events) = mpsc::unbounded_channel();
    let (size_sender, sizes) = watch::channel(None);
    let host = run_session(
        listener,
        capture,
        &settings,
        event_sender,
        size_sender,
        None,
    );

    let sent = [
        Event::Key {
            device: 0,
            usage: HID_A,
            down: true,
        },
        Event::Position {
            x: i32::from(WIDTH) / 2,
            y: i32::from(HEIGHT) / 2,
            display: 0,
        },
        Event::Clipboard(String::from(CLIENT_CLIPBOARD)),
    ];
    let mut writer = PacketWriter::new(Side::Client);
    for event in &sent {
        input.outgoing.send(writer.write(event).unwrap()).unwrap();
    }
    let frames = Arc::new(AtomicU64::new(0));
    let mut sink = CountingSink {
        frames: frames.clone(),
    };
    let mut client = client;
    let clock = WallClock::new();
    let client_run = run_simple_pipeline(&mut client, &mut sink, &clock, LINK_CAPACITY);

    let (host_result, client_result) =
        tokio::time::timeout(RUN_TIMEOUT, async { tokio::join!(host, client_run) })
            .await
            .expect("the session ends after the host's capture ends");
    host_result.expect("host session");
    client_result.expect("client session");

    assert_eq!(frames.load(Ordering::Relaxed), FRAMES as u64);
    let mut delivered = Vec::new();
    while let Ok(event) = host_events.try_recv() {
        delivered.push(event);
    }
    assert_eq!(delivered, sent);
    let mut reader = PacketReader::new(Side::Client);
    let mut from_host = Vec::new();
    while let Ok(packet) = input.incoming.try_recv() {
        from_host.push(reader.read(&packet).expect("a valid host packet"));
    }
    let cursor = CursorImage::new(
        CURSOR_SIDE as u16,
        CURSOR_SIDE as u16,
        1,
        2,
        cursor_shape.rgba,
    )
    .unwrap();
    assert!(
        from_host.contains(&Some(Event::Cursor(cursor))),
        "the host sent the cursor: {from_host:?}"
    );
    assert!(
        from_host.contains(&Some(Event::Clipboard(String::from(HOST_CLIPBOARD)))),
        "the host sent its clipboard: {from_host:?}"
    );
    assert_eq!(
        *sizes.borrow(),
        Some(ScreenSize {
            width: WIDTH,
            height: HEIGHT
        })
    );
}

struct RejectedSession {
    host_result: Result<g2g_core::runtime::RunStats, HostError>,
    client_failed: bool,
    frames: u64,
    size: Option<ScreenSize>,
}

async fn run_with_wrong_token(
    listener: HostListener,
    cert_hash: &str,
    settings: &HostSettings,
) -> RejectedSession {
    let source = format!("videotestsrc num-buffers={FRAMES} width={WIDTH} height={HEIGHT}");
    let (mut client, _input) = client_source(&listener, cert_hash, "wrong token");
    let (event_sender, _host_events) = mpsc::unbounded_channel();
    let (size_sender, sizes) = watch::channel(None);
    let host = run_session(
        listener,
        capture(&source),
        settings,
        event_sender,
        size_sender,
        None,
    );
    let frames = Arc::new(AtomicU64::new(0));
    let mut sink = CountingSink {
        frames: frames.clone(),
    };
    let clock = WallClock::new();
    let client_run = run_simple_pipeline(&mut client, &mut sink, &clock, LINK_CAPACITY);

    let (host_result, client_result) =
        tokio::time::timeout(RUN_TIMEOUT, async { tokio::join!(host, client_run) })
            .await
            .expect("the host closes the connection");
    let size = *sizes.borrow();
    RejectedSession {
        host_result,
        client_failed: client_result.is_err(),
        frames: frames.load(Ordering::Relaxed),
        size,
    }
}

fn settings() -> HostSettings {
    HostSettings {
        token: TOKEN.into(),
        bitrate: BITRATE,
        audio: false,
        send_window: None,
        video_mode: None,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_wrong_token_closes_the_connection_before_any_video() {
    let settings = settings();
    let (listener, cert_hash) = host_listener(&settings);
    let session = run_with_wrong_token(listener, &cert_hash, &settings).await;
    assert!(matches!(session.host_result, Err(HostError::BadToken)));
    assert!(session.client_failed);
    assert_eq!(session.frames, 0);
    assert_eq!(session.size, None);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_host_listens_on_its_port_again_right_after_a_closed_connection() {
    let settings = settings();
    let (listener, cert_hash) = host_listener(&settings);
    let address = listener.local_addr();
    run_with_wrong_token(listener, &cert_hash, &settings).await;
    HostListener::bind_when_free(address, generate_certificate().unwrap(), &settings)
        .await
        .expect("the host binds its port again once the closed connection drains");
}
