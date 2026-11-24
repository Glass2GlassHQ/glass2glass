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
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn client_sees_host_video_and_cursor_and_host_receives_client_input() {
    let settings = HostSettings {
        token: TOKEN.into(),
        bitrate: BITRATE,
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
    let capture = HostCapture {
        cursor_shapes: Some(cursor_shapes),
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
    let from_host = reader
        .read(&input.incoming.try_recv().expect("the host sent the cursor"))
        .expect("a valid host packet");
    let expected = CursorImage::new(
        CURSOR_SIDE as u16,
        CURSOR_SIDE as u16,
        1,
        2,
        cursor_shape.rgba,
    )
    .unwrap();
    assert_eq!(from_host, Some(Event::Cursor(expected)));
    assert_eq!(
        *sizes.borrow(),
        Some(ScreenSize {
            width: WIDTH,
            height: HEIGHT
        })
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_wrong_token_closes_the_connection_before_any_video() {
    let settings = HostSettings {
        token: TOKEN.into(),
        bitrate: BITRATE,
        send_window: None,
        video_mode: None,
    };
    let (listener, cert_hash) = host_listener(&settings);
    let source = format!("videotestsrc num-buffers={FRAMES} width={WIDTH} height={HEIGHT}");
    let (mut client, _input) = client_source(&listener, &cert_hash, "wrong token");
    let (event_sender, _host_events) = mpsc::unbounded_channel();
    let (size_sender, sizes) = watch::channel(None);
    let host = run_session(
        listener,
        capture(&source),
        &settings,
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
    assert!(matches!(host_result, Err(HostError::BadToken)));
    assert!(client_result.is_err());
    assert_eq!(frames.load(Ordering::Relaxed), 0);
    assert_eq!(*sizes.borrow(), None);
}
