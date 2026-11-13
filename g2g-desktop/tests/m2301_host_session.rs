// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "session"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{
    parse_element, parse_source, run_duplex_session, DynSourceLoop, GraphNodeRef,
};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, OutputSink, PipelinePacket,
};
use g2g_desktop::client::exchange_sdp;
use g2g_desktop::host::{run_session, HostSettings, INPUT_LABEL};
use g2g_desktop::packet::{Event, PacketWriter, Side};
use g2g_desktop::uinput::ScreenSize;
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;
use g2g_plugins::webrtcduplex::{SdpChannel, SignalRole, WebRtcDuplexSession};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, watch};

const TOKEN: &str = "correct horse battery staple";
const WIDTH: u16 = 320;
const HEIGHT: u16 = 240;
const FRAMES: usize = 90;
// the handshake drops the first frames
const MIN_RECEIVED: u64 = FRAMES as u64 / 3;
const BITRATE: u32 = 1_000_000;
const RUN_TIMEOUT: Duration = Duration::from_secs(30);
const SDP_CHANNEL_DEPTH: usize = 4;
const LINK_CAPACITY: usize = 2;
const HID_A: u16 = 0x04;

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

#[tokio::test(flavor = "multi_thread")]
async fn client_sees_host_video_and_host_receives_client_input() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let settings = HostSettings {
        token: TOKEN.into(),
        bitrate: BITRATE,
        stun_server: None,
        turn_server: None,
    };
    let registry = default_registry();
    let source = format!("videotestsrc num-buffers={FRAMES} width={WIDTH} height={HEIGHT}");
    let capture = vec![
        GraphNodeRef::Source(parse_source(&registry, &source).unwrap()),
        GraphNodeRef::Element(parse_element(&registry, "clocksync").unwrap()),
    ];
    let (event_sender, mut host_events) = mpsc::unbounded_channel();
    let (size_sender, sizes) = watch::channel(None);
    let host = async {
        let (stream, _) = listener.accept().await.unwrap();
        run_session(stream, capture, &settings, event_sender, size_sender).await
    };

    let (offer_sender, offers) = mpsc::channel(SDP_CHANNEL_DEPTH);
    let (answers, answer_receiver) = mpsc::channel(SDP_CHANNEL_DEPTH);
    let mut client = WebRtcDuplexSession::new(
        SignalRole::Offerer,
        SdpChannel::from_halves(offer_sender, answer_receiver),
        1,
    )
    .with_receive_only();
    let input = client.open_data_channel(INPUT_LABEL);
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
    let clock = WallClock::new();
    let client_run = async {
        let stream = TcpStream::connect(address).await.unwrap();
        let sources: Vec<&mut dyn DynSourceLoop> = Vec::new();
        let sinks: Vec<&mut dyn DynAsyncElement> = vec![&mut sink];
        tokio::join!(
            exchange_sdp(stream, TOKEN, offers, answers),
            run_duplex_session(sources, &mut client, sinks, &clock, LINK_CAPACITY)
        )
    };

    let (host_result, (exchanged, client_result)) =
        tokio::time::timeout(RUN_TIMEOUT, async { tokio::join!(host, client_run) })
            .await
            .expect("the session ends after the host's capture ends");
    host_result.expect("host session");
    exchanged.expect("SDP exchange");
    client_result.expect("client session");

    let received = frames.load(Ordering::Relaxed);
    assert!(
        received >= MIN_RECEIVED,
        "client: {received} of {FRAMES} frames"
    );
    let mut delivered = Vec::new();
    while let Ok(event) = host_events.try_recv() {
        delivered.push(event);
    }
    assert_eq!(delivered, sent);
    assert_eq!(
        *sizes.borrow(),
        Some(ScreenSize {
            width: WIDTH,
            height: HEIGHT
        })
    );
}
