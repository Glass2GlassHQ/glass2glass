//! M1190 - a native graph that serves its stream to whoever dials in.
//!
//! `RemoteWsSink` dials a `RemoteWsSrc`, which is the wrong direction for a
//! browser peer: a browser can only dial out, so a native graph feeding one has
//! to be the listening side. `listen=true` makes the sink wait for a client and
//! push the same wire stream down the accepted socket, unsolicited.
//!
//! The client here is a plain `tokio-tungstenite` connection decoding the wire
//! codec, which is exactly what a browser `WsWireSrc` does: the assertion is
//! that the caps and every frame arrive in order over a connection the receiver
//! opened.
#![cfg(feature = "remote-ws")]

use futures_util::StreamExt;
use tokio_tungstenite::tungstenite::Message;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::wire::decode_packet;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PropValue, PushOutcome, Rate, RawVideoFormat,
};

use g2g_plugins::remotewssink::RemoteWsSink;

const FRAMES: u8 = 5;
const FRAME_LEN: usize = 4 * 4 * 4;

struct NullOut;
impl OutputSink for NullOut {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn test_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(4),
        height: Dim::Fixed(4),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn frame(index: u8) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(
            vec![index; FRAME_LEN].into_boxed_slice(),
        )),
        timing: FrameTiming {
            pts_ns: u64::from(index) * 1_000_000,
            dts_ns: u64::from(index) * 1_000_000,
            duration_ns: 33_000,
            keyframe: index == 0,
            ..FrameTiming::default()
        },
        sequence: u64::from(index),
        meta: Default::default(),
    }
}

/// A free port to serve on: bound, read back, and dropped so the sink can bind
/// it itself (the listen side owns its socket).
fn free_port() -> u16 {
    let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("bind probe");
    probe.local_addr().expect("addr").port()
}

#[tokio::test]
async fn a_listening_sink_pushes_its_stream_to_a_client_that_dials_in() {
    let port = free_port();
    let url = format!("ws://127.0.0.1:{port}");

    // The graph side: serve on the port, then push caps and frames.
    let serve = async {
        let mut sink = RemoteWsSink::new(url.clone());
        sink.set_property("listen", PropValue::Bool(true))
            .expect("listen is settable");
        sink.configure_pipeline(&test_caps()).expect("configure");
        let mut null = NullOut;
        for i in 0..FRAMES {
            sink.process(PipelinePacket::DataFrame(frame(i)), &mut null)
                .await
                .expect("frame sent");
        }
        sink.process(PipelinePacket::Eos, &mut null)
            .await
            .expect("eos sent");
        sink.sent()
    };

    // The receiving side: dial in and read the unsolicited stream, as a browser
    // client does.
    let receive = async {
        // Retry while the sink is still binding: it owns the socket, so the
        // first dial can beat the listen.
        let mut client = loop {
            match tokio_tungstenite::connect_async(&url).await {
                Ok((socket, _)) => break socket,
                Err(_) => tokio::time::sleep(std::time::Duration::from_millis(20)).await,
            }
        };
        let mut caps = None;
        let mut frames = Vec::new();
        while let Some(Ok(message)) = client.next().await {
            let Message::Binary(bytes) = message else {
                continue;
            };
            match decode_packet(&bytes).expect("wire packet") {
                PipelinePacket::CapsChanged(c) => caps = Some(c),
                PipelinePacket::DataFrame(f) => {
                    let byte = f.domain.as_system_slice().map(|s| s[0]).unwrap_or(0);
                    frames.push((f.sequence, byte));
                }
                PipelinePacket::Eos => break,
                _ => {}
            }
        }
        (caps, frames)
    };

    let (sent, (caps, frames)) = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        tokio::join!(serve, receive)
    })
    .await
    .expect("finishes within 10s");

    assert_eq!(
        caps,
        Some(test_caps()),
        "the client learned the caps off the wire, unsolicited"
    );
    assert_eq!(frames.len(), FRAMES as usize, "every frame arrived");
    for (i, (sequence, byte)) in frames.iter().enumerate() {
        assert_eq!(*sequence, i as u64, "order preserved");
        assert_eq!(*byte, i as u8, "the frame's bytes are its own");
    }
    assert!(
        sent > u64::from(FRAMES),
        "the sink sent caps + {FRAMES} frames: {sent}"
    );
}

/// `listen` is a runtime property, so a launch line picks the role.
#[test]
fn listen_is_a_runtime_property() {
    let mut sink = RemoteWsSink::new("ws://127.0.0.1:1");
    assert_eq!(sink.get_property("listen"), Some(PropValue::Bool(false)));
    sink.set_property("listen", PropValue::Bool(true))
        .expect("listen is settable");
    assert_eq!(sink.get_property("listen"), Some(PropValue::Bool(true)));
}
