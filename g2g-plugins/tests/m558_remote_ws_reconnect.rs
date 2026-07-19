//! M558: reconnection for the WebSocket transport, the browser-facing sibling of
//! the TCP reconnect (`m558_remote_reconnect`). A `RemoteWsSink::with_reconnect`
//! retries the WebSocket handshake until the `RemoteWsSrc` server is up.
#![cfg(feature = "remote-ws")]

use core::future::Future;
use core::pin::Pin;
use std::net::TcpListener as StdTcpListener;
use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline, LatencyProfile};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};

use g2g_plugins::remotewssink::RemoteWsSink;
use g2g_plugins::remotewssrc::RemoteWsSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

struct NullOut;
impl OutputSink for NullOut {
    fn push<'a>(
        &'a mut self,
        _p: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

#[derive(Default)]
struct CollectSink {
    caps: Vec<Caps>,
    seqs: Vec<u64>,
}
impl AsyncElement for CollectSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn configure_pipeline(&mut self, c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.caps.push(c.clone());
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => self.seqs.push(frame.sequence),
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                _ => {}
            }
            Ok(())
        })
    }
}

fn test_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn frame(seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(
            vec![seq as u8; 16].into_boxed_slice(),
        )),
        timing: FrameTiming {
            pts_ns: seq * 1_000_000,
            ..FrameTiming::default()
        },
        sequence: seq,
        meta: Default::default(),
    }
}

/// The WebSocket sink retries the handshake until the far side comes up.
#[tokio::test]
async fn ws_sink_retries_connect_until_server_is_up() {
    const N: u64 = 4;

    let port = {
        let l = StdTcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };

    let sender = async move {
        let url = format!("ws://127.0.0.1:{port}");
        let mut sink = RemoteWsSink::new(url).with_reconnect(200);
        sink.configure_pipeline(&test_caps()).expect("configure");
        let mut null = NullOut;
        for i in 0..N {
            sink.process(PipelinePacket::DataFrame(frame(i)), &mut null)
                .await
                .expect("frame delivered after reconnect");
        }
        let _ = sink.process(PipelinePacket::Eos, &mut null).await;
    };

    let receiver = async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        let listener = StdTcpListener::bind(("127.0.0.1", port)).expect("late bind");
        let mut src = RemoteWsSrc::from_listener(listener)
            .unwrap()
            .with_frame_limit(N);
        let mut sink = CollectSink::default();
        let clock = ZeroClock;
        let stats = run_simple_pipeline(
            &mut src,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        )
        .await
        .expect("receive ok");
        (stats.frames_emitted, sink)
    };

    let recv = tokio::time::timeout(Duration::from_secs(10), receiver);
    let (recv_res, ()) = tokio::join!(recv, sender);
    let (emitted, sink) = recv_res.expect("finishes within 10s");

    assert_eq!(
        emitted, N,
        "all frames crossed after the WS sink retried the handshake"
    );
    assert_eq!(
        sink.caps.first(),
        Some(&test_caps()),
        "caps discovered from the wire"
    );
    assert_eq!(
        sink.seqs,
        (0..N).collect::<Vec<_>>(),
        "every frame in order"
    );
}
