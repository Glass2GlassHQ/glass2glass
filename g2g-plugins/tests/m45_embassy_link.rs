#![cfg(feature = "embassy-link")]
//! M45: a real source streams packets through an embassy-sync zero-alloc
//! channel to a consumer, both driven by `embassy_futures::block_on`. Proves
//! the §6.2 stack-channel link carries the pipeline's `PipelinePacket`s between
//! two halves on an Embassy executor primitive, with the channel storage static
//! (no allocation).

use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, Dim, PipelinePacket, Rate, RawVideoFormat};
use g2g_plugins::embassylink::SinglePacketChannel;
use g2g_plugins::videotestsrc::VideoTestSrc;

#[test]
fn source_streams_through_embassy_sync_channel_to_consumer() {
    let channel: SinglePacketChannel<8> = SinglePacketChannel::new();
    let mut src = VideoTestSrc::new(16, 8, 30, 3);
    src.configure_pipeline(&Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(16),
        height: Dim::Fixed(8),
        framerate: Rate::Fixed(30 << 16),
    })
    .expect("configure");

    let mut sink = channel.sink();
    let rx = channel.receiver();

    let producer = src.run(&mut sink);
    let consumer = async {
        let mut frames = 0u32;
        loop {
            match rx.receive().await {
                PipelinePacket::DataFrame(_) => frames += 1,
                PipelinePacket::Eos => break,
                _ => {}
            }
        }
        frames
    };

    let (run, frames) = embassy_futures::block_on(embassy_futures::join::join(producer, consumer));
    run.expect("source run completes");
    assert_eq!(frames, 3, "all frames cross the embassy-sync channel");
}
