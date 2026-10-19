#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph_with_bus};
use g2g_core::{Bus, BusMessage, PipelineClock};
use g2g_plugins::registry::default_registry;

const BUFFERS: usize = 5;
// smaller than BUFFERS, so a sink that dropped lines on a full bus loses some
const BUS_CAPACITY: usize = 2;
const SINK_NAME: &str = "video";

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn launched_checksumsink_posts_every_buffer_under_its_name() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        &format!("videotestsrc num-buffers={BUFFERS} ! checksumsink name={SINK_NAME}"),
    )
    .expect("checksum pipeline parses");
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let run = async move {
        let result = run_graph_with_bus(graph, &ZeroClock, 4, &handle).await;
        drop(handle);
        result
    };
    let collect = async {
        let mut checksums = Vec::new();
        while let Some(message) = bus.recv().await {
            if let BusMessage::Checksum {
                element,
                pts_ns,
                digest,
            } = message
            {
                checksums.push((element, pts_ns, digest));
            }
        }
        checksums
    };
    let (result, checksums) = tokio::join!(run, collect);
    result.expect("checksum pipeline runs");

    assert_eq!(
        checksums.len(),
        BUFFERS,
        "one line per buffer, none dropped"
    );
    assert!(checksums.iter().all(|(element, _, _)| element == SINK_NAME));
    assert!(
        checksums.windows(2).all(|pair| pair[0].1 < pair[1].1),
        "pts rise frame by frame: {checksums:?}"
    );
    assert_ne!(
        checksums[0].2, checksums[1].2,
        "each frame of the moving test pattern has its own digest"
    );
}
