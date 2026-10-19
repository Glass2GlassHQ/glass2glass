// runs a launched graph and counts the checksums each named `checksumsink` posts

use std::time::Duration;

use g2g_core::runtime::{run_graph_with_bus, GraphNode, LatencyProfile};
use g2g_core::{Bus, BusMessage, Graph};
use g2g_plugins::clock::WallClock;

const BUS_CAPACITY: usize = 64;

// stops once every sink has `enough` checksums, or when the graph ends on its own
pub(crate) fn count_checksums(
    graph: Graph<GraphNode>,
    sinks: &[&str],
    enough: usize,
    deadline: Duration,
) -> Vec<usize> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build a runtime");
    runtime.block_on(async {
        let (bus, handle) = Bus::new(BUS_CAPACITY);
        let clock = WallClock::new();
        let run = run_graph_with_bus(graph, &clock, LatencyProfile::Throughput, &handle);
        tokio::pin!(run);
        let mut counts = vec![0; sinks.len()];
        let collect = async {
            loop {
                tokio::select! {
                    result = &mut run => {
                        result.expect("the graph runs");
                        while let Some(message) = bus.try_recv() {
                            count(&message, sinks, &mut counts);
                        }
                        return;
                    }
                    Some(message) = bus.recv() => {
                        count(&message, sinks, &mut counts);
                        if counts.iter().all(|&n| n >= enough) {
                            return;
                        }
                    }
                }
            }
        };
        let _ = tokio::time::timeout(deadline, collect).await;
        counts
    })
}

fn count(message: &BusMessage, sinks: &[&str], counts: &mut [usize]) {
    let BusMessage::Checksum { element, .. } = message else {
        return;
    };
    if let Some(index) = sinks.iter().position(|sink| sink == element) {
        counts[index] += 1;
    }
}
