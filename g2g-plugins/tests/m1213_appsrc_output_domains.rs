#![cfg(all(unix, feature = "std"))]

use std::os::fd::IntoRawFd;

use g2g_core::memory::{DomainSet, MemoryDomain, MemoryDomainKind, OwnedDmaBuf};
use g2g_core::runtime::{parse_launch, run_graph, SourceLoop};
use g2g_core::{G2gError, PipelineClock, PropValue};
use g2g_plugins::appsink::{register_appsink_pull, Pull};
use g2g_plugins::appsrc::{register_appsrc, AppSrc};
use g2g_plugins::registry::default_registry;

const CAPS: &str = "video/x-raw,format=RGBA,width=2,height=2,framerate=30/1";
const RGBA_FRAME_BYTES: usize = 2 * 2 * 4;
const STRIDE: u32 = 8;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

// the sink only passes the descriptor on, so any closeable fd stands in for a dma-buf
fn placeholder_dmabuf() -> OwnedDmaBuf {
    let fd = std::fs::File::open("/dev/null")
        .expect("open /dev/null")
        .into_raw_fd();
    // SAFETY: `fd` is a fresh fd this test solely owns, the OwnedDmaBuf closes it once.
    unsafe { OwnedDmaBuf::from_raw(fd, STRIDE, 0) }
}

fn dmabuf_only_line(name: &str) -> String {
    format!(
        "appsrc channel={name}_in caps={CAPS} output-domains=dmabuf ! appsink input-domains=dmabuf channel={name}_out"
    )
}

#[test]
fn output_domains_property_sets_the_declared_domains() {
    let mut src = AppSrc::new();
    src.set_property("output-domains", PropValue::Str("system,dmabuf".into()))
        .expect("known domain names");
    assert_eq!(
        src.output_domains(),
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::DmaBuf)
    );
    assert_eq!(src.output_memory(), MemoryDomainKind::DmaBuf);
    assert_eq!(
        src.get_property("output-domains"),
        Some(PropValue::Str("dmabuf,system".into()))
    );
}

#[tokio::test]
async fn dmabuf_appsrc_negotiates_into_a_dmabuf_consumer() {
    let name = "m1213_negotiates";
    let feed = register_appsrc(&format!("{name}_in"));
    assert!(feed.push_dmabuf(placeholder_dmabuf(), 0));
    feed.end_of_stream();
    let pull = register_appsink_pull(&format!("{name}_out"));

    let graph = parse_launch(&default_registry(), &dmabuf_only_line(name)).expect("parses");
    run_graph(graph, &ZeroClock, 4).await.expect("runs");
    let Pull::Frame(frame) = pull.try_pull() else {
        panic!("no frame reached the appsink");
    };
    assert!(matches!(frame.domain, MemoryDomain::DmaBuf(_)));
}

#[tokio::test]
async fn system_push_into_a_dmabuf_appsrc_fails_the_run() {
    let name = "m1213_system_push";
    let feed = register_appsrc(&format!("{name}_in"));
    assert!(feed.push(&[0u8; RGBA_FRAME_BYTES], 0));
    feed.end_of_stream();
    let _pull = register_appsink_pull(&format!("{name}_out"));

    let graph = parse_launch(&default_registry(), &dmabuf_only_line(name)).expect("parses");
    let result = run_graph(graph, &ZeroClock, 4).await;
    assert!(
        matches!(result, Err(G2gError::UnsupportedDomain)),
        "{result:?}"
    );
}
