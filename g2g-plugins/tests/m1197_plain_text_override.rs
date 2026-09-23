//! M1197: `filesrc bytestream-format=text` types a file as plain UTF-8 text
//! whatever its extension, so a prompt kept as `.md` feeds a text element.

#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

const PROMPT: &[u8] = b"# Prompt\n\nTell me a joke about a cat.\n";
const LINK_CAPACITY: usize = 4;

fn text_pipeline(path: &std::path::Path, filesrc_properties: &str) -> String {
    format!(
        "filesrc location={} {filesrc_properties} ! capsfilter caps=text/x-raw,format=utf8 ! fakesink",
        path.display()
    )
}

#[tokio::test]
async fn filesrc_runs_a_markdown_file_as_plain_text() {
    let path = std::env::temp_dir().join(format!("g2g-m1197-{}-prompt.md", std::process::id()));
    std::fs::write(&path, PROMPT).expect("write temp");
    let registry = default_registry();

    // Prose sniffs as nothing, so without the override the `.md` file has no type.
    let untyped = text_pipeline(&path, "");
    let untyped_fails = match parse_launch(&registry, &untyped) {
        Ok(graph) => run_graph(graph, &ZeroClock, LINK_CAPACITY).await.is_err(),
        Err(_) => true,
    };

    let line = text_pipeline(&path, "bytestream-format=text");
    let graph = parse_launch(&registry, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let consumed = run_graph(graph, &ZeroClock, LINK_CAPACITY)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"))
        .frames_consumed;
    std::fs::remove_file(&path).ok();

    assert!(
        untyped_fails,
        "an unsniffable .md file should not negotiate as text on its own"
    );
    assert!(consumed >= 1, "the text chunk reached the sink: {consumed}");
}
