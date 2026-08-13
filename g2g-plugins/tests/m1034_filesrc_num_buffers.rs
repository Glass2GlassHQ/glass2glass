//! M1034: `FileSrc` honours `num-buffers`, the GStreamer `filesrc` property that
//! caps how many chunks a source emits before EOS.

use std::path::PathBuf;

use g2g_core::element::PushOutcome;
use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, G2gError, OutputSink, PipelinePacket, PropValue};
use g2g_plugins::filesrc::FileSrc;

const CHUNK_SIZE: usize = 8;
const FIXTURE_LEN: usize = CHUNK_SIZE * 10;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m1034_{}_{}.bin", std::process::id(), name))
}

fn ts_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    }
}

/// Collects every packet a directly-driven source pushes.
#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl Collect {
    fn data_frames(&self) -> usize {
        self.packets
            .iter()
            .filter(|p| matches!(p, PipelinePacket::DataFrame(_)))
            .count()
    }

    fn eos_seen(&self) -> bool {
        self.packets
            .iter()
            .any(|p| matches!(p, PipelinePacket::Eos))
    }
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Writes a fixture of `FIXTURE_LEN` bytes and runs a `FileSrc` over it,
/// applying `num_buffers` first when the caller asked for a limit.
async fn run_over_fixture(name: &str, num_buffers: Option<i64>) -> Collect {
    let path = temp_path(name);
    let payload: Vec<u8> = (0..FIXTURE_LEN).map(|i| i as u8).collect();
    std::fs::write(&path, &payload).expect("write fixture");

    let mut src = FileSrc::new(&path, ts_caps()).with_chunk_size(CHUNK_SIZE);
    if let Some(n) = num_buffers {
        src.set_property("num-buffers", PropValue::Int(n))
            .expect("num-buffers is settable");
    }
    let caps = src.intercept_caps().await.expect("declared caps");
    src.configure_pipeline(&caps).expect("configure");

    let mut out = Collect::default();
    src.run(&mut out).await.expect("run to EOS");
    let _ = std::fs::remove_file(&path);
    out
}

#[test]
fn num_buffers_round_trips() {
    let mut src = FileSrc::new(temp_path("props"), ts_caps());
    assert_eq!(
        src.get_property("num-buffers"),
        Some(PropValue::Int(-1)),
        "a fresh FileSrc is unlimited"
    );

    src.set_property("num-buffers", PropValue::Int(4)).unwrap();
    assert_eq!(src.get_property("num-buffers"), Some(PropValue::Int(4)));

    src.set_property("num-buffers", PropValue::Int(-1)).unwrap();
    assert_eq!(src.get_property("num-buffers"), Some(PropValue::Int(-1)));
}

#[test]
fn untyped_filesrc_defaults_to_unlimited() {
    let src = FileSrc::untyped();
    assert_eq!(src.get_property("num-buffers"), Some(PropValue::Int(-1)));
}

#[tokio::test]
async fn num_buffers_caps_emitted_chunks() {
    let out = run_over_fixture("limited", Some(3)).await;
    assert_eq!(out.data_frames(), 3, "num-buffers bounds the chunk count");
    assert!(out.eos_seen(), "the limit must still end with Eos");
}

#[tokio::test]
async fn num_buffers_zero_emits_only_eos() {
    let out = run_over_fixture("zero", Some(0)).await;
    assert_eq!(out.data_frames(), 0);
    assert!(out.eos_seen());
    assert_eq!(out.packets.len(), 1, "Eos is the only packet");
}

#[tokio::test]
async fn unlimited_reads_the_whole_file() {
    let out = run_over_fixture("unlimited", None).await;
    assert_eq!(out.data_frames(), FIXTURE_LEN / CHUNK_SIZE);
    assert!(out.eos_seen());
}
