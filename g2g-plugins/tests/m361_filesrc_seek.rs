//! M361 - `FileSrc` byte-offset seek. A `FileSrc` is a byte source, so a seek it
//! observes is in BYTES format: `start` is a file offset. On a flushing seek it
//! emits `Flush`, repositions the read, and resumes from there. This is the
//! source half of demuxer seeking (a demuxer resolves a time seek to a byte
//! offset and drives this controller).

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::{BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, PipelinePacket};
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{ByteStreamEncoding, Caps, G2gError, Seek};
use g2g_plugins::filesrc::FileSrc;

use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m361_{}_{}.bin", std::process::id(), name))
}

/// Records the bytes of each forwarded chunk, and whether a `Flush` was seen,
/// firing a one-shot byte-seek the moment the first chunk arrives (the source
/// awaits the push, so the seek is pending before the next chunk is read).
struct SeekOnFirst {
    ctl: SeekController,
    target: u64,
    armed: bool,
    before_flush: Vec<u8>,
    after_flush: Vec<u8>,
    flushed: bool,
}

impl OutputSink for SeekOnFirst {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(Frame {
                    domain: MemoryDomain::System(s),
                    ..
                }) => {
                    if self.flushed {
                        self.after_flush.extend_from_slice(s.as_slice());
                    } else {
                        self.before_flush.extend_from_slice(s.as_slice());
                    }
                    if self.armed {
                        self.ctl.seek(Seek::flush_to(self.target));
                        self.armed = false;
                    }
                }
                PipelinePacket::Flush => self.flushed = true,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[tokio::test]
async fn flushing_byte_seek_repositions_the_read() {
    let path = temp_path("seek");
    // 200 distinct bytes so the post-seek window is byte-identifiable.
    let data: Vec<u8> = (0..200u32).map(|i| (i % 256) as u8).collect();
    std::fs::write(&path, &data).unwrap();

    let seek_to = 120u64;
    let ctl = SeekController::new();
    let mut src = FileSrc::new(
        &path,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        },
    )
    .with_chunk_size(50)
    .with_seek(ctl.clone());

    let caps = {
        let c: Pin<Box<dyn Future<Output = _>>> = Box::pin(src.intercept_caps());
        c.await.expect("caps")
    };
    src.configure_pipeline(&caps).expect("configure");

    let mut sink = SeekOnFirst {
        ctl,
        target: seek_to,
        armed: true,
        before_flush: Vec::new(),
        after_flush: Vec::new(),
        flushed: false,
    };
    src.run(&mut sink).await.expect("run");

    assert!(sink.flushed, "the byte-seek flushed downstream");
    // First chunk (offset 0..50) arrived before the seek.
    assert_eq!(
        sink.before_flush,
        &data[0..50],
        "pre-seek bytes are the file head"
    );
    // After the flush, the read resumed at the requested byte offset and ran to
    // EOF: exactly the file tail from `seek_to`.
    assert_eq!(
        sink.after_flush,
        &data[seek_to as usize..],
        "post-seek bytes are the file tail from the byte offset"
    );
    let _ = std::fs::remove_file(&path);
}
