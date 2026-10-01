//! M466 `gstwrap`: host a real GStreamer element inside a g2g graph.
//!
//! Needs the host GStreamer runtime + dev libs and `gst-plugins-good` (for
//! `videoflip`); like the g2g-bridge smoke scripts, run it locally, not in CI:
//!
//! ```sh
//! cargo test -p g2g-plugins --features gstreamer --test gstwrap
//! ```
//!
//! The test drives the element directly (a crafted input frame + a capturing
//! output sink, the g2g graph boundary) rather than through `parse_launch`,
//! because the launch DSL v1 cannot carry a quoted property value with spaces
//! (`element="videoflip method=horizontal-flip"`). It asserts the pixels come
//! back horizontally flipped, which only a real GStreamer `videoflip` produces.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, IntoRawFd};
use std::os::unix::fs::{FileExt, MetadataExt};
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::log::{LogLevel, LogRecord, LogSink};
use g2g_core::memory::{MemoryDomain, MemoryDomainKind, OwnedDmaBuf, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, G2gError, HardwareError, OutputSink, PipelineClock, PipelinePacket, PropError,
    PropValue, PushOutcome,
};

use g2g_plugins::capsfilter::parse_caps;
use g2g_plugins::gstwrap::{GstWrap, OutputMemory};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Collects the bytes of every system `DataFrame` and every dma-buf the element
/// emits.
#[derive(Default)]
struct Collect {
    frames: Vec<Vec<u8>>,
    dmabufs: Vec<OwnedDmaBuf>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::DmaBuf(dmabuf) = &f.domain {
                    self.dmabufs.push(dmabuf.clone());
                } else if let Some(s) = f.domain.as_system_slice() {
                    self.frames.push(s.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[derive(Clone, Default)]
struct ErrorCategories(Arc<Mutex<Vec<String>>>);

impl LogSink for ErrorCategories {
    fn emit(&self, record: &LogRecord<'_>) {
        if record.level == LogLevel::Error {
            self.0.lock().unwrap().push(record.category.to_owned());
        }
    }
}

const RGBA_CAPS: &str = "video/x-raw,format=RGBA,width=2,height=2,framerate=1/1";
const RGBA_HEIGHT: u32 = 2;
// Two RGBA pixels take 8 bytes, the rest of each row is padding.
const PADDED_STRIDE: u32 = 16;
const PLANE_OFFSET: u32 = 64;

fn first_frame(domain: MemoryDomain) -> Frame {
    let timing = FrameTiming {
        pts_ns: 0,
        dts_ns: 0,
        ..FrameTiming::default()
    };
    Frame::new(domain, timing, 0)
}

fn memfd_holding(bytes: &[u8]) -> File {
    // SAFETY: the name is NUL-terminated; memfd_create returns a new fd or -1.
    let fd = unsafe { libc::memfd_create(c"g2g-gstwrap-test".as_ptr(), libc::MFD_CLOEXEC) };
    assert!(fd >= 0, "memfd_create failed");
    // SAFETY: `fd` is a fresh descriptor nothing else owns.
    let mut file = unsafe { File::from_raw_fd(fd) };
    file.write_all(bytes).expect("fill the memfd");
    file
}

fn file_behind(dmabuf: &OwnedDmaBuf) -> File {
    // SAFETY: the OwnedDmaBuf keeps the fd open for this borrow.
    let borrowed = unsafe { BorrowedFd::borrow_raw(dmabuf.as_raw()) };
    File::from(borrowed.try_clone_to_owned().expect("dup the dma-buf fd"))
}

fn dmabuf_wrap() -> GstWrap {
    let mut el = GstWrap::new().with_output_memory(OutputMemory::DmaBuf);
    el.set_property("element", PropValue::Str("identity".into()))
        .expect("element property");
    el.configure_pipeline(&parse_caps(RGBA_CAPS).expect("caps parse"))
        .expect("gst pipeline builds (needs host GStreamer)");
    el
}

#[tokio::test]
async fn passes_a_dmabuf_through_without_copying() {
    let frame_end = PLANE_OFFSET + PADDED_STRIDE * RGBA_HEIGHT;
    let bytes: Vec<u8> = (0..frame_end).map(|i| i as u8).collect();
    let memfd = memfd_holding(&bytes);
    let input_inode = memfd.metadata().expect("fstat memfd").ino();
    // SAFETY: the memfd's fd moves into the OwnedDmaBuf, which closes it once.
    let input = unsafe { OwnedDmaBuf::from_raw(memfd.into_raw_fd(), PADDED_STRIDE, PLANE_OFFSET) };

    let mut el = dmabuf_wrap();
    let mut sink = Collect::default();
    el.process(
        PipelinePacket::DataFrame(first_frame(MemoryDomain::DmaBuf(input.clone()))),
        &mut sink,
    )
    .await
    .expect("process frame");
    el.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("process eos");

    assert!(
        sink.frames.is_empty(),
        "no frame was copied to system memory"
    );
    assert_eq!(sink.dmabufs.len(), 1, "identity produced one dma-buf frame");
    let output = &sink.dmabufs[0];
    assert_eq!(output.stride, PADDED_STRIDE);
    assert_eq!(output.offset, PLANE_OFFSET);
    let output_file = file_behind(output);
    assert_eq!(
        output_file.metadata().expect("fstat output").ino(),
        input_inode,
        "the output frame is the input buffer"
    );
    let mut output_bytes = vec![0u8; bytes.len()];
    output_file
        .read_exact_at(&mut output_bytes, 0)
        .expect("read the output buffer");
    assert_eq!(output_bytes, bytes);

    assert!(
        input.share_count() > 1,
        "GStreamer still holds the input while the output frame is in use"
    );
    drop(el);
    assert_eq!(
        input.share_count(),
        1,
        "tearing down the pipeline released the input"
    );
}

// videoflip maps the dma-buf, so this checks the GstVideoMeta stride and offset.
#[tokio::test]
async fn maps_a_padded_dmabuf_into_a_system_element() {
    const PIXEL_BYTES: usize = 4;
    let pixels: [[u8; PIXEL_BYTES]; 4] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ];
    let mut bytes = vec![0xEE_u8; (PLANE_OFFSET + PADDED_STRIDE * RGBA_HEIGHT) as usize];
    for (index, pixel) in pixels.iter().enumerate() {
        let row = index / 2;
        let column = index % 2;
        let start = PLANE_OFFSET as usize + row * PADDED_STRIDE as usize + column * PIXEL_BYTES;
        bytes[start..start + PIXEL_BYTES].copy_from_slice(pixel);
    }
    let memfd = memfd_holding(&bytes);
    // SAFETY: the memfd's fd moves into the OwnedDmaBuf, which closes it once.
    let input = unsafe { OwnedDmaBuf::from_raw(memfd.into_raw_fd(), PADDED_STRIDE, PLANE_OFFSET) };

    let mut el = GstWrap::new();
    el.set_property(
        "element",
        PropValue::Str("videoflip method=horizontal-flip".into()),
    )
    .expect("element property");
    el.configure_pipeline(&parse_caps(RGBA_CAPS).expect("caps parse"))
        .expect("gst pipeline builds (needs host GStreamer + gst-plugins-good videoflip)");
    let mut sink = Collect::default();
    el.process(
        PipelinePacket::DataFrame(first_frame(MemoryDomain::DmaBuf(input))),
        &mut sink,
    )
    .await
    .expect("process frame");
    el.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("process eos");

    let flipped: Vec<u8> = [pixels[1], pixels[0], pixels[3], pixels[2]].concat();
    assert_eq!(sink.frames, vec![flipped], "each row came back mirrored");
}

#[tokio::test]
async fn dmabuf_output_rejects_a_system_sample() {
    let mut el = dmabuf_wrap();
    let bytes = vec![0u8; (PADDED_STRIDE * RGBA_HEIGHT) as usize];
    let frame = first_frame(MemoryDomain::System(SystemSlice::from_boxed(
        bytes.into_boxed_slice(),
    )));
    let mut sink = Collect::default();
    let result = match el
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
    {
        Ok(()) => el.process(PipelinePacket::Eos, &mut sink).await,
        error => error,
    };
    assert!(
        matches!(result, Err(G2gError::UnsupportedDomain)),
        "a system sample failed loud: {result:?}"
    );
    assert!(sink.frames.is_empty() && sink.dmabufs.is_empty());
}

// A write-only fd cannot be mapped for reading, so the system copy-out fails.
#[tokio::test]
async fn an_unmappable_sample_fails_the_stream() {
    let bytes = vec![0u8; (PLANE_OFFSET + PADDED_STRIDE * RGBA_HEIGHT) as usize];
    let memfd = memfd_holding(&bytes);
    let write_only = OpenOptions::new()
        .write(true)
        .open(format!("/proc/self/fd/{}", memfd.as_raw_fd()))
        .expect("reopen the memfd write-only");
    // SAFETY: the write-only fd moves into the OwnedDmaBuf, which closes it once.
    let input =
        unsafe { OwnedDmaBuf::from_raw(write_only.into_raw_fd(), PADDED_STRIDE, PLANE_OFFSET) };

    let mut el = GstWrap::new();
    el.set_property("element", PropValue::Str("identity".into()))
        .expect("element property");
    el.configure_pipeline(&parse_caps(RGBA_CAPS).expect("caps parse"))
        .expect("gst pipeline builds (needs host GStreamer)");
    let error_categories = ErrorCategories::default();
    let log_sink_id = g2g_core::log::add_sink(Box::new(error_categories.clone()));
    let mut sink = Collect::default();
    let result = match el
        .process(
            PipelinePacket::DataFrame(first_frame(MemoryDomain::DmaBuf(input))),
            &mut sink,
        )
        .await
    {
        Ok(()) => el.process(PipelinePacket::Eos, &mut sink).await,
        error => error,
    };
    g2g_core::log::remove_sink(log_sink_id);
    assert!(
        matches!(result, Err(G2gError::Hardware(HardwareError::Other))),
        "the unmappable sample failed loud: {result:?}"
    );
    assert!(sink.frames.is_empty() && sink.dmabufs.is_empty());
    assert!(
        error_categories
            .0
            .lock()
            .unwrap()
            .iter()
            .any(|category| category == el.log_category()),
        "the failure was logged under the element's category"
    );
}

#[test]
fn output_memory_is_a_launch_property() {
    let mut el = GstWrap::new();
    let spec = el
        .properties()
        .iter()
        .find(|spec| spec.name == "output-memory")
        .expect("output-memory is declared");
    let default = spec
        .parse_value(spec.default.expect("declares a default"))
        .expect("default parses");
    assert_eq!(el.get_property("output-memory"), Some(default));
    assert_eq!(el.output_memory(), MemoryDomainKind::System);
    assert!(!el.input_domains().contains(MemoryDomainKind::DmaBuf));

    let dmabuf = spec.parse_value("dmabuf").expect("dmabuf parses");
    el.set_property("output-memory", dmabuf.clone())
        .expect("dmabuf is accepted");
    assert_eq!(el.get_property("output-memory"), Some(dmabuf));
    assert_eq!(el.output_memory(), MemoryDomainKind::DmaBuf);
    assert!(el.input_domains().contains(MemoryDomainKind::DmaBuf));

    assert!(matches!(
        el.set_property("output-memory", PropValue::Str("gpu".into())),
        Err(PropError::Value)
    ));
}

#[tokio::test]
async fn hosts_a_real_gstreamer_videoflip() {
    // A 2x1 RGBA frame: left pixel opaque white, right pixel a distinct colour.
    let caps =
        parse_caps("video/x-raw,format=RGBA,width=2,height=1,framerate=1/1").expect("caps parse");

    let mut el = GstWrap::new();
    el.set_property(
        "element",
        PropValue::Str("videoflip method=horizontal-flip".into()),
    )
    .expect("element property");
    el.configure_pipeline(&caps)
        .expect("gst pipeline builds (needs host GStreamer + gst-plugins-good videoflip)");

    let input: Vec<u8> = vec![0xFF, 0xFF, 0xFF, 0xFF, 0x10, 0x20, 0x30, 0x40];
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(input.into_boxed_slice())),
        FrameTiming {
            pts_ns: 0,
            dts_ns: 0,
            ..FrameTiming::default()
        },
        0,
    );

    let mut sink = Collect::default();
    el.process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .expect("process frame");
    // EOS flushes videoflip's buffered frame; drain collects it.
    el.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("process eos");

    assert_eq!(
        sink.frames.len(),
        1,
        "the hosted GStreamer element produced one frame"
    );
    // Horizontal flip of a 2x1 image swaps the two pixels.
    assert_eq!(
        sink.frames[0],
        vec![0x10, 0x20, 0x30, 0x40, 0xFF, 0xFF, 0xFF, 0xFF],
        "pixels came back horizontally flipped by the real GStreamer videoflip"
    );
}

/// The quote-aware launch tokenizer carries a multi-word element description into
/// `gstwrap` from a gst-launch line, so a hosted GStreamer element runs straight
/// from `g2g-launch` (not only via the programmatic API).
#[tokio::test]
async fn runs_from_a_launch_line_with_a_spaced_property() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=3 ! gstwrap element=\"videoflip method=horizontal-flip\" ! fakesink",
    )
    .expect("quoted gstwrap line parses and builds");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("pipeline runs");
    assert_eq!(
        stats.frames_consumed, 3,
        "all frames flowed through the hosted GStreamer element"
    );
}
