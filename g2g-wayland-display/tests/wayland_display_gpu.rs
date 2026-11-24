// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "display"))]

mod wayland_client;

use std::os::fd::{BorrowedFd, OwnedFd};
use std::path::{Path, PathBuf};

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomain, MemoryDomainKind, OwnedDmaBuf};
use g2g_core::runtime::{run_graph, GraphNode};
use g2g_core::{AsyncElement, Graph, PropValue};
use g2g_plugins::appsink::{register_appsink_pull, AppSink};
use g2g_plugins::dmabufmap::DmaBufReadMap;
use g2g_wayland_display::waylanddisplaysrc::WaylandDisplaySrc;
use smithay::backend::allocator::dmabuf::{Dmabuf, DmabufFlags};
use smithay::backend::allocator::gbm::GbmDevice;
use smithay::backend::allocator::{Fourcc, Modifier};
use smithay::backend::drm::DrmDeviceFd;
use smithay::backend::egl::{EGLContext, EGLDisplay};
use smithay::backend::renderer::gles::GlesRenderer;
use smithay::backend::renderer::{Bind, ExportMem, ImportDma, TextureMapping};
use smithay::utils::{DeviceFd, Rectangle};
use wayland_client::{
    client_globals, connect_client, shows_client, wait_for_client_event, ClientEvent, ZeroClock,
    BYTES_PER_PIXEL, FRAMERATE, HEIGHT, TIMEOUT, WIDTH,
};

const DRI_DIRECTORY: &str = "/dev/dri";
const RENDER_NODE_PREFIX: &str = "renderD";
const LINUX_DMABUF_INTERFACE: &str = "zwp_linux_dmabuf_v1";
// drm's abgr8888 is the RGBA the source outputs
const OUTPUT_FOURCC: Fourcc = Fourcc::Abgr8888;
const NV12_CAPS: &str = "video/x-raw,format=NV12";
// BT.709 limited range Y, Cb, Cr of the client's upper and lower colours
const UPPER_YCBCR: [u8; 3] = [112, 166, 83];
const LOWER_YCBCR: [u8; 3] = [91, 94, 186];
const YCBCR_TOLERANCE: u8 = 2;

fn render_nodes() -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(DRI_DIRECTORY) else {
        return Vec::new();
    };
    let mut nodes: Vec<PathBuf> = entries
        .map(|entry| entry.expect("lists /dev/dri").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(RENDER_NODE_PREFIX))
        })
        .collect();
    nodes.sort();
    nodes
}

struct DmaBufReader {
    gles: GlesRenderer,
}

impl DmaBufReader {
    fn open(render_node: &Path) -> Self {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(render_node)
            .expect("opens the render node");
        let gbm = GbmDevice::new(DrmDeviceFd::new(DeviceFd::from(OwnedFd::from(file))))
            .expect("opens gbm");
        // SAFETY: the gbm device was created above, so no other code holds its handle to get or terminate the same EGL display.
        let display = unsafe { EGLDisplay::new(gbm) }.expect("creates the EGL display");
        let context = EGLContext::new(&display).expect("creates the EGL context");
        // SAFETY: the context was created on this thread above and is current on no other thread.
        let gles = unsafe { GlesRenderer::new(context) }.expect("creates the GLES renderer");
        Self { gles }
    }

    fn renders_linear(&self, fourcc: Fourcc) -> bool {
        Bind::<Dmabuf>::supported_formats(&self.gles).is_some_and(|formats| {
            formats
                .iter()
                .any(|format| format.code == fourcc && format.modifier == Modifier::Linear)
        })
    }

    fn read(&mut self, dmabuf: &OwnedDmaBuf) -> Vec<u8> {
        // SAFETY: the fd is a live dma-buf owned by `dmabuf` for this call.
        let fd = unsafe { BorrowedFd::borrow_raw(dmabuf.as_raw()) }
            .try_clone_to_owned()
            .expect("duplicates the dma-buf");
        let size = (WIDTH as i32, HEIGHT as i32);
        let mut builder = Dmabuf::builder(
            size,
            OUTPUT_FOURCC,
            Modifier::from(dmabuf.modifier()),
            DmabufFlags::empty(),
        );
        builder.add_plane(fd, dmabuf.offset, dmabuf.stride);
        let imported = builder.build().expect("one plane");
        let texture = self
            .gles
            .import_dmabuf(&imported, None)
            .expect("imports the dma-buf");
        let mapping = self
            .gles
            .copy_texture(&texture, Rectangle::from_size(size.into()), OUTPUT_FOURCC)
            .expect("reads the texture");
        let bytes = self.gles.map_texture(&mapping).expect("maps the read back");
        let row_bytes = WIDTH as usize * BYTES_PER_PIXEL;
        // smithay calls a mapping flipped when its first row is the top one
        match mapping.flipped() {
            true => bytes.to_vec(),
            false => bytes
                .chunks_exact(row_bytes)
                .rev()
                .flatten()
                .copied()
                .collect(),
        }
    }
}

fn frame_shows_client(
    frame: &Frame,
    expected: MemoryDomainKind,
    reader: &mut DmaBufReader,
) -> bool {
    assert_eq!(frame.domain.kind(), expected, "the negotiated memory");
    match &frame.domain {
        MemoryDomain::DmaBuf(dmabuf) => {
            let picture = reader.read(dmabuf);
            shows_client(&picture, WIDTH as usize * BYTES_PER_PIXEL)
        }
        domain => {
            let picture = domain.as_system_slice().expect("system memory");
            assert_eq!(
                picture.len(),
                WIDTH as usize * HEIGHT as usize * BYTES_PER_PIXEL
            );
            shows_client(picture, WIDTH as usize * BYTES_PER_PIXEL)
        }
    }
}

fn ycbcr_at(planes: &[u8], stride: usize, x: u32, y: u32) -> [u8; 3] {
    let luma = planes[y as usize * stride + x as usize];
    let chroma = (HEIGHT as usize + y as usize / 2) * stride + x as usize / 2 * 2;
    [luma, planes[chroma], planes[chroma + 1]]
}

fn near(actual: [u8; 3], expected: [u8; 3]) -> bool {
    actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| actual.abs_diff(expected) <= YCBCR_TOLERANCE)
}

fn nv12_shows_client(planes: &[u8], stride: usize) -> bool {
    let x = WIDTH / 2;
    near(ycbcr_at(planes, stride, x, HEIGHT / 4), UPPER_YCBCR)
        && near(ycbcr_at(planes, stride, x, HEIGHT * 3 / 4), LOWER_YCBCR)
}

fn nv12_frame_shows_client(frame: &Frame, expected: MemoryDomainKind) -> bool {
    assert_eq!(frame.domain.kind(), expected, "the negotiated memory");
    match &frame.domain {
        MemoryDomain::DmaBuf(dmabuf) => {
            let mapping = DmaBufReadMap::read(dmabuf).expect("maps the linear planes");
            nv12_shows_client(mapping.as_slice(), dmabuf.stride as usize)
        }
        domain => {
            let planes = domain.as_system_slice().expect("system memory");
            assert_eq!(planes.len(), WIDTH as usize * HEIGHT as usize * 3 / 2);
            nv12_shows_client(planes, WIDTH as usize)
        }
    }
}

fn nv12_sink(domains: DomainSet) -> AppSink {
    let mut sink = AppSink::new().with_input_domains(domains);
    sink.set_property("caps", PropValue::Str(NV12_CAPS.into()))
        .expect("takes caps");
    sink
}

fn gpu_source(render_node: &Path) -> WaylandDisplaySrc {
    WaylandDisplaySrc::new()
        .with_size(WIDTH, HEIGHT)
        .with_framerate(FRAMERATE)
        .with_render_node(render_node.to_string_lossy())
}

async fn client_reaches_frames(
    render_node: &Path,
    sink: AppSink,
    channel: &str,
    mut shows_client: impl FnMut(&Frame) -> bool,
) {
    let source = gpu_source(render_node);
    let mut display_names = source.display_names();
    let pull = register_appsink_pull(channel);
    let mut graph: Graph<GraphNode> = Graph::new();
    let source_node = graph.add_source(GraphNode::source(source));
    let sink_node = graph.add_sink(GraphNode::element(sink.with_channel(channel)));
    graph.link(source_node, sink_node).expect("links");
    let checks = async {
        let mut events = connect_client(&mut display_names).await;
        let interfaces = client_globals(&mut events).await;
        assert!(
            interfaces.iter().any(|name| name == LINUX_DMABUF_INTERFACE),
            "{} offers {LINUX_DMABUF_INTERFACE}: {interfaces:?}",
            render_node.display()
        );
        wait_for_client_event(
            &mut events,
            &ClientEvent::Configured {
                width: WIDTH,
                height: HEIGHT,
            },
        )
        .await;
        let client_shown = tokio::time::timeout(TIMEOUT, async {
            loop {
                let frame = pull.pull().await.expect("the source keeps producing");
                if shows_client(&frame) {
                    return;
                }
            }
        })
        .await;
        assert!(
            client_shown.is_ok(),
            "a frame rendered on {} shows the client's window",
            render_node.display()
        );
    };
    tokio::select! {
        result = run_graph(graph, &ZeroClock, 2) => panic!("the graph ended first: {result:?}"),
        () = checks => {}
    }
}

fn system_or_dmabuf() -> DomainSet {
    DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::DmaBuf)
}

async fn rgba_client_reaches_frames(
    render_node: &Path,
    domains: DomainSet,
    expected: MemoryDomainKind,
) {
    let mut reader = DmaBufReader::open(render_node);
    let channel = format!("m2535_rgba_{}_{expected:?}", render_node.display());
    let sink = AppSink::new().with_input_domains(domains);
    client_reaches_frames(render_node, sink, &channel, |frame| {
        frame_shows_client(frame, expected, &mut reader)
    })
    .await;
}

#[tokio::test]
async fn gpu_rendering_shares_a_dma_buf_with_a_sink_that_takes_one() {
    for render_node in render_nodes() {
        rgba_client_reaches_frames(&render_node, system_or_dmabuf(), MemoryDomainKind::DmaBuf)
            .await;
    }
}

#[tokio::test]
async fn gpu_rendering_copies_to_system_memory_for_a_system_sink() {
    for render_node in render_nodes() {
        let domains = DomainSet::only(MemoryDomainKind::System);
        rgba_client_reaches_frames(&render_node, domains, MemoryDomainKind::System).await;
    }
}

#[tokio::test]
async fn gpu_rendering_writes_nv12_planes_for_an_nv12_sink() {
    for render_node in render_nodes() {
        let reader = DmaBufReader::open(&render_node);
        let renders_planes =
            reader.renders_linear(Fourcc::R8) && reader.renders_linear(Fourcc::Gr88);
        for (domains, expected) in [
            (system_or_dmabuf(), MemoryDomainKind::DmaBuf),
            (
                DomainSet::only(MemoryDomainKind::System),
                MemoryDomainKind::System,
            ),
        ] {
            let channel = format!("m2536_nv12_{}_{expected:?}", render_node.display());
            if renders_planes {
                client_reaches_frames(&render_node, nv12_sink(domains), &channel, |frame| {
                    nv12_frame_shows_client(frame, expected)
                })
                .await;
                continue;
            }
            let mut graph: Graph<GraphNode> = Graph::new();
            let source_node = graph.add_source(GraphNode::source(gpu_source(&render_node)));
            let sink_node = graph.add_sink(GraphNode::element(
                nv12_sink(domains).with_channel(&channel),
            ));
            graph.link(source_node, sink_node).expect("links");
            let result = tokio::time::timeout(TIMEOUT, run_graph(graph, &ZeroClock, 2))
                .await
                .expect("the run ends");
            assert!(
                result.is_err(),
                "{} renders no linear NV12 planes, so the run fails",
                render_node.display()
            );
        }
    }
}

#[cfg(feature = "ffmpeg")]
fn i420_ycbcr_at(planes: &[u8], x: u32, y: u32) -> [u8; 3] {
    let (width, height) = (WIDTH as usize, HEIGHT as usize);
    let chroma = (y as usize / 2) * (width / 2) + x as usize / 2;
    let cb_plane = width * height;
    let cr_plane = cb_plane + width / 2 * (height / 2);
    [
        planes[y as usize * width + x as usize],
        planes[cb_plane + chroma],
        planes[cr_plane + chroma],
    ]
}

#[cfg(feature = "ffmpeg")]
fn decoded_shows_client(frame: &Frame) -> bool {
    // h.264 at its default quantizer moves flat colours by a few levels
    const ENCODE_TOLERANCE: u8 = 8;
    let near = |actual: [u8; 3], expected: [u8; 3]| {
        actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.abs_diff(expected) <= ENCODE_TOLERANCE)
    };
    let planes = frame
        .domain
        .as_system_slice()
        .expect("decoded system memory");
    let x = WIDTH / 2;
    near(i420_ycbcr_at(planes, x, HEIGHT / 4), UPPER_YCBCR)
        && near(i420_ycbcr_at(planes, x, HEIGHT * 3 / 4), LOWER_YCBCR)
}

#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn vaapi_encodes_nv12_dma_bufs_from_the_display() {
    use g2g_plugins::ffmpegdec::FfmpegVideoDec;
    use g2g_plugins::ffmpegenc::{Backend, FfmpegVideoEnc, VaapiCodec};
    // waylanddisplaysrc repeats an undamaged picture once a second, so a few frames cover the client's first draw
    const ENCODED_FRAMES: u64 = 5;
    for render_node in render_nodes() {
        let reader = DmaBufReader::open(&render_node);
        if !(reader.renders_linear(Fourcc::R8) && reader.renders_linear(Fourcc::Gr88)) {
            continue;
        }
        let channel = format!("m2539_{}", render_node.display());
        let node_path = render_node.to_string_lossy();
        let source = gpu_source(&render_node).with_num_buffers(ENCODED_FRAMES);
        let mut display_names = source.display_names();
        let encoder = FfmpegVideoEnc::new()
            .with_backend(Backend::Vaapi(VaapiCodec::H264))
            .with_device_path(&node_path);
        let pull = register_appsink_pull(&channel);
        let mut graph: Graph<GraphNode> = Graph::new();
        let source_node = graph.add_source(GraphNode::source(source));
        let encoder_node = graph.add_transform(GraphNode::element(encoder));
        let decoder_node = graph.add_transform(GraphNode::element(FfmpegVideoDec::new()));
        let sink_node = graph.add_sink(GraphNode::element(AppSink::new().with_channel(&channel)));
        graph.link(source_node, encoder_node).expect("links");
        graph.link(encoder_node, decoder_node).expect("links");
        graph.link(decoder_node, sink_node).expect("links");
        let checks = async {
            let mut events = connect_client(&mut display_names).await;
            client_globals(&mut events).await;
            let mut client_shown = false;
            while let Some(frame) = pull.pull().await {
                client_shown |= decoded_shows_client(&frame);
            }
            client_shown
        };
        let (stats, client_shown) = tokio::join!(run_graph(graph, &ZeroClock, 2), checks);
        let stats = stats.expect("the encode runs");
        assert_eq!(
            stats.allocation.map(|allocation| allocation.domain),
            Some(MemoryDomainKind::DmaBuf),
            "the encoder takes the display's dma-bufs on {}",
            render_node.display()
        );
        assert!(
            client_shown,
            "a decoded frame encoded on {} shows the client's window",
            render_node.display()
        );
    }
}

#[cfg(feature = "cuda")]
#[tokio::test]
async fn nvenc_encodes_dma_bufs_from_the_display_through_cuda() {
    use g2g_plugins::dmabuftocuda::DmaBufToCuda;
    use g2g_plugins::ffmpegdec::FfmpegVideoDec;
    use g2g_plugins::nvenc::NvEnc;
    use g2g_wayland_display::waylanddisplaysrc::render_node_driver;
    const NVIDIA_DRIVER: &str = "nvidia";
    // waylanddisplaysrc repeats an undamaged picture once a second, so a few frames cover the client's first draw
    const ENCODED_FRAMES: u64 = 5;
    for render_node in render_nodes() {
        if render_node_driver(&render_node).ok().as_deref() != Some(NVIDIA_DRIVER) {
            continue;
        }
        let channel = format!("m2538_{}", render_node.display());
        let source = gpu_source(&render_node).with_num_buffers(ENCODED_FRAMES);
        let mut display_names = source.display_names();
        let pull = register_appsink_pull(&channel);
        let mut graph: Graph<GraphNode> = Graph::new();
        let source_node = graph.add_source(GraphNode::source(source));
        let import_node = graph.add_transform(GraphNode::element(DmaBufToCuda::new()));
        let encoder_node = graph.add_transform(GraphNode::element(NvEnc::new()));
        let decoder_node = graph.add_transform(GraphNode::element(FfmpegVideoDec::new()));
        let sink_node = graph.add_sink(GraphNode::element(AppSink::new().with_channel(&channel)));
        graph.link(source_node, import_node).expect("links");
        graph.link(import_node, encoder_node).expect("links");
        graph.link(encoder_node, decoder_node).expect("links");
        graph.link(decoder_node, sink_node).expect("links");
        let checks = async {
            let mut events = connect_client(&mut display_names).await;
            client_globals(&mut events).await;
            let mut client_shown = false;
            while let Some(frame) = pull.pull().await {
                client_shown |= decoded_shows_client(&frame);
            }
            client_shown
        };
        let (stats, client_shown) = tokio::join!(run_graph(graph, &ZeroClock, 2), checks);
        let stats = stats.expect("the encode runs");
        assert_eq!(
            stats.allocation.map(|allocation| allocation.domain),
            Some(MemoryDomainKind::DmaBuf),
            "dmabuftocuda takes the display's dma-bufs on {}",
            render_node.display()
        );
        assert!(
            client_shown,
            "a decoded frame encoded from {} shows the client's window",
            render_node.display()
        );
    }
}
