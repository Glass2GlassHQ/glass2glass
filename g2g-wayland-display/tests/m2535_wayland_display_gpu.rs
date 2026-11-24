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
use g2g_core::Graph;
use g2g_plugins::appsink::{register_appsink_pull, AppSink};
use g2g_wayland_display::waylanddisplaysrc::WaylandDisplaySrc;
use smithay::backend::allocator::dmabuf::{Dmabuf, DmabufFlags};
use smithay::backend::allocator::gbm::GbmDevice;
use smithay::backend::allocator::{Fourcc, Modifier};
use smithay::backend::drm::DrmDeviceFd;
use smithay::backend::egl::{EGLContext, EGLDisplay};
use smithay::backend::renderer::gles::GlesRenderer;
use smithay::backend::renderer::{ExportMem, ImportDma, TextureMapping};
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

async fn client_reaches_frames(render_node: &Path, sink: AppSink, expected: MemoryDomainKind) {
    let mut reader = DmaBufReader::open(render_node);
    let channel = format!("m2535_{}_{expected:?}", render_node.display());
    let source = WaylandDisplaySrc::new()
        .with_size(WIDTH, HEIGHT)
        .with_framerate(FRAMERATE)
        .with_render_node(render_node.to_string_lossy());
    let mut socket_names = source.socket_names();
    let pull = register_appsink_pull(&channel);
    let mut graph: Graph<GraphNode> = Graph::new();
    let source_node = graph.add_source(GraphNode::source(source));
    let sink_node = graph.add_sink(GraphNode::element(sink.with_channel(&channel)));
    graph.link(source_node, sink_node).expect("links");
    let checks = async {
        let mut events = connect_client(&mut socket_names).await;
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
                if frame_shows_client(&frame, expected, &mut reader) {
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

#[tokio::test]
async fn gpu_rendering_shares_a_dma_buf_with_a_sink_that_takes_one() {
    for render_node in render_nodes() {
        let sink = AppSink::new().with_input_domains(
            DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::DmaBuf),
        );
        client_reaches_frames(&render_node, sink, MemoryDomainKind::DmaBuf).await;
    }
}

#[tokio::test]
async fn gpu_rendering_copies_to_system_memory_for_a_system_sink() {
    for render_node in render_nodes() {
        let sink = AppSink::new().with_input_domains(DomainSet::only(MemoryDomainKind::System));
        client_reaches_frames(&render_node, sink, MemoryDomainKind::System).await;
    }
}
