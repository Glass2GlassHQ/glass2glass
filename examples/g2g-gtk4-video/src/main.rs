use std::thread::{self, JoinHandle};

use g2g_core::clock::MonotonicClock;
use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::{parse_launch, run_graph, ParseError};
use g2g_core::{Caps, Dim, G2gError, RawVideoFormat};
use g2g_plugins::appsink::{register_appsink_pull, AppSinkPull, HostFrame};
use g2g_plugins::registry::default_registry;
use gtk4::prelude::*;
use gtk4::{gdk, gio, glib};

const APPSINK_CHANNEL: &str = "g2g-gtk4-video";
const LINK_CAPACITY: usize = 2;
const DEFAULT_PIPELINE: &str = "videotestsrc pattern=ball \
    ! video/x-raw,format=RGBA,width=1280,height=720,framerate=30/1 ! clocksync";
const DRM_FORMAT_MOD_LINEAR: u64 = 0;

fn main() -> glib::ExitCode {
    let description = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_PIPELINE.to_owned());
    let app = gtk4::Application::builder()
        .application_id("dev.g2g.Gtk4Video")
        .build();
    app.connect_activate(move |app| build_window(app, &description));
    // the pipeline description is ours, GTK must not parse it as its own options
    app.run_with_args::<&str>(&[])
}

fn build_window(app: &gtk4::Application, description: &str) {
    let picture = gtk4::Picture::new();
    picture.set_vexpand(true);
    let status = gtk4::Label::new(Some(description));
    status.set_wrap(true);
    let layout = gtk4::Box::new(gtk4::Orientation::Vertical, 6);
    layout.append(&picture);
    layout.append(&status);
    gtk4::ApplicationWindow::builder()
        .application(app)
        .title("g2g gtk4 video")
        .default_width(960)
        .default_height(640)
        .child(&layout)
        .build()
        .present();

    // registered before the launch line builds its appsink, which looks the channel up
    let frames = register_appsink_pull(APPSINK_CHANNEL);
    let launch = format!("{description} ! appsink channel={APPSINK_CHANNEL}");
    let pipeline = thread::spawn(move || run_pipeline(&launch));
    glib::spawn_future_local(show_frames(frames, picture, status, pipeline));
}

enum PipelineError {
    Parse(ParseError),
    Run(G2gError),
}

// a launch graph is not Send, so it is built on the thread that runs it
fn run_pipeline(launch: &str) -> Result<u64, PipelineError> {
    let graph = parse_launch(&default_registry(), launch).map_err(PipelineError::Parse)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("build the pipeline thread's tokio runtime");
    let stats = runtime
        .block_on(run_graph(graph, &MonotonicClock, LINK_CAPACITY))
        .map_err(PipelineError::Run)?;
    Ok(stats.frames_consumed)
}

async fn show_frames(
    frames: AppSinkPull,
    picture: gtk4::Picture,
    status: gtk4::Label,
    pipeline: JoinHandle<Result<u64, PipelineError>>,
) {
    let display = picture.display();
    let mut shown = 0u64;
    while let Some(frame) = frames.pull().await {
        let caps = frames
            .caps()
            .expect("appsink sends caps before the first frame");
        let kind = match frame.domain {
            MemoryDomain::DmaBuf(_) => "dmabuf",
            _ => "memory",
        };
        match texture_for(&display, frame, &caps) {
            Ok(texture) => picture.set_paintable(Some(&texture)),
            Err(error) => {
                status.set_text(&error);
                return;
            }
        }
        shown += 1;
        status.set_text(&format!("{shown} frames, {kind} textures"));
    }
    let outcome = gio::spawn_blocking(move || pipeline.join()).await;
    status.set_text(&match outcome {
        Ok(Ok(Ok(frames))) => format!("ended after {frames} frames"),
        Ok(Ok(Err(PipelineError::Parse(error)))) => format!("pipeline does not parse: {error:?}"),
        Ok(Ok(Err(PipelineError::Run(error)))) => format!("pipeline failed: {error:?}"),
        _ => "the pipeline thread panicked".to_owned(),
    });
}

fn texture_for(display: &gdk::Display, frame: Frame, caps: &Caps) -> Result<gdk::Texture, String> {
    let Caps::RawVideo {
        format,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        ..
    } = caps
    else {
        return Err(format!("expected fixed raw video, got {caps:?}"));
    };
    match frame.domain {
        MemoryDomain::DmaBuf(_) => dmabuf_texture(display, frame, *format, *width, *height),
        _ => memory_texture(frame, *format, *width, *height),
    }
}

// GTK imports the buffer itself, the pixels are never copied
fn dmabuf_texture(
    display: &gdk::Display,
    frame: Frame,
    format: RawVideoFormat,
    width: u32,
    height: u32,
) -> Result<gdk::Texture, String> {
    let MemoryDomain::DmaBuf(buffer) = &frame.domain else {
        unreachable!("texture_for routes only dmabuf frames here");
    };
    let (fourcc, planes) = drm_format(format)?;
    let (fd, stride, offset) = (buffer.as_raw(), buffer.stride, buffer.offset);
    let mut builder = gdk::DmabufTextureBuilder::new()
        .set_display(display)
        .set_width(width)
        .set_height(height)
        .set_fourcc(fourcc)
        .set_modifier(DRM_FORMAT_MOD_LINEAR)
        .set_n_planes(planes);
    for plane in 0..planes {
        // SAFETY: the frame holds `fd` open until GTK calls the release function below
        builder = unsafe { builder.set_fd(plane, fd) }
            .set_stride(plane, stride)
            .set_offset(plane, offset + plane * stride * height);
    }
    // SAFETY: the release function drops the frame, the only thing keeping `fd` open
    unsafe { builder.build_with_release_func(move || drop(frame)) }
        .map_err(|error| format!("GTK rejected the dmabuf: {error}"))
}

// one copy into GTK's memory, which GTK then uploads
fn memory_texture(
    frame: Frame,
    format: RawVideoFormat,
    width: u32,
    height: u32,
) -> Result<gdk::Texture, String> {
    let memory_format = match format {
        RawVideoFormat::Rgba8 => gdk::MemoryFormat::R8g8b8a8,
        RawVideoFormat::Bgra8 => gdk::MemoryFormat::B8g8r8a8,
        RawVideoFormat::Rgb8 => gdk::MemoryFormat::R8g8b8,
        other => {
            return Err(format!(
                "{other:?} in memory: add videoconvert ! video/x-raw,format=RGBA"
            ))
        }
    };
    let host = HostFrame::new(frame);
    let bytes = host
        .bytes()
        .ok_or("a GPU frame: download it to memory or a dmabuf first")?;
    let stride = bytes.len() / height as usize;
    let texture = gdk::MemoryTexture::new(
        width as i32,
        height as i32,
        memory_format,
        &glib::Bytes::from(bytes),
        stride,
    );
    Ok(texture.upcast())
}

// DRM fourccs name the byte order in memory, so RGBA bytes are DRM's ABGR8888
fn drm_format(format: RawVideoFormat) -> Result<(u32, u32), String> {
    let (code, planes) = match format {
        RawVideoFormat::Yuyv => (b"YUYV", 1),
        RawVideoFormat::Nv12 => (b"NV12", 2),
        RawVideoFormat::Rgba8 => (b"AB24", 1),
        RawVideoFormat::Bgra8 => (b"AR24", 1),
        other => return Err(format!("{other:?} has no dmabuf mapping in this example")),
    };
    Ok((u32::from_le_bytes(*code), planes))
}
