use core::future::Future;
use core::pin::Pin;
use std::sync::Mutex;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::element::QosMessage;
use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ClockSync, ConfigureOutcome, Dim, G2gError,
    HardwareError, Interlace, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PresentationPacer, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
    MAX_LATENESS_PROPERTY, QOS_INTERVAL_PROPERTY,
};
use ndk::hardware_buffer_format::HardwareBufferFormat;
use ndk::native_window::NativeWindow;

use crate::clock::wait_to_present;
#[cfg(feature = "mediacodec-wgpu")]
use crate::gpu::GpuContext;
#[cfg(feature = "mediacodec-wgpu")]
use crate::mediacodec_wgpu::{create_android_surface, WgpuRgbaTexture};
#[cfg(feature = "mediacodec-wgpu")]
use crate::wgpusink::WgpuSink;
#[cfg(feature = "mediacodec-wgpu")]
use g2g_core::memory::{MemoryDomain, OwnedWgpuTexture};

const DEFAULT_CHANNEL: &str = "default";
#[cfg(not(feature = "mediacodec-wgpu"))]
const ACCEPTED_DOMAINS: DomainSet = DomainSet::only(MemoryDomainKind::System);
#[cfg(feature = "mediacodec-wgpu")]
const ACCEPTED_DOMAINS: DomainSet =
    DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::WgpuTexture);
const RGBA_BYTES_PER_PIXEL: usize = 4;

// Sinks draw while holding this lock, so detach_surface returns only after the last draw.
static SURFACES: Mutex<BTreeMap<String, AttachedWindow>> = Mutex::new(BTreeMap::new());

struct AttachedWindow {
    // must drop before `window`, which the wgpu surface inside was created over
    #[cfg(feature = "mediacodec-wgpu")]
    mode: WindowMode,
    window: NativeWindow,
}

// ANativeWindow_lock fails on a window that has a wgpu surface over it.
#[cfg(feature = "mediacodec-wgpu")]
enum WindowMode {
    NoFrameYet,
    CpuCopy,
    Wgpu(Box<WgpuPresenter>),
}

#[cfg(feature = "mediacodec-wgpu")]
struct WgpuPresenter {
    instance: wgpu::Instance,
    sink: WgpuSink,
}

pub fn attach_surface(channel: &str, window: NativeWindow) {
    let attached = AttachedWindow {
        #[cfg(feature = "mediacodec-wgpu")]
        mode: WindowMode::NoFrameYet,
        window,
    };
    surfaces().insert(channel.to_string(), attached);
}

pub fn detach_surface(channel: &str) {
    surfaces().remove(channel);
}

fn surfaces() -> std::sync::MutexGuard<'static, BTreeMap<String, AttachedWindow>> {
    SURFACES.lock().expect("a sink panicked while drawing")
}

// Draws RGBA frames, from host memory or a wgpu texture, into the window attached under its channel, and drops them while none is.
#[derive(Debug)]
pub struct SurfaceSink {
    channel: String,
    width: usize,
    height: usize,
    presented: u64,
    pacer: PresentationPacer,
}

impl Default for SurfaceSink {
    fn default() -> Self {
        Self::new()
    }
}

impl SurfaceSink {
    pub fn new() -> Self {
        Self {
            channel: DEFAULT_CHANNEL.to_string(),
            width: 0,
            height: 0,
            presented: 0,
            pacer: PresentationPacer::for_video_sink(),
        }
    }

    pub fn presented(&self) -> u64 {
        self.presented
    }

    fn present(&mut self, frame: &Frame) -> Result<(), G2gError> {
        #[cfg(feature = "mediacodec-wgpu")]
        if let MemoryDomain::WgpuTexture(owned) = &frame.domain {
            return self.present_texture(&frame.domain, owned);
        }
        let bytes = frame
            .domain
            .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
        let row_bytes = self.width * RGBA_BYTES_PER_PIXEL;
        if bytes.len() != row_bytes * self.height {
            return Err(G2gError::CapsMismatch);
        }
        let mut surfaces = surfaces();
        let Some(attached) = surfaces.get_mut(&self.channel) else {
            return Ok(());
        };
        #[cfg(feature = "mediacodec-wgpu")]
        {
            if let WindowMode::Wgpu(_) = attached.mode {
                return Err(G2gError::UnsupportedDomain);
            }
            attached.mode = WindowMode::CpuCopy;
        }
        let window = &attached.window;
        let hardware = |_| G2gError::Hardware(HardwareError::Other);
        let width = i32::try_from(self.width).map_err(|_| G2gError::CapsMismatch)?;
        let height = i32::try_from(self.height).map_err(|_| G2gError::CapsMismatch)?;
        window
            .set_buffers_geometry(width, height, Some(HardwareBufferFormat::R8G8B8A8_UNORM))
            .map_err(hardware)?;
        let mut buffer = window.lock(None).map_err(hardware)?;
        if (buffer.width(), buffer.height()) != (self.width, self.height) {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        let lines = buffer
            .lines()
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        for (line, row) in lines.zip(bytes.chunks_exact(row_bytes)) {
            for (pixel, byte) in line.iter_mut().zip(row) {
                pixel.write(*byte);
            }
        }
        self.presented += 1;
        Ok(())
    }

    #[cfg(feature = "mediacodec-wgpu")]
    fn present_texture(
        &mut self,
        domain: &MemoryDomain,
        owned: &OwnedWgpuTexture,
    ) -> Result<(), G2gError> {
        let mut surfaces = surfaces();
        let Some(attached) = surfaces.get_mut(&self.channel) else {
            return Ok(());
        };
        let presenter = wgpu_presenter_for(attached, owned)?;
        let presented_before = presenter.sink.presented_count();
        presenter.sink.present_frame(domain)?;
        self.presented += presenter.sink.presented_count() - presented_before;
        Ok(())
    }
}

#[cfg(feature = "mediacodec-wgpu")]
fn wgpu_presenter_for<'a>(
    attached: &'a mut AttachedWindow,
    owned: &OwnedWgpuTexture,
) -> Result<&'a mut WgpuPresenter, G2gError> {
    let context = owned
        .keep_alive()
        .as_any()
        .downcast_ref::<WgpuRgbaTexture>()
        .ok_or(G2gError::UnsupportedDomain)?
        .context();
    let needs_new_presenter = match &attached.mode {
        WindowMode::CpuCopy => return Err(G2gError::UnsupportedDomain),
        WindowMode::NoFrameYet => true,
        // a texture binds only to the device that made it, which another decoder does not share
        WindowMode::Wgpu(presenter) => presenter.instance != context.instance,
    };
    if needs_new_presenter {
        // the old surface has to release the window before a new one is created over it
        attached.mode = WindowMode::NoFrameYet;
        attached.mode = WindowMode::Wgpu(Box::new(new_wgpu_presenter(context, &attached.window)?));
    }
    let WindowMode::Wgpu(presenter) = &mut attached.mode else {
        unreachable!("the mode is Wgpu once the presenter exists")
    };
    Ok(presenter.as_mut())
}

// The surface takes the window's size and WgpuSink scales each frame to it.
#[cfg(feature = "mediacodec-wgpu")]
fn new_wgpu_presenter(
    context: &GpuContext,
    window: &NativeWindow,
) -> Result<WgpuPresenter, G2gError> {
    let side =
        |pixels: i32| u32::try_from(pixels).map_err(|_| G2gError::Hardware(HardwareError::Other));
    let (surface, config) = create_android_surface(
        context,
        window,
        side(window.width())?,
        side(window.height())?,
    )?;
    Ok(WgpuPresenter {
        instance: context.instance.clone(),
        sink: WgpuSink::with_surface(context.clone(), surface, config),
    })
}

impl AsyncElement for SurfaceSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        ACCEPTED_DOMAINS
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&rgba_any())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(rgba_any()))
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync);
    }

    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pacer.take_qos()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SURFACESINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if name == "channel" {
            self.channel = value.as_str().ok_or(PropError::Type)?.to_string();
            return Ok(());
        }
        self.pacer
            .set_property(name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if name == "channel" {
            return Some(PropValue::Str(self.channel.clone()));
        }
        self.pacer.get_property(name)
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(Caps::RawVideo {
                    format: RawVideoFormat::Rgba8,
                    width,
                    height,
                    ..
                }) => {
                    let fixed = |dim: Dim| dim.fixed().ok_or(G2gError::CapsMismatch);
                    self.width = fixed(width)? as usize;
                    self.height = fixed(height)? as usize;
                }
                PipelinePacket::CapsChanged(_) => return Err(G2gError::CapsMismatch),
                PipelinePacket::DataFrame(frame) => {
                    let paced = self.pacer.judge(frame.timing.pts_ns, self.presented);
                    if wait_to_present(paced).await {
                        self.present(&frame)?;
                    }
                }
                PipelinePacket::Segment(segment) => self.pacer.set_segment(segment),
                PipelinePacket::Flush => self.pacer.flush(),
                _ => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for SurfaceSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::one(rgba_any()))])
    }
}

fn rgba_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

static SURFACESINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "channel",
        PropKind::Str,
        "name the application attaches its Surface under (default \"default\")",
    ),
    MAX_LATENESS_PROPERTY,
    QOS_INTERVAL_PROPERTY,
];
