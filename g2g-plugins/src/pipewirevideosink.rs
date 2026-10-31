use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};

use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use pipewire as pw;
use pw::spa;
use spa::param::ParamType;
use spa::utils::Fraction;

use g2g_core::caps::framerate_q16_to_fraction;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    HardwareError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat,
};

use crate::pwvideo::{
    fixed_format_pod_bytes, mapped_buffers_pod_bytes, spa_format, supported_formats, PlaneLayout,
};

const DEFAULT_CLIENT_NAME: &str = "g2g-pipewirevideosink";
// bounds the wait for the PipeWire daemon to accept the stream
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

enum Ctrl {
    Frame,
    Terminate,
}

// the newest frame, which the next graph cycle sends
type LatestFrame = Arc<Mutex<Option<Vec<u8>>>>;

// publishes raw video as a PipeWire Video/Source node with the Camera role, the camera other applications see
pub struct PipeWireVideoSink {
    client_name: String,
    ctrl_tx: Option<pw::channel::Sender<Ctrl>>,
    worker: Option<JoinHandle<()>>,
    latest: LatestFrame,
    caps: Option<Caps>,
    frames_sent: Arc<AtomicU64>,
}

struct StreamCfg {
    client_name: String,
    format: RawVideoFormat,
    width: u32,
    height: u32,
    framerate: Fraction,
    layout: PlaneLayout,
}

impl core::fmt::Debug for PipeWireVideoSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PipeWireVideoSink")
            .field("client_name", &self.client_name)
            .field("caps", &self.caps)
            .field("frames_sent", &self.frames_sent.load(Ordering::Relaxed))
            .finish()
    }
}

impl Default for PipeWireVideoSink {
    fn default() -> Self {
        Self::new()
    }
}

impl PipeWireVideoSink {
    pub fn new() -> Self {
        Self {
            client_name: String::from(DEFAULT_CLIENT_NAME),
            ctrl_tx: None,
            worker: None,
            latest: Arc::new(Mutex::new(None)),
            caps: None,
            frames_sent: Arc::new(AtomicU64::new(0)),
        }
    }

    // the node name consumers pick the camera by
    pub fn with_client_name(mut self, name: impl Into<String>) -> Self {
        self.client_name = name.into();
        self
    }

    // frames a consumer's graph cycle took
    pub fn frames_sent(&self) -> u64 {
        self.frames_sent.load(Ordering::Relaxed)
    }

    fn shutdown(&mut self) {
        if let Some(tx) = self.ctrl_tx.take() {
            let _ = tx.send(Ctrl::Terminate);
        }
        if let Some(join) = self.worker.take() {
            let _ = join.join();
        }
    }

    fn stream_cfg(&self, caps: &Caps) -> Result<StreamCfg, G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate,
            ..
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        let layout = PlaneLayout::new(*format, *width, *height).ok_or(G2gError::CapsMismatch)?;
        // a stream with no fixed rate offers a variable one, 0/1 in SPA terms
        let (num, denom) = match framerate {
            Rate::Fixed(q16) => framerate_q16_to_fraction(*q16),
            _ => (0, 1),
        };
        Ok(StreamCfg {
            client_name: self.client_name.clone(),
            format: *format,
            width: *width,
            height: *height,
            framerate: Fraction { num, denom },
            layout,
        })
    }
}

impl Drop for PipeWireVideoSink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn accepted_caps() -> CapsSet {
    CapsSet::from_alternatives(
        supported_formats()
            .map(|format| Caps::RawVideo {
                format,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            })
            .collect(),
    )
}

impl AsyncElement for PipeWireVideoSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        crate::dmabufmap::audio_input_domains()
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::RawVideo { format, .. } if spa_format(*format).is_some() => {
                Ok(upstream_caps.clone())
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(accepted_caps())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.worker.is_some() {
            if self.caps.as_ref() == Some(absolute_caps) {
                return Ok(ConfigureOutcome::Accepted);
            }
            self.shutdown();
        }
        let cfg = self.stream_cfg(absolute_caps)?;
        let (ctrl_tx, ctrl_rx) = pw::channel::channel::<Ctrl>();
        let (ready_tx, ready_rx) = sync_channel::<Result<(), i32>>(1);
        let latest = Arc::clone(&self.latest);
        let frames_sent = Arc::clone(&self.frames_sent);
        let join = thread::Builder::new()
            .name(String::from(DEFAULT_CLIENT_NAME))
            .spawn(move || {
                if let Err(code) = build_and_run(&cfg, latest, frames_sent, ctrl_rx, &ready_tx) {
                    let _ = ready_tx.send(Err(code));
                }
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let ready = ready_rx.recv_timeout(CONNECT_TIMEOUT).unwrap_or(Err(-1));
        if let Err(code) = ready {
            let _ = join.join();
            return Err(G2gError::Hardware(HardwareError::PipeWire(code)));
        }
        self.ctrl_tx = Some(ctrl_tx);
        self.worker = Some(join);
        self.caps = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "PipeWire video sink",
            "Sink/Video",
            "Publishes raw video as a PipeWire camera node other applications capture from",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[PropertySpec::new(
            "client-name",
            PropKind::Str,
            "node name the camera is published under",
        )
        .with_default(DEFAULT_CLIENT_NAME)];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "client-name" => {
                self.client_name = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "client-name" => Some(PropValue::Str(self.client_name.clone())),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let (Some(tx), Some(caps)) = (self.ctrl_tx.as_ref(), self.caps.as_ref()) else {
                        return Err(G2gError::NotConfigured);
                    };
                    let expected = self.stream_cfg(caps)?.layout.frame_bytes();
                    let element = g2g_core::log::short_type_name::<Self>();
                    let payload = crate::dmabufmap::frame_bytes(&frame, element)?;
                    let bytes = payload.as_slice();
                    if bytes.len() != expected {
                        g2g_core::g2g_error!(
                            g2g_core::log::Target::category(element),
                            "frame of {} bytes, the negotiated format needs {expected}",
                            bytes.len()
                        );
                        return Err(G2gError::CapsMismatch);
                    }
                    *self
                        .latest
                        .lock()
                        .map_err(|_| G2gError::Hardware(HardwareError::PipeWire(-1)))? =
                        Some(bytes.to_vec());
                    tx.send(Ctrl::Frame)
                        .map_err(|_| G2gError::Hardware(HardwareError::PipeWire(-1)))?;
                    Ok(())
                }
                PipelinePacket::CapsChanged(c) => self.configure_pipeline(&c).map(|_| ()),
                PipelinePacket::Eos => {
                    self.shutdown();
                    Ok(())
                }
                _ => Ok(()),
            }
        })
    }
}

impl PadTemplates for PipeWireVideoSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(accepted_caps())])
    }
}

fn build_and_run(
    cfg: &StreamCfg,
    latest: LatestFrame,
    frames_sent: Arc<AtomicU64>,
    ctrl_rx: pw::channel::Receiver<Ctrl>,
    ready: &std::sync::mpsc::SyncSender<Result<(), i32>>,
) -> Result<(), i32> {
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None).map_err(|_| -1)?;
    let context = pw::context::ContextRc::new(&mainloop, None).map_err(|_| -1)?;
    let core = context.connect_rc(None).map_err(|_| -1)?;

    let props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE => "Video",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Camera",
        *pw::keys::MEDIA_CLASS => "Video/Source",
        *pw::keys::NODE_NAME => cfg.client_name.as_str(),
        *pw::keys::NODE_DESCRIPTION => cfg.client_name.as_str(),
    };
    let stream =
        pw::stream::StreamRc::new(core, cfg.client_name.as_str(), props).map_err(|_| -1)?;

    let frame_bytes = cfg.layout.frame_bytes();
    let stride = cfg.layout.first_row_bytes();
    let buffers = mapped_buffers_pod_bytes(frame_bytes, stride).map_err(|_| -1)?;
    let _listener = stream
        .add_local_listener_with_user_data(())
        // the buffers are allocated after the format settles, so their size is announced then
        .param_changed(move |stream, (), id, param| {
            if id != ParamType::Format.as_raw() || param.is_none() {
                return;
            }
            if let Some(pod) = spa::pod::Pod::from_bytes(&buffers) {
                let _ = stream.update_params(&mut [pod]);
            }
        })
        .process(move |stream, ()| {
            let Some(frame) = latest.lock().ok().and_then(|mut slot| slot.take()) else {
                return;
            };
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let datas = buffer.datas_mut();
            let Some(data) = datas.first_mut() else {
                return;
            };
            let Some(slice) = data.data().filter(|slice| slice.len() >= frame_bytes) else {
                return;
            };
            slice[..frame_bytes].copy_from_slice(&frame);
            let chunk = data.chunk_mut();
            *chunk.offset_mut() = 0;
            *chunk.stride_mut() = stride as _;
            *chunk.size_mut() = frame_bytes as _;
            frames_sent.fetch_add(1, Ordering::Relaxed);
        })
        .register()
        .map_err(|_| -1)?;

    let spa_format = spa_format(cfg.format).ok_or(-1)?;
    let format = fixed_format_pod_bytes(
        ParamType::EnumFormat,
        spa_format,
        cfg.width,
        cfg.height,
        cfg.framerate,
    );
    let mut params = [spa::pod::Pod::from_bytes(&format).ok_or(-1)?];
    stream
        .connect(
            spa::utils::Direction::Output,
            None,
            pw::stream::StreamFlags::DRIVER | pw::stream::StreamFlags::MAP_BUFFERS,
            &mut params,
        )
        .map_err(|_| -1)?;

    let weak_loop = mainloop.downgrade();
    let driven = stream.clone();
    let _recv = ctrl_rx.attach(mainloop.loop_(), move |ctrl| match ctrl {
        // a driver with no consumer has no cycle to run, so a failed trigger drops the frame
        Ctrl::Frame => {
            let _ = driven.trigger_process();
        }
        Ctrl::Terminate => {
            if let Some(main_loop) = weak_loop.upgrade() {
                main_loop.quit();
            }
        }
    });

    let _ = ready.send(Ok(()));
    mainloop.run();
    Ok(())
}
