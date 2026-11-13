// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use alloc::string::String;

use crate::worker_ready::Handshake;
use g2g_core::log::Target;
use g2g_core::metrics::{monotonic_ns, LatencyHistogram};
#[cfg(feature = "wgpu-present")]
use g2g_core::Caps;
use g2g_core::{G2gError, HardwareError};

pub(crate) enum WorkerCmd<F> {
    Frame {
        frame: F,
        arrival_ns: u64,
        ack: tokio::sync::oneshot::Sender<()>,
    },
    /// New negotiated caps for the renderer, plus the size to ask the compositor
    /// for, so a mid-stream format / geometry change costs a reconfigure instead
    /// of tearing the window and its GPU device down and building them again.
    /// Only the wgpu sink follows caps in place; the GL sinks respawn.
    #[cfg(feature = "wgpu-present")]
    Reconfigure {
        caps: Caps,
        width: u32,
        height: u32,
    },
    Shutdown,
}

/// The per-sink half of the worker: how one frame's pixels reach the window's
/// surface. The window, the event loop, the readiness handshake and the pacing
/// ack are shared.
pub(crate) trait WindowRenderer: 'static {
    /// Payload the sink hands over the channel (a byte vector, a GPU frame).
    type Frame: Send + 'static;

    /// Draw `frame` and present it to the surface.
    fn present(&mut self, frame: &Self::Frame) -> Result<(), G2gError>;

    /// Take new negotiated caps mid-stream. Default: nothing to do, for a
    /// renderer whose sink respawns the worker on a caps change instead.
    #[cfg(feature = "wgpu-present")]
    fn reconfigure(&mut self, _caps: &Caps) -> Result<(), G2gError> {
        Ok(())
    }

    /// Follow the compositor's configured window size. Default: ignore it and
    /// keep drawing at the size the renderer was built for, letting the
    /// compositor letterbox.
    fn resize(&mut self, _width: u32, _height: u32) {}

    #[cfg(target_os = "linux")]
    fn picture_point(&self, surface_x: f64, surface_y: f64) -> Option<(u32, u32)>;
}

/// Window geometry + identity the worker opens with.
pub(crate) struct WindowParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) title: String,
    #[cfg(target_os = "linux")]
    pub(crate) app_id: String,
    /// Map the window fullscreen rather than as a normal toplevel.
    pub(crate) fullscreen: bool,
    /// Log category the worker's error lines go under (the sink's element name).
    pub(crate) log_tag: &'static str,
}

/// The handles the sink shares with its worker: the frame channel plus the
/// counters and readiness handshake the sink reads.
pub(crate) struct WorkerChannels<Receiver> {
    pub(crate) rx: Receiver,
    pub(crate) presented: Arc<AtomicU64>,
    pub(crate) latency: Arc<LatencyHistogram>,
    pub(crate) ready: Arc<Handshake>,
    pub(crate) window_closed: Arc<AtomicBool>,
    #[cfg(target_os = "linux")]
    pub(crate) input: Option<tokio::sync::mpsc::UnboundedSender<crate::windowinput::WindowInput>>,
}

// the channel types the window workers take commands on
pub(crate) trait CommandSender<F> {
    // false once the worker has dropped its receiver
    fn send_command(&self, command: WorkerCmd<F>) -> bool;
}

#[cfg(target_os = "linux")]
impl<F> CommandSender<F>
    for smithay_client_toolkit::reexports::calloop::channel::Sender<WorkerCmd<F>>
{
    fn send_command(&self, command: WorkerCmd<F>) -> bool {
        self.send(command).is_ok()
    }
}

#[cfg(target_os = "windows")]
impl<F> CommandSender<F> for std::sync::mpsc::Sender<WorkerCmd<F>> {
    fn send_command(&self, command: WorkerCmd<F>) -> bool {
        self.send(command).is_ok()
    }
}

pub(crate) async fn present_on_worker<F>(
    tx: &impl CommandSender<F>,
    frame: F,
    arrival_ns: u64,
    window_closed: &AtomicBool,
) -> Result<(), G2gError> {
    let (ack, presented) = tokio::sync::oneshot::channel();
    if !tx.send_command(WorkerCmd::Frame {
        frame,
        arrival_ns,
        ack,
    }) || presented.await.is_err()
    {
        return Err(worker_gone(window_closed));
    }
    Ok(())
}

pub(crate) fn worker_gone(window_closed: &AtomicBool) -> G2gError {
    if window_closed.load(Ordering::Acquire) {
        G2gError::WindowClosed
    } else {
        G2gError::Hardware(HardwareError::Other)
    }
}

// what a window worker reports about each frame it presents
pub(crate) struct PresentReports {
    pub(crate) presented: Arc<AtomicU64>,
    pub(crate) latency: Arc<LatencyHistogram>,
    pub(crate) log_tag: &'static str,
}

impl PresentReports {
    pub(crate) fn acknowledge<F>(
        &self,
        drawn: Result<(), G2gError>,
        frame: F,
        arrival_ns: u64,
        ack: tokio::sync::oneshot::Sender<()>,
    ) {
        // Let the frame go before acking. The ack is what lets the pipeline move
        // on, and at end of stream moving on means tearing the producers down: a
        // GPU frame whose memory belongs to one of them (a decoder's CUDA
        // context) must be released while that producer is still alive.
        drop(frame);
        if let Err(e) = drawn {
            g2g_core::g2g_error!(Target::category(self.log_tag), "cannot draw frame: {e:?}");
            // Release the producer so a transient GPU error doesn't deadlock
            // the pipeline; the frame just didn't paint.
            let _ = ack.send(());
            return;
        }
        self.presented.fetch_add(1, Ordering::Relaxed);
        if arrival_ns != 0 {
            let now = monotonic_ns();
            if now >= arrival_ns {
                self.latency.record(now - arrival_ns);
            }
        }
        let _ = ack.send(());
    }
}
