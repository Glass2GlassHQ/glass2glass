// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::sync::mpsc as std_mpsc;
use std::thread;
use std::time::{Duration, Instant};

use alloc::string::String;

use tokio::sync::watch;

use g2g_core::frame::Frame;
use g2g_core::{
    CaptureAnchor, ClockSync, FrameTiming, G2gError, HardwareError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket,
};

use crate::clock::{attach_wall_clock, UnixEpochClock};

const WORKER_START_TIMEOUT: Duration = Duration::from_secs(5);
const POLL_INTERVAL: Duration = Duration::from_millis(1);

pub(crate) trait DesktopGrabber {
    // returns without waiting, None when the desktop has not changed since the last call
    fn next_image(&mut self) -> Result<Option<MemoryDomain>, G2gError>;
}

#[derive(Debug)]
struct CapturedFrame {
    image: MemoryDomain,
    capture_ns: u64,
}

#[derive(Debug)]
struct DesktopPacer {
    period: Duration,
    last_image: Option<MemoryDomain>,
    last_change: Option<Instant>,
    last_sent: Instant,
}

impl DesktopPacer {
    fn new(period: Duration, now: Instant) -> Self {
        Self {
            period,
            last_image: None,
            last_change: None,
            last_sent: now,
        }
    }

    fn next_change_allowed(&self) -> Option<Instant> {
        self.last_change.map(|changed_at| changed_at + self.period)
    }

    fn image_to_send(
        &mut self,
        changed: Option<MemoryDomain>,
        now: Instant,
    ) -> Option<MemoryDomain> {
        match changed {
            Some(mut image) => {
                image.make_shareable();
                self.last_image = Some(image);
                self.last_change = Some(now);
            }
            None if now < self.last_sent + self.period => return None,
            None => {}
        }
        let image = self.last_image.as_ref()?.share();
        self.last_sent = now;
        Some(image)
    }
}

#[derive(Debug)]
pub(crate) struct CaptureWorker {
    period: Duration,
    frames: watch::Sender<Option<CapturedFrame>>,
    ready: std_mpsc::SyncSender<Result<(), G2gError>>,
}

impl CaptureWorker {
    pub(crate) fn pace(self, grabber: &mut impl DesktopGrabber) -> Result<(), G2gError> {
        let _ = self.ready.try_send(Ok(()));
        let mut pacer = DesktopPacer::new(self.period, Instant::now());
        loop {
            if let Some(allowed) = pacer.next_change_allowed() {
                thread::sleep(allowed.saturating_duration_since(Instant::now()));
            }
            let changed = grabber.next_image()?;
            let Some(image) = pacer.image_to_send(changed, Instant::now()) else {
                thread::sleep(POLL_INTERVAL);
                continue;
            };
            let captured = CapturedFrame {
                image,
                capture_ns: g2g_core::metrics::monotonic_ns(),
            };
            if self.frames.send(Some(captured)).is_err() {
                return Ok(());
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct DesktopCaptureSettings {
    pub(crate) thread_name: &'static str,
    pub(crate) period: Duration,
    pub(crate) frame_limit: u64,
    pub(crate) clock_sync: Option<ClockSync>,
}

pub(crate) async fn run_desktop_capture(
    settings: DesktopCaptureSettings,
    out: &mut dyn OutputSink,
    capture: impl FnOnce(CaptureWorker) -> Result<(), G2gError> + Send + 'static,
) -> Result<u64, G2gError> {
    let DesktopCaptureSettings {
        thread_name,
        period,
        frame_limit,
        clock_sync,
    } = settings;
    // a frame the pipeline has not taken yet is replaced by the newer one
    let (frame_sender, mut frame_receiver) = watch::channel::<Option<CapturedFrame>>(None);
    let (ready_sender, ready_receiver) = std_mpsc::sync_channel(1);
    let worker = CaptureWorker {
        period,
        frames: frame_sender,
        ready: ready_sender.clone(),
    };
    let worker = thread::Builder::new()
        .name(String::from(thread_name))
        .spawn(move || {
            let result = capture(worker);
            if let Err(e) = &result {
                let _ = ready_sender.try_send(Err(e.clone()));
            }
            result
        })
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
    match ready_receiver.recv_timeout(WORKER_START_TIMEOUT) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            let _ = worker.join();
            return Err(e);
        }
        Err(_) => return Err(G2gError::Timeout),
    }

    let period_ns = period.as_nanos() as u64;
    let mut anchor = CaptureAnchor::new();
    let mut first_capture_ns = None;
    let mut sequence = 0u64;
    let mut downstream_open = true;
    while sequence < frame_limit {
        if frame_receiver.changed().await.is_err() {
            break;
        }
        let captured = match frame_receiver.borrow_and_update().as_ref() {
            Some(captured) => CapturedFrame {
                image: captured.image.share(),
                capture_ns: captured.capture_ns,
            },
            None => continue,
        };
        let elapsed = captured.capture_ns - *first_capture_ns.get_or_insert(captured.capture_ns);
        let pts = match &clock_sync {
            Some(sync) => anchor.stamp(sync, elapsed, period_ns),
            None => elapsed,
        };
        let mut frame = Frame {
            domain: captured.image,
            timing: FrameTiming {
                pts_ns: pts,
                dts_ns: pts,
                duration_ns: period_ns,
                capture_ns: pts,
                arrival_ns: captured.capture_ns,
                keyframe: true,
                ..FrameTiming::default()
            },
            sequence,
            meta: Default::default(),
        };
        let age_ns = g2g_core::metrics::monotonic_ns() - captured.capture_ns;
        attach_wall_clock(&mut frame, (UnixEpochClock.now_ns() - age_ns) as i64);
        if out.push(PipelinePacket::DataFrame(frame)).await.is_err() {
            downstream_open = false;
            break;
        }
        sequence += 1;
    }
    drop(frame_receiver);
    let captured = worker
        .join()
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
    if downstream_open {
        captured?;
        out.push(PipelinePacket::Eos).await?;
    }
    Ok(sequence)
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::boxed::Box;

    use g2g_core::memory::SystemSlice;

    const PERIOD: Duration = Duration::from_millis(16);

    fn image(byte: u8) -> MemoryDomain {
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([byte])))
    }

    fn bytes(image: &MemoryDomain) -> &[u8] {
        image.require_system_slice("test").expect("system image")
    }

    #[test]
    fn a_change_is_sent_at_once_and_an_unchanged_desktop_repeats_once_per_period() {
        let start = Instant::now();
        let mut pacer = DesktopPacer::new(PERIOD, start);
        let sent = pacer.image_to_send(Some(image(7)), start).expect("change");
        assert_eq!(bytes(&sent), [7]);
        assert_eq!(pacer.next_change_allowed(), Some(start + PERIOD));
        assert!(pacer.image_to_send(None, start + PERIOD / 2).is_none());
        let repeated = pacer.image_to_send(None, start + PERIOD).expect("repeat");
        assert_eq!(bytes(&repeated), [7]);
        assert!(pacer.image_to_send(None, start + PERIOD * 3 / 2).is_none());
    }

    #[test]
    fn nothing_is_sent_before_the_first_image() {
        let start = Instant::now();
        let mut pacer = DesktopPacer::new(PERIOD, start);
        assert!(pacer.image_to_send(None, start + PERIOD * 4).is_none());
        assert_eq!(pacer.next_change_allowed(), None);
    }
}
