use alloc::boxed::Box;
use std::sync::mpsc::{self, Sender};
use std::sync::{Condvar, Mutex, OnceLock, PoisonError};

use g2g_core::{g2g_error, G2gError};

pub trait GpuFence: core::fmt::Debug + Send {
    fn wait(&self) -> Result<(), G2gError>;
}

#[derive(Debug)]
struct Release {
    fence: Box<dyn GpuFence>,
    resources: Box<dyn core::fmt::Debug + Send>,
}

#[derive(Debug)]
struct ReleaseQueue {
    sender: Mutex<Sender<Release>>,
    pending: Mutex<usize>,
    drained: Condvar,
}

impl ReleaseQueue {
    fn finish_one(&self) {
        let mut pending = self.pending.lock().unwrap_or_else(PoisonError::into_inner);
        *pending -= 1;
        if *pending == 0 {
            self.drained.notify_all();
        }
    }
}

fn release_queue() -> &'static ReleaseQueue {
    static QUEUE: OnceLock<ReleaseQueue> = OnceLock::new();
    QUEUE.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<Release>();
        std::thread::Builder::new()
            .name("g2g-gpu-release".into())
            .spawn(move || {
                for release in receiver {
                    if let Err(error) = release.fence.wait() {
                        g2g_error!(
                            g2g_core::log::Target::category("gpu"),
                            "waiting for GPU work before freeing {:?} failed: {:?}",
                            release.resources,
                            error
                        );
                    }
                    drop(release);
                    release_queue().finish_one();
                }
            })
            .expect("spawn the GPU release thread");
        ReleaseQueue {
            sender: Mutex::new(sender),
            pending: Mutex::new(0),
            drained: Condvar::new(),
        }
    })
}

// drops `resources` on a background thread once the GPU work before `fence` completes
pub fn release_after(fence: impl GpuFence + 'static, resources: Box<dyn core::fmt::Debug + Send>) {
    let queue = release_queue();
    *queue.pending.lock().unwrap_or_else(PoisonError::into_inner) += 1;
    let sender = queue.sender.lock().unwrap_or_else(PoisonError::into_inner);
    // the receiver lives in a thread that never exits
    let _ = sender.send(Release {
        fence: Box::new(fence),
        resources,
    });
}

#[cfg(feature = "hip")]
pub(crate) fn wait_until_released() {
    let queue = release_queue();
    let pending = queue.pending.lock().unwrap_or_else(PoisonError::into_inner);
    drop(
        queue
            .drained
            .wait_while(pending, |pending| *pending > 0)
            .unwrap_or_else(PoisonError::into_inner),
    );
}
