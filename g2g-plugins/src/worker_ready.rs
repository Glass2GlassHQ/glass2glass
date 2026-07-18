//! One-shot worker-readiness latch shared by the platform display sinks
//! (`D3d11Sink`, `WaylandSink`, `CudaGlSink`): the element thread blocks in
//! [`Handshake::wait`] until the render worker has opened its window/device and
//! called [`Handshake::notify`], so a failed setup surfaces before the first
//! frame instead of racing it.

use std::sync::{Condvar, Mutex};
use std::time::Duration;

#[derive(Debug)]
pub(crate) struct Handshake {
    flag: Mutex<bool>,
    cv: Condvar,
}

impl Handshake {
    pub(crate) fn new() -> Self {
        Self {
            flag: Mutex::new(false),
            cv: Condvar::new(),
        }
    }

    pub(crate) fn notify(&self) {
        *self.flag.lock().unwrap() = true;
        self.cv.notify_all();
    }

    /// Returns true if notified within `timeout`, false on timeout.
    pub(crate) fn wait(&self, timeout: Duration) -> bool {
        let guard = self.flag.lock().unwrap();
        let (guard, _) = self
            .cv
            .wait_timeout_while(guard, timeout, |notified| !*notified)
            .unwrap();
        *guard
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn handshake_round_trips() {
        let hs = Arc::new(Handshake::new());
        let hs2 = Arc::clone(&hs);
        let join = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            hs2.notify();
        });
        assert!(hs.wait(Duration::from_secs(2)), "notify should land");
        join.join().unwrap();
    }

    #[test]
    fn handshake_times_out_without_notify() {
        let hs = Handshake::new();
        assert!(!hs.wait(Duration::from_millis(20)));
    }
}
