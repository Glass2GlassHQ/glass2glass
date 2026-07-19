//! Browser implementation of [`PipelineClock`] / [`AsyncClock`], backed by
//! `performance.now()` and `setTimeout`. The wasm analog of `WallClock`, which
//! is tokio-backed and does not tick on `wasm32-unknown-unknown`.
//!
//! `!Send` (the sleep future owns JS values); wasm builds without the
//! `multi-thread` feature, so the empty `ElementBound` accepts it.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{AsyncClock, PipelineClock};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::webutil::ms_to_ns;

#[derive(Debug, Clone)]
pub struct WasmClock {
    /// `performance.now()` captured at construction, so `now_ns` starts near
    /// zero like `WallClock`'s `Instant` epoch.
    epoch_ms: f64,
}

impl WasmClock {
    pub fn new() -> Self {
        Self {
            epoch_ms: performance_now_ms(),
        }
    }
}

impl Default for WasmClock {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineClock for WasmClock {
    fn now_ns(&self) -> u64 {
        ms_to_ns(performance_now_ms() - self.epoch_ms)
    }
}

impl AsyncClock for WasmClock {
    type SleepFuture<'a> = Pin<Box<dyn Future<Output = ()> + 'a>>;

    fn sleep_until_ns<'a>(&'a self, deadline_ns: u64) -> Self::SleepFuture<'a> {
        let now = self.now_ns();
        Box::pin(async move {
            if deadline_ns > now {
                let delay_ms = ((deadline_ns - now) as f64) / 1.0e6;
                sleep_ms(delay_ms).await;
            }
        })
    }
}

/// `window.performance.now()`, or 0.0 if unavailable (no `window`). A missing
/// clock degrades to a zero reading rather than panicking.
fn performance_now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

/// Await `setTimeout(delay_ms)` as a future. Resolves immediately if there is
/// no `window` to schedule on, so a pipeline never hangs on a missing timer.
async fn sleep_ms(delay_ms: f64) {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        match web_sys::window() {
            Some(win) => {
                let _ = win.set_timeout_with_callback_and_timeout_and_arguments_0(
                    &resolve,
                    delay_ms as i32,
                );
            }
            // No scheduler: resolve now so the await completes.
            None => {
                let _ = resolve.call0(&JsValue::NULL);
            }
        }
    });
    let _ = JsFuture::from(promise).await;
}
