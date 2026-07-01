//! Tokio-backed [`GraphSpawner`] for the opt-in thread-per-arm graph runner
//! ([`g2g_core::runtime::run_graph_threaded`]).
//!
//! Each graph arm gets its own OS thread running a dedicated **current-thread**
//! tokio runtime (`enable_all`, so a network source / paced element has its own
//! reactor + timer). The arm's `!Send` future is built and driven entirely on
//! that thread; only the element and its channels (all `Send`) crossed the
//! boundary at setup. This is the GStreamer streaming-thread model: CPU-bound
//! stages (software decode/encode) run on separate cores instead of serialising
//! on one cooperative executor, at the cost of a per-stage thread handoff (so it
//! is opt-in; cooperative `run_graph` stays the low-latency default).
//!
//! Prefer this over core's `ThreadSpawner` whenever any element uses tokio
//! (network, timers): `ThreadSpawner` drives arms on a bare park-based executor
//! with no reactor.

use std::boxed::Box;
use std::thread;

use g2g_core::element::BoxFuture;
use g2g_core::error::G2gError;
use g2g_core::runtime::{GraphSpawner, LocalArmFuture};

/// Runs each graph arm on its own OS thread with a private current-thread tokio
/// runtime. See the module docs.
#[derive(Debug, Default, Clone, Copy)]
pub struct TokioThreadSpawner;

impl GraphSpawner for TokioThreadSpawner {
    fn spawn_arm(
        &self,
        build: Box<dyn FnOnce() -> LocalArmFuture + Send>,
    ) -> BoxFuture<'static, Result<u64, G2gError>> {
        // A oneshot carries the arm's single result back to the caller thread,
        // which awaits it (join). A failed handshake (thread panic / dropped
        // handle) collapses to `Shutdown`, matching how a closed link surfaces.
        let (tx, rx) = tokio::sync::oneshot::channel::<Result<u64, G2gError>>();
        thread::spawn(move || {
            let result = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                // `build()` is invoked inside `block_on` so the arm's future is
                // constructed with this runtime as its ambient context.
                Ok(rt) => rt.block_on(async move { build().await }),
                Err(_) => Err(G2gError::Shutdown),
            };
            let _ = tx.send(result);
        });
        Box::pin(async move { rx.await.unwrap_or(Err(G2gError::Shutdown)) })
    }
}
