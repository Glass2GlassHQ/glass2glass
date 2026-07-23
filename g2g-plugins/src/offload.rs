//! Element-level cooperative offload (M760). A heavy synchronous element moves
//! its per-frame compute onto tokio's blocking pool, so its arm future is
//! pending at the `await` and the cooperative `run_graph` keeps polling the
//! sibling arms (the sink renders while the convert runs), without opting into
//! `run_graph_threaded`.
//!
//! This is done at the element, not the runner: the runner joins every arm into
//! one task, so `block_in_place` there would block every sibling arm too (it
//! only yields to *other* tasks, and the arms are the same task). Only pushing
//! the blocking work into a separate pool task lets the join keep making
//! progress. See DESIGN-caps.md 4.13.3.

/// Run `f` on tokio's blocking pool and await its result. With no tokio runtime
/// active (a non-tokio executor like g2g-core's park-based `block_on`, where
/// `try_current` errs) it runs `f` inline, so the offload path is transparent
/// off-tokio. A panic inside `f` is re-raised on the caller.
pub async fn run_blocking<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => match handle.spawn_blocking(f).await {
            Ok(r) => r,
            Err(e) if e.is_panic() => std::panic::resume_unwind(e.into_panic()),
            Err(e) => panic!("offload blocking task did not complete: {e}"),
        },
        Err(_) => f(),
    }
}
