//! Helpers shared by the browser/wasm elements (`WasmClock`, `WebSocketSrc`,
//! the canvas sinks). The pure half carries no JS bindings so it is
//! unit-testable on the host: the `performance.now()` millisecond-to-nanosecond
//! conversion, and the callback-to-async `Inbox` that turns a JS event handler
//! (`WebSocket.onmessage`) into an awaitable stream. The wasm-only half resolves
//! the global scope and the presentation canvas, both of which differ between
//! the page's main thread and a dedicated worker.

use alloc::collections::VecDeque;
use alloc::rc::Rc;
use core::cell::RefCell;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};

/// Convert a `performance.now()` millisecond delta to integer nanoseconds.
/// Non-finite or non-positive readings clamp to 0 so a clock can never emit a
/// garbage timestamp; the `as u64` cast saturates on overflow.
pub(crate) fn ms_to_ns(ms: f64) -> u64 {
    if ms.is_finite() && ms > 0.0 {
        (ms * 1.0e6) as u64
    } else {
        0
    }
}

/// Single-consumer queue bridging a JS event callback to an `async` stream.
/// The JS side (a `Closure` registered as `WebSocket.onmessage` / `onclose`)
/// holds an [`InboxSender`] and calls `push` / `close`; the element's `run`
/// loop awaits [`Inbox::next`]. `Rc`-based and `!Send`, matching the
/// single-threaded browser executor (wasm builds without `multi-thread`).
pub(crate) struct Inbox<T> {
    state: Rc<RefCell<InboxState<T>>>,
}

struct InboxState<T> {
    queue: VecDeque<T>,
    waker: Option<Waker>,
    closed: bool,
}

impl<T> Inbox<T> {
    pub(crate) fn new() -> Self {
        Self {
            state: Rc::new(RefCell::new(InboxState {
                queue: VecDeque::new(),
                waker: None,
                closed: false,
            })),
        }
    }

    /// A sender handle for the JS callback. Cloning is a cheap `Rc` bump.
    pub(crate) fn sender(&self) -> InboxSender<T> {
        InboxSender {
            state: self.state.clone(),
        }
    }

    /// Await the next item: `Some` while items remain (even after close, so a
    /// final burst drains), then `None` once closed and empty.
    pub(crate) fn next(&self) -> NextItem<'_, T> {
        NextItem { state: &self.state }
    }

    /// Non-blocking pop of a ready item, used by `WebCodecsDecode` to drain
    /// decoder output frames that arrived since the last poll without awaiting.
    /// `None` when the queue is momentarily empty.
    #[cfg(any(test, feature = "web-codecs"))]
    pub(crate) fn try_pop(&self) -> Option<T> {
        self.state.borrow_mut().queue.pop_front()
    }

    /// Whether a consumer is currently parked (a waker is stored). Test-only
    /// introspection of the wake wiring.
    #[cfg(test)]
    pub(crate) fn waiting(&self) -> bool {
        self.state.borrow().waker.is_some()
    }
}

pub(crate) struct InboxSender<T> {
    state: Rc<RefCell<InboxState<T>>>,
}

impl<T> InboxSender<T> {
    /// Enqueue an item and wake the parked consumer, if any.
    pub(crate) fn push(&self, item: T) {
        // Take the waker out before releasing the borrow, then wake outside
        // the borrow so a re-entrant poll can't hit a double borrow.
        let waker = {
            let mut st = self.state.borrow_mut();
            st.queue.push_back(item);
            st.waker.take()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }

    /// Signal end-of-stream and wake the parked consumer. Items already queued
    /// still drain before `next` yields `None`.
    pub(crate) fn close(&self) {
        let waker = {
            let mut st = self.state.borrow_mut();
            st.closed = true;
            st.waker.take()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }
}

pub(crate) struct NextItem<'a, T> {
    state: &'a Rc<RefCell<InboxState<T>>>,
}

impl<T> Future for NextItem<'_, T> {
    type Output = Option<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut st = self.state.borrow_mut();
        if let Some(item) = st.queue.pop_front() {
            Poll::Ready(Some(item))
        } else if st.closed {
            Poll::Ready(None)
        } else {
            st.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// The global scope this wasm instance runs in. A graph driven from the page
/// gets a `Window`; one driven from a dedicated worker gets a
/// `WorkerGlobalScope`, which has no `window` and no `document`, so every
/// global lookup goes through here rather than `web_sys::window()`.
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub(crate) enum GlobalScope {
    Window(web_sys::Window),
    Worker(web_sys::WorkerGlobalScope),
}

/// Resolve `globalThis` to whichever scope it is. `None` in neither (a bare
/// wasm host), which every caller degrades on rather than failing.
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub(crate) fn global_scope() -> Option<GlobalScope> {
    use wasm_bindgen::JsCast;

    let global = js_sys::global();
    match global.dyn_into::<web_sys::Window>() {
        Ok(window) => Some(GlobalScope::Window(window)),
        Err(global) => global
            .dyn_into::<web_sys::WorkerGlobalScope>()
            .ok()
            .map(GlobalScope::Worker),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "web"))]
impl GlobalScope {
    pub(crate) fn performance(&self) -> Option<web_sys::Performance> {
        match self {
            Self::Window(window) => window.performance(),
            Self::Worker(worker) => worker.performance(),
        }
    }

    /// `setTimeout(handler, delay_ms)`, ignoring the timer id (nothing cancels).
    pub(crate) fn set_timeout(&self, handler: &js_sys::Function, delay_ms: i32) {
        let _ = match self {
            Self::Window(window) => {
                window.set_timeout_with_callback_and_timeout_and_arguments_0(handler, delay_ms)
            }
            Self::Worker(worker) => {
                worker.set_timeout_with_callback_and_timeout_and_arguments_0(handler, delay_ms)
            }
        };
    }

    /// `navigator.gpu`, which both scopes expose.
    #[cfg(feature = "web-gpu")]
    pub(crate) fn navigator_gpu(&self) -> web_sys::Gpu {
        match self {
            Self::Window(window) => window.navigator().gpu(),
            Self::Worker(worker) => worker.navigator().gpu(),
        }
    }
}

/// Where a browser sink presents: a `<canvas>` on the page, named by its `id`
/// and looked up in the document, or an `OffscreenCanvas` transferred into a
/// worker (`canvas.transferControlToOffscreen()`), which is the only way to
/// draw to the page from off the main thread.
#[cfg(all(target_arch = "wasm32", feature = "web"))]
#[derive(Debug, Clone)]
pub(crate) enum CanvasTarget {
    ElementId(alloc::string::String),
    Offscreen(web_sys::OffscreenCanvas),
}

/// A [`CanvasTarget`] resolved to the canvas object itself.
#[cfg(all(target_arch = "wasm32", feature = "web"))]
#[derive(Debug, Clone)]
pub(crate) enum Canvas {
    Element(web_sys::HtmlCanvasElement),
    Offscreen(web_sys::OffscreenCanvas),
}

#[cfg(all(target_arch = "wasm32", feature = "web"))]
impl CanvasTarget {
    /// Look the canvas up, which for an id needs the document (so it only works
    /// on the main thread). Called from `configure_pipeline`.
    pub(crate) fn resolve(&self) -> Result<Canvas, g2g_core::G2gError> {
        use wasm_bindgen::JsCast;

        let err = || g2g_core::G2gError::Hardware(g2g_core::HardwareError::Other);
        match self {
            Self::Offscreen(canvas) => Ok(Canvas::Offscreen(canvas.clone())),
            Self::ElementId(id) => {
                let Some(GlobalScope::Window(window)) = global_scope() else {
                    return Err(err());
                };
                let element = window
                    .document()
                    .ok_or_else(err)?
                    .get_element_by_id(id)
                    .ok_or_else(err)?;
                Ok(Canvas::Element(element.dyn_into().map_err(|_| err())?))
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "web"))]
impl Canvas {
    /// `canvas.getContext(context_id)`, whichever canvas kind this is.
    pub(crate) fn context(&self, context_id: &str) -> Result<js_sys::Object, g2g_core::G2gError> {
        let err = || g2g_core::G2gError::Hardware(g2g_core::HardwareError::Other);
        match self {
            Self::Element(canvas) => canvas.get_context(context_id),
            Self::Offscreen(canvas) => canvas.get_context(context_id),
        }
        .map_err(|_| err())?
        .ok_or_else(err)
    }

    /// Size the drawing buffer to the video. Only the WebGPU sink needs it: the
    /// swap-chain texture is sized from these, while the 2D sink keeps whatever
    /// size the page declared.
    #[cfg(feature = "web-gpu")]
    pub(crate) fn set_size(&self, width: u32, height: u32) {
        match self {
            Self::Element(canvas) => {
                canvas.set_width(width);
                canvas.set_height(height);
            }
            Self::Offscreen(canvas) => {
                canvas.set_width(width);
                canvas.set_height(height);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::task::{RawWaker, RawWakerVTable};

    fn noop_waker() -> Waker {
        fn clone(_: *const ()) -> RawWaker {
            RawWaker::new(core::ptr::null(), &VTABLE)
        }
        fn no_op(_: *const ()) {}
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
        // SAFETY: every vtable fn is a no-op over a null data pointer and
        // never dereferences it, so the RawWaker contract holds trivially.
        unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) }
    }

    fn poll_once<F: Future + Unpin>(fut: &mut F) -> Poll<F::Output> {
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        Pin::new(fut).poll(&mut cx)
    }

    #[test]
    fn ms_to_ns_converts_and_clamps() {
        assert_eq!(ms_to_ns(1.0), 1_000_000);
        assert_eq!(ms_to_ns(2.5), 2_500_000);
        assert_eq!(ms_to_ns(0.0), 0);
        assert_eq!(ms_to_ns(-5.0), 0);
        assert_eq!(ms_to_ns(f64::INFINITY), 0);
        assert_eq!(ms_to_ns(f64::NAN), 0);
    }

    #[test]
    fn inbox_drains_queue_in_order() {
        let inbox: Inbox<u32> = Inbox::new();
        let tx = inbox.sender();
        tx.push(10);
        tx.push(20);
        assert_eq!(poll_once(&mut inbox.next()), Poll::Ready(Some(10)));
        assert_eq!(poll_once(&mut inbox.next()), Poll::Ready(Some(20)));
    }

    #[test]
    fn inbox_parks_then_wakes_on_push() {
        let inbox: Inbox<u32> = Inbox::new();
        let tx = inbox.sender();
        assert_eq!(poll_once(&mut inbox.next()), Poll::Pending);
        assert!(inbox.waiting(), "a pending poll must register a waker");
        tx.push(7);
        assert!(!inbox.waiting(), "push must take the registered waker");
        assert_eq!(poll_once(&mut inbox.next()), Poll::Ready(Some(7)));
    }

    #[test]
    fn inbox_closes_after_draining() {
        let inbox: Inbox<u32> = Inbox::new();
        let tx = inbox.sender();
        tx.push(1);
        tx.close();
        assert_eq!(
            poll_once(&mut inbox.next()),
            Poll::Ready(Some(1)),
            "queued item drains first"
        );
        assert_eq!(
            poll_once(&mut inbox.next()),
            Poll::Ready(None),
            "then close yields None"
        );
    }

    #[test]
    fn inbox_try_pop_is_nonblocking() {
        let inbox: Inbox<u32> = Inbox::new();
        assert_eq!(inbox.try_pop(), None);
        inbox.sender().push(42);
        assert_eq!(inbox.try_pop(), Some(42));
        assert_eq!(inbox.try_pop(), None);
    }
}
