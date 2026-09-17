//! Per-node property requests a running graph's arms answer (M1175): a
//! [`GraphMutator`](crate::runtime::GraphMutator) set or get is queued here and
//! the arm that owns the element performs it at its next packet boundary, so the
//! element is only ever touched from the task driving it.
//!
//! The data plane pays one acquire load per packet on a node that has a mailbox
//! at all, and allocates nothing while the mailbox is empty.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::property::{PropError, PropValue};
use crate::runtime::channel::Sender;

/// The property surface a mailbox drains into, implemented for the erased
/// element traits the arms hold, so one drain path serves a transform, a sink
/// and a fan-in element (as [`ControlTarget`](crate::controller::ControlTarget)
/// does for animated properties).
pub(crate) trait PropertyTarget {
    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError>;
    fn get_property(&self, name: &str) -> Option<PropValue>;
}

impl PropertyTarget for dyn crate::element::DynAsyncElement + '_ {
    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        crate::element::DynAsyncElement::set_property(self, name, value)
    }
    fn get_property(&self, name: &str) -> Option<PropValue> {
        crate::element::DynAsyncElement::get_property(self, name)
    }
}

impl PropertyTarget for dyn crate::runtime::DynMultiInputElement + '_ {
    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        crate::runtime::DynMultiInputElement::set_property(self, name, value)
    }
    fn get_property(&self, name: &str) -> Option<PropValue> {
        crate::runtime::DynMultiInputElement::get_property(self, name)
    }
}

/// One queued property operation and the channel its answer goes back on. The
/// answer is the element's own: a refused set carries the element's
/// [`PropError`], and a get of a name the element does not know is `None`.
#[derive(Debug)]
pub(crate) enum PropertyRequest {
    Set {
        name: String,
        value: PropValue,
        reply: Sender<Result<(), PropError>>,
    },
    Get {
        name: String,
        reply: Sender<Option<PropValue>>,
    },
}

/// One node's queue of property operations, shared between the arm that owns the
/// element and the [`MutationService`](crate::runtime::mutate::MutationService)
/// that fills it.
#[derive(Debug, Clone)]
pub(crate) struct PropertyMailbox {
    inner: Arc<Queue>,
}

#[derive(Debug)]
struct Queue {
    pending: AtomicBool,
    requests: spin::Mutex<Vec<PropertyRequest>>,
}

impl PropertyMailbox {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(Queue {
                pending: AtomicBool::new(false),
                requests: spin::Mutex::new(Vec::new()),
            }),
        }
    }

    /// Queue `request` for the arm to perform at its next packet boundary.
    pub(crate) fn push(&self, request: PropertyRequest) {
        let mut queued = self.inner.requests.lock();
        queued.push(request);
        // Raised under the lock, so a drain that has just taken the queue cannot
        // clear this flag after the push that set it.
        self.inner.pending.store(true, Ordering::Release);
    }

    /// Perform everything queued against `elem` and answer each request. The
    /// empty case is one atomic load.
    pub(crate) fn drain<E: PropertyTarget + ?Sized>(&self, elem: &mut E) {
        if !self.inner.pending.load(Ordering::Acquire) {
            return;
        }
        let requests = {
            let mut queued = self.inner.requests.lock();
            self.inner.pending.store(false, Ordering::Release);
            core::mem::take(&mut *queued)
        };
        for request in requests {
            match request {
                PropertyRequest::Set { name, value, reply } => {
                    let _ = reply.try_send(elem.set_property(&name, value));
                }
                PropertyRequest::Get { name, reply } => {
                    let _ = reply.try_send(elem.get_property(&name));
                }
            }
        }
    }
}

/// [`PropertyMailbox::drain`] for an arm that may have no mailbox: a node the
/// run was not asked to make mutable carries `None` and pays one branch.
pub(crate) fn drain_properties<E: PropertyTarget + ?Sized>(
    mailbox: Option<&PropertyMailbox>,
    elem: &mut E,
) {
    if let Some(mailbox) = mailbox {
        mailbox.drain(elem);
    }
}
