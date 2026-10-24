//! Why a [`GraphMutator`](crate::runtime::GraphMutator) operation was refused,
//! and which operation the bus is reporting. Kept out of the `std`-gated
//! mutator module because [`BusMessage`](crate::BusMessage) carries both.

use alloc::string::String;

use crate::error::G2gError;
use crate::property::PropError;

/// Why a structural mutation was refused. Every one of these leaves the graph
/// running exactly as it was.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MutationError {
    /// No element of the running graph carries this instance name.
    UnknownNode(String),
    /// The named position names no single mutable edge on the side asked for: a
    /// node with several edges there (a tee below, a muxer above), an end with
    /// none (a source above, a sink below), or a node that is not a transform
    /// when one is being lifted out.
    NotMutable(String),
    /// The element refused the caps flowing on the edge, or refused to fixate
    /// on them.
    Refused(G2gError),
    /// The caps this operation would change the edge to are not in the set the
    /// chain below it accepts, or there is no snapshot of that set to check
    /// them against. Consent has to be established before the caps move, since
    /// an element that turns one down mid-stream fails the whole run.
    DownstreamRefused,
    /// Nothing has crossed the edge yet, so there are no caps to configure the
    /// new element against.
    NoCaps,
    /// The run has ended (or its producer's arm has), so there is no graph left
    /// to mutate.
    GraphEnded,
    /// The element refused the property value: its own
    /// [`PropError`], so an unknown name, a kind mismatch and an out-of-range
    /// value are told apart the way a build-time `set_property` tells them apart.
    PropertyRejected(PropError),
    /// The producer the operation waits on did not reach a packet boundary
    /// within the handle's [`timeout`](crate::runtime::GraphMutator::with_timeout).
    /// Nothing was changed.
    TimedOut,
    /// The handle carries a timeout but the run has no clock that can sleep on a
    /// deadline, so nothing can measure it.
    NoTimer,
    /// Every pad the muxer declares is already taken.
    NoFreePad,
}

/// Which [`GraphMutator`](crate::runtime::GraphMutator) operation a
/// [`BusMessage::GraphMutated`](crate::BusMessage::GraphMutated) reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MutationKind {
    /// A chain of transforms spliced onto an edge.
    Insert,
    /// A transform lifted off an edge.
    Remove,
    ReplaceSink,
    ReplaceSource,
    SetProperty,
    /// A branch attached under a tee.
    AddBranch,
    /// A branch dropped from under a tee.
    RemoveBranch,
    /// A source attached to a free muxer pad.
    AddInput,
}
