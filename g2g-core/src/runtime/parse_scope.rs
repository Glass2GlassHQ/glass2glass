extern crate std;

use core::cell::Cell;
use core::sync::atomic::{AtomicU64, Ordering};

static NEXT_PARSE_ID: AtomicU64 = AtomicU64::new(1);

std::thread_local! {
    static CURRENT_PARSE_ID: Cell<u64> = const { Cell::new(0) };
}

/// The parse running on this thread, `0` outside any parse. An element a launch
/// factory built from a bare `fn` has no other way to tell which pipeline it
/// belongs to, and one that has to find a sibling by name needs this so two
/// pipelines using the same name do not join up.
pub fn current_parse_id() -> u64 {
    CURRENT_PARSE_ID.with(Cell::get)
}

/// Marks its whole lifetime as one parse. Restores the previous id on drop, so
/// an early return or a nested parse leaves the slot as it found it.
#[derive(Debug)]
pub(crate) struct ParseScope {
    previous: u64,
}

impl ParseScope {
    pub(crate) fn enter() -> Self {
        let id = NEXT_PARSE_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            previous: CURRENT_PARSE_ID.with(|slot| slot.replace(id)),
        }
    }
}

impl Drop for ParseScope {
    fn drop(&mut self) {
        CURRENT_PARSE_ID.with(|slot| slot.set(self.previous));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scope_nests_and_restores() {
        assert_eq!(current_parse_id(), 0);
        let outer = ParseScope::enter();
        let first = current_parse_id();
        assert_ne!(first, 0);
        {
            let _inner = ParseScope::enter();
            assert_ne!(current_parse_id(), first);
        }
        assert_eq!(current_parse_id(), first);
        drop(outer);
        assert_eq!(current_parse_id(), 0);
    }
}
