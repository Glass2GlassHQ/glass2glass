use std::cell::RefCell;
use std::collections::HashMap;

const CONTINUITY_COUNTER_MODULUS: u8 = 16;

thread_local! {
    static NEXT_CONTINUITY_COUNTERS: RefCell<HashMap<u16, u8>> = RefCell::new(HashMap::new());
}

pub(crate) fn next_continuity_counter(pid: u16) -> u8 {
    NEXT_CONTINUITY_COUNTERS.with_borrow_mut(|counters| {
        let next = counters.entry(pid).or_insert(0);
        let counter = *next;
        *next = (counter + 1) % CONTINUITY_COUNTER_MODULUS;
        counter
    })
}
