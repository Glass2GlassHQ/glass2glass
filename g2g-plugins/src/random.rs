//! Deterministic pseudo-randomness for the test / debug elements, so the same
//! seed replays the same bytes and the same buffer sizes.

/// The start state a zero seed maps to. Xorshift stays at zero forever from a
/// zero state, and the `seed` properties default to 0, so a seed is mixed into
/// this rather than used directly.
pub(crate) const XORSHIFT_BASE_STATE: u32 = 0x2545_f491;

/// Marsaglia xorshift32. Deterministic and dependency-free, which is all a fill
/// pattern or a buffer-size sequence needs; it is not a source of randomness
/// for anything else.
pub(crate) fn next_random(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// The next draw scaled into `(0, 1]`, for a `draw <= probability` comparison.
/// Xorshift never returns zero, so a probability of zero never hits.
pub(crate) fn next_unit(state: &mut u32) -> f64 {
    f64::from(next_random(state)) / f64::from(u32::MAX)
}
