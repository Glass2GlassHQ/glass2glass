// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_core::Reconfigure;

// gst's videodecoder max-errors default: decoding goes on through this many failures in a row
const MAX_CONSECUTIVE_DECODE_ERRORS: u32 = 10;

#[derive(Debug, Default)]
pub(crate) struct DecodeErrorRecovery {
    consecutive_errors: u32,
    awaiting_keyframe: bool,
    keyframe_request_pending: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DecodeErrorBudgetSpent;

impl DecodeErrorRecovery {
    pub(crate) fn admits(&mut self, is_keyframe: impl FnOnce() -> bool) -> bool {
        // after a failure the pictures up to the next keyframe reference what failed to decode
        if self.awaiting_keyframe && !is_keyframe() {
            return false;
        }
        self.awaiting_keyframe = false;
        true
    }

    pub(crate) fn decoded(&mut self) {
        self.consecutive_errors = 0;
    }

    pub(crate) fn failed(&mut self) -> Result<(), DecodeErrorBudgetSpent> {
        self.consecutive_errors += 1;
        if self.consecutive_errors > MAX_CONSECUTIVE_DECODE_ERRORS {
            return Err(DecodeErrorBudgetSpent);
        }
        self.awaiting_keyframe = true;
        self.keyframe_request_pending = true;
        Ok(())
    }

    pub(crate) fn take_keyframe_request(&mut self) -> Option<Reconfigure> {
        core::mem::take(&mut self.keyframe_request_pending).then_some(Reconfigure::ForceKeyframe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failure_drops_until_a_keyframe_and_asks_for_one_once() {
        let mut recovery = DecodeErrorRecovery::default();
        assert!(recovery.admits(|| false), "nothing failed yet");
        recovery.failed().expect("within the budget");
        assert_eq!(
            recovery.take_keyframe_request(),
            Some(Reconfigure::ForceKeyframe)
        );
        assert_eq!(recovery.take_keyframe_request(), None);
        assert!(!recovery.admits(|| false));
        assert!(recovery.admits(|| true));
        assert!(recovery.admits(|| false));
    }

    #[test]
    fn failures_in_a_row_past_the_budget_end_decoding() {
        let mut recovery = DecodeErrorRecovery::default();
        for _ in 0..MAX_CONSECUTIVE_DECODE_ERRORS {
            recovery.failed().expect("within the budget");
        }
        assert_eq!(recovery.failed(), Err(DecodeErrorBudgetSpent));
        recovery.decoded();
        assert!(
            recovery.failed().is_ok(),
            "a decoded picture resets the count"
        );
    }
}
