use g2g_core::{PropError, PropKind, PropValue, PropertySpec};

/// Reconnect policy of the network elements. Off by default: a session failure
/// surfaces as an error and the element ends. When enabled, the element
/// retries up to `max_attempts` times with exponential backoff between
/// `initial_backoff_ms` and `max_backoff_ms`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReconnectPolicy {
    pub(crate) max_attempts: u32,
    pub(crate) initial_backoff_ms: u64,
    pub(crate) max_backoff_ms: u64,
}

impl ReconnectPolicy {
    #[cfg(feature = "rtsp")]
    pub(crate) const DISABLED: Self = Self {
        max_attempts: 0,
        initial_backoff_ms: 0,
        max_backoff_ms: 0,
    };

    /// Fill the backoff bounds a bare `with_reconnect(n)` / `reconnect=n`
    /// leaves at zero, so a caller that only asks for retries gets the
    /// documented 250 ms .. 5 s doubling.
    pub(crate) fn fill_backoff_defaults(&mut self) {
        if self.initial_backoff_ms == 0 {
            self.initial_backoff_ms = DEFAULT_INITIAL_BACKOFF_MS;
        }
        if self.max_backoff_ms == 0 {
            self.max_backoff_ms = DEFAULT_MAX_BACKOFF_MS;
        }
    }

    /// Sets `reconnect`, `reconnect-backoff` or `reconnect-backoff-max`, `None` for any other name.
    pub(crate) fn set_property(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Option<Result<(), PropError>> {
        let uint = || value.as_uint().ok_or(PropError::Type);
        let result = match name {
            "reconnect" => uint().and_then(|attempts| {
                self.max_attempts = u32::try_from(attempts).map_err(|_| PropError::Value)?;
                self.fill_backoff_defaults();
                Ok(())
            }),
            "reconnect-backoff" => uint().map(|ms| self.initial_backoff_ms = ms),
            "reconnect-backoff-max" => uint().map(|ms| self.max_backoff_ms = ms),
            _ => return None,
        };
        Some(result)
    }

    pub(crate) fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "reconnect" => Some(PropValue::Uint(u64::from(self.max_attempts))),
            "reconnect-backoff" => Some(PropValue::Uint(self.initial_backoff_ms)),
            "reconnect-backoff-max" => Some(PropValue::Uint(self.max_backoff_ms)),
            _ => None,
        }
    }
}

/// Wait before the first retry when only an attempt count was asked for.
pub(crate) const DEFAULT_INITIAL_BACKOFF_MS: u64 = 250;
/// Cap on the doubling retry backoff.
pub(crate) const DEFAULT_MAX_BACKOFF_MS: u64 = 5_000;

/// Retry counter for one [`ReconnectPolicy`], shared by the network elements
/// so all of them count attempts and double the backoff the same way.
#[derive(Debug, Clone)]
pub(crate) struct ReconnectState {
    /// Retries taken so far, for the log line.
    pub(crate) attempt: u32,
    backoff_ms: u64,
}

impl ReconnectState {
    pub(crate) fn new(policy: &ReconnectPolicy) -> Self {
        Self {
            attempt: 0,
            backoff_ms: policy.initial_backoff_ms.max(1),
        }
    }

    // a session that carried media again ends the loss, so the next loss gets every attempt
    #[cfg(feature = "rtsp")]
    pub(crate) fn recovered(&mut self, policy: &ReconnectPolicy) {
        *self = Self::new(policy);
    }

    /// Milliseconds to wait before the next attempt, or `None` once
    /// `max_attempts` retries have been handed out.
    pub(crate) fn next_delay_ms(&mut self, policy: &ReconnectPolicy) -> Option<u64> {
        if self.attempt >= policy.max_attempts {
            return None;
        }
        self.attempt += 1;
        let wait = self.backoff_ms;
        self.backoff_ms = wait.saturating_mul(2).min(policy.max_backoff_ms.max(wait));
        Some(wait)
    }
}

// the `reconnect` of a sink that keeps taking frames while it reconnects
#[cfg(any(feature = "rtsp-client-sink", feature = "webrtc"))]
pub(crate) const PUBLISHER_RECONNECT_PROP: PropertySpec = PropertySpec::new(
    "reconnect",
    PropKind::Uint,
    "max reconnect attempts after each loss of the session, frames are dropped meanwhile (0 = no reconnect)",
)
.with_default("0");

pub(crate) const RECONNECT_BACKOFF_PROP: PropertySpec = PropertySpec::new(
    "reconnect-backoff",
    PropKind::Uint,
    "wait before the first reconnect attempt, milliseconds (doubles per retry)",
)
.with_default("250");

pub(crate) const RECONNECT_BACKOFF_MAX_PROP: PropertySpec = PropertySpec::new(
    "reconnect-backoff-max",
    PropKind::Uint,
    "cap on the doubling reconnect backoff, milliseconds",
)
.with_default("5000");

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "rtsp")]
    #[test]
    fn each_recovered_loss_gets_the_whole_reconnect_budget() {
        let policy = ReconnectPolicy {
            max_attempts: 1,
            initial_backoff_ms: DEFAULT_INITIAL_BACKOFF_MS,
            max_backoff_ms: DEFAULT_MAX_BACKOFF_MS,
        };
        let mut retry = ReconnectState::new(&policy);
        assert_eq!(
            retry.next_delay_ms(&policy),
            Some(DEFAULT_INITIAL_BACKOFF_MS)
        );
        retry.recovered(&policy);
        assert_eq!(
            retry.next_delay_ms(&policy),
            Some(DEFAULT_INITIAL_BACKOFF_MS),
            "the second loss retries from the first backoff"
        );
        assert_eq!(
            retry.next_delay_ms(&policy),
            None,
            "within one loss the budget holds"
        );
    }
}
