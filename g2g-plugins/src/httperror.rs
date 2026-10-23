use g2g_core::{G2gError, HardwareError, HttpTransportFailure};

type FailedAt = fn(&reqwest::Error) -> bool;

// timeout comes first because a connect that times out also reports is_connect
const TRANSPORT_FAILURES: [(FailedAt, HttpTransportFailure); 6] = [
    (reqwest::Error::is_timeout, HttpTransportFailure::Timeout),
    (reqwest::Error::is_connect, HttpTransportFailure::Connect),
    (reqwest::Error::is_body, HttpTransportFailure::Body),
    (reqwest::Error::is_decode, HttpTransportFailure::Decode),
    (reqwest::Error::is_redirect, HttpTransportFailure::Redirect),
    (reqwest::Error::is_request, HttpTransportFailure::Request),
];

/// reqwest transport / status failures map to a hardware-ish I/O error; the run
/// fails loud and the pipeline surfaces it, naming the HTTP status when the
/// server answered with one, else the stage at which the request failed.
pub(crate) fn http_err(e: reqwest::Error) -> G2gError {
    if let Some(status) = e.status() {
        return G2gError::Hardware(HardwareError::Http(status.as_u16()));
    }
    match TRANSPORT_FAILURES
        .iter()
        .find(|(failed_at, _)| failed_at(&e))
    {
        Some((_, failure)) => G2gError::Hardware(HardwareError::HttpTransport(*failure)),
        None => G2gError::Hardware(HardwareError::Other),
    }
}
