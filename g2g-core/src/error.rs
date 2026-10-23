#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum G2gError {
    /// Phase 1: no non-empty intersection between proposed upstream caps
    /// and this element's supported caps.
    CapsMismatch,
    /// Element received a `DataFrame` before `configure_pipeline` succeeded.
    NotConfigured,
    /// Phase 2: caller should retry Phase 1 with the proposal returned in
    /// `ConfigureOutcome::ReFixate`.
    FixationFailed,
    /// Buffer pool exhausted; transient, retry after upstream drain.
    PoolExhausted,
    /// A `MemoryDomain` variant was handed to an element that cannot consume it.
    UnsupportedDomain,
    /// Two branches of a fan-out (a tee's diamond) proposed allocation
    /// parameters with no common memory domain, so there is no pool the shared
    /// producer can allocate that satisfies both. The negotiation fails loud
    /// rather than silently honouring one branch and copying for the other.
    AllocationConflict,
    /// Backend-specific hardware or driver failure.
    Hardware(HardwareError),
    /// Pipeline is shutting down; element should drain and propagate `Eos`.
    Shutdown,
    /// A `CopyPolicy` enforced on the graph was
    /// violated: the negotiated pipeline performs more memory-domain frame copies
    /// (device<->host or cross-device transfers of a raw buffer) than the budget
    /// allows. Raised before any frame flows, so a pipeline that must stay
    /// zero-copy refuses to start rather than paying the copy at runtime. Inspect
    /// `copy_plan` for the offending transfers.
    CopyBudget,
    /// A fan-in element refused an input added at runtime
    /// (`MultiInputElement::accepts_runtime_input`):
    /// it has no pad for that source, so the add fails and the run continues on
    /// the inputs it already has.
    InputRefused,
    /// An animated property binding (M882) does not fit the element it was
    /// attached to: an unknown or non-animatable property name (raised at
    /// startup, before any frame flows), or a sampled value the element refused
    /// mid-run. The offending node and property are named on the log's runtime
    /// category.
    ControlBinding,
    /// No data reached an element within the time it was given: the `watchdog`
    /// transform's `timeout` elapsed with the stream stalled.
    Timeout,
    /// A demux built with `ports` output ports was handed a batch of `streams`
    /// streams (M1233): each stream needs its own port.
    PortCountMismatch {
        ports: usize,
        streams: usize,
    },
    /// The frame of mux input `index` was already taken out of its batch by an
    /// earlier consumer (M1233).
    BatchFrameTaken {
        index: usize,
    },
    // the element holding the seek controller can no longer serve the seek it was given
    SeekRefused,
    // a network session failed at this stage, with the peer's or protocol library's message logged by the element
    Network(NetworkFailure),
    // an RTSP session failed after media started flowing, in the transport or the payload, with retina's message logged by the element
    RtspStream,
}

// the stage of a network session that failed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NetworkFailure {
    // before media flows: connecting, the handshake, or session setup such as RTSP DESCRIBE, SETUP and PLAY
    Connect,
    // after media flows: the peer closed or reset the session, or sent something the protocol forbids
    Stream,
}

impl G2gError {
    // a transport fault inside session setup is a connect failure, whatever stage the transport mapper assumed
    pub fn during_connect(self) -> Self {
        match self {
            G2gError::Network(_) => G2gError::Network(NetworkFailure::Connect),
            other => other,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HardwareError {
    /// `VkResult` code from a Vulkan call.
    Vulkan(i32),
    /// `errno` from a V4L2 ioctl.
    V4l2(i32),
    /// `wgpu` device or queue error.
    Wgpu,
    /// `HRESULT` from a Windows Media Foundation / COM call.
    MediaFoundation(i32),
    /// `CUresult` code from a CUDA Driver API call.
    Cuda(i32),
    /// Raw OS error code from a filesystem operation (`FileSrc` /
    /// `FileSink`), zero when the OS reported none.
    Io(i32),
    /// ALSA `snd_*` return code (negative errno) from `alsasink`.
    Alsa(i32),
    /// PulseAudio error code (`pa_error_code_t`) from `pulsesink`.
    PulseAudio(i32),
    /// PipeWire / SPA failure from `pipewiresink` / `pipewiresrc`; carries a
    /// negative errno where one is available, else -1.
    PipeWire(i32),
    /// MCU peripheral bus transfer failure (SPI / I2C / I2S via the
    /// `embedded-hal` seams in `g2g-mcu`). No payload: HAL error types are
    /// generic per implementation and the `no_std` core cannot carry them.
    Peripheral,
    /// Other backend-specific failure.
    Other,
    // HRESULT from a call into the Blackmagic Design DeckLink SDK
    DeckLink(i32),
    // a vendor library loaded at runtime is not installed, named by its file name
    LibraryNotFound(&'static str),
    // the status of an HTTP response that was not a success, such as 429 or 404
    Http(u16),
    // an HTTP request that failed before any response status arrived
    HttpTransport(HttpTransportFailure),
}

// the stage at which an HTTP request failed with no response status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HttpTransportFailure {
    Timeout,
    Connect,
    Request,
    Body,
    Decode,
    Redirect,
}
