#[derive(Debug, Clone, PartialEq, Eq)]
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Other backend-specific failure.
    Other,
}
