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
    /// Other backend-specific failure.
    Other,
}
