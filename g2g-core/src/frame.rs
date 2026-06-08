use crate::caps::Caps;
use crate::memory::MemoryDomain;

#[derive(Debug)]
pub enum PipelinePacket {
    CapsChanged(Caps),
    DataFrame(Frame),
    Eos,
}

#[derive(Debug)]
pub struct Frame {
    pub domain: MemoryDomain,
    pub caps: Caps,
    pub timing: FrameTiming,
    pub sequence: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameTiming {
    pub pts_ns: u64,
    pub dts_ns: u64,
    pub duration_ns: u64,
    pub capture_ns: u64,
}
