use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq)]
pub enum Caps {
    Video {
        format: VideoFormat,
        width: Dim,
        height: Dim,
        framerate: Rate,
    },
    Audio {
        format: AudioFormat,
        channels: u8,
        sample_rate: u32,
    },
    Tensor {
        dtype: TensorDType,
        shape: TensorShape,
        layout: TensorLayout,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    Any,
    Range { min: u32, max: u32 },
    Fixed(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rate {
    Any,
    /// Min/max framerate in Q16 fixed-point fps.
    Range { min_q16: u32, max_q16: u32 },
    /// Framerate in Q16 fixed-point fps.
    Fixed(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VideoFormat {
    H264,
    H265,
    Av1,
    Vp9,
    Nv12,
    I420,
    Rgba8,
    Bgra8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AudioFormat {
    Aac,
    Opus,
    PcmS16Le,
    PcmF32Le,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorDType {
    F16,
    F32,
    I8,
    U8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorShape(pub Vec<u32>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorLayout {
    Nchw,
    Nhwc,
}
