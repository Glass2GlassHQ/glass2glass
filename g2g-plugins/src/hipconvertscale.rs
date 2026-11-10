use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use g2g_core::caps_transform::{FieldTransform, RawVideoShape};
use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::{DomainSet, GpuDevice, MemoryDomainKind, OwnedHipBuffer};
use g2g_core::meta::{Orientation, PlaneLayout};
use g2g_core::{
    g2g_error, AsyncElement, Caps, CapsConstraint, CapsTransform, Colorimetry, ConfigureOutcome,
    Dim, ElementMetadata, Frame, G2gError, MemoryDomain, OutputSink, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

use crate::compositor::{CompositorOperator, CompositorPad};
use crate::hip::{
    HipDevice, HipEvent, HipKernel, HipKernelModule, HipMemory, HipMemoryPool, HipStream,
    KernelArgument,
};
use crate::hipdeviceid::{HipDeviceChoice, HIP_DEVICE_ID_PROP, HIP_VENDOR_PROP};
use crate::pixel::carries_yuv;

// the kernel names each format FMT_<name>, numbered by its position here
const FORMATS: [(RawVideoFormat, &str); 15] = [
    (RawVideoFormat::Nv12, "NV12"),
    (RawVideoFormat::I420, "I420"),
    (RawVideoFormat::Yuyv, "YUY2"),
    (RawVideoFormat::P010, "P010"),
    (RawVideoFormat::Rgba8, "RGBA8"),
    (RawVideoFormat::Bgra8, "BGRA8"),
    (RawVideoFormat::Rgb8, "RGB8"),
    (RawVideoFormat::Rgbp8, "RGBP8"),
    (RawVideoFormat::Bgrp8, "BGRP8"),
    (RawVideoFormat::RgbaF16, "RGBA_F16"),
    (RawVideoFormat::RgbaF32, "RGBA_F32"),
    (RawVideoFormat::RgbF16, "RGB_F16"),
    (RawVideoFormat::RgbF32, "RGB_F32"),
    (RawVideoFormat::RgbpF16, "RGBP_F16"),
    (RawVideoFormat::RgbpF32, "RGBP_F32"),
];

const KERNEL_SOURCE: &str = r#"
struct Color {
    float r, g, b, a;
};

struct Yuv {
    float y, pb, pr;
};

struct Frame {
    const unsigned char* base;
    unsigned int width, height;
    unsigned int offset[3];
    unsigned int stride[3];
};

__device__ float clamp01(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ float half_to_float(unsigned short half) {
    unsigned int exponent = (half >> 10) & 0x1fu;
    unsigned int mantissa = half & 0x3ffu;
    float sign = (half & 0x8000u) ? -1.0f : 1.0f;
    if (exponent == 0) {
        return sign * (float)mantissa * (1.0f / 16777216.0f);
    }
    if (exponent == 31) {
        return __uint_as_float(((half & 0x8000u) << 16) | 0x7f800000u | (mantissa << 13));
    }
    return __uint_as_float(((half & 0x8000u) << 16) | ((exponent + 112u) << 23) | (mantissa << 13));
}

__device__ unsigned short float_to_half(float value) {
    // callers clamp to 0..1, so no sign, infinity or overflow reaches this
    unsigned int bits = __float_as_uint(value);
    int exponent = (int)((bits >> 23) & 0xffu) - 112;
    unsigned int mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) {
            return 0;
        }
        mantissa |= 0x800000u;
        unsigned int shift = (unsigned int)(14 - exponent);
        unsigned int half = mantissa >> shift;
        unsigned int remainder = mantissa & ((1u << shift) - 1u);
        unsigned int halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (half & 1u))) {
            half++;
        }
        return (unsigned short)half;
    }
    unsigned int half = ((unsigned int)exponent << 10) | (mantissa >> 13);
    unsigned int remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
        half++;
    }
    return (unsigned short)half;
}

__device__ const unsigned char* row(const Frame& frame, int plane, int y) {
    return frame.base + frame.offset[plane] + (unsigned long long)y * frame.stride[plane];
}

__device__ float code_unit(int bit_depth) {
    return (float)(1 << (bit_depth - 8));
}

__device__ float decode_luma(float code, int bit_depth, int full_range) {
    if (full_range) {
        return code / (float)((1 << bit_depth) - 1);
    }
    return (code - 16.0f * code_unit(bit_depth)) / (219.0f * code_unit(bit_depth));
}

__device__ float decode_chroma(float code, int bit_depth, int full_range) {
    float zero = (float)(1 << (bit_depth - 1));
    if (full_range) {
        return (code - zero) / (float)((1 << bit_depth) - 1);
    }
    return (code - zero) / (224.0f * code_unit(bit_depth));
}

__device__ float encode_luma(float luma, int bit_depth, int full_range) {
    float maximum = (float)((1 << bit_depth) - 1);
    float code = full_range ? luma * maximum : 16.0f * code_unit(bit_depth) + 219.0f * code_unit(bit_depth) * luma;
    return fminf(fmaxf(rintf(code), 0.0f), maximum);
}

__device__ float encode_chroma(float chroma, int bit_depth, int full_range) {
    float maximum = (float)((1 << bit_depth) - 1);
    float zero = (float)(1 << (bit_depth - 1));
    float code = zero + (full_range ? maximum : 224.0f * code_unit(bit_depth)) * chroma;
    return fminf(fmaxf(rintf(code), 0.0f), maximum);
}

__device__ Color yuv_to_rgb(Yuv yuv, float kr, float kb) {
    float r = yuv.y + 2.0f * (1.0f - kr) * yuv.pr;
    float b = yuv.y + 2.0f * (1.0f - kb) * yuv.pb;
    float g = (yuv.y - kr * r - kb * b) / (1.0f - kr - kb);
    Color color = {clamp01(r), clamp01(g), clamp01(b), 1.0f};
    return color;
}

__device__ Yuv rgb_to_yuv(Color color, float kr, float kb) {
    float y = kr * color.r + (1.0f - kr - kb) * color.g + kb * color.b;
    Yuv yuv = {y, (color.b - y) / (2.0f * (1.0f - kb)), (color.r - y) / (2.0f * (1.0f - kr))};
    return yuv;
}

#if IN_YUV
__device__ float read_luma_code(const Frame& frame, int x, int y) {
#if IN_FORMAT == FMT_P010
    return (float)(((const unsigned short*)row(frame, 0, y))[x] >> 6);
#elif IN_FORMAT == FMT_YUY2
    return (float)row(frame, 0, y)[2 * x];
#else
    return (float)row(frame, 0, y)[x];
#endif
}

__device__ void read_chroma_codes(const Frame& frame, int x, int y, float* u, float* v) {
#if IN_FORMAT == FMT_NV12
    const unsigned char* pair = row(frame, 1, y) + 2 * x;
    *u = (float)pair[0];
    *v = (float)pair[1];
#elif IN_FORMAT == FMT_P010
    const unsigned short* pair = (const unsigned short*)row(frame, 1, y) + 2 * x;
    *u = (float)(pair[0] >> 6);
    *v = (float)(pair[1] >> 6);
#elif IN_FORMAT == FMT_YUY2
    const unsigned char* pair = row(frame, 0, y) + 4 * x;
    *u = (float)pair[1];
    *v = (float)pair[3];
#else
    *u = (float)row(frame, 1, y)[x];
    *v = (float)row(frame, 2, y)[x];
#endif
}
#else
__device__ Color read_rgb(const Frame& frame, int x, int y) {
    Color color;
    color.a = 1.0f;
#if IN_FORMAT == FMT_RGBA8 || IN_FORMAT == FMT_BGRA8 || IN_FORMAT == FMT_RGB8
    const int channels = IN_FORMAT == FMT_RGB8 ? 3 : 4;
    const unsigned char* pixel = row(frame, 0, y) + channels * x;
    color.r = pixel[IN_FORMAT == FMT_BGRA8 ? 2 : 0] / 255.0f;
    color.g = pixel[1] / 255.0f;
    color.b = pixel[IN_FORMAT == FMT_BGRA8 ? 0 : 2] / 255.0f;
    if (channels == 4) {
        color.a = pixel[3] / 255.0f;
    }
#elif IN_FORMAT == FMT_RGBP8 || IN_FORMAT == FMT_BGRP8
    const int red_plane = IN_FORMAT == FMT_BGRP8 ? 2 : 0;
    color.r = row(frame, red_plane, y)[x] / 255.0f;
    color.g = row(frame, 1, y)[x] / 255.0f;
    color.b = row(frame, 2 - red_plane, y)[x] / 255.0f;
#elif IN_FORMAT == FMT_RGBA_F16 || IN_FORMAT == FMT_RGB_F16
    const int channels = IN_FORMAT == FMT_RGB_F16 ? 3 : 4;
    const unsigned short* pixel = (const unsigned short*)row(frame, 0, y) + channels * x;
    color.r = half_to_float(pixel[0]);
    color.g = half_to_float(pixel[1]);
    color.b = half_to_float(pixel[2]);
    if (channels == 4) {
        color.a = half_to_float(pixel[3]);
    }
#elif IN_FORMAT == FMT_RGBA_F32 || IN_FORMAT == FMT_RGB_F32
    const int channels = IN_FORMAT == FMT_RGB_F32 ? 3 : 4;
    const float* pixel = (const float*)row(frame, 0, y) + channels * x;
    color.r = pixel[0];
    color.g = pixel[1];
    color.b = pixel[2];
    if (channels == 4) {
        color.a = pixel[3];
    }
#elif IN_FORMAT == FMT_RGBP_F16
    color.r = half_to_float(((const unsigned short*)row(frame, 0, y))[x]);
    color.g = half_to_float(((const unsigned short*)row(frame, 1, y))[x]);
    color.b = half_to_float(((const unsigned short*)row(frame, 2, y))[x]);
#elif IN_FORMAT == FMT_RGBP_F32
    color.r = ((const float*)row(frame, 0, y))[x];
    color.g = ((const float*)row(frame, 1, y))[x];
    color.b = ((const float*)row(frame, 2, y))[x];
#endif
    return color;
}
#endif

struct Taps {
    int x0, x1, y0, y1;
    float fx, fy;
};

__device__ Taps taps(float x, float y, int width, int height) {
    // `x` and `y` count samples, sample i centred on i
    float cx = fminf(fmaxf(x, 0.0f), (float)(width - 1));
    float cy = fminf(fmaxf(y, 0.0f), (float)(height - 1));
    Taps taps;
    taps.x0 = (int)floorf(cx);
    taps.y0 = (int)floorf(cy);
    taps.x1 = min(taps.x0 + 1, width - 1);
    taps.y1 = min(taps.y0 + 1, height - 1);
    taps.fx = cx - (float)taps.x0;
    taps.fy = cy - (float)taps.y0;
    return taps;
}

__device__ float mix4(float a, float b, float c, float d, Taps t) {
    float top = a + (b - a) * t.fx;
    float bottom = c + (d - c) * t.fx;
    return top + (bottom - top) * t.fy;
}

#if IN_YUV
__device__ float sample_luma(const Frame& frame, float x, float y) {
    // `x` and `y` are continuous source coordinates, pixel i spanning i to i + 1
    Taps t = taps(x - 0.5f, y - 0.5f, frame.width, frame.height);
    float code = mix4(read_luma_code(frame, t.x0, t.y0), read_luma_code(frame, t.x1, t.y0),
                      read_luma_code(frame, t.x0, t.y1), read_luma_code(frame, t.x1, t.y1), t);
    return decode_luma(code, IN_BIT_DEPTH, IN_FULL_RANGE);
}

__device__ void sample_chroma(const Frame& frame, float x, float y, float* pb, float* pr) {
#if IN_FORMAT == FMT_YUY2
    Taps t = taps(x * 0.5f - 0.5f, y - 0.5f, (frame.width + 1) / 2, frame.height);
#else
    Taps t = taps(x * 0.5f - 0.5f, y * 0.5f - 0.5f, (frame.width + 1) / 2, (frame.height + 1) / 2);
#endif
    float u[4], v[4];
    read_chroma_codes(frame, t.x0, t.y0, &u[0], &v[0]);
    read_chroma_codes(frame, t.x1, t.y0, &u[1], &v[1]);
    read_chroma_codes(frame, t.x0, t.y1, &u[2], &v[2]);
    read_chroma_codes(frame, t.x1, t.y1, &u[3], &v[3]);
    *pb = decode_chroma(mix4(u[0], u[1], u[2], u[3], t), IN_BIT_DEPTH, IN_FULL_RANGE);
    *pr = decode_chroma(mix4(v[0], v[1], v[2], v[3], t), IN_BIT_DEPTH, IN_FULL_RANGE);
}

__device__ Yuv sample_yuv(const Frame& frame, float x, float y) {
    Yuv yuv;
    yuv.y = sample_luma(frame, x, y);
    sample_chroma(frame, x, y, &yuv.pb, &yuv.pr);
    return yuv;
}

__device__ Color sample_color(const Frame& frame, float x, float y) {
    return yuv_to_rgb(sample_yuv(frame, x, y), IN_KR, IN_KB);
}
#else
__device__ Color sample_color(const Frame& frame, float x, float y) {
    Taps t = taps(x - 0.5f, y - 0.5f, frame.width, frame.height);
    Color c00 = read_rgb(frame, t.x0, t.y0);
    Color c10 = read_rgb(frame, t.x1, t.y0);
    Color c01 = read_rgb(frame, t.x0, t.y1);
    Color c11 = read_rgb(frame, t.x1, t.y1);
    Color color = {mix4(c00.r, c10.r, c01.r, c11.r, t), mix4(c00.g, c10.g, c01.g, c11.g, t),
                   mix4(c00.b, c10.b, c01.b, c11.b, t), mix4(c00.a, c10.a, c01.a, c11.a, t)};
    return color;
}
#endif

#if OUT_YUV
__device__ Yuv sample_out_yuv(const Frame& frame, float x, float y) {
#if SAME_MATRIX
    return sample_yuv(frame, x, y);
#else
    return rgb_to_yuv(sample_color(frame, x, y), OUT_KR, OUT_KB);
#endif
}

__device__ void write_luma(unsigned char* plane, unsigned int stride, int x, int y, float luma) {
    float code = encode_luma(luma, OUT_BIT_DEPTH, OUT_FULL_RANGE);
#if OUT_FORMAT == FMT_P010
    ((unsigned short*)(plane + (unsigned long long)y * stride))[x] = (unsigned short)code << 6;
#elif OUT_FORMAT == FMT_YUY2
    plane[(unsigned long long)y * stride + 2 * x] = (unsigned char)code;
#else
    plane[(unsigned long long)y * stride + x] = (unsigned char)code;
#endif
}

__device__ void write_chroma(unsigned char* dst, const unsigned int* offset, const unsigned int* stride,
                             int x, int y, float pb, float pr) {
    float u = encode_chroma(pb, OUT_BIT_DEPTH, OUT_FULL_RANGE);
    float v = encode_chroma(pr, OUT_BIT_DEPTH, OUT_FULL_RANGE);
#if OUT_FORMAT == FMT_NV12
    unsigned char* pair = dst + offset[1] + (unsigned long long)y * stride[1] + 2 * x;
    pair[0] = (unsigned char)u;
    pair[1] = (unsigned char)v;
#elif OUT_FORMAT == FMT_P010
    unsigned short* pair = (unsigned short*)(dst + offset[1] + (unsigned long long)y * stride[1]) + 2 * x;
    pair[0] = (unsigned short)u << 6;
    pair[1] = (unsigned short)v << 6;
#elif OUT_FORMAT == FMT_YUY2
    unsigned char* pair = dst + offset[0] + (unsigned long long)y * stride[0] + 4 * x;
    pair[1] = (unsigned char)u;
    pair[3] = (unsigned char)v;
#else
    dst[offset[1] + (unsigned long long)y * stride[1] + x] = (unsigned char)u;
    dst[offset[2] + (unsigned long long)y * stride[2] + x] = (unsigned char)v;
#endif
}
#else
__device__ void write_rgb(unsigned char* dst, const unsigned int* offset, const unsigned int* stride,
                          int x, int y, Color color) {
    float r = clamp01(color.r), g = clamp01(color.g), b = clamp01(color.b), a = clamp01(color.a);
#if OUT_FORMAT == FMT_RGBA8 || OUT_FORMAT == FMT_BGRA8 || OUT_FORMAT == FMT_RGB8
    const int channels = OUT_FORMAT == FMT_RGB8 ? 3 : 4;
    unsigned char* pixel = dst + offset[0] + (unsigned long long)y * stride[0] + channels * x;
    pixel[OUT_FORMAT == FMT_BGRA8 ? 2 : 0] = (unsigned char)rintf(r * 255.0f);
    pixel[1] = (unsigned char)rintf(g * 255.0f);
    pixel[OUT_FORMAT == FMT_BGRA8 ? 0 : 2] = (unsigned char)rintf(b * 255.0f);
    if (channels == 4) {
        pixel[3] = (unsigned char)rintf(a * 255.0f);
    }
#elif OUT_FORMAT == FMT_RGBP8 || OUT_FORMAT == FMT_BGRP8
    const int red_plane = OUT_FORMAT == FMT_BGRP8 ? 2 : 0;
    dst[offset[red_plane] + (unsigned long long)y * stride[red_plane] + x] = (unsigned char)rintf(r * 255.0f);
    dst[offset[1] + (unsigned long long)y * stride[1] + x] = (unsigned char)rintf(g * 255.0f);
    dst[offset[2 - red_plane] + (unsigned long long)y * stride[2 - red_plane] + x] = (unsigned char)rintf(b * 255.0f);
#elif OUT_FORMAT == FMT_RGBA_F16 || OUT_FORMAT == FMT_RGB_F16
    const int channels = OUT_FORMAT == FMT_RGB_F16 ? 3 : 4;
    unsigned short* pixel = (unsigned short*)(dst + offset[0] + (unsigned long long)y * stride[0]) + channels * x;
    pixel[0] = float_to_half(r);
    pixel[1] = float_to_half(g);
    pixel[2] = float_to_half(b);
    if (channels == 4) {
        pixel[3] = float_to_half(a);
    }
#elif OUT_FORMAT == FMT_RGBA_F32 || OUT_FORMAT == FMT_RGB_F32
    const int channels = OUT_FORMAT == FMT_RGB_F32 ? 3 : 4;
    float* pixel = (float*)(dst + offset[0] + (unsigned long long)y * stride[0]) + channels * x;
    pixel[0] = r;
    pixel[1] = g;
    pixel[2] = b;
    if (channels == 4) {
        pixel[3] = a;
    }
#elif OUT_FORMAT == FMT_RGBP_F16
    ((unsigned short*)(dst + offset[0] + (unsigned long long)y * stride[0]))[x] = float_to_half(r);
    ((unsigned short*)(dst + offset[1] + (unsigned long long)y * stride[1]))[x] = float_to_half(g);
    ((unsigned short*)(dst + offset[2] + (unsigned long long)y * stride[2]))[x] = float_to_half(b);
#elif OUT_FORMAT == FMT_RGBP_F32
    ((float*)(dst + offset[0] + (unsigned long long)y * stride[0]))[x] = r;
    ((float*)(dst + offset[1] + (unsigned long long)y * stride[1]))[x] = g;
    ((float*)(dst + offset[2] + (unsigned long long)y * stride[2]))[x] = b;
#endif
}
#endif

__device__ bool inside(float x, float y, unsigned int rect_x, unsigned int rect_y,
                       unsigned int rect_width, unsigned int rect_height) {
    return x > (float)rect_x && x < (float)(rect_x + rect_width) && y > (float)rect_y &&
           y < (float)(rect_y + rect_height);
}

__device__ void source_point(float x, float y, unsigned int src_width, unsigned int src_height,
                             unsigned int rect_x, unsigned int rect_y, unsigned int rect_width,
                             unsigned int rect_height, float* source_x, float* source_y) {
    float u = x - (float)rect_x, v = y - (float)rect_y;
    float w = (float)rect_width, h = (float)rect_height;
    float sw = (float)src_width, sh = (float)src_height;
    *source_x = SOURCE_X;
    *source_y = SOURCE_Y;
}

extern "C" __global__ void convert_scale(
    const unsigned char* src, unsigned char* dst,
    unsigned int src_width, unsigned int src_height,
    unsigned int src_offset0, unsigned int src_offset1, unsigned int src_offset2,
    unsigned int src_stride0, unsigned int src_stride1, unsigned int src_stride2,
    unsigned int dst_width, unsigned int dst_height,
    unsigned int dst_offset0, unsigned int dst_offset1, unsigned int dst_offset2,
    unsigned int dst_stride0, unsigned int dst_stride1, unsigned int dst_stride2,
    unsigned int rect_x, unsigned int rect_y, unsigned int rect_width, unsigned int rect_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= (int)dst_width || y >= (int)dst_height) {
        return;
    }
    Frame frame = {src, src_width, src_height, {src_offset0, src_offset1, src_offset2},
                   {src_stride0, src_stride1, src_stride2}};
    unsigned int offset[3] = {dst_offset0, dst_offset1, dst_offset2};
    unsigned int stride[3] = {dst_stride0, dst_stride1, dst_stride2};
    float centre_x = (float)x + 0.5f;
    float centre_y = (float)y + 0.5f;
    bool covered = inside(centre_x, centre_y, rect_x, rect_y, rect_width, rect_height);
    float source_x, source_y;
    source_point(centre_x, centre_y, src_width, src_height, rect_x, rect_y, rect_width,
                 rect_height, &source_x, &source_y);
#if OUT_YUV
    float luma = covered ? sample_out_yuv(frame, source_x, source_y).y : 0.0f;
    write_luma(dst + offset[0], stride[0], x, y, luma);
#if OUT_FORMAT == FMT_YUY2
    if ((x & 1) == 0) {
        float chroma_y = (float)y + 0.5f;
#else
    if ((x & 1) == 0 && (y & 1) == 0) {
        float chroma_y = (float)y + 1.0f;
#endif
        float chroma_x = (float)x + 1.0f;
        float pb = 0.0f, pr = 0.0f;
        if (inside(chroma_x, chroma_y, rect_x, rect_y, rect_width, rect_height)) {
            float chroma_source_x, chroma_source_y;
            source_point(chroma_x, chroma_y, src_width, src_height, rect_x, rect_y, rect_width,
                         rect_height, &chroma_source_x, &chroma_source_y);
            Yuv yuv = sample_out_yuv(frame, chroma_source_x, chroma_source_y);
            pb = yuv.pb;
            pr = yuv.pr;
        }
#if OUT_FORMAT == FMT_YUY2
        write_chroma(dst, offset, stride, x / 2, y, pb, pr);
#else
        write_chroma(dst, offset, stride, x / 2, y / 2, pb, pr);
#endif
    }
#else
    Color black = {0.0f, 0.0f, 0.0f, 1.0f};
    write_rgb(dst, offset, stride, x, y, covered ? sample_color(frame, source_x, source_y) : black);
#endif
}

extern "C" __global__ void blend_into_canvas(
    const unsigned char* src, float* canvas,
    unsigned int src_width, unsigned int src_height,
    unsigned int src_offset0, unsigned int src_offset1, unsigned int src_offset2,
    unsigned int src_stride0, unsigned int src_stride1, unsigned int src_stride2,
    unsigned int canvas_width, unsigned int canvas_height,
    int rect_x, int rect_y, unsigned int rect_width, unsigned int rect_height, float alpha,
    unsigned int source_operator) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= (int)canvas_width || y >= (int)canvas_height) {
        return;
    }
    float inside_x = (float)(x - rect_x) + 0.5f;
    float inside_y = (float)(y - rect_y) + 0.5f;
    if (inside_x <= 0.0f || inside_y <= 0.0f || inside_x >= (float)rect_width ||
        inside_y >= (float)rect_height) {
        return;
    }
    Frame frame = {src, src_width, src_height, {src_offset0, src_offset1, src_offset2},
                   {src_stride0, src_stride1, src_stride2}};
    Color color = sample_color(frame, inside_x * (float)src_width / (float)rect_width,
                               inside_y * (float)src_height / (float)rect_height);
    float a = clamp01(color.a) * alpha;
    float* pixel = canvas + 4 * ((unsigned long long)y * canvas_width + x);
    if (source_operator) {
        pixel[0] = color.r;
        pixel[1] = color.g;
        pixel[2] = color.b;
        pixel[3] = a;
        return;
    }
    pixel[0] = color.r * a + pixel[0] * (1.0f - a);
    pixel[1] = color.g * a + pixel[1] * (1.0f - a);
    pixel[2] = color.b * a + pixel[2] * (1.0f - a);
    pixel[3] = a + pixel[3] * (1.0f - a);
}
"#;

const BLOCK_SIZE: u32 = 16;

fn format_index(format: RawVideoFormat) -> Option<usize> {
    FORMATS.iter().position(|&(known, _)| known == format)
}

pub(crate) fn converts(format: RawVideoFormat) -> bool {
    format_index(format).is_some()
}

pub(crate) fn any_size_formats() -> impl Iterator<Item = Caps> {
    FORMATS.iter().map(|&(format, _)| Caps::RawVideo {
        format,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    })
}

pub(crate) fn narrow_to_converted_format(upstream_caps: &Caps) -> Result<Caps, G2gError> {
    any_size_formats()
        .find_map(|candidate| upstream_caps.intersect(&candidate).ok())
        .ok_or(G2gError::CapsMismatch)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Picture {
    format: RawVideoFormat,
    kernel_format: usize,
    width: u32,
    height: u32,
    colorimetry: Colorimetry,
}

impl Picture {
    fn of(caps: &Caps) -> Result<Self, G2gError> {
        match caps {
            Caps::RawVideo {
                format,
                width: Dim::Fixed(width),
                height: Dim::Fixed(height),
                colorimetry,
                ..
            } if *width > 0 && *height > 0 => Ok(Self {
                format: *format,
                kernel_format: format_index(*format).ok_or(G2gError::CapsMismatch)?,
                width: *width,
                height: *height,
                colorimetry: *colorimetry,
            }),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn tight_layout(&self) -> Result<PlaneLayout, G2gError> {
        PlaneLayout::tight(self.format, self.width, self.height).ok_or(G2gError::CapsMismatch)
    }

    fn defines(&self, side: &str) -> Vec<String> {
        let conversion = self.colorimetry.yuv_conversion();
        Vec::from([
            format!("-D{side}_FORMAT={}", self.kernel_format),
            format!("-D{side}_YUV={}", u8::from(carries_yuv(self.format))),
            format!("-D{side}_BIT_DEPTH={}", self.format.bit_depth()),
            format!("-D{side}_FULL_RANGE={}", u8::from(conversion.full_range)),
            format!("-D{side}_KR={:?}f", conversion.luma.kr),
            format!("-D{side}_KB={:?}f", conversion.luma.kb),
        ])
    }
}

#[derive(Debug, Clone, Copy)]
struct PictureRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

fn picture_rect(
    input: &Picture,
    output: &Picture,
    add_borders: bool,
    direction: Orientation,
) -> PictureRect {
    if !add_borders {
        return PictureRect {
            x: 0,
            y: 0,
            width: output.width,
            height: output.height,
        };
    }
    let (input_width, input_height) = match direction.swaps_dims() {
        true => (input.height, input.width),
        false => (input.width, input.height),
    };
    let scale = f64::min(
        f64::from(output.width) / f64::from(input_width),
        f64::from(output.height) / f64::from(input_height),
    );
    let subsampled = output.format.plane_rows(1, output.height) != Some(output.height);
    // a 4:2:0 output keeps the picture on whole chroma samples
    let even = |value: u32| if subsampled { value & !1 } else { value };
    let width = even((f64::from(input_width) * scale).round() as u32).clamp(1, output.width);
    let height = even((f64::from(input_height) * scale).round() as u32).clamp(1, output.height);
    PictureRect {
        x: even((output.width - width) / 2),
        y: even((output.height - height) / 2),
        width,
        height,
    }
}

// the source point for (u, v) inside a w x h output rect, read from a sw x sh source
fn source_point_expressions(direction: Orientation) -> (&'static str, &'static str) {
    match direction {
        Orientation::Identity => ("u * sw / w", "v * sh / h"),
        Orientation::Rotate90Cw => ("v * sw / h", "(w - u) * sh / w"),
        Orientation::Rotate180 => ("(w - u) * sw / w", "(h - v) * sh / h"),
        Orientation::Rotate90Ccw => ("(h - v) * sw / h", "u * sh / w"),
        Orientation::HorizontalMirror => ("(w - u) * sw / w", "v * sh / h"),
        Orientation::VerticalMirror => ("u * sw / w", "(h - v) * sh / h"),
        Orientation::Transpose => ("v * sw / h", "u * sh / w"),
        Orientation::Transverse => ("(h - v) * sw / h", "(w - u) * sh / w"),
    }
}

fn compile(
    device: &Arc<HipDevice>,
    input: &Picture,
    output: &Picture,
    direction: Orientation,
) -> Result<Arc<HipKernelModule>, G2gError> {
    let mut options: Vec<String> = FORMATS
        .iter()
        .enumerate()
        .map(|(index, (_, name))| format!("-DFMT_{name}={index}"))
        .collect();
    options.extend(input.defines("IN"));
    options.extend(output.defines("OUT"));
    let (source_x, source_y) = source_point_expressions(direction);
    options.push(format!("-DSOURCE_X={source_x}"));
    options.push(format!("-DSOURCE_Y={source_y}"));
    let (input_luma, output_luma) = (
        input.colorimetry.yuv_conversion().luma,
        output.colorimetry.yuv_conversion().luma,
    );
    let same_matrix =
        carries_yuv(input.format) && carries_yuv(output.format) && input_luma == output_luma;
    options.push(format!("-DSAME_MATRIX={}", u8::from(same_matrix)));
    let options: Vec<&str> = options.iter().map(String::as_str).collect();
    HipKernelModule::compile(device, KERNEL_SOURCE, &options)
}

fn source_planes(input: &OwnedHipBuffer, picture: &Picture) -> Result<[(u32, u32); 3], G2gError> {
    match input.plane_layout() {
        Some(layout) => kernel_planes(layout),
        None => kernel_planes(&picture.tight_layout()?),
    }
}

// the kernel takes three planes of (offset, stride), zero past the format's last plane
fn kernel_planes(layout: &PlaneLayout) -> Result<[(u32, u32); 3], G2gError> {
    let mut planes = [(0, 0); 3];
    for (plane, slot) in planes.iter_mut().enumerate() {
        if let Some(found) = layout.plane(plane) {
            *slot = (
                u32::try_from(found.offset).map_err(|_| G2gError::CapsMismatch)?,
                u32::try_from(found.stride).map_err(|_| G2gError::CapsMismatch)?,
            );
        }
    }
    Ok(planes)
}

#[derive(Debug)]
pub struct HipConverter {
    kernel: HipKernel,
    input: Picture,
    output: Picture,
    rect: PictureRect,
}

impl HipConverter {
    pub fn new(
        device: &Arc<HipDevice>,
        input: &Caps,
        output: &Caps,
        add_borders: bool,
        direction: Orientation,
    ) -> Result<Self, G2gError> {
        let input = Picture::of(input)?;
        let output = Picture::of(output)?;
        let module = compile(device, &input, &output, direction)?;
        Ok(Self {
            kernel: module.kernel("convert_scale")?,
            rect: picture_rect(&input, &output, add_borders, direction),
            input,
            output,
        })
    }

    pub fn output_len(&self) -> Result<usize, G2gError> {
        self.output
            .format
            .unpadded_frame_bytes(self.output.width, self.output.height)
            .and_then(|len| usize::try_from(len).ok())
            .ok_or(G2gError::CapsMismatch)
    }

    pub fn convert(
        &self,
        stream: &HipStream,
        input: &OwnedHipBuffer,
        output: &mut HipMemory,
    ) -> Result<(), G2gError> {
        if output.size() < self.output_len()? {
            return Err(G2gError::CapsMismatch);
        }
        let source = source_planes(input, &self.input)?;
        let destination = kernel_planes(&self.output.tight_layout()?)?;
        let (width, height) = (self.output.width, self.output.height);
        stream.launch(
            &self.kernel,
            [width.div_ceil(BLOCK_SIZE), height.div_ceil(BLOCK_SIZE), 1],
            [BLOCK_SIZE, BLOCK_SIZE, 1],
            &[
                KernelArgument::Read(input),
                KernelArgument::Write(output),
                KernelArgument::U32(self.input.width),
                KernelArgument::U32(self.input.height),
                KernelArgument::U32(source[0].0),
                KernelArgument::U32(source[1].0),
                KernelArgument::U32(source[2].0),
                KernelArgument::U32(source[0].1),
                KernelArgument::U32(source[1].1),
                KernelArgument::U32(source[2].1),
                KernelArgument::U32(width),
                KernelArgument::U32(height),
                KernelArgument::U32(destination[0].0),
                KernelArgument::U32(destination[1].0),
                KernelArgument::U32(destination[2].0),
                KernelArgument::U32(destination[0].1),
                KernelArgument::U32(destination[1].1),
                KernelArgument::U32(destination[2].1),
                KernelArgument::U32(self.rect.x),
                KernelArgument::U32(self.rect.y),
                KernelArgument::U32(self.rect.width),
                KernelArgument::U32(self.rect.height),
            ],
        )
    }
}

#[derive(Debug)]
pub(crate) struct HipBlender {
    kernel: HipKernel,
    input: Picture,
}

impl HipBlender {
    pub(crate) fn new(device: &Arc<HipDevice>, input: &Caps) -> Result<Self, G2gError> {
        let input = Picture::of(input)?;
        let canvas = Picture::of(&Caps::RawVideo {
            format: RawVideoFormat::RgbaF32,
            width: Dim::Fixed(1),
            height: Dim::Fixed(1),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        })?;
        let module = compile(device, &input, &canvas, Orientation::Identity)?;
        Ok(Self {
            kernel: module.kernel("blend_into_canvas")?,
            input,
        })
    }

    // `canvas` holds `canvas_width` x `canvas_height` RGBA pixels of four f32 each
    pub(crate) fn blend(
        &self,
        stream: &HipStream,
        input: &OwnedHipBuffer,
        canvas: &mut HipMemory,
        (canvas_width, canvas_height): (u32, u32),
        pad: &CompositorPad,
    ) -> Result<(), G2gError> {
        let source = source_planes(input, &self.input)?;
        let rect = pad.dest_rect(self.input.width, self.input.height);
        stream.launch(
            &self.kernel,
            [
                canvas_width.div_ceil(BLOCK_SIZE),
                canvas_height.div_ceil(BLOCK_SIZE),
                1,
            ],
            [BLOCK_SIZE, BLOCK_SIZE, 1],
            &[
                KernelArgument::Read(input),
                KernelArgument::Write(canvas),
                KernelArgument::U32(self.input.width),
                KernelArgument::U32(self.input.height),
                KernelArgument::U32(source[0].0),
                KernelArgument::U32(source[1].0),
                KernelArgument::U32(source[2].0),
                KernelArgument::U32(source[0].1),
                KernelArgument::U32(source[1].1),
                KernelArgument::U32(source[2].1),
                KernelArgument::U32(canvas_width),
                KernelArgument::U32(canvas_height),
                // the kernel reads these two as signed ints
                KernelArgument::U32(rect.x as u32),
                KernelArgument::U32(rect.y as u32),
                KernelArgument::U32(rect.width),
                KernelArgument::U32(rect.height),
                KernelArgument::F32(f32::from(pad.alpha) / 255.0),
                KernelArgument::U32(u32::from(pad.operator == CompositorOperator::Source)),
            ],
        )
    }
}

#[derive(Debug)]
struct ConvertGpu {
    device: Arc<HipDevice>,
    stream: HipStream,
    converter: HipConverter,
    pool: HipMemoryPool,
}

#[derive(Debug)]
struct HipConvertVariant {
    metadata: (&'static str, &'static str, &'static str),
    properties: &'static [PropertySpec],
    changes_format: bool,
    changes_size: bool,
}

const ADD_BORDERS_PROP: PropertySpec = PropertySpec::new(
    "add-borders",
    PropKind::Bool,
    "Add borders if necessary to keep the display aspect ratio",
)
.with_default("true");

static CONVERT_SCALE: HipConvertVariant = HipConvertVariant {
    metadata: (
        "HIP converter and scaler",
        "Filter/Converter/Video/Scaler/Hardware",
        "Converts the pixel format, resizes and rotates HIP frames on the GPU",
    ),
    properties: &[
        HIP_DEVICE_ID_PROP,
        HIP_VENDOR_PROP,
        ADD_BORDERS_PROP,
        PropertySpec::new(
            "video-direction",
            PropKind::Str,
            "Video direction: rotation and flipping",
        )
        .with_enum_values("identity | 90r | 180 | 90l | horiz | vert | ul-lr | ur-ll")
        .with_default("identity"),
    ],
    changes_format: true,
    changes_size: true,
};

static CONVERT: HipConvertVariant = HipConvertVariant {
    metadata: (
        "HIP colorspace converter",
        "Filter/Converter/Video/Hardware",
        "Converts the pixel format of HIP frames on the GPU",
    ),
    properties: &[HIP_DEVICE_ID_PROP, HIP_VENDOR_PROP],
    changes_format: true,
    changes_size: false,
};

static SCALE: HipConvertVariant = HipConvertVariant {
    metadata: (
        "HIP video scaler",
        "Filter/Converter/Video/Scaler/Hardware",
        "Resizes HIP frames on the GPU",
    ),
    properties: &[HIP_DEVICE_ID_PROP, HIP_VENDOR_PROP, ADD_BORDERS_PROP],
    changes_format: false,
    changes_size: true,
};

// the nicknames of GStreamer's GstVideoOrientationMethod
fn video_direction_from_str(name: &str) -> Option<Orientation> {
    match name {
        "identity" => Some(Orientation::Identity),
        "90r" => Some(Orientation::Rotate90Cw),
        "180" => Some(Orientation::Rotate180),
        "90l" => Some(Orientation::Rotate90Ccw),
        "horiz" => Some(Orientation::HorizontalMirror),
        "vert" => Some(Orientation::VerticalMirror),
        "ul-lr" => Some(Orientation::Transpose),
        "ur-ll" => Some(Orientation::Transverse),
        _ => None,
    }
}

fn video_direction_name(direction: Orientation) -> &'static str {
    match direction {
        Orientation::Identity => "identity",
        Orientation::Rotate90Cw => "90r",
        Orientation::Rotate180 => "180",
        Orientation::Rotate90Ccw => "90l",
        Orientation::HorizontalMirror => "horiz",
        Orientation::VerticalMirror => "vert",
        Orientation::Transpose => "ul-lr",
        Orientation::Transverse => "ur-ll",
    }
}

fn swap_dims(caps: &Caps) -> Caps {
    match caps {
        Caps::RawVideo {
            format,
            width,
            height,
            framerate,
            interlace,
            colorimetry,
            pixel_aspect: _,
        } => Caps::RawVideo {
            format: *format,
            width: height.clone(),
            height: width.clone(),
            framerate: framerate.clone(),
            interlace: *interlace,
            colorimetry: *colorimetry,
            pixel_aspect: g2g_core::PixelAspect::Any,
        },
        other => other.clone(),
    }
}

#[derive(Debug)]
pub struct HipConvertScale {
    variant: &'static HipConvertVariant,
    device: HipDeviceChoice,
    add_borders: bool,
    direction: Orientation,
    configured: bool,
    input_caps: Option<Caps>,
    output_caps: Option<Caps>,
    last_caps: Option<Caps>,
    gpu: Option<ConvertGpu>,
}

impl Default for HipConvertScale {
    fn default() -> Self {
        Self::of(&CONVERT_SCALE)
    }
}

impl HipConvertScale {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn convert_only() -> Self {
        Self::of(&CONVERT)
    }

    pub fn scale_only() -> Self {
        Self::of(&SCALE)
    }

    fn of(variant: &'static HipConvertVariant) -> Self {
        Self {
            variant,
            device: HipDeviceChoice::default(),
            add_borders: true,
            direction: Orientation::Identity,
            configured: false,
            input_caps: None,
            output_caps: None,
            last_caps: None,
            gpu: None,
        }
    }

    fn gpu_for(&mut self, buffer: &OwnedHipBuffer) -> Result<&mut ConvertGpu, G2gError> {
        let on_device = |gpu: &ConvertGpu| {
            gpu.device.vendor() == buffer.vendor && gpu.device.ordinal() == buffer.device_ordinal
        };
        let gpu = match self.gpu.take() {
            Some(gpu) if on_device(&gpu) => gpu,
            _ => {
                let input = self.input_caps.as_ref().ok_or(G2gError::NotConfigured)?;
                let output = self.output_caps.as_ref().ok_or(G2gError::NotConfigured)?;
                let device = HipDevice::open(buffer.vendor, buffer.device_ordinal)?;
                let converter =
                    HipConverter::new(&device, input, output, self.add_borders, self.direction)?;
                ConvertGpu {
                    stream: HipStream::new(&device)?,
                    pool: HipMemoryPool::new(&device, converter.output_len()?),
                    converter,
                    device,
                }
            }
        };
        Ok(self.gpu.insert(gpu))
    }

    fn convert(&mut self, buffer: &OwnedHipBuffer) -> Result<OwnedHipBuffer, G2gError> {
        if !self.device.holds(buffer) {
            g2g_error!(
                Target::category(short_type_name::<Self>()),
                "runs on {:?}, and a frame arrived on {:?} device {}",
                self.device,
                buffer.vendor,
                buffer.device_ordinal
            );
            return Err(G2gError::UnsupportedDomain);
        }
        let gpu = self.gpu_for(buffer)?;
        let mut memory = gpu.pool.acquire()?;
        gpu.converter.convert(&gpu.stream, buffer, &mut memory)?;
        let ready = HipEvent::new(&gpu.device)?;
        gpu.stream.record(&ready)?;
        let len = gpu.converter.output_len()?;
        memory.into_buffer(ready, len)
    }

    fn passthrough(&self) -> bool {
        let picture = |caps: &Option<Caps>| caps.as_ref().and_then(|caps| Picture::of(caps).ok());
        let (input, output) = (picture(&self.input_caps), picture(&self.output_caps));
        let same_picture = input.is_some_and(|input| {
            output.is_some_and(|output| {
                (input.format, input.width, input.height)
                    == (output.format, output.width, output.height)
            })
        });
        same_picture && self.direction == Orientation::Identity
    }

    fn has_property(&self, name: &str) -> bool {
        self.variant.properties.iter().any(|spec| spec.name == name)
    }
}

impl AsyncElement for HipConvertScale {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        let (long_name, classification, description) = self.variant.metadata;
        ElementMetadata::new(long_name, classification, description, crate::AARON_BOXER)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        narrow_to_converted_format(upstream_caps)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let formats: Vec<RawVideoFormat> = FORMATS.iter().map(|&(format, _)| format).collect();
        let any_size = |shape: RawVideoShape| {
            let range = Dim::Range {
                min: 1,
                max: u32::from(u16::MAX),
            };
            shape
                .with_width(FieldTransform::Fixed(range.clone()))
                .with_height(FieldTransform::Fixed(range))
        };
        let mut converted: Vec<RawVideoShape> = Vec::from([RawVideoShape::PASSTHROUGH]);
        if self.variant.changes_format {
            converted.extend(formats.iter().map(|&format| {
                RawVideoShape::PASSTHROUGH.with_format(FieldTransform::Fixed(format))
            }));
        }
        let shapes = converted
            .into_iter()
            .flat_map(|shape| {
                let resized = self.variant.changes_size.then(|| any_size(shape.clone()));
                core::iter::once(shape).chain(resized)
            })
            .collect();
        let transform = CapsTransform::RawVideo {
            accept: formats.clone(),
            produce: formats,
            shapes,
        };
        if !self.direction.swaps_dims() {
            return CapsConstraint::DerivedFields(transform);
        }
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            transform.derive(&swap_dims(input))
        }))
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::Hip)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::Hip
    }

    fn gpu_device(&self) -> Option<GpuDevice> {
        self.device.declared()
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Picture::of(absolute_caps)?;
        self.input_caps = Some(absolute_caps.clone());
        self.gpu = None;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        Picture::of(output_caps)?;
        self.output_caps = Some(output_caps.clone());
        self.gpu = None;
        Ok(())
    }

    fn properties(&self) -> &'static [PropertySpec] {
        self.variant.properties
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if !self.has_property(name) {
            return Err(PropError::Unknown);
        }
        if let Some(applied) = self.device.set_property(name, &value) {
            self.gpu = None;
            return applied;
        }
        match name {
            "add-borders" => self.add_borders = value.as_bool().ok_or(PropError::Type)?,
            "video-direction" => {
                let name = value.as_str().ok_or(PropError::Type)?;
                self.direction = video_direction_from_str(name).ok_or(PropError::Value)?;
            }
            _ => return Err(PropError::Unknown),
        }
        self.gpu = None;
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if !self.has_property(name) {
            return None;
        }
        if let Some(value) = self.device.get_property(name) {
            return Some(value);
        }
        match name {
            "add-borders" => Some(PropValue::Bool(self.add_borders)),
            "video-direction" => Some(PropValue::Str(video_direction_name(self.direction).into())),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let output_caps = self.output_caps.clone().ok_or(G2gError::NotConfigured)?;
                    if self.last_caps.as_ref() != Some(&output_caps) {
                        out.push(PipelinePacket::CapsChanged(output_caps.clone()))
                            .await?;
                        self.last_caps = Some(output_caps);
                    }
                    if self.passthrough() {
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                        return Ok(());
                    }
                    let MemoryDomain::Hip(buffer) = &frame.domain else {
                        g2g_error!(
                            Target::category(short_type_name::<Self>()),
                            "converts Hip frames, and the link delivered a {:?} frame",
                            frame.domain.kind()
                        );
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let converted = self.convert(buffer)?;
                    out.push(PipelinePacket::DataFrame(Frame {
                        domain: MemoryDomain::Hip(converted),
                        timing: frame.timing,
                        sequence: frame.sequence,
                        meta: Default::default(),
                    }))
                    .await?;
                }
                // the runner sends this element's own output caps, which go downstream as they are
                PipelinePacket::CapsChanged(caps) => {
                    out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                    self.last_caps = Some(caps);
                }
                PipelinePacket::Flush => {
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // the runner forwards end-of-stream after `process` returns
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}
