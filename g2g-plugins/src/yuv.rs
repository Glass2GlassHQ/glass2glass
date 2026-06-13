//! Pure chroma-resampling helpers for the decode elements. Kept out of the
//! OS-gated decoder modules (`ffmpegdec`) so the resampling math is
//! unit-testable on any host; the ffmpeg frame plumbing that feeds it is
//! Linux-only.

use alloc::vec;
use alloc::vec::Vec;

/// Downsample a full-resolution `w x h` 8-bit chroma plane (4:4:4) to 4:2:0
/// (`ceil(w/2) x ceil(h/2)`) by a 2x2 box average. `pitch` is the source row
/// stride in bytes (`>= w`). Used to accept a YUV444P decoder frame on the
/// 4:2:0 output path; the reduction is lossy in chroma resolution. Returns a
/// tightly-packed `cw * ch` plane (row stride `cw`).
pub(crate) fn downsample_chroma_420(src: &[u8], pitch: usize, w: usize, h: usize) -> Vec<u8> {
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let mut out = vec![0u8; cw * ch];
    for oy in 0..ch {
        let y0 = 2 * oy;
        let y1 = (y0 + 1).min(h - 1); // clamp the bottom row for odd heights
        for ox in 0..cw {
            let x0 = 2 * ox;
            let x1 = (x0 + 1).min(w - 1); // clamp the right column for odd widths
            let sum = src[y0 * pitch + x0] as u32
                + src[y0 * pitch + x1] as u32
                + src[y1 * pitch + x0] as u32
                + src[y1 * pitch + x1] as u32;
            // round-to-nearest (ties up); max (1020+2)/4 = 255.
            out[oy * cw + ox] = ((sum + 2) / 4) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_block_averages_each_2x2() {
        // 2x2 plane, no padding -> a single output sample = mean of all four.
        let src = [10u8, 20, 30, 40];
        let out = downsample_chroma_420(&src, 2, 2, 2);
        assert_eq!(out, vec![25]); // (10+20+30+40+2)/4 = 25
    }

    #[test]
    fn four_by_two_yields_two_samples() {
        // 4x2: two independent 2x2 blocks across the width.
        let src = [0u8, 0, 100, 100, 0, 0, 100, 100];
        let out = downsample_chroma_420(&src, 4, 4, 2);
        assert_eq!(out, vec![0, 100]);
    }

    #[test]
    fn honours_source_pitch_padding() {
        // 2x2 visible with a padded stride of 4: padding bytes must be ignored.
        let src = [10u8, 20, 0xFF, 0xFF, 30, 40, 0xFF, 0xFF];
        let out = downsample_chroma_420(&src, 4, 2, 2);
        assert_eq!(out, vec![25], "stride padding must not enter the average");
    }

    #[test]
    fn odd_dimensions_clamp_edge_samples() {
        // 3x3 -> 2x2 output. The right column / bottom row clamp to the edge
        // sample, so each output averages the 2x2 it can reach.
        let src = [
            10u8, 20, 60, // row 0
            30, 40, 80, // row 1
            90, 100, 200, // row 2
        ];
        let out = downsample_chroma_420(&src, 3, 3, 3);
        // (0,0): mean(10,20,30,40)=25
        // (1,0): x clamps to col 2 -> mean(60,60,80,80)=70
        // (0,1): y clamps to row 2 -> mean(30,40? no) ... mean(30,40 from row1?) :
        //   y0=2,y1=min(3,2)=2 -> rows 2,2; x0=0,x1=1 -> mean(90,100,90,100)=95
        // (1,1): rows 2,2 cols 2,2 -> mean(200)=200
        assert_eq!(out, vec![25, 70, 95, 200]);
    }
}
