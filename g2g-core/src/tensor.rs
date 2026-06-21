//! Strided tensor *view* over a byte buffer: the zero-copy substrate for raw
//! numeric media (M180). A [`TensorView`] describes how to read a flat `[u8]`
//! backing as an n-dimensional array, by NumPy-style byte strides plus a byte
//! offset, so a layout-preserving transform (flip, transpose, crop, channel
//! reorder) is expressed as a *new view over the same bytes* rather than a
//! copy. The bytes themselves live in a [`MemoryDomain`](crate::memory), eg the
//! shared-CPU [`SystemView`](crate::memory::SystemView); this type is pure
//! metadata.
//!
//! Why a view and not just `Caps::Tensor`: `Caps` is the *negotiated logical*
//! shape of a link; a `TensorView` is the *physical* layout of one concrete
//! buffer, including the non-contiguous strides a copy would otherwise
//! materialize. Strides are in bytes (NumPy's convention, not PyTorch's
//! element-count), so a single field describes packed pixels, sub-byte plane
//! offsets, and reversed axes uniformly over a `[u8]` allocation.
//!
//! Planar / subsampled video (NV12, I420) is deliberately out of scope for a
//! single `TensorView`: its planes have different resolutions, so a frame is a
//! *list* of views, not one. This type handles the packed and audio cases; the
//! plane-list lands with the first planar consumer.

use alloc::boxed::Box;
use alloc::vec;

use crate::caps::TensorDType;

/// Maximum tensor rank a [`TensorView`] can describe. Covers the media cases,
/// packed video `[H, W, C]` (3), planar audio `[frames, channels]` (2), ML
/// `[N, C, H, W]` (4), with headroom. Fixed so the view is `Copy` and needs no
/// heap, keeping the `no_std` / RTOS baseline allocation-free.
pub const MAX_TENSOR_RANK: usize = 6;

/// An n-dimensional strided view over a byte buffer. See the module docs. The
/// view owns no memory; it indexes a `&[u8]` supplied at read time (the backing
/// of the [`MemoryDomain`](crate::memory) it travels with).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorView {
    dtype: TensorDType,
    /// Number of valid entries in `shape` / `strides` (the tensor rank).
    rank: u8,
    /// Logical extent of each axis; `shape[..rank]` is valid.
    shape: [u32; MAX_TENSOR_RANK],
    /// Byte stride of each axis (bytes to step one element along it). Signed so
    /// a reversed axis (a flip) is a negative stride. `strides[..rank]` valid.
    strides: [isize; MAX_TENSOR_RANK],
    /// Byte offset of logical element `[0, 0, ...]` from the start of the
    /// backing buffer. A reversed axis moves this to the far end of that axis.
    offset: usize,
}

impl TensorView {
    /// A dense, row-major (C-order) view of `shape` at offset 0: the natural
    /// layout of a freshly-allocated contiguous buffer. The innermost (last)
    /// axis is contiguous (stride = element size).
    ///
    /// Panics if `shape` is empty or exceeds [`MAX_TENSOR_RANK`].
    pub fn contiguous(dtype: TensorDType, shape: &[u32]) -> Self {
        assert!(
            !shape.is_empty() && shape.len() <= MAX_TENSOR_RANK,
            "tensor rank must be 1..={MAX_TENSOR_RANK}"
        );
        let rank = shape.len();
        let mut s = [0u32; MAX_TENSOR_RANK];
        let mut st = [0isize; MAX_TENSOR_RANK];
        // Row-major: innermost stride = element size, each outer stride is the
        // inner stride times the inner extent.
        let mut acc = dtype.size() as isize;
        for i in (0..rank).rev() {
            s[i] = shape[i];
            st[i] = acc;
            acc *= shape[i] as isize;
        }
        Self { dtype, rank: rank as u8, shape: s, strides: st, offset: 0 }
    }

    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    pub fn rank(&self) -> usize {
        self.rank as usize
    }

    pub fn shape(&self) -> &[u32] {
        &self.shape[..self.rank as usize]
    }

    pub fn strides(&self) -> &[isize] {
        &self.strides[..self.rank as usize]
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Total number of logical elements (the product of the extents).
    pub fn num_elements(&self) -> usize {
        self.shape().iter().map(|&d| d as usize).product()
    }

    /// Bytes a dense row-major materialization of this view occupies.
    pub fn materialized_len(&self) -> usize {
        self.num_elements() * self.dtype.size()
    }

    /// True if the view is dense row-major at offset 0: a consumer may read the
    /// backing bytes directly, ignoring strides. A flipped / transposed view is
    /// not contiguous and must be honored through the strides (or materialized).
    pub fn is_contiguous(&self) -> bool {
        let mut acc = self.dtype.size() as isize;
        for i in (0..self.rank as usize).rev() {
            if self.strides[i] != acc {
                return false;
            }
            acc *= self.shape[i] as isize;
        }
        self.offset == 0
    }

    /// Reverse the iteration order along `axis` (a mirror / flip). Zero-copy:
    /// the offset jumps to the far end of the axis and the axis stride negates.
    /// Panics if `axis >= rank`.
    pub fn reversed_axis(mut self, axis: usize) -> Self {
        assert!(axis < self.rank as usize, "axis out of range");
        let extent = self.shape[axis] as isize;
        if extent > 0 {
            self.offset = (self.offset as isize + (extent - 1) * self.strides[axis]) as usize;
            self.strides[axis] = -self.strides[axis];
        }
        self
    }

    /// Swap two axes (a transpose). Zero-copy: the shape and stride entries
    /// swap. A 90-degree image rotation is a transpose followed by a
    /// [`reversed_axis`](Self::reversed_axis). Panics if either axis is out of
    /// range.
    pub fn transposed(mut self, a: usize, b: usize) -> Self {
        let r = self.rank as usize;
        assert!(a < r && b < r, "transpose axis out of range");
        self.shape.swap(a, b);
        self.strides.swap(a, b);
        self
    }

    /// Copy the logically-ordered elements into a fresh dense row-major buffer.
    /// This is the *one* copy a strided chain pays, and only when a consumer
    /// needs contiguous bytes (eg writing a file); a stride-aware consumer skips
    /// it. `backing` is the full byte buffer the view indexes into.
    ///
    /// Panics if a computed source range falls outside `backing` (a view whose
    /// extents/strides don't fit the buffer it was paired with).
    pub fn materialize(&self, backing: &[u8]) -> Box<[u8]> {
        let elem = self.dtype.size();
        let rank = self.rank as usize;
        let total = self.num_elements();
        let mut out = vec![0u8; total * elem];
        // Row-major odometer over the logical index space: gather each element
        // from `backing` at offset + sum(index[i] * strides[i]), last axis
        // fastest. Bounded by `total` so it terminates even if a stride is 0.
        let mut index = [0u32; MAX_TENSOR_RANK];
        let mut dst = 0usize;
        for _ in 0..total {
            let mut src = self.offset as isize;
            for (idx, stride) in index[..rank].iter().zip(&self.strides[..rank]) {
                src += *idx as isize * *stride;
            }
            let src = src as usize;
            out[dst..dst + elem].copy_from_slice(&backing[src..src + elem]);
            dst += elem;
            for axis in (0..rank).rev() {
                index[axis] += 1;
                if index[axis] < self.shape[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
        out.into_boxed_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use alloc::vec::Vec;

    #[test]
    fn contiguous_strides_and_materialize_roundtrip() {
        // 2x3 single-channel u8, row-major 0..6.
        let v = TensorView::contiguous(TensorDType::U8, &[2, 3]);
        assert_eq!(v.shape(), &[2, 3]);
        assert_eq!(v.strides(), &[3, 1]);
        assert!(v.is_contiguous());
        assert_eq!(v.materialized_len(), 6);
        let backing: Vec<u8> = (0..6).collect();
        assert_eq!(&*v.materialize(&backing), &[0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn contiguous_strides_in_bytes_for_wide_dtype() {
        // f32 [2,2]: innermost stride is 4 bytes, outer is 8.
        let v = TensorView::contiguous(TensorDType::F32, &[2, 2]);
        assert_eq!(v.strides(), &[8, 4]);
        assert_eq!(v.materialized_len(), 16);
    }

    #[test]
    fn reversed_axis_is_a_flip_and_aliases_the_backing() {
        // [[0,1,2],[3,4,5]] as 2x3 u8.
        let bytes: Box<[u8]> = (0u8..6).collect::<Vec<_>>().into_boxed_slice();
        let backing: Arc<[u8]> = Arc::from(bytes);
        let v = TensorView::contiguous(TensorDType::U8, &[2, 3]);

        // reverse columns (horizontal mirror): [[2,1,0],[5,4,3]]
        let h = v.reversed_axis(1);
        assert!(!h.is_contiguous());
        assert_eq!(&*h.materialize(&backing), &[2, 1, 0, 5, 4, 3]);

        // reverse rows (vertical mirror): [[3,4,5],[0,1,2]]
        let vert = v.reversed_axis(0);
        assert_eq!(&*vert.materialize(&backing), &[3, 4, 5, 0, 1, 2]);

        // rotate-180 = reverse both axes: full reversal 5..=0.
        let r = v.reversed_axis(0).reversed_axis(1);
        assert_eq!(&*r.materialize(&backing), &[5, 4, 3, 2, 1, 0]);

        // The flip moved zero bytes: a SystemView built from the flipped view
        // still points at the same allocation as the original.
        use crate::memory::SystemView;
        let original = SystemView::new(backing.clone(), v);
        let flipped = SystemView::new(backing.clone(), r);
        assert!(Arc::ptr_eq(original.backing(), flipped.backing()));
        assert_eq!(&*flipped.materialize(), &[5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn transpose_swaps_axes_and_extents() {
        // 2x3 [[0,1,2],[3,4,5]] transposed -> 3x2 [[0,3],[1,4],[2,5]].
        let backing: Vec<u8> = (0..6).collect();
        let t = TensorView::contiguous(TensorDType::U8, &[2, 3]).transposed(0, 1);
        assert_eq!(t.shape(), &[3, 2]);
        assert!(!t.is_contiguous());
        assert_eq!(&*t.materialize(&backing), &[0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn rotate90_cw_is_transpose_then_reverse_columns() {
        // 2x3 image -> rotate 90 CW -> 3x2. out[i,j] = in[H-1-j, i].
        // in = [[0,1,2],[3,4,5]]; CW -> [[3,0],[4,1],[5,2]].
        let backing: Vec<u8> = (0..6).collect();
        let v = TensorView::contiguous(TensorDType::U8, &[2, 3]);
        let cw = v.transposed(0, 1).reversed_axis(1);
        assert_eq!(cw.shape(), &[3, 2]);
        assert_eq!(&*cw.materialize(&backing), &[3, 0, 4, 1, 5, 2]);
    }

    #[test]
    fn rgba_pixels_stay_intact_under_flip() {
        // 1x2 RGBA: pixel0 = [0,1,2,3], pixel1 = [4,5,6,7]. Horizontal mirror
        // (reverse axis 1) swaps the two pixels but keeps each pixel's 4 bytes
        // in order, because the channel axis (2) is untouched.
        let backing: Vec<u8> = (0..8).collect();
        let v = TensorView::contiguous(TensorDType::U8, &[1, 2, 4]);
        let m = v.reversed_axis(1);
        assert_eq!(&*m.materialize(&backing), &[4, 5, 6, 7, 0, 1, 2, 3]);
    }
}
