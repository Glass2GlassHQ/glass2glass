//! M1041: `TensorBatcher` stacks a round by concatenating each slot's bytes on
//! the CPU, so it declares system memory and the allocation cascade downloads a
//! GPU-resident tensor feeding it rather than letting `process` reject it.

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::{Caps, MultiInputElement, TensorDType, TensorLayout, TensorShape};
use g2g_ml::batcher::TensorBatcher;

#[test]
fn batcher_takes_system_frames() {
    let slot = Caps::Tensor {
        dtype: TensorDType::U8,
        shape: TensorShape::new([1, 4]),
        layout: TensorLayout::Nchw,
    };
    let batcher = TensorBatcher::new(2, slot).unwrap();
    assert_eq!(
        batcher.input_domains(),
        DomainSet::only(MemoryDomainKind::System)
    );
}
