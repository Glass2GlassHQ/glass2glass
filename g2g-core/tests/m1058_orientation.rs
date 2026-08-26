//! M1058: the `Orientation` descriptor a display sink absorbs.
//!
//! Three properties matter: `Orientation` really is the dihedral group of the
//! square (so composing two flips gives the one turn they add up to, and the
//! runtime can reason about a chain of them), `OrientationMeta` round-trips
//! through a frame's meta set, and it drops under a crop, whose rectangle is
//! chosen in the coordinates the descriptor has not been applied to yet.
//!
//! The group table needs no features; the meta needs the real `FrameMetaSet`.

use g2g_core::meta::Orientation;

/// Every member, in the order the property strings list them.
const ALL: [Orientation; 8] = [
    Orientation::Identity,
    Orientation::Rotate90Cw,
    Orientation::Rotate180,
    Orientation::Rotate90Ccw,
    Orientation::HorizontalMirror,
    Orientation::VerticalMirror,
    Orientation::Transpose,
    Orientation::Transverse,
];

/// Where the pixel at `(x, y)` of a `w` by `h` picture lands once `orientation`
/// is applied. Written out per member from the geometry, independently of the
/// mirror-then-rotate factoring `Orientation` composes with, so this is a real
/// check on that factoring rather than a restatement of it.
fn map_pixel(orientation: Orientation, x: u32, y: u32, w: u32, h: u32) -> (u32, u32) {
    match orientation {
        Orientation::Identity => (x, y),
        Orientation::Rotate90Cw => (h - 1 - y, x),
        Orientation::Rotate180 => (w - 1 - x, h - 1 - y),
        Orientation::Rotate90Ccw => (y, w - 1 - x),
        Orientation::HorizontalMirror => (w - 1 - x, y),
        Orientation::VerticalMirror => (x, h - 1 - y),
        Orientation::Transpose => (y, x),
        Orientation::Transverse => (h - 1 - y, w - 1 - x),
    }
}

/// Dims after `orientation`, from `swaps_dims`.
fn map_dims(orientation: Orientation, w: u32, h: u32) -> (u32, u32) {
    if orientation.swaps_dims() {
        (h, w)
    } else {
        (w, h)
    }
}

const W: u32 = 5;
const H: u32 = 3;

#[test]
fn compose_matches_applying_both_turns_in_order() {
    for first in ALL {
        for then in ALL {
            let composed = first.compose(then);
            let (mid_w, mid_h) = map_dims(first, W, H);
            assert_eq!(
                map_dims(composed, W, H),
                map_dims(then, mid_w, mid_h),
                "{first:?} then {then:?} must agree on the output dims"
            );
            for y in 0..H {
                for x in 0..W {
                    let (mx, my) = map_pixel(first, x, y, W, H);
                    let expected = map_pixel(then, mx, my, mid_w, mid_h);
                    assert_eq!(
                        map_pixel(composed, x, y, W, H),
                        expected,
                        "{first:?} then {then:?} at ({x},{y})"
                    );
                }
            }
        }
    }
}

#[test]
fn composition_is_closed_over_the_eight_members() {
    for first in ALL {
        for then in ALL {
            assert!(
                ALL.contains(&first.compose(then)),
                "{first:?} then {then:?} left the group"
            );
        }
    }
}

#[test]
fn every_member_composed_with_its_inverse_is_identity() {
    for orientation in ALL {
        assert_eq!(
            orientation.compose(orientation.inverse()),
            Orientation::Identity,
            "{orientation:?} then its inverse"
        );
        assert_eq!(
            orientation.inverse().compose(orientation),
            Orientation::Identity,
            "the inverse of {orientation:?} then {orientation:?}"
        );
    }
}

#[test]
fn identity_is_the_unit_and_only_four_members_swap_dims() {
    for orientation in ALL {
        assert_eq!(orientation.compose(Orientation::Identity), orientation);
        assert_eq!(Orientation::Identity.compose(orientation), orientation);
    }
    assert_eq!(
        ALL.iter().filter(|o| o.swaps_dims()).count(),
        4,
        "the two quarter turns and the two diagonals"
    );
    for orientation in [
        Orientation::Rotate90Cw,
        Orientation::Rotate90Ccw,
        Orientation::Transpose,
        Orientation::Transverse,
    ] {
        assert!(orientation.swaps_dims(), "{orientation:?}");
    }
}

#[cfg(feature = "metadata")]
mod meta {
    use super::*;
    use g2g_core::frame::{Frame, FrameTiming};
    use g2g_core::memory::SystemSlice;
    use g2g_core::meta::{FrameMetaSet, OrientationMeta, Propagation, Transform};
    use g2g_core::MemoryDomain;

    fn frame_with(orientation: Orientation) -> Frame {
        let mut meta = FrameMetaSet::default();
        meta.attach(OrientationMeta { orientation });
        Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
            timing: FrameTiming::default(),
            sequence: 0,
            meta,
        }
    }

    #[test]
    fn attaches_to_a_frame_and_reads_back() {
        let frame = frame_with(Orientation::Transverse);
        assert_eq!(
            frame.meta.get::<OrientationMeta>().map(|m| m.orientation),
            Some(Orientation::Transverse)
        );
    }

    #[test]
    fn attaching_again_replaces_the_turn() {
        let mut frame = frame_with(Orientation::Rotate90Cw);
        frame.meta.attach(OrientationMeta {
            orientation: Orientation::Rotate180,
        });
        assert_eq!(
            frame.meta.get::<OrientationMeta>().map(|m| m.orientation),
            Some(Orientation::Rotate180),
            "one turn per frame, not a list"
        );
    }

    /// A scale or a colour convert keeps every row a row, so the turn still
    /// applies; a crop picks its rectangle in the stored coordinates, so the
    /// picture that comes out is not the one the turn described.
    #[test]
    fn survives_a_resample_and_drops_under_a_crop() {
        let meta = OrientationMeta {
            orientation: Orientation::Rotate90Cw,
        };
        for transform in [Transform::Copy, Transform::Scale, Transform::Encode] {
            assert_eq!(
                g2g_core::meta::FrameMeta::propagate(&meta, transform),
                Propagation::Keep,
                "{transform:?}"
            );
        }
        assert_eq!(
            g2g_core::meta::FrameMeta::propagate(&meta, Transform::Crop),
            Propagation::Drop
        );
    }

    #[test]
    fn the_meta_set_drops_it_on_a_crop() {
        let mut frame = frame_with(Orientation::Rotate90Cw);
        frame.meta.propagate(Transform::Scale);
        assert!(frame.meta.get::<OrientationMeta>().is_some());
        frame.meta.propagate(Transform::Crop);
        assert!(frame.meta.get::<OrientationMeta>().is_none());
    }
}
