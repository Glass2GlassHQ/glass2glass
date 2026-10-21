use alloc::string::String;
use alloc::vec::Vec;

use drm::control::{crtc, plane, property, Device as ControlDevice, ResourceHandles};
use drm::ClientCapability;

use g2g_core::meta::{Orientation, OrientationSet};
use g2g_core::{G2gError, HardwareError};

const ORIENTATIONS: [Orientation; 8] = [
    Orientation::Identity,
    Orientation::Rotate90Cw,
    Orientation::Rotate180,
    Orientation::Rotate90Ccw,
    Orientation::HorizontalMirror,
    Orientation::VerticalMirror,
    Orientation::Transpose,
    Orientation::Transverse,
];

// The `rotation` property of the primary plane on a sink's CRTC.
#[derive(Debug)]
pub(crate) struct PlaneRotation {
    plane: plane::Handle,
    property: property::Handle,
    // each value name the plane lists, with the bit it sets in the property
    values: Vec<(String, u32)>,
    // the property value as last read or written
    value: u64,
}

impl PlaneRotation {
    pub(crate) fn shown_orientations(&self) -> OrientationSet {
        ORIENTATIONS
            .into_iter()
            .filter(|&orientation| {
                rotation_value(drm_rotation(orientation), &self.values).is_some()
            })
            .collect()
    }

    // The property value that shows `orientation`, None when the plane already shows it.
    pub(crate) fn changed_value(&self, orientation: Orientation) -> Result<Option<u64>, G2gError> {
        let value = rotation_value(drm_rotation(orientation), &self.values)
            .ok_or(G2gError::CapsMismatch)?;
        Ok((value != self.value).then_some(value))
    }

    // Leaves `crtc` off, for the caller's next set_crtc to light it again.
    pub(crate) fn write(
        &mut self,
        card: &impl ControlDevice,
        crtc: crtc::Handle,
        value: u64,
    ) -> Result<(), G2gError> {
        // a quarter turn on a lit plane fails the kernel's size check until set_crtc sizes it again
        card.set_crtc(crtc, None, (0, 0), &[], None)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        card.set_property(self.plane, self.property, value)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        self.value = value;
        Ok(())
    }
}

// A plane without a `rotation` property shows the buffer as stored.
pub(crate) fn require_unrotated(orientation: Orientation) -> Result<(), G2gError> {
    match orientation {
        Orientation::Identity => Ok(()),
        _ => Err(G2gError::CapsMismatch),
    }
}

// A plane rotation: an optional `reflect-x`, then counter-clockwise quarter turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DrmRotation {
    counter_clockwise_quarter_turns: u8,
    reflect_x: bool,
}

// The plane applies the rotation as given, where `set_buffer_transform` applies its inverse.
fn drm_rotation(orientation: Orientation) -> DrmRotation {
    let (counter_clockwise_quarter_turns, reflect_x) = match orientation {
        Orientation::Identity => (0, false),
        Orientation::Rotate90Ccw => (1, false),
        Orientation::Rotate180 => (2, false),
        Orientation::Rotate90Cw => (3, false),
        Orientation::HorizontalMirror => (0, true),
        Orientation::Transpose => (1, true),
        Orientation::VerticalMirror => (2, true),
        Orientation::Transverse => (3, true),
    };
    DrmRotation {
        counter_clockwise_quarter_turns,
        reflect_x,
    }
}

const ROTATE_VALUE_NAMES: [&str; 4] = ["rotate-0", "rotate-90", "rotate-180", "rotate-270"];

// The `rotation` property value for `rotation`, None when the plane lacks a bit it needs.
fn rotation_value(rotation: DrmRotation, plane_values: &[(String, u32)]) -> Option<u64> {
    let bit = |name: &str| {
        plane_values
            .iter()
            .find(|(value_name, _)| value_name == name)
            .and_then(|&(_, position)| 1u64.checked_shl(position))
    };
    let rotate = |quarter_turns: usize| bit(ROTATE_VALUE_NAMES[quarter_turns % 4]);
    let quarter_turns = usize::from(rotation.counter_clockwise_quarter_turns);
    if !rotation.reflect_x {
        return rotate(quarter_turns);
    }
    // reflect-y is reflect-x followed by a half turn
    let reflected_in_x = bit("reflect-x").zip(rotate(quarter_turns));
    let reflected_in_y = || bit("reflect-y").zip(rotate(quarter_turns + 2));
    reflected_in_x
        .or_else(reflected_in_y)
        .map(|(reflect, turn)| reflect | turn)
}

// The value names of a bitmask property with the bit each one sets, as the kernel lists them.
fn bitmask_values(
    card: &impl ControlDevice,
    property: property::Handle,
) -> Result<Vec<(String, u32)>, G2gError> {
    let mut values = Vec::new();
    let mut enums = Vec::new();
    drm_ffi::mode::get_property(
        card.as_fd(),
        property.into(),
        Some(&mut values),
        Some(&mut enums),
    )
    .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
    enums
        .into_iter()
        .map(|entry| {
            let entry = property::EnumValue::from(entry);
            let name = entry
                .name()
                .to_str()
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            let position = u32::try_from(entry.value())
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            Ok((String::from(name), position))
        })
        .collect()
}

// None when the primary plane on `crtc` has no `rotation` property.
pub(crate) fn primary_plane_rotation(
    card: &impl ControlDevice,
    resources: &ResourceHandles,
    crtc: crtc::Handle,
) -> Result<Option<PlaneRotation>, G2gError> {
    // a kernel without universal planes lists no primary plane
    if card
        .set_client_capability(ClientCapability::UniversalPlanes, true)
        .is_err()
    {
        return Ok(None);
    }
    let planes = card
        .plane_handles()
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
    for plane in planes {
        let info = card
            .get_plane(plane)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        if !resources
            .filter_crtcs(info.possible_crtcs())
            .contains(&crtc)
        {
            continue;
        }
        let properties = card
            .get_properties(plane)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let mut primary = false;
        let mut rotation = None;
        for (&property, &value) in properties.iter() {
            let property_info = card
                .get_property(property)
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            if property_info.name() == c"type" {
                primary = property_info
                    .value_type()
                    .convert_value(value)
                    .as_enum()
                    .is_some_and(|plane_type| plane_type.name() == c"Primary");
            } else if property_info.name() == c"rotation" {
                rotation = Some((property, value));
            }
        }
        if !primary {
            continue;
        }
        let Some((property, value)) = rotation else {
            return Ok(None);
        };
        return Ok(Some(PlaneRotation {
            plane,
            property,
            values: bitmask_values(card, property)?,
            value,
        }));
    }
    Ok(None)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    // Bit positions run opposite to the uapi header's, to catch a hardcoded bit.
    fn plane_values(names: &[&str]) -> Vec<(String, u32)> {
        names
            .iter()
            .rev()
            .zip(0..)
            .map(|(&name, position)| (String::from(name), position))
            .collect()
    }

    const EVERY_VALUE: [&str; 6] = [
        "rotate-0",
        "rotate-90",
        "rotate-180",
        "rotate-270",
        "reflect-x",
        "reflect-y",
    ];

    fn bit(values: &[(String, u32)], name: &str) -> u64 {
        let &(_, position) = values
            .iter()
            .find(|(value_name, _)| value_name == name)
            .expect("the fixture lists the value");
        1 << position
    }

    // Plane handles the sink never passes to a device, the plane showing rotate-0.
    fn fixture_plane_rotation(values: Vec<(String, u32)>) -> PlaneRotation {
        PlaneRotation {
            plane: drm::control::from_u32(1).expect("non-zero handle"),
            property: drm::control::from_u32(2).expect("non-zero handle"),
            value: bit(&values, "rotate-0"),
            values,
        }
    }

    pub(crate) fn plane_with_every_value() -> PlaneRotation {
        fixture_plane_rotation(plane_values(&EVERY_VALUE))
    }

    pub(crate) fn plane_without_quarter_turns() -> PlaneRotation {
        fixture_plane_rotation(plane_values(&[
            "rotate-0",
            "rotate-180",
            "reflect-x",
            "reflect-y",
        ]))
    }

    pub(crate) fn half_turns_and_axis_mirrors() -> OrientationSet {
        [
            Orientation::Identity,
            Orientation::Rotate180,
            Orientation::HorizontalMirror,
            Orientation::VerticalMirror,
        ]
        .into_iter()
        .collect()
    }

    // The turn the plane shows for `rotation`: the reflection, then counter-clockwise turns.
    fn shown_orientation(rotation: DrmRotation) -> Orientation {
        let reflected = if rotation.reflect_x {
            Orientation::HorizontalMirror
        } else {
            Orientation::Identity
        };
        (0..rotation.counter_clockwise_quarter_turns).fold(reflected, |shown, _| {
            shown.compose(Orientation::Rotate90Ccw)
        })
    }

    // A quarter turn copied from waylandsink's inverted mapping fails here.
    #[test]
    fn plane_rotation_shows_the_descriptor() {
        for orientation in ORIENTATIONS {
            assert_eq!(
                shown_orientation(drm_rotation(orientation)),
                orientation,
                "{orientation:?}"
            );
        }
    }

    #[test]
    fn rotation_value_reads_bits_from_the_plane() {
        let values = plane_values(&EVERY_VALUE);
        assert_eq!(
            rotation_value(drm_rotation(Orientation::Rotate90Cw), &values),
            Some(bit(&values, "rotate-270"))
        );
        assert_eq!(
            rotation_value(drm_rotation(Orientation::Transpose), &values),
            Some(bit(&values, "reflect-x") | bit(&values, "rotate-90"))
        );
        assert_eq!(
            plane_with_every_value().shown_orientations(),
            OrientationSet::ALL
        );
    }

    #[test]
    fn reflect_y_stands_in_for_reflect_x() {
        let values = plane_values(&EVERY_VALUE);
        assert_eq!(
            rotation_value(drm_rotation(Orientation::HorizontalMirror), &values),
            Some(bit(&values, "reflect-x") | bit(&values, "rotate-0"))
        );

        let without_reflect_x = plane_values(&[
            "rotate-0",
            "rotate-90",
            "rotate-180",
            "rotate-270",
            "reflect-y",
        ]);
        assert_eq!(
            rotation_value(
                drm_rotation(Orientation::HorizontalMirror),
                &without_reflect_x
            ),
            Some(bit(&without_reflect_x, "reflect-y") | bit(&without_reflect_x, "rotate-180"))
        );
        assert_eq!(
            fixture_plane_rotation(without_reflect_x).shown_orientations(),
            OrientationSet::ALL
        );
    }

    #[test]
    fn plane_without_quarter_turns_shows_only_half_turns() {
        let plane = plane_without_quarter_turns();
        assert_eq!(
            plane.changed_value(Orientation::Rotate180),
            Ok(Some(bit(&plane.values, "rotate-180")))
        );
        assert_eq!(
            plane.changed_value(Orientation::Rotate90Cw),
            Err(G2gError::CapsMismatch)
        );
        assert_eq!(plane.shown_orientations(), half_turns_and_axis_mirrors());
    }

    #[test]
    fn unchanged_rotation_needs_no_write() {
        assert_eq!(
            plane_with_every_value().changed_value(Orientation::Identity),
            Ok(None)
        );
    }

    #[test]
    fn plane_without_rotation_shows_only_the_stored_picture() {
        assert_eq!(require_unrotated(Orientation::Identity), Ok(()));
        assert_eq!(
            require_unrotated(Orientation::Rotate90Cw),
            Err(G2gError::CapsMismatch)
        );
    }
}
