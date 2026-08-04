//! Animated properties (M882), the `gst-controller` analog: a property's value
//! becomes a function of stream time instead of a constant set once at build
//! time. A [`ControlSource`] is a keyframed curve, a [`ControlProgram`] binds
//! curves to one node's property names, and the runner samples the bindings at
//! each frame's PTS before handing that frame to the element.
//!
//! The program is checked against the element's own
//! [`PropertySpec`] table when the graph starts ([`ControlProgram::resolve`]), so
//! a misspelled or non-animatable property fails the run before any frame flows
//! rather than animating nothing.

use alloc::string::String;
use alloc::vec::Vec;

use crate::property::{PropError, PropKind, PropValue, PropertySpec};

/// The log category a controller fault is reported on, so
/// `G2G_DEBUG=controller:debug` follows the animation independently of element
/// logging (as [`CAPS_CATEGORY`](crate::log::CAPS_CATEGORY) does for the solver).
pub const CONTROL_CATEGORY: &str = "controller";

/// A keyframed value curve, sampled by PTS. Values are `f64` regardless of the
/// property's kind; the conversion to the property's type happens at
/// [`apply`](ArmController::apply).
///
/// Both variants clamp outside their keyframe range: before the first keyframe
/// the first value holds, after the last one the last value holds.
// Closed set: two interpolations cover the animation cases that exist. A cubic /
// LFO source is a real addition when something needs one, not a placeholder.
#[derive(Debug, Clone, PartialEq)]
pub enum ControlSource {
    /// Hold each keyframe's value until the next keyframe's time (a discrete
    /// knob: a mode switch, a boolean).
    Step(Vec<(u64, f64)>),
    /// Interpolate linearly between the surrounding keyframes (a smooth pan,
    /// fade, or zoom).
    Linear(Vec<(u64, f64)>),
}

impl ControlSource {
    /// A step curve over `keys` (`(pts_ns, value)`), sorted here so the caller
    /// need not supply them in order.
    pub fn step(keys: impl IntoIterator<Item = (u64, f64)>) -> Self {
        ControlSource::Step(sorted(keys))
    }

    /// A linear curve over `keys` (`(pts_ns, value)`), sorted here.
    pub fn linear(keys: impl IntoIterator<Item = (u64, f64)>) -> Self {
        ControlSource::Linear(sorted(keys))
    }

    fn keys(&self) -> &[(u64, f64)] {
        match self {
            ControlSource::Step(k) | ControlSource::Linear(k) => k,
        }
    }

    /// The curve's value at `t_ns`. Clamps to the end values outside the
    /// keyframe range; `0.0` for a curve with no keyframes, which
    /// [`ControlProgram::resolve`] rejects before a run can sample it.
    pub fn value_at(&self, t_ns: u64) -> f64 {
        let keys = self.keys();
        let (Some(&(first_t, first_v)), Some(&(last_t, last_v))) = (keys.first(), keys.last())
        else {
            return 0.0;
        };
        if t_ns <= first_t {
            return first_v;
        }
        if t_ns >= last_t {
            return last_v;
        }
        // `t_ns` is strictly inside the range, so the neighbours both exist.
        match self {
            // The latest keyframe at or before `t_ns`: a keyframe's value takes
            // effect exactly at its own time.
            ControlSource::Step(_) => keys[keys.partition_point(|&(kt, _)| kt <= t_ns) - 1].1,
            ControlSource::Linear(_) => {
                let i = keys.partition_point(|&(kt, _)| kt < t_ns);
                let (t1, v1) = keys[i];
                let (t0, v0) = keys[i - 1];
                // t0 < t_ns <= t1, so the span is non-zero.
                let span = (t1 - t0) as f64;
                let into = (t_ns - t0) as f64;
                v0 + (v1 - v0) * (into / span)
            }
        }
    }
}

fn sorted(keys: impl IntoIterator<Item = (u64, f64)>) -> Vec<(u64, f64)> {
    let mut keys: Vec<(u64, f64)> = keys.into_iter().collect();
    keys.sort_by_key(|&(t, _)| t);
    keys
}

/// The animated properties of one graph node: property name -> curve. Attach it
/// with [`Graph::set_node_control`](crate::Graph::set_node_control).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ControlProgram {
    bindings: Vec<(String, ControlSource)>,
}

impl ControlProgram {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind `property` to `source`. Re-binding a name replaces its curve, so a
    /// program cannot hold two curves fighting over one property.
    pub fn bind(mut self, property: &str, source: ControlSource) -> Self {
        match self.bindings.iter_mut().find(|(n, _)| n == property) {
            Some(slot) => slot.1 = source,
            None => self.bindings.push((String::from(property), source)),
        }
        self
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Check every binding against the target element's declared properties and
    /// resolve each one's [`PropKind`], so the run's per-frame sampling is a
    /// straight conversion with no name lookup. The first offending binding
    /// fails.
    pub fn resolve(self, specs: &[PropertySpec]) -> Result<ArmController, ControlFault> {
        let mut bound = Vec::with_capacity(self.bindings.len());
        for (property, source) in self.bindings {
            let fault = |reason| ControlFault {
                property: property.clone(),
                reason,
            };
            if source.keys().is_empty() {
                return Err(fault(ControlReason::NoKeyframes));
            }
            let spec = specs
                .iter()
                .find(|s| s.name == property)
                .ok_or_else(|| fault(ControlReason::UnknownProperty))?;
            if !animatable(spec.kind) {
                return Err(fault(ControlReason::NotAnimatable(spec.kind)));
            }
            bound.push(Bound {
                property,
                kind: spec.kind,
                source,
            });
        }
        Ok(ArmController { bound })
    }
}

/// The property kinds a curve can drive: the numeric ones, plus `Bool` as a
/// threshold at 0.5. A fraction / string / flags property has no meaningful
/// interpolation, so binding one is a startup error rather than a silent skip.
fn animatable(kind: PropKind) -> bool {
    matches!(
        kind,
        PropKind::Bool | PropKind::Int | PropKind::Uint | PropKind::Double
    )
}

#[derive(Debug, Clone, PartialEq)]
struct Bound {
    property: String,
    kind: PropKind,
    source: ControlSource,
}

/// A [`ControlProgram`] resolved against its element, held by the arm that owns
/// that element. The runner builds one per controlled node at startup and
/// [`apply`](Self::apply)s it before each `DataFrame`.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmController {
    bound: Vec<Bound>,
}

impl ArmController {
    /// Sample every binding at `pts_ns` and set it on `target`. An element that
    /// rejects a sampled value fails the run loud: a curve that walks a property
    /// out of its accepted range is a broken program, not something to swallow.
    pub fn apply<T: ControlTarget + ?Sized>(
        &self,
        target: &mut T,
        pts_ns: u64,
    ) -> Result<(), ControlFault> {
        for b in &self.bound {
            let value = convert(b.kind, b.source.value_at(pts_ns));
            target
                .set_control(&b.property, value)
                .map_err(|e| ControlFault {
                    property: b.property.clone(),
                    reason: ControlReason::Rejected(e),
                })?;
        }
        Ok(())
    }
}

/// Convert a sampled curve value to the property's kind: round to nearest and
/// clamp into the kind's representable range (so a negative sample cannot wrap a
/// `Uint`), `>= 0.5` for a `Bool`. `kind` is one [`animatable`] accepted.
fn convert(kind: PropKind, sample: f64) -> PropValue {
    match kind {
        PropKind::Bool => PropValue::Bool(sample >= 0.5),
        PropKind::Double => PropValue::Double(sample),
        // `as` truncates toward zero, saturates at the integer bounds, and maps
        // NaN to 0, so the half-step is what rounds (`f64::round` is a `std`
        // method and the core is `no_std`, like `segment`'s `fabs`).
        PropKind::Uint => PropValue::Uint(if sample <= 0.0 {
            0
        } else {
            (sample + 0.5) as u64
        }),
        _ => PropValue::Int(if sample < 0.0 {
            (sample - 0.5) as i64
        } else {
            (sample + 0.5) as i64
        }),
    }
}

/// Why a [`ControlProgram`] could not be resolved or applied, and to which
/// property. The runner logs it and fails the run
/// ([`G2gError::ControlBinding`](crate::G2gError::ControlBinding)).
#[derive(Debug, Clone, PartialEq)]
pub struct ControlFault {
    pub property: String,
    pub reason: ControlReason,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ControlReason {
    /// The element declares no property of that name.
    UnknownProperty,
    /// The property exists but its kind carries no number to animate.
    NotAnimatable(PropKind),
    /// The binding has no keyframes, so there is nothing to sample.
    NoKeyframes,
    /// The element refused a sampled value.
    Rejected(PropError),
}

impl core::fmt::Display for ControlFault {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.reason {
            ControlReason::UnknownProperty => {
                write!(f, "no property named `{}` on this element", self.property)
            }
            ControlReason::NotAnimatable(kind) => write!(
                f,
                "property `{}` is a {} and cannot be animated",
                self.property,
                kind.label()
            ),
            ControlReason::NoKeyframes => {
                write!(f, "property `{}` is bound to an empty curve", self.property)
            }
            ControlReason::Rejected(e) => write!(
                f,
                "property `{}` rejected a sampled value ({e})",
                self.property
            ),
        }
    }
}

/// The property surface a controller drives. Implemented for the erased element
/// traits the runner's arms hold, so one sampling path serves a transform, a
/// sink, and a fan-in element. Those traits are `std`-only (a `no_std` graph runs
/// monomorphised elements), so the impls are too; an element type can implement
/// this directly to be driven without them.
pub trait ControlTarget {
    fn set_control(&mut self, name: &str, value: PropValue) -> Result<(), PropError>;
}

#[cfg(feature = "std")]
impl ControlTarget for dyn crate::element::DynAsyncElement + '_ {
    fn set_control(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.set_property(name, value)
    }
}

#[cfg(feature = "std")]
impl ControlTarget for dyn crate::runtime::DynMultiInputElement + '_ {
    fn set_control(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.set_property(name, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys() -> [(u64, f64); 3] {
        [(0, 0.0), (100, 10.0), (200, 20.0)]
    }

    #[test]
    fn linear_interpolates_and_clamps_to_the_end_values() {
        let s = ControlSource::linear(keys());
        assert_eq!(s.value_at(0), 0.0);
        assert_eq!(s.value_at(50), 5.0, "halfway between two keyframes");
        assert_eq!(s.value_at(100), 10.0, "on a keyframe");
        assert_eq!(s.value_at(150), 15.0);
        assert_eq!(s.value_at(1_000), 20.0, "past the end holds the last value");
    }

    #[test]
    fn step_holds_each_keyframe_until_the_next() {
        let s = ControlSource::step(keys());
        assert_eq!(s.value_at(0), 0.0);
        assert_eq!(s.value_at(99), 0.0, "still the first value");
        assert_eq!(s.value_at(100), 10.0);
        assert_eq!(s.value_at(199), 10.0);
        assert_eq!(s.value_at(500), 20.0);
    }

    #[test]
    fn keyframes_are_sorted_at_construction() {
        let s = ControlSource::linear([(200, 20.0), (0, 0.0), (100, 10.0)]);
        assert_eq!(s.value_at(50), 5.0);
    }

    #[test]
    fn conversion_rounds_clamps_and_thresholds() {
        assert_eq!(convert(PropKind::Int, -2.4), PropValue::Int(-2));
        assert_eq!(convert(PropKind::Int, 2.5), PropValue::Int(3));
        assert_eq!(
            convert(PropKind::Uint, -7.0),
            PropValue::Uint(0),
            "a negative sample clamps instead of wrapping"
        );
        assert_eq!(convert(PropKind::Bool, 0.49), PropValue::Bool(false));
        assert_eq!(convert(PropKind::Bool, 0.5), PropValue::Bool(true));
        assert_eq!(convert(PropKind::Double, 1.25), PropValue::Double(1.25));
    }

    #[test]
    fn rebinding_a_property_replaces_its_curve() {
        let p = ControlProgram::new()
            .bind("x", ControlSource::step([(0, 1.0)]))
            .bind("x", ControlSource::step([(0, 2.0)]));
        assert_eq!(p.bindings.len(), 1);
        assert_eq!(p.bindings[0].1.value_at(0), 2.0);
    }
}
