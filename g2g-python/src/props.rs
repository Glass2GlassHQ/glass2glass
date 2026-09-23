//! The property face shared by every hosted gst-python-ml element.
//!
//! A `pyelement` and a `pyaggregator` host the same family of Python classes, so
//! they accept the same tunables and forward them the same way. One list here so
//! adding a property to one host cannot leave the other behind, plus the parsing
//! for the caps-valued properties both hosts read themselves rather than forward.

use std::collections::BTreeMap;
use std::sync::Mutex;

use g2g_core::property::UNDECLARED_PROPERTIES;
use g2g_core::{Caps, CapsSet, PropError, PropKind, PropValue, PropertySpec};

/// What a property the hosted class declares is, for a `gst-inspect` dump.
const CLASS_PROPERTY_BLURB: &str = "declared by the hosted Python class, passed to it as text";

/// What a hosted element's undeclared properties are, for a `gst-inspect` dump.
pub(crate) const FORWARDED_BLURB: &str =
    "any other property is forwarded to the hosted Python class, which declares \
     the real set (model-name, engine-name, device, ...); a name it does not \
     declare fails when the pipeline starts";

/// The one concrete caps a `input-caps=` / `output-caps=` description names.
/// A description that names a set (a `{a,b}` list, an unbounded geometry) has no
/// single answer, so it is rejected rather than silently resolved.
pub(crate) fn fixed_caps(desc: &str) -> Result<Caps, PropError> {
    CapsSet::from_gst_string(desc)
        .and_then(|set| set.fixate())
        .ok_or(PropError::Value)
}

/// Record a property the host does not read itself, to forward to the hosted
/// Python instance. Re-setting one replaces it, and the order they were set in
/// is the order they are applied.
pub(crate) fn forward(params: &mut Vec<(String, PropValue)>, name: &str, value: PropValue) {
    match params.iter_mut().find(|(key, _)| key == name) {
        Some(slot) => slot.1 = value,
        None => params.push((name.to_string(), value)),
    }
}

/// Build a host element's property list: its own entries, then the marker saying
/// every other name goes to the hosted class. `PropertySpec` resolves at the
/// call site, which already imports it.
macro_rules! hosted_element_props {
    ($($own:expr),+ $(,)?) => {
        &[
            $($own,)+
            PropertySpec::undeclared(crate::props::FORWARDED_BLURB),
        ]
    };
}

pub(crate) use hosted_element_props;

/// A host element's property list, extended with each hosted class's own
/// properties once that class has loaded.
///
/// `properties()` returns `&'static`, so each class's list is leaked once and
/// reused, which bounds the leak by the number of distinct classes loaded.
#[derive(Debug)]
pub(crate) struct HostedClassProperties {
    own: &'static [PropertySpec],
    by_class: Mutex<BTreeMap<(String, String), &'static [PropertySpec]>>,
}

impl HostedClassProperties {
    pub(crate) const fn new(own: &'static [PropertySpec]) -> Self {
        Self {
            own,
            by_class: Mutex::new(BTreeMap::new()),
        }
    }

    /// The host's own properties followed by the ones `class` from `module`
    /// declares, in place of the entry that lets any name through. Just the
    /// host's own list while no class is named, or when the class declares
    /// nothing or does not load.
    pub(crate) fn for_class(&self, module: &str, class: &str) -> &'static [PropertySpec] {
        if module.is_empty() || class.is_empty() {
            return self.own;
        }
        let mut by_class = self
            .by_class
            .lock()
            .expect("hosted class property cache lock poisoned");
        let key = (module.to_string(), class.to_string());
        if let Some(specs) = by_class.get(&key) {
            return specs;
        }
        let specs = match declared_by_class(module, class) {
            Some(declared) => with_class_properties(self.own, declared),
            None => self.own,
        };
        by_class.insert(key, specs);
        specs
    }
}

#[cfg(feature = "python")]
use crate::host::class_declared_properties as declared_by_class;

#[cfg(not(feature = "python"))]
fn declared_by_class(_module: &str, _class: &str) -> Option<Vec<String>> {
    None
}

/// `own` minus the entry that lets any name through, then each Python attribute
/// the class declares under its gst name, skipping one the host reads itself.
fn with_class_properties(
    own: &'static [PropertySpec],
    declared: Vec<String>,
) -> &'static [PropertySpec] {
    let class_specs = declared
        .into_iter()
        .map(|attribute| attribute.replace('_', "-"))
        .filter(|name| !own.iter().any(|spec| spec.name == name))
        .map(|name| PropertySpec::new(name.leak(), PropKind::Str, CLASS_PROPERTY_BLURB));
    own.iter()
        .filter(|spec| spec.name != UNDECLARED_PROPERTIES)
        .copied()
        .chain(class_specs)
        .collect::<Vec<_>>()
        .leak()
}
