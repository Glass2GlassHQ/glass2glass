//! Validation of everything a plugin hands the host.
//!
//! Every field below is attacker-controlled: a `.so` on the plugin path may be
//! corrupt, built against a different ABI generation, or hostile. The rules are
//! the repo's parser rules applied to a binary interface rather than a
//! bitstream: bound every count before it is used as a length, require a
//! non-null pointer before dereferencing, check UTF-8 before a byte range
//! becomes a `str`, and reject a malformed value rather than repairing it.
//!
//! Validation runs in two stages, and the split is the point of the design:
//!
//! 1. [`validate_descriptor`] reads the exported static. No plugin code has run
//!    yet, so the capability list it returns is a *declaration* the caller's
//!    policy can act on before the plugin gets control.
//! 2. [`validate_element`] checks each registration the plugin then makes.
//!    The loader additionally requires each one to match a declared capability,
//!    so the declaration is a promise the plugin is held to, not a hint.

use std::boxed::Box;
use std::string::{String, ToString};
use std::vec::Vec;

use g2g_core::caps::CapsSet;
use g2g_core::property::{ElementMetadata, PropKind, PropValue, PropertySpec};

use super::caps::{caps_set_from_ffi, is_fixable, CapsCodeError};
use super::{
    read_versioned, FfiCapability, FfiElementRegistration, FfiElementVtable, FfiPluginDescriptor,
    FfiPropertySpec, FfiRegistrar, FfiStatus, FfiStr, ELEMENT_SINK, ELEMENT_TRANSFORM,
    MAX_CAPABILITIES, MAX_ELEMENTS, MAX_PROPERTIES, PROP_BOOL, PROP_DOUBLE, PROP_FRACTION,
    PROP_INT, PROP_STR, PROP_UINT, V2_ABI_VERSION, V2_MAGIC,
};

/// Longest element or property name accepted. Long enough for any real name,
/// short enough that a corrupt length field cannot become a wall of text in a
/// `gst-inspect` dump or a log line.
pub const MAX_NAME_LEN: usize = 64;

/// What a v2 element is, for the capability declaration and the policy hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementKind {
    /// A 1-in / 1-out transform.
    Transform,
    /// A terminal sink.
    Sink,
}

impl ElementKind {
    fn from_code(code: u32) -> Option<ElementKind> {
        match code {
            ELEMENT_TRANSFORM => Some(ElementKind::Transform),
            ELEMENT_SINK => Some(ElementKind::Sink),
            _ => None,
        }
    }
}

/// One thing a plugin declares it will register, read before any plugin code
/// runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginCapability {
    /// An element by name and kind.
    Element {
        /// The `gst-launch` element name.
        name: String,
        /// Transform or sink.
        kind: ElementKind,
    },
    /// A capability kind this host does not know. Carried through to the policy
    /// rather than refused, so a plugin built against a newer ABI generation is
    /// a decision the caller makes rather than a hard failure. Nothing in this
    /// host can act on it, so allowing one registers nothing.
    Unknown {
        /// The declared kind code.
        kind: u32,
        /// The declared name.
        name: String,
    },
}

/// A validated descriptor: what the plugin says it is and what it will
/// register, plus the one entry point the host may call.
#[derive(Debug)]
pub struct PluginDeclaration {
    /// Plugin name, for diagnostics and policy.
    pub name: String,
    /// Plugin version string.
    pub version: String,
    /// Everything the plugin declares it will register.
    pub capabilities: Vec<PluginCapability>,
    /// The registration entry point. Call only after the policy allows the
    /// capabilities above.
    pub register: unsafe extern "C" fn(registrar: *const FfiRegistrar) -> FfiStatus,
}

impl PluginDeclaration {
    /// The declared kind for `name`, or `None` if the plugin never declared it.
    pub fn declared_kind(&self, name: &str) -> Option<ElementKind> {
        self.capabilities.iter().find_map(|c| match c {
            PluginCapability::Element { name: n, kind } if n == name => Some(*kind),
            _ => None,
        })
    }
}

/// A validated element registration, with every plugin-supplied string copied
/// into host-owned memory and every count checked.
#[derive(Debug)]
pub struct ValidatedElement {
    /// The `gst-launch` element name.
    pub name: String,
    /// Transform or sink.
    pub kind: ElementKind,
    /// Introspection metadata (`gst-inspect` Factory Details).
    pub metadata: ElementMetadata,
    /// Caps the element accepts. Empty means any.
    pub sink_caps: CapsSet,
    /// Caps the element produces. Empty means pass-through.
    pub source_caps: CapsSet,
    /// The element's properties.
    pub properties: Vec<PropertySpec>,
    /// The element's entry points, read into a host-sized copy with any absent
    /// tail zeroed.
    pub vtable: FfiElementVtable,
    /// Build one instance.
    pub create: unsafe extern "C" fn() -> *mut core::ffi::c_void,
}

/// Why something a plugin supplied was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// The descriptor symbol resolved to a null pointer.
    NullDescriptor,
    /// The first eight bytes are not [`V2_MAGIC`]: not a v2 descriptor.
    BadMagic {
        /// What was there instead.
        found: u64,
    },
    /// The plugin declares an ABI generation this host does not implement.
    AbiVersion {
        /// What the plugin declared.
        plugin: u32,
        /// What this host implements.
        host: u32,
    },
    /// A `struct_size` too small to describe even its own header.
    StructTooSmall {
        /// The declared size.
        declared: usize,
    },
    /// A string with a length but a null pointer.
    NullString,
    /// A string longer than [`MAX_STRING_LEN`](super::MAX_STRING_LEN).
    StringTooLong {
        /// The declared length.
        len: usize,
    },
    /// A string that is not valid UTF-8.
    NotUtf8,
    /// An element or property name that is empty, over-long, or contains
    /// anything but lowercase ASCII letters, digits, `-`, and `_`. A name
    /// reaches `gst-launch` lines, log output, and `gst-inspect` dumps, so a
    /// control character or a space in one corrupts more than the plugin.
    BadName {
        /// The offending name, escaped for display.
        name: String,
    },
    /// More capabilities than [`MAX_CAPABILITIES`].
    TooManyCapabilities {
        /// The declared count.
        count: usize,
    },
    /// A non-zero capability count with a null list pointer.
    NullCapabilities,
    /// A required function pointer was null.
    MissingFunction {
        /// Which one.
        name: &'static str,
    },
    /// The element registration's vtable pointer was null.
    NullVtable,
    /// More properties than [`MAX_PROPERTIES`].
    TooManyProperties {
        /// The declared count.
        count: usize,
    },
    /// A non-zero property count with a null list pointer.
    NullProperties,
    /// A property value kind outside the v2 set. The flag-set kind is
    /// deliberately absent, so an element declaring one lands here.
    BadPropertyKind {
        /// The offending kind code.
        kind: u32,
    },
    /// A property's declared default text does not parse for its kind.
    BadPropertyDefault {
        /// The property name.
        name: String,
    },
    /// An element kind code outside [`ELEMENT_TRANSFORM`] / [`ELEMENT_SINK`].
    BadElementKind {
        /// The offending code.
        kind: u32,
    },
    /// A declared source caps set with an `Any` dimension or framerate. The
    /// wrapper returns these caps from `intercept_caps`, where an `Any` cannot
    /// be fixated, so the registration is refused rather than failing later
    /// inside the solver.
    UnfixableSourceCaps,
    /// A caps value could not cross the boundary.
    Caps(CapsCodeError),
    /// More elements registered than [`MAX_ELEMENTS`].
    TooManyElements,
    /// The plugin registered an element it never declared. The capability list
    /// is a promise, and this is the loader catching it being broken.
    UndeclaredElement {
        /// The element name the plugin tried to register.
        name: String,
    },
    /// The plugin registered a declared name under a different kind.
    KindMismatch {
        /// The element name.
        name: String,
        /// What the descriptor declared.
        declared: ElementKind,
        /// What registration attempted.
        attempted: ElementKind,
    },
    /// The same element name was registered twice.
    DuplicateElement {
        /// The repeated name.
        name: String,
    },
}

impl From<CapsCodeError> for ValidationError {
    fn from(e: CapsCodeError) -> Self {
        ValidationError::Caps(e)
    }
}

impl core::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ValidationError::NullDescriptor => f.write_str("plugin descriptor symbol is null"),
            ValidationError::BadMagic { found } => {
                write!(
                    f,
                    "descriptor magic {found:#018x} is not a g2g v2 descriptor"
                )
            }
            ValidationError::AbiVersion { plugin, host } => write!(
                f,
                "plugin declares ABI generation {plugin}; this host implements {host}"
            ),
            ValidationError::StructTooSmall { declared } => {
                write!(
                    f,
                    "declared struct size {declared} is too small to be valid"
                )
            }
            ValidationError::NullString => f.write_str("string has a length but a null pointer"),
            ValidationError::StringTooLong { len } => write!(f, "string length {len} is too long"),
            ValidationError::NotUtf8 => f.write_str("string is not valid UTF-8"),
            ValidationError::BadName { name } => write!(f, "'{name}' is not a usable name"),
            ValidationError::TooManyCapabilities { count } => {
                write!(f, "{count} declared capabilities is too many")
            }
            ValidationError::NullCapabilities => {
                f.write_str("capability count is non-zero but the list is null")
            }
            ValidationError::MissingFunction { name } => {
                write!(f, "required entry point `{name}` is null")
            }
            ValidationError::NullVtable => f.write_str("element registration has a null vtable"),
            ValidationError::TooManyProperties { count } => {
                write!(f, "{count} properties is too many")
            }
            ValidationError::NullProperties => {
                f.write_str("property count is non-zero but the list is null")
            }
            ValidationError::BadPropertyKind { kind } => {
                write!(f, "property kind {kind} does not cross the v2 ABI")
            }
            ValidationError::BadPropertyDefault { name } => {
                write!(f, "property '{name}' has a default that does not parse")
            }
            ValidationError::BadElementKind { kind } => write!(f, "unknown element kind {kind}"),
            ValidationError::UnfixableSourceCaps => {
                f.write_str("declared source caps contain an unfixable `any` field")
            }
            ValidationError::Caps(e) => write!(f, "{e}"),
            ValidationError::TooManyElements => f.write_str("too many elements registered"),
            ValidationError::UndeclaredElement { name } => write!(
                f,
                "plugin registered '{name}', which its descriptor never declared"
            ),
            ValidationError::KindMismatch {
                name,
                declared,
                attempted,
            } => write!(
                f,
                "plugin declared '{name}' as {declared:?} but registered it as {attempted:?}"
            ),
            ValidationError::DuplicateElement { name } => {
                write!(f, "element '{name}' registered twice")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Read a plugin string into host-owned memory.
///
/// # Safety
/// See [`FfiStr::as_str`]: the pointer/length pair must describe readable bytes.
unsafe fn owned_string(s: &FfiStr) -> Result<String, ValidationError> {
    // SAFETY: forwarded from this function's own contract.
    unsafe { s.as_str() }.map(ToString::to_string)
}

/// Read a plugin string and leak it, so it can be stored in the `&'static str`
/// fields `PropertySpec` / `ElementMetadata` use.
///
/// Leaking is the honest lifetime here: a loaded plugin's code is kept mapped
/// for the life of the process (unmapping it would dangle the element factories
/// pointing into it), so its introspection strings are equally permanent. The
/// amount is bounded by [`MAX_ELEMENTS`] x [`MAX_PROPERTIES`] x
/// [`MAX_STRING_LEN`](super::MAX_STRING_LEN).
///
/// # Safety
/// See [`FfiStr::as_str`].
unsafe fn leaked_str(s: &FfiStr) -> Result<&'static str, ValidationError> {
    // SAFETY: forwarded from this function's own contract.
    let owned = unsafe { owned_string(s) }?;
    Ok(Box::leak(owned.into_boxed_str()))
}

/// Accept only lowercase ASCII names: letters, digits, `-`, `_`, starting with
/// a letter. This is the character set `gst-launch` element names and property
/// names already use, and it keeps a plugin from injecting whitespace, quotes,
/// or control characters into a pipeline description or a log line.
fn check_name(name: &str) -> Result<(), ValidationError> {
    let bad = name.is_empty()
        || name.len() > MAX_NAME_LEN
        || !name.starts_with(|c: char| c.is_ascii_lowercase())
        || !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_');
    if bad {
        return Err(ValidationError::BadName {
            name: name.escape_debug().to_string(),
        });
    }
    Ok(())
}

/// Read and validate the exported descriptor. Runs before any plugin code, so
/// the capabilities it returns are a pre-execution declaration.
///
/// # Safety
/// `ptr` must be the address `dlsym` returned for
/// [`V2_DESCRIPTOR_SYMBOL`](super::V2_DESCRIPTOR_SYMBOL) in a library that
/// stays loaded for at least the duration of the call. Everything reachable
/// from it is treated as untrusted data.
pub unsafe fn validate_descriptor(
    ptr: *const FfiPluginDescriptor,
) -> Result<PluginDeclaration, ValidationError> {
    if ptr.is_null() {
        return Err(ValidationError::NullDescriptor);
    }

    // Magic and generation come first, from the fixed 16-byte head every
    // generation of this descriptor shares. Nothing else is read until both
    // agree, so a stale or foreign symbol is rejected before its `struct_size`
    // is believed.
    // SAFETY: the caller guarantees `ptr` addresses a descriptor static; the
    // magic / abi_version / struct_size fields sit at fixed offsets that no ABI
    // generation moves.
    let magic = unsafe { core::ptr::addr_of!((*ptr).magic).read_unaligned() };
    if magic != V2_MAGIC {
        return Err(ValidationError::BadMagic { found: magic });
    }
    // SAFETY: as above.
    let abi_version = unsafe { core::ptr::addr_of!((*ptr).abi_version).read_unaligned() };
    if abi_version != V2_ABI_VERSION {
        return Err(ValidationError::AbiVersion {
            plugin: abi_version,
            host: V2_ABI_VERSION,
        });
    }
    // SAFETY: as above.
    let declared_size =
        unsafe { core::ptr::addr_of!((*ptr).struct_size).read_unaligned() } as usize;
    // SAFETY: the magic matched, so this is a v2 descriptor whose author wrote
    // at least `declared_size` bytes; `read_versioned` copies no more than that
    // and no more than the host's own struct.
    let desc = unsafe { read_versioned(ptr, declared_size) }?;

    // SAFETY: descriptor strings point into the loaded library's data.
    let name = unsafe { owned_string(&desc.name) }?;
    // SAFETY: as above.
    let version = unsafe { owned_string(&desc.version) }?;

    let Some(register) = desc.register else {
        return Err(ValidationError::MissingFunction { name: "register" });
    };

    if desc.capability_count > MAX_CAPABILITIES {
        return Err(ValidationError::TooManyCapabilities {
            count: desc.capability_count,
        });
    }
    if desc.capability_count > 0 && desc.capabilities.is_null() {
        return Err(ValidationError::NullCapabilities);
    }
    let declared: &[FfiCapability] = if desc.capability_count == 0 {
        &[]
    } else {
        // SAFETY: the count is bounded and the pointer non-null; the plugin
        // author's contract is that it addresses that many capabilities.
        unsafe { core::slice::from_raw_parts(desc.capabilities, desc.capability_count) }
    };

    let mut capabilities = Vec::with_capacity(declared.len());
    for cap in declared {
        // SAFETY: capability strings point into the loaded library's data.
        let cap_name = unsafe { owned_string(&cap.name) }?;
        match ElementKind::from_code(cap.kind) {
            Some(kind) => {
                check_name(&cap_name)?;
                capabilities.push(PluginCapability::Element {
                    name: cap_name,
                    kind,
                });
            }
            None => capabilities.push(PluginCapability::Unknown {
                kind: cap.kind,
                name: cap_name,
            }),
        }
    }

    Ok(PluginDeclaration {
        name,
        version,
        capabilities,
        register,
    })
}

/// Read and validate one element registration.
///
/// # Safety
/// `ptr` must be the pointer the plugin passed to
/// [`FfiRegistrar::register_element`], addressing a registration whose strings,
/// caps arrays, property array, and vtable stay valid for the life of the
/// process.
pub unsafe fn validate_element(
    ptr: *const FfiElementRegistration,
) -> Result<ValidatedElement, ValidationError> {
    if ptr.is_null() {
        return Err(ValidationError::NullDescriptor);
    }
    // SAFETY: `struct_size` is the first field of every generation of this
    // struct, so it is readable before the rest is trusted.
    let declared_size =
        unsafe { core::ptr::addr_of!((*ptr).struct_size).read_unaligned() } as usize;
    // SAFETY: the caller guarantees the plugin wrote `declared_size` bytes here.
    let reg = unsafe { read_versioned(ptr, declared_size) }?;

    let Some(kind) = ElementKind::from_code(reg.kind) else {
        return Err(ValidationError::BadElementKind { kind: reg.kind });
    };
    // SAFETY: registration strings point into the loaded library's data.
    let name = unsafe { owned_string(&reg.name) }?;
    check_name(&name)?;

    let metadata = ElementMetadata {
        // SAFETY: as above.
        long_name: unsafe { leaked_str(&reg.metadata.long_name) }?,
        // SAFETY: as above.
        klass: unsafe { leaked_str(&reg.metadata.klass) }?,
        // SAFETY: as above.
        description: unsafe { leaked_str(&reg.metadata.description) }?,
        // SAFETY: as above.
        author: unsafe { leaked_str(&reg.metadata.author) }?,
    };

    // SAFETY: the caps arrays are bounds-checked inside `caps_set_from_ffi`
    // before either pointer is dereferenced.
    let sink_caps = unsafe { caps_set_from_ffi(reg.sink_caps.alternatives, reg.sink_caps.count) }?;
    // SAFETY: as above.
    let source_caps =
        unsafe { caps_set_from_ffi(reg.source_caps.alternatives, reg.source_caps.count) }?;
    if source_caps.alternatives().iter().any(|c| !is_fixable(c)) {
        return Err(ValidationError::UnfixableSourceCaps);
    }

    // SAFETY: the property count is bounds-checked inside `read_properties`
    // before its pointer is dereferenced.
    let properties = unsafe { read_properties(reg.properties, reg.property_count) }?;

    if reg.vtable.is_null() {
        return Err(ValidationError::NullVtable);
    }
    // SAFETY: `struct_size` is the first field of every vtable generation.
    let vtable_size =
        unsafe { core::ptr::addr_of!((*reg.vtable).struct_size).read_unaligned() } as usize;
    // SAFETY: the plugin's contract is that it wrote `vtable_size` bytes at
    // `reg.vtable`; `read_versioned` reads no more than that.
    let vtable: FfiElementVtable = unsafe { read_versioned(reg.vtable, vtable_size) }?;

    // Only these two are required. Everything else absent means "use the
    // AsyncElement trait default", which is what makes a shorter vtable from an
    // older plugin loadable.
    if vtable.process.is_none() {
        return Err(ValidationError::MissingFunction { name: "process" });
    }
    if vtable.destroy.is_none() {
        return Err(ValidationError::MissingFunction { name: "destroy" });
    }
    let Some(create) = reg.create else {
        return Err(ValidationError::MissingFunction { name: "create" });
    };

    Ok(ValidatedElement {
        name,
        kind,
        metadata,
        sink_caps,
        source_caps,
        properties,
        vtable,
        create,
    })
}

/// # Safety
/// `ptr` must address `count` initialised [`FfiPropertySpec`] values whose
/// strings stay valid for the life of the process.
unsafe fn read_properties(
    ptr: *const FfiPropertySpec,
    count: usize,
) -> Result<Vec<PropertySpec>, ValidationError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if count > MAX_PROPERTIES {
        return Err(ValidationError::TooManyProperties { count });
    }
    if ptr.is_null() {
        return Err(ValidationError::NullProperties);
    }
    // SAFETY: the count is bounded and the pointer non-null; the caller's
    // contract covers the rest.
    let specs = unsafe { core::slice::from_raw_parts(ptr, count) };

    let mut out = Vec::with_capacity(count);
    for spec in specs {
        let kind = match spec.kind {
            PROP_BOOL => PropKind::Bool,
            PROP_INT => PropKind::Int,
            PROP_UINT => PropKind::Uint,
            PROP_DOUBLE => PropKind::Double,
            PROP_FRACTION => PropKind::Fraction,
            PROP_STR => PropKind::Str,
            other => return Err(ValidationError::BadPropertyKind { kind: other }),
        };
        // SAFETY: property strings point into the loaded library's data.
        let name = unsafe { leaked_str(&spec.name) }?;
        check_name(name)?;
        // SAFETY: as above.
        let blurb = unsafe { leaked_str(&spec.blurb) }?;
        // SAFETY: as above.
        let default = unsafe { leaked_str(&spec.default_value) }?;

        let mut out_spec = PropertySpec::new(name, kind, blurb);
        if !default.is_empty() {
            if PropValue::parse(kind, default).is_err() {
                return Err(ValidationError::BadPropertyDefault {
                    name: name.to_string(),
                });
            }
            out_spec = out_spec.with_default(default);
        }
        if spec.writable == 0 {
            out_spec = out_spec.read_only();
        }
        out.push(out_spec);
    }
    Ok(out)
}

/// Check a validated registration against what the descriptor declared, and
/// against what has already been registered in this load.
///
/// This is the second half of the capability gate: the declaration was allowed
/// by policy *before* the plugin ran, so anything the plugin then attempts that
/// the declaration did not cover fails the whole load.
pub fn check_against_declaration(
    declaration: &PluginDeclaration,
    already: &[String],
    element: &ValidatedElement,
) -> Result<(), ValidationError> {
    if already.len() >= MAX_ELEMENTS {
        return Err(ValidationError::TooManyElements);
    }
    if already.contains(&element.name) {
        return Err(ValidationError::DuplicateElement {
            name: element.name.clone(),
        });
    }
    match declaration.declared_kind(&element.name) {
        None => Err(ValidationError::UndeclaredElement {
            name: element.name.clone(),
        }),
        Some(declared) if declared != element.kind => Err(ValidationError::KindMismatch {
            name: element.name.clone(),
            declared,
            attempted: element.kind,
        }),
        Some(_) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::{FfiCapsSet, FfiElementMetadata, FfiStatus, MAX_STRING_LEN};
    use core::ffi::c_void;

    /// Safe face over [`validate_descriptor`] for the tests: every descriptor
    /// below is a live local (or a deliberate null), which is the whole
    /// contract, so the one `unsafe` lives here instead of at 15 call sites.
    fn read_descriptor(
        ptr: *const FfiPluginDescriptor,
    ) -> Result<PluginDeclaration, ValidationError> {
        // SAFETY: `ptr` is either null or a live local in this module, and any
        // hostile field values it carries are exactly what is under test.
        unsafe { validate_descriptor(ptr) }
    }

    /// As [`read_descriptor`], for [`validate_element`].
    fn read_registration(
        ptr: *const FfiElementRegistration,
    ) -> Result<ValidatedElement, ValidationError> {
        // SAFETY: `ptr` is a live local in this module.
        unsafe { validate_element(ptr) }
    }

    unsafe extern "C" fn stub_register(_r: *const FfiRegistrar) -> FfiStatus {
        super::super::STATUS_OK
    }
    unsafe extern "C" fn stub_create() -> *mut c_void {
        core::ptr::null_mut()
    }
    unsafe extern "C" fn stub_destroy(_e: *mut c_void) {}
    unsafe extern "C" fn stub_set_property(
        _e: *mut c_void,
        _n: FfiStr,
        _v: *const crate::abi::FfiPropValue,
    ) -> FfiStatus {
        super::super::STATUS_OK
    }
    unsafe extern "C" fn stub_process(
        _e: *mut c_void,
        _p: crate::abi::FfiPacket,
        _o: crate::abi::FfiOutputSink,
    ) -> async_ffi::LocalFfiFuture<FfiStatus> {
        async_ffi::LocalFfiFuture::new(async { super::super::STATUS_OK })
    }

    fn good_descriptor(caps: &'static [FfiCapability]) -> FfiPluginDescriptor {
        FfiPluginDescriptor {
            magic: V2_MAGIC,
            abi_version: V2_ABI_VERSION,
            struct_size: core::mem::size_of::<FfiPluginDescriptor>() as u32,
            name: FfiStr::borrowed("test-plugin"),
            version: FfiStr::borrowed("0.1.0"),
            capabilities: caps.as_ptr(),
            capability_count: caps.len(),
            register: Some(stub_register),
            reserved: [None; 4],
        }
    }

    const ONE_TRANSFORM: &[FfiCapability] = &[FfiCapability {
        kind: ELEMENT_TRANSFORM,
        reserved: 0,
        name: FfiStr::borrowed("countfilter"),
    }];

    #[test]
    fn a_well_formed_descriptor_validates() {
        let desc = good_descriptor(ONE_TRANSFORM);
        let decl = read_descriptor(&desc).expect("a good descriptor loads");
        assert_eq!(decl.name, "test-plugin");
        assert_eq!(
            decl.capabilities,
            [PluginCapability::Element {
                name: "countfilter".to_string(),
                kind: ElementKind::Transform,
            }]
        );
    }

    #[test]
    fn a_null_descriptor_is_refused() {
        let err = read_descriptor(core::ptr::null()).expect_err("null is refused");
        assert_eq!(err, ValidationError::NullDescriptor);
    }

    #[test]
    fn a_foreign_symbol_fails_on_magic_before_anything_else_is_read() {
        // The pointer here has a garbage magic and a garbage struct_size. The
        // magic check must fire first, so the bogus size is never believed.
        let mut desc = good_descriptor(ONE_TRANSFORM);
        desc.magic = 0xdead_beef_dead_beef;
        desc.struct_size = u32::MAX;
        let err = read_descriptor(&desc).expect_err("a foreign symbol is refused");
        assert_eq!(
            err,
            ValidationError::BadMagic {
                found: 0xdead_beef_dead_beef
            }
        );
    }

    #[test]
    fn a_different_abi_generation_is_refused() {
        let mut desc = good_descriptor(ONE_TRANSFORM);
        desc.abi_version = 99;
        let err = read_descriptor(&desc).expect_err("a foreign generation is refused");
        assert_eq!(
            err,
            ValidationError::AbiVersion {
                plugin: 99,
                host: V2_ABI_VERSION
            }
        );
    }

    #[test]
    fn a_descriptor_with_no_register_entry_is_refused() {
        let mut desc = good_descriptor(ONE_TRANSFORM);
        desc.register = None;
        let err = read_descriptor(&desc).expect_err("no register entry is refused");
        assert_eq!(err, ValidationError::MissingFunction { name: "register" });
    }

    #[test]
    fn an_absurd_capability_count_is_refused_before_the_pointer_is_read() {
        let mut desc = good_descriptor(ONE_TRANSFORM);
        desc.capability_count = usize::MAX;
        let err = read_descriptor(&desc).expect_err("an absurd count is refused");
        assert!(matches!(err, ValidationError::TooManyCapabilities { .. }));
    }

    #[test]
    fn a_null_capability_list_with_a_count_is_refused() {
        let mut desc = good_descriptor(ONE_TRANSFORM);
        desc.capabilities = core::ptr::null();
        desc.capability_count = 2;
        let err = read_descriptor(&desc).expect_err("a null list is refused");
        assert_eq!(err, ValidationError::NullCapabilities);
    }

    #[test]
    fn a_non_utf8_name_is_refused() {
        static BAD: &[u8] = &[0xff, 0xfe, 0xfd];
        let mut desc = good_descriptor(ONE_TRANSFORM);
        desc.name = FfiStr {
            ptr: BAD.as_ptr(),
            len: BAD.len(),
        };
        let err = read_descriptor(&desc).expect_err("non-UTF-8 is refused");
        assert_eq!(err, ValidationError::NotUtf8);
    }

    #[test]
    fn an_over_long_string_is_refused_before_it_is_read() {
        // The length is checked against MAX_STRING_LEN before the bytes are
        // touched, so a hostile length cannot walk off a short buffer.
        let mut desc = good_descriptor(ONE_TRANSFORM);
        desc.name = FfiStr {
            ptr: b"x".as_ptr(),
            len: MAX_STRING_LEN + 1,
        };
        let err = read_descriptor(&desc).expect_err("an over-long string is refused");
        assert_eq!(
            err,
            ValidationError::StringTooLong {
                len: MAX_STRING_LEN + 1
            }
        );
    }

    #[test]
    fn a_string_with_a_length_and_a_null_pointer_is_refused() {
        let mut desc = good_descriptor(ONE_TRANSFORM);
        desc.name = FfiStr {
            ptr: core::ptr::null(),
            len: 4,
        };
        let err = read_descriptor(&desc).expect_err("a null string is refused");
        assert_eq!(err, ValidationError::NullString);
    }

    #[test]
    fn an_unknown_capability_kind_survives_as_a_policy_decision() {
        // Forward compatibility: a kind this host cannot act on is reported,
        // not fatal. Nothing can register under it, so allowing one is inert.
        const ODD: &[FfiCapability] = &[FfiCapability {
            kind: 4242,
            reserved: 0,
            name: FfiStr::borrowed("future"),
        }];
        let desc = good_descriptor(ODD);
        let decl = read_descriptor(&desc).expect("an unknown kind is not fatal");
        assert_eq!(
            decl.capabilities,
            [PluginCapability::Unknown {
                kind: 4242,
                name: "future".to_string()
            }]
        );
    }

    #[test]
    fn hostile_element_names_are_refused() {
        for bad in [
            "",
            "Has Spaces",
            "line\nbreak",
            "UPPER",
            "9leading",
            "semi;colon",
        ] {
            assert!(
                check_name(bad).is_err(),
                "'{}' must not be accepted as a name",
                bad.escape_debug()
            );
        }
        check_name("countfilter").expect("a plain name is fine");
        check_name("h264-parse_2").expect("digits, dash and underscore are fine");
    }

    fn good_vtable() -> FfiElementVtable {
        FfiElementVtable {
            struct_size: core::mem::size_of::<FfiElementVtable>() as u32,
            version: 1,
            configure_pipeline: None,
            configure_output: None,
            process: Some(stub_process),
            set_property: None,
            get_property: None,
            destroy: Some(stub_destroy),
            reserved: [None; 6],
        }
    }

    fn good_registration(vtable: &FfiElementVtable) -> FfiElementRegistration {
        FfiElementRegistration {
            struct_size: core::mem::size_of::<FfiElementRegistration>() as u32,
            kind: ELEMENT_TRANSFORM,
            name: FfiStr::borrowed("countfilter"),
            metadata: FfiElementMetadata::EMPTY,
            sink_caps: FfiCapsSet::EMPTY,
            source_caps: FfiCapsSet::EMPTY,
            properties: core::ptr::null(),
            property_count: 0,
            vtable,
            create: Some(stub_create),
            reserved: [None; 4],
        }
    }

    #[test]
    fn a_well_formed_registration_validates() {
        let vt = good_vtable();
        let reg = good_registration(&vt);
        let elem = read_registration(&reg).expect("a good registration validates");
        assert_eq!(elem.name, "countfilter");
        assert_eq!(elem.kind, ElementKind::Transform);
        assert!(elem.sink_caps.is_empty(), "an empty sink set means any");
    }

    #[test]
    fn a_registration_without_process_is_refused() {
        let mut vt = good_vtable();
        vt.process = None;
        let reg = good_registration(&vt);
        let err = read_registration(&reg).expect_err("no process entry is refused");
        assert_eq!(err, ValidationError::MissingFunction { name: "process" });
    }

    #[test]
    fn a_registration_without_destroy_is_refused() {
        // Without it the host would leak every instance it builds, and worse,
        // would have no way to tell the plugin its state is finished with.
        let mut vt = good_vtable();
        vt.destroy = None;
        let reg = good_registration(&vt);
        let err = read_registration(&reg).expect_err("no destroy entry is refused");
        assert_eq!(err, ValidationError::MissingFunction { name: "destroy" });
    }

    #[test]
    fn a_null_vtable_is_refused() {
        let vt = good_vtable();
        let mut reg = good_registration(&vt);
        reg.vtable = core::ptr::null();
        let err = read_registration(&reg).expect_err("a null vtable is refused");
        assert_eq!(err, ValidationError::NullVtable);
    }

    #[test]
    fn a_short_vtable_loads_with_host_defaults_for_what_it_lacks() {
        // The compatibility promise: a plugin that wrote only the two required
        // entries loads, and everything past its declared size comes back absent
        // rather than as a garbage function pointer read off the end.
        let mut vt = good_vtable();
        vt.set_property = Some(stub_set_property);
        vt.struct_size = core::mem::offset_of!(FfiElementVtable, configure_pipeline) as u32;
        let reg = good_registration(&vt);
        let elem = read_registration(&reg).expect("a minimal vtable is enough");
        assert!(elem.vtable.process.is_some());
        assert!(elem.vtable.destroy.is_some());
        assert!(
            elem.vtable.set_property.is_none(),
            "an entry past the declared size is not read, even though it is set"
        );
        assert!(elem.vtable.configure_pipeline.is_none());
        assert!(elem.vtable.reserved.iter().all(Option::is_none));
    }

    #[test]
    fn a_flags_property_is_refused() {
        // The flag-set kind deliberately does not cross v2; an element that
        // declares one is refused rather than silently losing the property.
        const SPEC: &[FfiPropertySpec] = &[FfiPropertySpec {
            name: FfiStr::borrowed("mode"),
            kind: 99,
            readable: 1,
            writable: 1,
            reserved: 0,
            blurb: FfiStr::EMPTY,
            default_value: FfiStr::EMPTY,
        }];
        let vt = good_vtable();
        let mut reg = good_registration(&vt);
        reg.properties = SPEC.as_ptr();
        reg.property_count = SPEC.len();
        let err = read_registration(&reg).expect_err("a flags property is refused");
        assert_eq!(err, ValidationError::BadPropertyKind { kind: 99 });
    }

    #[test]
    fn an_absurd_property_count_is_refused_before_the_pointer_is_read() {
        let vt = good_vtable();
        let mut reg = good_registration(&vt);
        reg.properties = core::ptr::null();
        reg.property_count = usize::MAX;
        let err = read_registration(&reg).expect_err("an absurd count is refused");
        assert!(matches!(err, ValidationError::TooManyProperties { .. }));
    }

    #[test]
    fn an_unfixable_source_caps_set_is_refused() {
        use g2g_core::caps::{Dim, Interlace, Rate, RawVideoFormat};
        // A source set with `Any` geometry becomes the wrapper's intercept_caps
        // answer, which the solver cannot fixate. Catch it at registration.
        let ffi = crate::abi::caps_into_ffi(&g2g_core::Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        })
        .expect("rgba crosses");
        let alternatives = [ffi];
        let vt = good_vtable();
        let mut reg = good_registration(&vt);
        reg.source_caps = FfiCapsSet {
            alternatives: alternatives.as_ptr(),
            count: 1,
        };
        let err = read_registration(&reg).expect_err("an unfixable source set is refused");
        assert_eq!(err, ValidationError::UnfixableSourceCaps);
    }

    #[test]
    fn registering_an_undeclared_element_is_refused() {
        let desc = good_descriptor(ONE_TRANSFORM);
        let decl = read_descriptor(&desc).expect("descriptor validates");
        let vt = good_vtable();
        let mut reg = good_registration(&vt);
        reg.name = FfiStr::borrowed("sneaky");
        let elem = read_registration(&reg).expect("the registration itself is well formed");
        let err = check_against_declaration(&decl, &[], &elem)
            .expect_err("an undeclared element is refused");
        assert_eq!(
            err,
            ValidationError::UndeclaredElement {
                name: "sneaky".to_string()
            }
        );
    }

    #[test]
    fn registering_a_declared_name_under_another_kind_is_refused() {
        let desc = good_descriptor(ONE_TRANSFORM);
        let decl = read_descriptor(&desc).expect("descriptor validates");
        let vt = good_vtable();
        let mut reg = good_registration(&vt);
        reg.kind = ELEMENT_SINK;
        let elem = read_registration(&reg).expect("registration validates");
        let err =
            check_against_declaration(&decl, &[], &elem).expect_err("a kind mismatch is refused");
        assert_eq!(
            err,
            ValidationError::KindMismatch {
                name: "countfilter".to_string(),
                declared: ElementKind::Transform,
                attempted: ElementKind::Sink,
            }
        );
    }

    #[test]
    fn registering_the_same_name_twice_is_refused() {
        let desc = good_descriptor(ONE_TRANSFORM);
        let decl = read_descriptor(&desc).expect("descriptor validates");
        let vt = good_vtable();
        let reg = good_registration(&vt);
        let elem = read_registration(&reg).expect("registration validates");
        let already = ["countfilter".to_string()];
        let err = check_against_declaration(&decl, &already, &elem)
            .expect_err("a repeated name is refused");
        assert_eq!(
            err,
            ValidationError::DuplicateElement {
                name: "countfilter".to_string()
            }
        );
    }

    #[test]
    fn a_declared_registration_passes_the_gate() {
        let desc = good_descriptor(ONE_TRANSFORM);
        let decl = read_descriptor(&desc).expect("descriptor validates");
        let vt = good_vtable();
        let reg = good_registration(&vt);
        let elem = read_registration(&reg).expect("registration validates");
        assert_eq!(check_against_declaration(&decl, &[], &elem), Ok(()));
    }
}
