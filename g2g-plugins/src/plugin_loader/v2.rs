//! Host side of the **v2 plugin ABI**: the wrapper element that drives a
//! plugin's `repr(C)` vtable as a normal [`AsyncElement`], and the registration
//! plumbing the loader hands the plugin.
//!
//! The frozen boundary types live in [`g2g_plugin::abi`]; everything here is
//! the host's half of the contract.
//!
//! **What the wrapper answers itself.** A v2 vtable carries six entry points.
//! Every other `AsyncElement` hook (the clock election, QoS, metadata
//! propagation, the allocation cascade, the reverse-channel signals) keeps its
//! trait default, so a v2 element is a plain System-memory transform or sink and
//! nothing more. [`AsyncElement::input_domains`] is narrowed to System, which is
//! what makes that structural: a GPU-resident upstream gets a domain converter
//! spliced in front rather than a frame the plugin cannot read.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::ffi::c_void;
use core::task::Poll;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Mutex, OnceLock};

use async_ffi::{FfiContext, FfiPoll};

use g2g_core::caps::{Caps, CapsSet};
use g2g_core::element::{AsyncElement, BoxFuture, ConfigureOutcome, DynAsyncElement, OutputSink};
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::pad_template::PadTemplate;
use g2g_core::property::{ElementMetadata, PropError, PropValue, PropertySpec};
use g2g_core::runtime::{LaunchFactory, Registry};
use g2g_core::{G2gError, PipelinePacket};

use g2g_plugin::abi::{
    caps_from_ffi, caps_into_ffi, check_against_declaration, packet_from_ffi, packet_into_ffi,
    prop_from_ffi, prop_into_ffi, validate_element, ElementKind, FfiCaps, FfiElementRegistration,
    FfiElementVtable, FfiOutputSink, FfiOutputSinkVtable, FfiPacket, FfiPropValue, FfiRegistrar,
    FfiStatus, FfiStr, PluginDeclaration, ValidatedElement, ValidationError, CAPS_NONE,
    PACKET_NONE, STATUS_ERROR, STATUS_OK, STATUS_PROPERTY_UNKNOWN, STATUS_PROPERTY_VALUE,
};

/// How many v2 elements one process can register.
///
/// The `gst-launch` registry builds an element from a bare `fn()` pointer with
/// no context argument, so each v2 element needs its own trampoline, and the
/// trampolines have to exist at compile time. A fixed table of 64 is the honest
/// consequence: past it the loader refuses rather than silently dropping an
/// element. Nothing frees a slot, because a loaded plugin's code is never
/// unmapped.
pub const MAX_V2_ELEMENT_SLOTS: usize = 64;

/// One registered v2 element type, shared by every instance the registry builds.
#[derive(Debug)]
struct ElementSlot {
    vtable: FfiElementVtable,
    create: unsafe extern "C" fn() -> *mut c_void,
    sink_caps: CapsSet,
    source_caps: CapsSet,
    properties: &'static [PropertySpec],
    metadata: ElementMetadata,
    log_category: &'static str,
}

fn slots() -> &'static Mutex<Vec<&'static ElementSlot>> {
    static SLOTS: OnceLock<Mutex<Vec<&'static ElementSlot>>> = OnceLock::new();
    SLOTS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Install a validated element into the next free slot, returning its index.
fn claim_slot(element: ValidatedElement, name: &'static str) -> Option<usize> {
    let slot: &'static ElementSlot = Box::leak(Box::new(ElementSlot {
        vtable: element.vtable,
        create: element.create,
        sink_caps: element.sink_caps,
        source_caps: element.source_caps,
        properties: Box::leak(element.properties.into_boxed_slice()),
        metadata: element.metadata,
        log_category: name,
    }));
    let mut slots = slots().lock().expect("v2 element slot table poisoned");
    if slots.len() >= MAX_V2_ELEMENT_SLOTS {
        return None;
    }
    slots.push(slot);
    Some(slots.len() - 1)
}

fn slot(index: usize) -> &'static ElementSlot {
    slots().lock().expect("v2 element slot table poisoned")[index]
}

/// One instantiation of the registry's context-free `fn()` constructor, bound
/// to a slot by its const parameter. See [`MAX_V2_ELEMENT_SLOTS`].
fn build_slot<const INDEX: usize>() -> Box<dyn DynAsyncElement> {
    Box::new(V2Element::new(slot(INDEX)))
}

macro_rules! slot_builders {
    ($($index:literal),* $(,)?) => {
        [ $( build_slot::<$index> as fn() -> Box<dyn DynAsyncElement> ),* ]
    };
}

static SLOT_BUILDERS: [fn() -> Box<dyn DynAsyncElement>; MAX_V2_ELEMENT_SLOTS] = slot_builders![
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
];

/// The pad templates a v2 element's declared caps sets amount to.
fn pad_templates(slot: &ElementSlot, kind: ElementKind) -> Vec<PadTemplate> {
    let mut out = Vec::new();
    out.push(if slot.sink_caps.is_empty() {
        PadTemplate::sink_any()
    } else {
        PadTemplate::sink(slot.sink_caps.clone())
    });
    if kind == ElementKind::Transform {
        let produced = if slot.source_caps.is_empty() {
            slot.sink_caps.clone()
        } else {
            slot.source_caps.clone()
        };
        out.push(PadTemplate::source(produced));
    }
    out
}

/// Build the registry factory for a slot the loader has just claimed.
fn launch_factory(index: usize, name: &'static str, kind: ElementKind) -> LaunchFactory {
    LaunchFactory::new(name, pad_templates(slot(index), kind), SLOT_BUILDERS[index])
}

// ---------------------------------------------------------------------------
// The wrapper element
// ---------------------------------------------------------------------------

/// A plugin element instance, driven through its vtable.
#[derive(Debug)]
struct V2Element {
    /// The plugin's opaque instance, or null when `create` failed. A null
    /// instance is not silently tolerated: `configure_pipeline` refuses, so the
    /// pipeline fails to start rather than running a no-op element.
    instance: *mut c_void,
    slot: &'static ElementSlot,
}

// SAFETY: the runner owns an element exclusively (it is moved into one arm and
// borrowed `&mut` for every call), so `instance` is never touched from two
// threads at once, and the multi-thread runner only *moves* it between threads.
// The plugin contract in `g2g_plugin_v2.h` states the same requirement: an
// element instance must not be thread-affine. This is an assumption about
// plugin code the host cannot verify, and it is the reason a v2 plugin that
// binds its state to a specific thread (a COM apartment, a GL context) is
// outside the ABI's contract.
unsafe impl Send for V2Element {}

impl V2Element {
    fn new(slot: &'static ElementSlot) -> V2Element {
        // SAFETY: `create` was validated non-null, and the plugin's contract is
        // that it builds one instance or returns null.
        let instance = unsafe { (slot.create)() };
        V2Element { instance, slot }
    }

    fn require_instance(&self) -> Result<*mut c_void, G2gError> {
        if self.instance.is_null() {
            return Err(G2gError::Hardware(g2g_core::error::HardwareError::Other));
        }
        Ok(self.instance)
    }

    /// Hand `caps` to the plugin's optional `configure_*` entry, converting
    /// them first. Caps v2 cannot express are refused here rather than at the
    /// first frame, so a v2 element only ever runs on caps it can see.
    fn configure(
        &mut self,
        caps: &Caps,
        entry: Option<unsafe extern "C" fn(*mut c_void, *const FfiCaps, *mut FfiCaps) -> FfiStatus>,
        refixate: Option<&mut FfiCaps>,
    ) -> Result<(), G2gError> {
        let instance = self.require_instance()?;
        let ffi = caps_into_ffi(caps).map_err(|_| G2gError::CapsMismatch)?;
        let Some(entry) = entry else {
            return Ok(());
        };
        let mut scratch = FfiCaps::NONE;
        let out = match refixate {
            Some(slot) => slot as *mut FfiCaps,
            None => &mut scratch as *mut FfiCaps,
        };
        // SAFETY: `instance` came from this element's `create`, `ffi` is a live
        // local, and `out` addresses a live `FfiCaps` the callee may overwrite.
        let status = unsafe { entry(instance, &ffi, out) };
        status.into_error().map_or(Ok(()), Err)
    }
}

impl Drop for V2Element {
    fn drop(&mut self) {
        if self.instance.is_null() {
            return;
        }
        if let Some(destroy) = self.slot.vtable.destroy {
            // SAFETY: `destroy` was validated non-null at registration, and
            // this runs once, from the single owner, at the end of the
            // instance's life.
            unsafe { destroy(self.instance) };
        }
    }
}

impl AsyncElement for V2Element {
    type ProcessFuture<'a>
        = BoxFuture<'a, Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if !self.slot.sink_caps.is_empty() && !self.slot.sink_caps.accepts(upstream_caps) {
            return Err(G2gError::CapsMismatch);
        }
        match self.slot.source_caps.alternatives().first() {
            Some(produced) => Ok(produced.clone()),
            None => Ok(upstream_caps.clone()),
        }
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let mut refixate = FfiCaps::NONE;
        let entry = self.slot.vtable.configure_pipeline;
        self.configure(absolute_caps, entry, Some(&mut refixate))?;
        if refixate.tag == CAPS_NONE {
            return Ok(ConfigureOutcome::Accepted);
        }
        let proposal = caps_from_ffi(&refixate).map_err(|_| G2gError::CapsMismatch)?;
        Ok(ConfigureOutcome::ReFixate(proposal))
    }

    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        let Some(entry) = self.slot.vtable.configure_output else {
            return Ok(());
        };
        let instance = self.require_instance()?;
        let ffi = caps_into_ffi(output_caps).map_err(|_| G2gError::CapsMismatch)?;
        // SAFETY: `instance` came from this element's `create` and `ffi` is a
        // live local borrowed only for the call.
        let status = unsafe { entry(instance, &ffi) };
        status.into_error().map_or(Ok(()), Err)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        // Segment and Tick never reach a v2 plugin: the segment is forwarded
        // unchanged (a v2 element does not remap time) and a tick is only ever
        // delivered to a fan-in element, which v2 has no way to be.
        match packet {
            PipelinePacket::Segment(_) => {
                return Box::pin(async move { out.push(packet).await.map(|_| ()) })
            }
            PipelinePacket::Tick => return Box::pin(async move { Ok(()) }),
            _ => {}
        }

        let instance = self.instance;
        let process = self.slot.vtable.process;
        Box::pin(async move {
            if instance.is_null() {
                return Err(G2gError::Hardware(g2g_core::error::HardwareError::Other));
            }
            let Some(process) = process else {
                return Err(G2gError::Hardware(g2g_core::error::HardwareError::Other));
            };
            let ffi_packet = packet_into_ffi(packet)?;

            // Boxed so the plugin's raw `ctx` pointer addresses a stable heap
            // slot rather than a local whose place in this future's state the
            // compiler is free to choose.
            let mut shim = Box::new(SinkShim { out, pending: None });
            let sink = FfiOutputSink {
                ctx: (&mut *shim as *mut SinkShim) as *mut c_void,
                vtable: &SINK_VTABLE,
            };
            // SAFETY: `instance` came from this element's `create`, the packet's
            // payload ownership passes to the plugin with the struct, and `sink`
            // stays valid because `shim` is dropped only after the returned
            // future has completed below.
            let future = unsafe { process(instance, ffi_packet, sink) };
            let status = future.await;
            drop(shim);
            status.into_error().map_or(Ok(()), Err)
        })
    }

    /// A v2 element reads frame bytes through a plain pointer, so it can only
    /// be fed System memory. Declaring that lets the runner splice a domain
    /// converter in front of a GPU-resident producer instead of handing the
    /// plugin a frame it cannot read.
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn is_format_boundary(&self) -> bool {
        !self.slot.source_caps.is_empty()
    }

    fn propose_output_caps(&self, input: &Caps) -> Caps {
        match self.slot.source_caps.alternatives().first() {
            Some(produced) => produced.clone(),
            None => input.clone(),
        }
    }

    fn properties(&self) -> &'static [PropertySpec] {
        self.slot.properties
    }

    fn metadata(&self) -> ElementMetadata {
        self.slot.metadata
    }

    fn log_category(&self) -> &'static str {
        self.slot.log_category
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let Some(entry) = self.slot.vtable.set_property else {
            return Err(PropError::Unknown);
        };
        if self.instance.is_null() {
            return Err(PropError::Unknown);
        }
        let ffi = prop_into_ffi(&value).ok_or(PropError::Type)?;
        // SAFETY: `name` and the borrowed string inside `ffi` outlive the call,
        // and `instance` came from this element's `create`.
        let status = unsafe { entry(self.instance, FfiStr::borrowed(name), &ffi) };
        match status {
            STATUS_OK => Ok(()),
            STATUS_PROPERTY_UNKNOWN => Err(PropError::Unknown),
            STATUS_PROPERTY_VALUE => Err(PropError::Value),
            _ => Err(PropError::Value),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let entry = self.slot.vtable.get_property?;
        if self.instance.is_null() {
            return None;
        }
        let mut out = FfiPropValue::NONE;
        // SAFETY: `name` outlives the call and `out` is a live local the callee
        // may write.
        let status = unsafe { entry(self.instance, FfiStr::borrowed(name), &mut out) };
        if status != STATUS_OK {
            return None;
        }
        // SAFETY: the plugin reported success, so `out` holds a value tagged by
        // its `kind`; `prop_from_ffi` reads only the member that tag selects and
        // releases an owned string before returning.
        unsafe { prop_from_ffi(&out) }
    }
}

// ---------------------------------------------------------------------------
// The output sink the plugin pushes through
// ---------------------------------------------------------------------------

/// Host state behind [`FfiOutputSink::ctx`]: the real downstream sink plus the
/// packet slot a multi-poll push needs.
struct SinkShim<'a> {
    out: &'a mut dyn OutputSink,
    /// The packet taken from the plugin, held across pending polls. Dropping
    /// the shim with a packet still here releases it, which is what happens
    /// when the plugin abandons a push.
    pending: Option<PipelinePacket>,
}

static SINK_VTABLE: FfiOutputSinkVtable = FfiOutputSinkVtable {
    struct_size: core::mem::size_of::<FfiOutputSinkVtable>() as u32,
    version: 1,
    poll_push: shim_poll_push,
    reserved: [None; 4],
};

/// Drive one packet from a plugin toward downstream.
///
/// Ownership of the packet moves into the host on the **first** poll: the slot
/// is cleared to [`PACKET_NONE`] there and the packet is held in the shim until
/// downstream commits it. A re-poll therefore passes an already-empty slot,
/// which is why the shim's own `pending` is what decides whether there is work,
/// not the slot's tag.
///
/// # Safety
/// `ctx` must be the [`SinkShim`] pointer the host installed, `cx` a live
/// context, and `packet` a live slot the plugin owns.
unsafe extern "C" fn shim_poll_push(
    ctx: *mut c_void,
    cx: *mut FfiContext,
    packet: *mut FfiPacket,
) -> FfiPoll<FfiStatus> {
    if ctx.is_null() || cx.is_null() {
        return FfiPoll::Ready(STATUS_ERROR);
    }
    // A panic must not unwind into the plugin's frame: that is undefined
    // behaviour across `extern "C"`, whatever language the plugin is in.
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: forwarded from this function's contract.
        let shim = unsafe { &mut *ctx.cast::<SinkShim>() };
        if shim.pending.is_none() {
            if packet.is_null() {
                return FfiPoll::Ready(STATUS_ERROR);
            }
            // SAFETY: as above; `FfiPacket` is `Copy`.
            let taken = unsafe { *packet };
            if taken.tag == PACKET_NONE {
                return FfiPoll::Ready(STATUS_OK);
            }
            // Clear the slot before the conversion can fail, so the plugin can
            // never free a payload the host has already taken.
            // SAFETY: as above.
            unsafe { (*packet).tag = PACKET_NONE };
            // SAFETY: the plugin filled this packet and transferred its payload.
            match unsafe { packet_from_ffi(taken) } {
                Ok(converted) => {
                    shim.out.begin_push();
                    shim.pending = Some(converted);
                }
                Err(status) => return FfiPoll::Ready(status),
            }
        }
        let SinkShim { out, pending } = shim;
        // SAFETY: `cx` is a live context borrowed for this call.
        unsafe { &mut *cx }.with_context(|rust_cx| match out.poll_push(rust_cx, pending) {
            Poll::Pending => FfiPoll::Pending,
            Poll::Ready(Ok(_)) => FfiPoll::Ready(STATUS_OK),
            Poll::Ready(Err(e)) => FfiPoll::Ready(FfiStatus::from_error(&e)),
        })
    }));
    outcome.unwrap_or(FfiPoll::Ready(STATUS_ERROR))
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Host state behind [`FfiRegistrar::ctx`] while a plugin registers.
///
/// Registrations are **staged**, not committed: the loader writes into the
/// caller's `Registry` only once the plugin's `register` has returned cleanly
/// and every element it attempted matched the declaration. A plugin that
/// registers three good elements and one undeclared one contributes nothing.
struct RegistrarCtx<'d> {
    declaration: &'d PluginDeclaration,
    staged: Vec<ValidatedElement>,
    names: Vec<String>,
    error: Option<ValidationError>,
}

/// # Safety
/// `ctx` must be the [`RegistrarCtx`] the host installed, and `element` a
/// registration the plugin filled in.
unsafe extern "C" fn registrar_register_element(
    ctx: *mut c_void,
    element: *const FfiElementRegistration,
) -> FfiStatus {
    if ctx.is_null() {
        return STATUS_ERROR;
    }
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: forwarded from this function's contract.
        let state = unsafe { &mut *ctx.cast::<RegistrarCtx>() };
        if state.error.is_some() {
            return STATUS_ERROR;
        }
        // SAFETY: as above; every field is validated before it is trusted.
        let validated = match unsafe { validate_element(element) } {
            Ok(v) => v,
            Err(e) => {
                state.error = Some(e);
                return STATUS_ERROR;
            }
        };
        if let Err(e) = check_against_declaration(state.declaration, &state.names, &validated) {
            state.error = Some(e);
            return STATUS_ERROR;
        }
        state.names.push(validated.name.clone());
        state.staged.push(validated);
        STATUS_OK
    }));
    outcome.unwrap_or(STATUS_ERROR)
}

/// What a plugin registered, once its `register` entry has returned.
#[derive(Debug)]
pub(super) struct Registered {
    /// The elements it staged, in registration order.
    pub elements: Vec<ValidatedElement>,
}

/// Call a validated plugin's `register` entry and collect what it staged.
///
/// # Safety
/// `declaration` must come from `validate_descriptor` on a library that is
/// still loaded.
pub(super) unsafe fn run_registration(
    declaration: &PluginDeclaration,
) -> Result<Registered, RegistrationFailure> {
    let mut state = RegistrarCtx {
        declaration,
        staged: Vec::new(),
        names: Vec::new(),
        error: None,
    };
    let registrar = FfiRegistrar {
        struct_size: core::mem::size_of::<FfiRegistrar>() as u32,
        version: 1,
        ctx: (&mut state as *mut RegistrarCtx) as *mut c_void,
        register_element: registrar_register_element,
        reserved: [None; 4],
    };
    // SAFETY: `register` was validated non-null and the library is loaded; the
    // registrar it receives is a live local that outlives the call.
    let status = unsafe { (declaration.register)(&registrar) };

    if let Some(error) = state.error {
        return Err(RegistrationFailure::Invalid(error));
    }
    if status != STATUS_OK {
        return Err(RegistrationFailure::Status(status.0));
    }
    Ok(Registered {
        elements: state.staged,
    })
}

/// Why a plugin's `register` call did not yield a usable element set.
#[derive(Debug)]
pub(super) enum RegistrationFailure {
    /// Something the plugin handed the registrar was refused.
    Invalid(ValidationError),
    /// The plugin's own `register` reported failure.
    Status(i32),
    /// The process has no free element slot left.
    NoSlots,
}

/// Commit staged elements into the caller's registry, one slot each.
pub(super) fn commit(
    registered: Registered,
    reg: &mut Registry,
) -> Result<Vec<String>, RegistrationFailure> {
    // Claim every slot first: a half-committed plugin would leave the registry
    // holding elements from a load that failed.
    let mut claimed = Vec::with_capacity(registered.elements.len());
    for element in registered.elements {
        let name: &'static str = Box::leak(element.name.clone().into_boxed_str());
        let kind = element.kind;
        let Some(index) = claim_slot(element, name) else {
            return Err(RegistrationFailure::NoSlots);
        };
        claimed.push((index, name, kind));
    }
    let mut names = Vec::with_capacity(claimed.len());
    for (index, name, kind) in claimed {
        reg.register_launch(launch_factory(index, name, kind));
        names.push(name.to_string());
    }
    Ok(names)
}
