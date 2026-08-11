//! Plugin side of the v2 ABI: everything [`declare_plugin_v2!`] needs to turn a
//! plain [`AsyncElement`] into a `repr(C)` vtable.
//!
//! A plugin author writes a normal element and one macro invocation. The shims
//! below are what the macro instantiates: they convert at the boundary, keep
//! panics from unwinding across `extern "C"`, and publish the element's caps,
//! properties, and metadata as the data the host validates.

use std::boxed::Box;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::vec::Vec;

use core::ffi::c_void;
use core::task::{Context, Poll};

use async_ffi::{ContextExt, LocalFfiFuture};

use g2g_core::element::{AsyncElement, ConfigureOutcome, OutputSink, PushOutcome};
use g2g_core::pad_template::{PadCaps, PadDirection, PadTemplate, PadTemplates};
use g2g_core::{G2gError, PipelinePacket};

use crate::abi::{
    caps_from_ffi, caps_into_ffi, packet_from_ffi, packet_into_ffi, prop_from_ffi, prop_into_ffi,
    release_packet, spec_into_ffi, FfiCaps, FfiCapsSet, FfiElementMetadata, FfiElementRegistration,
    FfiElementVtable, FfiOutputSink, FfiPacket, FfiPropValue, FfiPropertySpec, FfiRegistrar,
    FfiStatus, FfiStr, STATUS_ERROR, STATUS_OK, STATUS_PROPERTY_UNKNOWN, STATUS_PROPERTY_VALUE,
};

/// The host's downstream, as an [`OutputSink`] the element can `await`.
///
/// The mirror image of the loader's shim: that one wraps a real `OutputSink` in
/// the C vtable, this one wraps the C vtable in a real `OutputSink`.
#[derive(Debug)]
pub struct HostSink {
    sink: FfiOutputSink,
    /// The packet handed to the host, kept across pending polls. Held here (not
    /// in the caller's slot) because the host takes ownership on the first poll
    /// and clears the slot.
    staged: Option<FfiPacket>,
}

impl HostSink {
    /// Wrap the sink the host passed to `process`.
    pub fn new(sink: FfiOutputSink) -> HostSink {
        HostSink { sink, staged: None }
    }

    fn discard_staged(&mut self) {
        if let Some(packet) = self.staged.take() {
            // SAFETY: a staged packet is one this side still owns: the host
            // takes ownership only when its poll returns ready, and a staged
            // packet by definition has not reached that point.
            unsafe { release_packet(&packet) };
        }
    }
}

impl Drop for HostSink {
    fn drop(&mut self) {
        self.discard_staged();
    }
}

impl OutputSink for HostSink {
    fn poll_push(
        &mut self,
        cx: &mut Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        if self.staged.is_none() {
            let Some(taken) = packet.take() else {
                return Poll::Ready(Ok(PushOutcome::Accepted));
            };
            match packet_into_ffi(taken) {
                Ok(converted) => self.staged = Some(converted),
                Err(e) => return Poll::Ready(Err(e)),
            }
        }
        let HostSink { sink, staged } = self;
        let slot = staged.as_mut().expect("just staged");
        let poll = cx.with_ffi_context(|ffi_cx| {
            // SAFETY: the host keeps its sink vtable and context valid for as
            // long as the `process` future it handed them to.
            unsafe { ((*sink.vtable).poll_push)(sink.ctx, ffi_cx, slot) }
        });
        match poll.try_into_poll() {
            Ok(Poll::Pending) => Poll::Pending,
            Ok(Poll::Ready(status)) => {
                // The host owns the packet once it answers; forget it here so
                // `discard_staged` cannot free it a second time.
                *staged = None;
                match status.into_error() {
                    None => Poll::Ready(Ok(PushOutcome::Accepted)),
                    Some(e) => Poll::Ready(Err(e)),
                }
            }
            // The host's push panicked. It cannot unwind across the boundary,
            // so it comes back as a status; the element treats it as shutdown.
            Err(_) => {
                *staged = None;
                Poll::Ready(Err(G2gError::Shutdown))
            }
        }
    }

    fn begin_push(&mut self) {
        // A packet still staged here belongs to a push the element abandoned.
        self.discard_staged();
    }
}

/// # Safety
/// Called by the host to build one instance, paired with [`destroy_shim`].
pub unsafe extern "C" fn create_shim<E: AsyncElement + Default + 'static>() -> *mut c_void {
    match catch_unwind(|| Box::into_raw(Box::new(E::default())).cast::<c_void>()) {
        Ok(ptr) => ptr,
        Err(_) => core::ptr::null_mut(),
    }
}

/// # Safety
/// `elem` must be a pointer [`create_shim`] returned for the same `E`,
/// destroyed exactly once.
pub unsafe extern "C" fn destroy_shim<E: AsyncElement + 'static>(elem: *mut c_void) {
    if elem.is_null() {
        return;
    }
    // SAFETY: forwarded from this function's contract.
    let owned = unsafe { Box::from_raw(elem.cast::<E>()) };
    let _ = catch_unwind(AssertUnwindSafe(move || drop(owned)));
}

/// # Safety
/// `elem` must be a live instance of `E`, `caps` a readable value, and
/// `refixate` a writable slot.
pub unsafe extern "C" fn configure_pipeline_shim<E: AsyncElement + 'static>(
    elem: *mut c_void,
    caps: *const FfiCaps,
    refixate: *mut FfiCaps,
) -> FfiStatus {
    if elem.is_null() || caps.is_null() {
        return STATUS_ERROR;
    }
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: forwarded from this function's contract.
        let element = unsafe { &mut *elem.cast::<E>() };
        // SAFETY: as above.
        let Ok(caps) = caps_from_ffi(unsafe { &*caps }) else {
            return FfiStatus::from_error(&G2gError::CapsMismatch);
        };
        match element.configure_pipeline(&caps) {
            Ok(ConfigureOutcome::Accepted) => STATUS_OK,
            Ok(ConfigureOutcome::ReFixate(proposal)) => match caps_into_ffi(&proposal) {
                Ok(ffi) if !refixate.is_null() => {
                    // SAFETY: `refixate` is non-null and the host's to write.
                    unsafe { *refixate = ffi };
                    STATUS_OK
                }
                _ => FfiStatus::from_error(&G2gError::CapsMismatch),
            },
            Err(e) => FfiStatus::from_error(&e),
        }
    }));
    outcome.unwrap_or(STATUS_ERROR)
}

/// # Safety
/// `elem` must be a live instance of `E` and `caps` a readable value.
pub unsafe extern "C" fn configure_output_shim<E: AsyncElement + 'static>(
    elem: *mut c_void,
    caps: *const FfiCaps,
) -> FfiStatus {
    if elem.is_null() || caps.is_null() {
        return STATUS_ERROR;
    }
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: forwarded from this function's contract.
        let element = unsafe { &mut *elem.cast::<E>() };
        // SAFETY: as above.
        let Ok(caps) = caps_from_ffi(unsafe { &*caps }) else {
            return FfiStatus::from_error(&G2gError::CapsMismatch);
        };
        match element.configure_output(&caps) {
            Ok(()) => STATUS_OK,
            Err(e) => FfiStatus::from_error(&e),
        }
    }));
    outcome.unwrap_or(STATUS_ERROR)
}

/// # Safety
/// `elem` must be a live instance of `E`, and `out` a sink the host keeps valid
/// until the returned future is dropped. The host must not touch the instance
/// while that future is alive.
pub unsafe extern "C" fn process_shim<E: AsyncElement + 'static>(
    elem: *mut c_void,
    packet: FfiPacket,
    out: FfiOutputSink,
) -> LocalFfiFuture<FfiStatus> {
    // Convert before building the future, not inside it. The raw `FfiPacket` has
    // no destructor, so a host that drops the returned future without ever
    // polling it would leak the payload; the converted `PipelinePacket` releases
    // itself when the future is dropped, polled or not.
    // SAFETY: payload ownership arrived with the packet.
    let converted = unsafe { packet_from_ffi(packet) };
    LocalFfiFuture::new(async move {
        let packet = match converted {
            Ok(p) => p,
            Err(status) => return status,
        };
        if elem.is_null() {
            return STATUS_ERROR;
        }
        // SAFETY: the host's contract is exclusive access for the life of this
        // future.
        let element = unsafe { &mut *elem.cast::<E>() };
        let mut sink = HostSink::new(out);
        match element.process(packet, &mut sink).await {
            Ok(()) => STATUS_OK,
            Err(e) => FfiStatus::from_error(&e),
        }
    })
}

/// # Safety
/// `elem` must be a live instance of `E`; `name` and `value` are borrowed for
/// the call.
pub unsafe extern "C" fn set_property_shim<E: AsyncElement + 'static>(
    elem: *mut c_void,
    name: FfiStr,
    value: *const FfiPropValue,
) -> FfiStatus {
    if elem.is_null() || value.is_null() {
        return STATUS_ERROR;
    }
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the host owns `name` for the duration of the call.
        let Ok(name) = (unsafe { name.as_str() }) else {
            return STATUS_PROPERTY_UNKNOWN;
        };
        // SAFETY: forwarded from this function's contract.
        let Some(value) = (unsafe { prop_from_ffi(&*value) }) else {
            return STATUS_PROPERTY_VALUE;
        };
        // SAFETY: as above.
        let element = unsafe { &mut *elem.cast::<E>() };
        match element.set_property(name, value) {
            Ok(()) => STATUS_OK,
            Err(g2g_core::property::PropError::Unknown) => STATUS_PROPERTY_UNKNOWN,
            Err(_) => STATUS_PROPERTY_VALUE,
        }
    }));
    outcome.unwrap_or(STATUS_ERROR)
}

/// # Safety
/// `elem` must be a live instance of `E`, `name` borrowed for the call, and
/// `out` a writable slot.
pub unsafe extern "C" fn get_property_shim<E: AsyncElement + 'static>(
    elem: *mut c_void,
    name: FfiStr,
    out: *mut FfiPropValue,
) -> FfiStatus {
    if elem.is_null() || out.is_null() {
        return STATUS_ERROR;
    }
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the host owns `name` for the duration of the call.
        let Ok(name) = (unsafe { name.as_str() }) else {
            return STATUS_PROPERTY_UNKNOWN;
        };
        // SAFETY: forwarded from this function's contract.
        let element = unsafe { &*elem.cast::<E>() };
        let Some(value) = element.get_property(name) else {
            return STATUS_PROPERTY_UNKNOWN;
        };
        // The string a `Str` property yields is borrowed from a temporary here,
        // so it is copied onto the heap with a `free` the host will call.
        let Some(ffi) = owned_prop_into_ffi(&value) else {
            return STATUS_PROPERTY_VALUE;
        };
        // SAFETY: `out` is the host's slot, checked non-null above.
        unsafe { *out = ffi };
        STATUS_OK
    }));
    outcome.unwrap_or(STATUS_ERROR)
}

/// # Safety
/// `user` must be a pointer produced by `Box::into_raw` on a `Box<[u8]>` of the
/// matching length, released once.
unsafe extern "C" fn drop_owned_string(user: *mut c_void) {
    if user.is_null() {
        return;
    }
    // SAFETY: forwarded from this function's contract; the box was built from a
    // `String`'s bytes in `owned_prop_into_ffi`.
    drop(unsafe { Box::from_raw(user.cast::<std::string::String>()) });
}

/// A property value whose string payload the receiver owns and must free. The
/// borrowed form ([`prop_into_ffi`]) is only valid for one call, which is wrong
/// for a value read back out of an element.
fn owned_prop_into_ffi(value: &g2g_core::property::PropValue) -> Option<FfiPropValue> {
    let mut ffi = prop_into_ffi(value)?;
    if let g2g_core::property::PropValue::Str(s) = value {
        let owned = Box::new(s.clone());
        let ptr = owned.as_ptr();
        let len = owned.len();
        ffi.body.string = crate::abi::FfiPropStr {
            ptr,
            len,
            free: Some(drop_owned_string),
            free_user: Box::into_raw(owned).cast::<c_void>(),
        };
    }
    Some(ffi)
}

/// The element's vtable, filled from the shims for `E`.
pub fn element_vtable<E: AsyncElement + Default + 'static>() -> FfiElementVtable {
    FfiElementVtable {
        struct_size: core::mem::size_of::<FfiElementVtable>() as u32,
        version: 1,
        configure_pipeline: Some(configure_pipeline_shim::<E>),
        configure_output: Some(configure_output_shim::<E>),
        process: Some(process_shim::<E>),
        set_property: Some(set_property_shim::<E>),
        get_property: Some(get_property_shim::<E>),
        destroy: Some(destroy_shim::<E>),
        reserved: [None; 6],
    }
}

/// The caps a pad template declares, as the ABI's alternative list. `None` for
/// a wildcard template, which crosses as the empty set.
fn template_caps(template: Option<&PadTemplate>) -> Result<Vec<FfiCaps>, G2gError> {
    let Some(PadTemplate {
        caps: PadCaps::Fixed(set),
        ..
    }) = template
    else {
        return Ok(Vec::new());
    };
    set.alternatives()
        .iter()
        .map(|c| caps_into_ffi(c).map_err(|_| G2gError::CapsMismatch))
        .collect()
}

fn leak_caps(caps: Vec<FfiCaps>) -> FfiCapsSet {
    if caps.is_empty() {
        return FfiCapsSet::EMPTY;
    }
    let leaked: &'static [FfiCaps] = Box::leak(caps.into_boxed_slice());
    FfiCapsSet {
        alternatives: leaked.as_ptr(),
        count: leaked.len(),
    }
}

/// Register one element with the host, building its registration from the
/// element type itself.
///
/// The caps sets, property specs, and metadata come from a throwaway default
/// instance: they are all type-level tables the trait exposes through `&self`.
/// Everything the registration points at is leaked, which is the correct
/// lifetime, the host never unloads a plugin it accepted.
///
/// Source caps are published only for an element that declares itself a format
/// boundary. A pass-through publishes none, and the host reads that as "this
/// element produces what it was given", which is the only output shape v2 can
/// express as data.
///
/// # Safety
/// `registrar` must be the host's live registrar.
pub unsafe fn register_element<E>(
    registrar: &FfiRegistrar,
    name: &'static str,
    kind: u32,
) -> FfiStatus
where
    E: AsyncElement + PadTemplates + Default + 'static,
{
    let probe = E::default();
    let templates = E::pad_templates();

    let sink = template_caps(templates.iter().find(|t| t.direction == PadDirection::Sink));
    let source = if probe.is_format_boundary() {
        template_caps(
            templates
                .iter()
                .find(|t| t.direction == PadDirection::Source),
        )
    } else {
        Ok(Vec::new())
    };
    let (Ok(sink), Ok(source)) = (sink, source) else {
        return STATUS_ERROR;
    };

    let specs: Option<Vec<FfiPropertySpec>> =
        probe.properties().iter().map(spec_into_ffi).collect();
    let Some(specs) = specs else {
        return STATUS_ERROR;
    };
    let (properties, property_count) = if specs.is_empty() {
        (core::ptr::null(), 0)
    } else {
        let leaked: &'static [FfiPropertySpec] = Box::leak(specs.into_boxed_slice());
        (leaked.as_ptr(), leaked.len())
    };

    let metadata = probe.metadata();
    let vtable: &'static FfiElementVtable = Box::leak(Box::new(element_vtable::<E>()));

    let registration = FfiElementRegistration {
        struct_size: core::mem::size_of::<FfiElementRegistration>() as u32,
        kind,
        name: FfiStr::borrowed(name),
        metadata: FfiElementMetadata {
            long_name: FfiStr::borrowed(metadata.long_name),
            klass: FfiStr::borrowed(metadata.klass),
            description: FfiStr::borrowed(metadata.description),
            author: FfiStr::borrowed(metadata.author),
        },
        sink_caps: leak_caps(sink),
        source_caps: leak_caps(source),
        properties,
        property_count,
        vtable,
        create: Some(create_shim::<E>),
        reserved: [None; 4],
    };
    // SAFETY: forwarded from this function's contract; everything the
    // registration points at is leaked, so it outlives the call.
    unsafe { (registrar.register_element)(registrar.ctx, &registration) }
}

/// Declare a **v2** dynamically loadable plugin: emit the `repr(C)` descriptor
/// the host reads and the vtables it drives.
///
/// Unlike [`declare_plugin!`](crate::declare_plugin), nothing but `repr(C)`
/// data crosses the boundary, so the resulting plugin loads into a host built
/// by a different `rustc`, against a different `g2g-core` build.
///
/// ```ignore
/// // In the plugin crate (crate-type = ["cdylib"]):
/// g2g_plugin::declare_plugin_v2! {
///     name: "my-plugin",
///     version: "1.0.0",
///     elements: [
///         ("myfilter", MyFilter, transform),
///     ]
/// }
/// ```
///
/// Each element is `(name, Type, kind)` where `Type` implements `AsyncElement`,
/// `PadTemplates`, and `Default`, and `kind` is `transform` or `sink`. The
/// names are also written into the descriptor's capability list, which the host
/// reads before running any of this plugin's code.
#[macro_export]
macro_rules! declare_plugin_v2 {
    (@kind transform) => { $crate::abi::ELEMENT_TRANSFORM };
    (@kind sink) => { $crate::abi::ELEMENT_SINK };
    (
        name: $plugin_name:expr,
        version: $plugin_version:expr,
        elements: [ $( ( $name:expr, $ty:ty, $kind:ident ) ),* $(,)? ] $(,)?
    ) => {
        /// The descriptor's capability list: what this plugin will register,
        /// readable by the host before any of its code runs.
        static G2G_V2_CAPABILITIES: $crate::abi::AbiStatic<[$crate::abi::FfiCapability; 0 $( + { let _ = $name; 1 } )*]> =
            $crate::abi::AbiStatic([
                $(
                    $crate::abi::FfiCapability {
                        kind: $crate::declare_plugin_v2!(@kind $kind),
                        reserved: 0,
                        name: $crate::abi::FfiStr::borrowed($name),
                    },
                )*
            ]);

        /// Registration entry point. Called only after the host has validated
        /// the descriptor and its policy has allowed the capabilities above.
        ///
        /// # Safety
        /// Called by the host with a live registrar. Every registration is
        /// built from the element type itself, so it cannot disagree with the
        /// declaration above.
        unsafe extern "C" fn g2g_v2_register(
            registrar: *const $crate::abi::FfiRegistrar,
        ) -> $crate::abi::FfiStatus {
            if registrar.is_null() {
                return $crate::abi::STATUS_ERROR;
            }
            // SAFETY: the host passes a live registrar for this call.
            let registrar = unsafe { &*registrar };
            let outcome = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                $(
                    // SAFETY: `registrar` is live for the duration of this call.
                    let status = unsafe {
                        $crate::v2::register_element::<$ty>(
                            registrar,
                            $name,
                            $crate::declare_plugin_v2!(@kind $kind),
                        )
                    };
                    if !status.is_ok() {
                        return status;
                    }
                )*
                $crate::abi::STATUS_OK
            }));
            outcome.unwrap_or($crate::abi::STATUS_ERROR)
        }

        /// The one symbol the host looks up: a `static`, so the declaration is
        /// readable without running plugin code.
        #[no_mangle]
        #[allow(non_upper_case_globals)]
        pub static g2g_plugin_v2_descriptor:
            $crate::abi::AbiStatic<$crate::abi::FfiPluginDescriptor> =
            $crate::abi::AbiStatic($crate::abi::FfiPluginDescriptor {
                magic: $crate::abi::V2_MAGIC,
                abi_version: $crate::abi::V2_ABI_VERSION,
                struct_size: ::core::mem::size_of::<$crate::abi::FfiPluginDescriptor>() as u32,
                name: $crate::abi::FfiStr::borrowed($plugin_name),
                version: $crate::abi::FfiStr::borrowed($plugin_version),
                capabilities: G2G_V2_CAPABILITIES.0.as_ptr(),
                capability_count: G2G_V2_CAPABILITIES.0.len(),
                register: Some(g2g_v2_register),
                reserved: [None; 4],
            });
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::caps::{Caps, CapsSet, Dim, Interlace, Rate, RawVideoFormat};

    #[derive(Debug, Default)]
    struct Passthrough;

    impl AsyncElement for Passthrough {
        type ProcessFuture<'a>
            = core::pin::Pin<Box<dyn core::future::Future<Output = Result<(), G2gError>> + 'a>>
        where
            Self: 'a;

        fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
            Ok(upstream.clone())
        }

        fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
            Ok(ConfigureOutcome::Accepted)
        }

        fn process<'a>(
            &'a mut self,
            packet: PipelinePacket,
            out: &'a mut dyn OutputSink,
        ) -> Self::ProcessFuture<'a> {
            Box::pin(async move { out.push(packet).await.map(|_| ()) })
        }
    }

    impl PadTemplates for Passthrough {
        fn pad_templates() -> Vec<PadTemplate> {
            let set = CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: Interlace::Any,
            });
            Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
        }
    }

    #[test]
    fn a_pass_through_publishes_no_source_caps() {
        // The rule the host relies on: a non-boundary element declares nothing
        // on its source pad, and the host reads that as "produces its input".
        // Publishing the `Any` source template instead would be refused by
        // validation, since it cannot be fixated.
        let templates = Passthrough::pad_templates();
        let source = template_caps(
            templates
                .iter()
                .find(|t| t.direction == PadDirection::Source),
        )
        .expect("the template converts");
        assert_eq!(source.len(), 1, "the template itself does carry caps");
        assert!(
            !Passthrough.is_format_boundary(),
            "but the element is not a boundary, so registration publishes none"
        );
    }

    #[test]
    fn the_generated_vtable_fills_every_entry() {
        // A macro-built element gets the full surface: nothing is left for the
        // host to default, which is the difference from a hand-written plugin.
        let vtable = element_vtable::<Passthrough>();
        assert_eq!(
            vtable.struct_size as usize,
            core::mem::size_of::<FfiElementVtable>()
        );
        assert!(vtable.process.is_some());
        assert!(vtable.destroy.is_some());
        assert!(vtable.configure_pipeline.is_some());
        assert!(vtable.configure_output.is_some());
        assert!(vtable.set_property.is_some());
        assert!(vtable.get_property.is_some());
        assert!(
            vtable.reserved.iter().all(Option::is_none),
            "reserved slots stay null until a future revision claims one"
        );
    }

    #[test]
    fn create_and_destroy_round_trip() {
        // SAFETY: the pair is used exactly as the host would.
        let instance = unsafe { create_shim::<Passthrough>() };
        assert!(!instance.is_null());
        // SAFETY: `instance` came from the matching `create_shim`.
        unsafe { destroy_shim::<Passthrough>(instance) };
    }
}
