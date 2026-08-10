//! An out-of-tree third-party g2g plugin written directly against the **v2**
//! plugin ABI: a `repr(C)` descriptor, a `repr(C)` element vtable, and nothing
//! else crossing the boundary.
//!
//! Written by hand, without the SDK macro, because that is the property under
//! test: the boundary has to be writable by something that is not this
//! workspace's `rustc`. The same shape in C lives in `tests/fixtures/c-plugin`.
//!
//! The element is `v2counter`: it counts data frames and forwards them
//! unchanged, and exposes `count` (read-only) and `enabled` (drops frames when
//! false) as runtime properties.

use core::ffi::c_void;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use async_ffi::{ContextExt, LocalFfiFuture};

use g2g_plugin::abi::{
    AbiStatic, FfiCapability, FfiCapsSet, FfiElementMetadata, FfiElementRegistration,
    FfiElementVtable, FfiOutputSink, FfiPacket, FfiPluginDescriptor, FfiPropValue,
    FfiPropValueBody, FfiPropertySpec, FfiRegistrar, FfiStatus, FfiStr, ELEMENT_TRANSFORM,
    PACKET_DATA_FRAME, PACKET_EOS, PROP_BOOL, PROP_UINT, STATUS_ERROR, STATUS_OK,
    STATUS_PROPERTY_UNKNOWN, STATUS_PROPERTY_VALUE, V2_ABI_VERSION, V2_MAGIC,
};

/// Per-instance state.
#[derive(Debug, Default)]
struct Counter {
    seen: u64,
    enabled: bool,
}

/// # Safety
/// Called by the host to build one instance; the host pairs it with `destroy`.
unsafe extern "C" fn create() -> *mut c_void {
    Box::into_raw(Box::new(Counter {
        seen: 0,
        enabled: true,
    }))
    .cast()
}

/// # Safety
/// `elem` must be a pointer `create` returned, destroyed exactly once.
unsafe extern "C" fn destroy(elem: *mut c_void) {
    drop(unsafe { Box::from_raw(elem.cast::<Counter>()) });
}

/// One in-flight push toward the host's downstream, in the poll form the ABI
/// uses. This is the whole shape a v2 plugin needs to be backpressure-aware.
struct Push {
    sink: FfiOutputSink,
    packet: FfiPacket,
}

impl Future for Push {
    type Output = FfiStatus;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<FfiStatus> {
        let this = self.get_mut();
        let poll = cx.with_ffi_context(|ffi_cx| {
            // SAFETY: the host's sink vtable is valid for as long as the future
            // this push lives in, which the host guarantees by outliving it.
            unsafe { ((*this.sink.vtable).poll_push)(this.sink.ctx, ffi_cx, &mut this.packet) }
        });
        poll.try_into_poll().unwrap_or(Poll::Ready(STATUS_ERROR))
    }
}

/// # Safety
/// `elem` must be a live instance and `out` a sink the host keeps valid until
/// the returned future is dropped.
unsafe extern "C" fn process(
    elem: *mut c_void,
    packet: FfiPacket,
    out: FfiOutputSink,
) -> LocalFfiFuture<FfiStatus> {
    // SAFETY: the host calls `process` with exclusive access to the instance.
    let counter = unsafe { &mut *elem.cast::<Counter>() };
    let is_frame = packet.tag == PACKET_DATA_FRAME;
    if is_frame {
        counter.seen += 1;
    }
    // The runner emits the pipeline's single EOS, so a transform must not push
    // one of its own. A disabled counter drops frames instead of forwarding.
    let drop_it = packet.tag == PACKET_EOS || (is_frame && !counter.enabled);

    LocalFfiFuture::new(async move {
        if drop_it {
            // Releasing the payload is the plugin's job once it owns the packet.
            if let Some(free) = packet.frame.free {
                // SAFETY: ownership of the payload came with the packet, and
                // this releases it exactly once.
                unsafe { free(packet.frame.free_user) };
            }
            return STATUS_OK;
        }
        Push { sink: out, packet }.await
    })
}

/// # Safety
/// `elem` is a live instance; `value` is borrowed for the call.
unsafe extern "C" fn set_property(
    elem: *mut c_void,
    name: FfiStr,
    value: *const FfiPropValue,
) -> FfiStatus {
    // SAFETY: the host passes a name it owns for the duration of the call.
    let Ok(name) = (unsafe { name.as_str() }) else {
        return STATUS_PROPERTY_UNKNOWN;
    };
    if value.is_null() {
        return STATUS_PROPERTY_VALUE;
    }
    // SAFETY: the host passes a live value for the duration of the call.
    let value = unsafe { &*value };
    // SAFETY: the host calls with exclusive access to the instance.
    let counter = unsafe { &mut *elem.cast::<Counter>() };
    match name {
        "enabled" => {
            if value.kind != PROP_BOOL {
                return STATUS_PROPERTY_VALUE;
            }
            // SAFETY: the kind tag says the boolean member is live.
            counter.enabled = unsafe { value.body.boolean } != 0;
            STATUS_OK
        }
        "count" => STATUS_PROPERTY_VALUE,
        _ => STATUS_PROPERTY_UNKNOWN,
    }
}

/// # Safety
/// `elem` is a live instance; `out` is a slot the host owns.
unsafe extern "C" fn get_property(
    elem: *mut c_void,
    name: FfiStr,
    out: *mut FfiPropValue,
) -> FfiStatus {
    // SAFETY: as in `set_property`.
    let Ok(name) = (unsafe { name.as_str() }) else {
        return STATUS_PROPERTY_UNKNOWN;
    };
    if out.is_null() {
        return STATUS_ERROR;
    }
    // SAFETY: the host calls with exclusive access to the instance.
    let counter = unsafe { &*elem.cast::<Counter>() };
    let value = match name {
        "count" => FfiPropValue {
            kind: PROP_UINT,
            reserved: 0,
            body: FfiPropValueBody { uint: counter.seen },
        },
        "enabled" => FfiPropValue {
            kind: PROP_BOOL,
            reserved: 0,
            body: FfiPropValueBody {
                boolean: u32::from(counter.enabled),
            },
        },
        _ => return STATUS_PROPERTY_UNKNOWN,
    };
    // SAFETY: `out` is a live slot the host provided for this write.
    unsafe { *out = value };
    STATUS_OK
}

static PROPERTIES: AbiStatic<[FfiPropertySpec; 2]> = AbiStatic([
    FfiPropertySpec {
        name: FfiStr::borrowed("count"),
        kind: PROP_UINT,
        readable: 1,
        writable: 0,
        reserved: 0,
        blurb: FfiStr::borrowed("data frames seen so far"),
        default_value: FfiStr::EMPTY,
    },
    FfiPropertySpec {
        name: FfiStr::borrowed("enabled"),
        kind: PROP_BOOL,
        readable: 1,
        writable: 1,
        reserved: 0,
        blurb: FfiStr::borrowed("forward frames; drop them when false"),
        default_value: FfiStr::borrowed("true"),
    },
]);

static VTABLE: AbiStatic<FfiElementVtable> = AbiStatic(FfiElementVtable {
    struct_size: core::mem::size_of::<FfiElementVtable>() as u32,
    version: 1,
    configure_pipeline: None,
    configure_output: None,
    process: Some(process),
    set_property: Some(set_property),
    get_property: Some(get_property),
    destroy: Some(destroy),
    reserved: [None; 6],
});

/// # Safety
/// `registrar` is the host-owned object, valid for the duration of this call.
unsafe extern "C" fn register(registrar: *const FfiRegistrar) -> FfiStatus {
    if registrar.is_null() {
        return STATUS_ERROR;
    }
    // SAFETY: the host passes a live registrar for the duration of the call.
    let registrar = unsafe { &*registrar };
    let element = FfiElementRegistration {
        struct_size: core::mem::size_of::<FfiElementRegistration>() as u32,
        kind: ELEMENT_TRANSFORM,
        name: FfiStr::borrowed("v2counter"),
        metadata: FfiElementMetadata {
            long_name: FfiStr::borrowed("v2 counting filter"),
            klass: FfiStr::borrowed("Filter/Effect/Video"),
            description: FfiStr::borrowed("Counts data frames and forwards them unchanged."),
            author: FfiStr::borrowed("third-party"),
        },
        // Empty sets: accepts anything, produces what it was given. A
        // pass-through element declares no caps of its own.
        sink_caps: FfiCapsSet::EMPTY,
        source_caps: FfiCapsSet::EMPTY,
        properties: PROPERTIES.0.as_ptr(),
        property_count: PROPERTIES.0.len(),
        vtable: &VTABLE.0,
        create: Some(create),
        reserved: [None; 4],
    };
    // SAFETY: `element` is a live local valid for the duration of the call, and
    // every pointer in it addresses a `static` in this library.
    let status = unsafe { (registrar.register_element)(registrar.ctx, &element) };
    if !status.is_ok() {
        return status;
    }

    // A plugin that breaks its own declaration, for the loader's gate to catch.
    // Enabled only by the `undeclared` feature, which the test turns on.
    #[cfg(feature = "undeclared")]
    {
        let sneaky = FfiElementRegistration {
            name: FfiStr::borrowed("sneaky"),
            ..element
        };
        // SAFETY: as above.
        return unsafe { (registrar.register_element)(registrar.ctx, &sneaky) };
    }
    #[cfg(not(feature = "undeclared"))]
    status
}

static CAPABILITIES: AbiStatic<[FfiCapability; 1]> = AbiStatic([FfiCapability {
    kind: ELEMENT_TRANSFORM,
    reserved: 0,
    name: FfiStr::borrowed("v2counter"),
}]);

/// The one symbol the host looks up. A `static`, not a function: the host reads
/// and validates it, and decides whether to allow the declared capabilities,
/// before any code in this library runs.
#[no_mangle]
#[allow(non_upper_case_globals)]
pub static g2g_plugin_v2_descriptor: AbiStatic<FfiPluginDescriptor> =
    AbiStatic(FfiPluginDescriptor {
        magic: V2_MAGIC,
        abi_version: V2_ABI_VERSION,
        struct_size: core::mem::size_of::<FfiPluginDescriptor>() as u32,
        name: FfiStr::borrowed("g2g-v2-example-plugin"),
        version: FfiStr::borrowed("0.1.0"),
        capabilities: CAPABILITIES.0.as_ptr(),
        capability_count: CAPABILITIES.0.len(),
        register: Some(register),
        reserved: [None; 4],
    });
