//! Compile-time layout pins for the v2 ABI.
//!
//! Every `repr(C)` type that crosses the boundary is checked against a
//! hand-written field table, so the size and the field offsets the validator,
//! the `struct_size` growth rule, and `g2g_plugin_v2.h` all assume are the ones
//! rustc actually produced.
//!
//! A failure here means the frozen surface moved. The fix is to bump
//! [`V2_ABI_VERSION`](super::V2_ABI_VERSION) and update the C header, never to
//! edit the table until the assert passes: an already-shipped plugin is built
//! against the old layout and would be read at the wrong offsets.
//!
//! Sizes are derived rather than written as byte counts, because a
//! pointer-bearing struct differs between a 32- and a 64-bit target, and a
//! `u64` field is 4-byte aligned on some 32-bit ABIs (i686 System V) and
//! 8-byte aligned on others (armv7, wasm32). The tables carry each field's real
//! `(size, align)` and the helpers below re-apply the C layout rule, so one
//! table holds on every target the crate builds for.

// Everything lives in one anonymous const because a `const _` item does not
// count as a use for dead-code analysis: helpers at module scope would be
// reported unused.
const _: () = {
    use super::{
        AbiStatic, FfiAudioCaps, FfiByteStreamCaps, FfiCapability, FfiCaps, FfiCapsBody,
        FfiCapsSet, FfiCompressedVideoCaps, FfiDim, FfiElementMetadata, FfiElementRegistration,
        FfiElementVtable, FfiFraction, FfiFrame, FfiOutputSink, FfiOutputSinkVtable, FfiPacket,
        FfiPluginDescriptor, FfiPropStr, FfiPropValue, FfiPropValueBody, FfiPropertySpec, FfiRate,
        FfiRawVideoCaps, FfiRegistrar, FfiStatus, FfiStr, FfiTextCaps, VERSIONED_HEADER_SIZE,
    };
    use core::ffi::c_void;
    use core::mem::{align_of, offset_of, size_of};

    /// End of the first `count` fields, before the struct's own trailing
    /// padding: each field starts at the next multiple of its alignment.
    const fn offset_after(fields: &[(usize, usize)], count: usize) -> usize {
        let mut offset = 0usize;
        let mut index = 0usize;
        while index < count {
            let (size, align) = fields[index];
            offset = offset.next_multiple_of(align) + size;
            index += 1;
        }
        offset
    }

    const fn largest_alignment(fields: &[(usize, usize)]) -> usize {
        let mut alignment = 1usize;
        let mut index = 0usize;
        while index < fields.len() {
            if fields[index].1 > alignment {
                alignment = fields[index].1;
            }
            index += 1;
        }
        alignment
    }

    const fn field_offset(fields: &[(usize, usize)], index: usize) -> usize {
        offset_after(fields, index).next_multiple_of(fields[index].1)
    }

    const fn struct_size(fields: &[(usize, usize)]) -> usize {
        offset_after(fields, fields.len()).next_multiple_of(largest_alignment(fields))
    }

    /// A union is as large as its widest member, rounded up to the widest
    /// alignment among them.
    const fn union_size(members: &[(usize, usize)]) -> usize {
        let mut widest = 0usize;
        let mut index = 0usize;
        while index < members.len() {
            if members[index].0 > widest {
                widest = members[index].0;
            }
            index += 1;
        }
        widest.next_multiple_of(largest_alignment(members))
    }

    const fn shape(fields: &[(usize, usize)]) -> (usize, usize) {
        (struct_size(fields), largest_alignment(fields))
    }

    const fn union_shape(members: &[(usize, usize)]) -> (usize, usize) {
        (union_size(members), largest_alignment(members))
    }

    // -----------------------------------------------------------------------
    // Primitive field shapes
    // -----------------------------------------------------------------------

    const U32: (usize, usize) = (size_of::<u32>(), align_of::<u32>());
    const U64: (usize, usize) = (size_of::<u64>(), align_of::<u64>());
    const I64: (usize, usize) = (size_of::<i64>(), align_of::<i64>());
    const F64: (usize, usize) = (size_of::<f64>(), align_of::<f64>());
    const PTR: (usize, usize) = (size_of::<*const c_void>(), align_of::<*const c_void>());

    /// One function-pointer slot. A null slot is what "absent, use the host
    /// default" is spelled as, so `Option` must stay the same width as a bare
    /// pointer or every optional vtable entry shifts.
    const FUNC: (usize, usize) = (
        size_of::<Option<extern "C" fn()>>(),
        align_of::<Option<extern "C" fn()>>(),
    );
    assert!(FUNC.0 == PTR.0 && FUNC.1 == PTR.1);

    /// `n` reserved function-pointer slots as one array field.
    const fn func_array(slots: usize) -> (usize, usize) {
        (FUNC.0 * slots, FUNC.1)
    }

    // `usize` is the C `size_t` half of every pointer + length pair, and the
    // tables spell those pairs as two pointer-shaped fields.
    assert!(size_of::<usize>() == PTR.0 && align_of::<usize>() == PTR.1);

    // -----------------------------------------------------------------------
    // Strings
    // -----------------------------------------------------------------------

    const STR_FIELDS: &[(usize, usize)] = &[PTR, PTR];
    const STR: (usize, usize) = shape(STR_FIELDS);
    assert!(size_of::<FfiStr>() == STR.0);
    assert!(align_of::<FfiStr>() == STR.1);
    assert!(offset_of!(FfiStr, ptr) == field_offset(STR_FIELDS, 0));
    assert!(offset_of!(FfiStr, len) == field_offset(STR_FIELDS, 1));

    // -----------------------------------------------------------------------
    // Caps
    // -----------------------------------------------------------------------

    const DIM_FIELDS: &[(usize, usize)] = &[U32, U32, U32];
    const DIM: (usize, usize) = shape(DIM_FIELDS);
    assert!(size_of::<FfiDim>() == DIM.0);
    assert!(offset_of!(FfiDim, kind) == field_offset(DIM_FIELDS, 0));
    assert!(offset_of!(FfiDim, min) == field_offset(DIM_FIELDS, 1));
    assert!(offset_of!(FfiDim, max) == field_offset(DIM_FIELDS, 2));

    const RATE_FIELDS: &[(usize, usize)] = &[U32, U32, U32];
    const RATE: (usize, usize) = shape(RATE_FIELDS);
    assert!(size_of::<FfiRate>() == RATE.0);
    assert!(offset_of!(FfiRate, kind) == field_offset(RATE_FIELDS, 0));
    assert!(offset_of!(FfiRate, min_q16) == field_offset(RATE_FIELDS, 1));
    assert!(offset_of!(FfiRate, max_q16) == field_offset(RATE_FIELDS, 2));

    const RAW_VIDEO_FIELDS: &[(usize, usize)] = &[U32, DIM, DIM, RATE, U32];
    const RAW_VIDEO: (usize, usize) = shape(RAW_VIDEO_FIELDS);
    assert!(size_of::<FfiRawVideoCaps>() == RAW_VIDEO.0);
    assert!(offset_of!(FfiRawVideoCaps, format) == field_offset(RAW_VIDEO_FIELDS, 0));
    assert!(offset_of!(FfiRawVideoCaps, width) == field_offset(RAW_VIDEO_FIELDS, 1));
    assert!(offset_of!(FfiRawVideoCaps, height) == field_offset(RAW_VIDEO_FIELDS, 2));
    assert!(offset_of!(FfiRawVideoCaps, framerate) == field_offset(RAW_VIDEO_FIELDS, 3));
    assert!(offset_of!(FfiRawVideoCaps, interlace) == field_offset(RAW_VIDEO_FIELDS, 4));

    const COMPRESSED_VIDEO_FIELDS: &[(usize, usize)] = &[U32, DIM, DIM, RATE];
    const COMPRESSED_VIDEO: (usize, usize) = shape(COMPRESSED_VIDEO_FIELDS);
    assert!(size_of::<FfiCompressedVideoCaps>() == COMPRESSED_VIDEO.0);
    assert!(offset_of!(FfiCompressedVideoCaps, codec) == field_offset(COMPRESSED_VIDEO_FIELDS, 0));
    assert!(offset_of!(FfiCompressedVideoCaps, width) == field_offset(COMPRESSED_VIDEO_FIELDS, 1));
    assert!(offset_of!(FfiCompressedVideoCaps, height) == field_offset(COMPRESSED_VIDEO_FIELDS, 2));
    assert!(
        offset_of!(FfiCompressedVideoCaps, framerate) == field_offset(COMPRESSED_VIDEO_FIELDS, 3)
    );

    const AUDIO_FIELDS: &[(usize, usize)] = &[U32, U32, U32];
    const AUDIO: (usize, usize) = shape(AUDIO_FIELDS);
    assert!(size_of::<FfiAudioCaps>() == AUDIO.0);
    assert!(offset_of!(FfiAudioCaps, format) == field_offset(AUDIO_FIELDS, 0));
    assert!(offset_of!(FfiAudioCaps, channels) == field_offset(AUDIO_FIELDS, 1));
    assert!(offset_of!(FfiAudioCaps, sample_rate) == field_offset(AUDIO_FIELDS, 2));

    const BYTE_STREAM_FIELDS: &[(usize, usize)] = &[U32];
    const BYTE_STREAM: (usize, usize) = shape(BYTE_STREAM_FIELDS);
    assert!(size_of::<FfiByteStreamCaps>() == BYTE_STREAM.0);
    assert!(offset_of!(FfiByteStreamCaps, encoding) == field_offset(BYTE_STREAM_FIELDS, 0));

    const TEXT_FIELDS: &[(usize, usize)] = &[U32];
    const TEXT: (usize, usize) = shape(TEXT_FIELDS);
    assert!(size_of::<FfiTextCaps>() == TEXT.0);
    assert!(offset_of!(FfiTextCaps, format) == field_offset(TEXT_FIELDS, 0));

    const CAPS_BODY_MEMBERS: &[(usize, usize)] =
        &[RAW_VIDEO, COMPRESSED_VIDEO, AUDIO, BYTE_STREAM, TEXT];
    const CAPS_BODY: (usize, usize) = union_shape(CAPS_BODY_MEMBERS);
    assert!(size_of::<FfiCapsBody>() == CAPS_BODY.0);
    assert!(align_of::<FfiCapsBody>() == CAPS_BODY.1);

    const CAPS_FIELDS: &[(usize, usize)] = &[U32, U32, CAPS_BODY];
    const CAPS: (usize, usize) = shape(CAPS_FIELDS);
    assert!(size_of::<FfiCaps>() == CAPS.0);
    assert!(offset_of!(FfiCaps, tag) == field_offset(CAPS_FIELDS, 0));
    assert!(offset_of!(FfiCaps, reserved) == field_offset(CAPS_FIELDS, 1));
    assert!(offset_of!(FfiCaps, body) == field_offset(CAPS_FIELDS, 2));
    // The `reserved` u32 exists so the payload starts on an 8-byte boundary
    // whatever the union's own alignment works out to.
    assert!(offset_of!(FfiCaps, body) == 8);

    const CAPS_SET_FIELDS: &[(usize, usize)] = &[PTR, PTR];
    const CAPS_SET: (usize, usize) = shape(CAPS_SET_FIELDS);
    assert!(size_of::<FfiCapsSet>() == CAPS_SET.0);
    assert!(offset_of!(FfiCapsSet, alternatives) == field_offset(CAPS_SET_FIELDS, 0));
    assert!(offset_of!(FfiCapsSet, count) == field_offset(CAPS_SET_FIELDS, 1));

    // -----------------------------------------------------------------------
    // Frames and packets
    // -----------------------------------------------------------------------

    const FRAME_FIELDS: &[(usize, usize)] = &[
        PTR,  // data
        PTR,  // len
        FUNC, // free
        PTR,  // free_user
        U64,  // pts_ns
        U64,  // dts_ns
        U64,  // duration_ns
        U64,  // capture_ns
        U64,  // arrival_ns
        U64,  // sequence
        U32,  // keyframe
        U32,  // reserved
    ];
    const FRAME: (usize, usize) = shape(FRAME_FIELDS);
    assert!(size_of::<FfiFrame>() == FRAME.0);
    assert!(offset_of!(FfiFrame, data) == field_offset(FRAME_FIELDS, 0));
    assert!(offset_of!(FfiFrame, len) == field_offset(FRAME_FIELDS, 1));
    assert!(offset_of!(FfiFrame, free) == field_offset(FRAME_FIELDS, 2));
    assert!(offset_of!(FfiFrame, free_user) == field_offset(FRAME_FIELDS, 3));
    assert!(offset_of!(FfiFrame, pts_ns) == field_offset(FRAME_FIELDS, 4));
    assert!(offset_of!(FfiFrame, dts_ns) == field_offset(FRAME_FIELDS, 5));
    assert!(offset_of!(FfiFrame, duration_ns) == field_offset(FRAME_FIELDS, 6));
    assert!(offset_of!(FfiFrame, capture_ns) == field_offset(FRAME_FIELDS, 7));
    assert!(offset_of!(FfiFrame, arrival_ns) == field_offset(FRAME_FIELDS, 8));
    assert!(offset_of!(FfiFrame, sequence) == field_offset(FRAME_FIELDS, 9));
    assert!(offset_of!(FfiFrame, keyframe) == field_offset(FRAME_FIELDS, 10));
    assert!(offset_of!(FfiFrame, reserved) == field_offset(FRAME_FIELDS, 11));

    const PACKET_FIELDS: &[(usize, usize)] = &[U32, U32, CAPS, FRAME];
    assert!(size_of::<FfiPacket>() == struct_size(PACKET_FIELDS));
    assert!(offset_of!(FfiPacket, tag) == field_offset(PACKET_FIELDS, 0));
    assert!(offset_of!(FfiPacket, reserved) == field_offset(PACKET_FIELDS, 1));
    assert!(offset_of!(FfiPacket, caps) == field_offset(PACKET_FIELDS, 2));
    assert!(offset_of!(FfiPacket, frame) == field_offset(PACKET_FIELDS, 3));

    // -----------------------------------------------------------------------
    // Output sink
    // -----------------------------------------------------------------------

    const OUTPUT_SINK_VTABLE_FIELDS: &[(usize, usize)] = &[
        U32,           // struct_size
        U32,           // version
        FUNC,          // poll_push
        func_array(4), // reserved
    ];
    assert!(size_of::<FfiOutputSinkVtable>() == struct_size(OUTPUT_SINK_VTABLE_FIELDS));
    assert!(offset_of!(FfiOutputSinkVtable, struct_size) == 0);
    assert!(offset_of!(FfiOutputSinkVtable, version) == field_offset(OUTPUT_SINK_VTABLE_FIELDS, 1));
    assert!(
        offset_of!(FfiOutputSinkVtable, poll_push) == field_offset(OUTPUT_SINK_VTABLE_FIELDS, 2)
    );
    assert!(
        offset_of!(FfiOutputSinkVtable, reserved) == field_offset(OUTPUT_SINK_VTABLE_FIELDS, 3)
    );

    const OUTPUT_SINK_FIELDS: &[(usize, usize)] = &[PTR, PTR];
    assert!(size_of::<FfiOutputSink>() == struct_size(OUTPUT_SINK_FIELDS));
    assert!(offset_of!(FfiOutputSink, ctx) == field_offset(OUTPUT_SINK_FIELDS, 0));
    assert!(offset_of!(FfiOutputSink, vtable) == field_offset(OUTPUT_SINK_FIELDS, 1));

    // -----------------------------------------------------------------------
    // Properties
    // -----------------------------------------------------------------------

    const FRACTION_FIELDS: &[(usize, usize)] = &[U32, U32];
    const FRACTION: (usize, usize) = shape(FRACTION_FIELDS);
    assert!(size_of::<FfiFraction>() == FRACTION.0);
    assert!(offset_of!(FfiFraction, num) == field_offset(FRACTION_FIELDS, 0));
    assert!(offset_of!(FfiFraction, den) == field_offset(FRACTION_FIELDS, 1));

    const PROP_STR_FIELDS: &[(usize, usize)] = &[PTR, PTR, FUNC, PTR];
    const PROP_STR: (usize, usize) = shape(PROP_STR_FIELDS);
    assert!(size_of::<FfiPropStr>() == PROP_STR.0);
    assert!(offset_of!(FfiPropStr, ptr) == field_offset(PROP_STR_FIELDS, 0));
    assert!(offset_of!(FfiPropStr, len) == field_offset(PROP_STR_FIELDS, 1));
    assert!(offset_of!(FfiPropStr, free) == field_offset(PROP_STR_FIELDS, 2));
    assert!(offset_of!(FfiPropStr, free_user) == field_offset(PROP_STR_FIELDS, 3));

    const PROP_VALUE_BODY_MEMBERS: &[(usize, usize)] = &[U32, I64, U64, F64, FRACTION, PROP_STR];
    const PROP_VALUE_BODY: (usize, usize) = union_shape(PROP_VALUE_BODY_MEMBERS);
    assert!(size_of::<FfiPropValueBody>() == PROP_VALUE_BODY.0);
    assert!(align_of::<FfiPropValueBody>() == PROP_VALUE_BODY.1);

    const PROP_VALUE_FIELDS: &[(usize, usize)] = &[U32, U32, PROP_VALUE_BODY];
    assert!(size_of::<FfiPropValue>() == struct_size(PROP_VALUE_FIELDS));
    assert!(offset_of!(FfiPropValue, kind) == field_offset(PROP_VALUE_FIELDS, 0));
    assert!(offset_of!(FfiPropValue, reserved) == field_offset(PROP_VALUE_FIELDS, 1));
    assert!(offset_of!(FfiPropValue, body) == field_offset(PROP_VALUE_FIELDS, 2));
    // Same reason as `FfiCaps::body`: the padding u32 puts the payload on 8.
    assert!(offset_of!(FfiPropValue, body) == 8);

    const PROPERTY_SPEC_FIELDS: &[(usize, usize)] = &[
        STR, // name
        U32, // kind
        U32, // readable
        U32, // writable
        U32, // reserved
        STR, // blurb
        STR, // default_value
    ];
    assert!(size_of::<FfiPropertySpec>() == struct_size(PROPERTY_SPEC_FIELDS));
    assert!(offset_of!(FfiPropertySpec, name) == field_offset(PROPERTY_SPEC_FIELDS, 0));
    assert!(offset_of!(FfiPropertySpec, kind) == field_offset(PROPERTY_SPEC_FIELDS, 1));
    assert!(offset_of!(FfiPropertySpec, readable) == field_offset(PROPERTY_SPEC_FIELDS, 2));
    assert!(offset_of!(FfiPropertySpec, writable) == field_offset(PROPERTY_SPEC_FIELDS, 3));
    assert!(offset_of!(FfiPropertySpec, reserved) == field_offset(PROPERTY_SPEC_FIELDS, 4));
    assert!(offset_of!(FfiPropertySpec, blurb) == field_offset(PROPERTY_SPEC_FIELDS, 5));
    assert!(offset_of!(FfiPropertySpec, default_value) == field_offset(PROPERTY_SPEC_FIELDS, 6));

    // -----------------------------------------------------------------------
    // Element vtable and registration
    // -----------------------------------------------------------------------

    const ELEMENT_VTABLE_FIELDS: &[(usize, usize)] = &[
        U32,           // struct_size
        U32,           // version
        FUNC,          // process
        FUNC,          // destroy
        FUNC,          // configure_pipeline
        FUNC,          // configure_output
        FUNC,          // set_property
        FUNC,          // get_property
        func_array(6), // reserved
    ];
    assert!(size_of::<FfiElementVtable>() == struct_size(ELEMENT_VTABLE_FIELDS));
    assert!(offset_of!(FfiElementVtable, struct_size) == 0);
    assert!(offset_of!(FfiElementVtable, version) == field_offset(ELEMENT_VTABLE_FIELDS, 1));
    assert!(offset_of!(FfiElementVtable, process) == field_offset(ELEMENT_VTABLE_FIELDS, 2));
    assert!(offset_of!(FfiElementVtable, destroy) == field_offset(ELEMENT_VTABLE_FIELDS, 3));
    assert!(
        offset_of!(FfiElementVtable, configure_pipeline) == field_offset(ELEMENT_VTABLE_FIELDS, 4)
    );
    assert!(
        offset_of!(FfiElementVtable, configure_output) == field_offset(ELEMENT_VTABLE_FIELDS, 5)
    );
    assert!(offset_of!(FfiElementVtable, set_property) == field_offset(ELEMENT_VTABLE_FIELDS, 6));
    assert!(offset_of!(FfiElementVtable, get_property) == field_offset(ELEMENT_VTABLE_FIELDS, 7));
    assert!(offset_of!(FfiElementVtable, reserved) == field_offset(ELEMENT_VTABLE_FIELDS, 8));
    // The two required entries come first, so the smallest `struct_size` a
    // plugin can declare and still be usable stops right after `destroy`.
    assert!(
        offset_of!(FfiElementVtable, configure_pipeline)
            == offset_of!(FfiElementVtable, destroy) + FUNC.0
    );

    const ELEMENT_METADATA_FIELDS: &[(usize, usize)] = &[STR, STR, STR, STR];
    const ELEMENT_METADATA: (usize, usize) = shape(ELEMENT_METADATA_FIELDS);
    assert!(size_of::<FfiElementMetadata>() == ELEMENT_METADATA.0);
    assert!(offset_of!(FfiElementMetadata, long_name) == field_offset(ELEMENT_METADATA_FIELDS, 0));
    assert!(offset_of!(FfiElementMetadata, klass) == field_offset(ELEMENT_METADATA_FIELDS, 1));
    assert!(
        offset_of!(FfiElementMetadata, description) == field_offset(ELEMENT_METADATA_FIELDS, 2)
    );
    assert!(offset_of!(FfiElementMetadata, author) == field_offset(ELEMENT_METADATA_FIELDS, 3));

    const ELEMENT_REGISTRATION_FIELDS: &[(usize, usize)] = &[
        U32,              // struct_size
        U32,              // kind
        STR,              // name
        ELEMENT_METADATA, // metadata
        CAPS_SET,         // sink_caps
        CAPS_SET,         // source_caps
        PTR,              // properties
        PTR,              // property_count
        PTR,              // vtable
        FUNC,             // create
        func_array(4),    // reserved
    ];
    assert!(size_of::<FfiElementRegistration>() == struct_size(ELEMENT_REGISTRATION_FIELDS));
    assert!(offset_of!(FfiElementRegistration, struct_size) == 0);
    assert!(
        offset_of!(FfiElementRegistration, kind) == field_offset(ELEMENT_REGISTRATION_FIELDS, 1)
    );
    assert!(
        offset_of!(FfiElementRegistration, name) == field_offset(ELEMENT_REGISTRATION_FIELDS, 2)
    );
    assert!(
        offset_of!(FfiElementRegistration, metadata)
            == field_offset(ELEMENT_REGISTRATION_FIELDS, 3)
    );
    assert!(
        offset_of!(FfiElementRegistration, sink_caps)
            == field_offset(ELEMENT_REGISTRATION_FIELDS, 4)
    );
    assert!(
        offset_of!(FfiElementRegistration, source_caps)
            == field_offset(ELEMENT_REGISTRATION_FIELDS, 5)
    );
    assert!(
        offset_of!(FfiElementRegistration, properties)
            == field_offset(ELEMENT_REGISTRATION_FIELDS, 6)
    );
    assert!(
        offset_of!(FfiElementRegistration, property_count)
            == field_offset(ELEMENT_REGISTRATION_FIELDS, 7)
    );
    assert!(
        offset_of!(FfiElementRegistration, vtable) == field_offset(ELEMENT_REGISTRATION_FIELDS, 8)
    );
    assert!(
        offset_of!(FfiElementRegistration, create) == field_offset(ELEMENT_REGISTRATION_FIELDS, 9)
    );
    assert!(
        offset_of!(FfiElementRegistration, reserved)
            == field_offset(ELEMENT_REGISTRATION_FIELDS, 10)
    );

    const REGISTRAR_FIELDS: &[(usize, usize)] = &[
        U32,           // struct_size
        U32,           // version
        PTR,           // ctx
        FUNC,          // register_element
        func_array(4), // reserved
    ];
    assert!(size_of::<FfiRegistrar>() == struct_size(REGISTRAR_FIELDS));
    assert!(offset_of!(FfiRegistrar, struct_size) == 0);
    assert!(offset_of!(FfiRegistrar, version) == field_offset(REGISTRAR_FIELDS, 1));
    assert!(offset_of!(FfiRegistrar, ctx) == field_offset(REGISTRAR_FIELDS, 2));
    assert!(offset_of!(FfiRegistrar, register_element) == field_offset(REGISTRAR_FIELDS, 3));
    assert!(offset_of!(FfiRegistrar, reserved) == field_offset(REGISTRAR_FIELDS, 4));

    const CAPABILITY_FIELDS: &[(usize, usize)] = &[U32, U32, STR];
    assert!(size_of::<FfiCapability>() == struct_size(CAPABILITY_FIELDS));
    assert!(offset_of!(FfiCapability, kind) == field_offset(CAPABILITY_FIELDS, 0));
    assert!(offset_of!(FfiCapability, reserved) == field_offset(CAPABILITY_FIELDS, 1));
    assert!(offset_of!(FfiCapability, name) == field_offset(CAPABILITY_FIELDS, 2));

    // -----------------------------------------------------------------------
    // Descriptor
    // -----------------------------------------------------------------------

    const DESCRIPTOR_FIELDS: &[(usize, usize)] = &[
        U64,           // magic
        U32,           // abi_version
        U32,           // struct_size
        STR,           // name
        STR,           // version
        PTR,           // capabilities
        PTR,           // capability_count
        FUNC,          // register
        func_array(4), // reserved
    ];
    assert!(size_of::<FfiPluginDescriptor>() == struct_size(DESCRIPTOR_FIELDS));
    assert!(offset_of!(FfiPluginDescriptor, magic) == field_offset(DESCRIPTOR_FIELDS, 0));
    assert!(offset_of!(FfiPluginDescriptor, abi_version) == field_offset(DESCRIPTOR_FIELDS, 1));
    assert!(offset_of!(FfiPluginDescriptor, struct_size) == field_offset(DESCRIPTOR_FIELDS, 2));
    assert!(offset_of!(FfiPluginDescriptor, name) == field_offset(DESCRIPTOR_FIELDS, 3));
    assert!(offset_of!(FfiPluginDescriptor, version) == field_offset(DESCRIPTOR_FIELDS, 4));
    assert!(offset_of!(FfiPluginDescriptor, capabilities) == field_offset(DESCRIPTOR_FIELDS, 5));
    assert!(
        offset_of!(FfiPluginDescriptor, capability_count) == field_offset(DESCRIPTOR_FIELDS, 6)
    );
    assert!(offset_of!(FfiPluginDescriptor, register) == field_offset(DESCRIPTOR_FIELDS, 7));
    assert!(offset_of!(FfiPluginDescriptor, reserved) == field_offset(DESCRIPTOR_FIELDS, 8));
    // `validate_descriptor` reads these three at fixed offsets, through
    // `addr_of` on a pointer it has not yet checked, so no ABI generation may
    // move them.
    assert!(offset_of!(FfiPluginDescriptor, magic) == 0);
    assert!(offset_of!(FfiPluginDescriptor, abi_version) == 8);
    assert!(offset_of!(FfiPluginDescriptor, struct_size) == 12);

    // The wrapper that puts a descriptor in a `static`: `dlsym` hands the host
    // the symbol's address and the host reads a descriptor there, so the
    // wrapper must add nothing.
    assert!(size_of::<AbiStatic<FfiPluginDescriptor>>() == size_of::<FfiPluginDescriptor>());
    assert!(align_of::<AbiStatic<FfiPluginDescriptor>>() == align_of::<FfiPluginDescriptor>());

    // A C plugin returns a plain `int32_t`.
    assert!(size_of::<FfiStatus>() == size_of::<i32>());

    // The floor `read_versioned` enforces: a `struct_size` plus the `u32` after
    // it, which is exactly the head every versioned struct opens with.
    assert!(VERSIONED_HEADER_SIZE == field_offset(ELEMENT_VTABLE_FIELDS, 1) + U32.0);
    assert!(VERSIONED_HEADER_SIZE == field_offset(REGISTRAR_FIELDS, 1) + U32.0);
    assert!(VERSIONED_HEADER_SIZE == field_offset(OUTPUT_SINK_VTABLE_FIELDS, 1) + U32.0);
    assert!(VERSIONED_HEADER_SIZE == field_offset(ELEMENT_REGISTRATION_FIELDS, 1) + U32.0);
};
