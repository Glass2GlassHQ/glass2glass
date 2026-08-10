//! The **v2 plugin ABI**: a frozen `repr(C)` surface a plugin written in any
//! language (Rust built with a different toolchain, C, C++, Zig) can emit, and
//! the host can read without trusting the producing compiler.
//!
//! The v1 path ([`crate::declare_plugin!`]) passes Rust types (`Registry`,
//! `Frame`, `Box<dyn ...>`) straight across `dlopen`, which is only sound when
//! plugin and host share a toolchain, a `g2g-core` version, and a layout-
//! affecting feature set. The ABI tag enforces that, at the price of locking
//! plugins to one build. v2 is the cross-toolchain tier: nothing but `repr(C)`
//! structs, integers, pointer+length pairs, and `extern "C"` function pointers
//! crosses the boundary.
//!
//! # Shape
//!
//! A v2 plugin exports one **data** symbol, [`V2_DESCRIPTOR_SYMBOL`], pointing
//! at a `static` [`FfiPluginDescriptor`]. The host reads and validates it
//! before calling a single line of plugin code, which is what makes the
//! capability declaration (the element names and kinds the plugin intends to
//! register) a *pre-execution* fact the caller's policy can act on.
//!
//! Only then does the host call the descriptor's `register` entry with a host-
//! owned [`FfiRegistrar`], through which the plugin hands back one
//! [`FfiElementRegistration`] per element.
//!
//! # Versioning
//!
//! Two independent mechanisms, both needed:
//!
//! - **`abi_version`** on the descriptor gates the whole surface. The host
//!   refuses anything but [`V2_ABI_VERSION`]: a semantic change to an existing
//!   field bumps it.
//! - **`struct_size` + reserved slots** let a struct grow inside one
//!   `abi_version`. Every versioned struct starts with its own byte size, so a
//!   host reads `min(plugin_size, host_size)` bytes into a zeroed local: an
//!   older, smaller plugin struct leaves the host's newer fields null, and the
//!   host substitutes its own default. The trailing `reserved` function-pointer
//!   slots are the other half: a future entry point takes a reserved slot
//!   without changing the size, so an *older* host reading a *newer* plugin
//!   simply ignores it. A host therefore never rejects a non-null reserved
//!   slot, it ignores it, and a plugin that cannot work without the new entry
//!   declares a higher `abi_version` instead.
//!
//! # What crosses
//!
//! Deliberately small. Frames are **System memory only**, as pointer + length +
//! an owner-side free function (the shape
//! [`SystemSlice::from_foreign`](g2g_core::memory::SystemSlice::from_foreign)
//! already takes). GPU memory domains, the clock election, QoS, metadata, and
//! the allocation cascade do not cross v2 at all: the host-side wrapper element
//! answers those with the `AsyncElement` trait defaults.

use core::ffi::c_void;

use g2g_core::G2gError;

mod caps;
mod convert;
mod validate;

pub use caps::{
    caps_from_ffi, caps_into_ffi, caps_set_from_ffi, CapsCodeError, AUDIO_FORMAT_CODES,
    BYTE_STREAM_ENCODING_CODES, RAW_VIDEO_FORMAT_CODES, TEXT_FORMAT_CODES, VIDEO_CODEC_CODES,
};
pub use convert::{
    packet_from_ffi, packet_into_ffi, prop_from_ffi, prop_into_ffi, prop_kind_code, release_packet,
    spec_into_ffi,
};
pub use validate::{
    check_against_declaration, validate_descriptor, validate_element, ElementKind,
    PluginCapability, PluginDeclaration, ValidatedElement, ValidationError, MAX_NAME_LEN,
};

/// The exported **data** symbol a v2 plugin defines: a `static`
/// [`FfiPluginDescriptor`]. The host `dlsym`s this first and falls back to the
/// v1 `g2g_plugin_abi` / `g2g_plugin_register` pair when it is absent.
pub const V2_DESCRIPTOR_SYMBOL: &[u8] = b"g2g_plugin_v2_descriptor";

/// [`FfiPluginDescriptor::magic`]. The ASCII bytes `G2GABIv2` read as a
/// little-endian `u64`, so a descriptor is recognisable in a hex dump and a
/// wrong-endian or garbage symbol fails the very first check.
pub const V2_MAGIC: u64 = 0x3276_4942_4147_3247;

/// The ABI generation this header describes. A plugin declaring anything else
/// is refused: the host has no way to know what its fields mean.
pub const V2_ABI_VERSION: u32 = 2;

/// Longest string (element name, blurb, metadata field) accepted from a plugin.
/// A length beyond this is garbage or a hostile length field, not a name.
pub const MAX_STRING_LEN: usize = 4096;

/// Most capabilities one descriptor may declare.
pub const MAX_CAPABILITIES: usize = 256;

/// Most elements one plugin may register. Also bounds the capability list it
/// is checked against.
pub const MAX_ELEMENTS: usize = 256;

/// Most alternatives in one declared caps set (a pad template).
pub const MAX_CAPS_ALTERNATIVES: usize = 64;

/// Most properties one element may declare.
pub const MAX_PROPERTIES: usize = 256;

/// Largest frame payload accepted across the boundary, 1 GiB. Nothing in a
/// media pipeline legitimately hands a single System-memory frame larger than
/// this, so a bigger length is a corrupt or hostile field.
pub const MAX_FRAME_BYTES: usize = 1 << 30;

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

/// Result of a v2 call: `0` on success, one of the negative `STATUS_*` codes on
/// failure. Transparent over `i32` so C writes a plain `int32_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct FfiStatus(pub i32);

/// Success.
pub const STATUS_OK: FfiStatus = FfiStatus(0);
/// Unclassified failure.
pub const STATUS_ERROR: FfiStatus = FfiStatus(-1);
/// The caps offered do not intersect what this element accepts.
pub const STATUS_CAPS_MISMATCH: FfiStatus = FfiStatus(-2);
/// The element received data before a successful `configure_pipeline`.
pub const STATUS_NOT_CONFIGURED: FfiStatus = FfiStatus(-3);
/// A memory domain the element cannot consume (a v2 element sees System only).
pub const STATUS_UNSUPPORTED_DOMAIN: FfiStatus = FfiStatus(-4);
/// The pipeline is shutting down.
pub const STATUS_SHUTDOWN: FfiStatus = FfiStatus(-5);
/// No property by that name.
pub const STATUS_PROPERTY_UNKNOWN: FfiStatus = FfiStatus(-6);
/// The value handed to `set_property` had the wrong kind, or an out-of-range
/// value.
pub const STATUS_PROPERTY_VALUE: FfiStatus = FfiStatus(-7);

impl FfiStatus {
    /// Whether this is the success code.
    pub fn is_ok(self) -> bool {
        self == STATUS_OK
    }

    /// The host-side error a non-success status maps to. Any code the host does
    /// not recognise (including a *positive* one, which no version of this ABI
    /// ever returns) collapses to the unclassified error rather than being
    /// silently treated as success.
    pub fn into_error(self) -> Option<G2gError> {
        match self {
            STATUS_OK => None,
            STATUS_CAPS_MISMATCH => Some(G2gError::CapsMismatch),
            STATUS_NOT_CONFIGURED => Some(G2gError::NotConfigured),
            STATUS_UNSUPPORTED_DOMAIN => Some(G2gError::UnsupportedDomain),
            STATUS_SHUTDOWN => Some(G2gError::Shutdown),
            _ => Some(G2gError::Hardware(g2g_core::error::HardwareError::Other)),
        }
    }

    /// The status a host-side error crosses back as.
    pub fn from_error(error: &G2gError) -> FfiStatus {
        match error {
            G2gError::CapsMismatch => STATUS_CAPS_MISMATCH,
            G2gError::NotConfigured => STATUS_NOT_CONFIGURED,
            G2gError::UnsupportedDomain => STATUS_UNSUPPORTED_DOMAIN,
            G2gError::Shutdown => STATUS_SHUTDOWN,
            _ => STATUS_ERROR,
        }
    }
}

// ---------------------------------------------------------------------------
// Strings
// ---------------------------------------------------------------------------

/// A borrowed UTF-8 string as pointer + length. Not NUL-terminated: the length
/// is authoritative, so a plugin can point at a slice of a larger buffer.
///
/// Nothing about the bytes is trusted. The host bounds `len` by
/// [`MAX_STRING_LEN`], requires a non-null pointer whenever `len > 0`, and
/// runs a UTF-8 check before the bytes reach any `str`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiStr {
    /// First byte, or null for the empty string.
    pub ptr: *const u8,
    /// Length in bytes.
    pub len: usize,
}

impl FfiStr {
    /// The empty string.
    pub const EMPTY: FfiStr = FfiStr {
        ptr: core::ptr::null(),
        len: 0,
    };

    /// Borrow a Rust string for the duration of a call.
    pub const fn borrowed(s: &str) -> FfiStr {
        FfiStr {
            ptr: s.as_ptr(),
            len: s.len(),
        }
    }

    /// Read the string, rejecting a null pointer, an over-long length, or
    /// non-UTF-8 bytes.
    ///
    /// # Safety
    /// When `len > 0`, `ptr` must point at `len` initialised bytes that stay
    /// valid for the returned borrow. This is the one property the host cannot
    /// check and must take on trust from the plugin.
    pub unsafe fn as_str(&self) -> Result<&str, ValidationError> {
        if self.len == 0 {
            return Ok("");
        }
        if self.ptr.is_null() {
            return Err(ValidationError::NullString);
        }
        if self.len > MAX_STRING_LEN {
            return Err(ValidationError::StringTooLong { len: self.len });
        }
        // SAFETY: the caller guarantees `ptr` covers `len` initialised bytes,
        // and `len` is now known to be within `MAX_STRING_LEN`.
        let bytes = unsafe { core::slice::from_raw_parts(self.ptr, self.len) };
        core::str::from_utf8(bytes).map_err(|_| ValidationError::NotUtf8)
    }
}

// ---------------------------------------------------------------------------
// Caps
// ---------------------------------------------------------------------------

/// A [`Caps`](g2g_core::Caps) dimension, flattened. `kind` is one of the
/// `DIM_*` constants; `min` / `max` carry the bounds a `Range` needs and a
/// `Fixed` puts in `min`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FfiDim {
    /// `DIM_ANY` / `DIM_FIXED` / `DIM_RANGE`.
    pub kind: u32,
    /// `Fixed`'s value, or a `Range`'s lower bound.
    pub min: u32,
    /// A `Range`'s upper bound; ignored otherwise.
    pub max: u32,
}

/// Unconstrained dimension.
pub const DIM_ANY: u32 = 0;
/// A single concrete value, in [`FfiDim::min`].
pub const DIM_FIXED: u32 = 1;
/// An inclusive `[min, max]` interval.
pub const DIM_RANGE: u32 = 2;

/// A framerate constraint in Q16 fixed-point frames per second, laid out like
/// [`FfiDim`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FfiRate {
    /// `DIM_ANY` / `DIM_FIXED` / `DIM_RANGE`.
    pub kind: u32,
    /// Q16 fps: `Fixed`'s value, or a `Range`'s lower bound.
    pub min_q16: u32,
    /// Q16 fps upper bound; ignored unless `kind` is `DIM_RANGE`.
    pub max_q16: u32,
}

/// Raw pixel-buffer caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FfiRawVideoCaps {
    /// A code from [`RAW_VIDEO_FORMAT_CODES`].
    pub format: u32,
    /// Frame width.
    pub width: FfiDim,
    /// Frame height.
    pub height: FfiDim,
    /// Framerate.
    pub framerate: FfiRate,
    /// `INTERLACE_ANY` / `INTERLACE_PROGRESSIVE` / `INTERLACE_INTERLEAVED`.
    pub interlace: u32,
}

/// Scan structure unconstrained.
pub const INTERLACE_ANY: u32 = 0;
/// Progressive scan.
pub const INTERLACE_PROGRESSIVE: u32 = 1;
/// Both fields woven into one frame.
pub const INTERLACE_INTERLEAVED: u32 = 2;

/// Compressed video-bitstream caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FfiCompressedVideoCaps {
    /// A code from [`VIDEO_CODEC_CODES`].
    pub codec: u32,
    /// Nominal width until the bitstream parser confirms it.
    pub width: FfiDim,
    /// Nominal height.
    pub height: FfiDim,
    /// Nominal framerate.
    pub framerate: FfiRate,
}

/// Audio caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FfiAudioCaps {
    /// A code from [`AUDIO_FORMAT_CODES`].
    pub format: u32,
    /// Channel count; `0` is the "any / unknown" wildcard, and anything above
    /// 255 is refused (the host's channel count is a `u8`).
    pub channels: u32,
    /// Sample rate in Hz; `0` is the "any / unknown" wildcard.
    pub sample_rate: u32,
}

/// Opaque container / elementary byte-stream caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FfiByteStreamCaps {
    /// A code from [`BYTE_STREAM_ENCODING_CODES`].
    pub encoding: u32,
}

/// Text-stream caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FfiTextCaps {
    /// A code from [`TEXT_FORMAT_CODES`].
    pub format: u32,
}

/// The payload half of [`FfiCaps`]. Which member is live is decided entirely by
/// [`FfiCaps::tag`]; reading any other member is undefined, so the host reads it
/// only through [`caps_from_ffi`], which switches on the tag first.
#[repr(C)]
pub union FfiCapsBody {
    /// Live when the tag is [`CAPS_RAW_VIDEO`].
    pub raw_video: FfiRawVideoCaps,
    /// Live when the tag is [`CAPS_COMPRESSED_VIDEO`].
    pub compressed_video: FfiCompressedVideoCaps,
    /// Live when the tag is [`CAPS_AUDIO`].
    pub audio: FfiAudioCaps,
    /// Live when the tag is [`CAPS_BYTE_STREAM`].
    pub byte_stream: FfiByteStreamCaps,
    /// Live when the tag is [`CAPS_TEXT`].
    pub text: FfiTextCaps,
}

impl core::fmt::Debug for FfiCapsBody {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("FfiCapsBody(union)")
    }
}

impl Clone for FfiCapsBody {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for FfiCapsBody {}

/// A caps value: a discriminant plus the union its variant selects.
///
/// The v2 vocabulary is deliberately a subset of [`Caps`](g2g_core::Caps):
/// raw video, compressed video, audio, byte streams, and text. Tensor, KLV,
/// closed-caption, and sub-picture links stay host-native, and a host `Caps`
/// outside the subset fails conversion loudly rather than being coerced into a
/// neighbouring variant.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiCaps {
    /// One of the `CAPS_*` constants. `CAPS_NONE` means "no caps present".
    pub tag: u32,
    /// Padding so `body` is 8-byte aligned in every build. Must be zero.
    pub reserved: u32,
    /// The variant payload named by `tag`.
    pub body: FfiCapsBody,
}

/// No caps present (a zeroed [`FfiCaps`]).
pub const CAPS_NONE: u32 = 0;
/// [`FfiCapsBody::raw_video`] is live.
pub const CAPS_RAW_VIDEO: u32 = 1;
/// [`FfiCapsBody::compressed_video`] is live.
pub const CAPS_COMPRESSED_VIDEO: u32 = 2;
/// [`FfiCapsBody::audio`] is live.
pub const CAPS_AUDIO: u32 = 3;
/// [`FfiCapsBody::byte_stream`] is live.
pub const CAPS_BYTE_STREAM: u32 = 4;
/// [`FfiCapsBody::text`] is live.
pub const CAPS_TEXT: u32 = 5;

impl FfiCaps {
    /// A zeroed value, tagged [`CAPS_NONE`].
    pub const NONE: FfiCaps = FfiCaps {
        tag: CAPS_NONE,
        reserved: 0,
        body: FfiCapsBody {
            byte_stream: FfiByteStreamCaps { encoding: 0 },
        },
    };
}

/// An ordered set of caps alternatives, highest preference first: the data form
/// of a pad template. `count == 0` means "any" on a sink pad and "same as the
/// input" on a source pad.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiCapsSet {
    /// First alternative, or null when `count` is 0.
    pub alternatives: *const FfiCaps,
    /// Number of alternatives, bounded by [`MAX_CAPS_ALTERNATIVES`].
    pub count: usize,
}

impl FfiCapsSet {
    /// The empty set.
    pub const EMPTY: FfiCapsSet = FfiCapsSet {
        alternatives: core::ptr::null(),
        count: 0,
    };
}

// ---------------------------------------------------------------------------
// Frames and packets
// ---------------------------------------------------------------------------

/// One System-memory frame crossing the boundary, with its payload as
/// pointer + length + an owner-side free function.
///
/// **Ownership transfers with the struct.** Whoever receives an `FfiFrame` owns
/// the payload and must eventually call `free(free_user)` exactly once. A null
/// `free` means the bytes are static or otherwise outlive the pipeline and
/// nothing is released. This is exactly the contract
/// [`SystemSlice::from_foreign`](g2g_core::memory::SystemSlice::from_foreign)
/// takes, so a frame crosses in either direction without a copy.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiFrame {
    /// First payload byte, or null when `len` is 0.
    pub data: *const u8,
    /// Payload length, bounded by [`MAX_FRAME_BYTES`].
    pub len: usize,
    /// Releases the payload. Called once by the owner, with `free_user`.
    pub free: Option<unsafe extern "C" fn(*mut c_void)>,
    /// Opaque argument handed back to `free`.
    pub free_user: *mut c_void,
    /// Presentation timestamp, ns.
    pub pts_ns: u64,
    /// Decode timestamp, ns.
    pub dts_ns: u64,
    /// Frame duration, ns.
    pub duration_ns: u64,
    /// Media-clock capture time, ns, stream-relative.
    pub capture_ns: u64,
    /// Monotonic wall-clock time stamped at source ingestion, ns.
    pub arrival_ns: u64,
    /// Monotonically increasing frame counter.
    pub sequence: u64,
    /// Non-zero when this frame starts an independently decodable unit.
    pub keyframe: u32,
    /// Padding. Must be zero.
    pub reserved: u32,
}

impl FfiFrame {
    /// An empty frame with no payload and no owner.
    pub const EMPTY: FfiFrame = FfiFrame {
        data: core::ptr::null(),
        len: 0,
        free: None,
        free_user: core::ptr::null_mut(),
        pts_ns: 0,
        dts_ns: 0,
        duration_ns: 0,
        capture_ns: 0,
        arrival_ns: 0,
        sequence: 0,
        keyframe: 0,
        reserved: 0,
    };
}

/// A pipeline packet in its ABI form.
///
/// Only four of the host's [`PipelinePacket`](g2g_core::PipelinePacket)
/// variants cross: caps changes, data frames, end of stream, and flush. The
/// host-side wrapper forwards `Segment` downstream itself and never delivers a
/// `Tick` (a v2 element declares no tick interval), so a plugin has no case for
/// either.
///
/// A plain struct rather than a union: the two payloads together are small, and
/// a struct is one less thing a hand-written C plugin can get wrong.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiPacket {
    /// One of the `PACKET_*` constants.
    pub tag: u32,
    /// Padding. Must be zero.
    pub reserved: u32,
    /// Live when `tag` is [`PACKET_CAPS_CHANGED`].
    pub caps: FfiCaps,
    /// Live when `tag` is [`PACKET_DATA_FRAME`]; its payload ownership travels
    /// with the packet.
    pub frame: FfiFrame,
}

/// The packet slot is empty: either nothing to send, or the callee has taken
/// what was there. [`FfiOutputSinkVtable::poll_push`] writes this tag back into
/// the caller's slot when it commits a packet.
pub const PACKET_NONE: u32 = 0;
/// [`FfiPacket::caps`] is live.
pub const PACKET_CAPS_CHANGED: u32 = 1;
/// [`FfiPacket::frame`] is live.
pub const PACKET_DATA_FRAME: u32 = 2;
/// End of stream. The runner emits the pipeline's single EOS, so a transform
/// must not push one of its own; this is the cue to flush buffered output.
pub const PACKET_EOS: u32 = 3;
/// Discard in-flight and buffered data and reset position state.
pub const PACKET_FLUSH: u32 = 4;

impl FfiPacket {
    /// An empty packet slot.
    pub const NONE: FfiPacket = FfiPacket {
        tag: PACKET_NONE,
        reserved: 0,
        caps: FfiCaps::NONE,
        frame: FfiFrame::EMPTY,
    };
}

/// Downstream push outcome, as reported by
/// [`FfiOutputSinkVtable::poll_push`]. Only "accepted" crosses v2: the
/// reverse-channel signals (renegotiation, QoS, bitrate targets) stay
/// host-native, and the wrapper answers them with the trait defaults.
pub const PUSH_ACCEPTED: i32 = 0;

// ---------------------------------------------------------------------------
// Output sink
// ---------------------------------------------------------------------------

/// The host-provided downstream push, in poll form so a plugin awaits
/// backpressure instead of failing on a full link. Mirrors
/// [`OutputSink::poll_push`](g2g_core::OutputSink::poll_push).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiOutputSinkVtable {
    /// `size_of::<FfiOutputSinkVtable>()` as the host wrote it.
    pub struct_size: u32,
    /// Vtable revision within [`V2_ABI_VERSION`].
    pub version: u32,
    /// Drive one packet downstream.
    ///
    /// `packet` is an in/out slot. On [`core::task::Poll::Ready`] with
    /// [`STATUS_OK`] the sink has taken the packet and rewritten the slot's tag
    /// to [`PACKET_NONE`]; on pending it has taken nothing and the caller must
    /// re-poll with the same slot. The `cx` context is borrowed for the call
    /// only.
    pub poll_push: unsafe extern "C" fn(
        ctx: *mut c_void,
        cx: *mut async_ffi::FfiContext,
        packet: *mut FfiPacket,
    ) -> async_ffi::FfiPoll<FfiStatus>,
    /// Reserved for future entries; null in this revision. See the module
    /// "Versioning" note: a host ignores any it does not know.
    pub reserved: [Option<extern "C" fn()>; 4],
}

/// A downstream output: the host's opaque context plus its vtable.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiOutputSink {
    /// Opaque host state, passed back as the vtable's first argument.
    pub ctx: *mut c_void,
    /// Never null.
    pub vtable: *const FfiOutputSinkVtable,
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

/// A `num/den` fraction property value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FfiFraction {
    /// Numerator.
    pub num: i32,
    /// Denominator; never zero.
    pub den: i32,
}

/// The string payload of an [`FfiPropValue`].
///
/// When `free` is null the string is **borrowed** and valid only for the
/// duration of the call that carried it. When `free` is non-null the receiver
/// **owns** the string and must call `free(free_user)` once. That single rule
/// covers both directions: the host lends a borrowed string to `set_property`,
/// and a plugin hands back an owned one from `get_property`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiPropStr {
    /// First byte, or null for the empty string.
    pub ptr: *const u8,
    /// Length in bytes, bounded by [`MAX_STRING_LEN`].
    pub len: usize,
    /// Releases the string, or null when it is borrowed.
    pub free: Option<unsafe extern "C" fn(*mut c_void)>,
    /// Opaque argument handed back to `free`.
    pub free_user: *mut c_void,
}

/// The payload half of [`FfiPropValue`], selected by its `kind`.
#[repr(C)]
pub union FfiPropValueBody {
    /// Live for [`PROP_BOOL`]: zero is false, non-zero true.
    pub boolean: u32,
    /// Live for [`PROP_INT`].
    pub int: i64,
    /// Live for [`PROP_UINT`].
    pub uint: u64,
    /// Live for [`PROP_DOUBLE`].
    pub double: f64,
    /// Live for [`PROP_FRACTION`].
    pub fraction: FfiFraction,
    /// Live for [`PROP_STR`].
    pub string: FfiPropStr,
}

impl core::fmt::Debug for FfiPropValueBody {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("FfiPropValueBody(union)")
    }
}

impl Clone for FfiPropValueBody {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for FfiPropValueBody {}

/// A runtime property value: a kind discriminant plus its payload.
///
/// The v2 kinds are a subset of [`PropKind`](g2g_core::property::PropKind):
/// the flag-set kind does not cross, because its value is a list of strings
/// whose ownership rules would double the surface for one rarely-used property
/// shape. An element registration that declares a flags property is refused.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiPropValue {
    /// One of the `PROP_*` constants.
    pub kind: u32,
    /// Padding. Must be zero.
    pub reserved: u32,
    /// The payload named by `kind`.
    pub body: FfiPropValueBody,
}

/// No value present (a zeroed [`FfiPropValue`]).
pub const PROP_NONE: u32 = 0;
/// Boolean.
pub const PROP_BOOL: u32 = 1;
/// Signed 64-bit integer.
pub const PROP_INT: u32 = 2;
/// Unsigned 64-bit integer.
pub const PROP_UINT: u32 = 3;
/// Double-precision float.
pub const PROP_DOUBLE: u32 = 4;
/// `num/den` fraction.
pub const PROP_FRACTION: u32 = 5;
/// UTF-8 string.
pub const PROP_STR: u32 = 6;

impl FfiPropValue {
    /// A zeroed value, tagged [`PROP_NONE`].
    pub const NONE: FfiPropValue = FfiPropValue {
        kind: PROP_NONE,
        reserved: 0,
        body: FfiPropValueBody { uint: 0 },
    };
}

/// Static description of one property an element exposes: the data form of
/// [`PropertySpec`](g2g_core::property::PropertySpec).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiPropertySpec {
    /// Property name as a `gst-launch` line writes it (`key=value`).
    pub name: FfiStr,
    /// One of the `PROP_*` value kinds. [`PROP_NONE`] is not a kind.
    pub kind: u32,
    /// Non-zero if the property can be read back.
    pub readable: u32,
    /// Non-zero if the property can be set.
    pub writable: u32,
    /// Padding. Must be zero.
    pub reserved: u32,
    /// One-line human description.
    pub blurb: FfiStr,
    /// Default value as text, or empty for none.
    pub default_value: FfiStr,
}

// ---------------------------------------------------------------------------
// Element vtable and registration
// ---------------------------------------------------------------------------

/// Per-instance entry points of a v2 element.
///
/// `process` and `destroy` are required; the rest are optional, and the host
/// substitutes the `AsyncElement` trait default for any that is null (accept
/// the caps, ignore the output caps, report no such property). A vtable whose
/// `struct_size` is smaller than the host's is read as a prefix and the missing
/// tail defaults the same way.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiElementVtable {
    /// `size_of::<FfiElementVtable>()` as the plugin wrote it. At least
    /// [`VERSIONED_HEADER_SIZE`].
    pub struct_size: u32,
    /// Vtable revision within [`V2_ABI_VERSION`].
    pub version: u32,

    /// Accept the negotiated input caps. Optional; absent means "accept".
    ///
    /// Returning [`STATUS_CAPS_MISMATCH`] rejects the caps. `refixate` is an
    /// out-parameter: writing a caps value with a tag other than
    /// [`CAPS_NONE`] asks the solver to try again with that proposal, which is
    /// a hard error at pipeline start (the caps are already fixated by then).
    pub configure_pipeline: Option<
        unsafe extern "C" fn(
            elem: *mut c_void,
            caps: *const FfiCaps,
            refixate: *mut FfiCaps,
        ) -> FfiStatus,
    >,

    /// Receive the element's own negotiated **output** caps. Optional.
    pub configure_output:
        Option<unsafe extern "C" fn(elem: *mut c_void, caps: *const FfiCaps) -> FfiStatus>,

    /// Handle one packet, pushing any output through `out`. **Required.**
    ///
    /// Ownership of `packet`'s payload transfers to the element. The returned
    /// future borrows both `elem` and `out`; the host drops it before either
    /// goes away, and polls it to completion or drops it, never both.
    pub process: Option<
        unsafe extern "C" fn(
            elem: *mut c_void,
            packet: FfiPacket,
            out: FfiOutputSink,
        ) -> async_ffi::LocalFfiFuture<FfiStatus>,
    >,

    /// Set a property by name. Optional; absent means the element has none.
    /// `value` is borrowed for the call.
    pub set_property: Option<
        unsafe extern "C" fn(
            elem: *mut c_void,
            name: FfiStr,
            value: *const FfiPropValue,
        ) -> FfiStatus,
    >,

    /// Read a property back by name. Optional. Writes the value through `out`
    /// and returns [`STATUS_OK`], or [`STATUS_PROPERTY_UNKNOWN`].
    pub get_property: Option<
        unsafe extern "C" fn(elem: *mut c_void, name: FfiStr, out: *mut FfiPropValue) -> FfiStatus,
    >,

    /// Destroy an instance built by [`FfiElementRegistration::create`].
    /// **Required.** Called exactly once per instance.
    pub destroy: Option<unsafe extern "C" fn(elem: *mut c_void)>,

    /// Reserved for future entries; null in this revision.
    pub reserved: [Option<extern "C" fn()>; 6],
}

/// An element kind code: a 1-in / 1-out transform.
pub const ELEMENT_TRANSFORM: u32 = 1;
/// An element kind code: a terminal sink.
pub const ELEMENT_SINK: u32 = 2;

/// Static introspection metadata, the data form of
/// [`ElementMetadata`](g2g_core::property::ElementMetadata).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiElementMetadata {
    /// Human-readable name, e.g. `"Example filter"`.
    pub long_name: FfiStr,
    /// Classification, e.g. `"Filter/Effect/Video"`.
    pub klass: FfiStr,
    /// One-paragraph description.
    pub description: FfiStr,
    /// Author / origin.
    pub author: FfiStr,
}

impl FfiElementMetadata {
    /// All fields empty.
    pub const EMPTY: FfiElementMetadata = FfiElementMetadata {
        long_name: FfiStr::EMPTY,
        klass: FfiStr::EMPTY,
        description: FfiStr::EMPTY,
        author: FfiStr::EMPTY,
    };
}

/// One element a plugin hands to the host's registrar.
///
/// Every pointer in here must stay valid for the life of the process: the host
/// keeps a loaded plugin's code and data mapped forever, and the registered
/// element is built from these fields long after `register` returned.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiElementRegistration {
    /// `size_of::<FfiElementRegistration>()` as the plugin wrote it.
    pub struct_size: u32,
    /// [`ELEMENT_TRANSFORM`] or [`ELEMENT_SINK`]. Must match the kind the
    /// descriptor declared for this name.
    pub kind: u32,
    /// The `gst-launch` element name.
    pub name: FfiStr,
    /// Introspection metadata.
    pub metadata: FfiElementMetadata,
    /// Caps the element accepts on its input pad. Empty means any.
    pub sink_caps: FfiCapsSet,
    /// Caps the element produces. Empty means "the input caps unchanged", the
    /// pass-through shape. A non-empty set must be fully concrete: an `Any`
    /// dimension or framerate here cannot survive fixation and is refused.
    pub source_caps: FfiCapsSet,
    /// First property spec, or null.
    pub properties: *const FfiPropertySpec,
    /// Number of property specs, bounded by [`MAX_PROPERTIES`].
    pub property_count: usize,
    /// The element's entry points. Never null.
    pub vtable: *const FfiElementVtable,
    /// Build one instance. **Required.** Returns null on failure.
    pub create: Option<unsafe extern "C" fn() -> *mut c_void>,
    /// Reserved for future entries; null in this revision.
    pub reserved: [Option<extern "C" fn()>; 4],
}

/// The host-owned object a plugin registers its elements through.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiRegistrar {
    /// `size_of::<FfiRegistrar>()` as the host wrote it.
    pub struct_size: u32,
    /// Registrar revision within [`V2_ABI_VERSION`].
    pub version: u32,
    /// Opaque host state, passed back as the first argument.
    pub ctx: *mut c_void,
    /// Register one element. Returns [`STATUS_OK`], or an error when the host
    /// refuses it (an undeclared name, a malformed registration, too many
    /// elements). A refusal fails the whole load, so a plugin need not unwind.
    pub register_element:
        unsafe extern "C" fn(ctx: *mut c_void, element: *const FfiElementRegistration) -> FfiStatus,
    /// Reserved for future entries; null in this revision.
    pub reserved: [Option<extern "C" fn()>; 4],
}

/// One capability a descriptor declares up front: what the plugin intends to
/// register, readable before any plugin code runs.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiCapability {
    /// [`ELEMENT_TRANSFORM`] or [`ELEMENT_SINK`]. An unknown code is carried
    /// through to the policy rather than rejected, so a future capability kind
    /// is a policy decision on an old host rather than a hard failure.
    pub kind: u32,
    /// Padding. Must be zero.
    pub reserved: u32,
    /// The element name this capability covers.
    pub name: FfiStr,
}

/// The static a v2 plugin exports under [`V2_DESCRIPTOR_SYMBOL`].
///
/// Read and validated before any plugin code runs. `register` is the only entry
/// point, and the host calls it only after the descriptor validates *and* the
/// caller's policy has allowed the declared capabilities.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FfiPluginDescriptor {
    /// [`V2_MAGIC`]. First field so a wrong symbol fails immediately.
    pub magic: u64,
    /// [`V2_ABI_VERSION`].
    pub abi_version: u32,
    /// `size_of::<FfiPluginDescriptor>()` as the plugin wrote it.
    pub struct_size: u32,
    /// Plugin name, for diagnostics and policy.
    pub name: FfiStr,
    /// Plugin version string, for diagnostics.
    pub version: FfiStr,
    /// First capability, or null.
    pub capabilities: *const FfiCapability,
    /// Number of capabilities, bounded by [`MAX_CAPABILITIES`].
    pub capability_count: usize,
    /// Register the declared elements. **Required.**
    pub register: Option<unsafe extern "C" fn(registrar: *const FfiRegistrar) -> FfiStatus>,
    /// Reserved for future entries; null in this revision.
    pub reserved: [Option<extern "C" fn()>; 4],
}

/// Wrapper that lets an ABI table live in a `static`.
///
/// Every table a plugin exports (the descriptor, its capability array, its
/// vtables, its property specs) holds raw pointers, so Rust will not put one in
/// a `static` on its own. `repr(transparent)`, so the symbol's address is the
/// table's address, which is what `dlsym` hands the host.
///
/// Only for **immutable** v2 ABI tables, whose pointers address other immutable
/// statics in the same library. Nothing else may use it.
#[derive(Debug)]
#[repr(transparent)]
pub struct AbiStatic<T>(pub T);

// SAFETY: the wrapped value is an immutable ABI table. Nothing ever writes it,
// and the pointers it holds address other immutable statics in the same
// library, so sharing it across threads is a read of constant data.
unsafe impl<T> Sync for AbiStatic<T> {}

/// Bytes every versioned struct's `struct_size` must at least cover: its own
/// size field plus the `u32` after it. A smaller declared size cannot even
/// describe itself and is refused.
pub const VERSIONED_HEADER_SIZE: usize = 8;

/// Copy a plugin-declared struct into a host-sized, zero-filled local.
///
/// This is the whole forward/backward-compatibility mechanism: the plugin's
/// `struct_size` says how many bytes it actually wrote, so the host copies
/// `min(declared, host)` of them and leaves the rest zero. Zero reads as `None`
/// for every optional function pointer and as null for every pointer, which is
/// exactly "absent, use the host default".
///
/// # Safety
/// `ptr` must be non-null, aligned for `T`, and point at `declared_size` bytes
/// the plugin actually wrote.
pub unsafe fn read_versioned<T: Copy>(
    ptr: *const T,
    declared_size: usize,
) -> Result<T, ValidationError> {
    if declared_size < VERSIONED_HEADER_SIZE {
        return Err(ValidationError::StructTooSmall {
            declared: declared_size,
        });
    }
    let host_size = core::mem::size_of::<T>();
    let copy = declared_size.min(host_size);
    // SAFETY: `T` is a `repr(C)` POD whose fields are integers, raw pointers,
    // unions of those, and `Option<extern "C" fn>`. An all-zero bit pattern is
    // a valid value for every one of them (null pointers, `None`), so a zeroed
    // `T` is initialised before the prefix copy overwrites the leading `copy`
    // bytes.
    let mut out: T = unsafe { core::mem::zeroed() };
    // SAFETY: the caller guarantees `ptr` covers `declared_size` readable
    // bytes, and `copy` is at most that and at most `size_of::<T>()`, so
    // neither side of the copy runs past its allocation. The regions cannot
    // overlap: `out` is a fresh local.
    unsafe {
        core::ptr::copy_nonoverlapping(
            ptr.cast::<u8>(),
            core::ptr::addr_of_mut!(out).cast::<u8>(),
            copy,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_spells_the_ascii_tag() {
        // A hex dump of a real descriptor should read `G2GABIv2`, which is what
        // makes a wrong or garbage symbol obvious rather than mysterious.
        assert_eq!(&V2_MAGIC.to_le_bytes(), b"G2GABIv2");
    }

    #[test]
    fn status_round_trips_the_mapped_errors() {
        for error in [
            G2gError::CapsMismatch,
            G2gError::NotConfigured,
            G2gError::UnsupportedDomain,
            G2gError::Shutdown,
        ] {
            let status = FfiStatus::from_error(&error);
            assert_eq!(status.into_error(), Some(error));
        }
    }

    #[test]
    fn unknown_status_codes_are_errors_not_success() {
        // The dangerous direction: a code the host does not know must never be
        // read as "the call succeeded". Positive codes are not part of the ABI
        // at all and must fail the same way.
        assert!(FfiStatus(-999).into_error().is_some());
        assert!(FfiStatus(7).into_error().is_some());
        assert!(STATUS_OK.into_error().is_none());
    }

    #[test]
    fn read_versioned_zero_fills_a_short_struct() {
        // The compatibility guarantee: a plugin built against an older, smaller
        // vtable leaves the host's newer entries null, not garbage.
        let full = FfiElementVtable {
            struct_size: core::mem::size_of::<FfiElementVtable>() as u32,
            version: 1,
            configure_pipeline: None,
            configure_output: None,
            process: None,
            set_property: None,
            get_property: None,
            destroy: None,
            reserved: [None; 6],
        };
        // Pretend the plugin only wrote up to and including `process`.
        let short = core::mem::offset_of!(FfiElementVtable, set_property);
        // SAFETY: `full` is a live, fully written local, so reading any prefix
        // of it is in bounds.
        let read: FfiElementVtable =
            unsafe { read_versioned(&full as *const _, short) }.expect("prefix reads");
        assert!(read.set_property.is_none());
        assert!(read.get_property.is_none());
        assert!(read.destroy.is_none());
        assert!(read.reserved.iter().all(Option::is_none));
    }

    #[test]
    fn read_versioned_refuses_a_size_that_cannot_describe_itself() {
        let full = FfiElementVtable {
            struct_size: 4,
            version: 0,
            configure_pipeline: None,
            configure_output: None,
            process: None,
            set_property: None,
            get_property: None,
            destroy: None,
            reserved: [None; 6],
        };
        // SAFETY: `full` is a live local; the declared size is rejected before
        // any read happens.
        let err = unsafe { read_versioned::<FfiElementVtable>(&full as *const _, 4) }
            .expect_err("4 bytes is not a struct");
        assert!(matches!(
            err,
            ValidationError::StructTooSmall { declared: 4 }
        ));
    }

    #[test]
    fn read_versioned_ignores_a_longer_plugin_struct() {
        // A plugin built against a NEWER header declares a bigger size; the host
        // must read its own prefix and ignore the tail rather than refuse.
        let full = FfiElementVtable {
            struct_size: 4096,
            version: 99,
            configure_pipeline: None,
            configure_output: None,
            process: None,
            set_property: None,
            get_property: None,
            destroy: None,
            reserved: [None; 6],
        };
        // SAFETY: `full` is a live local; the copy is capped at the host's own
        // struct size, so the over-large declared size reads nothing extra.
        let read: FfiElementVtable =
            unsafe { read_versioned(&full as *const _, 4096) }.expect("longer struct reads");
        assert_eq!(read.version, 99);
    }
}
