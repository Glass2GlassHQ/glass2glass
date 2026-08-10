//! Packet, frame, and property conversion across the v2 boundary.
//!
//! Both sides need both directions (the host lends a frame down and takes one
//! back; a plugin does the mirror image), so the conversions live here rather
//! than once in the loader and once in the SDK, where the two copies could
//! drift on an ownership rule and leak or double-free.
//!
//! # Payload ownership
//!
//! A frame's bytes cross as pointer + length + `free` + `free_user`, and
//! ownership moves with the [`FfiPacket`]. Every path below either hands that
//! ownership on or releases it, never both and never neither:
//!
//! - [`packet_into_ffi`] boxes the host's [`SystemSlice`] and hands out a `free`
//!   that drops the box, so the receiver's eventual `free` call is what ends the
//!   allocation's life.
//! - [`packet_from_ffi`] wraps the pointer with
//!   [`SystemSlice::from_foreign`], which calls `free` when the frame is
//!   dropped. If it rejects the frame first, it calls `free` itself, because the
//!   sender has already let go.
//! - [`release_packet`] is for a packet that was taken but never delivered (a
//!   cancelled push, a dropped output).

use std::boxed::Box;
use std::string::{String, ToString};

use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::property::{PropKind, PropValue, PropertySpec};
use g2g_core::{Frame, FrameTiming, G2gError, PipelinePacket};

use core::ffi::c_void;

use super::caps::{caps_from_ffi, caps_into_ffi};
use super::{
    FfiFraction, FfiFrame, FfiPacket, FfiPropStr, FfiPropValue, FfiPropValueBody, FfiPropertySpec,
    FfiStatus, FfiStr, MAX_FRAME_BYTES, MAX_STRING_LEN, PACKET_CAPS_CHANGED, PACKET_DATA_FRAME,
    PACKET_EOS, PACKET_FLUSH, PROP_BOOL, PROP_DOUBLE, PROP_FRACTION, PROP_INT, PROP_STR, PROP_UINT,
    STATUS_ERROR,
};

/// Release a `SystemSlice` that was lent across the boundary.
///
/// # Safety
/// `user` must be a pointer [`packet_into_ffi`] produced with `Box::into_raw`,
/// released exactly once.
unsafe extern "C" fn drop_lent_slice(user: *mut c_void) {
    if user.is_null() {
        return;
    }
    // SAFETY: forwarded from this function's contract.
    drop(unsafe { Box::from_raw(user.cast::<SystemSlice>()) });
}

/// Convert a host packet into its ABI form, transferring payload ownership to
/// whoever receives the result.
///
/// Only the four packet kinds v2 carries convert. `Segment` and `Tick` are the
/// host wrapper's business and never reach a plugin.
pub fn packet_into_ffi(packet: PipelinePacket) -> Result<FfiPacket, G2gError> {
    match packet {
        PipelinePacket::CapsChanged(caps) => {
            let caps = caps_into_ffi(&caps).map_err(|_| G2gError::CapsMismatch)?;
            Ok(FfiPacket {
                tag: PACKET_CAPS_CHANGED,
                caps,
                ..FfiPacket::NONE
            })
        }
        PipelinePacket::DataFrame(frame) => {
            let MemoryDomain::System(slice) = frame.domain else {
                return Err(G2gError::UnsupportedDomain);
            };
            // The slice moves to the heap so its address survives this call; the
            // payload it points at is untouched, so nothing is copied.
            let boxed = Box::new(slice);
            let bytes = boxed.as_slice();
            let data = bytes.as_ptr();
            let len = bytes.len();
            let free_user = Box::into_raw(boxed).cast::<c_void>();
            Ok(FfiPacket {
                tag: PACKET_DATA_FRAME,
                frame: FfiFrame {
                    data,
                    len,
                    free: Some(drop_lent_slice),
                    free_user,
                    pts_ns: frame.timing.pts_ns,
                    dts_ns: frame.timing.dts_ns,
                    duration_ns: frame.timing.duration_ns,
                    capture_ns: frame.timing.capture_ns,
                    arrival_ns: frame.timing.arrival_ns,
                    sequence: frame.sequence,
                    keyframe: u32::from(frame.timing.keyframe),
                    reserved: 0,
                },
                ..FfiPacket::NONE
            })
        }
        PipelinePacket::Eos => Ok(FfiPacket {
            tag: PACKET_EOS,
            ..FfiPacket::NONE
        }),
        PipelinePacket::Flush => Ok(FfiPacket {
            tag: PACKET_FLUSH,
            ..FfiPacket::NONE
        }),
        _ => Err(G2gError::UnsupportedDomain),
    }
}

/// Convert a received packet into a host packet, taking ownership of its
/// payload. A rejected frame is released here, since the sender has already let
/// go of it.
///
/// # Safety
/// `packet` must be a value the sender filled in, whose `data` / `len` describe
/// that many readable bytes for as long as the frame lives, and whose `free` may
/// be called once. That is the one property no amount of validation can check;
/// it comes from the ABI contract.
pub unsafe fn packet_from_ffi(packet: FfiPacket) -> Result<PipelinePacket, FfiStatus> {
    match packet.tag {
        PACKET_CAPS_CHANGED => {
            let caps = caps_from_ffi(&packet.caps).map_err(|_| STATUS_ERROR)?;
            Ok(PipelinePacket::CapsChanged(caps))
        }
        PACKET_DATA_FRAME => {
            let f = packet.frame;
            if f.len > MAX_FRAME_BYTES || (f.len > 0 && f.data.is_null()) {
                // SAFETY: ownership came with the packet; releasing it is the
                // only way not to leak a payload that is about to be rejected.
                unsafe { release_frame(&f) };
                return Err(STATUS_ERROR);
            }
            // SAFETY: forwarded from this function's contract; `free` /
            // `free_user` are the release pair the ABI requires alongside them.
            let slice = unsafe { SystemSlice::from_foreign(f.data, f.len, f.free, f.free_user) };
            Ok(PipelinePacket::DataFrame(Frame::new(
                MemoryDomain::System(slice),
                FrameTiming {
                    pts_ns: f.pts_ns,
                    dts_ns: f.dts_ns,
                    duration_ns: f.duration_ns,
                    capture_ns: f.capture_ns,
                    arrival_ns: f.arrival_ns,
                    keyframe: f.keyframe != 0,
                },
                f.sequence,
            )))
        }
        PACKET_EOS => Ok(PipelinePacket::Eos),
        PACKET_FLUSH => Ok(PipelinePacket::Flush),
        _ => Err(STATUS_ERROR),
    }
}

/// # Safety
/// The frame's `free` / `free_user` pair must be one this call may run once.
unsafe fn release_frame(frame: &FfiFrame) {
    if let Some(free) = frame.free {
        // SAFETY: forwarded from this function's contract.
        unsafe { free(frame.free_user) };
    }
}

/// Release a packet that was taken but never delivered: a push the peer
/// abandoned, or an output nobody consumed.
///
/// # Safety
/// The packet's payload must still be owned by the caller, and this must run
/// once for it.
pub unsafe fn release_packet(packet: &FfiPacket) {
    if packet.tag == PACKET_DATA_FRAME {
        // SAFETY: forwarded from this function's contract.
        unsafe { release_frame(&packet.frame) };
    }
}

/// Convert a property value for a call that borrows it. A string crosses as a
/// borrow (`free` null), valid only while the call runs.
///
/// `None` for the flag-set kind, which does not cross v2. Element validation
/// refuses a registration that declares one, so no live element reaches it.
pub fn prop_into_ffi(value: &PropValue) -> Option<FfiPropValue> {
    let (kind, body) = match value {
        PropValue::Bool(b) => (
            PROP_BOOL,
            FfiPropValueBody {
                boolean: u32::from(*b),
            },
        ),
        PropValue::Int(v) => (PROP_INT, FfiPropValueBody { int: *v }),
        PropValue::Uint(v) => (PROP_UINT, FfiPropValueBody { uint: *v }),
        PropValue::Double(v) => (PROP_DOUBLE, FfiPropValueBody { double: *v }),
        PropValue::Fraction(num, den) => (
            PROP_FRACTION,
            FfiPropValueBody {
                fraction: FfiFraction {
                    num: *num,
                    den: *den,
                },
            },
        ),
        PropValue::Str(s) => (
            PROP_STR,
            FfiPropValueBody {
                string: FfiPropStr {
                    ptr: s.as_ptr(),
                    len: s.len(),
                    free: None,
                    free_user: core::ptr::null_mut(),
                },
            },
        ),
        _ => return None,
    };
    Some(FfiPropValue {
        kind,
        reserved: 0,
        body,
    })
}

/// Convert a received property value, releasing an owned string payload.
///
/// # Safety
/// `value`'s `kind` must name the live union member, and a string payload with
/// a non-null `free` must be one this call may release once.
pub unsafe fn prop_from_ffi(value: &FfiPropValue) -> Option<PropValue> {
    match value.kind {
        PROP_BOOL => {
            // SAFETY: the kind tag is the ABI's sole authority on which union
            // member is live.
            Some(PropValue::Bool(unsafe { value.body.boolean } != 0))
        }
        PROP_INT => {
            // SAFETY: as above.
            Some(PropValue::Int(unsafe { value.body.int }))
        }
        PROP_UINT => {
            // SAFETY: as above.
            Some(PropValue::Uint(unsafe { value.body.uint }))
        }
        PROP_DOUBLE => {
            // SAFETY: as above.
            Some(PropValue::Double(unsafe { value.body.double }))
        }
        PROP_FRACTION => {
            // SAFETY: as above.
            let f = unsafe { value.body.fraction };
            (f.den != 0).then_some(PropValue::Fraction(f.num, f.den))
        }
        PROP_STR => {
            // SAFETY: as above.
            let s = unsafe { value.body.string };
            let text = if s.len == 0 {
                Some(String::new())
            } else if s.ptr.is_null() || s.len > MAX_STRING_LEN {
                None
            } else {
                // SAFETY: the pointer is non-null and the length bounded; the
                // sender's contract is that they describe its string.
                let bytes = unsafe { core::slice::from_raw_parts(s.ptr, s.len) };
                core::str::from_utf8(bytes).ok().map(ToString::to_string)
            };
            if let Some(free) = s.free {
                // SAFETY: a non-null `free` means ownership transferred to this
                // call, which releases it exactly once.
                unsafe { free(s.free_user) };
            }
            text.map(PropValue::Str)
        }
        _ => None,
    }
}

/// The ABI code for a property kind, or `None` for the flag-set kind, which
/// does not cross v2.
pub fn prop_kind_code(kind: PropKind) -> Option<u32> {
    match kind {
        PropKind::Bool => Some(PROP_BOOL),
        PropKind::Int => Some(PROP_INT),
        PropKind::Uint => Some(PROP_UINT),
        PropKind::Double => Some(PROP_DOUBLE),
        PropKind::Fraction => Some(PROP_FRACTION),
        PropKind::Str => Some(PROP_STR),
        _ => None,
    }
}

/// Convert a host property spec into its ABI form, for a plugin publishing what
/// it exposes. `None` for a kind v2 does not carry.
pub fn spec_into_ffi(spec: &PropertySpec) -> Option<FfiPropertySpec> {
    Some(FfiPropertySpec {
        name: FfiStr::borrowed(spec.name),
        kind: prop_kind_code(spec.kind)?,
        readable: u32::from(spec.flags.readable),
        writable: u32::from(spec.flags.writable),
        reserved: 0,
        blurb: FfiStr::borrowed(spec.blurb),
        default_value: FfiStr::borrowed(spec.default.unwrap_or("")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::caps::{Caps, TextFormat};

    #[test]
    fn a_frame_round_trips_without_copying_its_bytes() {
        // The zero-copy claim: what comes back points at the same allocation
        // that went out, and the payload is released exactly once (leak / double
        // free would show up under the test runner's allocator otherwise).
        let payload: Box<[u8]> = Box::new([1u8, 2, 3, 4]);
        let original = payload.as_ptr();
        let packet = PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(payload)),
            FrameTiming {
                pts_ns: 42,
                keyframe: true,
                ..FrameTiming::default()
            },
            7,
        ));

        let ffi = packet_into_ffi(packet).expect("a System frame crosses");
        assert_eq!(ffi.frame.data, original, "the bytes were lent, not copied");
        assert_eq!(ffi.frame.len, 4);

        // SAFETY: `ffi` is the value just produced, whose payload this test owns.
        let back = unsafe { packet_from_ffi(ffi) }.expect("and comes back");
        match back {
            PipelinePacket::DataFrame(frame) => {
                assert_eq!(frame.sequence, 7);
                assert_eq!(frame.timing.pts_ns, 42);
                assert!(frame.timing.keyframe);
                let MemoryDomain::System(slice) = &frame.domain else {
                    panic!("still System memory");
                };
                assert_eq!(slice.as_slice(), &[1, 2, 3, 4]);
                assert_eq!(slice.as_slice().as_ptr(), original);
            }
            other => panic!("expected a data frame, got {other:?}"),
        }
    }

    #[test]
    fn a_frame_with_an_absurd_length_is_rejected_and_released() {
        // A hostile length must not become a slice. The payload is released on
        // the way out, which is what the free-count assertion below checks.
        static FREED: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
        unsafe extern "C" fn count_free(_user: *mut c_void) {
            FREED.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        }
        let bytes = [0u8; 4];
        let ffi = FfiPacket {
            tag: PACKET_DATA_FRAME,
            frame: FfiFrame {
                data: bytes.as_ptr(),
                len: MAX_FRAME_BYTES + 1,
                free: Some(count_free),
                ..FfiFrame::EMPTY
            },
            ..FfiPacket::NONE
        };
        // SAFETY: the length is the hostile field under test; nothing reads it.
        let err = unsafe { packet_from_ffi(ffi) }.expect_err("an absurd length is rejected");
        assert_eq!(err, STATUS_ERROR);
        assert_eq!(
            FREED.load(core::sync::atomic::Ordering::SeqCst),
            1,
            "the rejected payload was released, not leaked"
        );
    }

    #[test]
    fn a_null_pointer_with_a_length_is_rejected() {
        let ffi = FfiPacket {
            tag: PACKET_DATA_FRAME,
            frame: FfiFrame {
                data: core::ptr::null(),
                len: 16,
                ..FfiFrame::EMPTY
            },
            ..FfiPacket::NONE
        };
        // SAFETY: the null pointer is the case under test; nothing reads it.
        assert!(unsafe { packet_from_ffi(ffi) }.is_err());
    }

    #[test]
    fn caps_and_control_packets_round_trip() {
        for packet in [
            PipelinePacket::CapsChanged(Caps::Text {
                format: TextFormat::Utf8,
            }),
            PipelinePacket::Eos,
            PipelinePacket::Flush,
        ] {
            let expected = alloc_debug(&packet);
            let ffi = packet_into_ffi(packet).expect("crosses");
            // SAFETY: the value just produced, with no payload to own.
            let back = unsafe { packet_from_ffi(ffi) }.expect("comes back");
            assert_eq!(alloc_debug(&back), expected);
        }
    }

    fn alloc_debug(packet: &PipelinePacket) -> String {
        std::format!("{packet:?}")
    }

    #[test]
    fn a_segment_does_not_cross() {
        // Segment and Tick are the host wrapper's business; a conversion attempt
        // must fail rather than silently dropping the packet.
        assert!(packet_into_ffi(PipelinePacket::Tick).is_err());
    }

    #[test]
    fn property_values_round_trip() {
        for value in [
            PropValue::Bool(true),
            PropValue::Int(-9),
            PropValue::Uint(9),
            PropValue::Double(1.5),
            PropValue::Fraction(30, 1),
            PropValue::Str("hello".to_string()),
        ] {
            let ffi = prop_into_ffi(&value).expect("crosses v2");
            // SAFETY: the value just produced; its string is borrowed from
            // `value`, which outlives this call, and carries no `free`.
            let back = unsafe { prop_from_ffi(&ffi) }.expect("comes back");
            assert_eq!(back, value);
        }
    }

    #[test]
    fn a_flag_set_property_does_not_cross() {
        let flags = PropValue::Flags(std::vec![String::from("a")]);
        assert!(prop_into_ffi(&flags).is_none());
    }
}
