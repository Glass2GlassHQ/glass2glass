//! GStreamer's `shmpipe` control protocol (`sys/shm/shmpipe.c` in
//! gst-plugins-bad), the wire the `shmsink` / `shmsrc` pair speaks. A unix
//! socket carries fixed-size commands; the buffers themselves live in a POSIX
//! shared-memory area whose name travels in the first command.
//!
//! A command is the C `struct CommandBuffer` sent raw, so it is native-endian
//! and native-width and both ends must share an ABI. The offsets here come from
//! `offset_of!` on `#[repr(C)]` mirrors of the C declarations rather than from
//! hand-counted numbers, and a 64-bit build additionally asserts the 24-byte
//! size a C probe of the original reports.
//!
//! Everything a peer sends is untrusted: [`Command::decode`] rejects a short or
//! unknown command, [`buffer_range`] rejects an offset / size pair that does not
//! land inside the mapped area, and [`valid_area_name`] rejects a shm name that
//! is not a single path component.

use core::ffi::{c_int, c_uint, c_ulong, c_void};
use core::mem::{offset_of, size_of};
use core::ops::Range;
use core::sync::atomic::{AtomicUsize, Ordering};

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use std::ffi::CString;
use std::io::{Error as IoError, ErrorKind};

/// Server to client: a shared-memory area exists, here are its id, size and
/// name. The name follows the command as `path_size` further bytes.
pub const COMMAND_NEW_SHM_AREA: u32 = 1;
/// Server to client: this area is gone, unmap it.
pub const COMMAND_CLOSE_SHM_AREA: u32 = 2;
/// Server to client: a buffer occupies `offset .. offset + size` of the area.
pub const COMMAND_NEW_BUFFER: u32 = 3;
/// Client to server: done with the buffer at `offset`, its space can be reused.
pub const COMMAND_ACK_BUFFER: u32 = 4;

/// Longest shm area name accepted from a peer. A POSIX shm name is a filename
/// under `/dev/shm`, so `NAME_MAX` plus the leading slash bounds it.
pub const MAX_AREA_NAME_BYTES: usize = 256;

/// Largest area a peer may announce. GStreamer's `shm-size` is a `guint`
/// property, so no gst sink can ever name a larger one.
pub const MAX_AREA_BYTES: u64 = u32::MAX as u64;

/// The C `struct CommandBuffer` of `shmpipe.c`. Only its layout is used: the
/// codec below reads and writes the bytes at the offsets `offset_of!` reports,
/// which keeps the wire format tied to these declarations.
#[repr(C)]
#[derive(Clone, Copy)]
struct CommandBuffer {
    command_type: c_uint,
    area_id: c_int,
    payload: CommandPayload,
}

#[repr(C)]
#[derive(Clone, Copy)]
union CommandPayload {
    new_shm_area: NewShmAreaPayload,
    buffer: BufferPayload,
    ack_buffer: AckBufferPayload,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NewShmAreaPayload {
    size: usize,
    path_size: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BufferPayload {
    offset: c_ulong,
    size: c_ulong,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AckBufferPayload {
    offset: c_ulong,
}

/// Bytes in one command, the whole struct including its tail padding: the C
/// `send_command` writes `sizeof (struct CommandBuffer)` every time.
pub const COMMAND_BYTES: usize = size_of::<CommandBuffer>();

const TYPE_OFFSET: usize = offset_of!(CommandBuffer, command_type);
const AREA_ID_OFFSET: usize = offset_of!(CommandBuffer, area_id);
const PAYLOAD_OFFSET: usize = offset_of!(CommandBuffer, payload);
const NEW_AREA_SIZE_OFFSET: usize = PAYLOAD_OFFSET + offset_of!(NewShmAreaPayload, size);
const NEW_AREA_PATH_SIZE_OFFSET: usize = PAYLOAD_OFFSET + offset_of!(NewShmAreaPayload, path_size);
const BUFFER_OFFSET_OFFSET: usize = PAYLOAD_OFFSET + offset_of!(BufferPayload, offset);
const BUFFER_SIZE_OFFSET: usize = PAYLOAD_OFFSET + offset_of!(BufferPayload, size);
const ACK_OFFSET_OFFSET: usize = PAYLOAD_OFFSET + offset_of!(AckBufferPayload, offset);

/// Bytes in a `c_uint` / `c_int` field. The `u32` codec below only compiles
/// while this is 4, which is the assert that it matches the C type.
const UINT_BYTES: usize = size_of::<c_uint>();
/// Bytes in a `size_t` / `unsigned long` field.
const NATIVE_BYTES: usize = size_of::<usize>();

// size_t and unsigned long are the same width on every unix ABI this builds
// for, so one native reader covers both payload shapes.
const _: () = assert!(size_of::<usize>() == size_of::<c_ulong>());
// the layout a C probe of shmpipe.c's struct reports on a 64-bit ABI
#[cfg(target_pointer_width = "64")]
const _: () = assert!(COMMAND_BYTES == 24 && PAYLOAD_OFFSET == 8);

fn put_uint(bytes: &mut [u8; COMMAND_BYTES], offset: usize, value: u32) {
    bytes[offset..offset + UINT_BYTES].copy_from_slice(&value.to_ne_bytes());
}

fn put_native(bytes: &mut [u8; COMMAND_BYTES], offset: usize, value: u64) {
    bytes[offset..offset + NATIVE_BYTES].copy_from_slice(&(value as usize).to_ne_bytes());
}

fn get_uint(bytes: &[u8], offset: usize) -> u32 {
    let mut raw = [0u8; UINT_BYTES];
    raw.copy_from_slice(&bytes[offset..offset + UINT_BYTES]);
    u32::from_ne_bytes(raw)
}

fn get_native(bytes: &[u8], offset: usize) -> u64 {
    let mut raw = [0u8; NATIVE_BYTES];
    raw.copy_from_slice(&bytes[offset..offset + NATIVE_BYTES]);
    usize::from_ne_bytes(raw) as u64
}

/// One command off the control socket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// The name follows on the socket as `path_size` further bytes, NUL
    /// included.
    NewShmArea {
        area_id: i32,
        size: u64,
        path_size: u32,
    },
    CloseShmArea {
        area_id: i32,
    },
    NewBuffer {
        area_id: i32,
        offset: u64,
        size: u64,
    },
    AckBuffer {
        area_id: i32,
        offset: u64,
    },
}

impl Command {
    /// The bytes to put on the socket. The C side zeroes the whole struct
    /// before filling one union arm, so the unused tail is zero here too.
    pub fn encode(&self) -> [u8; COMMAND_BYTES] {
        let mut bytes = [0u8; COMMAND_BYTES];
        match *self {
            Command::NewShmArea {
                area_id,
                size,
                path_size,
            } => {
                put_uint(&mut bytes, TYPE_OFFSET, COMMAND_NEW_SHM_AREA);
                put_uint(&mut bytes, AREA_ID_OFFSET, area_id as u32);
                put_native(&mut bytes, NEW_AREA_SIZE_OFFSET, size);
                put_uint(&mut bytes, NEW_AREA_PATH_SIZE_OFFSET, path_size);
            }
            Command::CloseShmArea { area_id } => {
                put_uint(&mut bytes, TYPE_OFFSET, COMMAND_CLOSE_SHM_AREA);
                put_uint(&mut bytes, AREA_ID_OFFSET, area_id as u32);
            }
            Command::NewBuffer {
                area_id,
                offset,
                size,
            } => {
                put_uint(&mut bytes, TYPE_OFFSET, COMMAND_NEW_BUFFER);
                put_uint(&mut bytes, AREA_ID_OFFSET, area_id as u32);
                put_native(&mut bytes, BUFFER_OFFSET_OFFSET, offset);
                put_native(&mut bytes, BUFFER_SIZE_OFFSET, size);
            }
            Command::AckBuffer { area_id, offset } => {
                put_uint(&mut bytes, TYPE_OFFSET, COMMAND_ACK_BUFFER);
                put_uint(&mut bytes, AREA_ID_OFFSET, area_id as u32);
                put_native(&mut bytes, ACK_OFFSET_OFFSET, offset);
            }
        }
        bytes
    }

    /// `None` when `bytes` is not exactly one command or names a type the
    /// protocol does not define.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != COMMAND_BYTES {
            return None;
        }
        let area_id = get_uint(bytes, AREA_ID_OFFSET) as i32;
        match get_uint(bytes, TYPE_OFFSET) {
            COMMAND_NEW_SHM_AREA => Some(Command::NewShmArea {
                area_id,
                size: get_native(bytes, NEW_AREA_SIZE_OFFSET),
                path_size: get_uint(bytes, NEW_AREA_PATH_SIZE_OFFSET),
            }),
            COMMAND_CLOSE_SHM_AREA => Some(Command::CloseShmArea { area_id }),
            COMMAND_NEW_BUFFER => Some(Command::NewBuffer {
                area_id,
                offset: get_native(bytes, BUFFER_OFFSET_OFFSET),
                size: get_native(bytes, BUFFER_SIZE_OFFSET),
            }),
            COMMAND_ACK_BUFFER => Some(Command::AckBuffer {
                area_id,
                offset: get_native(bytes, ACK_OFFSET_OFFSET),
            }),
            _ => None,
        }
    }
}

/// The announced buffer as a range of the mapped area, or `None` when the pair
/// is empty, wraps, or reaches past `area_len`.
pub fn buffer_range(area_len: usize, offset: u64, size: u64) -> Option<Range<usize>> {
    if size == 0 {
        return None;
    }
    let end = offset.checked_add(size)?;
    if end > area_len as u64 {
        return None;
    }
    Some(offset as usize..end as usize)
}

/// Whether `name` is a shm name safe to hand `shm_open`: one leading slash and
/// a single path component after it, so a peer cannot walk out of `/dev/shm`.
pub fn valid_area_name(name: &str) -> bool {
    let Some(component) = name.strip_prefix('/') else {
        return false;
    };
    !component.is_empty()
        && name.len() <= MAX_AREA_NAME_BYTES
        && !component.contains('/')
        && !component.contains('\0')
        && component != "."
        && component != ".."
}

/// Serial for the generated area names, so two sinks in one process never
/// collide on the first try.
static NEXT_AREA_SERIAL: AtomicUsize = AtomicUsize::new(0);

/// The name template `sp_open_shm` generates, `%5d` widths included, so a gst
/// reader sees the names it would see from a gst writer.
fn generated_area_name(serial: usize) -> String {
    // SAFETY: getpid takes no arguments and only reads the calling process id.
    let pid = unsafe { libc::getpid() };
    format!("/shmpipe.{pid:5}.{serial:5}")
}

/// A POSIX shared-memory area mapped into this process. The writer creates and
/// unlinks it; a reader opens the announced name read-only.
#[derive(Debug)]
pub struct MappedArea {
    id: i32,
    name: String,
    fd: c_int,
    address: *mut c_void,
    len: usize,
    writer: bool,
}

// SAFETY: the area owns its mapping and its descriptor exclusively, and neither
// is bound to the thread that made them, so moving one between threads leaves
// every access still going through this owner.
unsafe impl Send for MappedArea {}

impl MappedArea {
    /// Create a fresh area of `len` bytes with mode `perms`, retrying the name
    /// while one is taken, and map it read-write.
    pub fn create(id: i32, len: usize, perms: u32) -> Result<Self, IoError> {
        if len == 0 {
            return Err(IoError::new(ErrorKind::InvalidInput, "shm-size is zero"));
        }
        let flags = libc::O_RDWR | libc::O_CREAT | libc::O_TRUNC | libc::O_EXCL;
        let (name, fd) = loop {
            let serial = NEXT_AREA_SERIAL.fetch_add(1, Ordering::Relaxed);
            let name = generated_area_name(serial);
            let c_name = CString::new(name.as_str())
                .map_err(|_| IoError::new(ErrorKind::InvalidInput, "shm name holds a NUL"))?;
            // SAFETY: c_name is a NUL-terminated C string valid for the call,
            // and the flags are the create flags shm_open documents.
            let fd = unsafe { libc::shm_open(c_name.as_ptr(), flags, perms as libc::mode_t) };
            if fd >= 0 {
                break (name, fd);
            }
            let error = IoError::last_os_error();
            if error.raw_os_error() != Some(libc::EEXIST) {
                return Err(error);
            }
        };

        let mut area = Self {
            id,
            name,
            fd,
            address: libc::MAP_FAILED,
            len,
            writer: true,
        };
        // shm_open's mode is masked by the umask, so fchmod is what actually
        // applies `perms`.
        // SAFETY: fd is the descriptor just returned by shm_open.
        if unsafe { libc::fchmod(fd, perms as libc::mode_t) } < 0 {
            return Err(IoError::last_os_error());
        }
        // SAFETY: fd is a fresh shm descriptor and len is the size just checked
        // to be nonzero.
        if unsafe { libc::ftruncate(fd, len as libc::off_t) } < 0 {
            return Err(IoError::last_os_error());
        }
        area.map(libc::PROT_READ | libc::PROT_WRITE)?;
        Ok(area)
    }

    /// Open the area a peer announced, read-only. `name` is peer-supplied, so it
    /// is checked against [`valid_area_name`] before it reaches `shm_open`.
    pub fn open_readonly(id: i32, name: &str, len: usize) -> Result<Self, IoError> {
        if !valid_area_name(name) {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "peer announced an unusable shm area name",
            ));
        }
        if len == 0 || len as u64 > MAX_AREA_BYTES {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "peer announced an out of range shm area size",
            ));
        }
        let c_name = CString::new(name)
            .map_err(|_| IoError::new(ErrorKind::InvalidInput, "shm name holds a NUL"))?;
        // SAFETY: c_name is a NUL-terminated C string valid for the call; a mode
        // is ignored without O_CREAT.
        let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_RDONLY, 0) };
        if fd < 0 {
            return Err(IoError::last_os_error());
        }
        let mut area = Self {
            id,
            name: String::from(name),
            fd,
            address: libc::MAP_FAILED,
            len,
            writer: false,
        };
        // A peer that announces more than it created would leave the tail of the
        // mapping unbacked, and touching it raises SIGBUS.
        if area.file_size()? < len as u64 {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "shm area is smaller than the announced size",
            ));
        }
        area.map(libc::PROT_READ)?;
        Ok(area)
    }

    fn file_size(&self) -> Result<u64, IoError> {
        // SAFETY: stat is a plain out parameter the call fills, and self.fd is
        // the open shm descriptor.
        let mut stat = unsafe { core::mem::zeroed::<libc::stat>() };
        // SAFETY: fd is open and stat is a valid, writable libc::stat.
        if unsafe { libc::fstat(self.fd, &mut stat) } < 0 {
            return Err(IoError::last_os_error());
        }
        Ok(stat.st_size.max(0) as u64)
    }

    fn map(&mut self, protection: c_int) -> Result<(), IoError> {
        // SAFETY: a null hint lets the kernel place the mapping, self.fd is an
        // open shm descriptor sized to self.len by create (or announced as that
        // size by the peer, in which case a short file makes the later access
        // fault rather than read out of the mapping).
        let address = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                self.len,
                protection,
                libc::MAP_SHARED,
                self.fd,
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(IoError::last_os_error());
        }
        self.address = address;
        Ok(())
    }

    pub fn id(&self) -> i32 {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The bytes of `range`, which the caller took from [`buffer_range`] so it
    /// is inside the mapping.
    pub fn read(&self, range: Range<usize>) -> &[u8] {
        debug_assert!(range.end <= self.len);
        // SAFETY: the mapping is len bytes of readable memory for as long as
        // self lives, and range was bounded against len.
        let all = unsafe { core::slice::from_raw_parts(self.address as *const u8, self.len) };
        &all[range]
    }

    /// Copy `bytes` to `offset`. Only a writer area can take this, and the
    /// destination is checked against the mapping.
    pub fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), IoError> {
        if !self.writer || buffer_range(self.len, offset as u64, bytes.len() as u64).is_none() {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "write lands outside the shm area",
            ));
        }
        // SAFETY: the mapping is len bytes of writable memory (writer areas map
        // PROT_WRITE) for as long as self lives, and the destination was just
        // bounded against len.
        let all = unsafe { core::slice::from_raw_parts_mut(self.address as *mut u8, self.len) };
        all[offset..offset + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
}

impl Drop for MappedArea {
    fn drop(&mut self) {
        if self.address != libc::MAP_FAILED {
            // SAFETY: address and len are the pair mmap returned and nothing
            // else holds a reference into the mapping at drop.
            unsafe { libc::munmap(self.address, self.len) };
        }
        if self.fd >= 0 {
            // SAFETY: fd came from shm_open and is closed once.
            unsafe { libc::close(self.fd) };
        }
        if self.writer {
            if let Ok(c_name) = CString::new(self.name.as_str()) {
                // SAFETY: c_name is a NUL-terminated C string valid for the
                // call; only the creator unlinks.
                unsafe { libc::shm_unlink(c_name.as_ptr()) };
            }
        }
    }
}

/// A block handed out by [`AreaAllocator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AllocatedBlock {
    offset: usize,
    size: usize,
}

/// First-fit allocator over a shm area, the shape gst's `shmalloc.c` uses: live
/// blocks in offset order, a new block taking the first gap it fits in.
#[derive(Debug, Default)]
pub struct AreaAllocator {
    len: usize,
    blocks: Vec<AllocatedBlock>,
}

impl AreaAllocator {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            blocks: Vec::new(),
        }
    }

    /// The offset of a fresh `size`-byte block, or `None` when no gap fits.
    pub fn alloc(&mut self, size: usize) -> Option<usize> {
        if size == 0 || size > self.len {
            return None;
        }
        let mut gap_start = 0usize;
        for (index, block) in self.blocks.iter().enumerate() {
            if block.offset - gap_start >= size {
                self.blocks.insert(
                    index,
                    AllocatedBlock {
                        offset: gap_start,
                        size,
                    },
                );
                return Some(gap_start);
            }
            gap_start = block.offset + block.size;
        }
        if self.len - gap_start < size {
            return None;
        }
        self.blocks.push(AllocatedBlock {
            offset: gap_start,
            size,
        });
        Some(gap_start)
    }

    /// Release the block starting at `offset`. Unknown offsets are ignored, the
    /// way a double free of an already-reused block has to be.
    pub fn free(&mut self, offset: usize) {
        if let Some(index) = self.blocks.iter().position(|b| b.offset == offset) {
            self.blocks.remove(index);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_round_trip() {
        for command in [
            Command::NewShmArea {
                area_id: 1,
                size: 4096,
                path_size: 12,
            },
            Command::CloseShmArea { area_id: 7 },
            Command::NewBuffer {
                area_id: 2,
                offset: 64,
                size: 1024,
            },
            Command::AckBuffer {
                area_id: 2,
                offset: 64,
            },
        ] {
            assert_eq!(Command::decode(&command.encode()), Some(command));
        }
    }

    #[test]
    fn unknown_type_and_short_command_are_rejected() {
        let mut bytes = Command::CloseShmArea { area_id: 1 }.encode();
        put_uint(&mut bytes, TYPE_OFFSET, 99);
        assert_eq!(Command::decode(&bytes), None);
        assert_eq!(Command::decode(&bytes[..COMMAND_BYTES - 1]), None);
    }

    #[test]
    fn allocator_reuses_a_freed_gap() {
        let mut allocator = AreaAllocator::new(100);
        let first = allocator.alloc(40).expect("the first block fits");
        let second = allocator.alloc(40).expect("the second block fits");
        assert_eq!((first, second), (0, 40));
        assert_eq!(allocator.alloc(40), None, "only 20 bytes are left");
        allocator.free(first);
        assert_eq!(allocator.alloc(40), Some(0), "the freed gap is reused");
    }
}
