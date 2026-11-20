import mmap, os, struct, sys
SHARED_REPORT_ADDRESS = 0x200C6000
fd = os.open("/dev/mem", os.O_RDONLY | os.O_SYNC)
page = mmap.mmap(fd, mmap.PAGESIZE, mmap.MAP_SHARED, mmap.PROT_READ, offset=SHARED_REPORT_ADDRESS)
magic, runs, wrong, low, high = struct.unpack_from("<5I", page, 0)
print(f"magic={magic.to_bytes(4, 'little')!r} runs={runs} wrong_checksums={wrong} last_checksum={(high << 32) | low}")
