/* QEMU MPS2-AN386 (Cortex-M4): code in ZBT SSRAM1 at 0x00000000, data in the
 * SSRAM2/3 block at 0x20000000. Sizes are conservative slices of the board's
 * 4 MB banks; the pipeline needs ~2 KB ROM and ~1.3 KB stack. */
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 512K
  RAM   : ORIGIN = 0x20000000, LENGTH = 512K
}
