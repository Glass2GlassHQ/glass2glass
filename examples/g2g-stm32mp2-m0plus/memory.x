/* the board's device tree reserves these for the M0+ (cm0-cube-fw, cm0-cube-data),
 * the same split ST's CM0PLUS_DEMO links against */
MEMORY
{
  FLASH : ORIGIN = 0x200C0000, LENGTH = 16K
  RAM   : ORIGIN = 0x200C4000, LENGTH = 8K
}
