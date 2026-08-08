# g2g-mcu

MCU peripheral elements for
[glass2glass](https://github.com/boxerab/glass2glass) over `embedded-hal` 1.0
trait seams: SPI display sink, frame grabber, and PCM sink. `no_std` with no
`alloc`, so it links on targets that have no allocator, and the element logic is
host-testable against mock peripherals.
