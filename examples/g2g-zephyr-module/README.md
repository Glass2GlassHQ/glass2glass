# g2g Zephyr module

A reusable [Zephyr module](https://docs.zephyrproject.org/latest/develop/modules.html)
that wires the g2g static pipeline library and its C header into a Zephyr
application, so a C shop integrates g2g without touching Cargo or hand-passing a
staticlib path.

## Use it

1. **Build the prebuilt staticlib** for your board's ISA (Zephyr's CMake does
   not run cargo). For the `qemu_cortex_m3` board (thumbv7m, soft float):

   ```sh
   cargo build --manifest-path examples/g2g-noalloc/Cargo.toml \
       --release --target thumbv7m-none-eabi
   # -> examples/g2g-noalloc/target/thumbv7m-none-eabi/release/libg2g_noalloc.a
   ```

   Use `thumbv7em-none-eabihf` for a Cortex-M4F board, etc. The archive is pure
   `no_std` Rust with no allocator and no panic machinery (proven by
   `tools/noalloc-check.sh`).

2. **Add the module** to your build. With west, list it in your manifest;
   without west, pass it directly:

   ```sh
   west build -b <board> app -- \
       -DZEPHYR_EXTRA_MODULES=/path/to/examples/g2g-zephyr-module \
       -DG2G_STATICLIB=/path/to/libg2g_noalloc.a
   ```

3. **Call it** from your application. No CMake wiring, no `extern`
   declarations, no include paths:

   ```c
   #include <g2g.h>

   void run(void) {
       if (g2g_audio_run() == g2g_audio_expected()) {
           /* the flagship audio graph ran on-target */
       }
   }
   ```

## What the module does

`zephyr/module.yml` declares the module; `CMakeLists.txt` imports the archive
named by `G2G_STATICLIB`, adds `include/` to the Zephyr include path
(`zephyr_include_directories`), and links the archive into the image
(`zephyr_link_libraries`). The application's calls to `g2g_noalloc_run()` /
`g2g_audio_run()` resolve with no per-app link directives.

## Proof

`tools/zephyr-check.sh` builds the staticlib, configures the `examples/g2g-zephyr`
application against this module (via `ZEPHYR_EXTRA_MODULES`), builds it, and
boots it on QEMU's lm3s6965evb Cortex-M3, asserting the on-target checksum
banner. It runs in CI.
