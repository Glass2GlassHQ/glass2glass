# Zephyr application consuming the g2g Zephyr module (M647, promoting the M637
# proof). Note what is NOT here: no staticlib path, no include directory, no
# target_link_libraries for g2g. The g2g module (discovered via
# ZEPHYR_EXTRA_MODULES / the west manifest) provides `<g2g.h>` and links the
# prebuilt library into the image. This is the consumable-packaging path,
# replacing the old hand-linked -DG2G_NOALLOC_LIB. tools/zephyr-check.sh builds
# the Rust staticlib and points Zephyr at both the module and the archive.
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(g2g_zephyr)

target_sources(app PRIVATE src/main.c)
