# The g2g Zephyr module's build fragment: Zephyr includes this (because the
# module is on ZEPHYR_EXTRA_MODULES / in the west manifest), and it exposes the
# prebuilt g2g static pipeline library plus its header to the application. An
# app then just `#include <g2g.h>` and calls the entry points; it never names
# the staticlib or an include path. This is the difference from the M637
# hand-linked app: the packaging does the wiring.
#
# The Rust staticlib is built out-of-band (Zephyr's CMake cannot run cargo);
# its path arrives as G2G_STATICLIB. See README.md for the recipe.

if(NOT DEFINED G2G_STATICLIB)
  message(FATAL_ERROR
    "g2g module: set -DG2G_STATICLIB=/path/to/libg2g_noalloc.a "
    "(build it with: cargo build --manifest-path examples/g2g-noalloc/Cargo.toml "
    "--release --target <board-triple>; see this module's README).")
endif()
if(NOT EXISTS "${G2G_STATICLIB}")
  message(FATAL_ERROR "g2g module: G2G_STATICLIB does not exist: ${G2G_STATICLIB}")
endif()

# Import the prebuilt archive as a linkable target and add it (plus the public
# header directory) to the Zephyr image. `zephyr_link_libraries` folds it into
# the final link, so the app's calls to g2g_noalloc_run() / g2g_audio_run()
# resolve without the app declaring anything.
add_library(g2g_imported STATIC IMPORTED GLOBAL)
set_target_properties(g2g_imported PROPERTIES IMPORTED_LOCATION "${G2G_STATICLIB}")

zephyr_include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
zephyr_link_libraries(g2g_imported)
