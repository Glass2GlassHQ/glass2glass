# Build the glass2glass C example against the g2g-capi static library.
#
#   make run     # build the capi staticlib + this program, then run it
#   make         # build only
#   make clean   # remove the built binary
#
# The Rust static lib is produced by `cargo build -p g2g-capi` at the workspace
# root; we link it plus the system libraries a Rust staticlib needs on Linux.

ROOT      := ../..
HEADER    := $(ROOT)/g2g-capi/include
LIBDIR    := $(ROOT)/target/release
STATICLIB := $(LIBDIR)/libg2g_capi.a

CC      ?= cc
CFLAGS  += -std=c11 -Wall -Wextra -I$(HEADER)
# A Rust staticlib pulls in pthread / dl / m and the unwinder on Linux.
LDLIBS  += -lpthread -ldl -lm

g2g-c-example: main.c cargo-lib
	$(CC) $(CFLAGS) -o $@ main.c $(STATICLIB) $(LDLIBS)

# Always let cargo decide whether a rebuild is needed (it tracks the Rust
# sources; a plain file rule would miss edits to g2g-capi).
cargo-lib:
	cargo build --release -p g2g-capi --manifest-path $(ROOT)/Cargo.toml

.PHONY: run clean cargo-lib
run: g2g-c-example
	./g2g-c-example

clean:
	rm -f g2g-c-example
