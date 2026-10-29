ROOT      := ../..
HEADER    := $(ROOT)/g2g-capi/include
CARGO_TARGET_DIR ?= $(ROOT)/target
STATICLIB := $(CARGO_TARGET_DIR)/release/libg2g_capi.a

CXX      ?= c++
CXXFLAGS += -std=c++17 -Wall -Wextra -Wpedantic -I$(HEADER)
LDLIBS   += -lpthread -ldl -lm

g2g-cpp-example: main.cpp $(HEADER)/g2g.hpp $(HEADER)/g2g.h cargo-lib
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(STATICLIB) $(LDLIBS)

# cargo tracks the Rust sources, a file rule would link a stale library
cargo-lib:
	cargo build --release -p g2g-capi --manifest-path $(ROOT)/Cargo.toml

.PHONY: run clean cargo-lib
run: g2g-cpp-example
	./g2g-cpp-example

clean:
	rm -f g2g-cpp-example
