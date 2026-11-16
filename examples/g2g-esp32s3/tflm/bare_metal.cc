// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#include <cstddef>

extern "C" void tflm_log(const char* text);

// esp-nn's error paths call puts.
extern "C" int puts(const char* text) {
  tflm_log(text);
  return 0;
}

// virtual destructors reference operator delete, which nothing calls without a heap.
void operator delete(void*, size_t) noexcept { __builtin_trap(); }
void operator delete(void*) noexcept { __builtin_trap(); }
