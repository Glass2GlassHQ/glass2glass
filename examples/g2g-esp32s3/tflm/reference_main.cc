// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void* tflm_load(const uint8_t* model, uint8_t* arena, size_t arena_bytes);
extern "C" int tflm_infer(void* model, const uint8_t* input, size_t input_bytes, uint8_t* output,
                          size_t output_bytes);
extern const unsigned char g_person_detect_model_data[];

extern "C" void tflm_log(const char* text) { std::fprintf(stderr, "%s\n", text); }

alignas(16) static uint8_t arena[256 * 1024];

int main(int argc, char** argv) {
  void* model = tflm_load(g_person_detect_model_data, arena, sizeof(arena));
  if (model == nullptr) return 1;
  for (int index = 1; index < argc; ++index) {
    std::FILE* file = std::fopen(argv[index], "rb");
    if (file == nullptr) return 1;
    std::vector<uint8_t> image(96 * 96);
    if (std::fread(image.data(), 1, image.size(), file) != image.size()) return 1;
    std::fclose(file);
    for (uint8_t& pixel : image) pixel = static_cast<uint8_t>(pixel - 128);
    int8_t scores[2];
    if (tflm_infer(model, image.data(), image.size(), reinterpret_cast<uint8_t*>(scores), 2) != 0) return 1;
    std::printf("[%d, %d],\n", scores[0], scores[1]);
  }
  return 0;
}
