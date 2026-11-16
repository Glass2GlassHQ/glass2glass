// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

#ifdef ESP_NN
#include <esp_nn.h>
// esp-nn falls back to its portable C kernels without an error when the shim's sdkconfig.h is not in effect.
#ifndef ARCH_ESP32_S3
#error "esp-nn did not select its ESP32-S3 kernels"
#endif
#endif

extern "C" void tflm_log(const char* text);

extern "C" void DebugLog(const char* format, va_list) { tflm_log(format); }

extern "C" int DebugVsnprintf(char* buffer, size_t buffer_size, const char*, va_list) {
  if (buffer_size > 0) buffer[0] = 0;
  return 0;
}

namespace {
constexpr int kMaxModels = 2;
using Resolver = tflite::MicroMutableOpResolver<6>;
alignas(Resolver) uint8_t resolver_storage[sizeof(Resolver)];
Resolver* resolver = nullptr;
alignas(tflite::MicroInterpreter) uint8_t interpreter_storage[kMaxModels][sizeof(tflite::MicroInterpreter)];
int models_loaded = 0;
}  // namespace

extern "C" void* tflm_load(const uint8_t* model, uint8_t* arena, size_t arena_bytes) {
  if (models_loaded == kMaxModels) return nullptr;
  if (resolver == nullptr) {
    resolver = new (resolver_storage) Resolver();
    resolver->AddAveragePool2D();
    resolver->AddConv2D();
    resolver->AddDepthwiseConv2D();
    resolver->AddFullyConnected();
    resolver->AddReshape();
    resolver->AddSoftmax();
  }
  auto* interpreter = new (interpreter_storage[models_loaded])
      tflite::MicroInterpreter(tflite::GetModel(model), *resolver, arena, arena_bytes);
  if (interpreter->AllocateTensors() != kTfLiteOk) return nullptr;
  ++models_loaded;
  return interpreter;
}

extern "C" size_t tflm_arena_used(void* model) {
  return static_cast<tflite::MicroInterpreter*>(model)->arena_used_bytes();
}

extern "C" int tflm_infer(void* model, const uint8_t* input, size_t input_bytes, uint8_t* output,
                          size_t output_bytes) {
  auto* interpreter = static_cast<tflite::MicroInterpreter*>(model);
  TfLiteTensor* input_tensor = interpreter->input(0);
  TfLiteTensor* output_tensor = interpreter->output(0);
  if (input_tensor->bytes != input_bytes || output_tensor->bytes != output_bytes) return -2;
  std::memcpy(input_tensor->data.raw, input, input_bytes);
  if (interpreter->Invoke() != kTfLiteOk) return -1;
  std::memcpy(output, output_tensor->data.raw, output_bytes);
  return 0;
}
