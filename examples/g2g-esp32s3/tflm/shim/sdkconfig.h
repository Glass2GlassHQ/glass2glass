// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

// esp-nn reads these ESP-IDF Kconfig values to pick its ESP32-S3 kernels.
#define CONFIG_IDF_TARGET_ESP32S3 1
#define CONFIG_IDF_TARGET_ARCH_XTENSA 1
#define CONFIG_NN_OPTIMIZED 1
#define CONFIG_FREERTOS_UNICORE 1
#define CONFIG_ESP32S3_DATA_CACHE_SIZE 32768
