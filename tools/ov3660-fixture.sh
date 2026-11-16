#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# usage: tools/ov3660-fixture.sh, records the SCCB transactions esp32-camera's OV3660 driver makes
# for 96x96 grayscale into g2g-mcu/tests/fixtures/ov3660/grayscale_96x96.txt
set -euo pipefail

ESP32_CAMERA_COMMIT=b3f7fe802f41f17fd7a104d517c53f1f62c8414d
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/g2g-mcu/tests/fixtures/ov3660/grayscale_96x96.txt"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git -C "$WORK" init -q esp32-camera
git -C "$WORK/esp32-camera" fetch -q --depth 1 https://github.com/espressif/esp32-camera.git "$ESP32_CAMERA_COMMIT"
git -C "$WORK/esp32-camera" checkout -q FETCH_HEAD
CAMERA="$WORK/esp32-camera"

mkdir -p "$WORK/stub/freertos"
printf '#pragma once\n#define DRAM_ATTR\n' > "$WORK/stub/esp_attr.h"
printf '#pragma once\n#define ESP_LOGI(...)\n#define ESP_LOGD(...)\n#define ESP_LOGE(...)\n#define ESP_LOGW(...)\n' > "$WORK/stub/esp_log.h"
printf '#pragma once\n#define portTICK_PERIOD_MS 1\n' > "$WORK/stub/freertos/FreeRTOS.h"
printf '#pragma once\nstatic inline void vTaskDelay(int ticks) { (void)ticks; }\n' > "$WORK/stub/freertos/task.h"
printf '#pragma once\nint xclk_timer_conf(int ledc_timer, int xclk_freq_hz);\n' > "$WORK/stub/xclk.h"

cat > "$WORK/harness.c" <<'HARNESS'
#include <stdio.h>
#include <stdint.h>
#include "sensor.h"

int esp32_camera_ov3660_detect(int slv_addr, sensor_id_t *id);
int esp32_camera_ov3660_init(sensor_t *sensor);

// the set_xclk hook, which the 96x96 setup never calls
int xclk_timer_conf(int ledc_timer, int xclk_freq_hz) { return 0; }

// one flat register file, so read-modify-write sequences read back what was written
static uint8_t registers[0x10000];

int SCCB_Write16(uint8_t slv_addr, uint16_t reg, uint8_t data) {
    printf("write 0x%04x 0x%02x\n", reg, data);
    registers[reg] = data;
    return 0;
}

uint8_t SCCB_Read16(uint8_t slv_addr, uint16_t reg) {
    printf("read 0x%04x\n", reg);
    return registers[reg];
}

int main(void) {
    sensor_t sensor = {0};
    sensor.slv_addr = 0x3c;
    sensor.xclk_freq_hz = 20000000;
    registers[0x300a] = OV3660_PID >> 8;
    registers[0x300b] = OV3660_PID & 0xff;
    if (!esp32_camera_ov3660_detect(sensor.slv_addr, &sensor.id)) return 1;
    esp32_camera_ov3660_init(&sensor);
    sensor.reset(&sensor);
    sensor.status.framesize = FRAMESIZE_96X96;
    sensor.pixformat = PIXFORMAT_GRAYSCALE;
    sensor.set_framesize(&sensor, FRAMESIZE_96X96);
    sensor.set_pixformat(&sensor, PIXFORMAT_GRAYSCALE);
    return 0;
}
HARNESS

cc -w -I"$WORK/stub" -I"$CAMERA/driver/include" -I"$CAMERA/driver/private_include" \
   -I"$CAMERA/sensors/private_include" \
   "$WORK/harness.c" "$CAMERA/sensors/ov3660.c" "$CAMERA/driver/sensor.c" -o "$WORK/harness"
mkdir -p "$(dirname "$OUT")"
"$WORK/harness" > "$OUT"
echo "wrote $(wc -l < "$OUT") transactions to $OUT"
