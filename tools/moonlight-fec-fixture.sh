#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# moonlight decodes video parity with nanors, so its encoder writes the reference parity
set -euo pipefail

nanors_commit=a9b1636
output="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/src/moonlight/testdata/nanors-parity.bin"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

git clone -q https://github.com/sleepybishop/nanors.git "$work/nanors"
git -C "$work/nanors" checkout -q "$nanors_commit"
cat > "$work/generate.c" <<'C'
#include <stdio.h>
#include <stdlib.h>
#include "rs.h"

#define SHARD_BYTES 24

int main(void)
{
    static const int layouts[][2] = {{1, 1}, {4, 2}, {10, 3}, {32, 7}, {200, 40}, {253, 2}};
    reed_solomon_init();
    for (unsigned l = 0; l < sizeof layouts / sizeof layouts[0]; l++) {
        int data = layouts[l][0], parity = layouts[l][1];
        reed_solomon *rs = reed_solomon_new(data, parity);
        uint8_t **shards = malloc(sizeof(uint8_t *) * (data + parity));
        for (int i = 0; i < data + parity; i++) {
            shards[i] = calloc(1, SHARD_BYTES);
            for (int j = 0; i < data && j < SHARD_BYTES; j++)
                shards[i][j] = (uint8_t)(i * 37 + j * 11 + 5);
        }
        if (!rs || reed_solomon_encode(rs, shards, data + parity, SHARD_BYTES))
            return 1;
        for (int i = data; i < data + parity; i++)
            fwrite(shards[i], 1, SHARD_BYTES, stdout);
        reed_solomon_release(rs);
    }
    return 0;
}
C
n="$work/nanors"
cc -O2 -std=c11 -I"$n" -I"$n/deps/obl" "$work/generate.c" "$n/rs.c" \
    "$n/deps/obl/oblas_common.c" "$n/deps/obl/oblas_lite.c" -o "$work/generate"
"$work/generate" > "$output"
