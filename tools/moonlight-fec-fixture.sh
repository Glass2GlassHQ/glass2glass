#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# moonlight decodes video and audio parity with nanors, so its encoder writes the reference parity
set -euo pipefail

nanors_commit=a9b1636
testdata="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/src/moonlight/testdata"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

git clone -q https://github.com/sleepybishop/nanors.git "$work/nanors"
git -C "$work/nanors" checkout -q "$nanors_commit"
cat > "$work/generate.c" <<'C'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rs.h"

#define SHARD_BYTES 24

/* moonlight's audio queue copies this matrix over the one nanors builds for 4 data and 2 parity shards */
static const uint8_t audio_matrix[] = {0x77, 0x40, 0x38, 0x0e, 0xc7, 0xa7, 0x0d, 0x6c};

static int encode(reed_solomon *rs, int data, int parity, FILE *out)
{
    uint8_t **shards = malloc(sizeof(uint8_t *) * (data + parity));
    for (int i = 0; i < data + parity; i++) {
        shards[i] = calloc(1, SHARD_BYTES);
        for (int j = 0; i < data && j < SHARD_BYTES; j++)
            shards[i][j] = (uint8_t)(i * 37 + j * 11 + 5);
    }
    if (reed_solomon_encode(rs, shards, data + parity, SHARD_BYTES))
        return 1;
    for (int i = data; i < data + parity; i++)
        fwrite(shards[i], 1, SHARD_BYTES, out);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 3)
        return 1;
    FILE *video = fopen(argv[1], "wb"), *audio = fopen(argv[2], "wb");
    if (!video || !audio)
        return 1;
    static const int layouts[][2] = {{1, 1}, {4, 2}, {10, 3}, {32, 7}, {200, 40}, {253, 2}};
    reed_solomon_init();
    for (unsigned l = 0; l < sizeof layouts / sizeof layouts[0]; l++) {
        reed_solomon *rs = reed_solomon_new(layouts[l][0], layouts[l][1]);
        if (!rs || encode(rs, layouts[l][0], layouts[l][1], video))
            return 1;
        reed_solomon_release(rs);
    }
    reed_solomon *rs = reed_solomon_new(4, 2);
    if (!rs)
        return 1;
    memcpy(rs->p, audio_matrix, sizeof audio_matrix);
    if (encode(rs, 4, 2, audio))
        return 1;
    reed_solomon_release(rs);
    fclose(video);
    fclose(audio);
    return 0;
}
C
n="$work/nanors"
cc -O2 -std=c11 -I"$n" -I"$n/deps/obl" "$work/generate.c" "$n/rs.c" \
    "$n/deps/obl/oblas_common.c" "$n/deps/obl/oblas_lite.c" -o "$work/generate"
"$work/generate" "$testdata/nanors-parity.bin" "$testdata/nanors-audio-parity.bin"
