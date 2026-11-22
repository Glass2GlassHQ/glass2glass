// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::process::Command;

pub(crate) fn mjpeg_source(ffmpeg: &mut Command) {
    ffmpeg.args([
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=64x48:rate=25",
        "-c:v",
        "mjpeg",
        "-pix_fmt",
        "yuvj420p",
        "-huffman",
        "default",
        "-force_duplicated_matrix",
        "1",
        "-threads:v",
        "1",
    ]);
}
