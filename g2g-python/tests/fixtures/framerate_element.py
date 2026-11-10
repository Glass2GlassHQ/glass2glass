# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

FRAMERATES_SEEN = []


class FramerateRecorder:
    def g2g_process(self, buf, width, height, fmt, meta):
        FRAMERATES_SEEN.append(meta.framerate)
