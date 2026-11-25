// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

extern "C" __global__ void scale_plane(
    const unsigned char* source,
    unsigned int source_pitch,
    unsigned int source_width,
    unsigned int source_height,
    unsigned char* destination,
    unsigned int destination_pitch,
    unsigned int destination_width,
    unsigned int destination_height,
    unsigned int channels)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= destination_width || y >= destination_height) {
        return;
    }
    float source_x = ((float)x + 0.5f) * (float)source_width / (float)destination_width - 0.5f;
    float source_y = ((float)y + 0.5f) * (float)source_height / (float)destination_height - 0.5f;
    source_x = fminf(fmaxf(source_x, 0.0f), (float)(source_width - 1));
    source_y = fminf(fmaxf(source_y, 0.0f), (float)(source_height - 1));
    unsigned int left = (unsigned int)source_x;
    unsigned int top = (unsigned int)source_y;
    unsigned int right = min(left + 1, source_width - 1);
    unsigned int bottom = min(top + 1, source_height - 1);
    float across = source_x - (float)left;
    float down = source_y - (float)top;
    const unsigned char* upper = source + (unsigned long long)top * source_pitch;
    const unsigned char* lower = source + (unsigned long long)bottom * source_pitch;
    unsigned char* output = destination + (unsigned long long)y * destination_pitch + x * channels;
    for (unsigned int channel = 0; channel < channels; channel++) {
        float upper_value = upper[left * channels + channel] * (1.0f - across)
            + upper[right * channels + channel] * across;
        float lower_value = lower[left * channels + channel] * (1.0f - across)
            + lower[right * channels + channel] * across;
        output[channel] = (unsigned char)rintf(upper_value * (1.0f - down) + lower_value * down);
    }
}
