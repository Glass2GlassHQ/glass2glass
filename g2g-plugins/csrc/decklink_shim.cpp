#include <cstdint>
#include <cstring>
#include <new>

#include "DeckLinkAPI.h"

// defined in DeckLinkAPIDispatch.cpp without a declaration in any header
bool IsDeckLinkAPIPresent(void);

extern "C" {

enum {
    G2G_DECKLINK_OPENED = 0,
    G2G_DECKLINK_LIBRARY_MISSING = 1,
    G2G_DECKLINK_NO_DEVICE = 2,
    G2G_DECKLINK_NO_OUTPUT = 3,
    G2G_DECKLINK_NO_ITERATOR = 4,
};

struct g2g_decklink_output {
    IDeckLink* device;
    IDeckLinkOutput* output;
};

int32_t g2g_decklink_open(uint32_t device_number, g2g_decklink_output** opened)
{
    IDeckLinkIterator* iterator = CreateDeckLinkIteratorInstance();
    if (iterator == nullptr)
        return IsDeckLinkAPIPresent() ? G2G_DECKLINK_NO_ITERATOR : G2G_DECKLINK_LIBRARY_MISSING;

    IDeckLink* device = nullptr;
    for (uint32_t index = 0; iterator->Next(&device) == S_OK; index++) {
        if (index == device_number)
            break;
        device->Release();
        device = nullptr;
    }
    iterator->Release();
    if (device == nullptr)
        return G2G_DECKLINK_NO_DEVICE;

    IDeckLinkOutput* output = nullptr;
    if (device->QueryInterface(IID_IDeckLinkOutput, (void**)&output) != S_OK) {
        device->Release();
        return G2G_DECKLINK_NO_OUTPUT;
    }
    g2g_decklink_output* handle = new (std::nothrow) g2g_decklink_output{device, output};
    if (handle == nullptr) {
        output->Release();
        device->Release();
        return E_OUTOFMEMORY;
    }
    *opened = handle;
    return G2G_DECKLINK_OPENED;
}

void g2g_decklink_close(g2g_decklink_output* handle)
{
    bool running = false;
    if (handle->output->IsScheduledPlaybackRunning(&running) == S_OK && running)
        handle->output->StopScheduledPlayback(0, nullptr, 0);
    handle->output->DisableAudioOutput();
    handle->output->DisableVideoOutput();
    handle->output->Release();
    handle->device->Release();
    delete handle;
}

int32_t g2g_decklink_enable_video(g2g_decklink_output* handle, uint32_t display_mode)
{
    return handle->output->EnableVideoOutput(display_mode, bmdVideoOutputFlagDefault);
}

int32_t g2g_decklink_schedule_video(g2g_decklink_output* handle, const uint8_t* pixels,
    int32_t width, int32_t height, int32_t row_bytes, uint32_t pixel_format,
    int64_t display_time, int64_t duration, int64_t time_scale)
{
    IDeckLinkMutableVideoFrame* frame = nullptr;
    HRESULT result = handle->output->CreateVideoFrame(width, height, row_bytes, pixel_format,
        bmdFrameFlagDefault, &frame);
    if (result != S_OK)
        return result;

    IDeckLinkVideoBuffer* buffer = nullptr;
    result = frame->QueryInterface(IID_IDeckLinkVideoBuffer, (void**)&buffer);
    if (result == S_OK) {
        result = buffer->StartAccess(bmdBufferAccessWrite);
        if (result == S_OK) {
            void* destination = nullptr;
            result = buffer->GetBytes(&destination);
            if (result == S_OK)
                std::memcpy(destination, pixels, (size_t)row_bytes * (size_t)height);
            buffer->EndAccess(bmdBufferAccessWrite);
        }
        buffer->Release();
    }
    if (result == S_OK)
        result = handle->output->ScheduleVideoFrame(frame, display_time, duration, time_scale);
    // the output keeps its own reference to a scheduled frame until it has been shown
    frame->Release();
    return result;
}

int32_t g2g_decklink_enable_audio(g2g_decklink_output* handle, uint32_t sample_type,
    uint32_t channels)
{
    return handle->output->EnableAudioOutput(bmdAudioSampleRate48kHz, sample_type, channels,
        bmdAudioOutputStreamContinuous);
}

int32_t g2g_decklink_begin_audio_preroll(g2g_decklink_output* handle)
{
    return handle->output->BeginAudioPreroll();
}

int32_t g2g_decklink_schedule_audio(g2g_decklink_output* handle, const void* samples,
    uint32_t sample_frames, int64_t stream_time, int64_t time_scale, uint32_t* written)
{
    return handle->output->ScheduleAudioSamples(const_cast<void*>(samples), sample_frames,
        stream_time, time_scale, written);
}

int32_t g2g_decklink_start_playback(g2g_decklink_output* handle, int64_t start_time,
    int64_t time_scale)
{
    return handle->output->StartScheduledPlayback(start_time, time_scale, 1.0);
}

int32_t g2g_decklink_stop_playback(g2g_decklink_output* handle, int64_t stop_time,
    int64_t time_scale)
{
    return handle->output->StopScheduledPlayback(stop_time, nullptr, time_scale);
}

int32_t g2g_decklink_hardware_time(g2g_decklink_output* handle, int64_t time_scale,
    int64_t* hardware_time)
{
    return handle->output->GetHardwareReferenceClock(time_scale, hardware_time, nullptr, nullptr);
}

}
