// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#include <atomic>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

#include "DeckLinkAPI.h"

// defined in DeckLinkAPIDispatch.cpp without a declaration in any header
bool IsDeckLinkAPIPresent(void);

static bool same_iid(REFIID left, REFIID right)
{
    return std::memcmp(&left, &right, sizeof(REFIID)) == 0;
}

extern "C" {

enum {
    G2G_DECKLINK_OPENED = 0,
    G2G_DECKLINK_LIBRARY_MISSING = 1,
    G2G_DECKLINK_NO_DEVICE = 2,
    G2G_DECKLINK_NO_OUTPUT = 3,
    G2G_DECKLINK_NO_ITERATOR = 4,
    G2G_DECKLINK_NO_INPUT = 5,
};

struct g2g_decklink_output {
    IDeckLink* device;
    IDeckLinkOutput* output;
};

struct g2g_decklink_ancillary_packet {
    const uint8_t* user_data;
    uint32_t user_data_count;
    uint32_t line_number;
    uint8_t did;
    uint8_t sdid;
};

}

// ST 291 data count is one byte
static const uint32_t MAX_USER_DATA_COUNT = 255;

class AncillaryPacket final : public IDeckLinkAncillaryPacket {
public:
    explicit AncillaryPacket(const g2g_decklink_ancillary_packet& packet)
        : m_refcount(1)
        , m_user_data_count(packet.user_data_count)
        , m_line_number(packet.line_number)
        , m_did(packet.did)
        , m_sdid(packet.sdid)
    {
        // the SDK keeps attached packets until the frame has been shown
        std::memcpy(m_user_data, packet.user_data, m_user_data_count);
    }

    HRESULT QueryInterface(REFIID iid, LPVOID* object) override
    {
        if (same_iid(iid, IID_IUnknown) || same_iid(iid, IID_IDeckLinkAncillaryPacket)) {
            AddRef();
            *object = static_cast<IDeckLinkAncillaryPacket*>(this);
            return S_OK;
        }
        *object = nullptr;
        return E_NOINTERFACE;
    }

    ULONG AddRef() override { return ++m_refcount; }

    ULONG Release() override
    {
        ULONG remaining = --m_refcount;
        if (remaining == 0)
            delete this;
        return remaining;
    }

    HRESULT GetBytes(BMDAncillaryPacketFormat format, const void** data, uint32_t* size) override
    {
        if (format != bmdAncillaryPacketFormatUInt8)
            return E_NOTIMPL;
        if (data != nullptr)
            *data = m_user_data;
        if (size != nullptr)
            *size = m_user_data_count;
        return S_OK;
    }

    uint8_t GetDID() override { return m_did; }
    uint8_t GetSDID() override { return m_sdid; }
    uint32_t GetLineNumber() override { return m_line_number; }
    uint8_t GetDataStreamIndex() override { return 0; }
    BMDAncillaryDataSpace GetDataSpace() override { return bmdAncillaryDataSpaceVANC; }

private:
    std::atomic<ULONG> m_refcount;
    uint8_t m_user_data[MAX_USER_DATA_COUNT];
    uint32_t m_user_data_count;
    uint32_t m_line_number;
    uint8_t m_did;
    uint8_t m_sdid;
};

static HRESULT attach_ancillary(IDeckLinkMutableVideoFrame* frame,
    const g2g_decklink_ancillary_packet* packets, uint32_t packet_count)
{
    if (packet_count == 0)
        return S_OK;
    IDeckLinkVideoFrameAncillaryPackets* attached = nullptr;
    HRESULT result = frame->QueryInterface(IID_IDeckLinkVideoFrameAncillaryPackets, (void**)&attached);
    if (result != S_OK)
        return result;
    for (uint32_t index = 0; index < packet_count && result == S_OK; index++) {
        if (packets[index].user_data_count > MAX_USER_DATA_COUNT) {
            result = E_INVALIDARG;
            break;
        }
        AncillaryPacket* packet = new (std::nothrow) AncillaryPacket(packets[index]);
        if (packet == nullptr) {
            result = E_OUTOFMEMORY;
            break;
        }
        result = attached->AttachPacket(packet);
        packet->Release();
    }
    attached->Release();
    return result;
}

extern "C" {

static int32_t find_device(uint32_t device_number, IDeckLink** found)
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
    *found = device;
    return G2G_DECKLINK_OPENED;
}

int32_t g2g_decklink_open(uint32_t device_number, g2g_decklink_output** opened)
{
    IDeckLink* device = nullptr;
    int32_t status = find_device(device_number, &device);
    if (status != G2G_DECKLINK_OPENED)
        return status;

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

int32_t g2g_decklink_enable_video(g2g_decklink_output* handle, uint32_t display_mode,
    bool vanc)
{
    BMDVideoOutputFlags flags = vanc ? bmdVideoOutputVANC : bmdVideoOutputFlagDefault;
    return handle->output->EnableVideoOutput(display_mode, flags);
}

int32_t g2g_decklink_schedule_video(g2g_decklink_output* handle, const uint8_t* pixels,
    int32_t width, int32_t height, int32_t row_bytes, uint32_t pixel_format,
    int64_t display_time, int64_t duration, int64_t time_scale,
    const g2g_decklink_ancillary_packet* packets, uint32_t packet_count)
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
        result = attach_ancillary(frame, packets, packet_count);
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

struct g2g_decklink_input_frame {
    const uint8_t* pixels;
    uint32_t row_bytes;
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    int64_t hardware_time;
    bool no_signal;
    const g2g_decklink_ancillary_packet* packets;
    uint32_t packet_count;
};

typedef void (*g2g_decklink_frame_arrived)(void* context, const g2g_decklink_input_frame* frame);
typedef void (*g2g_decklink_format_changed)(void* context, uint32_t display_mode,
    uint32_t detected_signal_flags);
typedef void (*g2g_decklink_capture_failed)(void* context, int32_t hresult);

}

class InputCallback final : public IDeckLinkInputCallback {
public:
    InputCallback(void* context, int64_t time_scale, g2g_decklink_frame_arrived frame_arrived,
        g2g_decklink_format_changed format_changed, g2g_decklink_capture_failed capture_failed)
        : m_refcount(1)
        , m_context(context)
        , m_time_scale(time_scale)
        , m_frame_arrived(frame_arrived)
        , m_format_changed(format_changed)
        , m_capture_failed(capture_failed)
    {
    }

    HRESULT QueryInterface(REFIID iid, LPVOID* object) override
    {
        if (same_iid(iid, IID_IUnknown) || same_iid(iid, IID_IDeckLinkInputCallback)) {
            AddRef();
            *object = static_cast<IDeckLinkInputCallback*>(this);
            return S_OK;
        }
        *object = nullptr;
        return E_NOINTERFACE;
    }

    ULONG AddRef() override { return ++m_refcount; }

    ULONG Release() override
    {
        ULONG remaining = --m_refcount;
        if (remaining == 0)
            delete this;
        return remaining;
    }

    HRESULT VideoInputFormatChanged(BMDVideoInputFormatChangedEvents, IDeckLinkDisplayMode* mode,
        BMDDetectedVideoInputFormatFlags detected_signal_flags) override
    {
        m_format_changed(m_context, mode->GetDisplayMode(), detected_signal_flags);
        return S_OK;
    }

    HRESULT VideoInputFrameArrived(IDeckLinkVideoInputFrame* frame, IDeckLinkAudioInputPacket*) override
    {
        if (frame == nullptr)
            return S_OK;
        HRESULT result = deliver(frame);
        if (result != S_OK)
            m_capture_failed(m_context, result);
        return S_OK;
    }

private:
    HRESULT deliver(IDeckLinkVideoInputFrame* frame)
    {
        g2g_decklink_input_frame captured = {};
        captured.row_bytes = (uint32_t)frame->GetRowBytes();
        captured.width = (uint32_t)frame->GetWidth();
        captured.height = (uint32_t)frame->GetHeight();
        captured.pixel_format = frame->GetPixelFormat();
        captured.no_signal = (frame->GetFlags() & bmdFrameHasNoInputSource) != 0;
        BMDTimeValue duration = 0;
        HRESULT result = frame->GetHardwareReferenceTimestamp(m_time_scale, &captured.hardware_time, &duration);
        if (result != S_OK)
            return result;

        std::vector<IDeckLinkAncillaryPacket*> held;
        std::vector<g2g_decklink_ancillary_packet> packets;
        result = collect_ancillary(frame, held, packets);
        IDeckLinkVideoBuffer* buffer = nullptr;
        if (result == S_OK)
            result = frame->QueryInterface(IID_IDeckLinkVideoBuffer, (void**)&buffer);
        if (result == S_OK) {
            result = buffer->StartAccess(bmdBufferAccessRead);
            if (result == S_OK) {
                void* pixels = nullptr;
                result = buffer->GetBytes(&pixels);
                if (result == S_OK) {
                    captured.pixels = static_cast<const uint8_t*>(pixels);
                    captured.packets = packets.data();
                    captured.packet_count = (uint32_t)packets.size();
                    m_frame_arrived(m_context, &captured);
                }
                buffer->EndAccess(bmdBufferAccessRead);
            }
            buffer->Release();
        }
        for (IDeckLinkAncillaryPacket* packet : held)
            packet->Release();
        return result;
    }

    static HRESULT collect_ancillary(IDeckLinkVideoInputFrame* frame,
        std::vector<IDeckLinkAncillaryPacket*>& held,
        std::vector<g2g_decklink_ancillary_packet>& packets)
    {
        IDeckLinkVideoFrameAncillaryPackets* attached = nullptr;
        if (frame->QueryInterface(IID_IDeckLinkVideoFrameAncillaryPackets, (void**)&attached) != S_OK)
            return S_OK;
        IDeckLinkAncillaryPacketIterator* iterator = nullptr;
        HRESULT result = attached->GetPacketIterator(&iterator);
        attached->Release();
        if (result != S_OK)
            return result;
        IDeckLinkAncillaryPacket* packet = nullptr;
        try {
            while (iterator->Next(&packet) == S_OK) {
                // a packet's user data stays valid while a reference to it is held
                held.push_back(packet);
                const void* user_data = nullptr;
                uint32_t user_data_count = 0;
                result = packet->GetBytes(bmdAncillaryPacketFormatUInt8, &user_data, &user_data_count);
                if (result != S_OK)
                    break;
                packets.push_back(g2g_decklink_ancillary_packet {
                    static_cast<const uint8_t*>(user_data), user_data_count,
                    packet->GetLineNumber(), packet->GetDID(), packet->GetSDID() });
            }
        } catch (const std::bad_alloc&) {
            result = E_OUTOFMEMORY;
        }
        iterator->Release();
        return result;
    }

    std::atomic<ULONG> m_refcount;
    void* m_context;
    int64_t m_time_scale;
    g2g_decklink_frame_arrived m_frame_arrived;
    g2g_decklink_format_changed m_format_changed;
    g2g_decklink_capture_failed m_capture_failed;
};

extern "C" {

struct g2g_decklink_input {
    IDeckLink* device;
    IDeckLinkInput* input;
    InputCallback* callback;
};

int32_t g2g_decklink_open_input(uint32_t device_number, g2g_decklink_input** opened)
{
    IDeckLink* device = nullptr;
    int32_t status = find_device(device_number, &device);
    if (status != G2G_DECKLINK_OPENED)
        return status;

    IDeckLinkInput* input = nullptr;
    if (device->QueryInterface(IID_IDeckLinkInput, (void**)&input) != S_OK) {
        device->Release();
        return G2G_DECKLINK_NO_INPUT;
    }
    g2g_decklink_input* handle = new (std::nothrow) g2g_decklink_input{device, input, nullptr};
    if (handle == nullptr) {
        input->Release();
        device->Release();
        return E_OUTOFMEMORY;
    }
    *opened = handle;
    return G2G_DECKLINK_OPENED;
}

static void stop_input(g2g_decklink_input* handle)
{
    if (handle->callback == nullptr)
        return;
    handle->input->StopStreams();
    handle->input->SetCallback(nullptr);
    handle->callback->Release();
    handle->callback = nullptr;
}

void g2g_decklink_close_input(g2g_decklink_input* handle)
{
    stop_input(handle);
    handle->input->DisableVideoInput();
    handle->input->Release();
    handle->device->Release();
    delete handle;
}

int32_t g2g_decklink_set_input_connection(g2g_decklink_input* handle, uint32_t connection)
{
    IDeckLinkConfiguration* configuration = nullptr;
    HRESULT result = handle->device->QueryInterface(IID_IDeckLinkConfiguration, (void**)&configuration);
    if (result != S_OK)
        return result;
    result = configuration->SetInt(bmdDeckLinkConfigVideoInputConnection, connection);
    configuration->Release();
    return result;
}

int32_t g2g_decklink_supports_format_detection(g2g_decklink_input* handle, bool* supported)
{
    IDeckLinkProfileAttributes* attributes = nullptr;
    HRESULT result = handle->device->QueryInterface(IID_IDeckLinkProfileAttributes, (void**)&attributes);
    if (result != S_OK)
        return result;
    result = attributes->GetFlag(BMDDeckLinkSupportsInputFormatDetection, supported);
    attributes->Release();
    return result;
}

int32_t g2g_decklink_enable_video_input(g2g_decklink_input* handle, uint32_t display_mode,
    uint32_t pixel_format, bool format_detection)
{
    BMDVideoInputFlags flags = format_detection ? bmdVideoInputEnableFormatDetection : bmdVideoInputFlagDefault;
    return handle->input->EnableVideoInput(display_mode, pixel_format, flags);
}

int32_t g2g_decklink_start_input(g2g_decklink_input* handle, void* context, int64_t time_scale,
    g2g_decklink_frame_arrived frame_arrived, g2g_decklink_format_changed format_changed,
    g2g_decklink_capture_failed capture_failed)
{
    stop_input(handle);
    InputCallback* callback = new (std::nothrow)
        InputCallback(context, time_scale, frame_arrived, format_changed, capture_failed);
    if (callback == nullptr)
        return E_OUTOFMEMORY;
    HRESULT result = handle->input->SetCallback(callback);
    if (result != S_OK) {
        callback->Release();
        return result;
    }
    handle->callback = callback;
    return handle->input->StartStreams();
}

int32_t g2g_decklink_restart_video_input(g2g_decklink_input* handle, uint32_t display_mode,
    uint32_t pixel_format)
{
    // the steps GStreamer's decklink input takes when the SDK reports a new input format
    HRESULT result = handle->input->PauseStreams();
    if (result == S_OK)
        result = handle->input->EnableVideoInput(display_mode, pixel_format, bmdVideoInputEnableFormatDetection);
    if (result == S_OK)
        result = handle->input->FlushStreams();
    if (result == S_OK)
        result = handle->input->StartStreams();
    return result;
}

int32_t g2g_decklink_input_hardware_time(g2g_decklink_input* handle, int64_t time_scale,
    int64_t* hardware_time)
{
    return handle->input->GetHardwareReferenceClock(time_scale, hardware_time, nullptr, nullptr);
}

}
