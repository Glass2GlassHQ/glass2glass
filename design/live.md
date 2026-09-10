# Live capture, ingress and egress

Camera and audio capture, device discovery, the RTP, RTMP, RTSP and SRT paths
in both directions, and the fallback switching that keeps a live source
running. Part of the design in [README.md](README.md).

## Live capture

### V4l2Src

`V4l2Src` (`v4l2src.rs`, `v4l2` feature, Linux-only) streams frames off a
`/dev/videoN` device through V4L2 mmap streaming I/O, wrapping the pure-Rust
`v4l` crate with no libv4l C dependency. Packed YUYV, 4:2:2, the near-universal
UVC output, is the preferred format, and `VideoConvert` unpacks it to a planar
or RGB target (the raw formats in [README.md](README.md)), so the canonical
chain is
`V4l2Src -> VideoConvert(Yuyv -> Nv12) -> sink`.

V4L2 dequeue is a blocking ioctl, so capture runs on a dedicated `std::thread`
that owns the device and the mmap stream borrowing it, and copies each frame's
payload into a bounded channel that `SourceLoop::run` drains into `DataFrame`s.
The channel bound `BUFFER_COUNT` applies backpressure: the capture thread
blocks rather than growing memory when the pipeline falls behind. The source
reports a live `LatencyReport` of one frame period.

Negotiation runs up front on a probe device. It enumerates the pixel formats,
sets each one it can carry at the requested geometry and reads back what the
driver chose, which may snap to a supported mode. The probe device is dropped
and the capture thread re-opens the device under the negotiated format. Keeping
no device handle in the struct between negotiation and `run` sidesteps `Send`
and borrow entanglement with the stream. Errors surface as
`G2gError::Hardware(HardwareError::V4l2(errno))`.

Every confirmed format becomes one alternative of the source's
`CapsConstraint::Produces` set, in a fixed preference order: YUYV, NV12, I420,
then MJPEG last because it needs a decoder. A chain that constrains nothing
takes YUYV. A downstream `MjpegDec` or a pinned `image/jpeg` link drops the raw
alternatives during arc consistency and the camera runs in its MJPEG mode,
which is what fits 1080p over USB. `configure_pipeline` reads the solved caps
back to learn which mode the capture thread runs, and MJPEG's per-frame length
comes from the buffer's `bytesused`, not the format's `sizeimage`. What a pixel
format means on a link, its `Caps` and its frame size, is in
`capturepixelformat.rs`, shared with `LibCameraSrc`, which sits on a different
fourcc registry but agrees on the meaning.

`io-mode` selects how a buffer leaves the element. `auto` and `mmap` copy.
`dmabuf` exports each MMAP buffer once at stream start with `VIDIOC_EXPBUF` and
emits frames in `MemoryDomain::DmaBuf` carrying a share of the fd their buffer
was filled into, so a GPU consumer imports the camera buffer with no copy. The
buffer is the frame there, so it goes back to the driver only once every share
of its fd has dropped: the element holds one share per buffer for the whole
stream and re-queues when the count falls back to it, and the in-flight bound
stays below `BUFFER_COUNT` so the driver always has a buffer to fill. An
exported fd carries no payload length, so dmabuf mode advertises only the raw
formats and MJPEG stays on the copy path.

### LibCameraSrc

`LibCameraSrc` (`libcamerasrc.rs`, `libcamera` feature, Linux-only) captures
through the system libcamera via the `libcamera` crate. It drives UVC webcams
through the `uvcvideo` pipeline handler, the same devices as `V4l2Src`, plus
CSI/ISP cameras that need an ISP pipeline V4L2 alone cannot provide. It follows
`V4l2Src`'s two design points, blocking work off the async path and up-front
negotiation with a re-configure for capture, and differs in two ways.

It asks for NV12 and falls back to YUYV only when the camera does not offer
NV12, mapping whatever survives `validate()` to `Caps::RawVideo`, so a camera
producing planar frames needs no `VideoConvert`. `with_mjpeg(true)` or
`format=mjpeg` negotiates MJPEG instead and emits `CompressedVideo{Mjpeg}` for
`MjpegDec`, the on-camera-compression path for resolutions and frame rates
uncompressed YUYV cannot sustain over USB. Because libcamera is callback-driven
and thread-affine, the capture thread owns the whole libcamera object graph,
manager, camera, request-buffer ring and completion callback, rather than a
single device handle, and packs each completed request's planes contiguously,
Y then interleaved UV for NV12, before forwarding them over the bounded
channel.

The frame rate is bounded on the camera with a `FrameDurationLimits` start
control whose minimum frame duration caps the fastest rate, the maximum left
generous so an unachievable request degrades to the camera's own ceiling
instead of collapsing. Manual exposure and gain, `with_exposure` and
`with_gain`, turn auto-exposure off, ride the same start-control path, and are
the frame-rate lever in low light: auto-exposure lengthens exposure until the
rate collapses to about 9 fps on a dim webcam, the same rate in every format
and resolution, and a fixed short exposure restored 8.8 to 24.9 fps.
`Brightness`, `Contrast` and `Saturation` are post-capture adjustments that do
not touch exposure time, so they brighten a dim short-exposure frame without
giving back the frame rate, measured mean luma 16 to 117 at a fixed exposure.
`with_camera_id` selects the camera by id substring rather than enumeration
index, stable across reboots. Start controls are applied only after a support
check against the camera's `ControlInfoMap`, because libcamera aborts the
process with a C++ exception across the FFI boundary if a control list carries
an id the pipeline handler does not advertise, and a UVC webcam may expose
`ExposureTime` but not `AnalogueGain`. The `libcamera` crate requires libcamera
`>= 0.4`, newer than some distro packages, so the feature is host-validated
rather than built in CI.

The g2g-ml `libcamera-wgpu` feature chains
`LibCameraSrc -> VideoConvert(NV12) -> WgpuPreprocess` to turn live frames into
a normalized f32 NCHW tensor on the GPU, validated from camera to tensor on an
RTX 3060. Zero-copy dma-buf import of libcamera buffers into wgpu, the Linux
analog of the CUDA and AHardwareBuffer interop, sits under `libcamera-dmabuf`.
libcamera exports a real dma-buf fd, but on a USB camera plus discrete NVIDIA
GPU the driver advertises the buffer as importable through
`vkGetMemoryFdPropertiesKHR` while `vkAllocateMemory` fails to bind it, because
the buffer is CPU/vmalloc backed and the dGPU cannot map it. CPU upload is the
correct path for that configuration. Zero-copy is expected to work on an
integrated GPU with shared memory or a CSI/ISP camera with GPU-visible buffers,
so the import-to-texture element is gated behind the on-hardware probe rather
than shipped blind.

### PipeWireSrc and PipeWireVideoSrc

Both follow the blocking-work-off-the-async-path shape. `PipeWireSrc`
(`pipewire` feature, Linux) captures interleaved PCM off the PipeWire graph by
running a `pw::stream` input on a dedicated main-loop worker thread feeding the
`run` loop over a channel, and requests a fixed PCM format the PipeWire adapter
converts to, so the produced caps are deterministic.

`PipeWireVideoSrc` captures raw frames from any PipeWire video node, a camera,
another client, or a portal-opened screen-cast node named through
`target-object`, and its `io-mode` property picks the buffer path. `mmap`
copies each frame out of the mapped block into `System` memory. `dmabuf`
negotiates a `Buffers` param accepting `SPA_DATA_DmaBuf` alone and hands the
descriptor on as `MemoryDomain::DmaBuf`, holding each buffer until every share
of its frame is released. The domain is fixed by negotiation, hence a property
rather than a per-caps feature.

Screen capture on a Wayland desktop goes through `portal=true` (`portal`
feature) instead of `target-object`, which only reaches the session's own
PipeWire remote. The element runs the xdg-desktop-portal `ScreenCast`
handshake, `CreateSession` then `SelectSources` then `Start`, each answered on
an `org.freedesktop.portal.Request` object, then `OpenPipeWireRemote`, on the
capture worker thread over a blocking zbus connection, and connects the stream
to the granted node id on the private remote fd the portal returns. Every step
is bounded by `portal-timeout`, so an unattended consent dialog fails the
capture instead of hanging, and `portal-restore-token` re-opens an earlier
grant without asking.

### MfVideoSrc

`MfVideoSrc` (`mf-video-src`, Windows) is the camera sibling of `WasapiSrc`. It
enumerates video capture devices and drains NV12 and YUY2 frames through an
`IMFSourceReader` on a COM/MTA worker thread.

## Linux audio output

The audible-output end of the audio path on Linux mirrors the Windows-only
`WasapiSink` across the three Linux audio stacks. Each is a `std`-gated element
with a dedicated render worker thread: `AlsaSink` (`alsa-sink`, libasound,
lowest level), `PulseSink` (`pulse-sink`, the blocking libpulse "simple" API),
and `PipeWireSink` (`pipewire`). ALSA and Pulse backpressure through the
blocking write. PipeWire's `process` callback pulls on its own clock and cannot
backpressure, so that sink's hand-off queue is leaky, bounded to about 1 s and
dropping the oldest bytes, the `LinkPolicy::DropOldest` analog for an external
clock. All accept interleaved `PcmS16Le` and `PcmF32Le` and reject compressed
audio structurally. Errors surface as
`HardwareError::{Alsa,PulseAudio,PipeWire}`.

## Device discovery

The `GstDeviceProvider` and `GstDeviceMonitor` analog is in
`g2g-core/src/runtime/device.rs` and `g2g-plugins/src/devicemon.rs`. A
`DeviceProvider` probes one backend for the devices it can see. A
`DeviceMonitor` aggregates providers behind class and caps filters with
`gst_device_has_classes` semantics, where `Video/Source` requires both parts
and `Source` matches any source, and once started it watches for hotplug.
Events arrive on the monitor's own channel, not the pipeline bus: a monitor is
application-side, not part of a running graph.

A `Device` does not own an element factory. It carries the launch name of the
element that drives it plus the textual `key=value` properties that select it,
so construction rides the same `Registry` and `PropertySpec::parse_value` path
as `parse_launch`. `Device::create` builds and configures the element, and
`Device::launch_fragment` prints the `v4l2src device=/dev/video0` fragment a
text pipeline would use. `persistent_id` is the monitor's hotplug diff key and
is chosen per backend for cross-reboot stability: USB/PCI bus info for v4l2,
the direction-prefixed hint name for ALSA, and `node.name` for PipeWire, which
survives daemon restarts where `object.serial` does not.

### Hotplug

A provider with a native event source implements `watch()`: PipeWire registers
a registry listener on a dedicated loop thread, relies on the daemon replaying
existing globals for the initial `Added` set, and posts through a
filter-applying `DeviceSink`, a try-send retry loop so N watcher threads never
depend on the channel's single send waker. Shutdown closes the receiver first,
so a watcher blocked on a full queue exits instead of deadlocking the join.
Providers without events, v4l2, ALSA and GPU, are covered by the monitor's
poll-and-diff fallback thread keyed on `persistent_id`.

### Standard providers

`default_device_monitor` gathers them, mirroring `default_registry`'s
per-feature gating.

- v4l2: capture nodes with YUYV modes probed into real caps alternatives.
  Other fourccs are listed in `detail`.
- ALSA: PCM hints in both directions, formats probed through `HwParams`. A busy
  device is still listed, with empty caps.
- PipeWire: media-class nodes mapped to `pipewiresrc`, `pipewiresink` and
  `pipewirevideosrc`, selected through their `target-object` property.
- GPU: `Compute/GPU`, a g2g extension beyond GStreamer's capture/render model,
  covering wgpu adapters, CUDA ordinals and VAAPI render nodes. Only the render
  nodes name a driving element, the rest are informational.
- `mfdevice` and `wasapidevice` on Windows, `avfdevice` and `coreaudiodevice`
  on macOS.

The `g2g-device-monitor` binary is the CLI over all of this, the
`gst-device-monitor-1.0` analog: one-shot listing, class filter, `--json`, and
`--follow` for live hotplug.

### Windows and macOS providers

`mfdevice` lists the `MFEnumDeviceSources` video capture devices with the NV12
and YUY2 native modes `mfvideosrc` can deliver. Reading them activates the
source, so a camera another application holds open lists with empty caps.
`wasapidevice` lists the active `IMMDeviceEnumerator` render and capture
endpoints with their shared-mode mix format as caps. `avfdevice` lists cameras
from an `AVCaptureDeviceDiscoverySession`, and `coreaudiodevice` the HAL's
`kAudioHardwarePropertyDevices` entries, one device record per direction a
duplex device carries.

WASAPI is the one non-Linux backend with a native watch, an
`IMMNotificationClient` whose callbacks wake a re-probe on the watch thread,
since the callback carries only an id and must not block. The other three are
polled, because MF has no hotplug callback short of a `WM_DEVICECHANGE` window
and the AVFoundation and CoreAudio listeners need a run loop a library has no
business owning.

On these platforms the selection property is the persistent id, so no separate
`device-id` is needed. `mfvideosrc device-path=` takes the MF symbolic link,
`wasapisrc` and `wasapisink device=` the endpoint id, `avfvideosrc` and
`avfaudiosrc device=` the `AVCaptureDevice` unique id, and `coreaudiosrc` and
`coreaudiosink device=` the Core Audio device UID, since the `AudioDeviceID` is
reassigned every boot and the UID is not. Each element has that selector and
shares one open-by-id helper with its provider, `wasapipcm` on Windows, so the
id a listing reports and the id `device=` accepts cannot drift.

### v4l2src device-id

V4L2 is the exception: its selection handle is the node path, which the kernel
renumbers across a replug, so `v4l2src` takes a separate `device-id` carrying
the provider's `bus_info:card:path` id and resolves it against a fresh probe at
negotiation. The exact id wins. Failing that, the hardware half, bus plus card,
matches, which is what survives a replug into the same port, and the
lowest-numbered node of a multi-node camera is chosen, the capture node on
every UVC device. An id nothing carries fails the negotiation with
`HardwareError::V4l2(ENODEV)` rather than silently falling back to `device`.

### V4L2 camera controls

`v4l2src` exposes the user and camera control classes as runtime properties
under the names `v4l2-ctl` uses: exposure, focus and white balance
(`exposure-auto`, `exposure-absolute`, `focus-auto`, `focus-absolute`,
`white-balance-temperature-auto`, `white-balance-temperature`), the picture
controls GStreamer's `v4l2src` also names (`brightness`, `contrast`,
`saturation`, `hue`, `gamma`, `gain`, `sharpness`, `backlight-compensation`,
`power-line-frequency`), and pan, tilt and zoom (`pan-absolute`,
`tilt-absolute`, `zoom-absolute`).

One table drives both the property specs and the `VIDIOC_S_EXT_CTRLS` ids, and
its order is the apply order: an auto switch precedes the manual value it
gates, because a driver rejects a manual exposure while auto exposure is on.
Each is applied with its own ioctl, since a batch may not span the user and
camera control classes. Only a control that was set is touched, and one the
camera does not implement fails the negotiation instead of being quietly
ignored. A control's property kind follows its range: a switch is `Bool`, the
four picture controls GStreamer also gives a signed property to plus pan and
tilt are `Int`, and the rest, whose range starts at zero on every device that
has them, are `Uint`.

Anything past that table is reachable through `extra-controls`, GStreamer's own
spelling, as a comma-separated `name=value` list. The names are the driver's
own, kebab-cased, which is how `g2g-device-monitor` lists them: the provider
walks `VIDIOC_QUERY_EXT_CTRL` and records every numeric control as a
`control.<name>` detail entry with the range and default the driver reports, so
a listing tells the caller exactly what an `extra-controls` entry may say. The
walk is g2g's own rather than the `v4l` crate's `query_controls`, which panics
on a control type its enum predates, uvcvideo's region-of-interest rectangle.
A malformed list fails the property. A name this device does not offer, or a
value outside its reported range, fails the negotiation and logs the names the
device does carry.

## Live egress

The receive path in [decode.md](decode.md) has an inverse:
encoded video out over RTP. The protocol logic is sans-IO. A pure packetizer
produces the RTP packets and a thin sink does the UDP I/O.

`RtpH264Packetizer` (`rtppay.rs`) implements RFC 3550 and RFC 6184. An H.264
access unit becomes a single-NAL RTP packet when the NAL fits the MTU, and FU-A
fragments when it does not. The marker bit lands on the access unit's last
packet, sequence numbers increment across packets and across calls, and one RTP
timestamp covers one access unit. The logic is pure `no_std` and host-testable.

`UdpSink` (`udpsink.rs`, `udp-egress` feature) is the `AsyncElement` sink that
drives the packetizer over each Annex-B access unit and sends the packets to a
destination on a tokio `UdpSocket`. The RTP timestamp is the 90 kHz image of
`FrameTiming::pts_ns`. `with_rtp(pt, ssrc)` and `with_max_payload(mtu)`
configure the flow. The sink also keeps a bounded history of recently sent
packets and honours a receive-side RTCP NACK by retransmitting from it
(`with_retransmit(enabled, capacity)`).

### SRTP

The `srtp` module (`srtp` feature) protects raw RTP and RTCP packets under an
`SrtpPolicy` of cipher (NULL, AES-128-CM, AES-256-CM, AES-128-GCM,
AES-256-GCM) and authentication (NULL, HMAC-SHA1-32, HMAC-SHA1-80), covering
RFC 3711, RFC 6188 and RFC 7714. `SrtpSender` and `SrtpReceiver` derive
separate RTP and RTCP session keys with the RFC 3711 KDF, track one SSRC's
rollover and SRTCP indices, and reject replayed packets. `SrtpReceiverSet`
keeps one receiver per SSRC and asks an `SrtpKeyProvider` for the master key
and initial rollover counter when a new SSRC appears.

`srtpenc` and `srtpdec` are the pipeline elements. Each instance carries one
flow, RTP or RTCP, chosen by the negotiated `ByteStreamEncoding`. The `key`
property is the hex master key followed by the master salt, 12 bytes for GCM
and 14 for counter mode. `rtp-cipher`, `rtcp-cipher`, `rtp-auth` and
`rtcp-auth` take GStreamer's names. Left unset, the cipher follows the key
length (28 or 44 bytes GCM, 30 or 46 counter mode) and the authentication is
HMAC-SHA1-80 for the non-AEAD ciphers.

The encoder takes its SSRC from the first packet, keeps its packet indices,
resets its usage counters on `replace_key`, and posts a bus `Info` once the
soft key-use limit is reached. The decoder keys every SSRC from `key` and `roc`
or from a programmatic provider, drops packets that fail authentication or
replay checks, and keeps its indices across a rekey. SRTCP supports encrypted
and authentication-only packets.

The replay window is `replay-window-size`, 64 to 32768 packets, default 128, a
`Vec<u64>` bitmap sized once. `allow-repeat-tx` lets the sender protect a
repeated index again under the same nonce. An MKI selects among the receive
keys a provider returns for one SSRC, placed after the AES-GCM tag, or between
the body and the HMAC tag for RFC 3711. `stats()` on both elements reports
packet counts and each context's rollover counter.

The module is `no_std + alloc` with software AES and no socket, thread, or OS
dependency. RFC 6904 encrypted header extensions, hardware AES, and heapless
packet storage are outside this layer.

### DTLS-SRTP

`dtlssrtpenc` and `dtlssrtpdec` (`dtls-srtp` feature, `std`) key that layer
from a DTLS 1.2 handshake run over the media socket, RFC 5764, on the `dimpl`
sans-IO DTLS stack. Both elements share one `DtlsSrtpConnection`, found by
`connection-id` or passed in.

`dtlssrtpenc` is a fan-in with one `rtp_%u` or `rtcp_%u` pad per flow and one
`application/x-dtls` output. Its tick services the handshake, meaning
retransmits and outbound flights, and it holds a bounded queue of media per
input until the keys arrive. `dtlssrtpdec` is the fan-out taking that datagram
stream, splitting DTLS from SRTP on the first byte (RFC 7983) and RTP from RTCP
by payload type (RFC 5761), with RTP on port 0 and RTCP on port 1.

The exporter block splits into client and server key and salt for the
negotiated profile, either of the two GCM ones or `SRTP_AES128_CM_SHA1_80`, the
only one GStreamer's `dtls` plugin offers. The client sends under the client
key. The local certificate is a self-signed ECDSA P-256 one unless `pem`
supplies it, `peer-pem` reads the peer's, and `peer-fingerprint`, in the SDP
`a=fingerprint` value form, aborts a handshake whose peer certificate does not
match. Any DTLS failure stops the run: nothing falls back to clear media.
[PORTING.md](../PORTING.md#srtp-gaps) compares the host integration surface with
GStreamer's SRTP elements.

## Live ingress over UDP and RTP

`UdpSrc` (`udpsrc.rs`, `udp-ingress` feature) is the receive-side inverse of
`UdpSink`. It receives RTP on a tokio `UdpSocket` and depayloads H.264 into
Annex-B access units pushed downstream as `CompressedVideo` H.264, so the
canonical chain is `UdpSrc -> FfmpegH264Dec -> sink`. The I/O is async, so
unlike `V4l2Src` it needs no capture thread.

The protocol logic is sans-IO, mirroring the egress split. `rtpdepay.rs`'s
`RtpH264Depayloader` is a pure, `no_std`, host-testable function that inverts
`RtpH264Packetizer`. Single-NAL and STAP-A payloads pass through, FU-A
fragments reassemble with the original NAL header rebuilt from the FU
indicator's F and NRI bits plus the FU header's type, and the RTP marker bit
closes an access unit. A sequence-number gap drops the in-flight reassembly, so
loss or reorder never welds two access units together.

### Jitter buffer, RTCP, NACK, RTX and FEC

Between the socket and the depayloader sits a sans-IO jitter buffer
(`rtpjitter.rs`, `RtpJitterBuffer`). It orders packets by an extended sequence
number, the 16-bit RTP sequence unrolled to a monotonic counter so wraparound is
handled, releases them in order, holds a gap only until its predecessors fill or
a bounded deadline elapses and then declares loss, and drops duplicates and
too-late packets.

RTCP (`rtcp.rs`) is sans-IO RFC 3550 SR, RR and BYE plus RFC 4585 Generic NACK,
with `ReceptionStats` carrying loss fraction, cumulative loss and interarrival
jitter. It runs RTP/RTCP-muxed on the one socket (RFC 5761). `UdpSrc` sends
periodic receiver reports and emits a NACK for each detected gap, and `UdpSink`
honours those NACKs from its send history. A retransmit arriving inside the
jitter hold window heals the gap before it is declared lost, so the loop
recovers packet loss end to end.

RFC 4588 RTX (`rtx.rs`) wraps a NACK resend in a distinct payload type and SSRC
with the original sequence prepended (`UdpSink::with_rtx`, `UdpSrc::with_rtx`),
which stays unambiguous under heavy loss. ULPFEC (`ulpfec.rs`, RFC 5109) adds
feedback-free recovery: the sender XORs each group of media packets into a
repair packet (`with_fec`), and the receiver reconstructs a single per-group
loss from the repair plus the survivors and injects it into the jitter buffer.
That costs no round trip, which is the better fit for one-way or high-RTT
paths. NACK, RTX and FEC compose.

This is raw RTP with no RTSP or SDP, so there is no out-of-band stream
description. The output geometry is a declared hint (`with_video_size`,
`with_framerate`), and since H.264 carries its real dimensions in the SPS a
downstream decoder re-derives and corrects them. `RtspSrc` covers the RTSP case
with its own jitter buffer ([decode.md](decode.md)).

## RTMP

`RtmpSrc` (`rtmpsrc.rs`, `rtmp` feature) accepts one RTMP publisher, ffmpeg or
OBS pushing `rtmp://host/app/key`, over TCP and streams the result downstream as
`Caps::ByteStream{Flv}`, so the chain is `RtmpSrc -> flvdemux -> h264parse`.
The protocol is sans-IO (`rtmp.rs`, `RtmpSession`): the simple non-digest
handshake publishers fall back to, the chunk-stream reassembly with
per-chunk-stream header inheritance and `Set Chunk Size`, and the AMF0
`connect` / `createStream` / `publish` command flow, where the session emits the
Window-Ack, Set-Peer-Bandwidth, `_result` and `onStatus` replies. An RTMP audio
or video message payload is exactly an FLV tag body, so the session reframes the
messages into an FLV byte stream that the existing `flvdemux`
([containers.md](containers.md)) recovers the H.264 and AAC access
units from. Scope is one publisher, one stream, H.264 plus AAC, AMF0.

`RtmpSink` (`rtmpsink.rs`, `rtmp` feature) is the inverse. It connects out to an
RTMP server and publishes an incoming FLV byte stream, so the chain is
`flvmux ! RtmpSink location=rtmp://host/app/key`. The protocol is sans-IO
(`rtmp.rs`, `RtmpPublisher`), the mirror of `RtmpSession`: it sends C0 and C1,
drives the `connect` / `createStream` / `publish` command ladder off the
server's `_result` and `onStatus` replies, then splits the FLV stream back into
tags and reframes each as an RTMP audio, video or data message, the tag body
being the message payload. Both directions share one `ChunkReader` and one
fragmenting `write_message` writer, so the publisher and the session are true
inverses rather than parallel re-implementations. The element opens the socket
lazily on the first buffer, after `flvmux`'s header, and drives the publish
ladder before sending media. It is validated sans-IO by pitting the publisher
against the server session, where an access unit survives the RTMP round trip.
Live publish to a real endpoint is operator-validated.

Both halves also speak the HMAC-SHA256 "genuine FMS / FP" digest handshake that
strict CDNs require (`rtmphandshake.rs`). `RtmpPublisher` sends a digest C1 and
response C2 by default and `RtmpSession` answers and validates it, each falling
back to the simple handshake against a non-genuine peer.

Window-acknowledgement back-pressure runs in both directions. `RtmpSession`
emits an `Acknowledgement` every Window-Ack-Size bytes received
(`with_window_ack_size`), and `RtmpPublisher` tracks the server's window against
the acknowledged sequence and exposes `throttled()`, on which `RtmpSink` blocks
feeding media, so a slow server back-pressures the pipeline instead of bloating
the socket buffer. Ingest is validated against a real peer
(`rtmp_ffmpeg_interop`, ffmpeg publishing into `RtmpSrc`).

## RTSP server

`RtspServerSink` (`rtspserversink.rs`, `rtsp-server` feature) hosts the server
side of RTSP. A player connects over TCP, runs OPTIONS, DESCRIBE, SETUP and
PLAY, and the sink streams the pipeline's H.264 to the player's negotiated UDP
port as RTP, reusing `RtpH264Packetizer`. The protocol is sans-IO
(`rtspserver.rs`, `RtspResponder`, `RtspRequest::parse`, `sdp_h264`): a
per-session state machine answering each method and returning an `RtspEvent`
(`Setup{client_rtp_port}`, `Play`, `Record`, `Teardown`) that the element acts
on. It also speaks the publisher path, ANNOUNCE and RECORD, served by the
receive-side `RtspServerSrc`.

The sink is multi-client, each player getting its own RTP session broadcast per
frame, on either transport: unicast UDP or TCP-interleaved, `$`-framed on the
control connection and validated against `ffmpeg -rtsp_transport tcp`. During
PLAY the sink runs RTCP and keepalive: periodic RFC 3550 sender reports per
player, over UDP from the socket adjacent to the RTP one so the advertised
`server_port` pair is real, or `$`-framed on the RTCP channel, a BYE at EOS, and
a session timeout advertised as `Session: id;timeout=N` at SETUP. A player is
reaped when it is silent past the timeout on both the control channel
(GET_PARAMETER, OPTIONS) and RTCP, whose receiver reports arrive `$`-framed
mid-stream on an interleaved control connection and are consumed there.
Validated end to end over loopback: handshake, RTP recovery, SR delivery on both
transports, RR-extended lifetime, and silent-client reap.

## SRT

`SrtSink` (caller, egress) and `SrtSrc` (listener, ingress, `srt` feature) carry
an MPEG-TS byte stream over UDP with SRT's reliable but low-latency ARQ, the
contribution-link transport. The protocol is sans-IO (`srt.rs`): the 16-byte
packet header plus data and control wire codec (HSv5 HANDSHAKE with the
HSREQ-latency and Stream-ID extensions, ACK, NAK loss report, ACKACK,
KEEPALIVE, SHUTDOWN), the caller and listener handshake driver (`SrtHandshake`,
induction to conclusion with a listener cookie challenge), and the ARQ pair
`SrtSender` and `SrtReceiver`. The sender buffers and resends on NAK with the
retransmit flag, and the receiver reorders by wrap-aware sequence, NAKs gaps,
and delivers in order, the same shape as the RTP jitter and NACK path.

Validated g2g to g2g end to end over a lossy loopback: handshake, data, and a
dropped packet recovered via NAK. The wire format follows the SRT draft so
real-peer interop is the design target. AES-256 encryption (`with_aes256`),
mid-stream key rotation (`with_key_rotation`), the TSBPD timing model, and
live-mode congestion control and pacing (`with_max_bandwidth`) are in place. A
rotated key's KM message is retransmitted until the peer answers with a KMRSP,
so a rekey survives KM-packet loss. Real-peer interop with libsrt and ffmpeg is
validated for the full matrix by the ignored `srt_ffmpeg_interop` test, which
needs ffmpeg built with libsrt: both directions, ffmpeg caller into `SrtSrc`
listener and `SrtSink` caller into an ffmpeg listener, across plaintext,
AES-128 and AES-256.

## Fallback switching

`fallbackswitch` is `input-selector` with the choice made for it. Input 0 is the
primary and each higher index the next fallback, so the input index is the
priority until `sinkN-priority` says otherwise. gst defaults a request pad's
`priority` to its pad serial and g2g defaults input N's to N, the same order. A
lower number is preferred and ties go to the lower index. gst spells it on the
request pad (`sink_1::priority`), which a `&'static` property table has no room
for, so g2g flattens it into the element's own table for inputs 0 to 7, the
convention `compositor`'s `sinkN-xpos` already uses. A switch with more inputs
sets the rest at construction. The best-priority input is also the one the
startup hold waits for, where gst tests literally for priority 0 and so holds the
output for a whole timeout when every pad was given a nonzero priority.

An input is healthy while it delivered a `DataFrame` within `timeout` plus
`latency` nanoseconds. `latency` is the slack an upstream running late is
allowed and what the element reports to the pipeline latency query, live, so
downstream buffers it. The lowest-index healthy input is forwarded, frames on
the rest are dropped, and when none is healthy the current input keeps the
output rather than blanking it. Health is re-checked on every arriving packet
and on the tick the element declares, `tick_interval_ns` being the timeout, so a
primary going silent is noticed while the other inputs are quiet too.

- `immediate-fallback=false` holds a lower-priority frame until the primary has
  had one stall window from the first frame the element saw
- `auto-switch=false` hands the choice back to `active-pad`
- `stop-on-eos` ends forwarding when any input ends
- `min-upstream-latency` floors what the fold reports for the branches feeding
  the switch, which is how a run makes room for an input slower than the ones it
  negotiated at startup. gst carries it on the aggregator base class and g2g on
  `MultiInputElement`, so any fan-in can raise it.

A switch re-announces the new input's caps downstream when they differ from the
last ones emitted, since the branches negotiate independently. The element is
`std`-only, because the health rule measures against the process monotonic
clock.

### fallbacksrc

`fallbacksrc uri=X` is the launch macro over the switch, flattened at parse time
like `uridecodebin`: the URI's source auto-plugged to raw on input 0, and on
input 1 either `fallback-uri`'s own decode chain or a dummy generator,
`videotestsrc pattern=black` or `audiotestsrc wave=silence` behind a
`clocksync`. Neither test source is live, so without the pacer the dummy would
run as fast as the CPU allows. `timeout` and `immediate-fallback` pass through to
the switch, and a `name=` names the switch, so a line can hang a further branch
off it by pad reference. Inline, the expansion is single-stream, as
`uridecodebin`'s is: `enable-video` and `enable-audio` say which kind the decode
chain may reach, video first. The dummy carries its own geometry rather than the
main stream's, which is not known until the main branch negotiates, and the
switch's caps re-announcement is what carries that difference downstream.

### Multi-stream fan-out

A `fallbacksrc uri=X` that is the whole pipeline instead carries every kind the
container holds, the way a lone `playbin` fans out. The split is the same one
`playbin` draws and for the same reason: the multi-stream form has an output per
kind and a launch line has no way to name a second one, so only the form with
nothing downstream of it can take it.

The URI is probed by a `Registry::register_uri_fanout` hook, the open-ported
sibling of the `playbin` hook. Where that returns a graph already closed on its
own sinks, this returns the byte source, its rebuild, the multi-output demuxer,
and each port's caps and `StreamType`, because a `fallbackswitch` has to sit
between every port and its sink. What produces the ports is either a restartable
byte source feeding one demuxer or, for a session protocol, one restartable
multi-output source with nothing after it: an RTSP stream's video and audio
tracks come off one `DESCRIBE`, so there is no byte stream to demux.

Each port gets its own decode chain into input 0 of its own switch, its own
fallback on input 1, and its own automatic sink: `autovideosink`, or
`audioconvert ! audioresample ! autoaudiosink`, the tail that fixes one PCM
format for the sink while the converters absorb the stream's real channels and
rate. A `fallback-uri` is probed by the same hooks and its matching port feeds
each switch. A kind the fallback does not carry falls back to the dummy
generator.

The switches are named after the keyword with the kind appended, so
`fallbacksrc name=fb` gives `fb-video` and `fb-audio`, and the sources take the
suffixes on the keyword's own name. A kind may repeat: a container carrying two
audio tracks gets two audio branches, and the second and later port of a kind
takes its ordinal too (`fb-audio-1`), so a one-track container's names are
unchanged. Each main port pairs with the fallback port of the same ordinal, so
the second audio track backs the second audio switch rather than every audio
switch backing onto the first.

Hooks are registered for Matroska, MPEG-TS, ISO-BMFF, MPEG program streams, HLS,
RTSP and Ogg, sharing each container's probe with its `playbin` hook. Ogg is
what needs the repeated kind, since every Ogg mapping g2g reads is audio, so a
grouped file's ports are all audio ports. An HLS variant fans out through the
demuxer its packaging needs, `TsDemuxN` for muxed MPEG-TS segments and
`Mp4DemuxN` for fMP4 and CMAF, with the tracks read from the `#EXT-X-MAP` init
segment. A rendition with its own playlist gets no port, because one fan-out has
one source, so a separate-audio variant reports its video alone and the line
falls back to single-stream. A hook declines a container it does not parse, and
the fan-out declines one whose ports are fewer than two or include a kind it has
no generator and no sink for, so either way the line falls back to the
single-stream expansion plus one automatic sink.

A demuxed head needs nothing new for restart: the demuxer downstream of a
rebuilt byte source copes with the rebuilt stream, so it is the existing
`RestartSrc` wrap with a demuxer after it. The rebuild comes from the hook
rather than from `Registry::uri_source_rebuilder`, because the container's byte
source is not what the URI's scheme handler builds, and for `file://` that
handler self-demuxes MP4. A session head takes `RestartFanoutSrc` instead, the
multi-output sibling of `RestartSrc`: the same policy over a life that pushes to
several ports, with one shared timeline offset for all of them, so a rebuilt
session keeps the alignment its tracks had. The two wrappers share the policy
state, meaning the timeline offset, the retry tally and the budget arithmetic,
and the timestamp-shift rule. What differs is that there is no per-port
configure to repeat, since a multi-output source answers `output_caps` itself,
and the terminal EOS goes to every port. The port count and each port's caps are
read once from the first session and answered from there, because between a
death and the next life there is no inner session to ask.

### Keeping a source alive

Both the `uri=` and the `fallback-uri=` source are wrapped in `RestartSrc`
(`g2g-plugins/src/fallbacksrc.rs`), a source that runs the inner one and rebuilds
it from the URI when it fails, delivers nothing for `restart-timeout` (5 s by
default, 0 disables the check), or ends under `restart-on-eos`. Both sources take
the same policy, where gst loops its fallback source on EOS unconditionally, so a
line over two files still runs to EOS unless it asks for the loop.

Each life is stitched onto one timeline through the `ShiftSink` adapter shared
with `gaplesssrc`, so a rebuilt file source that starts again at PTS 0 continues
where the last life stopped, and the inner `Eos` packets are swallowed. A fixed
one-second pause separates a death from its rebuild, gst's hardcoded sleep.
`retry-timeout` (60 s) is the budget for repeated failure, counted from the first
failure after the last delivered frame: a rebuild starts only while it would
begin inside it, and once it is spent the wrapper ends its stream, handing the
switch to the fallback for good. gst stores `retry-timeout` but never arms its
timer, so this is the property's plain reading rather than its upstream
behaviour.

The wrapper is installed through the registry's `RestartSourceHook`, with
`Registry::uri_source_rebuilder` handing it a closure that builds a fresh source
for the URI. Core owns the policy and the hook type and the plugin crate owns the
timer-driven wrapper, the same split as the `playbin` hooks. Without a registered
hook the keyword runs its sources unwrapped. The rebuilt source is negotiated by
the wrapper and configured with the caps the runner gave the wrapper at startup,
since a source arm cannot renegotiate the decode chain below it mid-run, and a
source that refuses them counts as a failed attempt.

`manual-unblock=true` holds every life, the first included, until the application
releases it, where gst blocks the restarted source's pads. gst releases through
an `unblock` action signal. g2g has no signals, so the release is an
`UnblockHandle` (`g2g-core/src/runtime/unblock.rs`), the app-holds and
source-holds-a-clone shape `GaplessController` already uses, registered on the
`Registry` and passed to the wrapper by the expansion. One `unblock` frees one
life and the handle re-arms, so an application releases again after every
restart. A held life delivers nothing and does not report itself `Running`. Both
sources take the handle, since either can be the one that restarted.
`manual-unblock=true` without a registered handle is a parse error rather than a
pipeline nothing could ever release.

### Restart status on the bus

What gst exposes as `fallbacksrc`'s read-only `status` and `statistics`
properties, g2g posts on the bus as `BusMessage::SourceRestart`, because a source
arm owns its element for the whole run and nothing can read a property off it
mid-run. The message carries the status, the rebuild tally gst calls `num-retry`,
and the reason the last life ended.

- status: `Running` as each life starts, `Retrying` once a rebuild is decided
  and before the one-second pause, `Stopped` at a clean end or a spent retry
  budget
- reason: `Error`, `Eos`, `Timeout` for the stall, and g2g's `Rebuild` and
  `Negotiate` where gst folds both into `StateChangeFailure`

gst's fourth status, `buffering`, has no analog, since there is no buffering
stage on this path. The main and the fallback source, which gst separates by
carrying `num-retry` and `num-fallback-retry` side by side, are told apart by the
message's `FallbackSourceRole`. The source's instance name rides along as a label
for logs rather than as the discriminator, since a launch line's `name=` chooses
it. A `RestartSrc` built directly from Rust has no role. The runner hands the bus
to a source through `SourceLoop::set_bus`, which only the graph runner calls.

The expansion also names the sources it builds off the `fallbacksrc`'s own name:
`fallbacksrc name=fb` yields `fb-source` and, with a `fallback-uri=`,
`fb-fallback-source`, so a name is what `GraphMutator::replace_source` addresses
to swap either one during a run. The generated names collide with a line's own
`name=` the way the generated switch name does, reported as
`ParseError::DuplicateName`. The other source macros, `uridecodebin` and
`playbin`, leave their nodes unnamed.

## livesync

`livesync` has one input and is still a fan-in, because
`MultiInputElement::tick_interval_ns` is the only way an element is called with
no packet arriving, and a stalled input is exactly that case. So `parse_launch`
builds a name registered only as a muxer into a fan-in at link degree one
(`is_muxer` in `g2g-core/src/runtime/launch.rs`), while a name registered both
ways, `mp4mux` and `textoverlay`, keeps its single-input element there.

The tick period is one output buffer, the frame period of the negotiated
`Caps::RawVideo` framerate or the last audio buffer's duration, and
`output_follows_input` makes the output caps the input's. On a tick past a
buffer's due time plus `latency` the element emits a filler at the next PTS and
counts it in `duplicate`: for video the last frame again, refcounted rather than
copied, and for PCM audio a silence run the size and span of the last buffer.

A frame behind the timeline already emitted is dropped and counted in `drop`,
unless it is more than `late-threshold` behind, in which case the output timeline
follows it, so an upstream that restarts its clock recovers instead of being
dropped forever. Under `single-segment` that frame is stamped at the slot the
output timeline expects next and every later one is shifted by the same offset,
so the output PTS never goes backwards. Under `sync` a buffer that arrives before
its due time is held and goes out on the first tick at or past it. It defaults to
false here, unlike gst, because the arm's tick period is fixed, so a held buffer
leaves up to one period late.

## togglerecord

`togglerecord` is an N-in N-out shape: each input has its own output, and the
streams share only a decision. There is no node kind for that, so it is several
1:1 `AsyncElement`s sharing a `RecordGroup`, flattened at parse time the way
every other g2g bin is. From Rust the group is an `Arc<RecordGroup>`.

GStreamer's spelling parses here: `togglerecord name=t` plus `t.sink_K` and
`t.src_K` request pad pairs. The inline keyword is stream 0, the main stream, and
each pad pair becomes a second element in the same group, named `t-K`. The group
is the keyword's own `name=` unless the line set `group=`. A pad with no partner
is a parse error, since it would leave a stream with one end dangling.

A `group=` name is scoped to one `parse_launch` call: the launch layer marks the
parse a factory is building for, so two pipelines in one process that pick the
same name get separate groups. Looking the name up from outside any parse reaches
the earliest parse still holding it, which is how an application toggles a text
pipeline once the run owns its elements.

The group holds the `record` flag, the `Stopped` / `Starting` / `Recording` /
`Stopping` state, the main stream's last frame end, and the keyframe-aligned
`[start, stop)` spans the main stream decided. Exactly one member declares
`main=true`. Its next keyframe opens a span when `record` goes true and closes
one when `record` goes false, so a recording is a whole number of GOPs. A
secondary looks its frame up in the spans, and parks on the group's notify until
the main stream's position has passed that frame's end, so it never decides ahead
of the main.

With `is-live=false` every member subtracts the same quantity, the recorded time
before its own timestamp, so the not-recording gaps vanish and the streams stay
aligned across a pause. `is-live=true` passes timestamps through.

Two departures from GStreamer. GStreamer blocks a non-live upstream while paused
where g2g drops, because a g2g source keeps producing. And GStreamer rejects any
delta frame on a secondary pad where g2g rejects one only when that pad's caps
are compressed video, since a raw or compressed-audio frame sets no `keyframe`
flag and can be cut anywhere.
