# Remote desktop

Plan for a remote-desktop session on g2g. Nothing in this file is built unless
it names an element that already exists. When a slice ships, delete it here and
from [TODO.md](TODO.md), and write the shipped behavior into the track that
owns it: capture in [live.md](live.md), the session channel in
[transports.md](transports.md). This file becomes the track document for the
session only once a host and a client run.

The session lets a person on one machine see and drive the desktop of another,
with the picture free to drop frames and the input required to arrive.

## Done

A host and a client for Linux, Windows, and macOS, plus a Chrome page.

- One display, chosen by index. Several displays come after the first session runs.
- H.264 from `nvenc` on an NVIDIA GPU, from `ffmpegenc` VA-API
  on a Linux drm buffer, from `vtencode` on macOS, from `mfencode` on Windows,
  and from `ffmpegenc` software where none of those open.
- Opus audio from the host's default capture device.
- Keyboard, mouse (relative and absolute), wheel, cursor bitmap back to the
  client, clipboard text, gamepad and rumble on Linux and Windows.
- A bearer token on the message channel. OIDC mints that token in a later slice.
- The client shows the newest frame. Audio uses a short queue and does not hold the picture.
- A Windows service whose only job is input injection on the secure desktop.
  Capture stays in the interactive user session.
- Packages that install the host and the client. No media-library build.

Out of this plan: phones, headsets, USB forwarding, a graphical client, file
transfer, AV1 as a default, and a relay. A direct QUIC connection covers a
LAN, and a host off the LAN needs a forwarded UDP port.

## Decisions

1. The session runs over one QUIC connection, not WebRTC. Native clients dial
   raw QUIC (`quic://`, a g2g ALPN), the browser dials WebTransport
   (`https://`), and the host accepts both ALPNs on one UDP port. The
   transport is a mode of the `remotewt` elements, so any g2g pipeline can use
   it: `remotewtsink` on the host, `remotewtsrc` on the client.
2. Video goes either one frame per unidirectional stream or as datagrams with
   Reed-Solomon repair (`reed-solomon-simd`). Input and the token ride a
   reliable message channel the `remotewt` elements open on the same
   connection, so a lost picture does not sit in front of a key release.
   Keyframe requests go from `remotewtsrc` to `remotewtsink` on the video
   stream's return direction and reach the encoder as a downstream
   `ForceKeyframe`, which every encoder already handles.
3. The client dials the host's address. There is no ICE, so a host behind NAT
   needs a forwarded port until a relay slice. The host makes a self-signed
   ECDSA P-256 certificate valid under 14 days (what WebTransport's
   `serverCertificateHashes` accepts) and prints its SHA-256 hash, which the
   client pins with `--cert-hash`.
4. The first slice encodes whatever memory the capture element already
   produces. In-place GPU encode is the next slice on each OS, and the session
   runs before it lands. `nvenc` takes `MemoryDomain::Cuda`, and a D3D11 NV12
   texture on Windows. `mfencode hardware=true` takes a D3D11 NV12 texture.
   `ffmpegenc` VA-API takes a drm buffer. `vtencode` takes the
   `screencapturesrc` `cv-output` path.
5. The client's video sink presents each frame as it is decoded, with
   `sync=false` and `present-mode=mailbox`. No `livesync`: receivers stamp
   frames with their arrival time, so network jitter puts frames closer than
   one frame period and `livesync` drops them as early.
6. Input types live in the `g2g-desktop` crate, not in `g2g-core`. Capture
   elements live in `g2g-plugins` behind their OS feature. `g2g-desktop` is a
   workspace member, uses `std`, and is not on the 1.86 MSRV line.
7. A resolution change is a `CapsChanged` the encoders already rebuild on
   (`mfencode` drains and reopens). The host rebuilds the graph only when the
   capture source itself dies, which DXGI does on a desktop switch.
8. Wayland capture is `pipewirevideosrc portal=true` with
   `portal-restore-token`, so a later start reuses the grant. Unattended X11
   gets its own source. The portal dialog is the wrong tool for a service.
9. The bearer token is the first message on the message channel and is
   compared in constant time before capture starts. A bad token closes the
   connection. OIDC, a URL scheme, and installers do not
   block the first LAN session.
10. A network that blocks UDP falls back to WebSocket over TLS on the same
    port number over TCP. The client dials `quic://` and dials `wss://` when
    the QUIC handshake has not finished after 1 s. The `wss://` location is a
    third scheme of the `remotewt` elements, so the message channel and
    keyframe requests are the same code on both transports.

## What the session builds on

| Need | Element |
| :--- | :--- |
| Wayland screen | `pipewirevideosrc` `portal=true`, `portal-restore-token`, `portal-timeout`, `portal-cursor-mode=metadata` for `cursor_shapes()`. `io-mode=dmabuf` or `mmap`. |
| macOS screen | `screencapturesrc`. `cv-output` keeps the IOSurface. It captures the first display until `display-index` exists. |
| Windows screen | `d3d11screencapturesrc`, `monitor-index`, `fps`, BGRA in system memory or a D3D11 texture. |
| Windows GPU colour convert | `d3d11convert`, BGRA D3D11 texture to NV12 on the video processor. |
| NVIDIA encode | `nvenc`. CUDA NV12, or a D3D11 NV12 texture on Windows, in, Annex-B out, no B-frames, `gop-size=-1` by default, `force_keyframe()` latched and set on the first frame. |
| Upload into CUDA | `cudaupload`, Linux and Windows. `dmabuftocuda` takes RGBA dma-bufs of any modifier on Linux. |
| VA-API encode | `ffmpegenc` backend `h264_vaapi` / `hevc_vaapi`. |
| macOS encode | `vtencode`. |
| Windows encode | `mfencode`, `MF_LOW_LATENCY` so the MFT releases one output per input. `hardware=true` reads D3D11 NV12 textures in place. |
| Software encode | `ffmpegenc`. |
| Audio capture | `pulsesrc`, `wasapisrc`, `coreaudiosrc`. |
| Audio encode | `opusenc`. |
| Present | `autovideosink`, and directly `waylandsink`, `metalvideosink`, `d3d11sink`. |
| Present without queueing | `wgpusink sync=false present-mode=mailbox`. |
| Browser decode and draw | `g2g-web` WebCodecs and `canvassink`. The page needs the same data channel the native client uses. |
| QUIC carrier | `remotewtsink`, `remotewtsrc` over `web-transport-quinn` (quinn 0.11). |

## QUIC session

- Encoder bitrate stays fixed until a later slice feeds it the bandwidth
  estimate.
- Stream-per-frame mode: a frame the client has not finished when a newer
  keyframe is queued is reset, since nothing after the keyframe needs it.
- The host destroys the virtual device a removal packet names.

## WebSocket fallback

`remotewtsink mode=listener` accepts a WebSocket over TLS on the same port
number when UDP is blocked, and `g2g-desktop client` dials it when QUIC has not
connected after 1 s. The carrier is described under the `remotewt` elements in
[transports.md](transports.md). A browser WebSocket accepts no pinned hash, so
the Chrome page's fallback needs a CA-signed certificate or plain `ws://` on a
LAN.

## Host graphs

The host sends video through `remotewtsink` in listening mode, and Opus audio
on the sink's track (`open_track`). A second display is a second track on the same connection.

Linux, NVIDIA, the fast path once `cudaupload` accepts the capture format:

```
pipewirevideosrc portal=true io-mode=mmap ! videoconvert ! cudaupload ! nvenc
```

Linux without CUDA:

```
pipewirevideosrc portal=true ! videoconvert ! ffmpegenc
```

or `ffmpegenc` with the VA-API backend when `io-mode=dmabuf` yields a buffer
that backend accepts. If it does not, the software line is the one that ships.

The Linux host sends what the desktop plays, the monitor of the default output,
unless `--audio false`:

```
pulsesrc device=@DEFAULT_MONITOR@ ! audioconvert ! opusenc frame-size=10 audio-type=restricted-lowdelay
```

`--run` sends no audio, since its app plays through the host's audio server
along with every other app.

The Linux host reads `portal-restore-token` from
`$XDG_STATE_HOME/g2g-desktop/portal-restore-token` before each connection and
writes each token `PipeWireVideoSrc::portal_restore_tokens` publishes back to
that file. Portal tokens are single-use, so the file always holds the newest one.

With `--run COMMAND` the Linux host captures a headless desktop instead of the
screen. At launch the host starts a `WaylandDisplay` with Xwayland at `--size`
(default 1920x1080) on `--render-node`, runs `sh -c COMMAND` with
`WAYLAND_DISPLAY` and `DISPLAY` naming it, and exits when the command exits.
Each connection attaches a `WaylandDisplaySrc::on_display` source to that
display, so the app and its windows stay across reconnects. A source that
detaches releases every key and button its session left held. The cursor the
focused client sets goes to the client as type 8, a named cursor comes from the
`XCURSOR_THEME` theme at `XCURSOR_SIZE`. The encoder follows the render node's
kernel driver:

```
waylanddisplaysrc ! dmabuftocuda ! nvenc      # nvidia
waylanddisplaysrc ! vah264enc                 # amdgpu, i915, xe
waylanddisplaysrc ! videoconvert ! ffmpegenc  # any other driver, or software rendering
```

The dma-bufs reach `vah264enc` with no copy and `nvenc` after one device copy.
Client input goes into the compositor through `WaylandDisplaySrc::input_sender`
rather than uinput.

With `--x11-display NAME` the Linux host captures that X display, for an
unattended X11 session where the portal dialog has nobody to answer it:

```
ximagesrc display-name=NAME show-pointer=false ! video/x-raw,framerate=FPS/1 ! videoconvert ! ffmpegenc
```

`XImageSrc::cursor_shapes` sends the XFixes pointer image beside the picture,
and client input goes to `XTestInjector`, which fakes it through the XTEST
extension on the root window. Keys take the evdev code plus 8, buttons and wheel
notches take the core button numbers (wheel up, down, left, right are 4 to 7),
and a keyboard removal releases every key it pressed. `--run` and
`--x11-display` are exclusive.

Windows, every frame on the GPU from capture to the encoder:

```
d3d11screencapturesrc ! d3d11convert ! mfh264enc hardware=true
wasapisrc ! audioconvert ! opusenc
```

Windows with NVIDIA:

```
d3d11screencapturesrc ! d3d11convert ! nvenc
```

`nvenc` opens its session on the texture's `ID3D11Device` and registers the
texture itself, so the frame never passes through CUDA.

macOS:

```
screencapturesrc cv-output=true display-index=0 ! vtencode
coreaudiosrc ! audioconvert ! opusenc
```

`screencapturesrc` today takes the first display and has no `display-index`.
That property is part of this plan.

The first frame is an IDR, which `nvenc` already does.

## Client graphs

The client dials with `remotewtsrc` and sends no media. Receive side:

```
h264parse ! <decoder> ! wgpusink sync=false present-mode=mailbox
opusdec ! audioconvert ! pulsesink buffer-time=40000
```

`pulsesink` starts playing once 40 ms of audio is queued. The client's audio
graph runs beside the video graph on the same runtime. The host's runs on its
own thread: on the video graph's runtime it adds 1 ms p50 and 3 ms p99 to each
video frame.

The decoder is `nvdec`, `mfdecode`, `vtdecode`, or `ffmpegdec`, chosen by the
same auto-plug `decodebin` uses. `software-decode=true` forces `ffmpegdec`.
The Linux client decodes with `ffmpegdec` and the Windows client with
`d3d12h264dec`, whose textures `wgpusink` presents on the decoder's device.

On exit the client prints glass-to-glass latency: the time from the host's
capture to the present, from the frame's arrival stamp. Encoders keep that stamp
by pts, and `remotewt` shifts it by half the QUIC round trip. The mean and max
are exact, and p50 and p99 read at most 1/16 high.

The browser client is the same WebTransport session and message channel, decoded with
WebCodecs into a canvas. Pointer lock and `navigator.keyboard.lock` are the
capture side. Chrome is the target. Firefox stays untested until the native
session is solid.

## Capture sources to add

### nvenc on the Windows host

The host encodes with `d3d11convert ! nvenc` on an NVIDIA GPU and falls back to
`mfh264enc hardware=true`. Capture runs in the interactive session, never in
session 0, and the host rebuilds the graph after `AccessLost` fails the source.

### display-index elsewhere

`screencapturesrc` gains `display-index` and stops hard-coding the first
display. The portal path records the restore token the user already granted.
Picking a monitor the portal did not grant is a failed start, not a guess.

## Input edges

Each edge is a small platform file in `g2g-desktop`. The packet and the router
stay shared and are tested without a desktop.

| Edge | Capture | Inject |
| :--- | :--- | :--- |
| Linux | the client window's keyboard and pointer, with the desktop's shortcuts inhibited while grabbed | `/dev/uinput`, which Fedora creates as root-only `0600`, so the host needs a udev rule that gives the `input` group write access. |
| Windows | the client window's key and mouse messages, plus Raw Input for grab | `SendInput` in the interactive session. The service injects when the interactive process cannot, which is the secure desktop. |
| macOS | the client window | `CGEvent`, which needs the accessibility permission. Screen recording stays the permission `screencapturesrc` already needs. |

Relative move is `EV_REL` on Linux and `MOUSEEVENTF_MOVE` on Windows.
Absolute position is the virtual desktop: `x` and `y` in pixels of
`display`, origin at that display's top left. The host maps `display` through
the same index the capture source uses.

Gamepad capture uses one crate on every client. Injection is a virtual uinput
gamepad on Linux and ViGEm on Windows. macOS injection waits until a virtual
HID device is chosen. The packet still parses there, and the router counts it
as undelivered.

Cursor: the host sends type 8 each time the pointer shape or its visibility
changes, starting with the first captured frame, and a 0x0 shape while the
pointer is hidden. `d3d11screencapturesrc` reads the shape from DXGI and
publishes it on `cursor_shapes()`. Windows inverts the screen under some pixels
of monochrome and masked-color pointers, and those pixels arrive black. The
Wayland client sets the bitmap as its window's pointer cursor through
`WgpuPresentSink::with_cursor_shapes`, so the pointer moves at the
compositor's rate with no video delay. Grab is pointer lock on the web client
and a window grab on the native client. On Linux the window holds a
`zwp_keyboard_shortcuts_inhibitor_v1` while it has keyboard focus, so Super and
Alt+Tab reach the host, and a click takes the grab again after Escape. A
compositor without the protocol still gets the Escape and click handling. Escape releases the grab locally and is not
forwarded.

Clipboard: each side watches its clipboard and sends type 9 when the text
changes and the other side did not just set it. A payload over the 64 KiB
message cap is dropped. The `--run` host sends what `WaylandDisplay::clipboard_texts`
publishes and hands received text to `WaylandDisplay::set_clipboard`. The Linux
client passes `WgpuPresentSink::sync_clipboard` the host's texts, which its
window sets as the selection once it has a keyboard serial, and forwards each
selection the desktop sends the focused window. Both sides remember the last
text that crossed, so neither echoes it.

## Windows service

The service process is installed separately, runs as SYSTEM, and accepts
injection packets on a named pipe from the interactive host. It does not
capture, encode, or speak QUIC. The interactive host captures. When
`SendInput` from the interactive process fails because the secure desktop is
up, the host forwards the same packets down the pipe. The service is optional
at build time and required for the UAC slice.

## g2g-desktop

Workspace member. Two binaries, or one binary with two subcommands: `host` and
`client`.

Host flags, all of them real properties the process applies:

| flag | meaning | default |
| :--- | :--- | :--- |
| `--port` | UDP port for QUIC and WebTransport, TCP port for the WebSocket fallback | 8080 |
| `--token` | bearer token | required |
| `--monitor-index` | which monitor to capture, Windows | -1 |
| `--fps` | capture rate | 60 |
| `--bitrate` | bits per second, passed to the encoder | 4000000 |
| `--audio` | capture and send audio | true |
| `--video-mode` | `stream`, `stream-per-frame` or `datagrams` | `stream` |
| `--x11-display` | capture this X display and inject through XTest, Linux | none |
| `--congestion-control` | `low-latency` (BBR), `throughput` or `default` (Cubic) | `low-latency` |

Client flags:

| flag | meaning | default |
| :--- | :--- | :--- |
| `--host` | host address | required |
| `--port` | host UDP and TCP port | 8080 |
| `--transport` | `auto` (QUIC, then `wss://` after 1 s), `quic` or `websocket` | `auto` |
| `--cert-hash` | SHA-256 of the host certificate | required |
| `--token` | bearer token | required |
| `--software-decode` | force `ffmpegdec` | false |
| `--audio` | play audio | true |
| `--inputs` | send input | true |
| `--clipboard` | sync clipboard | true |
| `--keyboard-grab` | grab on focus | true |

A URL `g2g://host:port/?token=...` parses onto those flags. The query keys
match the flag names with underscores. Unknown keys are an error and the
client exits without connecting. `token` in a URL is visible to other
processes of the same user. The slice that adds the URL also documents that,
and OIDC replaces a long-lived token with a short one.

## OIDC and packages

OIDC is a front end that mints the bearer token and hands it to the client.
The host still only knows the token. The refresh loop lives in the client
process that speaks to the identity provider, and a failed refresh closes the
session.

Packages install `g2g-desktop`, the URL-scheme handler, and on Windows the
service. They do not build ffmpeg or a media player.

## Tests

Hardware checks are probes, the same shape as `m739_screencapture`. A denied
capture permission asserts the error path. A granted desktop is manual: move
the pointer, hold a key, confirm the key-up arrives.

## Order

Each slice builds and is reviewable on its own. Later slices depend on earlier
ones.

3. **Linux host and Linux client on one machine.** Portal or `ximagesrc`,
   software or VA-API encode, window capture. `inputs=false` when both run on the
   same seat, or the pointer feeds itself.
4. **Windows host, then the QUIC transport.** `d3d11screencapturesrc`,
   `mfencode`, `SendInput`. Then FEC datagram video in `remotewt`. Then the
   WebSocket fallback. Then the Windows
   client, then D3D11 input on `nvenc`.
5. **macOS client, then the macOS host** on `screencapturesrc` `display-index`
   and `vtencode`.
6. **Audio, clipboard, cursor, grab.**
7. **Gamepad** on Linux and Windows, and the Windows service.
8. **Chrome page** on the same WebTransport session and message channel.
9. **OIDC, the URL scheme, packages.**

A LAN session that moves a pointer is slice 3. Matching the done list above is
the end of slice 9.
