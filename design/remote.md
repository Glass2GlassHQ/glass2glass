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
- H.264 from `nvenc` where a CUDA surface is available, from `ffmpegenc` VA-API
  on a Linux drm buffer, from `vtencode` on macOS, from `mfencode` on Windows,
  and from `ffmpegenc` software where none of those open.
- Opus audio from the host's default capture device.
- Keyboard, mouse (relative and absolute), wheel, cursor bitmap back to the
  client, clipboard text, gamepad and rumble on Linux and Windows.
- A bearer token on the signaling connection. OIDC mints that token in a later slice.
- The client shows the newest frame. Audio uses a short queue and does not hold the picture.
- A Windows service whose only job is input injection on the secure desktop.
  Capture stays in the interactive user session.
- Packages that install the host and the client. No media-library build.

Out of this plan: phones, headsets, USB forwarding, a graphical client, file
transfer, AV1 as a default, and a relay. ICE host candidates cover a LAN.
`stun-server` and `turn-server` on the duplex session are forwarded for a host
that is not on the same network.

## Decisions

1. One `WebRtcDuplexSession` per session. It already publishes and receives on
   one `Rtc`, with host ICE candidates in the SDP, PLI and bandwidth estimate
   on a reverse channel, and `stun-server` / `turn-server` properties.
   `webrtcsink` is WHIP: the viewer would be a second peer, and a data channel
   cannot cross that media server. `WebRtcDataSink` opens its own `Rtc` and its
   own ICE gathering, so a second one would be a second connection.
2. Input rides a reliable, ordered data channel on that same `Rtc`, label
   `input`. Video stays RTP, so a lost picture does not sit in front of a key
   release. str0m caps one data-channel message at 64 KiB
   (`webrtcdata::MAX_MESSAGE_SIZE`). A payload over that is dropped.
3. The client is the offerer. It connects out. The host is the answerer and
   listens. `SdpChannel::from_halves` is the seam: the desktop crate moves the
   SDP strings across one TCP connection, and the duplex session never learns
   about that socket.
4. The first slice encodes whatever memory the capture element already
   produces. In-place GPU encode is the next slice on each OS, and the session
   runs before it lands. `nvenc` rejects anything other than
   `MemoryDomain::Cuda`. `mfencode` takes tightly packed system NV12.
   `ffmpegenc` VA-API takes a drm buffer. `vtencode` takes the
   `screencapturesrc` `cv-output` path.
5. The client puts `livesync` in front of the video sink. A frame that arrives
   behind the timeline is dropped. A frame later than `late-threshold` rebases
   the timeline instead of being dropped forever. Audio does not gate that.
6. Input types live in the `g2g-desktop` crate, not in `g2g-core`. Capture
   elements live in `g2g-plugins` behind their OS feature. `g2g-desktop` is a
   workspace member, uses `std`, and is not on the 1.86 MSRV line.
7. A resolution change is a `CapsChanged` the encoders already rebuild on
   (`mfencode` drains and reopens). The host rebuilds the graph only when the
   capture source itself dies, which DXGI does on a desktop switch.
8. Wayland capture is `pipewirevideosrc portal=true` with
   `portal-restore-token`, so a later start reuses the grant. Unattended X11
   gets its own source. The portal dialog is the wrong tool for a service.
9. The bearer token is compared in constant time before any SDP is read. A bad
   token closes the TCP connection. OIDC, a URL scheme, and installers do not
   block the first LAN session.

## What the session builds on

| Need | Element |
| :--- | :--- |
| Wayland screen | `pipewirevideosrc` `portal=true`, `portal-restore-token`, `portal-timeout`. `io-mode=dmabuf` or `mmap`. |
| macOS screen | `screencapturesrc`. `cv-output` keeps the IOSurface. It captures the first display until `display-index` exists. |
| Windows camera, not the screen | `mfvideosrc`. There is no desktop-duplication source yet. |
| NVIDIA encode | `nvenc`. CUDA NV12 in, Annex-B out, no B-frames, `gop-size=-1` by default, `force_keyframe()` latched and set on the first frame. |
| Upload into CUDA | `cudaupload`, Linux and Windows. |
| VA-API encode | `ffmpegenc` backend `h264_vaapi` / `hevc_vaapi`. |
| macOS encode | `vtencode`. |
| Windows encode | `mfencode`, system NV12, `MF_LOW_LATENCY` so the MFT releases one output per input. |
| Software encode | `ffmpegenc`. |
| Audio capture | `pulsesrc`, `wasapisrc`, `coreaudiosrc`. |
| Audio encode | `opusenc`. |
| Picture | `WebRtcDuplexSession`, `run_duplex_session`. Reverse-channel PLI is wired to `NvEnc::force_keyframe`. |
| Present | `autovideosink`, and directly `waylandsink`, `metalvideosink`, `d3d11sink`. |
| Drop late frames | `livesync`. |
| Browser decode and draw | `g2g-web` WebCodecs and `canvassink`. The page needs the same data channel the native client uses. |
| Signaling seam | `SdpChannel`. |

## Signaling

One TCP connection, default port 8080. The client sends first.

```
u32be length | payload
```

1. Client sends the token, UTF-8, at most 256 bytes.
2. Host compares it and closes on a mismatch. Then it sends one byte, `0x01`.
3. Client sends the SDP offer.
4. Host sends the SDP answer.

Both SDP blobs are the strings `SdpChannel` already exchanges. ICE host
candidates are inside them, which is how the duplex session connects on one
machine today. Trickle ICE is a later slice: further messages with a leading
`u8` tag `1` for an ICE candidate and `2` for the SDP, once a peer off the LAN
needs them. Until then a message after the answer is a protocol error and
closes the connection.

The host binds the TCP port before it builds the graph. The graph starts when
the answer is sent, and stops when the TCP connection or the ICE state drops.
One client at a time. A second TCP connection is refused while the first is up.

## Data channel

Added inside `WebRtcDuplexSession::run`, on the `Rtc` that session already
owns, created by the offerer with `ChannelConfig` label `input`, `ordered:
true`, `Reliability::Reliable`. Bytes in and out are a pair of channels the
desktop crate holds. The duplex session gains no knowledge of keycodes.

Every message is one packet:

```
u8  type
u8  flags
u32le sequence
u16le payload_len
payload[payload_len]
```

`sequence` is per direction, starting at 0. The receiver drops a packet whose
sequence is older than the newest applied packet of that type, which is how a
stale absolute mouse position loses to a newer one. Reliable and ordered
delivery makes that rare. The rule still holds for a later unreliable mode.

| type | name | payload |
| :--- | :--- | :--- |
| 1 | key | `u8 device`, `u16le hid`, `u8 down` |
| 2 | button | `u8 button`, `u8 down` |
| 3 | move | `i32le dx`, `i32le dy` |
| 4 | position | `i32le x`, `i32le y`, `u8 display` |
| 5 | wheel | `i32le dx`, `i32le dy` |
| 6 | gamepad | `u8 device`, `u32le buttons`, six `i16le` axes (`lx ly rx ry lt rt`) |
| 7 | rumble | `u8 device`, `u16le low`, `u16le high`, `u16le duration_ms` |
| 8 | cursor | `u16le w`, `u16le h`, `i16le hot_x`, `i16le hot_y`, RGBA bytes `w*h*4` |
| 9 | clipboard | UTF-8 |

`hid` is the USB HID usage, not an OS keycode. Each platform maps at its edge.
Button 0 is left, 1 is right, 2 is middle. `flags` bit 0 means the packet is
from the host. Cursor and rumble set it. A cursor whose RGBA length is not
`w*h*4`, or whose encoded size exceeds 64 KiB, is dropped.

## Router

One router on each side. Producers push packets in. Consumers subscribe by
type. The network is both: it consumes local packets and produces remote ones.

The client router drops a packet before it is sent when the video window has
no focus, when a local shortcut matches, or when `inputs=false`. A swallowed
shortcut is not forwarded. The host router delivers every packet it is willing
to inject. `clipboard=false` unsubscribes both directions from type 9.

Devices of one type share a producer. A gamepad packet carries `device`.
Hot-plug is a packet with `flags` bit 1 set and an empty payload, meaning the
device is gone. The host destroys that virtual device.

## Host graphs

The host offers two send tracks, video then audio, and opens no receive media
track. `WebRtcDuplexSession::new(answerer, channel, 2)` matches a client that
offers two sendrecv m-lines. Confirm the pad count against `with_spare_tracks`
when the second display arrives. One display needs no spare.

Linux, NVIDIA, the fast path once `cudaupload` accepts the capture format:

```
pipewirevideosrc portal=true io-mode=mmap ! videoconvert ! cudaupload ! nvenc
pulsesrc ! audioconvert ! opusenc
```

Linux without CUDA:

```
pipewirevideosrc portal=true ! videoconvert ! ffmpegenc
```

or `ffmpegenc` with the VA-API backend when `io-mode=dmabuf` yields a buffer
that backend accepts. If it does not, the software line is the one that ships.

Windows, first slice, a copy:

```
dxgisrc ! videoconvert ! mfencode
wasapisrc ! audioconvert ! opusenc
```

Windows, in-place, after the bridge exists:

```
dxgisrc zero-copy=true ! nvenc
```

`dxgisrc` in that mode produces `MemoryDomain::Cuda` by registering the D3D11
texture with `cuGraphicsD3D11RegisterResource` and wrapping the mapped pointer
the way `nvenc` already requires: one NV12 base pointer, chroma at
`luma + pitch * height`. There is no D3D11 memory domain. The bridge is the
source's output, not a new domain.

macOS:

```
screencapturesrc cv-output=true display-index=0 ! vtencode
coreaudiosrc ! audioconvert ! opusenc
```

`screencapturesrc` today takes the first display and has no `display-index`.
That property is part of this plan.

On PLI the host calls `NvEnc::force_keyframe`. The other encoders need the same
latch: `vtencode`, `mfencode`, and `ffmpegenc` gain `force-keyframe` if they
do not have it, and the host pulses it. The first frame is an IDR, which
`nvenc` already does.

## Client graphs

The client offers the same two sendrecv m-lines and discards its own send
pads (it sends no media). Receive pads:

```
h264parse ! <decoder> ! livesync ! autovideosink
opusdec ! audioconvert ! autoaudiosink
```

The decoder is `nvdec`, `mfdecode`, `vtdecode`, or `ffmpegdec`, chosen by the
same auto-plug `decodebin` uses. `livesync latency=0`. `software-decode=true`
forces `ffmpegdec`.

The browser client is the same SDP and the same data channel, decoded with
WebCodecs into a canvas. Pointer lock and `navigator.keyboard.lock` are the
capture side. Chrome is the target. Firefox stays untested until the native
session is solid.

## Capture sources to add

### dxgisrc

`dxgi` feature, `cfg(windows)`, the `windows` crate, same MTA rule as
`mfencode`. Desktop Duplication on the output selected by `display-index`.
Default output is tightly packed system NV12 at a requested `fps` (default
60). An unchanged frame repeats the last image so the encoder's rate stays
steady. `AccessLost` fails the source. The host tears the graph down and
rebuilds it. Capture runs in the interactive session, never in session 0.

Properties: `display-index` (u32), `fps` (u32), `zero-copy` (bool, default
false). `zero-copy=true` without CUDA fails at configure time.

### ximagesrc

`x11` feature, Linux, xcb shared memory. `display-index` selects the screen.
Output is system memory in the format the server gives, and `videoconvert`
follows. `gst_compat` maps the name `ximagesrc` to `screencapturesrc` today.
That mapping changes to this element on Linux and stays `screencapturesrc` on
macOS.

### display-index elsewhere

`screencapturesrc` gains `display-index` and stops hard-coding the first
display. The portal path records the restore token the user already granted.
Picking a monitor the portal did not grant is a failed start, not a guess.

## Input edges

Each edge is a small platform file in `g2g-desktop`. The packet and the router
stay shared and are tested without a desktop.

| Edge | Capture | Inject |
| :--- | :--- | :--- |
| Linux | the client window's keyboard and pointer, plus `/dev/input` for grab | `/dev/uinput`. The host user is in the `input` group, or the process has a writable `/dev/uinput`. |
| Windows | Raw Input on the client window | `SendInput` in the interactive session. The service injects when the interactive process cannot, which is the secure desktop. |
| macOS | the client window | `CGEvent`, which needs the accessibility permission. Screen recording stays the permission `screencapturesrc` already needs. |

Relative move is `EV_REL` on Linux and `MOUSEEVENTF_MOVE` on Windows.
Absolute position is the virtual desktop: `x` and `y` in pixels of
`display`, origin at that display's top left. The host maps `display` through
the same index the capture source uses.

Gamepad capture uses one crate on every client. Injection is a virtual uinput
gamepad on Linux and ViGEm on Windows. macOS injection waits until a virtual
HID device is chosen. The packet still parses there, and the router counts it
as undelivered.

Cursor: the host sends type 8 on change, and at least once when the session
starts. The client hides its cursor over the video while grabbed and draws the
bitmap at the hotspot. Grab is pointer lock on the web client and a window
grab on the native client. Escape releases the grab locally and is not
forwarded.

Clipboard: each side watches its clipboard and sends type 9 when the text
changes and the other side did not just set it. A payload over the 64 KiB
message cap is dropped.

## Windows service

The service process is installed separately, runs as SYSTEM, and accepts
injection packets on a named pipe from the interactive host. It does not
capture, encode, or speak WebRTC. The interactive host captures. When
`SendInput` from the interactive process fails because the secure desktop is
up, the host forwards the same packets down the pipe. The service is optional
at build time and required for the UAC slice.

## g2g-desktop

Workspace member. Two binaries, or one binary with two subcommands: `host` and
`client`.

Host flags, all of them real properties the process applies:

| flag | meaning | default |
| :--- | :--- | :--- |
| `--port` | TCP signaling port | 8080 |
| `--token` | bearer token | required |
| `--display-index` | which display to capture | 0 |
| `--fps` | capture rate | 60 |
| `--bitrate` | bits per second, passed to the encoder | 4000000 |
| `--audio` | capture and send audio | true |
| `--stun-server` | forwarded to the duplex session | unset |
| `--turn-server`, `--turn-user`, `--turn-pass` | forwarded | unset |

Client flags:

| flag | meaning | default |
| :--- | :--- | :--- |
| `--host` | host address | required |
| `--port` | signaling port | 8080 |
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

These run without a GPU or a desktop, in `g2g-desktop`:

- Packet round trip for every type, including a truncated payload, a length
  that does not match the body, a sequence that goes backwards, and a cursor
  whose RGBA length is wrong.
- The router drops while unfocused, drops a matching shortcut, and delivers a
  cursor packet from the network side to the cursor consumer only.
- HID mapping tables: a key that exists on both platforms maps to the same
  usage, and an unknown usage fails the inject call rather than sending a
  zero.
- The signaling state machine: bad token, offer before the accept byte, a
  second connection refused, a hang-up before the answer.

Hardware checks are probes, the same shape as `m739_screencapture`. A denied
capture permission asserts the error path. A granted desktop is manual: move
the pointer, hold a key, confirm the key-up arrives, confirm a late video
frame was counted in `livesync`'s `drop`.

## Order

Each slice builds and is reviewable on its own. Later slices depend on earlier
ones.

1. **Packet, router, signaling state machine, and the tests above.** No
   sockets yet beyond the state machine fed from byte slices. Crate
   `g2g-desktop` added to the workspace.
2. **Data channel on `WebRtcDuplexSession`.** An in-process loopback sends a
   packet across the channel while a video frame crosses the same `Rtc`.
3. **Linux host and Linux client on one machine.** Portal or `ximagesrc`,
   software or VA-API encode, `livesync` on the client, uinput inject, window
   capture. `inputs=false` when both run on the same seat, or the pointer
   feeds itself.
4. **`dxgisrc` system-memory path and the Windows client.** `mfencode`,
   `SendInput`. Then the D3D11-to-CUDA output and `nvenc`.
5. **macOS client, then the macOS host** on `screencapturesrc` `display-index`
   and `vtencode`.
6. **Audio, clipboard, cursor, grab, `force-keyframe` on every encoder, PLI
   wired through.**
7. **Gamepad** on Linux and Windows, and the Windows service.
8. **Chrome page** speaking the same TCP signaling and the same data channel.
9. **OIDC, the URL scheme, packages.**

A LAN session that moves a pointer is slice 3. Matching the done list above is
the end of slice 9.
