# Moonlight host

`g2g-desktop host --protocol gamestream` is a host that stock Moonlight
clients pair with and stream from, the way they use Sunshine. It serves
discovery, pairing, the app list with launched `--app` commands, encrypted
H.264 or HEVC video from an X display, the desktop portal, a `--run` headless
display or the Windows desktop at the size the client asks for, reference
invalidation on Linux, encrypted stereo, 5.1 or 7.1 Opus audio from the
default output, and keyboard, mouse, scroll, text, touch, pen and gamepad
input. Open work is under "Moonlight host" in [TODO.md](TODO.md).

## Decisions

1. The host is a mode of `g2g-desktop host`, so it shares capture, the
   encoders, the injectors and `--run`. One Moonlight session runs at a time.
2. `/serverinfo` sends `appversion` 7.1.431.-1 and `GfeVersion` 3.23.0.74, the
   values Sunshine sends. Moonlight treats a negative fourth number as Sunshine
   and then uses SHA-256 pairing, RTSP over TCP and the encrypted control
   stream.
3. The code is written from the wire format. Sunshine, moonlight-common-c and
   moonlight-common-rust are GPL-3.0 and are read for behavior only. moonshine
   (BSD-2-Clause) and Wolf (MIT) can be ported with their notices kept.
4. Control runs on the `tokio-enet` crate, which moonshine ships against
   Moonlight's ENet fork.
5. Reed-Solomon parity comes from a g2g encoder for the Cauchy GF(2^8) matrix
   the nanors library builds, plus the fixed parity matrix Moonlight's audio
   uses. `reed-solomon-simd` computes a different code.
6. Video, audio and control are encrypted from the first slice that carries
   them, all under the `rikey` that `/launch` delivers: AES-GCM per video
   packet, AES-CBC per audio packet, AES-GCM per control message.
7. The stream has the size the client asks for, set by a caps filter after
   `cudascale` on Linux and after `d3d11convert` on Windows.
8. H.264 and HEVC come from `nvenc` on Linux and `mfh264enc`/`mfh265enc` on
   Windows. The software and VA-API encoders refuse `--protocol gamestream`.
9. A PIN reaches the host on a localhost-only `POST /submit-pin`, and
   `g2g-desktop pin <PIN>` calls it.
10. Video and audio leave through `udpsink`, which sends each
    `application/x-datagram` frame as one datagram from a fixed `bind-port`, to
    the address the client's ping came from.
11. `nvenc` encodes with the `ref-frames` count `ANNOUNCE` asks for. Moonlight
    asks for 1 when its decoder cannot invalidate references, and its decoder
    fails on every second P-frame of a stream with more. It asks for 0 when its
    decoder can, and the host then keeps 16, the most H.264 and HEVC allow.
12. On Linux `DESCRIBE` offers reference invalidation, and a Moonlight client
    whose decoder supports it reports lost frames instead of asking for an IDR.
    The Windows host does not offer it, because the Media Foundation encoders
    can only answer with a key frame.

## Ports

Fixed offsets from 47989, as in Sunshine: HTTP 47989, HTTPS 47984, RTSP 48010,
video UDP 47998, control UDP 47999, audio UDP 48000.

## Host graphs

```
ximagesrc ! video/x-raw,framerate=F ! videoconvert ! cudaupload ! cudascale ! video/x-raw,width=W,height=H ! nvenc ! moonlightvideopay ! udpsink
pulsesrc device=@DEFAULT_MONITOR@ ! audioconvert ! audio/x-raw,channels=C ! opusenc frame-size=5 bitrate-type=cbr audio-type=restricted-lowdelay ! moonlightaudiopay ! udpsink
```

The portal capture replaces `ximagesrc` and its caps filter with
`pipewirevideosrc ! videorate drop-only=true ! video/x-raw,framerate=F`, with
the pointer drawn into the picture. `--run` captures with `waylanddisplaysrc !
dmabuftocuda ! videorate drop-only=true ! video/x-raw,framerate=F`. Both
sources send a frame only when the picture changes, and a `videorate` that
repeats frames holds each one until the next change.
Windows replaces the video front with
`d3d11screencapturesrc ! d3d11convert ! video/x-raw,width=W,height=H ! mfh265enc`
and the audio source with `wasapisrc loopback=true`, which plays silence to
the output while it captures because Windows delivers no loopback packets while
nothing plays. The session builds both
graphs after RTSP `ANNOUNCE`, which carries the codec, size, frame rate,
bitrate and packet size.

## Elements

In `g2g-plugins`, behind a `gamestream` feature unless noted.

- `moonlightvideopay` takes H.264 or HEVC access units, Annex-B with
  `alignment=au`. It prepends the 8-byte frame header, which carries the frame
  type and the time since capture in 0.1 ms units. It cuts `packet-size`
  slices, each behind an RTP header and the NVIDIA packet header, adds parity
  per FEC block (at most four blocks per frame), and seals every packet with
  AES-GCM. It outputs one frame per datagram, and turns an IDR request from the
  session into `Reconfigure::ForceKeyframe`. It keeps the frame numbers and
  presentation times of the last 16 frames it sent, and turns the first frame
  of a lost range into `Reconfigure::InvalidateReferences` with that frame's
  presentation time, or into a key frame request once the frame is older.
  `nvenc` answers an invalidation with `nvEncInvalidateRefFrames`, which also
  invalidates every frame predicted from the lost one, and sends an IDR when
  the lost frame is no longer one of its references. Every other encoder
  answers it with a key frame. Properties: `packet-size`,
  `fec-percentage`, `min-required-fec-packets`, `key`, `encrypt`.
- `moonlightaudiopay` puts each Opus packet in one RTP packet, encrypts it with
  AES-CBC under an IV made from `key-id` plus the sequence number, and sends two
  parity packets after every four. Properties: `packet-duration`, `key`,
  `key-id`, `encrypt`.
- `opusenc` gets `bitrate-type` with gst's values `cbr`, `vbr` and
  `constrained-vbr`.

## Session

In `g2g-desktop`, module `gamestream`.

- mDNS advertises `_nvstream._tcp` on the HTTP port with no TXT records.
- Plain HTTP serves `/serverinfo`, `/pair` and `/submit-pin`.
- HTTPS asks for a client certificate, accepts any certificate in the
  handshake, then serves `/applist` only when that certificate is in the
  paired list. The list has a "Desktop" entry, and with `--run` one entry per
  `--app TITLE=COMMAND`. `/serverinfo` reports `PairStatus` 1 for
  a paired certificate. `/launch`, `/resume` and `/cancel` come with video.
  `/launch` of an `--app` entry runs its command with `sh -c` on the headless
  display, after stopping another app the host started. The app keeps running
  between sessions, so Moonlight can resume it, and `/cancel` (Moonlight's
  quit) kills the process. `/serverinfo` reports the running app as
  `currentgame`.
  The host serves no `/appasset`, and Moonlight shows its own box art.
- Pairing hashes the salt and PIN with SHA-256 into an AES-128-ECB key, swaps
  challenges signed with RSA-SHA256 over each certificate's signature, and ends
  with `pairchallenge` over HTTPS. A 2048-bit RSA host key (made once with the
  `rsa` crate, signed with `ring`), the host certificate, the `uniqueid` and
  the paired client certificates live in `g2g-desktop/gamestream` under the
  user's config directory.
- RTSP takes one request per TCP connection and closes it after the reply.
  `/launch` returns a plain `rtsp://` URL, so RTSP itself is not encrypted.
  `DESCRIBE` lists the HEVC marker and the encryption flags, and requests video
  and audio encryption.
- `SETUP` returns no ping payload, so Moonlight sends the 4-byte `PING`. The
  video and audio ports each take the address of the first `PING` from the
  client that launched the session. A failed audio graph is logged and video
  keeps streaming.
- `ANNOUNCE` sets the audio packet duration (5 or 10 ms), the audio
  encryption bit and the channel count. Opus runs CBR at Sunshine's rates:
  96 kbit/s for stereo, 256 kbit/s for 5.1 and 450 kbit/s for 7.1. `opusenc`
  encodes 5.1 and 7.1 as mapping family 1 from the ffmpeg channel order, which
  is the order Moonlight plays and the order `pulsesrc` captures in.
  `DESCRIBE` lists that layout as the one `surround-params` per channel count,
  so Moonlight never asks for its high-quality surround mode, which needs a
  second layout.
- Control: ENet with AES-GCM on every message. An IDR request (0x0302) and a
  reference invalidation (0x0301, the first and last lost frame numbers) go to
  `moonlightvideopay`. Input (0x0206) becomes an `Event` for the injector of
  the capture: XTest for `--x11-display`, uinput for the portal, the headless
  compositor for `--run` and `SendInput` on Windows. Keys map through a
  Windows virtual-key to HID usage table. Absolute pointer positions scale from
  the size Moonlight sends to the captured size. Mouse buttons 4 and 5 become
  the back and forward buttons. Text (0x17) becomes `Event::Text`. DESCRIBE
  sets feature flag 0x01, so Moonlight sends touch (0x55000002) and pen
  (0x55000003), and both become `Event::Touch` contacts with positions scaled
  from fractions of the video to the capture. The pen is one more contact, and
  hover events are dropped. uinput injects through a multitouch touchscreen,
  the `--run` compositor through `wl_touch`, `SendInput` through a synthetic
  touch pointer device, and XTest moves the pointer and holds the left button
  with the first contact. A controller report (0x0C) becomes `Event::Gamepad`
  with Moonlight's XInput button bits, and a report whose controller bit is
  clear in the connected mask removes that pad. On Linux every capture injects
  gamepads through one uinput Xbox 360 pad per controller number, with the
  xpad driver's codes and ranges and the Y axes inverted. A host that cannot
  open `/dev/uinput` warns once and drops gamepads. On Windows each controller
  becomes an Xbox 360 pad on ViGEmBus through the `vigem-client` crate, and a
  host without ViGEmBus installed warns once and drops gamepads. Keys still held when the control stream ends are
  released. The host ends a session with 0x0109 carrying 0x80030023, and ends it
  itself after 10 s without a control message.

## Tests

- FEC parity matches vectors generated once with nanors and checked in as a
  fixture.
- `tools/moonlight-interop.sh` runs the Moonlight flatpak under Xvfb against the
  host, each side with a fresh config directory: it pairs with `--pin`, enters
  the PIN through `g2g-desktop pin`, lists the apps, streams the Xvfb display
  for 8 seconds and checks that Moonlight's video stats report decoded frames
  and that its audio stream received packets and decrypted every one.
  `AUDIO=5.1-surround` and `AUDIO=7.1-surround` stream surround. A test in
  `g2g-plugins` encodes a front-center tone through `opusenc` and checks that
  `opusdec` puts it back on front center. Moonlight
  runs with SDL's dummy audio driver, so nothing plays aloud.
  `CODEC=HEVC` picks the codec. With `CAPTURE=run` the headless display runs
  `tools/moonlight-input-probe.py`, xdotool clicks and types into Moonlight's
  window, pastes text with Moonlight's Ctrl+Alt+Shift+V, and the script checks
  that the probe received a left click, a back click, the key and the pasted
  text. It also streams an `--app` entry that records its process id, and
  checks that Moonlight's quit killed that process. Moonlight on Xvfb has no touch source, so touch is tested by injecting
  `Event::Touch` into each Linux injector. `LOSS_PERCENT` runs the script in a
  network namespace of its own and drops that share of the host's video
  packets after pairing, and with `CODEC=HEVC` checks that Moonlight asked for
  reference invalidation and the host answered it. At 30% loss Moonlight
  invalidated 14 times in 8 seconds, asked for no IDR, and its decoder logged
  no missing reference. It runs before each push of this work and puts
  no window on the desktop.
- A uinput test creates the Xbox 360 pad, grabs it, injects buttons, a stick,
  a trigger and the dpad, and reads the pad's state back. Moonlight on Xvfb
  has no gamepad, so no interop run sends one. A Windows test plugs a pad into
  ViGEmBus, or checks that a host without ViGEmBus reports it once. Neither
  the Windows laptop nor japan has ViGEmBus installed.
- `nvenc` encodes H.264 and HEVC with four references, invalidates two frames,
  and checks that the next frame decodes without them as a predicted frame.
- The Windows host is checked by hand: the Moonlight flatpak on Linux streams
  over the LAN from a Windows host started in the logged-in session, and the
  host log at `g2g-desktop-gamestream:debug` lists the input it injected.
  Windows cancels a synthetic touch contact that is not injected again within
  about a second, so `SendInput` repeats the held contacts every 50 ms. A
  window logging the mouse events Windows makes from touch showed a press, a
  drag and a release for a contact held 1.5 s with the repeat, and a cancelled
  tap without it.

## References

- moonshine, a Rust host: https://github.com/hgaiser/moonshine
- Wolf protocol notes: https://games-on-whales.github.io/wolf/stable/protocols/index.html
- moonlight-common-c, the client side (GPL-3.0, behavior only):
  https://github.com/moonlight-stream/moonlight-common-c
