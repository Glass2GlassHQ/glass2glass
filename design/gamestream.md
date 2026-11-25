# Moonlight host

Plan for `g2g-desktop host --protocol gamestream`, a host that stock Moonlight
clients pair with and stream from, the way they use Sunshine. Discovery,
pairing, the app list and encrypted H.264 or HEVC video from an X display, the
desktop portal or a `--run` headless display, at the size the client asks for,
run today. Nothing else in this file is built unless it names an element that
already exists. When a slice ships, delete it
here and from [TODO.md](TODO.md), and write the shipped behavior into the track
that owns it.

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
    fails on every second P-frame of a stream with more.

## Ports

Fixed offsets from 47989, as in Sunshine: HTTP 47989, HTTPS 47984, RTSP 48010,
video UDP 47998, control UDP 47999, audio UDP 48000.

## Host graphs

```
ximagesrc ! video/x-raw,framerate=F ! videoconvert ! cudaupload ! cudascale ! video/x-raw,width=W,height=H ! nvenc ! moonlightvideopay ! udpsink
pulsesrc device=@DEFAULT_MONITOR@ ! audioconvert ! opusenc frame-size=5 bitrate-type=cbr audio-type=restricted-lowdelay ! moonlightaudiopay ! udpsink
```

The portal capture replaces `ximagesrc` and its caps filter with
`pipewirevideosrc ! videorate drop-only=true ! video/x-raw,framerate=F`, with
the pointer drawn into the picture. `--run` captures with `waylanddisplaysrc !
dmabuftocuda ! videorate drop-only=true ! video/x-raw,framerate=F`. Both
sources send a frame only when the picture changes, and a `videorate` that
repeats frames holds each one until the next change.
Windows replaces the video front with
`d3d11screencapturesrc ! d3d11convert ! video/x-raw,width=W,height=H ! mfh265enc`
and the audio source with `wasapisrc loopback=true`. The session builds both
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
  session into `Reconfigure::ForceKeyframe`. Properties: `packet-size`,
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
  handshake, then serves `/applist` (one "Desktop" entry) only when that
  certificate is in the paired list. `/serverinfo` reports `PairStatus` 1 for
  a paired certificate. `/launch`, `/resume` and `/cancel` come with video.
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
  video port takes the address of the first `PING` from the client that
  launched the session.
- Control: ENet with AES-GCM on every message. An IDR request (0x0302) goes to
  `moonlightvideopay`. Input (0x0206) becomes an `Event` for the existing
  injectors, with keys mapped through a new Windows virtual-key to HID usage
  table. The host ends a session with 0x0109 carrying 0x80030023, and ends it
  itself after 10 s without a control message.

## Order

1. Audio: `moonlightaudiopay` and CBR Opus in 5 ms frames.
2. Input: keyboard through the virtual-key table, mouse and scroll.
3. The Windows host.

After those: reference frame invalidation, gamepad, HDR, surround, and `--run`
applications as `/applist` entries.

## Tests

- FEC parity matches vectors generated once with nanors and checked in as a
  fixture.
- `tools/moonlight-interop.sh` runs the Moonlight flatpak under Xvfb against the
  host, each side with a fresh config directory: it pairs with `--pin`, enters
  the PIN through `g2g-desktop pin`, lists the apps, streams the Xvfb display
  for 8 seconds and checks that Moonlight's video stats report decoded frames.
  `CODEC=HEVC` picks the codec. It runs before each push of this work and puts
  no window on the desktop.

## References

- moonshine, a Rust host: https://github.com/hgaiser/moonshine
- Wolf protocol notes: https://games-on-whales.github.io/wolf/stable/protocols/index.html
- moonlight-common-c, the client side (GPL-3.0, behavior only):
  https://github.com/moonlight-stream/moonlight-common-c
