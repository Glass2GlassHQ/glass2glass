# Moonlight host

Plan for `g2g-desktop host --protocol gamestream`, a host that stock Moonlight
clients pair with and stream from, the way they use Sunshine. Nothing in this
file is built unless it names an element that already exists. When a slice
ships, delete it here and from [TODO.md](TODO.md), and write the shipped
behavior into the track that owns it.

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
7. The stream has the size the client asks for. Linux scales with a new
   `cudascale` before `nvenc`, Windows with `width` and `height` on
   `d3d11convert`.
8. H.264 and HEVC come from `nvenc` on Linux and `mfh264enc`/`mfh265enc` on
   Windows. The software and VA-API encoders refuse `--protocol gamestream`.
9. A PIN reaches the host on a localhost-only `POST /submit-pin`, and
   `g2g-desktop pin <PIN>` calls it.
10. Video and audio leave through `udpsink` in a new mode that sends each frame
    as one datagram from a fixed `bind-port`, to the address the client's ping
    came from.

## Ports

Fixed offsets from 47989, as in Sunshine: HTTP 47989, HTTPS 47984, RTSP 48010,
video UDP 47998, control UDP 47999, audio UDP 48000.

## Host graphs

```
<capture> ! dmabuftocuda ! cudascale width=W height=H ! nvenc ! moonlightvideopay ! udpsink
pulsesrc device=@DEFAULT_MONITOR@ ! audioconvert ! opusenc frame-size=5 bitrate-type=cbr audio-type=restricted-lowdelay ! moonlightaudiopay ! udpsink
```

Windows replaces the video front with
`d3d11screencapturesrc ! d3d11convert width=W height=H ! mfh265enc` and the
audio source with `wasapisrc loopback=true`. The session builds both graphs
after RTSP `ANNOUNCE`, which carries the codec, size, frame rate, bitrate and
packet size.

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
- `cudascale` resizes packed RGBA in CUDA memory, behind the `nvenc` feature.
  Properties: `width`, `height`.
- `udpsink` gets the datagram mode and `bind-port`.
- `d3d11convert` gets `width` and `height`.
- `opusenc` gets `bitrate-type` with gst's values `cbr`, `vbr` and
  `constrained-vbr`.

## Session

In `g2g-desktop`, module `gamestream`.

- mDNS advertises `_nvstream._tcp` on the HTTP port with no TXT records.
- Plain HTTP serves `/serverinfo`, `/pair` and `/submit-pin`.
- HTTPS asks for a client certificate, accepts any certificate in the
  handshake, then serves a request only when that certificate is in the paired
  list: `/serverinfo`, `/applist` with one "Desktop" entry, `/appasset`,
  `/launch`, `/resume`, `/cancel`.
- Pairing hashes the salt and PIN with SHA-256 into an AES-128-ECB key, swaps
  challenges signed with RSA-SHA256 over each certificate's signature, and ends
  with `pairchallenge` over HTTPS. The host certificate (from `rcgen`) and the
  paired client certificates are stored in the config directory.
- RTSP takes one request per TCP connection and closes it after the reply. It
  uses the encrypted framing when `/launch` returned `rtspenc://`. `DESCRIBE`
  lists the HEVC marker, the encryption flags and the `surround-params` lines.
- The video and audio ports learn the client's address from its ping, which
  carries the 16-byte payload `SETUP` returned.
- Control: ENet with AES-GCM on every message. An IDR request (0x0302) goes to
  `moonlightvideopay`. Input (0x0206) becomes an `Event` for the existing
  injectors, with keys mapped through a new Windows virtual-key to HID usage
  table. The host ends a session with 0x0109 carrying 0x80030023, and ends it
  itself after 10 s without a control message.

## Order

1. mDNS, `/serverinfo`, pairing, `/applist` and the pinned HTTPS port.
2. RTSP, `/launch`, ENet control, `moonlightvideopay`, the `udpsink` datagram
   mode, `cudascale`, the `d3d11convert` size, and graphs built from
   `ANNOUNCE`.
3. Audio: `moonlightaudiopay` and CBR Opus in 5 ms frames.
4. Input: keyboard through the virtual-key table, mouse and scroll.
5. The Windows host.

After those: reference frame invalidation, gamepad, HDR, surround, and `--run`
applications as `/applist` entries.

## Tests

- FEC parity matches vectors generated once with nanors and checked in as a
  fixture.
- `tools/moonlight-interop.sh` runs the Moonlight flatpak under Xvfb against the
  host: pair with `--pin`, list, stream for a few seconds, then check the
  Moonlight log for decoded frames and no connection error. It runs before
  each push of this work and puts no window on the desktop.

## References

- moonshine, a Rust host: https://github.com/hgaiser/moonshine
- Wolf protocol notes: https://games-on-whales.github.io/wolf/stable/protocols/index.html
- moonlight-common-c, the client side (GPL-3.0, behavior only):
  https://github.com/moonlight-stream/moonlight-common-c
