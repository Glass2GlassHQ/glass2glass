# g2g on ESP32-S3

Runs on a Seeed XIAO ESP32S3 Sense. At boot it checks the heap-free display,
audio and keyword-feature graphs from `noalloc-pipeline` against the checksums
the host computes, then the micro_speech and person_detect models against
TFLite's own scores. Then it runs two live graphs. The keyword graph scores the
PDM microphone for "yes" and "no". The person graph scores the OV3660 camera for
a person and streams each frame over Wi-Fi as RFC 4175 RTP.

## Setup

```bash
cargo install espup espflash
espup install
tools/esp-tflm-fetch.sh
```

`tools/esp-tflm-fetch.sh` downloads TFLite Micro and Espressif's ESP-NN kernels
into `third_party`, which `build.rs` compiles.

The serial port is `root:dialout`. Either join `dialout` or add a uaccess rule:

```bash
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="303a", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/70-espressif.rules
sudo udevadm control --reload && sudo udevadm trigger
```

Plug the Wi-Fi antenna into the U.FL socket on the main board. Without it the
board finds no networks.

## Build and run

The build reads three variables:

| Variable | Value |
| :--- | :--- |
| `G2G_WIFI_SSID` | a 2.4 GHz WPA2 network |
| `G2G_WIFI_PASSWORD` | its password |
| `G2G_RTP_DESTINATION` | the receiving host as `ip:port` |

They are compiled into the firmware, so keep them in a file outside the repo and
source it before building.

```bash
export PATH="$HOME/.rustup/toolchains/esp/xtensa-esp-elf/esp-15.2.0_20250920/xtensa-esp-elf/bin:$PATH"
source ~/.config/g2g-esp32s3/wifi.sh
cargo run --release
```

`cargo run` flashes with QIO at 80 MHz, which makes inference about 30% faster
than espflash's default DIO at 40 MHz, and opens the monitor. The board reports:

```
g2g on esp32s3
OK display: 2261 (expected 2261), 1291 us
OK audio: 214749012174 (expected 214749012174), 277203 us
OK keyword features: 17655463838127690952 (expected 17655463838127690952), 83007 us
OK speech: [128, 128, 127, 129] (TFLite interpreter [128, 128, 127, 129])
OK person: 10 of 10 sample images equal the reference kernels, 64037 us per image
PASS
listening: say yes or no
joining <network>
address 192.168.2.201/24, sending RTP to 192.168.2.21:5004
person  0.92 166 ms
yes      silence 0.00 unknown 0.00 yes 1.00 no 0.00 peak 3659
```

Keyword scores are per window with no averaging, so background noise sometimes
leans "no" at 0.4 to 0.75.

## Watch the camera

With g2g (`g2g-plugins` built with `std,wayland-sink,st2110`):

```bash
g2g-launch st2110videosrc port=5004 format=gray8 width=96 height=96 idle-timeout=3000000000 ! videoconvert ! videoscale ! video/x-raw,width=480,height=480 ! fpsdisplaysink video-sink=autovideosink sync=false
```

With GStreamer:

```bash
gst-launch-1.0 udpsrc port=5004 caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=RAW,sampling=YCbCr-4:2:2,depth=(string)8,width=(string)96,height=(string)96,colorimetry=(string)BT601-5,payload=96" ! rtpvrawdepay ! videoconvert ! videoscale ! video/x-raw,width=480,height=480 ! autovideosink sync=false
```

The board sends 7 to 9 frames per second at -74 dBm. A weaker signal sends
fewer, with longer gaps.

## Notes

- The release profile has no LTO: esp toolchains 1.94 to 1.98.0 fail to compile
  esp-hal's DMA code under LTO and later ones miscompile it (esp-rs/rust issue 283).
- Internal RAM is nearly full. Both TFLite Micro arenas and the microphone window
  live in the 8 MB PSRAM, and large buffers are static and left uninitialized so
  they are never built on the stack.
- The camera is an OV3660, not the OV2640 the board listing names.
