# g2g on ESP32-S3

Runs the heap-free display, audio and keyword graphs from `noalloc-pipeline` on a
Seeed XIAO ESP32S3 Sense and prints each checksum next to the value the host
computes, then runs the keyword graph on the board's PDM microphone, printing
the scores for a one-second window every 500 ms.

## Setup

```bash
cargo install espup espflash
espup install
```

The serial port is `root:dialout`. Either join `dialout` or add a uaccess rule:

```bash
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="303a", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/70-espressif.rules
sudo udevadm control --reload && sudo udevadm trigger
```

## Build and run

```bash
. ~/export-esp.sh
cargo run --release
```

`cargo run` flashes with `espflash` and opens its monitor. The board reports:

```
g2g on esp32s3
OK display: 2261 (expected 2261), 1405 us
OK audio: 214749012174 (expected 214749012174), 261805 us
OK keyword: 16635284155892619601 (expected 16635284155892619601), 214923 us
PASS
listening: say yes or no
no       silence 0.00 unknown 0.02 yes 0.02 no 0.96 peak 634
```

`LogMelSpectrogram` has none of the noise reduction and gain control of the
TFLite Micro front end that micro_speech was trained on, so live scores favour
"no" even in a quiet room.
