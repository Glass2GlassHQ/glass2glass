#!/usr/bin/env bash
# Regenerate the AAC fixtures whose object types ADTS cannot signal, for the
# LOAS-in-MPEG-TS test (g2g-plugins/tests/m1731_ts_loas_aac.rs). Each is half a
# second of a 440 Hz stereo sine at 48 kHz in an .m4a:
#
#   aac_ld_stereo_48000.m4a    AAC-LD (object type 23), GStreamer's fdkaacenc
#   aac_eld_stereo_48000.m4a   AAC-ELD (object type 39), the fdkaac program
#   aac_usac_stereo_48000.m4a  USAC (object type 42), exhale
#
# fdkaacenc offers no ELD profile, so ELD comes from fdkaac (github.com/nu774/fdkaac)
# built against the system libfdk-aac into $FDKAAC_PREFIX. exhale
# (gitlab.com/ecodis/exhale) is built into $EXHALE_DIR.
#
# usage: tools/aac-loas-fixtures.sh [ld|eld|usac]...   (no argument: all three)
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"
work="${AAC_FIXTURE_WORK:-$HOME/tmp/aac-loas-fixtures}"
FDKAAC_PREFIX="${FDKAAC_PREFIX:-$HOME/opt/fdk-aac}"
EXHALE_DIR="${EXHALE_DIR:-$HOME/opt/exhale}"
FDKAAC_TAG=v1.0.9
EXHALE_TAG=v1.2.2
mkdir -p "$work"

sine_wav() {
    ffmpeg -v error -y -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=0.5" \
        -ac 2 -c:a pcm_s16le "$work/sine.wav"
}

ld() {
    gst-launch-1.0 -q filesrc location="$work/sine.wav" ! wavparse ! audioconvert \
        ! fdkaacenc ! audio/mpeg,profile=ld,stream-format=raw \
        ! mp4mux ! filesink location="$fixtures/aac_ld_stereo_48000.m4a"
}

eld() {
    if [ ! -x "$FDKAAC_PREFIX/bin/fdkaac" ]; then
        pkg-config --exists fdk-aac
        rm -rf "$work/fdkaac"
        git clone --depth 1 --branch "$FDKAAC_TAG" https://github.com/nu774/fdkaac.git "$work/fdkaac"
        (cd "$work/fdkaac" && autoreconf -fi && ./configure --prefix="$FDKAAC_PREFIX" \
            && make -j"$(nproc)" && make install)
    fi
    "$FDKAAC_PREFIX/bin/fdkaac" -p 39 -b 64000 \
        -o "$fixtures/aac_eld_stereo_48000.m4a" "$work/sine.wav"
}

usac() {
    if [ ! -d "$EXHALE_DIR/bin" ]; then
        rm -rf "$EXHALE_DIR"
        git clone --depth 1 --branch "$EXHALE_TAG" https://gitlab.com/ecodis/exhale.git "$EXHALE_DIR"
        make -C "$EXHALE_DIR" -j"$(nproc)" release
    fi
    # make release puts the program in bin or a subdirectory of it
    local exhale
    exhale="$(find "$EXHALE_DIR/bin" -type f -name exhale | head -n 1)"
    # presets 0 to 9 code without eSBR
    "$exhale" 3 "$work/sine.wav" "$fixtures/aac_usac_stereo_48000.m4a"
}

kinds=("$@")
if [ ${#kinds[@]} -eq 0 ]; then
    kinds=(ld eld usac)
fi
sine_wav
for kind in "${kinds[@]}"; do
    case "$kind" in
        ld | eld | usac) "$kind" ;;
        *) echo "unknown fixture: $kind" >&2; exit 2 ;;
    esac
done
ls -l "$fixtures"/aac_*_stereo_48000.m4a
