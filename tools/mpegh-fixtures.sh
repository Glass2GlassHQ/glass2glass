#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# Fetch the MPEG-H test streams the MPEG-H tests read (g2g-plugins/tests/m2620_mpegh_demux.rs,
# m2621_mpeghdec.rs, m2622_mpeghuimanager.rs) into the gitignored
# g2g-plugins/tests/fixtures/mpegh, then decode reference WAVs from them with Fraunhofer's
# mpeghDecoder program, with mpeghdec's default DRC settings. The streams come from
# github.com/Fraunhofer-IIS/mpegh-test-content (CC BY-NC-ND 4.0, so not committed) and the
# androidx/media test assets, which hold the only mha1 sample here.
#
# mpeghDecoder is built with the library (see "MPEG-H" in README.md), in
# $MPEGH_SOURCE/build/bin.
#
# usage: tools/mpegh-fixtures.sh
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures/mpegh"
MPEGH_SOURCE="${MPEGH_SOURCE:-$HOME/opt/mpeghdec-src}"
decoder="$MPEGH_SOURCE/build/bin/mpeghDecoder"
content=https://media.githubusercontent.com/media/Fraunhofer-IIS/mpegh-test-content/main
transport="$content/transport_layer_test_content"
media3=https://raw.githubusercontent.com/androidx/media/main/libraries/test_data/src/test/assets/media/mp4
# the mpeghDecoder options for mpeghdec's defaults: -24 LKFS, general DRC, all DRC gains, no album mode
ELEMENT_DEFAULT_DRC=(-rl 96 -dse 6 -db 127 -dc 127 -dam 0)

fetch() {
    local name="$1" url="$2"
    if [ -s "$fixtures/$name" ]; then
        echo "have $name"
        return
    fi
    echo "fetching $name"
    curl -fsSL --retry 3 -o "$fixtures/$name.part" "$url"
    mv "$fixtures/$name.part" "$fixtures/$name"
}

fetch mpegh_cicp_02_baseline_32kbps.mp4 "$transport/CICP_02/mp4/baseline/32kbps/mpegh_cicp_02_baseline_32kbps.mp4"
fetch mpegh_cicp_02_baseline_32kbps_fragmented.mp4 "$transport/CICP_02/mp4/baseline/32kbps/mpegh_cicp_02_baseline_32kbps_fragmented.mp4"
fetch mpegh_cicp_06_baseline_32kbps.mp4 "$transport/CICP_06/mp4/baseline/32kbps/mpegh_cicp_06_baseline_32kbps.mp4"
for packing in single multi_5 cont_2000; do
    directory="${packing%%_*}"
    fetch "mpegh_cicp_02_baseline_32kbps_$packing.ts" "$transport/CICP_02/ts/$directory/baseline/32kbps/mpegh_cicp_02_baseline_32kbps_$packing.ts"
done
fetch mpegh_config_change_cicp_2_14_6_baseline_32kbps.mp4 "$transport/ConfigChange/mp4/baseline/32kbps/mpegh_config_change_cicp_2_14_6_baseline_32kbps.mp4"
fetch TRI_Fileset_17_514H_D1_D2_D3_O1_24bit2160p50.audio.mp4 "$content/TRI_Fileset_17_514H_D1_D2_D3_O1_24bit2160p50.audio.mp4"
fetch sample_mpegh_mha1.mp4 "$media3/sample_mpegh_mha1.mp4"

reference() {
    local stream="$1" layout="$2"
    local wav="${stream%.mp4}_cicp_$layout.wav"
    if [ -s "$fixtures/$wav" ]; then
        echo "have $wav"
        return
    fi
    echo "decoding $wav"
    "$decoder" -if "$fixtures/$stream" -of "$fixtures/$wav" -tl "$layout" "${ELEMENT_DEFAULT_DRC[@]}" >/dev/null
}

reference mpegh_cicp_02_baseline_32kbps.mp4 2
reference mpegh_cicp_06_baseline_32kbps.mp4 6
reference sample_mpegh_mha1.mp4 2

echo "fixtures ready in $fixtures"
