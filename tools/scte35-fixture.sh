#!/usr/bin/env bash
# Regenerate g2g-plugins/tests/fixtures/h264_scte35_splice_insert.ts for the
# SCTE-35 test (g2g-plugins/tests/m2038_scte35.rs): three seconds of 64x64
# H.264 from ffmpeg, with a TSDuck splice_insert (event 4660, splice at PTS
# 323000 plus a pts_adjustment of 1000) on PID 500. spliceinject writes into
# null packets, so ffmpeg muxes at a constant rate. TSDuck is the unpacked RPM
# under $TSDUCK_PREFIX.
set -euo pipefail

fixtures="$(cd "$(dirname "$0")/.." && pwd)/g2g-plugins/tests/fixtures"
work="${SCTE35_FIXTURE_WORK:-$HOME/tmp/scte35-fixture}"
TSDUCK_PREFIX="${TSDUCK_PREFIX:-$HOME/opt/tsduck/usr}"
export LD_LIBRARY_PATH="$TSDUCK_PREFIX/lib64" TSPLUGINS_PATH="$TSDUCK_PREFIX/lib64/tsduck"
export TSDUCK_NO_VERSION_CHECK=true
mkdir -p "$work"

cat > "$work/cue.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<tsduck>
  <splice_information_table pts_adjustment="1000">
    <splice_insert splice_event_id="4660" out_of_network="true" pts_time="323000" unique_program_id="7" avail_num="1" avails_expected="2">
      <break_duration auto_return="true" duration="270000"/>
    </splice_insert>
  </splice_information_table>
</tsduck>
EOF

ffmpeg -v error -y -f lavfi -i testsrc=size=64x64:rate=10 -t 3 \
    -c:v libx264 -pix_fmt yuv420p -g 10 -muxrate 150k -f mpegts "$work/src.ts"
"$TSDUCK_PREFIX/bin/tsp" -I file "$work/src.ts" \
    -P pmt --service 1 --add-pid 500/0x86 --add-programinfo-id 0x43554549 \
    -P spliceinject --service 1 --pid 500 --files "$work/cue.xml" --wait-first-batch \
    -O file "$fixtures/h264_scte35_splice_insert.ts"
