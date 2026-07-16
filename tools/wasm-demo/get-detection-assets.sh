#!/usr/bin/env bash
# Fetch the object-detection demo assets into models/ (git-ignored, ~13 MB):
#   - yolov8n.onnx : standard COCO YOLOv8n export (input images[1,3,640,640],
#                    output output0[1,84,8400]). Served same-origin; ort-shim.js
#                    loads onnxruntime-web from the CDN and runs it.
#   - bus.jpg      : the classic ultralytics sample (people + a bus).
#   - bus_640.h264 : bus.jpg encoded to a 640x640 Annex-B H.264 loop (square
#                    pixels: setsar=1, so no SAR/aspect skew), the fixture the
#                    "Detect (real YOLOv8 / ort-web)" demo mode runs on.
#
# Requires curl + ffmpeg. Run once before serving the demo with the ortdetect mode.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
models="$here/models"
mkdir -p "$models"

MODEL_URL="https://huggingface.co/Serotina/sentis-YOLOv8n-image/resolve/main/yolov8n.onnx"
BUS_URL="https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"

echo "fetching yolov8n.onnx ..."
curl -sL --fail -o "$models/yolov8n.onnx" "$MODEL_URL"

echo "fetching bus.jpg ..."
curl -sL --fail -o "$models/bus.jpg" "$BUS_URL"

echo "encoding bus_640.h264 (640x640, square pixels, no B-frames, no SEI) ..."
ffmpeg -y -loop 1 -i "$models/bus.jpg" -t 2 -r 15 \
  -vf "scale=640:640,setsar=1,format=yuv420p" -color_range tv \
  -c:v libx264 -profile:v baseline -x264-params keyint=15:scenecut=0 -bf 0 \
  -bsf:v "filter_units=remove_types=6" \
  -f h264 "$models/bus_640.h264"

echo
echo "done. Assets in $models"
echo "Serve the demo, then:"
echo "  ws-fixture-server 127.0.0.1:8080 $models/bus_640.h264 15"
echo "  open http://127.0.0.1:8000/ , pick 'Detect (real YOLOv8 / ort-web)', Start"
