#!/bin/bash
# Render separate hi-res morph (grid) + trail assets for 3 blog runs, K=30,
# then ping-pong (boomerang) each into the blog assets dir.
set -e
cd /home/jupyter-smearle/picbreeder-vlm
PY=.venv/bin/python
DST=/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets/traversals
TMP=archive_animations/out/_blogtmp
mkdir -p "$TMP" "$DST"

declare -A RUNS=(
  [default]=th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s3
  [randp1]=th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_randp1_rmode-all_nopersonalities_fixed-sesh_s5
  [randp2]=ag20_tb-1_scheme-toggle_randp2_rmode-all_nopersonalities_fixed-sesh_s3
)

boomerang () {  # in out crf
  ffmpeg -y -loglevel error -i "$1" -filter_complex \
    "[0:v]split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1[v]" \
    -map "[v]" -r 24 -c:v libx264 -preset slow -crf "$3" -pix_fmt yuv420p -movflags +faststart "$2"
}

for key in default randp1 randp2; do
  RUN="sweep_logs/sweep/${RUNS[$key]}"
  echo "=== $key: grid (cell128) ==="
  $PY archive_animations/walker_grid.py --run "$RUN" --out "$TMP/${key}_grid.mp4" \
    --max-n 600 -k 36 -K 30 --cell 128 --frames-per-edge 12 --hold 0 --jobs 8
  echo "=== $key: trails (600) ==="
  $PY archive_animations/walker_trails.py --run "$RUN" --out "$TMP/${key}_trails.mp4" \
    -k 36 --method best-first -K 30 --max-n 600 --require-genome \
    --size 600 --margin 16 --fps 24 --frames-per-edge 12 --hold 0 --theme light
  echo "=== $key: boomerang both ==="
  boomerang "$TMP/${key}_grid.mp4"   "$DST/${key}_grid.mp4"   22
  boomerang "$TMP/${key}_trails.mp4" "$DST/${key}_trails.mp4" 20
  # poster from the boomerang midpoint (fullest extent, before it reverses)
  ffmpeg -y -loglevel error -ss 15 -i "$DST/${key}_grid.mp4" -frames:v 1 "$DST/${key}_grid.jpg"
  ffmpeg -y -loglevel error -ss 15 -i "$DST/${key}_trails.mp4" -frames:v 1 "$DST/${key}_trails.jpg"
  echo "  grid=$(du -h "$DST/${key}_grid.mp4"|cut -f1)  trails=$(du -h "$DST/${key}_trails.mp4"|cut -f1)"
done
echo "ALL DONE"
ls -lh "$DST"