#!/bin/bash
# Rebuild the blog's "archive being born" scroll feeds (archive_scroll_lineage.py)
# for the three noise levels, then stage mp4 + poster into the blog assets dir.
# Same seed (s3) across all three -> controlled comparison, only epsilon varies.
set -e
cd /home/jupyter-smearle/picbreeder-vlm
PY=.venv/bin/python
DST=/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets/scrolls
TMP=archive_animations/out/_scrolltmp
mkdir -p "$TMP" "$DST"

declare -A RUNS=(
  [randp0]=th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s3
  [randp0.25]=th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_randp0.25_rmode-all_nopersonalities_fixed-sesh_s3
  [randp0.5]=th1_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_randp0.5_rmode-all_nopersonalities_fixed-sesh_s3
)

for key in randp0 randp0.25 randp0.5; do
  RUN="sweep_logs/sweep/${RUNS[$key]}"
  echo "=== $key: scroll feed (n=1500) ==="
  $PY archive_animations/archive_scroll_lineage.py --run "$RUN" \
    --out "$TMP/scroll_${key}.mp4" \
    --n 1500 --cols 9 --rows-visible 7 --cell 70 --gap 1 \
    --morph-steps 12 --descent-frames 26 --hold 36 --jobs 24
  # poster: a frame well into the steady feed (full grid)
  ffmpeg -y -loglevel error -ss 60 -i "$TMP/scroll_${key}.mp4" -frames:v 1 "$DST/scroll_${key}.jpg"
  cp "$TMP/scroll_${key}.mp4" "$DST/scroll_${key}.mp4"
  echo "  staged: $(du -h "$DST/scroll_${key}.mp4"|cut -f1)  scroll_${key}.mp4"
done
echo "ALL DONE"
ls -lh "$DST"
