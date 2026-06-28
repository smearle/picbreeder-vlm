#!/bin/bash
# Re-render the trio figure's RIGHT panel (archive-fills scroll) with the
# jiggle fix in archive_scroll_lineage.py, then composite it over the existing
# trio (left+middle reused) -> drop-in replacement, same 1680x1028 / 2882 frames.
set -e
cd /home/jupyter-smearle/picbreeder-vlm
PY=.venv/bin/python
RUN=sweep_logs/sweep/th10_ag20_model-gemini-2.5-pro_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s5
TMP=archive_animations/out/_scrolltmp
BLOG=/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets
NEWRIGHT=$TMP/trio_right_scroll_pop_fixed.mp4
OLDTRIO=$BLOG/trio_branched.mp4
OUTTRIO=$TMP/trio_branched_fixed.mp4

echo "[1/5] sanity"
test -f "$RUN/archive/archive_metadata.json" || { echo "MISSING run metadata"; exit 1; }
test -d "$RUN/archive/images" || { echo "MISSING archive/images"; exit 1; }
echo "  images: $(ls "$RUN/archive/images" | wc -l)   genomes: $(ls "$RUN/archive/genomes" 2>/dev/null | wc -l)"

echo "[2/5] backup current blog trio"
cp -n "$OLDTRIO" "$OLDTRIO.bak_prejigglefix" 2>/dev/null && echo "  backed up" || echo "  backup already exists (kept)"

echo "[3/5] render right panel (pop, fixed easing)"
$PY archive_animations/archive_scroll_lineage.py --run "$RUN" \
  --out "$NEWRIGHT" \
  --reveal pop --cols 7 --rows-visible 13 --cell 73 --gap 6 \
  --descent-frames 6 --hold 200 --n 3123 --fps 24 --jobs 24

echo "[3b] new right panel geometry:"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate -of csv=p=0 "$NEWRIGHT"

echo "[4/5] composite over old trio (reuse left+middle = crop 1126x1028)"
ffmpeg -y -loglevel error -i "$OLDTRIO" -i "$NEWRIGHT" \
  -filter_complex "[0:v]crop=1126:1028:0:0[lm];[1:v]scale=554:1028[r];[lm][r]hstack[v]" \
  -map "[v]" -r 24 -frames:v 2882 -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart "$OUTTRIO"
echo "  result geometry:"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate -of csv=p=0 "$OUTTRIO"

echo "[5/5] stage to blog + refresh poster"
cp "$OUTTRIO" "$OLDTRIO"
ffmpeg -y -loglevel error -ss 60 -i "$OLDTRIO" -frames:v 1 "$BLOG/trio_branched.jpg"
echo "  staged: $(du -h "$OLDTRIO" | cut -f1)  $OLDTRIO"
echo "ALL DONE"
