#!/bin/bash
# Rebuild the soda triptych (figure #trio-soda) with the CENTER panel swapped
# from the Top-Rated board to the Most-Branched board -- matching the trio
# figure above it (trio_branched), which also ranks by branch count.
#
# The three native panels from the Jun-7 build live in out/_sodatrio:
#   phylo_s8.mp4        707 frames  548x1010   (left)
#   lb_branched_s8.mp4 1792 frames  548x1010   (center, NEW -- was lb_rated_chrono)
#   reel_s8.mp4         882 frames  560x1034   (right)
# The previous trio anchored on the 848-frame rated-chrono center; we keep the
# same 848-frame length so phylo+reel pacing and the figure scrubber/poster are
# unchanged, and only retime the branched board (1792 -> 848) into that slot.
set -e
cd /home/jupyter-smearle/picbreeder-vlm
ST=archive_animations/out/_sodatrio
BLOG=/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets
N=848

test -f "$ST/phylo_s8.mp4"      || { echo "MISSING phylo_s8";      exit 1; }
test -f "$ST/lb_branched_s8.mp4"|| { echo "MISSING lb_branched_s8";exit 1; }
test -f "$ST/reel_s8.mp4"       || { echo "MISSING reel_s8";       exit 1; }

echo "[1/4] backup current blog trio"
cp -n "$BLOG/trio_soda.mp4" "$BLOG/trio_soda.mp4.bak_prebranched" && echo "  backed up" || echo "  backup exists (kept)"

echo "[2/4] composite raw (retime each panel to $N frames, hstack 1644x1010)"
ffmpeg -y -loglevel error \
  -i "$ST/phylo_s8.mp4" -i "$ST/lb_branched_s8.mp4" -i "$ST/reel_s8.mp4" \
  -filter_complex "\
    [0:v]setpts=PTS*($N/707),fps=24,scale=548:1010,setsar=1[l];\
    [1:v]setpts=PTS*($N/1792),fps=24,scale=548:1010,setsar=1[c];\
    [2:v]setpts=PTS*($N/882),fps=24,scale=548:1010,setsar=1[r];\
    [l][c][r]hstack=inputs=3[v]" \
  -map "[v]" -r 24 -frames:v $N -c:v libx264 -pix_fmt yuv420p -crf 18 \
  -preset slow -movflags +faststart "$ST/trio_soda_branched_raw.mp4"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate -of csv=p=0 "$ST/trio_soda_branched_raw.mp4"

echo "[3/4] web encode (crf 30, faststart -- ~36M, matches prior budget)"
ffmpeg -y -loglevel error -i "$ST/trio_soda_branched_raw.mp4" \
  -c:v libx264 -pix_fmt yuv420p -crf 30 -preset slow -movflags +faststart \
  "$ST/trio_soda_branched_web.mp4"

echo "[4/4] stage to blog + refresh poster"
cp "$ST/trio_soda_branched_web.mp4" "$BLOG/trio_soda.mp4"
ffmpeg -y -loglevel error -ss 18 -i "$BLOG/trio_soda.mp4" -frames:v 1 "$BLOG/trio_soda.jpg"
echo "  staged: $(du -h "$BLOG/trio_soda.mp4" | cut -f1)  $BLOG/trio_soda.mp4"
echo "ALL DONE"
