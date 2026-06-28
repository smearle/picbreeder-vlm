#!/bin/bash
# Fix the trio figure's RIGHT panel "stops early" effect.
#
# The right panel (archive-fills scroll) rendered 2682 frames of motion (447
# rows x DESC=6) + 200 frames of a FROZEN final grid (--hold 200) = 2882.
# So while phylogeny (left) + leaderboard (middle) keep animating to frame
# 2882, the right reel freezes for the last ~8.3s.
#
# Fix: take the existing right render, drop the 200 frozen frames, and
# time-stretch the 2682 motion frames to fill the full 2882 (factor 2882/2682
# = 1.074571). The reel now scrolls continuously to the end, all 3,123 images
# still shown, ~7% of frames duplicated (a tiny cadence change, no dead freeze).
# Then re-composite over the existing trio (left+middle reused, crop 1126x1028).
set -e
cd /home/jupyter-smearle/picbreeder-vlm
BLOG=/home/jupyter-smearle/smearle.github.io/picbreeder-vlm-06b0d76d/assets
TMP=archive_animations/out/_scrolltmp
RIGHT=$TMP/trio_right_scroll_pop_fixed.mp4   # 2882 = 2682 motion + 200 hold
OLDTRIO=$BLOG/trio_branched.mp4              # current (frozen-right) trio
OUTTRIO=$TMP/trio_branched_pacingfix.mp4

test -f "$RIGHT"   || { echo "MISSING $RIGHT"; exit 1; }
test -f "$OLDTRIO" || { echo "MISSING $OLDTRIO"; exit 1; }

echo "[1/4] backup current blog trio"
cp -n "$OLDTRIO" "$OLDTRIO.bak_prefreezefix" && echo "  backed up" || echo "  backup already exists (kept)"

echo "[2/4] composite: stretch right motion 2682->2882, hstack with cropped left+middle"
ffmpeg -y -loglevel error -i "$OLDTRIO" -i "$RIGHT" -filter_complex \
  "[0:v]crop=1126:1028:0:0[lm];\
   [1:v]trim=end_frame=2682,setpts=(PTS-STARTPTS)*1.074571,fps=24,scale=554:1028[r];\
   [lm][r]hstack[v]" \
  -map "[v]" -r 24 -frames:v 2882 -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart "$OUTTRIO"

echo "[3/4] result geometry:"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate -of csv=p=0 "$OUTTRIO"

echo "[4/4] stage to blog + refresh poster"
cp "$OUTTRIO" "$OLDTRIO"
ffmpeg -y -loglevel error -ss 60 -i "$OLDTRIO" -frames:v 1 "$BLOG/trio_branched.jpg"
echo "  staged: $(du -h "$OLDTRIO" | cut -f1)  $OLDTRIO"
echo "ALL DONE"
