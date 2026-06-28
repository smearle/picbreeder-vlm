#!/bin/bash
# Build an "archive growth" triptych for the HUMAN Picbreeder archive, for local
# inspection. Same three panels as the blog VLM trio:
#   LEFT   phylogeny tree growing from the centre   (archive_tree.py)
#   MIDDLE running "most branched" leaderboard       (leaderboard.py --board branched)
#   RIGHT  archive-fills scroll                       (archive_scroll_lineage.py --reveal pop)
#
# Human data has no VLM ratings (-> branched board, not rated) and no pickled
# genomes (-> pop scroll, no morph). The run dir is synthesized by
# tools/build_human_run.py from the 128px human thumbs + picbreeder branchFrom edges.
#
# Each panel has its own pacing model, so we aim each near ~TGT frames, then
# time-stretch all three to the common max in the final hstack (mild cadence
# change, no panel freezes early).
set -e
cd /home/jupyter-smearle/picbreeder-vlm
PY=.venv/bin/python
RUN=archive_animations/out/human_run
T=archive_animations/out/_humantrio
OUT=$T/trio_human.mp4
H=1028          # common panel height
JOBS=24
mkdir -p "$T"

echo "[0/5] build human run dir (metadata + image symlinks)"
$PY tools/build_human_run.py --out "$RUN"

echo "[1/5] LEFT: phylogeny tree (archive_tree.py)"
$PY archive_animations/archive_tree.py --run "$RUN" --out "$T/tree.mp4" \
  --size "$H" --thumb 9 --per-frame 4 --hold 60 --fps 24

echo "[2/5] MIDDLE: most-branched leaderboard (leaderboard.py)"
$PY archive_animations/leaderboard.py --run "$RUN" --out "$T/lb.mp4" \
  --board branched --layout list --k 12 --cell 64 \
  --quiet-stride 4 --eval-stride 16 --enter-frames 6 --tick-frames 1 \
  --hold 80 --fps 24

echo "[3/5] RIGHT: archive-fills scroll (archive_scroll_lineage.py, pop)"
$PY archive_animations/archive_scroll_lineage.py --run "$RUN" --out "$T/scroll.mp4" \
  --reveal pop --cols 7 --rows-visible 13 --cell 73 --gap 6 \
  --descent-frames 2 --hold 200 --fps 24 --jobs "$JOBS"

echo "[4/5] measure frame counts"
nf () { ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames -of csv=p=0 "$1"; }
FT=$(nf "$T/tree.mp4"); FL=$(nf "$T/lb.mp4"); FS=$(nf "$T/scroll.mp4")
MAX=$FT; [ "$FL" -gt "$MAX" ] && MAX=$FL; [ "$FS" -gt "$MAX" ] && MAX=$FS
echo "  tree=$FT  lb=$FL  scroll=$FS  -> target=$MAX"
# stretch factors (float) to bring each up to MAX
ST=$($PY -c "print($MAX/$FT)"); SL=$($PY -c "print($MAX/$FL)"); SS=$($PY -c "print($MAX/$FS)")

echo "[5/5] composite: scale each to height $H, stretch to $MAX frames, hstack"
ffmpeg -y -loglevel error \
  -i "$T/tree.mp4" -i "$T/lb.mp4" -i "$T/scroll.mp4" -filter_complex "\
   [0:v]setpts=PTS*$ST,fps=24,scale=-2:$H[l];\
   [1:v]setpts=PTS*$SL,fps=24,scale=-2:$H[m];\
   [2:v]setpts=PTS*$SS,fps=24,scale=-2:$H[r];\
   [l][m][r]hstack=inputs=3[v]" \
  -map "[v]" -r 24 -frames:v "$MAX" -c:v libx264 -pix_fmt yuv420p -crf 22 \
  -movflags +faststart "$OUT"

ffmpeg -y -loglevel error -ss 30 -i "$OUT" -frames:v 1 "$T/trio_human.jpg"
echo "DONE: $(du -h "$OUT" | cut -f1)  $OUT"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate -of csv=p=0 "$OUT"
