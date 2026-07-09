# Figures

TikZ sources and rendered assets for the figures that appear in the interactive
blog report (and, where useful, in slides and the top-level `README.md`). None of
these are used by the GECCO camera-ready — **the paper lives in Overleaf**, and its
export is kept, gitignored, under `manuscript/`.

| Directory | Figure |
| --- | --- |
| `system_fig/` | four-quadrant system overview |
| `metric_figs/` | semantic recall, visual coverage, semantic coverage, and the combined isometric stack |
| `intervention_figs/` | the noise / memory / agents triptych |

The PNG assets each `.tex` consumes are generated from run data:

```sh
python tools/build_system_fig_assets.py   # -> figures/system_fig/
python tools/build_metric_fig_assets.py   # -> figures/metric_figs/
```
