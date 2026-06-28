#!/usr/bin/env python
"""Pack a run's per-image thumbnails into a few sprite sheets + a tiny sprites.json,
so the blog gallery can render BOTH the chronological and SigLIP-ordered archives
from one download (position lives in layout.json; pixels are addressed by index).

Image i (0-based, sorted by filename) lands at:
  sheet = i // per_sheet ; within = i % per_sheet
  col = within % sheet_cols ; row = within // sheet_cols
  (x, y) = (col*cell, row*cell)
The renderer derives (sheet,x,y) from i the same way, so sprites.json stays tiny.
"""
import argparse, json, math, shutil
from pathlib import Path
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumbs", required=True, help="dir of per-image thumbnails (sorted by name = index order)")
    ap.add_argument("--out", required=True, help="output sprite dir")
    ap.add_argument("--layout", default="", help="optional layout.json to copy alongside (chronological/siglip rc)")
    ap.add_argument("--cell", type=int, default=128)
    ap.add_argument("--sheet-cols", type=int, default=20)
    ap.add_argument("--sheet-rows", type=int, default=20)
    ap.add_argument("--ext", default="webp", choices=["webp", "png", "jpg"])
    args = ap.parse_args()

    thumbs = sorted(p for p in Path(args.thumbs).iterdir()
                    if p.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg"))
    n = len(thumbs)
    per_sheet = args.sheet_cols * args.sheet_rows
    n_sheets = math.ceil(n / per_sheet)
    out = Path(args.out); (out / "sheets").mkdir(parents=True, exist_ok=True)
    cell, cols, rows = args.cell, args.sheet_cols, args.sheet_rows

    sheet_dims = []
    for s in range(n_sheets):
        chunk = thumbs[s * per_sheet:(s + 1) * per_sheet]
        used_rows = math.ceil(len(chunk) / cols)
        W, H = cols * cell, used_rows * cell
        sheet = Image.new("RGB", (W, H), (240, 240, 240))
        for j, tp in enumerate(chunk):
            im = Image.open(tp).convert("RGB")
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.NEAREST)
            sheet.paste(im, ((j % cols) * cell, (j // cols) * cell))
        fn = f"sheet_{s:03d}.{args.ext}"
        save_kw = {"quality": 88, "method": 6} if args.ext == "webp" else {}
        sheet.save(out / "sheets" / fn, **save_kw)
        sheet_dims.append([W, H])

    sprites = {"cell": cell, "sheet_cols": cols, "sheet_rows": rows,
               "per_sheet": per_sheet, "n": n, "n_sheets": n_sheets,
               "ext": args.ext, "sheet_dims": sheet_dims,
               "sheet_tmpl": "sheets/sheet_{s:03d}." + args.ext}
    (out / "sprites.json").write_text(json.dumps(sprites))
    if args.layout:
        shutil.copy(args.layout, out / "layout.json")

    total_mb = sum((out / "sheets" / f"sheet_{s:03d}.{args.ext}").stat().st_size for s in range(n_sheets)) / 1e6
    print(f"packed {n} imgs -> {n_sheets} sheets ({total_mb:.1f} MB total) in {out}")


if __name__ == "__main__":
    main()
