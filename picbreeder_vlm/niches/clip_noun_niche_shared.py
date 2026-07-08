from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

from picbreeder_vlm.niches.clip_noun_niche_config import ClipNounNicheConfig


def compress_run_images(run_dir: Path) -> None:
    """Compress images/ and elites/ directories into images.zip and remove originals."""
    zip_path = run_dir / "images.zip"
    paths_to_zip = [run_dir / "images", run_dir / "elites"]
    
    # Only compress if directories exist
    if not any(p.exists() for p in paths_to_zip):
        return

    print(f"Compressing images in {run_dir} to {zip_path}...")
    temp_zip = zip_path.with_suffix(".tmp.zip")
    try:
        with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in paths_to_zip:
                if p.exists():
                    for file_path in p.rglob("*"):
                        if file_path.is_file():
                            arcname = file_path.relative_to(run_dir)
                            zf.write(file_path, arcname)
        temp_zip.rename(zip_path)
    except Exception as e:
        if temp_zip.exists():
            temp_zip.unlink()
        raise e
    
    # Remove original directories
    for p in paths_to_zip:
        if p.exists():
            shutil.rmtree(p)
    print(f"Compression complete.")


def decompress_run_images(run_dir: Path, remove_zip: bool = True) -> None:
    """Decompress images.zip into run_dir and optionally remove the zip file."""
    zip_path = run_dir / "images.zip"
    if not zip_path.exists():
        return

    print(f"Decompressing images in {run_dir} from {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(run_dir)
    
    if remove_zip:
        zip_path.unlink()
    print("Decompression complete.")


def sanitize_tag(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return cleaned or "run"


def build_run_name(cfg: ClipNounNicheConfig) -> str:
    noun_tag = sanitize_tag(Path(cfg.nounlist).stem)
    config_tag = sanitize_tag(Path(cfg.config).stem)
    clip_tag = f"{sanitize_tag(cfg.clip_model)}-{sanitize_tag(cfg.clip_pretrained)}"
    parts = [
        f"nouns-{noun_tag}",
        f"stage{cfg.stage_length}",
        f"render{cfg.render_size}",
        f"batch{cfg.batch_size}",
        f"mut{cfg.mutation_strength}",
        f"rand{cfg.new_random_prob}",
        f"clip-{clip_tag}",
        f"cfg-{config_tag}",
    ]
    if cfg.crossover_strength > 0:
        parts.append(f"cross{cfg.crossover_strength}")
    if cfg.seed is not None:
        parts.append(f"seed{cfg.seed}")
    if cfg.save_images:
        parts.append("imgs")
        if cfg.save_offspring_grids:
            parts.append("grids")
    if cfg.save_offspring_grids and not cfg.save_images:
        parts.append("grids_flag_noimg")
    return "_".join(parts)


def resolve_path(path: Path, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p
