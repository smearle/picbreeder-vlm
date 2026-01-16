#!/usr/bin/env python3
"""Compress images/elites in CLIP noun niche runs to images.zip."""
import argparse
from pathlib import Path
import sys

# Add current directory to path to allow imports
sys.path.append(str(Path(__file__).parent))

from clip_noun_niche_shared import compress_run_images

def main():
    parser = argparse.ArgumentParser(description="Compress images/elites in CLIP noun niche runs.")
    parser.add_argument("--path", type=Path, help="Path to run directory or directory containing runs", default="clip_noun_niche_es_logs")
    args = parser.parse_args()

    target = args.path.resolve()
    
    if not target.exists():
        print(f"Path not found: {target}")
        return

    # Check if target is a run directory itself (has state.pkl)
    if (target / "state.pkl").exists():
        print(f"Processing single run: {target.name}")
        compress_run_images(target)
    else:
        # Assume it's a parent directory of runs
        print(f"Scanning {target} for runs...")
        found = False
        for child in target.iterdir():
            if child.is_dir() and (child / "state.pkl").exists():
                found = True
                compress_run_images(child)
        
        if not found:
            print("No runs found (directories with state.pkl).")

if __name__ == "__main__":
    main()
