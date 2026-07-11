#!/usr/bin/env python3
"""
Check if Picbreeder lineages respect chronological ordering based on numeric PIDs.
Specifically, checks if any child PID is smaller than its parent PID.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm

# Add the parent directory to sys.path to import fer modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    from fer.src.picbreeder_util import load_zip_xml_as_dict
except ImportError:
    # If running from different location, try adjusting path or imports
    import sys
    sys.path.append(str(Path.cwd()))
    try:
        from fer.src.picbreeder_util import load_zip_xml_as_dict
    except ImportError:
        print("Could not import fer.src.picbreeder_util. Make sure you are in the project root or fer/src is in python path.")
        sys.exit(1)

def get_parent_pid(pb_dir: Path, pid: str) -> Optional[str]:
    """
    Reads main.zip to find parent PID.
    """
    main_zip = pb_dir / pid / "main.zip"
    if not main_zip.exists():
        return None
        
    try:
        main_data = load_zip_xml_as_dict(str(main_zip))
    except Exception as exc:
        # tqdm.write(f"[WARN] Failed to read main.zip for {pid}: {exc}")
        return None

    genome_meta = main_data.get("genome", {})
    series = genome_meta.get("series", {})
    
    # Parent PID
    branch_from = series.get("branchFrom")
    if isinstance(branch_from, dict):
        return branch_from.get("@branch")
    return None

def main():
    parser = argparse.ArgumentParser(description="Check lineage chronological ordering.")
    parser.add_argument(
        "--pb-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "spaghetti/pbRender/genomeAll",
        help="Directory that contains pid subdirectories.",
    )
    args = parser.parse_args()
    
    pb_dir = args.pb_dir.expanduser().resolve()
    
    if not pb_dir.is_dir():
        print(f"Error: Directory not found: {pb_dir}")
        sys.exit(1)

    pids = sorted(
        pid.name
        for pid in pb_dir.iterdir()
        if pid.is_dir() and (pid / "main.zip").exists()
    )
    
    print(f"Checking {len(pids)} lineages...")
    
    violations = []
    checked_count = 0
    
    for pid in tqdm(pids):
        try:
            pid_int = int(pid)
        except ValueError:
            continue
            
        parent_pid = get_parent_pid(pb_dir, pid)
        
        if parent_pid:
            try:
                parent_pid_int = int(parent_pid)
                checked_count += 1
                
                if parent_pid_int > pid_int:
                    violations.append((pid_int, parent_pid_int))
            except ValueError:
                pass
    
    print(f"\nChecked {checked_count} lineages with parents.")
    
    if violations:
        print(f"\nFound {len(violations)} violations (where Parent PID > Child PID):")
        # Sort by child PID
        violations.sort(key=lambda x: x[0])
        for child, parent in violations:
            print(f"Child: {child} < Parent: {parent}")
    else:
        print("\nNo violations found. Ordering appears chronological.")

if __name__ == "__main__":
    main()
