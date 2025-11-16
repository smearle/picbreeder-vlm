#!/usr/bin/env python3
"""
Shared helpers for traversing Picbreeder lineages.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Generator, List, Optional

from fer.src.picbreeder_util import load_zip_xml_as_dict

PID_ZIP_PATTERN = re.compile(r"\d+\.zip$")


def _ensure_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def recursive_parse_all_genomes(root, all_genomes=None):
    """
    Collect every genome dictionary reachable from `root`.
    """
    if all_genomes is None:
        all_genomes = []
    if root is None:
        return all_genomes
    if isinstance(root, list):
        for r in root:
            recursive_parse_all_genomes(r, all_genomes=all_genomes)
        return all_genomes
    if isinstance(root, dict) and "nodes" in root and "links" in root:
        all_genomes.append(root)
        return all_genomes
    if isinstance(root, dict):
        for value in root.values():
            recursive_parse_all_genomes(value, all_genomes=all_genomes)
    return all_genomes


def get_pid_parent(pb_dir: Path, pid: str) -> Optional[str]:
    """
    Read the parent pid from the pid's main.zip metadata.
    """
    zip_file_path = pb_dir / pid / "main.zip"
    root = load_zip_xml_as_dict(str(zip_file_path))
    try:
        return root["genome"]["series"]["branchFrom"]["@branch"]
    except KeyError:
        return None


def get_pid_lineage(pb_dir: Path, pid: str) -> List[str]:
    """
    Traverse parent pointers until the lineage root is reached.
    """
    lineage = []
    while pid is not None:
        lineage.append(pid)
        try:
            pid = get_pid_parent(pb_dir, pid)
        except FileNotFoundError:
            break
    return list(reversed(lineage))


def iter_generation_archives(pb_dir: Path, pid: str) -> Generator[Path, None, None]:
    pid_dir = pb_dir / pid
    if not pid_dir.is_dir():
        return
    for entry in sorted(pid_dir.iterdir()):
        if PID_ZIP_PATTERN.fullmatch(entry.name):
            yield entry


def get_lineage_genomes(pb_dir: Path, pid: str) -> List[Dict]:
    """
    Load every stored genome across the lineage for a pid.
    """
    genomes = []
    for lineage_pid in get_pid_lineage(pb_dir, pid):
        for archive in iter_generation_archives(pb_dir, lineage_pid):
            root = load_zip_xml_as_dict(str(archive))
            generation = (
                root.get("genome", {})
                .get("storage", {})
                .get("generation")
            )
            genomes.extend(recursive_parse_all_genomes(generation))
    return sorted(genomes, key=lambda g: int(g.get("@age", 0)))
