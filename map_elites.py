import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch  # type: ignore
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    pipeline = None  # type: ignore

# Optional YOLO (Ultralytics)
try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore

from PIL import Image, ImageDraw

import neat
from neat.checkpoint import Checkpointer
from neat.population import CompleteExtinctionException

from artifacts import build_generation_state, save_neat_genome_diagrams, save_neat_population
from neat_components import (
    GenerationCheckpointer,
    InteractiveStagnation,
    PicbreederGenome,
    apply_picbreeder_config_defaults,
    seed_initial_population,
    sync_population_output_activations,
)
from picbreeder_reproduction import PicbreederReproduction
from rendering import create_numbered_grid, create_numbered_grid_with_overlays, decode_image, render_genome_diagram, try_load_font, draw_label, draw_bbox


REPO_ROOT = Path(__file__).resolve().parent


# -------------------- YOLO (COCO80) class names --------------------
_COCO80_NAMES: List[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
# -------------------- SigLIP utils --------------------
_SIGLIP_PIPELINE = None


def _get_siglip_pipeline(
    model_name: str = "google/siglip-base-patch16-224",
    device: Optional[str | int] = None,
):
    global _SIGLIP_PIPELINE
    if _SIGLIP_PIPELINE is not None:
        return _SIGLIP_PIPELINE
    if pipeline is None:
        raise ImportError("'transformers' is required for SigLIP scoring.")
    init_kwargs: Dict[str, Any] = {
        "task": "zero-shot-image-classification",
        "model": model_name,
    }
    if device is not None and str(device).lower() != "cpu":
        try:
            init_kwargs["device"] = int(device)
            if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
                init_kwargs["dtype"] = torch.bfloat16  # type: ignore[attr-defined]
        except Exception:
            pass
    _SIGLIP_PIPELINE = pipeline(**init_kwargs)
    return _SIGLIP_PIPELINE


# -------------------- YOLO utils --------------------
_YOLO_MODEL = None


def _get_yolo_model(model_name: str, device: Optional[str] = None):
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    if YOLO is None:
        raise ImportError("'ultralytics' is required for YOLO evaluation. Install 'ultralytics'.")
    _YOLO_MODEL = YOLO(model_name)
    # Device is applied at predict-time in Ultralytics API
    return _YOLO_MODEL

# -------------------- Archive --------------------


def _sanitize_label(label: str) -> str:
    safe = label.strip().lower().replace(" ", "_")
    return "".join(ch for ch in safe if ch.isalnum() or ch in ("-", "_"))


@dataclass
class EliteEntry:
    label: str
    score: float
    generation: int
    index: int
    genome_id: int
    image_path: Path
    diagram_path: Optional[Path]


class LabelArchive:
    def __init__(self, labels: Sequence[str], archive_dir: Path) -> None:
        self.labels = list(labels)
        self.archive_dir = archive_dir
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, EliteEntry] = {}

    def best_score(self, label: str) -> float:
        entry = self._entries.get(label)
        return entry.score if entry is not None else float("-inf")

    def update(
        self,
        *,
        label: str,
        score: float,
        generation: int,
        index: int,
        genome_id: int,
        image: Image.Image,
        genome: neat.DefaultGenome,
        config: neat.Config,
    ) -> EliteEntry:
        stem = _sanitize_label(label)
        image_path = self.archive_dir / f"{stem}.png"
        image.save(image_path, format="PNG")

        diag_stem = self.archive_dir / f"{stem}_genome"
        diagram_path: Optional[Path]
        try:
            diagram_path = render_genome_diagram(genome, config, diag_stem, fmt="svg")
        except Exception:
            diagram_path = None

        entry = EliteEntry(
            label=label,
            score=float(score),
            generation=int(generation),
            index=int(index),
            genome_id=int(genome_id),
            image_path=image_path,
            diagram_path=diagram_path,
        )
        self._entries[label] = entry
        self._write_summary()
        return entry

    def to_summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for label, entry in self._entries.items():
            out[label] = {
                "score": entry.score,
                "generation": entry.generation,
                "index": entry.index,
                "genome_id": entry.genome_id,
                "image_path": str(entry.image_path),
                "diagram_path": str(entry.diagram_path) if entry.diagram_path else None,
            }
        return out

    def _write_summary(self) -> None:
        (self.archive_dir / "elites.json").write_text(
            json.dumps(self.to_summary(), indent=2), encoding="utf-8"
        )

    def render_archive_image(self, thumb_size: int, columns: Optional[int] = None) -> Image.Image:
        labels = list(self.labels)
        count = len(labels)
        if count <= 0:
            return Image.new("RGB", (thumb_size, thumb_size), (16, 16, 20))

        if columns is None or columns < 1:
            columns = min(count, 8)
        rows = (count + columns - 1) // columns

        margin = 12
        width = (columns * thumb_size) + ((columns + 1) * margin)
        height = (rows * thumb_size) + ((rows + 1) * margin)
        canvas = Image.new("RGBA", (width, height), (16, 16, 20, 255))
        draw = ImageDraw.Draw(canvas)
        font = try_load_font(20)

        for i, label in enumerate(labels):
            row = i // columns
            col = i % columns
            x = margin + col * (thumb_size + margin)
            y = margin + row * (thumb_size + margin)

            entry = self._entries.get(label)
            if entry and entry.image_path.exists():
                try:
                    img = Image.open(entry.image_path).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (thumb_size, thumb_size), (30, 30, 40))
            else:
                img = Image.new("RGB", (thumb_size, thumb_size), (30, 30, 40))

            if img.size != (thumb_size, thumb_size):
                try:
                    img = img.resize((thumb_size, thumb_size))
                except Exception:
                    img = Image.new("RGB", (thumb_size, thumb_size), (30, 30, 40))

            canvas.paste(img, (x, y))
            draw_label(draw, (x + 6, y + 6), label, font)

        return canvas.convert("RGB")

    def get_entry(self, label: str) -> Optional[EliteEntry]:
        return self._entries.get(label)

    def render_archive_image_with_overlays(
        self,
        thumb_size: int,
        overlays_by_label: Dict[str, Dict[str, Any]],
        columns: Optional[int] = None,
        highlight_labels: Optional[Sequence[str]] = None,
    ) -> Image.Image:
        labels = list(self.labels)
        count = len(labels)
        if count <= 0:
            return Image.new("RGB", (thumb_size, thumb_size), (16, 16, 20))

        if columns is None or columns < 1:
            columns = min(count, 8)
        rows = (count + columns - 1) // columns

        margin = 12
        width = (columns * thumb_size) + ((columns + 1) * margin)
        height = (rows * thumb_size) + ((rows + 1) * margin)
        canvas = Image.new("RGBA", (width, height), (16, 16, 20, 255))
        draw = ImageDraw.Draw(canvas)
        font = try_load_font(20)
        highlight_set = set(highlight_labels) if highlight_labels else set()

        for i, label in enumerate(labels):
            row = i // columns
            col = i % columns
            x = margin + col * (thumb_size + margin)
            y = margin + row * (thumb_size + margin)

            entry = self._entries.get(label)
            if entry and entry.image_path.exists():
                try:
                    img = Image.open(entry.image_path).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (thumb_size, thumb_size), (30, 30, 40))
            else:
                img = Image.new("RGB", (thumb_size, thumb_size), (30, 30, 40))

            src_w, src_h = img.size
            if img.size != (thumb_size, thumb_size):
                try:
                    img = img.resize((thumb_size, thumb_size))
                except Exception:
                    img = Image.new("RGB", (thumb_size, thumb_size), (30, 30, 40))
            canvas.paste(img, (x, y))

            draw_label(draw, (x + 6, y + 6), label, font)

            # Overlay bbox for this label if provided
            det = overlays_by_label.get(label)
            if det and det.get("box") is not None and src_w > 0 and src_h > 0:
                bx1, by1, bx2, by2 = det["box"]
                sx = thumb_size / float(src_w)
                sy = thumb_size / float(src_h)
                gx1 = x + (float(bx1) * sx)
                gy1 = y + (float(by1) * sy)
                gx2 = x + (float(bx2) * sx)
                gy2 = y + (float(by2) * sy)
                score = det.get("score")
                text = f"{label} {score:.2f}" if isinstance(score, (int, float)) else label
                color = (0, 255, 0) if label in highlight_set else (0, 200, 255)
                draw_bbox(draw, (gx1, gy1, gx2, gy2), text, color=color, width=3, font=try_load_font(16))

        return canvas.convert("RGB")


# -------------------- Evolver --------------------


class MapElitesEvolver:
    def __init__(
        self,
        population: neat.Population,
        rows: int,
        cols: int,
        thumb_size: int,
        scheme: str,
        palette: str,
        experiment_dir: Path,
        population_dir: Path,
        query_dir: Path,
        labels: Sequence[str],
        *,
        select_k: int = 1,
        siglip_model: str = "google/siglip-base-patch16-224",
        siglip_device: Optional[str] = None,
        eval_mode: str = "siglip",
        yolo_model: str = "yolov8n.pt",
        yolo_device: Optional[str] = None,
        yolo_confidence: float = 0.1,
        prefer_elites_selection: bool = False,
        render_diagrams: bool = False,
        mutation_mode: str = "all",
        mutation_alternate_period: int = 10,
    ) -> None:
        self.population = population
        self.rows = rows
        self.cols = cols
        self.thumb_size = thumb_size
        self.scheme = scheme
        self.palette = palette
        self.experiment_dir = experiment_dir
        self.population_dir = population_dir
        self.query_dir = query_dir
        self.population_size = rows * cols
        self.select_k = max(1, int(select_k))
        self.siglip_model = siglip_model
        self.siglip_device = siglip_device
        self._clf = None
        self._render_diagrams = bool(render_diagrams)
        self.eval_mode = eval_mode
        self.yolo_model = yolo_model
        self.yolo_device = yolo_device
        self.yolo_confidence = float(yolo_confidence)
        self._yolo = None
        self.prefer_elites_selection = bool(prefer_elites_selection)

        self.archive = LabelArchive(labels, experiment_dir / "elites")
        self.metrics_dir = experiment_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.scores_path = self.metrics_dir / "label_scores.jsonl"
        # Mutation alternation settings
        self.mutation_mode = str(mutation_mode).lower()
        try:
            self.mutation_alternate_period = max(1, int(mutation_alternate_period))
        except Exception:
            self.mutation_alternate_period = 10

    def _ensure_siglip(self):
        if self._clf is None:
            self._clf = _get_siglip_pipeline(self.siglip_model, self.siglip_device)
        return self._clf

    def _score_images_for_labels(
        self, images: Sequence[Image.Image], labels: Sequence[str]
    ) -> List[Dict[str, float]]:
        clf = self._ensure_siglip()
        outputs: List[List[Dict[str, Any]]] = clf(images, candidate_labels=list(labels))
        scores_per_image: List[Dict[str, float]] = []
        for out in outputs:
            mapping: Dict[str, float] = {}
            for item in out:
                try:
                    mapping[str(item["label"])] = float(item["score"])  # type: ignore[index]
                except Exception:
                    continue
            scores_per_image.append(mapping)
        return scores_per_image

    def _ensure_yolo(self):
        if self._yolo is None:
            self._yolo = _get_yolo_model(self.yolo_model, self.yolo_device)
        return self._yolo

    def _score_images_with_yolo(self, images: Sequence[Image.Image]):
        """
        Returns (per-image mapping: {class_name: max_confidence}, per-image best det dict)
        Best det dict: {label, score, box=(x1,y1,x2,y2)} or None if no detections.
        """
        model = self._ensure_yolo()
        try:
            results = model.predict(
                images,
                conf=self.yolo_confidence,
                imgsz=max(384, self.thumb_size),
                device=self.yolo_device if self.yolo_device else None,
                verbose=False,
                half=False,
            )
        except TypeError:
            # Fallback for older ultralytics versions without 'device' kw
            results = model.predict(
                images,
                conf=self.yolo_confidence,
                imgsz=max(384, self.thumb_size),
                verbose=False,
            )

        per_image_scores: List[Dict[str, float]] = []
        per_image_best: List[Optional[Dict[str, Any]]] = []
        for res in results:
            name_map = getattr(res, "names", None) or {}
            mapping: Dict[str, float] = {}
            best: Optional[Dict[str, Any]] = None
            try:
                boxes = res.boxes  # type: ignore[attr-defined]
                if boxes is not None:
                    cls_list = boxes.cls.cpu().numpy().tolist() if hasattr(boxes.cls, "cpu") else []
                    conf_list = boxes.conf.cpu().numpy().tolist() if hasattr(boxes.conf, "cpu") else []
                    xyxy_list = boxes.xyxy.cpu().numpy().tolist() if hasattr(boxes, "xyxy") else []
                    for cls_id, conf, xyxy in zip(cls_list, conf_list, xyxy_list):
                        cls_i = int(cls_id)
                        label = (
                            name_map[cls_i]
                            if isinstance(name_map, dict) and cls_i in name_map
                            else str(cls_i)
                        )
                        prev = mapping.get(label, 0.0)
                        if float(conf) > prev:
                            mapping[label] = float(conf)
                        if best is None or float(conf) > best.get("score", -1.0):
                            best = {"label": label, "score": float(conf), "box": tuple(map(float, xyxy))}
            except Exception:
                pass
            per_image_scores.append(mapping)
            per_image_best.append(best)
        return per_image_scores, per_image_best

    def _yolo_best_det_for_each_label(
        self, images: Sequence[Image.Image], target_labels: Sequence[str]
    ) -> List[Optional[Dict[str, Any]]]:
        model = self._ensure_yolo()
        try:
            results = model.predict(
                images,
                conf=self.yolo_confidence,
                imgsz=max(384, self.thumb_size),
                device=self.yolo_device if self.yolo_device else None,
                verbose=False,
                half=False,
            )
        except TypeError:
            results = model.predict(
                images,
                conf=self.yolo_confidence,
                imgsz=max(384, self.thumb_size),
                verbose=False,
            )
        out: List[Optional[Dict[str, Any]]] = []
        for res, want_label in zip(results, target_labels):
            name_map = getattr(res, "names", None) or {}
            best: Optional[Dict[str, Any]] = None
            try:
                boxes = res.boxes  # type: ignore[attr-defined]
                if boxes is not None:
                    cls_list = boxes.cls.cpu().numpy().tolist() if hasattr(boxes.cls, "cpu") else []
                    conf_list = boxes.conf.cpu().numpy().tolist() if hasattr(boxes.conf, "cpu") else []
                    xyxy_list = boxes.xyxy.cpu().numpy().tolist() if hasattr(boxes, "xyxy") else []
                    for cls_id, conf, xyxy in zip(cls_list, conf_list, xyxy_list):
                        cls_i = int(cls_id)
                        label = (
                            name_map[cls_i]
                            if isinstance(name_map, dict) and cls_i in name_map
                            else str(cls_i)
                        )
                        if str(label) != str(want_label):
                            continue
                        if best is None or float(conf) > best.get("score", -1.0):
                            best = {"label": label, "score": float(conf), "box": tuple(map(float, xyxy))}
            except Exception:
                best = None
            out.append(best)
        return out

    def _random_select_indices(self, total: int, k: int) -> List[int]:
        k = max(1, min(k, total))
        return random.sample(list(range(total)), k=k)

    def evaluate_generation(
        self, genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config
    ) -> None:
        generation = int(self.population.generation)
        # Apply alternating mutation mode schedule (affects reproduction to next gen)
        if getattr(self, "mutation_mode", "all") == "alternate":
            period = max(1, int(getattr(self, "mutation_alternate_period", 10)))
            block_index = (generation // period)
            next_mode = "color_only" if (block_index % 2 == 0) else "structure_only"
            try:
                config.genome_config.picbreeder_mutation_mode = next_mode
            except Exception:
                pass
            print(f"Alternating mutation mode (period={period}) → next reproduction: {next_mode}")
        if len(genomes) != self.population_size:
            raise ValueError(
                f"Expected {self.population_size} genomes, received {len(genomes)}."
            )

        print(f"\n--- Generation {generation} ---")
        state, cache = build_generation_state(
            genomes,
            config,
            generation,
            self.rows,
            self.cols,
            self.thumb_size,
            self.scheme,
            self.palette,
        )

        save_neat_population(state, self.population_dir, generation, cache)
        if self._render_diagrams:
            try:
                save_neat_genome_diagrams(genomes, config, self.population_dir, generation)
            except Exception as exc:
                print(f"Skipping genome diagram export: {exc}")
                self._render_diagrams = False

        # Build PIL images for evaluation
        images: List[Image.Image] = [decode_image(entry) for entry in state["images"]]

        # Score each image against all labels
        labels = self.archive.labels
        yolo_best: Optional[List[Optional[Dict[str, Any]]]] = None
        if self.eval_mode == "yolo":
            scores_per_image, yolo_best = self._score_images_with_yolo(images)
        else:
            scores_per_image = self._score_images_for_labels(images, labels)

        # For each label, find best candidate and update archive if improved
        updates: List[EliteEntry] = []
        for label_index, label in enumerate(labels):
            best_score = float("-inf")
            best_idx = -1
            for idx, label_scores in enumerate(scores_per_image):
                score = float(label_scores.get(label, 0.0))
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0 and best_score > self.archive.best_score(label):
                genome_id, genome = genomes[best_idx]
                entry = self.archive.update(
                    label=label,
                    score=best_score,
                    generation=generation,
                    index=best_idx,
                    genome_id=genome_id,
                    image=images[best_idx],
                    genome=genome,
                    config=config,
                )
                updates.append(entry)

        if updates:
            msg = ", ".join(
                f"{u.label}={u.score:.4f} (idx {u.index})" for u in updates
            )
            print(f"Improved elites: {msg}")
            # Render the entire archive image for this generation
            try:
                archive_img = self.archive.render_archive_image(self.thumb_size)
                arc_path = self.experiment_dir / "elites" / f"gen_{generation:03d}_archive.png"
                archive_img.save(arc_path, format="PNG")
                print(f"Archive image saved to {arc_path}")
                # YOLO overlay variant of archive grid
                if self.eval_mode == "yolo":
                    # Collect existing entries and images
                    labels_with_entries: List[str] = []
                    images_for_entries: List[Image.Image] = []
                    for lbl in labels:
                        entry = self.archive.get_entry(lbl)
                        if entry and entry.image_path.exists():
                            try:
                                img = Image.open(entry.image_path).convert("RGB")
                            except Exception:
                                continue
                            labels_with_entries.append(lbl)
                            images_for_entries.append(img)
                    if images_for_entries:
                        dets = self._yolo_best_det_for_each_label(images_for_entries, labels_with_entries)
                        overlays_by_label: Dict[str, Dict[str, Any]] = {}
                        for lbl, det in zip(labels_with_entries, dets):
                            if det is not None:
                                overlays_by_label[lbl] = det
                        # Highlight labels that just updated this generation
                        updated_labels = [u.label for u in updates]
                        archive_overlay = self.archive.render_archive_image_with_overlays(
                            self.thumb_size,
                            overlays_by_label,
                            highlight_labels=updated_labels,
                        )
                        arc_path_overlay = self.experiment_dir / "elites" / f"gen_{generation:03d}_archive_yolo.png"
                        archive_overlay.save(arc_path_overlay, format="PNG")
                        print(f"YOLO archive image saved to {arc_path_overlay}")
            except Exception as exc:
                print(f"Failed to render archive image: {exc}")
        else:
            print("No elite improvements this generation.")

        # Write per-generation label scores summary (best per label this gen)
        best_by_label: Dict[str, float] = {}
        for label in labels:
            best_for_label = max(
                (float(m.get(label, 0.0)) for m in scores_per_image), default=0.0
            )
            best_by_label[label] = best_for_label
        with self.scores_path.open("a", encoding="utf-8") as fp:
            fp.write(
                json.dumps({
                    "generation": generation,
                    "best_scores": best_by_label,
                })
            )
            fp.write("\n")

        # Selection for reproduction
        selection_count = max(1, min(self.select_k, len(genomes)))
        selected_indices: List[int]
        if self.prefer_elites_selection and updates:
            updated_indices = sorted({u.index for u in updates})
            if updated_indices:
                take = min(selection_count, len(updated_indices))
                selected_indices = random.sample(updated_indices, take)
                # Fill any remainder randomly from non-selected
                if take < selection_count:
                    remaining_pool = [i for i in range(len(genomes)) if i not in selected_indices]
                    selected_indices.extend(self._random_select_indices(len(remaining_pool), selection_count - take))
                    # Map pooled indices back to global indices
                    selected_indices = selected_indices[:take] + [remaining_pool[i] for i in selected_indices[take:]]
            else:
                selected_indices = self._random_select_indices(len(genomes), selection_count)
        else:
            selected_indices = self._random_select_indices(len(genomes), selection_count)
        for idx, (_, genome) in enumerate(genomes):
            genome.fitness = 1.0 if idx in selected_indices else 0.0

        # Save grid and selection overlays
        self.query_dir.mkdir(parents=True, exist_ok=True)
        grid = create_numbered_grid(state)
        grid.save(self.query_dir / f"gen_{generation:03d}_grid.png", format="PNG")
        selection = create_numbered_grid(state, selected=selected_indices)
        selection.save(
            self.query_dir / f"gen_{generation:03d}_selection.png", format="PNG"
        )
        # YOLO overlay grid for selection
        if self.eval_mode == "yolo" and yolo_best is not None:
            overlays: Dict[int, Dict[str, Any]] = {}
            for idx, det in enumerate(yolo_best):
                if det is None:
                    continue
                overlays[idx] = {
                    "box": det.get("box"),
                    "label": det.get("label"),
                    "score": det.get("score"),
                }
            # Determine highlight: if an image's top det label produced a new elite at the same index
            update_pairs = {(e.label, e.index) for e in updates}
            highlight_indices = []
            for idx, det in enumerate(yolo_best):
                if det is None:
                    continue
                if (str(det.get("label")), idx) in update_pairs:
                    highlight_indices.append(idx)
            sel_yolo = create_numbered_grid_with_overlays(
                state,
                overlays,
                selected=selected_indices,
                highlight_indices=highlight_indices,
            )
            sel_yolo.save(
                self.query_dir / f"gen_{generation:03d}_selection_yolo.png",
                format="PNG",
            )
        if self.prefer_elites_selection and updates:
            print(f"Selected indices (prefer-elites): {selected_indices}")
        else:
            print(f"Selected indices (random baseline): {selected_indices}")


# -------------------- CLI and main --------------------


def _default_config_path(scheme: str) -> Path:
    base = REPO_ROOT / "picture2d"
    name = "interactive_config_color" if scheme == "color" else "interactive_config_gray"
    return (base / name).resolve()


def _parse_labels(args: argparse.Namespace) -> List[str]:
    # Prefer labels from file when provided (supports long captions with spaces)
    if args.labels_file:
        path = Path(args.labels_file).resolve()
        if not path.exists():
            raise SystemExit(f"Labels file not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        labels: List[str] = []
        if text.startswith("["):
            try:
                items = json.loads(text)
                if isinstance(items, list):
                    labels = [str(x).strip() for x in items if str(x).strip()]
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in labels file: {exc}") from exc
        else:
            # Newline-separated, allow comments with leading '#'
            for line in text.splitlines():
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                labels.append(raw)
    else:
        # Fallback to CLI list; to pass spaces, users must quote each label
        labels = []
        if args.labels:
            if isinstance(args.labels, list):
                labels.extend([str(p).strip() for p in args.labels if str(p).strip()])
            elif isinstance(args.labels, str):  # fallback for comma-separated usage
                parts = [p.strip() for p in args.labels.split(",")]
                labels.extend([p for p in parts if p])

    # De-duplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for label in labels:
        if label not in seen:
            uniq.append(label)
            seen.add(label)
    if not uniq:
        raise SystemExit("No labels provided. Use --labels and/or --labels-file.")
    return uniq


def _build_yolo_labels(max_classes: int) -> List[str]:
    n = int(max_classes)
    if n < 1:
        n = 1
    return list(_COCO80_NAMES[: min(n, len(_COCO80_NAMES))])


def _build_experiment_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "eval_mode", "siglip") == "yolo":
        label_count = int(getattr(args, "yolo_max_classes", 0) or 0)
        mode_tag = "yolo"
    else:
        label_count = len(getattr(args, "_labels", []) or [])
        mode_tag = "siglip"
    slug = (
        f"mapelites_r{args.rows}_c{args.cols}_ts{args.thumb_size}_nlbl{label_count}_{mode_tag}"
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = Path("logs") / f"{slug}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def _write_run_metadata(experiment_dir: Path, args: argparse.Namespace) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": args.rows,
        "cols": args.cols,
        "thumb_size": args.thumb_size,
        "generations_planned": args.generations,
        "scheme": args.scheme,
        "color_palette": args.color_palette,
        "config_path": str(args.config_path) if args.config_path else None,
        "labels": args._labels,
        "select_k": args.select_k,
        "eval_mode": getattr(args, "eval_mode", "siglip"),
        "siglip_model": getattr(args, "siglip_model", None),
        "siglip_device": getattr(args, "siglip_device", None),
        "yolo_model": getattr(args, "yolo_model", None),
        "yolo_device": getattr(args, "yolo_device", None),
        "yolo_confidence": getattr(args, "yolo_confidence", None),
        "yolo_max_classes": getattr(args, "yolo_max_classes", None),
        "prefer_elites_selection": getattr(args, "prefer_elites_selection", False),
        "mutation_mode": getattr(args, "mutation_mode", "all"),
        "mutation_mask": getattr(args, "mutation_mask", "strict"),
        "mutation_alternate_period": getattr(args, "mutation_alternate_period", None),
    }
    (experiment_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MAP-Elites with NEAT image generator. Eval modes: SigLIP (text labels) or YOLO (closed classes)."
        )
    )
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--thumb-size", type=int, default=128)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--scheme", choices=["color", "gray", "mono"], default="color")
    parser.add_argument("--color-palette", dest="color_palette", default="hsb")
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--output-activations", action="store_true", dest="output_activations")
    parser.add_argument("--eval-mode", choices=["siglip", "yolo"], default="siglip")
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        help=(
            "Labels as list: e.g. --labels dog cat ballerina. "
            "For multi-word labels, quote each (e.g. --labels 'golden retriever' 'tabby cat') "
            "or use --labels-file."
        ),
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        help="Optional file with JSON array or newline-separated labels",
    )
    parser.add_argument("--select-k", type=int, default=1, help="Random baseline: number of parents")
    parser.add_argument("--siglip-model", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--siglip-device", type=str)
    # YOLO options
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt")
    parser.add_argument("--yolo-device", type=str)
    parser.add_argument("--yolo-confidence", type=float, default=0.10)
    parser.add_argument(
        "--yolo-max-classes",
        type=int,
        default=80,
        help="Max number of YOLO classes to include in archive (set 80 for full COCO)",
    )
    parser.add_argument(
        "--prefer-elites-selection",
        action="store_true",
        help="If any elites improve this generation, select those genomes as parents (fill remainder randomly)",
    )
    parser.add_argument("--render-diagrams", action="store_true")
    # Mutation scoping
    parser.add_argument(
        "--mutation-mode",
        choices=["all", "color_only", "structure_only", "alternate"],
        default="all",
        help="Mutation scope: 'all', color-only (H/S), structure-only (B), or alternate blocks",
    )
    parser.add_argument(
        "--mutation-mask",
        choices=["strict", "soft"],
        default="strict",
        help="Policy for channel-masked mutation. Strict = avoid entangled genes.",
    )
    parser.add_argument(
        "--mutation-alternate-period",
        type=int,
        default=10,
        help="When --mutation-mode=alternate, number of generations per block before switching",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rows < 1 or args.cols < 1:
        raise ValueError("rows and cols must be positive integers")
    if args.thumb_size < 8:
        raise ValueError("thumb-size must be at least 8")
    if args.generations < 1:
        raise ValueError("generations must be at least 1")
    if args.select_k is not None and args.select_k < 1:
        raise ValueError("select-k must be at least 1")
    if getattr(args, "eval_mode", "siglip") == "yolo":
        if getattr(args, "yolo_max_classes", 0) < 1:
            raise ValueError("yolo-max-classes must be at least 1 for YOLO mode")
    if getattr(args, "mutation_mode", "all") == "alternate":
        if getattr(args, "mutation_alternate_period", 0) < 1:
            raise ValueError("mutation-alternate-period must be at least 1 when using alternate mode")


def run(args: argparse.Namespace, experiment_dir: Path, population_dir: Path, query_dir: Path) -> None:
    # Resolve config path
    if args.config_path is None:
        args.config_path = _default_config_path(args.scheme)
    else:
        args.config_path = args.config_path.resolve()

    # Build NEAT config and population
    config = neat.Config(
        PicbreederGenome,
        PicbreederReproduction,
        neat.DefaultSpeciesSet,
        InteractiveStagnation,
        str(args.config_path),
    )
    apply_picbreeder_config_defaults(config, enable_output_activations=bool(args.output_activations))
    config.pop_size = args.rows * args.cols
    # Apply mutation scoping preferences before creating population
    try:
        config.genome_config.picbreeder_mutation_mode = str(getattr(args, "mutation_mode", "all"))
    except Exception:
        pass
    try:
        config.genome_config.picbreeder_mask_policy = str(getattr(args, "mutation_mask", "strict"))
    except Exception:
        pass
    population = neat.Population(config)
    sync_population_output_activations(population, config.genome_config)
    seed_initial_population(population, config.genome_config)

    # Reporters
    population.add_reporter(GenerationCheckpointer(population_dir))

    evolver = MapElitesEvolver(
        population,
        args.rows,
        args.cols,
        args.thumb_size,
        args.scheme,
        args.color_palette,
        experiment_dir,
        population_dir,
        query_dir,
        labels=args._labels,
        select_k=args.select_k,
        siglip_model=args.siglip_model,
        siglip_device=args.siglip_device,
        eval_mode=args.eval_mode,
        yolo_model=args.yolo_model,
        yolo_device=args.yolo_device,
        yolo_confidence=args.yolo_confidence,
        prefer_elites_selection=args.prefer_elites_selection,
        render_diagrams=args.render_diagrams,
        mutation_mode=getattr(args, "mutation_mode", "all"),
        mutation_alternate_period=getattr(args, "mutation_alternate_period", 10),
    )

    try:
        population.run(evolver.evaluate_generation, args.generations)
    except CompleteExtinctionException as exc:
        raise SystemExit("Population went extinct; evolution cannot continue.") from exc

    print(f"\nRun complete. Next generation index: {population.generation}")
    # Save final archive image in metrics
    try:
        final_img = evolver.archive.render_archive_image(args.thumb_size)
        final_path = evolver.metrics_dir / "final_elites_archive.png"
        final_img.save(final_path, format="PNG")
        print(f"Final elites archive saved to {final_path}")
    except Exception as exc:
        print(f"Failed to save final elites archive: {exc}")


def main() -> None:
    args = parse_args()
    try:
        if getattr(args, "eval_mode", "siglip") == "yolo":
            args._labels = _build_yolo_labels(getattr(args, "yolo_max_classes", 80))
        else:
            args._labels = _parse_labels(args)
        validate_args(args)
    except Exception as exc:
        raise SystemExit(f"Argument error: {exc}") from exc

    experiment_dir = _build_experiment_dir(args)
    population_dir = experiment_dir / "populations"
    query_dir = experiment_dir / "queries"
    population_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)

    _write_run_metadata(experiment_dir, args)

    run(args, experiment_dir, population_dir, query_dir)


if __name__ == "__main__":
    main()


