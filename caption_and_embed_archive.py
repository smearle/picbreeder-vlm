#!/usr/bin/env python3
"""
Caption archive images using a VLM and embed the captions into a text embedding space.
Calculates coverage metrics on the text embeddings.
"""

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch_fpsample
import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from hydra.conf import HelpConf, HydraConf
from hydra.core.config_store import ConfigStore
from PIL import Image
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from config import PicbreederConfig
from vlm_backends import create_vlm_backend, VLMBackend
from rendering import create_captioned_grid

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CaptionEmbedConfig(PicbreederConfig):
    archive_path: str = "logs_collaborative/latest/archive" # Path to archive images
    caption_model: str = "gemini-2.5-pro"
    caption_batch_size: int = 64
    # embedding_model: str = "ViT-B-32"
    # embedding_pretrained: str = "laion2b_s34b_b79k"
    embedding_model: str = "gemini-embedding-001"
    embedding_pretrained: str = ""
    caption_prompt: str = "Provide a short description (one or a few words only) for each image."
    max_images: Optional[int] = None
    k_covering_ks: str = "1,5,10,20,50,100" # Comma separated list of k values for coverage radius
    resume_captions: bool = True
    render_grid: bool = False
    grid_thumb_size: int = 128
    hydra: HydraConf = field(
        default_factory=lambda: HydraConf(
            help=HelpConf(
                app_name="caption_and_embed_archive",
                header="Caption archive images with VLM and analyze text embeddings.",
                footer="Use +archive_path=... to specify the image archive.",
            )
        )
    )

ConfigStore.instance().store(name="caption_embed_config", node=CaptionEmbedConfig)

def load_images(archive_path: Path, max_images: Optional[int] = None) -> List[Path]:
    """Load image paths from the archive directory."""
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive path {archive_path} does not exist.")
    
    # Support both flat directory and 'archive/images' structure
    if (archive_path / "images").exists():
        search_path = archive_path / "images"
    else:
        search_path = archive_path

    image_extensions = {".png", ".jpg", ".jpeg"}
    images = sorted([
        p for p in search_path.iterdir() 
        if p.suffix.lower() in image_extensions
    ])
    
    if max_images:
        images = images[:max_images]
    
    logger.info(f"Found {len(images)} images in {search_path}")
    return images

def parse_json_response(text: str) -> Dict[str, Any]:
    """Extract and parse JSON from VLM response text."""
    # Try to find JSON block
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Try to find first { and last }
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = text
            
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON: {text}")
        return {}

def caption_images(
    backend: VLMBackend, 
    image_paths: List[Path], 
    batch_size: int, 
    prompt: str,
    output_path: Path,
    resume: bool = True
) -> Dict[str, str]:
    """Caption images in batches and save results incrementally."""
    
    captions = {}
    if resume and output_path.exists():
        try:
            with open(output_path, "r") as f:
                captions = json.load(f)
            logger.info(f"Resumed {len(captions)} captions from {output_path}")
        except Exception as e:
            logger.warning(f"Failed to resume captions: {e}")

    # Filter out already captioned images
    to_process = [p for p in image_paths if p.name not in captions]
    if not to_process:
        logger.info("All images already captioned.")
        return captions

    logger.info(f"Captioning {len(to_process)} images...")

    for i in tqdm(range(0, len(to_process), batch_size)):
        batch_paths = to_process[i : i + batch_size]
        batch_images = []
        batch_labels = []
        
        for idx, p in enumerate(batch_paths):
            try:
                with open(p, "rb") as f:
                    batch_images.append(f.read())
                batch_labels.append(f"Image {idx+1}")
            except Exception as e:
                logger.error(f"Failed to read image {p}: {e}")

        if not batch_images:
            continue

        system_instruction = (
            "You are a helpful assistant that captions images. "
            "Output valid JSON only. "
            "The format should be: {\"captions\": [\"caption1\", \"caption2\", ...]} "
            "where the order corresponds to the images provided."
        )

        full_prompt = f"{prompt} Return a JSON object with a 'captions' key containing a list of strings."

        try:
            # Retry loop for rate limits handled by backend, but we handle logic errors
            response = backend.query_multiple(
                image_bytes_list=batch_images,
                captions=batch_labels,
                prompt=full_prompt,
                system_instruction=system_instruction
            )
            
            result = parse_json_response(response.text)
            batch_captions = result.get("captions", [])
            
            if len(batch_captions) != len(batch_paths):
                logger.warning(f"Mismatch in captions count: expected {len(batch_paths)}, got {len(batch_captions)}. Saving raw response text as caption.")
                # Fallback or manual handle? We'll skip or use empty strings to keep alignment if partial? 
                # If mismatch, it's safer to discard or try to align if close.
                # For now, if mismatch, just log error and maybe set empty for this batch to avoid shifting.
                # Actually, better to retry or fail? 
                # Let's try to map what we have.
                pass
            
            for p, cap in zip(batch_paths, batch_captions):
                captions[p.name] = cap
                
            # Save incrementally
            with open(output_path, "w") as f:
                json.dump(captions, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error processing batch starting at {batch_paths[0]}: {e}")
            time.sleep(5) # Cooldown on error

    return captions

def embed_captions(
    captions: Dict[str, str], 
    model_name: str, 
    pretrained: str
) -> Tuple[Dict[str, List[float]], Any]:
    """Embed captions using OpenCLIP or Hugging Face Transformers."""
    
    logger.info(f"Loading embedding model {model_name} ({pretrained})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = None
    tokenizer = None
    use_open_clip = True

    # Detect if model is supported by OpenCLIP
    if model_name in open_clip.list_models():
        logger.info(f"Detected OpenCLIP model: {model_name}")
        # If pretrained is empty, convert to None for OpenCLIP
        pt = pretrained if pretrained else None
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pt, device=device
            )
            tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model: {e}")
            raise e
    else:
        logger.info(f"Model '{model_name}' not found in OpenCLIP. Assuming Hugging Face Transformers.")
        use_open_clip = False
        try:
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
            
            # Set pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
        except Exception as e:
             logger.error(f"Failed to load Transformers model: {e}")
             raise e

    embeddings = {}
    
    # Process in batches for efficiency
    batch_size = 32
    items = list(captions.items())
    
    logger.info("Embedding captions...")
    with torch.no_grad():
        for i in tqdm(range(0, len(items), batch_size)):
            batch_items = items[i : i + batch_size]
            filenames, texts = zip(*batch_items)
            
            if use_open_clip:
                # Truncate text if necessary, though CLIP tokenizer handles it
                tokens = tokenizer(texts).to(device)
                text_features = model.encode_text(tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                features_np = text_features.cpu().numpy()
            else:
                # Transformers logic (Mean Pooling)
                inputs = tokenizer(
                    list(texts), 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to(device)
                
                outputs = model(**inputs)
                
                # Perform mean pooling on last hidden state, paying attention to attention mask
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs.attention_mask
                
                # Masked mean
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                text_features = sum_embeddings / sum_mask
                
                # Normalize
                text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
                features_np = text_features.cpu().numpy()
            
            for fname, feat in zip(filenames, features_np):
                embeddings[fname] = feat.tolist()
                
    return embeddings, model

def compute_metrics_snapshot(vectors: np.ndarray, k_values: List[int]) -> Dict[str, Any]:
    """Calculate metrics for a specific set of vectors."""
    if vectors.shape[0] == 0:
        return {}
    
    N = vectors.shape[0]
    
    # Mean Pairwise Distance
    sample_size = N
    
    if N > sample_size:
        indices = np.random.choice(N, sample_size, replace=False)
        sample_vectors = vectors[indices]
    else:
        sample_vectors = vectors
        
    dists = pairwise_distances(sample_vectors, metric='euclidean')
    # Use upper triangle
    iu = np.triu_indices(sample_size, k=1)
    pairwise_dists = dists[iu]
    
    if pairwise_dists.size > 0:
        mean_pairwise_dist = float(np.mean(pairwise_dists))
        std_pairwise_dist = float(np.std(pairwise_dists))
    else:
        mean_pairwise_dist = 0.0
        std_pairwise_dist = 0.0
    
    # K-Covering Radius (Greedy K-Center / Farthest Point Sampling)
    max_k = max(k_values)
    eff_max_k = min(max_k, N)
    
    radii = {}
    vectors_th = torch.Tensor(vectors)
    
    if N > 0:
        sampled_pts, indices = torch_fpsample.sample(vectors_th, eff_max_k)
        dists_matrix = torch.cdist(sampled_pts, vectors_th)

        sorted_ks = sorted([k for k in k_values if k <= eff_max_k])
        current_min_dists = dists_matrix[0]
        last_k = 1

        for k in sorted_ks:
            if k == 1:
                radii[1] = float(current_min_dists.max())
                last_k = 1
                continue

            if k > last_k:
                new_dists_block = dists_matrix[last_k:k]
                min_new, _ = torch.min(new_dists_block, dim=0)
                current_min_dists = torch.min(current_min_dists, min_new)

            radii[k] = float(current_min_dists.max())
            last_k = k
            
    k_covering_results = {k: radii.get(k, 0.0) for k in k_values if k <= eff_max_k}
    
    return {
        "mean_pairwise_distance": mean_pairwise_dist,
        "std_pairwise_distance": std_pairwise_dist,
        "k_covering_radii": k_covering_results,
        "sample_size_pairwise": sample_size,
        "total_embeddings": N
    }

def calculate_caption_metrics_trajectory(
    embeddings_dict: Dict[str, List[float]], 
    image_paths_ordered: List[Path],
    k_values: List[int],
    data_path: Path,
) -> Dict[str, Any]:
    """Calculate metrics over time as the archive grows."""
    
    ordered_keys = [p.name for p in image_paths_ordered if p.name in embeddings_dict]
    if not ordered_keys:
        # Ensure we write a valid JSON (empty dict) to avoid stale data
        with open(data_path, "w") as f:
            json.dump({}, f)
        return {}

    all_vectors = np.array([embeddings_dict[k] for k in ordered_keys])
    total_images = len(ordered_keys)
    
    # Determine checkpoints
    # We want a reasonable resolution for the plot
    # step_size = max(1, total_images // 50)
    step_size = 20
    
    checkpoints = list(range(step_size, total_images, step_size))
    if not checkpoints or checkpoints[-1] != total_images:
        checkpoints.append(total_images)
        
    trajectory = {}
    
    logger.info(f"Computing metrics over {len(checkpoints)} checkpoints (Total {total_images} images)...")

    # if data_path.is_file():
    #     with open(data_path, "r") as f:
    #         trajectory = json.load(f)
    
    for n in tqdm(checkpoints):
        if n in trajectory:
            continue
        subset = all_vectors[:n]
        metrics = compute_metrics_snapshot(subset, k_values)
        trajectory[n] = metrics

    print(trajectory.keys())
    with open(data_path, "w") as f:
        json.dump(trajectory, f, indent=2)
        
    return trajectory


def plot_metrics(metrics_data: Dict[str, Any], output_dir: Path):
    """Plot metrics over time."""

    output_dir.mkdir(exist_ok=True, parents=True)
    trajectory = metrics_data
    if not trajectory:
        return
        
    xs = sorted([int(k) for k in trajectory.keys()])
    str_xs = [str(x) for x in xs]
    
    # 1. Mean Pairwise Distance
    means = [trajectory[x]["mean_pairwise_distance"] for x in str_xs]
    stds = [trajectory[x]["std_pairwise_distance"] for x in str_xs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(xs, means, marker='o', label="Mean Pairwise Dist")
    plt.fill_between(xs, 
                     [m - s for m, s in zip(means, stds)], 
                     [m + s for m, s in zip(means, stds)], 
                     alpha=0.2, label="Std Dev")
    plt.xlabel("Archive Size (Images)")
    plt.ylabel("Distance")
    plt.title("Diversity: Pairwise Distance over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "pairwise_distance_over_time.png")
    plt.close()
    
    # 2. K-Covering Radius
    if not str_xs:
        return

    # Determine which Ks are present in the final snapshot to ensure we plot relevant ones
    final_key = str_xs[-1]
    final_metrics = trajectory[final_key].get("k_covering_radii", {})
    ks_to_plot = sorted([int(k) for k in final_metrics.keys()])
    
    plt.figure(figsize=(10, 6))
    for k in ks_to_plot:
        # Extract series
        ys = []
        valid_xs = []
        for x, key in zip(xs, str_xs):
            radii = trajectory[key].get("k_covering_radii", {})
            if k in radii:
                ys.append(radii[k])
                valid_xs.append(x)
        
        if ys:
            plt.plot(valid_xs, ys, marker='.', label=f"k={k}")
            
    plt.xlabel("Archive Size (Images)")
    plt.ylabel("Covering Radius")
    plt.title("Coverage: K-Covering Radius over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "k_covering_radius_over_time.png")
    plt.close()

def render_grid_wrapper(image_paths: List[Path], captions_dict: Dict[str, str], output_path: Path, thumb_size: int):
    """Render a grid of images with short captions using rendering.py utility."""
    pil_images = []
    caption_list = []
    
    logger.info("Loading images for grid...")
    for p in tqdm(image_paths):
        try:
            img = Image.open(p).convert("RGB")
            pil_images.append(img)
            
            raw_cap = captions_dict.get(p.name, "")
            # Removed truncation to allow full caption rendering with wrapping
            caption_list.append(raw_cap)
        except Exception as e:
            logger.warning(f"Could not load {p} for grid: {e}")
            
    if not pil_images:
        logger.warning("No images loaded for grid.")
        return

    try:
        logger.info(f"Creating captioned grid with {len(pil_images)} images...")
        grid_img = create_captioned_grid(
            pil_images, 
            caption_list, 
            thumb_size=thumb_size, 
            margin=10, 
            font_size=10
        )
        grid_img.save(output_path)
        logger.info(f"Saved captioned grid to {output_path}")
    except Exception as e:
        logger.error(f"Failed to render grid: {e}")

def run_captioning_phase(cfg: CaptionEmbedConfig, backend: Optional[VLMBackend] = None):
    archive_path = Path(cfg.archive_path)
    if not archive_path.is_absolute():
        archive_path = Path(hydra.utils.to_absolute_path(str(archive_path)))
    
    if not archive_path.exists():
        logger.error(f"Archive not found: {archive_path}")
        return

    logger.info(f"Processing archive: {archive_path}")
    logger.info(f"VLM: {cfg.caption_model}")

    # 1. Load Images
    images = load_images(archive_path, cfg.max_images)
    if not images:
        logger.warning("No images found.")
        return

    # 2. Caption Images
    if backend is None:
        backend = create_vlm_backend(cfg.caption_model, max_model_len=10_000)
    
    captions_file = archive_path / f"captions_{cfg.caption_model}.json"
    
    captions = caption_images(
        backend, 
        images, 
        cfg.caption_batch_size, 
        cfg.caption_prompt,
        captions_file,
        resume=cfg.resume_captions
    )
    
    # Render Grid
    if cfg.render_grid:
        grid_path = archive_path / f"captioned_grid_{cfg.caption_model}.png"
        if not grid_path.exists():
            render_grid_wrapper(images, captions, grid_path, cfg.grid_thumb_size)
        else:
            logger.info(f"Grid {grid_path} already exists. Skipping render.")

def run_embedding_phase(cfg: CaptionEmbedConfig, embed_model: Any = None):
    archive_path = Path(cfg.archive_path)
    if not archive_path.is_absolute():
        archive_path = Path(hydra.utils.to_absolute_path(str(archive_path)))

    captions_file = archive_path / f"captions_{cfg.caption_model}.json"
    if not captions_file.exists():
        logger.warning(f"Captions file {captions_file} not found. Skipping embedding.")
        return
        
    with open(captions_file, "r") as f:
        captions = json.load(f)
        
    # 3. Embed Captions
    embeddings_file = archive_path / f"embeddings_{cfg.embedding_model.replace('/', '-')}_{cfg.embedding_pretrained}.json"
    
    # Only use captions that exist
    valid_captions = {k: v for k, v in captions.items() if v}

    # Optimization: if embeddings file exists and covers all captions, skip embedding
    embeddings = {}
    should_embed = True
    if embeddings_file.exists():
        try:
             with open(embeddings_file, "r") as f:
                 embeddings = json.load(f)
             if len(embeddings) >= len(valid_captions): # Approximate check
                 should_embed = False
                 logger.info("Embeddings file seems complete, skipping embedding step.")
        except Exception:
             pass

    if should_embed:
        if embed_model is None:
             embed_model = load_embedding_model(cfg.embedding_model, cfg.embedding_pretrained)
        
        embeddings = embed_captions_with_model(valid_captions, embed_model)
        
        # Save embeddings
        with open(embeddings_file, "w") as f:
            json.dump(embeddings, f)
            
    # 4. Calculate Metrics
    k_values = [int(k.strip()) for k in cfg.k_covering_ks.split(",") if k.strip()]
    metrics_file = archive_path / f"metrics_{cfg.caption_model}_{cfg.embedding_model.replace('/', '-')}.json"
    
    images = load_images(archive_path, cfg.max_images)
        
    metrics = calculate_caption_metrics_trajectory(embeddings, images, k_values, metrics_file)
            
    plot_metrics(metrics, archive_path / "plots")

def load_embedding_model(model_name: str, pretrained: str):
    """Load embedding model (OpenCLIP or HF) once."""
    logger.info(f"Loading embedding model {model_name} ({pretrained})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = None
    tokenizer = None
    use_open_clip = True

    # Detect if model is supported by OpenCLIP
    if model_name.startswith("gemini-embedding") or model_name.startswith("models/gemini-embedding"):
        logger.info(f"Detected Gemini Embedding model: {model_name}")
        # Initialize Gemini API
        import dotenv
        from google import genai
        
        dotenv.load_dotenv()
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
            
        # Initialize client
        client = genai.Client(
            api_key=api_key,
            http_options={'api_version': 'v1alpha'}
        )
        
        return {
            "model": client,
            "use_gemini": True,
            "model_name": model_name,
            "device": device # Not really used but keep interface consistent
        }
    elif model_name in open_clip.list_models():
        logger.info(f"Detected OpenCLIP model: {model_name}")
        pt = pretrained if pretrained else None
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pt, device=device
            )
            tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model: {e}")
            raise e
    else:
        # Try vLLM first
        use_vllm = False
        try:
            from vllm import LLM
            logger.info(f"Attempting to load '{model_name}' with vLLM...")
            # For embedding models, we might need task="embed" or similar depending on version
            # vLLM 0.13.0 might infer it?
            # 'enforce_eager=True' is often safe. 'trust_remote_code=True' for KaLM.
            model = LLM(model=model_name, trust_remote_code=True, enforce_eager=True)
            use_vllm = True
            logger.info(f"Loaded '{model_name}' with vLLM.")
        except Exception as vllm_err:
            logger.warning(f"vLLM load failed ({vllm_err}). Falling back to Transformers.")
            use_vllm = False
        
        if not use_vllm:
            logger.info(f"Assuming Hugging Face Transformers for '{model_name}'.")
            use_open_clip = False
            try:
                from transformers import AutoTokenizer, AutoModel
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception as e:
                 logger.error(f"Failed to load Transformers model: {e}")
                 raise e
             
    return {
        "model": model,
        "tokenizer": tokenizer,
        "use_open_clip": use_open_clip,
        "use_vllm": getattr(locals(), 'use_vllm', False),
        "device": device
    }

def embed_captions_with_model(
    captions: Dict[str, str],
    loaded_model_dict: Dict[str, Any]
) -> Dict[str, List[float]]:
    """Embed captions using pre-loaded model dict."""
    model = loaded_model_dict["model"]
    tokenizer = loaded_model_dict.get("tokenizer")
    use_open_clip = loaded_model_dict.get("use_open_clip", False)
    use_vllm = loaded_model_dict.get("use_vllm", False)
    use_gemini = loaded_model_dict.get("use_gemini", False)
    device = loaded_model_dict["device"]
    
    embeddings = {}
    batch_size = 32
    items = list(captions.items())
    
    logger.info("Embedding captions...")
    
    if use_gemini:
        # Gemini API processing
        model_name = loaded_model_dict.get("model_name", "models/gemini-embedding-001")
        if not model_name.startswith("models/"):
             model_name = f"models/{model_name}"
             
        client = model
        
        # Google GenAI API rate limits might apply, so process in smaller batches if needed
        # The API supports batch embedding.
        
        for i in tqdm(range(0, len(items), batch_size)):
            batch_items = items[i : i + batch_size]
            batch_texts = [text for _, text in batch_items]
            batch_filenames = [fname for fname, _ in batch_items]
            
            try:
                # Assuming google-genai client structure
                result = client.models.embed_content(
                    model=model_name,
                    contents=batch_texts,
                )
                
                # result.embeddings should be a list of embedding objects or lists
                # If using google-genai package:
                # result is EmbedContentResponse
                # result.embeddings is list of ContentEmbedding
                # ContentEmbedding.values is the vector
                
                if hasattr(result, 'embeddings'):
                    for j, emb_obj in enumerate(result.embeddings):
                        if hasattr(emb_obj, 'values'):
                            embeddings[batch_filenames[j]] = emb_obj.values
                        else:
                            # Fallback if structure differs
                            pass
                else:
                     logger.warning(f"Unexpected response format from Gemini Embedding API")

            except Exception as e:
                logger.error(f"Gemini embedding failed for batch starting at {i}: {e}")
                time.sleep(5) # Backoff
                
    elif use_vllm:
        # vLLM processing
        all_texts = [text for _, text in items]
        all_filenames = [fname for fname, _ in items]
        
        try:
            # vLLM encode returns list of EmbeddingRequestOutput?
            # or RequestOutput with embedding?
            outputs = model.encode(all_texts)
            # outputs is list of EmbeddingRequestOutput
            # each has .outputs which is EmbeddingOutput?
            
            for i, output in enumerate(outputs):
                # output.outputs.embedding is the vector?
                # Check vLLM API
                embedding = output.outputs.embedding
                embeddings[all_filenames[i]] = embedding
                
        except Exception as e:
            logger.error(f"vLLM encoding failed: {e}")
            raise e
            
    else:
        with torch.no_grad():
            for i in tqdm(range(0, len(items), batch_size)):
                batch_items = items[i : i + batch_size]
                filenames, texts = zip(*batch_items)
                
                if use_open_clip:
                    tokens = tokenizer(texts).to(device)
                    text_features = model.encode_text(tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    features_np = text_features.cpu().numpy()
                else:
                    inputs = tokenizer(
                        list(texts), 
                        padding=True, 
                        truncation=True, 
                        max_length=512, 
                        return_tensors="pt"
                    ).to(device)
                    outputs = model(**inputs)
                    last_hidden = outputs.last_hidden_state
                    attention_mask = inputs.attention_mask
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                    sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    text_features = sum_embeddings / sum_mask
                    text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
                    features_np = text_features.cpu().numpy()
                
                for fname, feat in zip(filenames, features_np):
                    embeddings[fname] = feat.tolist()
                
    return embeddings

@hydra.main(version_base=None, config_name="caption_embed_config")
def main(cfg: CaptionEmbedConfig):
    run_captioning_phase(cfg)
    
    # For main run, we load model ad-hoc
    # Or reuse the new functions
    embed_model = load_embedding_model(cfg.embedding_model, cfg.embedding_pretrained)
    run_embedding_phase(cfg, embed_model)

if __name__ == "__main__":
    main()
