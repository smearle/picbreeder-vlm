import torch
import numpy as np
import open_clip
import os
import math
from typing import List, Tuple, Sequence
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import logging

try:
    import tensorflow as tf
except ImportError:
    tf = None

logger = logging.getLogger(__name__)

class TFSigLIPWrapper(torch.nn.Module):
    def __init__(self, tf_model_path):
        super().__init__()
        if tf is None:
            raise ImportError("TensorFlow is required to use SigLIP2-B-alignet")
        
        logger.info(f"Loading TF model from {tf_model_path}")
        self.tf_model = tf.saved_model.load(tf_model_path)
        self.serving_fn = self.tf_model.signatures['serving_default']
        
        self.output_key = None
        self.target_dim = 1024 
        
    def encode_image(self, image):
        # image: torch.Tensor (B, C, H, W) -> (B, H, W, C)
        img_np = image.detach().cpu().numpy()
        img_np = np.transpose(img_np, (0, 2, 3, 1)) # B C H W -> B H W C
        
        out_dict = self.serving_fn(images=tf.constant(img_np))
        
        if self.output_key is None:
            # Look for 1024 dim output (triplet_logits)
            for k, v in out_dict.items():
                if len(v.shape) == 2 and v.shape[1] == self.target_dim:
                    self.output_key = k
                    break
            if self.output_key is None:
                # Fallback to known key from inspection
                if 'Identity_14' in out_dict:
                    self.output_key = 'Identity_14'
                else:
                    # Look for 'triplet_logits' if named correctly
                    if 'triplet_logits' in out_dict:
                         self.output_key = 'triplet_logits'
                    else:
                        raise RuntimeError(f"Could not find embedding output with dimension {self.target_dim}. Available keys: {list(out_dict.keys())}")

        emb = out_dict[self.output_key].numpy()
        return torch.from_numpy(emb)

    def encode_text(self, text):
        raise NotImplementedError("SigLIP2-B-alignet is a vision-only model and does not support text encoding.")
    
    def to(self, device):
        return self
    
    def eval(self):
        pass

def load_model_by_name(model_name, pretrained, device):
    """
    Load model, preprocess, and tokenizer by name.
    """
    if model_name == "SigLIP2-B-alignet":
        # Use a standard preprocess from a similar model (SigLIP-B)
        # We use ViT-B-16-SigLIP2 from open_clip to get the transform
        ref_model_name = "ViT-B-16-SigLIP2"
        logger.info(f"Loading preprocessing from {ref_model_name}...")
        try:
             _, _, preprocess = open_clip.create_model_and_transforms(
                ref_model_name,
                pretrained="webli" # default for SigLIP2 usually
            )
        except:
             # Fallback if webli not found or connection issue, use simple defaults
             # SigLIP usually uses 224 or 256, scale -1,1 or similar. 
             # Let's try to use a safe default if specific one fails, or just 'laion2b_s32b_b79k' from ViT-H-14 as fallback
             _, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")

        # Load TF model
        if Path(model_name).exists():
            tf_path = model_name
        else:
            tf_path = model_name
            
        model = TFSigLIPWrapper(tf_path)
        tokenizer = None # No tokenizer
        
    else:
        # Standard OpenCLIP
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
        except Exception as e:
            # Check for SafetensorError or specific errors
            err_str = str(e)
            if "SafetensorError" in str(type(e).__name__) or "SafetensorError" in err_str or "EOF" in err_str or "invalid JSON" in err_str:
                logger.warning(f"Failed to load model {model_name} with pretrained={pretrained} due to corruption: {e}")
                logger.warning("Attempting to identify and delete corrupted cache file...")
                
                try:
                    # Attempt to resolve the checkpoint path
                    pretrained_cfg = open_clip.get_pretrained_cfg(model_name, pretrained)
                    if pretrained_cfg:
                        checkpoint_path = open_clip.download_pretrained(pretrained_cfg)
                        
                        if checkpoint_path and os.path.exists(checkpoint_path):
                            logger.warning(f"Deleting cached file: {checkpoint_path}")
                            os.remove(checkpoint_path)
                        else:
                             logger.warning("Could not locate cached file to delete.")
                    else:
                        logger.warning("Could not get pretrained config.")
                except Exception as cleanup_e:
                    logger.error(f"Error during cache cleanup: {cleanup_e}")

                # Retry
                logger.info("Retrying model loading...")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name,
                    pretrained=pretrained,
                )
            else:
                raise e

        model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)

    return model, preprocess, tokenizer

def prepare_model(cfg, device):
    """
    Load model, preprocess, and tokenizer from config.
    """
    return load_model_by_name(cfg.embedding_model, cfg.pretrained, device)

def batch(iterable, n=32):
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i : i + n]

def embed_images(
    model: torch.nn.Module,
    preprocess,
    image_paths: Sequence[Path],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[List[str], np.ndarray]:
    """Return filenames and L2-normalized CLIP embeddings for a list of image paths."""
    model.eval()
    embeddings = []
    filenames = []
    total = math.ceil(len(image_paths) / batch_size)
    
    with torch.no_grad():
        for chunk in tqdm(
            batch(image_paths, batch_size),
            total=total,
            desc="Embedding images",
        ):
            tensors = []
            chunk_filenames = []
            for p in chunk:
                img = Image.open(p).convert("RGB")
                t = preprocess(img)
                tensors.append(t)
                chunk_filenames.append(p.name)

            x = torch.stack(tensors, dim=0).to(device)
            emb = model.encode_image(x)
            emb = emb.cpu().numpy()
            
            # L2 normalize
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            emb = emb / norm
            
            embeddings.append(emb)
            filenames.extend(chunk_filenames)
            
    embeddings = np.vstack(embeddings)
        
    return filenames, embeddings
