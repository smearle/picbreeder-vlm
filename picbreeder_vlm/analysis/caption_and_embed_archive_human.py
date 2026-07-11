from dataclasses import dataclass
from pathlib import Path
import shutil
import hydra
from hydra.core.config_store import ConfigStore
from picbreeder_vlm.core.constants import HUMAN_BASELINE_DIR
from picbreeder_vlm.analysis.caption_and_embed_archive import (
    CaptionEmbedConfig, 
    run_captioning_phase, 
    run_embedding_phase, 
    load_embedding_model
)

@dataclass
class HumanCaptionEmbedConfig(CaptionEmbedConfig):
    archive_path: str = "fer/src/archive_res-128"

cs = ConfigStore.instance()
cs.store(name="human_caption_embed_config", node=HumanCaptionEmbedConfig)

@hydra.main(version_base=None, config_name="human_caption_embed_config")
def main(cfg: HumanCaptionEmbedConfig):
    run_captioning_phase(cfg)
    
    embed_model = load_embedding_model(cfg.embedding_model, cfg.embedding_pretrained)
    metrics_file = run_embedding_phase(cfg, embed_model)
    shutil.copy(metrics_file, HUMAN_BASELINE_DIR / f"metrics_res{cfg.grid_thumb_size}_{cfg.caption_model}.json")

if __name__ == "__main__":
    main()