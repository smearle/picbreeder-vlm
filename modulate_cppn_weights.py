import pickle
import sys
from pathlib import Path
import neat
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import multiprocessing
import os
import matplotlib.pyplot as plt

# Add repo root to path
sys.path.append(str(Path(__file__).resolve().parent))

from neat_components import (
    PicbreederGenome,
    InteractiveStagnation,
    apply_picbreeder_config_defaults
)
from picbreeder_reproduction import PicbreederReproduction
from rendering import render_genome_image

def process_connection_row(conn_key, genome, config, deltas, image_size):
    conn_gene = genome.connections[conn_key]
    original_weight = conn_gene.weight
    
    row_images = []
    
    for delta in deltas:
        # Modulate
        conn_gene.weight = original_weight + delta
        
        try:
            # render_genome_image returns (gray, color)
            grayscale_image, color_image = render_genome_image(genome, config, image_size[0], image_size[1])
            row_images.append((grayscale_image, color_image))
        except Exception:
            row_images.append(None)
            
    return conn_key, row_images

def load_config(config_path: Path) -> neat.Config:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = neat.Config(
        PicbreederGenome,
        PicbreederReproduction,
        neat.DefaultSpeciesSet,
        InteractiveStagnation,
        str(config_path)
    )
    apply_picbreeder_config_defaults(config)
    return config

def save_modulation_grid(
    results,
    genome,
    deltas,
    output_path: Path,
    mode: str,
    title: str,
):
    num_rows = len(results)
    num_cols = len(deltas)

    if num_rows == 0:
        print(f"No rows to render for {title} ({mode}).")
        return

    # Use a per-row height so full-reference grids can be very tall.
    fig_height = max(3.0, num_rows * 2.2)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, fig_height), squeeze=False)

    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    fig.suptitle(title, fontsize=14)
    fig.text(0.5, 0.02, 'Sweeping Weight Value', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Weight ID', va='center', rotation='vertical', fontsize=14)

    for i, (conn_key, row_images) in enumerate(results):
        conn_gene = genome.connections[conn_key]
        original_weight = conn_gene.weight

        ax_first = axes[i, 0]
        ax_first.text(
            -0.1,
            0.5,
            f"Conn {conn_key}\n$w={original_weight:.2f}$",
            transform=ax_first.transAxes,
            va='center',
            ha='right',
            fontsize=10,
        )

        for j, imgs in enumerate(row_images):
            ax = axes[i, j]
            if imgs is None:
                ax.axis('off')
                continue

            grayscale_image, color_image = imgs

            if mode == 'color':
                ax.imshow(color_image)
            else:
                ax.imshow(grayscale_image, cmap='gray')

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if i == num_rows - 1 and j in [0, 2, 4]:
                delta_val = int(deltas[j])
                ax.text(
                    0.5,
                    -0.15,
                    rf"$\delta W = {delta_val}$",
                    transform=ax.transAxes,
                    ha='center',
                    va='top',
                    fontsize=12,
                )

    output_base_name = output_path.stem
    output_dir = output_path.parent
    current_output_path = output_dir / f"{output_base_name}_{mode}.png"
    plt.savefig(current_output_path, bbox_inches='tight', dpi=150)
    print(f"Saved to {current_output_path}")
    plt.close(fig)

def main():
    genome_path = Path("sweep_logs/sweep/th1_ag20_model-gemini-2.5-flash-lite_tb-1_scheme-toggle_nopersonalities_fixed-sesh_s5/archive/genomes/img_000447.pkl")
    config_path = Path("picture2d/interactive_config_color")
    output_path = Path("weight_modulation_grid.png")
    
    if not genome_path.exists():
        print(f"Genome not found: {genome_path}")
        return

    # Load config
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return
    
    # Load genome
    try:
        with open(genome_path, "rb") as f:
            genome = pickle.load(f)
    except Exception as e:
        print(f"Failed to load genome: {e}")
        return
        
    print(f"Loaded genome: {genome.key}")
    
    # Parameters for modulation
    deltas = np.linspace(-1.0, 1.0, 5)
    image_size = (128, 128)
    
    # Get all connections (weights)
    connections = list(genome.connections.keys())
    connections.sort() # Ensure consistent order
    
    num_rows = len(connections)
    num_cols = len(deltas)
    
    print(f"Processing {num_rows} weights with {num_cols} modulations each...")

    # Prepare for multiprocessing
    num_cores = os.cpu_count() or 1
    num_workers = max(1, num_cores - 1)
    print(f"Using {num_workers} workers.")
    
    tasks = [
        (conn_key, genome, config, deltas, image_size)
        for conn_key in connections
    ]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use starmap to pass arguments unpacked
        all_results = pool.starmap(process_connection_row, tasks)
    
    # Compute impact of modulation (pixel distance)
    # We want weights where modulation causes biggest change from the original (center) to extremes.
    # deltas are linspace(-1, 1, 5) -> indices: 0 (-1), 1 (-0.5), 2 (0), 3 (0.5), 4 (1)
    scored_results = []
    for conn_key, row_images in all_results:
        # Check if valid
        if not row_images or len(row_images) != 5 or any(img is None for img in row_images):
            score = -1.0
        else:
            # row_images[i] is (grayscale, color)
            # Use grayscale image for distance
            
            # Center image (original weight)
            center_img = np.array(row_images[2][0], dtype=np.float32)
            
            # Extremes
            left_img = np.array(row_images[0][0], dtype=np.float32)
            right_img = np.array(row_images[4][0], dtype=np.float32)
            
            # L1 distance
            dist_left = np.mean(np.abs(center_img - left_img))
            dist_right = np.mean(np.abs(center_img - right_img))
            
            score = dist_left + dist_right
            
        scored_results.append((score, conn_key, row_images))
        
    # Sort descending by score
    scored_results.sort(key=lambda x: x[0], reverse=True)
    
    # Take top k
    k = 4
    top_k = scored_results[:k]

    # Extract results for rendering
    top_k_results = [(item[1], item[2]) for item in top_k]
    all_weight_results = [(item[1], item[2]) for item in scored_results]

    print(f"Selected top {len(top_k_results)} weights with highest modulation impact.")
    print(f"Rendering full reference grid with all {len(all_weight_results)} weights.")

    top_k_output_path = output_path.with_name(f"{output_path.stem}_topk{output_path.suffix}")
    all_weights_output_path = output_path.with_name(f"{output_path.stem}_all_weights{output_path.suffix}")

    for mode in ['color', 'grayscale']:
        print(f"\nSaving top-k {mode} grid...")
        save_modulation_grid(
            top_k_results,
            genome,
            deltas,
            top_k_output_path,
            mode,
            title=f"Top-{k} Weight Modulations by Impact",
        )

        print(f"Saving all-weights {mode} grid...")
        save_modulation_grid(
            all_weight_results,
            genome,
            deltas,
            all_weights_output_path,
            mode,
            title="All Weight Modulations (Reference)",
        )

if __name__ == "__main__":
    main()
