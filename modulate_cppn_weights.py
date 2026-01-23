import pickle
import sys
from pathlib import Path
import neat
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import multiprocessing
import os

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
    deltas = np.linspace(-3.0, 3.0, 7)
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
        results = pool.starmap(process_connection_row, tasks)
    
    # Create grid image
    # Add space for labels
    label_width = 150
    header_height = 40
    
    grid_width = label_width + num_cols * image_size[0]
    grid_height = header_height + num_rows * image_size[1]
    
    grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
    grayscale_grid_image = Image.new("L", (grid_width, grid_height), 255)
    draw = ImageDraw.Draw(grid_image)
    draw_gray = ImageDraw.Draw(grayscale_grid_image)
    
    # Draw header
    for j, delta in enumerate(deltas):
        x = label_width + j * image_size[0]
        text = f"{delta:+.1f}"
        # Use default font
        bbox = draw.textbbox((0, 0), text)
        text_width = bbox[2] - bbox[0]
        draw.text((x + (image_size[0] - text_width) // 2, 10), text, fill=(0, 0, 0))
        draw_gray.text((x + (image_size[0] - text_width) // 2, 10), text, fill=0)

    # Process results and draw
    for i, (conn_key, row_images) in enumerate(results):
        conn_gene = genome.connections[conn_key]
        original_weight = conn_gene.weight
        
        # Draw row label
        y = header_height + i * image_size[1]
        label = f"Idx: {i}\nConn: {conn_key}\nW: {original_weight:.2f}"
        draw.text((10, y + image_size[1] // 2 - 20), label, fill=(0, 0, 0))
        draw_gray.text((10, y + image_size[1] // 2 - 20), label, fill=0)
        
        for j, imgs in enumerate(row_images):
            if imgs is None:
                continue
                
            grayscale_image, color_image = imgs
            x = label_width + j * image_size[0]
            
            # Paste
            grid_image.paste(color_image, (x, y))
            grayscale_grid_image.paste(grayscale_image, (x, y))

    print("\nSaving grid...")

    grid_image.save(output_path)
    grayscale_output_path = output_path.with_name(output_path.stem + "_grayscale" + output_path.suffix)
    grayscale_grid_image.save(grayscale_output_path)
    print(f"Saved to {output_path} and {grayscale_output_path}")

if __name__ == "__main__":
    main()
