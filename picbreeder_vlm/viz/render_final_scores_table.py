
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import asdict, replace, fields
from typing import List, Dict, Any, Tuple, Optional, Union
import sys
import os
import glob

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

from picbreeder_vlm.experiments.sweep_configs import _NAMED_SWEEPS, SweepConfig, PicbreederConfig, RandBaselineSweep
from picbreeder_vlm.experiments.sweep import _expand_sweep_configs, _build_run_config
from picbreeder_vlm.experiments.sweep_analysis_utils import load_human_baseline

def load_metric_from_trajectory(path: Path, key: str, limit: int) -> Tuple[Optional[float], int]:
    if not path.exists():
        return None, 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Handle dict format { "0": ..., "1": ... }
        if isinstance(data, dict):
            # Sort by index
            indices = sorted([int(k) for k in data.keys() if k.isdigit()])
            valid_indices = [i for i in indices if i <= limit]
            if not valid_indices:
                return None, 0
            last_idx = valid_indices[-1]
            record = data[str(last_idx)]
            return record.get(key), last_idx
        
        # Handle list format
        if isinstance(data, list):
            valid_records = [r for r in data if r.get("index") is not None and int(r["index"]) <= limit]
            if not valid_records:
                return None, 0
            last_record = valid_records[-1]
            return last_record.get(key), int(last_record["index"])
            
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return None, 0

def load_k_covering(path: Path, k: int, limit: int) -> Tuple[Optional[float], int]:
    if not path.exists():
        return None, 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        
        # Handle dict format { "index": { ... } }
        if isinstance(data, dict):
            indices = sorted([int(idx) for idx in data.keys() if idx.isdigit()])
            valid_indices = [i for i in indices if i <= limit]
            if not valid_indices:
                return None, 0
            last_idx = valid_indices[-1]
            record = data[str(last_idx)]
            radii = record.get("k_covering_radii")
            if isinstance(radii, dict):
                val = radii.get(str(k))
                if val is not None:
                     return float(val), last_idx
        
        # Handle list format [ { "index": ... }, ... ]
        elif isinstance(data, list):
            valid_records = [r for r in data if r.get("index") is not None and int(r["index"]) <= limit]
            if not valid_records:
                return None, 0
            last_record = valid_records[-1]
            last_idx = int(last_record["index"])
            radii = last_record.get("k_covering_radii")
            if isinstance(radii, dict):
                val = radii.get(str(k))
                if val is not None:
                     return float(val), last_idx
            
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return None, 0

def load_j1_index(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        val = data.get("j1_index")
        return float(val) if val is not None else None
    except Exception:
        return None

def get_value_at_limit(trajectory: Dict[int, float], limit: int) -> Optional[float]:
    if not trajectory:
        return None
    
    # Exact match
    if limit in trajectory:
        return trajectory[limit]
        
    # Find closest
    keys = sorted(trajectory.keys())
    # We prefer something close to limit. 
    # If limit is 2000, and we have 1951 and 2001. 
    # 2001 is closer. 
    
    # Let's find the key with minimum absolute difference
    closest_key = min(keys, key=lambda k: abs(k - limit))
    
    # If the difference is huge (e.g. > 100), maybe warn? But here we just return it.
    # For now, just return the closest.
    return trajectory[closest_key]

def get_experiment_metrics(run_cfg: PicbreederConfig, sweep_cfg: SweepConfig) -> Dict[str, Union[Optional[float], int]]:
    exp_dir = Path(run_cfg.experiment_dir)
    limit = run_cfg.num_agents
    
    # 1. Mean max noun similarity
    # Filename: noun_similarity_over_time_{nounlist}_{model}.json
    nounlist_name = Path(run_cfg.nounlist).stem
    
    text_image_model = sweep_cfg.text_image_embedding_model
    text_model_suffix = ""
    if text_image_model:
        sanitized = text_image_model.replace("/", "-")
        text_model_suffix = f"_{sanitized}"
        
    noun_path = exp_dir / f"noun_similarity_over_time_{nounlist_name}{text_model_suffix}.json"
    
    noun_val = None
    noun_idx = 0
    if noun_path.exists():
         noun_val, noun_idx = load_metric_from_trajectory(noun_path, "mean_max_similarity", limit)
    else:
         # Fallback search if exact match fails (e.g. if config didn't match exactly what was run)
         candidates = list(exp_dir.glob(f"noun_similarity_over_time_{nounlist_name}*.json"))
         if candidates:
             noun_val, noun_idx = load_metric_from_trajectory(candidates[0], "mean_max_similarity", limit)

    # 2. k=100 covering radius in image space
    # Filename: embedding_mean_pairwise_distance_over_time_{model}.json
    image_model = sweep_cfg.image_embedding_model
    image_model_suffix = ""
    if image_model:
        sanitized = image_model.replace("/", "-")
        image_model_suffix = f"_{sanitized}"
        
    novelty_path = exp_dir / f"embedding_mean_pairwise_distance_over_time{image_model_suffix}.json"
    
    image_k_val = None
    image_idx = 0
    
    if novelty_path.exists():
        image_k_val, image_idx = load_k_covering(novelty_path, 100, limit)
    else:
        # Fallback
        candidates = list(exp_dir.glob("embedding_mean_pairwise_distance_over_time*.json"))
        # Prioritize SigLIP if fallback
        siglip = [c for c in candidates if "SigLIP" in c.name]
        if siglip:
             image_k_val, image_idx = load_k_covering(siglip[0], 100, limit)
        elif candidates:
             image_k_val, image_idx = load_k_covering(candidates[0], 100, limit)
        
    # 3. k=100 covering radius in caption-embedding space
    # Filename: archive/metrics_{caption_model}_{caption_embed}.json
    caption_model = sweep_cfg.caption_model
    caption_embed = sweep_cfg.caption_embedding_model
    
    caption_suffix = ""
    if caption_model and caption_embed:
        embed_sanitized = caption_embed.replace("/", "-")
        caption_suffix = f"_{caption_model}_{embed_sanitized}"
    
    caption_path = exp_dir / "archive" / f"metrics{caption_suffix}.json"
    
    caption_k_val = None
    caption_idx = 0
    
    if caption_path.exists():
        caption_k_val, caption_idx = load_k_covering(caption_path, 100, limit)
    else:
        # Fallback
        candidates = list((exp_dir / "archive").glob("metrics_*.json"))
        if candidates:
             caption_k_val, caption_idx = load_k_covering(candidates[0], 100, limit)

    # 4. J1 index
    # archive/phylogeny_metrics.json
    j1_val = load_j1_index(exp_dir / "archive" / "phylogeny_metrics.json")
    
    # Determine max agents found
    max_agents = max(noun_idx, image_idx, caption_idx)
    
    # Warn if incomplete
    if max_agents > 0 and max_agents < 2000:
        print(f"  Warning: Incomplete run (agents={max_agents}) in {exp_dir.name}")
    
    return {
        "noun_sim": noun_val,
        "image_k100": image_k_val,
        "caption_k100": caption_k_val,
        "j1": j1_val,
        "num_agents": max_agents if max_agents > 0 else None
    }

def format_cell(mean, sem, is_int=False):
    if mean is None or np.isnan(mean):
        return "-"
    if is_int:
        return f"{int(mean)}" # No SEM for agents usually, or format differently?
        # User requested "mean number of agents", so it might have variance if runs crashed differently
        # But usually we want to see if it's 2000. 
        # Let's keep SEM if variance exists, but format mean as int if it's close to int
    return f"{mean:.3f} $\\pm$ {sem:.3f}"

def main():
    sweeps_to_process = {
        r"Random Action Prob. ($\epsilon$)": "full_rand_select_prob",
        r"Context Length ($CL$)": "chat_history_turns",
        r"Num. Agents ($NA$)": "traits"
    }
    
    results = []
    
    cwd = Path(os.getcwd())

    # Helper to map sweep name to config key
    sweep_name_to_key = {
        "full_rand_select_prob": "rand_select_prob",
        "chat_history_turns": "chat_history_turns",
        "traits": "n_personality_traits"
    }

    # 1. Process Named Sweeps
    for label, sweep_name in sweeps_to_process.items():
        print(f"Processing {label}...")
        sweep_cls = _NAMED_SWEEPS[sweep_name]
        
        cfg = sweep_cls()
        cfg.sweep_name = sweep_name 
        
        configs = _expand_sweep_configs(cfg)
        
        group_key = sweep_name_to_key.get(sweep_name, "unknown")
            
        grouped_data = {} 
        
        for base_run_cfg in configs:
            run_cfg = _build_run_config(base_run_cfg, cwd)
            
            if not Path(run_cfg.experiment_dir).exists():
                print(f"  Warning: Directory not found: {run_cfg.experiment_dir}")
                continue
            
            val = getattr(run_cfg, group_key)
            if isinstance(val, list): val = tuple(val)
            
            if group_key == "chat_history_turns" and val == -1:
                val = "20 (Full)" 
            
            metrics = get_experiment_metrics(run_cfg, base_run_cfg)
            
            if val not in grouped_data:
                grouped_data[val] = []
            grouped_data[val].append(metrics)
            
        try:
            sorted_keys = sorted(grouped_data.keys(), key=lambda x: (isinstance(x, str), x))
        except:
            sorted_keys = sorted(grouped_data.keys(), key=str)

        for val in sorted_keys:
            metrics_list = grouped_data[val]
            row = {"Sweep": label, "Condition": val}
            
            for m_key in ["noun_sim", "image_k100", "caption_k100", "j1", "num_agents"]:
                values = [m[m_key] for m in metrics_list if m[m_key] is not None]
                if values:
                    mean = np.mean(values)
                    sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
                    row[f"{m_key}_mean"] = mean
                    row[f"{m_key}_sem"] = sem
                else:
                    row[f"{m_key}_mean"] = np.nan
                    row[f"{m_key}_sem"] = np.nan
            
            results.append(row)

    # 2. Process Random Baseline
    print("Processing Random Baseline...")
    rand_cfg = RandBaselineSweep()
    rand_cfg.sweep_name = "rand_baseline"
    rand_cfg.num_agents = 2000 # Try 2000 first
    rand_configs = _expand_sweep_configs(rand_cfg)
    
    rand_metrics = []
    rand_found = False
    
    for base_run_cfg in rand_configs:
        run_cfg = _build_run_config(base_run_cfg, cwd)
        if Path(run_cfg.experiment_dir).exists():
            print(f"  Found Random Baseline (2000): {run_cfg.experiment_dir}")
            rand_found = True
            metrics = get_experiment_metrics(run_cfg, base_run_cfg)
            rand_metrics.append(metrics)
    
    if not rand_found:
        print("  Random Baseline (2000 agents) not found. Trying default config...")
        rand_cfg = RandBaselineSweep() # Reset to default (9377)
        rand_configs = _expand_sweep_configs(rand_cfg)
        for base_run_cfg in rand_configs:
            run_cfg = _build_run_config(base_run_cfg, cwd)
            if Path(run_cfg.experiment_dir).exists():
                print(f"  Found Random Baseline (Default): {run_cfg.experiment_dir}")
                rand_found = True
                run_cfg.num_agents = 2000 # Restrict analysis to 2000
                metrics = get_experiment_metrics(run_cfg, base_run_cfg)
                rand_metrics.append(metrics)
    
    if rand_metrics:
        row = {"Sweep": "Baselines", "Condition": "Random"}
        for m_key in ["noun_sim", "image_k100", "caption_k100", "j1", "num_agents"]:
            values = [m[m_key] for m in rand_metrics if m[m_key] is not None]
            if values:
                mean = np.mean(values)
                sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
                row[f"{m_key}_mean"] = mean
                row[f"{m_key}_sem"] = sem
            else:
                row[f"{m_key}_mean"] = np.nan
                row[f"{m_key}_sem"] = np.nan
        results.append(row)
    else:
        print("  Random Baseline not found.")

    # 3. Process Human Baseline
    print("Processing Human Baseline (Target: 2000 agents)...")
    
    hb_row = {"Sweep": "Baselines", "Condition": "Human"}
    hb_limit = 2000
    
    hb_noun = load_human_baseline("noun", 128, "ViT-SO400M-14-SigLIP2", nounlist="things_deduped")
    if hb_noun:
        hb_row["noun_sim_mean"] = get_value_at_limit(hb_noun, hb_limit)
        hb_row["noun_sim_sem"] = 0.0 
    
    hb_img = load_human_baseline("visual_k_covering", 128, "SigLIP2-B-alignet", k=100)
    if hb_img:
        hb_row["image_k100_mean"] = get_value_at_limit(hb_img, hb_limit)
        hb_row["image_k100_sem"] = 0.0

    hb_cap = load_human_baseline("caption_k_covering", 128, "gemini-2.5-pro", k=100)
    if hb_cap:
        hb_row["caption_k100_mean"] = get_value_at_limit(hb_cap, hb_limit)
        hb_row["caption_k100_sem"] = 0.0
    
    # Human baseline num_agents is conceptually 2000 (since we clipped it there)
    hb_row["num_agents_mean"] = 2000

    human_metrics_path = Path("human_lineages/lineages/lineage_phylogeny_metrics.json")
    if human_metrics_path.exists():
        try:
            hm = json.loads(human_metrics_path.read_text(encoding="utf-8"))
            m = hm.get("2000")
            # If 2000 not found, try to fallback but prefer 2000
            if m:
                hb_row["j1_mean"] = float(m.get("j1_index", 0.0))
                hb_row["j1_sem"] = 0.0
            else:
                 # Try finding closest key?
                 # Keys are strings.
                 keys = [int(k) for k in hm.keys() if k.isdigit()]
                 if keys:
                     closest = min(keys, key=lambda k: abs(k - hb_limit))
                     m = hm[str(closest)]
                     hb_row["j1_mean"] = float(m.get("j1_index", 0.0))
                     hb_row["j1_sem"] = 0.0
        except Exception:
            pass
            
    results.append(hb_row)

    df = pd.DataFrame(results)
    
    # Define metrics for console (including Agents)
    metrics_map_console = {
        "noun_sim": "Semantic Recall",
        "image_k100": "Visual Coverage",
        "caption_k100": "Semantic Coverage",
        "j1": "Tree Balance (J1)",
        "num_agents": "Agents"
    }
    
    # Define metrics for LaTeX (excluding Agents)
    metrics_map_latex = {
        "noun_sim": "Semantic Recall",
        "image_k100": "Visual Coverage",
        "caption_k100": "Semantic Coverage",
        "j1": r"Tree Balance ($J^1$)"
    }
    
    # Calculate Maxima for Highlighting
    # 1. Global Max per metric (including Baselines)
    global_max = {}
    for m_key in metrics_map_latex.keys():
        col = f"{m_key}_mean"
        if col in df.columns:
            global_max[m_key] = df[col].max()
            
    # 2. Sweep Max per metric (excluding Baselines)
    sweep_max = {}
    sweeps = df["Sweep"].unique()
    for sweep in sweeps:
        if sweep == "Baselines":
            continue
        sweep_df = df[df["Sweep"] == sweep]
        sweep_max[sweep] = {}
        for m_key in metrics_map_latex.keys():
            col = f"{m_key}_mean"
            if col in sweep_df.columns:
                sweep_max[sweep][m_key] = sweep_df[col].max()

    def format_cell_latex(row, m_key):
        mean = row.get(f"{m_key}_mean")
        sem = row.get(f"{m_key}_sem")
        sweep = row.get("Sweep")
        
        if mean is None or np.isnan(mean):
            return "-"
            
        # Basic format
        text = f"{mean:.3f} $\\pm$ {sem:.3f}"
        
        # Apply Bold (Sweep Max, excluding Baselines)
        if sweep != "Baselines" and sweep in sweep_max and m_key in sweep_max[sweep]:
            if np.isclose(mean, sweep_max[sweep][m_key]):
                text = f"\\textbf{{{text}}}"
                
        # Apply Green Highlight (Global Max)
        if m_key in global_max and np.isclose(mean, global_max[m_key]):
             text = f"\\cellcolor[rgb]{{0.75, 1, 0.75}}{{{text}}}"
             
        return text

    # Build Console DataFrame
    df_console = df.copy()
    final_cols_console = ["Sweep", "Condition"]
    for m_key, m_name in metrics_map_console.items():
        col_name = m_name
        is_int = (m_key == "num_agents")
        df_console[col_name] = df.apply(lambda r: format_cell(r.get(f"{m_key}_mean"), r.get(f"{m_key}_sem"), is_int=is_int), axis=1)
        final_cols_console.append(col_name)
    
    df_console = df_console[final_cols_console]
    
    # Build LaTeX DataFrame
    df_latex = df.copy()
    final_cols_latex = ["Sweep", "Condition"]
    for m_key, m_name in metrics_map_latex.items():
        col_name = m_name
        # Use new formatting function
        df_latex[col_name] = df.apply(lambda r: format_cell_latex(r, m_key), axis=1)
        final_cols_latex.append(col_name)
    
    df_latex = df_latex[final_cols_latex]
    
    print("\nFinal Table (Console):")
    print(df_console)
    
    # Custom LaTeX generation with multirow and lines
    latex_lines = []
    # Column alignment: l l c c c c
    # Assuming 4 metrics: l l c c c c
    n_metrics = len(metrics_map_latex)
    latex_lines.append(r"\begin{tabular}{ll" + "c"*n_metrics + "}")
    latex_lines.append(r"\toprule")
    
    # Header row
    metric_cols = [c for c in df_latex.columns if c not in ["Sweep", "Condition"]]
    header = ["Sweep", "Condition"] + metric_cols
    latex_lines.append(" & ".join(header) + r" \\")
    latex_lines.append(r"\midrule")
    
    # Iterate by group (preserve order)
    groups = df_latex.groupby("Sweep", sort=False)
    
    first_group = True
    for sweep_label, group in groups:
        if not first_group:
            latex_lines.append(r"\midrule")
        first_group = False
        
        n_rows = len(group)
        for i, (idx, row) in enumerate(group.iterrows()):
            condition = str(row["Condition"])
            vals = [str(row[c]) for c in metric_cols]
            
            # Check for default row
            is_default = False
            if "Random Action Prob" in sweep_label and condition == "0.0":
                is_default = True
            elif "Context Length" in sweep_label and condition == "1":
                is_default = True
            elif "Num. Agents" in sweep_label and condition == "0":
                is_default = True

            if is_default:
                gray_cmd = r"\cellcolor[gray]{0.9}"
                # Apply to Condition
                condition = f"{gray_cmd}{condition}"
                # Apply to Values if not already highlighted
                vals = [v if v.strip().startswith(r"\cellcolor") else f"{gray_cmd}{v}" for v in vals]
            
            if i == 0:
                s_label_cell = f"\\multirow{{{n_rows}}}{{*}}{{{sweep_label}}}"
            else:
                s_label_cell = ""
                
            line_cells = [s_label_cell, condition] + vals
            latex_lines.append(" & ".join(line_cells) + r" \\")
            
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    
    latex = "\n".join(latex_lines)

    print("\nLaTeX Code:")
    print(latex)
    
    output_file = "final_scores_table.tex"
    with open(output_file, "w") as f:
        f.write(latex)
    print(f"\nSaved to {output_file}")

if __name__ == "__main__":
    main()
