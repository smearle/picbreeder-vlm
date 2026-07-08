import json
import random
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate a LaTeX table of random personality traits.")
    parser.add_argument("--input", default="data/personality_traits.json", help="Input JSON file")
    parser.add_argument("--output", default="personality_table.tex", help="Output TeX file")
    parser.add_argument("--n", type=int, default=40, help="Number of traits to sample (default: 40, approx. 1 page)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found.")
        return

    with open(input_path, 'r') as f:
        data = json.load(f)
    
    traits = data.get("traits", [])
    
    if not traits:
        print("Error: No traits found in the input file.")
        return
        
    if len(traits) < args.n:
        print(f"Warning: requested {args.n} traits but only found {len(traits)}. Using all available.")
        sampled_traits = traits
    else:
        sampled_traits = random.sample(traits, args.n)
    
    # Generate LaTeX
    latex_content = generate_latex(sampled_traits)
    
    with open(args.output, 'w') as f:
        f.write(latex_content)
    
    print(f"Generated {args.output} with {len(sampled_traits)} traits.")

def generate_latex(traits):
    # Header
    latex = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}

% Set margins to fit more on the page if needed
\geometry{a4paper, margin=1in}

\begin{document}

\section*{Sampled Personality Traits}

\begin{longtable}{rp{12cm}}
\toprule
\textbf{\#} & \textbf{Trait Description} \\
\midrule
\endhead
"""
    
    for i, trait in enumerate(traits, 1):
        # Escape special LaTeX characters
        safe_trait = (trait.replace('\\', r'\\')
                           .replace('&', r'\&')
                           .replace('%', r'\%')
                           .replace('$', r'\$')
                           .replace('#', r'\#')
                           .replace('_', r'\_')
                           .replace('{', r'\{')
                           .replace('}', r'\}')
                           .replace('^', r'\textasciicircum{}')
                           .replace('~', r'\textasciitilde{}'))
        latex += f"{i} & {safe_trait} \\\\\n"
        
    latex += r"""\bottomrule
\end{longtable}

\end{document}
"""
    return latex

if __name__ == "__main__":
    main()
