import os

def dedupe_file(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist.")
        return

    with open(input_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    print(f"Original line count: {len(lines)}")

    # Dedupe while preserving order (if that matters, though the list looks sorted)
    # If the list is sorted, duplicates are adjacent.
    # If we want to strictly keep one instance.
    
    seen = set()
    deduped_lines = []
    for line in lines:
        if line and line not in seen: # also skip empty lines
            seen.add(line)
            deduped_lines.append(line)
    
    # Sort just in case, or keep original order? 
    # The original file seems sorted. Let's sort it to be clean.
    deduped_lines.sort()

    print(f"Deduped line count: {len(deduped_lines)}")

    with open(output_path, 'w') as f:
        for line in deduped_lines:
            f.write(line + '\n')
    
    print(f"Written deduped list to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "noun_lists", "things.txt")
    output_file = os.path.join(base_dir, "noun_lists", "things_deduped.txt")
    
    dedupe_file(input_file, output_file)
