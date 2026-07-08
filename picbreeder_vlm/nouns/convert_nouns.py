import sys

inp, outp = sys.argv[1], sys.argv[2]

with open(inp, "r", encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as g:
    for line in f:
        # split on first comma, take the noun, strip whitespace
        _, noun = line.split(",", 1)
        g.write(noun.strip() + "\n")