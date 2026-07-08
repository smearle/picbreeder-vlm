from collections import defaultdict, deque
from nltk.corpus import wordnet as wn
import nltk

nltk.download("wordnet")

from pathlib import Path
import re


def wnid_to_synset(wnid: str):
    pos = wnid[0]
    offset = int(wnid[1:])
    return wn.synset_from_pos_and_offset(pos, offset)

def synset_to_wnid(s) -> str:
    return f"{s.pos()}{s.offset():08d}"

def build_induced_graph(wnids):
    wnid_set = set(wnids)
    children = defaultdict(set)  # parent -> children
    parents  = defaultdict(set)  # child  -> parents

    synsets = {}
    for w in wnids:
        try:
            synsets[w] = wnid_to_synset(w)
        except Exception:
            # WordNet mismatch; skip
            pass

    for w, s in synsets.items():
        for hy in s.hyponyms():
            cw = synset_to_wnid(hy)
            if cw in wnid_set and cw in synsets:
                children[w].add(cw)
                parents[cw].add(w)

    return synsets, parents, children

def topo_order(nodes, adj_children, adj_parents):
    # Kahn over the induced graph
    indeg = {n: len(adj_parents[n]) for n in nodes}
    q = deque([n for n in nodes if indeg[n] == 0])
    out = []
    while q:
        n = q.popleft()
        out.append(n)
        for c in adj_children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)

    # WordNet should be a DAG; if not all nodes appear, fall back to partial order.
    if len(out) != len(nodes):
        # keep whatever we got, then append remaining deterministically
        remaining = [n for n in nodes if n not in set(out)]
        out.extend(sorted(remaining))
    return out

def compute_max_dist_to_ancestors(nodes, parents, children):
    # max distance to any ancestor (sources have 0)
    order = topo_order(nodes, children, parents)
    max_up = {n: 0 for n in nodes}
    for n in order:
        if parents[n]:
            max_up[n] = max(max_up[p] + 1 for p in parents[n])
    return max_up

def compute_max_dist_to_descendants(nodes, parents, children):
    # max distance to any descendant (sinks have 0)
    order = topo_order(nodes, children, parents)
    order.reverse()
    max_down = {n: 0 for n in nodes}
    for n in order:
        if children[n]:
            max_down[n] = max(max_down[c] + 1 for c in children[n])
    return max_down

def induced_middle_nodes(
    wnids,
    require_internal=True,
    exact_middle=True,
):
    """
    require_internal=True: must have at least one ancestor and one descendant in induced graph
    exact_middle=True: additionally require node to be exactly in the middle of its lineage
                       using max distances: allow |a-d|<=1
    """
    synsets, parents, children = build_induced_graph(wnids)
    nodes = list(synsets.keys())

    max_up = compute_max_dist_to_ancestors(nodes, parents, children)
    max_dn = compute_max_dist_to_descendants(nodes, parents, children)

    out = []
    for n in nodes:
        has_anc = max_up[n] > 0
        has_des = max_dn[n] > 0
        if require_internal and not (has_anc and has_des):
            continue

        if exact_middle:
            if abs(max_up[n] - max_dn[n]) > 1:
                continue

        out.append(n)

    middle_wnids = out

    # Now convert back to words
    middle_words = [synsets[w].lemma_names()[0] for w in middle_wnids]

    return middle_wnids, middle_words

wnids_path = "noun_lists/imagenet21k_wordnet_ids.txt"
with open(wnids_path) as f:
    wnids = [line.strip() for line in f]

all_words = [wnid_to_synset(w).lemma_names() for w in wnids]
# newlines separate wnids, commas separate words within each wnid
all_words_str = "\n".join([", ".join(words) for words in all_words])
print(f"{len(wnids)} total wnids in imagenet21k ({len(all_words)} unique words)")

# all_words_flat = [item for sublist in all_words for item in sublist]
with open("imagenet21k_all_words.txt", "w") as f:
    f.write(all_words_str + "\n")

middle_wnids, middle_words = induced_middle_nodes(wnids)
print(f"{len(middle_wnids)} middle wnids (within induced 21k hierarchy)")

with open("imagenet21k_middle_wnids.txt", "w") as f:
    f.write("\n".join(middle_wnids) + "\n")

with open("imagenet21k_middle_words.txt", "w") as f:
    f.write("\n".join(middle_words) + "\n")