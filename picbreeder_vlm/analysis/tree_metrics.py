import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

def compute_tree_metrics(roots: List[str], children: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Compute tree balance metrics (Sackin, Colless, J1) and other stats for a forest of trees.
    
    Args:
        roots: List of root node IDs.
        children: Adjacency list mapping node ID to list of child node IDs.
        
    Returns:
        Dictionary containing:
            - sackin_index: Sum of leaf depths (lower is more balanced).
            - colless_index: Sum of |L-R| for binary nodes (None if tree is not binary).
            - j1_index: Robust Universal Tree Balance Index (higher is more balanced).
            - num_trees: Number of roots.
            - num_leaves: Number of leaves.
            - max_depth: Maximum depth in the forest.
            - is_binary: Boolean indicating if the forest is strictly binary (<=2 children per node).
    """
    # Compute Sackin (sum of leaf depths) and basic traversals
    leaf_depths = []
    node_depth = {}
    is_binary = True
    
    stack = [(root, 0) for root in roots]
    visited = set()
    
    while stack:
        node, depth = stack.pop()
        visited.add(node)
        node_depth[node] = depth
        
        curr_children = children.get(node, [])
        if len(curr_children) > 2:
            is_binary = False
        
        if not curr_children:
            leaf_depths.append(depth)
        
        for child in curr_children:
            stack.append((child, depth + 1))
            
    sackin_index = float(sum(leaf_depths))
    
    # Compute metrics requiring post-order info (subtree sizes)
    # Sort nodes by depth descending to process children first
    sorted_nodes = sorted(node_depth.keys(), key=lambda k: node_depth[k], reverse=True)
    subtree_leaves = defaultdict(int)
    
    colless_sum = 0
    
    # J1 Index Calculation variables
    # J1 accumulates split entropy weighted by subtree size.
    # Formula: sum( Star[i] * K[i] / log(eff_children) ) / sum( Star[i] )
    j1_numerator = 0.0
    j1_denominator = 0.0

    for node in sorted_nodes:
        curr_children = children.get(node, [])
        if not curr_children:
            subtree_leaves[node] = 1
        else:
            s_leaves = 0
            child_leaf_counts = []
            for child in curr_children:
                c_leaves = subtree_leaves[child]
                s_leaves += c_leaves
                child_leaf_counts.append(c_leaves)
            
            subtree_leaves[node] = s_leaves
            
            # --- J1 Index Calculation ---
            # Star[i] corresponds to s_leaves (if population of internal node is 0)
            star_u = float(s_leaves)
            j1_denominator += star_u
            
            eff_children = len(curr_children)
            # Only internal nodes with > 1 children contribute to entropy balance
            if eff_children > 1:
                k_entropy = 0.0
                for count in child_leaf_counts:
                    p = count / star_u
                    if p > 0:
                        k_entropy += -p * math.log(p)
                
                max_entropy = math.log(eff_children)
                if max_entropy > 0:
                    # h_factor is 1 assuming 0 population for internal nodes
                    j1_numerator += star_u * k_entropy / max_entropy
            
            # --- Colless Index Calculation (only valid if binary tree) ---
            if is_binary:
                if len(curr_children) == 1:
                    colless_sum += child_leaf_counts[0]
                elif len(curr_children) == 2:
                    colless_sum += abs(child_leaf_counts[0] - child_leaf_counts[1])
    
    colless_index = float(colless_sum) if is_binary else None
    j1_index = j1_numerator / j1_denominator if j1_denominator > 0 else 0.0

    return {
        "sackin_index": sackin_index,
        "colless_index": colless_index,
        "j1_index": j1_index,
        "num_trees": len(roots),
        "num_leaves": len(leaf_depths),
        "max_depth": max(leaf_depths) if leaf_depths else 0,
        "is_binary": is_binary
    }
