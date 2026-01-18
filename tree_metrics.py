from collections import defaultdict
from typing import Any, Dict, List, Optional

def compute_tree_metrics(roots: List[str], children: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Compute tree balance metrics (Sackin, Colless) and other stats for a forest of trees.
    
    Args:
        roots: List of root node IDs.
        children: Adjacency list mapping node ID to list of child node IDs.
        
    Returns:
        Dictionary containing:
            - sackin_index: Sum of leaf depths (lower is more balanced).
            - colless_index: Sum of |L-R| for binary nodes (None if tree is not binary).
            - num_trees: Number of roots.
            - num_leaves: Number of leaves.
            - max_depth: Maximum depth in the forest.
            - is_binary: Boolean indicating if the forest is strictly binary (<=2 children per node).
    """
    # Compute Sackin (sum of leaf depths)
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
    
    # Compute Colless (for binary trees)
    colless_index = None
    if is_binary:
        # Sort nodes by depth descending to process children first
        sorted_nodes = sorted(node_depth.keys(), key=lambda k: node_depth[k], reverse=True)
        subtree_leaves = defaultdict(int)
        colless_sum = 0
        
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
                
                if len(curr_children) == 1:
                    colless_sum += child_leaf_counts[0]
                elif len(curr_children) == 2:
                    colless_sum += abs(child_leaf_counts[0] - child_leaf_counts[1])
        
        colless_index = float(colless_sum)

    return {
        "sackin_index": sackin_index,
        "colless_index": colless_index,
        "num_trees": len(roots),
        "num_leaves": len(leaf_depths),
        "max_depth": max(leaf_depths) if leaf_depths else 0,
        "is_binary": is_binary
    }
