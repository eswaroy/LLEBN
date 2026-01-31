from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field

@dataclass
class Node:
    id: str
    reputation: float = 0.5
    interactions_out: int = 0
    interactions_in: int = 0

class GraphEngine:
    """
    In-Memory Interaction Graph.
    Simulates a Graph DB for Reputation Propagation.
    """
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        # Adjacency: src -> [(dest, weight)]
        self.edges: Dict[str, List[Tuple[str, float]]] = {}
        # Reverse Adj for rapid lookup: dest -> [src]
        self.reverse_edges: Dict[str, List[str]] = {}

    def get_or_create_node(self, node_id: str) -> Node:
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(id=node_id)
        return self.nodes[node_id]

    def add_interaction(self, src: str, dest: str, weight: float = 1.0):
        self.get_or_create_node(src).interactions_out += 1
        self.get_or_create_node(dest).interactions_in += 1
        
        if src not in self.edges:
            self.edges[src] = []
        self.edges[src].append((dest, weight))
        
        if dest not in self.reverse_edges:
            self.reverse_edges[dest] = []
        self.reverse_edges[dest].append(src)

    def calculate_peer_reputation_weighted(self, author_id: str) -> float:
        """
        Calculates the sum of reputation of all peers interacting with the author.
        Simulates: Sum(Rater_Rep * Vote_Weight)
        """
        if author_id not in self.reverse_edges:
            return 0.0
            
        score_sum = 0.0
        raters = self.reverse_edges[author_id]
        
        for rater_id in raters:
            rater_node = self.nodes.get(rater_id)
            if rater_node:
                # Basic trusted peer summation
                score_sum += rater_node.reputation
                
        # Normalize roughly (e.g., sigmoid logic downstream handles the range)
        return min(1.0, score_sum / 10.0) # Saturation at 10 reputable peers

    def compute_interaction_density(self, user_id: str) -> float:
        """
        Computes localized density (Cluster Coefficient proxy).
        High density = Potential Ring.
        """
        neighbors = self.reverse_edges.get(user_id, [])
        if len(neighbors) < 2:
            return 0.0
            
        # Check connections between neighbors
        links = 0
        potential_links = len(neighbors) * (len(neighbors) - 1) / 2
        
        # Limited check (BFS depth 1 usually)
        # For simulation, we check if neighbor A knows neighbor B
        neighbor_set = set(neighbors)
        
        for n in neighbors:
            n_out = self.edges.get(n, [])
            for (target, _) in n_out:
                if target in neighbor_set and target != n:
                    links += 1
                    
        # Undirected conversion mostly, keep simple
        return min(1.0, links / potential_links) if potential_links > 0 else 0.0
