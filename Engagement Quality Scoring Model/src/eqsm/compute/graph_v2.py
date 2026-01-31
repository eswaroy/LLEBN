from typing import Dict, List, Tuple

class EigenTrustEngine:
    """
    Implements Global Trust Propagation using EigenTrust Algorithm.
    T = C * M.T * T + (1-C) * P
    """
    def __init__(self, convergence_threshold: float = 0.001, max_iterations: int = 20):
        # Adjacency: src -> {dest: trust_score}
        self.trust_matrix: Dict[str, Dict[str, float]] = {}
        # Pre-trusted peers (The "Founders" / "Bootstrap" set)
        self.pre_trusted_peers: List[str] = ["genesis_node"] 
        self.convergence = convergence_threshold
        self.max_iter = max_iterations
        self.C = 0.85 # Decay factor (like Damping factor in PageRank)

    def add_rating(self, src: str, dest: str, score: float):
        """
        Records a local trust rating s_{ij}
        """
        if src not in self.trust_matrix:
            self.trust_matrix[src] = {}
        # Normalize local trust? standard EigenTrust normalizes c_{ij}
        # We store raw, then normalize row-wise during compute
        self.trust_matrix[src][dest] = max(0.0, score) # Only positive trust propagates

    def _normalize_matrix(self, nodes: List[str]) -> Dict[str, Dict[str, float]]:
        """
        c_{ij} = max(s_{ij}, 0) / sum(max(s_{ik}, 0))
        """
        normalized = {}
        for src in nodes:
            normalized[src] = {}
            row_sum = sum(self.trust_matrix.get(src, {}).values())
            
            if row_sum > 0:
                for dest, val in self.trust_matrix.get(src, {}).items():
                    normalized[src][dest] = val / row_sum
            else:
                # If node trusts no one, it trusts the pre-trusted peers (or itself)
                # Standard EigenTrust fallback: uniform over P
                p_len = len(self.pre_trusted_peers)
                for p in self.pre_trusted_peers:
                    normalized[src][p] = 1.0 / p_len
        return normalized

    def compute_global_trust(self) -> Dict[str, float]:
        """
        Runs the iterative power method to find the Eigenvector.
        Returns map {node_id: global_trust_score}
        """
        # 1. Identify all nodes
        all_nodes = set(self.trust_matrix.keys())
        for src in self.trust_matrix:
            all_nodes.update(self.trust_matrix[src].keys())
        all_nodes.update(self.pre_trusted_peers)
        nodes_list = list(all_nodes)
        N = len(nodes_list)
        
        if N == 0: return {}

        # 2. Initialize Trust Vector t_0 (Uniform or P)
        t = {n: 1.0/N for n in nodes_list}
        
        # 3. Define P vector (Pre-trusted distribution)
        p_vec = {n: (1.0/len(self.pre_trusted_peers) if n in self.pre_trusted_peers else 0.0) for n in nodes_list}
        
        # 4. Normalize Cij Matrix
        matrix_c = self._normalize_matrix(nodes_list)

        # 5. Iterate
        for _ in range(self.max_iter):
            t_new = {n: 0.0 for n in nodes_list}
            
            # t_{k+1} = C * M.T * t_k + (1-C) * p
            
            # Compute M.T * t_k
            # i.e., node j's new trust is sum of (trust of i * normalized trust i->j)
            for i in nodes_list:
                weight_from_i = t[i]
                start_trusts = matrix_c.get(i, {})
                for j, trust_ij in start_trusts.items():
                    t_new[j] += (self.C * weight_from_i * trust_ij)
            
            # Add (1-C) * p
            for n in nodes_list:
                t_new[n] += (1.0 - self.C) * p_vec[n]
                
            # Check convergence
            diff = sum(abs(t_new[n] - t[n]) for n in nodes_list)
            t = t_new
            if diff < self.convergence:
                break
                
        return t
