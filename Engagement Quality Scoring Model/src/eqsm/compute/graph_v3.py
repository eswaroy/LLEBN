from typing import Dict, List, Set

class WeightedEigenTrustEngine:
    """
    EigenTrust V3: Confidence-Weighted & Settlement-Gated.
    Trust only propagates from Settled nodes.
    """
    def __init__(self, convergence_threshold: float = 0.001):
        self.trust_matrix: Dict[str, Dict[str, float]] = {}
        self.pre_trusted_peers: List[str] = ["genesis_node", "mod_team"] 
        self.convergence = convergence_threshold
        self.C = 0.85
        
        # Gate: Set of nodes that have at least one SETTLED reward
        self.settled_nodes: Set[str] = set(self.pre_trusted_peers)

    def mark_settled(self, node_id: str):
        self.settled_nodes.add(node_id)
        
    def add_rating(self, src: str, dest: str, score: float, confidence: float = 1.0):
        """
        Records trust edge w_ij.
        Weight = Score * Confidence.
        """
        if src not in self.trust_matrix:
            self.trust_matrix[src] = {}
            
        # Effectively, trust = raw_score * confidence
        # And we perform gating during compute or normalization
        self.trust_matrix[src][dest] = max(0.0, score * confidence)

    def compute_global_trust(self) -> Dict[str, float]:
        """
        Compute EigenTrust where unsettled votes count for ZERO.
        """
        # 1. Identify active nodes
        all_nodes = set(self.pre_trusted_peers)
        for src in self.trust_matrix:
            all_nodes.add(src)
            all_nodes.update(self.trust_matrix[src].keys())
        nodes_list = list(all_nodes)
        N = len(nodes_list)
        if N == 0: return {}

        # 2. Trusted Set P
        p_vec = {n: (1.0/len(self.pre_trusted_peers) if n in self.pre_trusted_peers else 0.0) for n in nodes_list}
        t = {n: 1.0/N for n in nodes_list}

        # 3. Normalize Matrix with Gating
        # c_{ij} exists ONLY if src is in self.settled_nodes
        matrix_c = {}
        for src in nodes_list:
            matrix_c[src] = {}
            row_sum = 0.0
            
            # GATING CHECK
            if src in self.settled_nodes:
                # Calculate sum for normalization
                for dest, val in self.trust_matrix.get(src, {}).items():
                    row_sum += val
                
                if row_sum > 0:
                    for dest, val in self.trust_matrix.get(src, {}).items():
                        matrix_c[src][dest] = val / row_sum
                else:
                    # Settled but no votes -> fallback to P
                    for p in self.pre_trusted_peers:
                        matrix_c[src][p] = 1.0 / len(self.pre_trusted_peers)
            else:
                # Unsettled Node -> Their trust vector is forced to match P (Genesis)
                # They cannot vote for their Sybil friends. They effectively vote for the system.
                for p in self.pre_trusted_peers:
                    matrix_c[src][p] = 1.0 / len(self.pre_trusted_peers)

        # 4. Iterate
        for _ in range(20): # Max 20 iters
            t_new = {n: 0.0 for n in nodes_list}
            
            for i in nodes_list:
                weight_from_i = t[i]
                for j, trust_ij in matrix_c.get(i, {}).items():
                    t_new[j] += (self.C * weight_from_i * trust_ij)
            
            for n in nodes_list:
                t_new[n] += (1.0 - self.C) * p_vec[n]
                
            diff = sum(abs(t_new[n] - t[n]) for n in nodes_list)
            t = t_new
            if diff < self.convergence:
                break
                
        return t
