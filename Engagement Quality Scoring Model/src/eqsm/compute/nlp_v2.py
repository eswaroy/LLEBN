import math
import random
from typing import List, Dict, Any

class MockVectorEncoder:
    """
    Simulates a Transformer Model (e.g., SBERT).
    In production, this would load 'all-MiniLM-L6-v2'.
    Here, we generate deterministically distinct vectors based on hash of content
    to simulate semantic space.
    """
    def encode(self, text: str) -> List[float]:
        # Dimension 384 (Standard for MiniLM)
        # We simulate semantics by hashing words.
        # Similar words -> Similar hash distribution -> params
        
        # Simple deterministic vector generation for demo
        seed = sum(ord(c) for c in text)  
        random.seed(seed)
        return [random.uniform(-1.0, 1.0) for _ in range(384)]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(a*b for a,b in zip(v1, v2))
    norm_a = math.sqrt(sum(a*a for a in v1))
    norm_b = math.sqrt(sum(b*b for b in v2))
    return dot_product / (norm_a * norm_b + 1e-9)

class VectorNLPProcessor:
    def __init__(self):
        self.encoder = MockVectorEncoder()

    def calculate_quality_metrics(self, text: str, context_texts: List[str]) -> Dict[str, float]:
        """
        Computes 3 dimensions:
        1. Semantic Quality (Coherence/Structure mapped to latent quality space)
        2. Novelty (Distance from context)
        3. Redundancy (Self-similarity vs thread)
        """
        if not text:
            return {"semantic_quality": 0.0, "novelty": 0.0, "coherence": 0.0}

        # 1. Embed Input
        doc_vector = self.encoder.encode(text)
        
        # 2. Compute Novelty (1 - max_similarity_to_context)
        max_sim = 0.0
        if context_texts:
            for ctx in context_texts:
                ctx_vector = self.encoder.encode(ctx)
                sim = cosine_similarity(doc_vector, ctx_vector)
                max_sim = max(max_sim, sim)
        
        novelty_score = 1.0 - max_sim

        # 3. Compute Coherence (Internal Consistency)
        # Split into "sentences" and check flow
        sentences = [s.strip() for s in text.split('.') if len(s) > 10]
        coherence_score = 0.5 # Default
        if len(sentences) > 1:
            sent_vectors = [self.encoder.encode(s) for s in sentences]
            flows = []
            for i in range(len(sent_vectors)-1):
                flows.append(cosine_similarity(sent_vectors[i], sent_vectors[i+1]))
            
            # High flow similarity = coherent argument
            # Too high (>0.95) = repetitive/bot
            avg_flow = sum(flows) / len(flows)
            coherence_score = avg_flow
            
        # 4. Semantic Quality (Gold Standard Distance)
        # In real sys, we measure cosine dist to "High Quality Centroids"
        # Here we simulate it based on length/structure properties encoded in vector
        # (Mocking return for verification)
        quality_score = min(1.0, len(text) / 200.0) * (0.5 + coherence_score/2)

        return {
            "semantic_quality": round(quality_score, 4),
            "novelty": round(novelty_score, 4),
            "coherence": round(coherence_score, 4)
        }
