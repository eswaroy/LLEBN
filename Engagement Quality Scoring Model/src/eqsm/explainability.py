from typing import Dict, List, Any

class ExplainabilityEngine:
    """
    Generates human-readable explanations for scores.
    """
    
    def explain(self, features: Dict[str, Any], score: float, penalty: float) -> Dict[str, Any]:
        """
        Produces a reasoning vector.
        """
        reasons = []
        
        # Positive Factors
        if features.get('semantic_depth_score', 0) > 0.7:
            reasons.append("High Semantic Depth")
        
        if features.get('novelty_index', 0) > 0.8:
            reasons.append("Highly Original Content")
            
        if features.get('peer_reputation_weighted', 0) > 0.5:
            reasons.append("Validated by Trusted Peers")
            
        # Negative Factors
        if penalty > 0.5:
            reasons.append("Suspicious Pattern Detected")
            
        if features.get('effective_reach', 0) > 10000 and score < 0.6:
            reasons.append("Engagement normalized for massive reach")
            
        return {
            "score": round(score, 3),
            "factors": reasons,
            "private_debug": {
                "penalty_applied": penalty,
                "raw_features": {k: v for k,v in features.items() if isinstance(v, (int, float))}
            }
        }
