from typing import Dict, Any, List
from .features import FeatureVector
from .scoring import ScoringEngine
from .safety import SafetyEngine

from typing import Dict, Any, List
from .features import FeatureVector
from .scoring import ScoringEngine
from .safety import SafetyEngine

from typing import Dict, Any, List
import time
from .features import FeatureVector
from .scoring import ScoringEngine
from .safety import SafetyEngine

# Production V3 Integrations
from .compute.nlp_v2 import VectorNLPProcessor
from .compute.graph_v3 import WeightedEigenTrustEngine # V3
from .ledger.settlement import SettlementLedger, RewardState
from .antigaming.temporal import TemporalGraphDetector # V3
from .uncertainty import UncertaintyEngine
from .explainability import ExplainabilityEngine
from .governance import GovernanceEnforcer

class EQSMPipeline:
    def __init__(self):
        self.scorer = ScoringEngine()
        self.safety = SafetyEngine()
        
        # V3 Engines
        self.nlp = VectorNLPProcessor()
        self.graph = WeightedEigenTrustEngine()
        self.ledger = SettlementLedger()
        self.temporal = TemporalGraphDetector()
        self.governance = GovernanceEnforcer()
        
        self.uncertainty = UncertaintyEngine()
        self.explainer = ExplainabilityEngine()
        
    def run_settlement_cycle(self, current_time: int):
        """
        Settles rewards and Updates Trust Graph.
        """
        settled_ids = self.ledger.process_settlements(current_time)
        
        for entry_id in settled_ids:
            entry = self.ledger.entries[entry_id]
            # When reward settles, user becomes 'Settled' in graph
            # unleashing their voting power for FUTURE interactions
            self.graph.mark_settled(entry.author_id)

    def process_content_v3(self, 
                        content_text: str, 
                        author_id: str, 
                        context_texts: List[str] = [],
                        feature_overrides: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Adversarial-Hardened Entry Point.
        """
        current_time = feature_overrides.get('timestamp', time.time())
        
        # 1. Feature Computation
        nlp_metrics = self.nlp.calculate_quality_metrics(content_text, context_texts)
        
        # 2. Temporal Analysis & Recording
        # Record this interaction attempt (Self-interaction implicitly for content creation for now)
        # In real flow this records Rating events.
        self.temporal.record_interaction(author_id, "system", 1.0, current_time)
        temporal_penalty = self.temporal.detect_temporal_anomalies(author_id, current_time)
        
        # 3. Trust Score
        trust_scores = self.graph.compute_global_trust()
        author_trust = trust_scores.get(author_id, 0.0)

        # 4. Construct Vector
        features = FeatureVector(
            author_id=author_id,
            semantic_depth_score=nlp_metrics['semantic_quality'],
            novelty_index=nlp_metrics['novelty'],
            peer_reputation_weighted=author_trust * 10.0, 
            interaction_density=0.0, # Handled by temporal now
            **{k:v for k,v in feature_overrides.items() if hasattr(FeatureVector, k)}
        )

        # 5. Scoring
        if not self.safety.validate_safety_thresholds(features):
            return {"score": 0.0, "status": "BLOCKED"}

        raw_score = self.scorer.calculate_final_score(features)
        
        # Apply Temporal Penalty (Severe)
        final_score = raw_score * (1.0 - temporal_penalty)

        # 6. Ledger
        ledger_entry_id = None
        if self.safety.is_monetizable(features) and final_score > 0.1:
            if temporal_penalty > 0.8:
                # Immediate block/burn if clearly gaming
                pass 
            else:
                ledger_entry_id = self.ledger.create_provisional_entry(
                    content_id=feature_overrides.get('content_id', 'unknown'),
                    author_id=author_id,
                    score=final_score
                )
                self.ledger.lock_entry(ledger_entry_id)

        # 7. Explain
        explanation = self.explainer.explain(features.to_dict(), final_score, temporal_penalty)
        return {
            "score": final_score,
            "ledger_id": ledger_entry_id,
            "temporal_penalty": temporal_penalty,
            "trust_rank": author_trust,
            "explanation": explanation
        }
