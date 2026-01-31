from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

class DeviceContextType(Enum):
    DESKTOP = "DESKTOP"
    MOBILE = "MOBILE"
    VOICE_ASSIST = "VOICE_ASSIST"

class EvaluationContextMode(Enum):
    STANDARD = "STANDARD"
    GROUP_CHALLENGE = "GROUP_CHALLENGE"
    HUMAN_VS_AI = "HUMAN_VS_AI"

@dataclass
class FeatureVector:
    """
    Standardized Feature Store Interface for EQSM.
    Contains all input signals required for scoring.
    """
    # A. Content Quality
    semantic_depth_score: float = 0.0  # F_C01
    novelty_index: float = 0.0         # F_C02
    toxicity_probability: float = 0.0  # F_C03
    
    # B. Temporal
    response_latency_s: int = 0        # F_T01
    voice_processing_latency_s: int = 0 # Helper for Voice Mode
    session_burstiness: float = 0.0    # F_T02
    
    # C. Trust & Graph
    peer_reputation_weighted: float = 0.0 # F_G01
    interaction_density: float = 0.0      # F_G02
    
    # D. User History
    domain_consistency: float = 0.0       # F_U01
    account_age_days: int = 0             # F_U02
    
    # E. Audience & Reach
    effective_reach: int = 0              # F_A01
    
    # F. Device Context
    device_context_type: DeviceContextType = DeviceContextType.DESKTOP # F_D01
    
    # G. Evaluation Mode
    evaluation_context_mode: EvaluationContextMode = EvaluationContextMode.STANDARD # F_M01
    
    # H. Cold Start / Safety
    confidence_interval_lower: float = 0.0 # F_S01
    is_new_entity: bool = False            # F_S02
    
    # Metadata (Not for scoring formula, but for routing)
    author_id: str = ""
    is_author_human_verified: bool = False
    author_type: str = "HUMAN" # HUMAN, AI_AGENT
    content_id: str = ""
    is_ai_generated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
