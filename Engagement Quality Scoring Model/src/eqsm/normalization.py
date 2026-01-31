import math
from .features import FeatureVector, DeviceContextType

class NormalizationEngine:
    """
    Handles score adjustments for Audience Size and Device Context.
    """
    
    def __init__(self, normalization_floor: float = 0.5, dampening_factor: float = 0.5):
        self.alpha_floor = normalization_floor # Safety Floor
        self.beta_dampening = dampening_factor
        self.epsilon = 1e-9

    def normalize_audience(self, raw_score: float, reach: int) -> float:
        """
        Applies Bounded Engagement Efficiency Ratio (BEER).
        Decouples Quality from Reach with a safety floor for popular creators.
        
        Formula:
        S_norm = S_c * max(alpha, (1 / (log(1 + Reach) + eps)) ^ beta)
        """
        if reach <= 0:
            return raw_score
            
        log_reach = math.log(1 + reach) + self.epsilon
        penalty_factor = (1.0 / log_reach) ** self.beta_dampening
        
        # Apply Safety Floor (max)
        final_modifier = max(self.alpha_floor, penalty_factor)
        
        return raw_score * final_modifier

    def adjust_for_device(self, features: FeatureVector) -> FeatureVector:
        """
        Returns a COPY of features with device-specific adjustments applied.
        Does not mutate the original input.
        """
        adjusted = FeatureVector(**features.to_dict())
        
        if features.device_context_type == DeviceContextType.MOBILE:
            # Mobile: Relax semantic depth expectation
            # We don't change the score, we conceptually lower the threshold in the scoring function.
            # But here, we can boost the feature slightly to match the standard classifier.
            # Strategy: Boost semantic_depth by 20% to normalize against Desktop baseline
            adjusted.semantic_depth_score *= 1.2
            
            # Mobile: Reduce toxicity false positives (often due to slang/brevity)? 
            # (Skipped for now, safety first)
            
        elif features.device_context_type == DeviceContextType.VOICE_ASSIST:
            # Voice: Subtract processing latency
            net_latency = max(0, features.response_latency_s - features.voice_processing_latency_s)
            adjusted.response_latency_s = net_latency
            
        return adjusted
