from .features import FeatureVector, DeviceContextType

class SafetyEngine:
    """
    Enforces Hard Safety & Economic Rules.
    """
    
    @staticmethod
    def is_monetizable(features: FeatureVector) -> bool:
        """
        CRITICAL: Determines if content is eligible for Rewards.
        AI Code and AI Agents are strictly EXCLUDED.
        """
        # 1. AI Exclusion Policy
        if features.author_type == "AI_AGENT":
            return False
            
        if features.is_ai_generated:
            return False
            
        # 2. Human Verification Policy
        # Rewards require verified human ID
        if not features.is_author_human_verified:
            return False
            
        return True

    @staticmethod
    def validate_safety_thresholds(features: FeatureVector) -> bool:
        """
        Checks if content meets minimum safety standards to be scored at all.
        """
        # Global Toxicity Floor (Cannot be overridden by Community Config)
        if features.toxicity_probability > 0.8:
            return False
            
        return True
