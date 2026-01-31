from typing import Dict, Any

class GovernanceEnforcer:
    """
    Ensures that only Settled Reputation translates to Governance Power.
    """
    
    @staticmethod
    def calculate_governance_weight(trust_score: float, is_settled: bool) -> float:
        """
        Returns the voting weight for a user.
        """
        if not is_settled:
            return 0.0
            
        # Non-linear power ramp to prevent "buying" influence linearly
        # Power = Trust^1.5 (Rewards high trust exponentially more than med trust)
        return pow(trust_score, 1.5)

    @staticmethod
    def validate_action_eligibility(user_trust: float, required_tier: float) -> bool:
        """
        Gates sensitive actions (e.g. creating proposals).
        """
        return user_trust >= required_tier
