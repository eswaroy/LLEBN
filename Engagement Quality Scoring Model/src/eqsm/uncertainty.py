import math

class UncertaintyEngine:
    """
    Computes statistical confidence intervals for scores.
    """
    
    @staticmethod
    def wilson_score_lower_bound(positive: int, total: int, confidence: float = 0.95) -> float:
        """
        Computes the Lower Bound of the Wilson Score Interval.
        Used for Cold-Start handling: we trust the lower bound, not the average.
        """
        if total == 0:
            return 0.0
            
        z = 1.96 # For 95% confidence
        phat = positive / total
        
        numerator = phat + (z*z)/(2*total) - z * math.sqrt((phat*(1-phat)/total) + (z*z)/(4*total*total))
        denominator = 1 + (z*z)/total
        
        return max(0.0, numerator / denominator)

    @staticmethod
    def calculate_confidence_decay(last_update_ts: int, current_ts: int) -> float:
        """
        Time-based confidence decay. 
        If info is old, confidence drops.
        """
        # Simple linear decay over 30 days
        seconds_in_day = 86400
        days_diff = (current_ts - last_update_ts) / seconds_in_day
        
        if days_diff < 0: return 1.0
        
        decay = max(0.5, 1.0 - (days_diff * 0.01)) # Drops 1% per day, floor at 0.5
        return decay
