from typing import Dict, List, Any
from collections import defaultdict
import time

class AntiGamingDetector:
    """
    Detects Sybil rings, coordinated behavior, and farming patterns.
    """
    def __init__(self):
        # pair -> [timestamps]
        self.interaction_history: Dict[str, List[float]] = defaultdict(list)
        
    def record_interaction(self, src: str, dest: str):
        key = tuple(sorted((src, dest)))
        self.interaction_history[key].append(time.time())

    def calculate_mutual_decay(self, src: str, dest: str) -> float:
        """
        Returns a multiplier (0.0 to 1.0) based on interaction frequency.
        If src and dest interact too often, their votes are worth less.
        """
        key = tuple(sorted((src, dest)))
        count = len(self.interaction_history.get(key, []))
        
        if count <= 3:
            return 1.0
        
        # Exponential decay: 0.9 ^ (count - 3)
        return max(0.1, 0.9 ** (count - 3))

    def detect_rings(self, graph_density: float, burstiness: float) -> float:
        """
        Returns Penalty Magnitude (0.0 to 1.0).
        High density + Low variance (Burstiness) = Automated Ring.
        """
        # Feature: Regularity (Low Burstiness) is suspicious for bots
        # Feature: High Density is suspicious for rings
        
        is_suspicious_structure = graph_density > 0.7
        is_robotic_timing = burstiness < 0.1 # Very regular intervals
        
        if is_suspicious_structure and is_robotic_timing:
            return 1.0 # Max Penalty
        elif is_suspicious_structure:
            return 0.5
        elif is_robotic_timing:
            return 0.3
            
        return 0.0
