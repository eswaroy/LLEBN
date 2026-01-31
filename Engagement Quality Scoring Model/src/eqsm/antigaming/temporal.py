from typing import List, Dict, Tuple
from collections import defaultdict
import time

class TemporalGraphDetector:
    """
    Analyzes interaction graphs over time windows (7d, 30d, 90d).
    Detects 'Slow Burn' farming and 'Trust Laundering' cycles.
    """
    def __init__(self):
        # (src, dest) -> [(timestamp, score)]
        self.interactions: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
        
    def record_interaction(self, src: str, dest: str, score: float, timestamp: float = None):
        if timestamp is None:
            timestamp = time.time()
        self.interactions[(src, dest)].append((timestamp, score))

    def detect_temporal_anomalies(self, author_id: str, current_time: float = None) -> float:
        """
        Returns a penalty score (0.0 - 1.0) based on temporal patterns.
        """
        if current_time is None:
            current_time = time.time()
            
        penalty = 0.0
        
        # 1. Check for Trust Laundering Cycles (A->B->C->A)
        # Simplified 3-hop check with time constraints
        # Get recent partners (last 24h)
        recent_partners = self._get_partners(author_id, current_time - 86400, current_time)
        
        for p1 in recent_partners:
            p1_partners = self._get_partners(p1, current_time - 86400 * 2, current_time) # Look slightly wider
            if author_id in p1_partners:
                # Direct ping-pong (A <-> B)
                penalty = max(penalty, 0.5)
            
            for p2 in p1_partners:
                if p2 == author_id: continue # Handled above
                p2_partners = self._get_partners(p2, current_time - 86400 * 3, current_time)
                if author_id in p2_partners:
                    # A -> B -> C -> A Cycle
                    penalty = max(penalty, 0.8) # High penalty for laundering rings
                    
        # 2. Check for "Slow Burn" Awakening
        # If user was dormant for > 30d then suddenly has high interactions
        if self._is_sudden_awakening(author_id, current_time):
             # Cap penalty, as it could be viral content, but suspicious if combined with other signals
             penalty = max(penalty, 0.2) 

        return penalty

    def _get_partners(self, node: str, start_time: float, end_time: float) -> List[str]:
        # Expensive scan (In prod: Graph DB query)
        partners = []
        for (src, dest), history in self.interactions.items():
            if src == node:
                for ts, _ in history:
                    if start_time <= ts <= end_time:
                        partners.append(dest)
                        break
        return partners
        
    def _is_sudden_awakening(self, node: str, current_time: float) -> bool:
        # Check interaction volume 90d ago vs 2d ago
        old_volume = 0
        recent_volume = 0
        
        for (src, dest), history in self.interactions.items():
            if src == node or dest == node:
                for ts, _ in history:
                    if ts > current_time - 172800: # Last 2 days
                        recent_volume += 1
                    elif ts < current_time - 2592000: # Older than 30d
                        old_volume += 1
        
        if old_volume == 0 and recent_volume > 10:
            return True # New account bursting or sleeper cell waking
        return False
