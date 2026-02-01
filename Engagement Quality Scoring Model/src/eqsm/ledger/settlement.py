from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import hashlib

class RewardState(Enum):
    PROVISIONAL = "PROVISIONAL" # Calculated, shown in UI, not promised
    LOCKED = "LOCKED"           # Promised, waiting for anti-gaming window (T+7)
    SETTLED = "SETTLED"         # Finalized, ready for payout
    BURNED = "BURNED"           # Flagged by gaming, removed

@dataclass
class LedgerEntry:
    entry_id: str
    content_id: str
    author_id: str
    amount: float
    state: RewardState
    created_at: int
    settle_at: int
    audit_hash: str = ""

class SettlementLedger:
    def __init__(self, settlement_window_seconds: int = 7 * 86400):
        self.entries: Dict[str, LedgerEntry] = {}
        self.settlement_window = settlement_window_seconds

    def create_provisional_entry(self, content_id: str, author_id: str, score: float) -> str:
        """
        Creates a new entry.
        """
        entry_id = hashlib.sha256(f"{content_id}{author_id}{time.time()}".encode()).hexdigest()
        
        # Logic: Score 0-1 maps to generic Reward Units
        amount = score * 100 
        
        entry = LedgerEntry(
            entry_id=entry_id,
            content_id=content_id,
            author_id=author_id,
            amount=amount,
            state=RewardState.PROVISIONAL,
            created_at=int(time.time()),
            settle_at=int(time.time()) + self.settlement_window
        )
        self.entries[entry_id] = entry
        return entry_id

    def lock_entry(self, entry_id: str):
        """
        Moves to LOCKED state. Author sees this in 'Pending Balance'.
        """
        if entry_id in self.entries:
            self.entries[entry_id].state = RewardState.LOCKED

    def process_settlements(self, current_time: int) -> List[str]:
        """
        called by cron. Moves eligible LOCKED -> SETTLED.
        Returns list of settled entry IDs.
        """
        settled = []
        for eid, entry in self.entries.items():
            if entry.state == RewardState.LOCKED and current_time >= entry.settle_at:
                entry.state = RewardState.SETTLED
                self.finalize_audit(entry)
                settled.append(eid)
        return settled

    def flag_and_burn(self, content_id: str, reason: str):
        """
        Anti-Gaming Trigger. Burns any non-settled rewards for content.
        """
        for entry in self.entries.values():
            if entry.content_id == content_id:
                if entry.state in [RewardState.PROVISIONAL, RewardState.LOCKED]:
                    entry.state = RewardState.BURNED
                    entry.audit_hash = f"BURNED: {reason}"

    def finalize_audit(self, entry: LedgerEntry):
        # Create immutable hash of the final state
        raw = f"{entry.entry_id}:{entry.amount}:{entry.state}:{entry.settle_at}"
        entry.audit_hash = hashlib.sha256(raw.encode()).hexdigest()
