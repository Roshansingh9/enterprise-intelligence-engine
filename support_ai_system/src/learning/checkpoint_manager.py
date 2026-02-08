"""
Checkpoint Manager
==================
Manages training checkpoints for resumable learning.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Saves and restores training checkpoints.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        learning_config = config.get('learning', {})
        self.checkpoint_file = Path(learning_config.get('checkpoint_file', 'learning_state/checkpoints.json'))
        self.interval = learning_config.get('checkpoint_interval', 50)
        
        self._checkpoints: List[Dict] = []
        self._load()
    
    def save(
        self,
        round_num: int,
        batch_id: str = None,
        metrics: Dict = None,
        state: Dict = None
    ) -> str:
        """
        Save a checkpoint.
        
        Returns:
            Checkpoint ID
        """
        checkpoint = {
            'id': f"CP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'round': round_num,
            'batch_id': batch_id,
            'metrics': metrics or {},
            'state': state or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self._checkpoints.append(checkpoint)
        self._persist()
        
        logger.info(f"Checkpoint saved: {checkpoint['id']} (round {round_num})")
        return checkpoint['id']
    
    def get_latest(self) -> Optional[Dict]:
        """Get most recent checkpoint."""
        return self._checkpoints[-1] if self._checkpoints else None
    
    def get_by_round(self, round_num: int) -> Optional[Dict]:
        """Get checkpoint for specific round."""
        for cp in reversed(self._checkpoints):
            if cp.get('round') == round_num:
                return cp
        return None
    
    def get_all(self) -> List[Dict]:
        """Get all checkpoints."""
        return self._checkpoints.copy()
    
    def clear(self) -> None:
        """Clear all checkpoints."""
        self._checkpoints = []
        self._persist()
    
    def should_checkpoint(self, batch_count: int) -> bool:
        """Check if we should save based on interval."""
        return batch_count > 0 and batch_count % self.interval == 0
    
    def _persist(self) -> None:
        """Save checkpoints to disk."""
        try:
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self._checkpoints, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to persist checkpoints: {e}")
    
    def _load(self) -> None:
        """Load checkpoints from disk."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self._checkpoints = json.load(f)
                logger.info(f"Loaded {len(self._checkpoints)} checkpoints")
            except Exception as e:
                logger.warning(f"Failed to load checkpoints: {e}")
    
    def stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics."""
        return {
            'count': len(self._checkpoints),
            'latest_round': self._checkpoints[-1].get('round') if self._checkpoints else 0,
            'interval': self.interval
        }
