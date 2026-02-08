"""
Rolling Buffer
==============
Manages a fixed-size buffer of training samples.
"""

import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import deque

logger = logging.getLogger(__name__)


class RollingBuffer:
    """
    Fixed-size buffer for training samples with persistence.
    
    New samples push out oldest when full.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        learning_config = config.get('learning', {})
        self.size = learning_config.get('buffer_size', 500)
        self.buffer_file = Path(learning_config.get('buffer_file', 'learning_state/buffer.pkl'))
        
        self._buffer: deque = deque(maxlen=self.size)
        self._load()
    
    def add(self, sample: Dict[str, Any]) -> None:
        """Add a sample to the buffer."""
        self._buffer.append(sample)
    
    def add_batch(self, samples: List[Dict[str, Any]]) -> int:
        """Add multiple samples. Returns count added."""
        for s in samples:
            self._buffer.append(s)
        return len(samples)
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all samples."""
        return list(self._buffer)
    
    def get_batch(self, size: int) -> List[Dict[str, Any]]:
        """Get most recent N samples."""
        items = list(self._buffer)
        return items[-size:] if size < len(items) else items
    
    def get_split(self, train_ratio: float = 0.7) -> tuple:
        """
        Split buffer into train/validate sets.
        
        Returns:
            (train_samples, validate_samples)
        """
        items = list(self._buffer)
        split_idx = int(len(items) * train_ratio)
        return items[:split_idx], items[split_idx:]
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
    
    def save(self) -> None:
        """Persist buffer to disk."""
        try:
            self.buffer_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.buffer_file, 'wb') as f:
                pickle.dump(list(self._buffer), f)
            logger.debug(f"Buffer saved: {len(self._buffer)} samples")
        except Exception as e:
            logger.warning(f"Failed to save buffer: {e}")
    
    def _load(self) -> None:
        """Load buffer from disk."""
        if self.buffer_file.exists():
            try:
                with open(self.buffer_file, 'rb') as f:
                    items = pickle.load(f)
                self._buffer = deque(items, maxlen=self.size)
                logger.info(f"Buffer loaded: {len(self._buffer)} samples")
            except Exception as e:
                logger.warning(f"Failed to load buffer: {e}")
    
    def __len__(self) -> int:
        return len(self._buffer)
    
    def stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'count': len(self._buffer),
            'capacity': self.size,
            'utilization': len(self._buffer) / self.size if self.size else 0
        }
