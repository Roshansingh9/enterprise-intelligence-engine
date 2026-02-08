"""
Retrieval Cache
===============
Caches frequent queries and results.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import OrderedDict

logger = logging.getLogger(__name__)


class RetrievalCache:
    """
    LRU cache for retrieval results.
    
    Features:
    - Configurable size and TTL
    - Disk persistence
    - Cache statistics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config['optimization'].get('cache_enabled', True)
        self.max_size = config['optimization'].get('cache_size', 1000)
        self.ttl = config['optimization'].get('cache_ttl', 3600)  # seconds
        
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, datetime] = {}
        self._hits = 0
        self._misses = 0
        
        # Optional disk persistence
        self._cache_path = Path(config['paths'].get('cache', 'cache')) / 'retrieval_cache.json'
    
    def _make_key(self, query: str, filters: Optional[Dict] = None) -> str:
        """Generate cache key from query and filters."""
        key_data = {'query': query.lower().strip()}
        if filters:
            key_data['filters'] = sorted(filters.items())
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, filters: Optional[Dict] = None) -> Optional[List[Dict]]:
        """
        Get cached results for a query.
        
        Args:
            query: Search query
            filters: Optional filters
            
        Returns:
            Cached results or None
        """
        if not self.enabled:
            return None
        
        key = self._make_key(query, filters)
        
        if key not in self._cache:
            self._misses += 1
            return None
        
        # Check TTL
        timestamp = self._timestamps.get(key)
        if timestamp and (datetime.now() - timestamp).total_seconds() > self.ttl:
            self._remove(key)
            self._misses += 1
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        
        logger.debug(f"Cache hit for query: {query[:50]}...")
        return self._cache[key]
    
    def set(self, query: str, results: List[Dict], filters: Optional[Dict] = None) -> None:
        """
        Cache results for a query.
        
        Args:
            query: Search query
            results: Results to cache
            filters: Optional filters
        """
        if not self.enabled:
            return
        
        key = self._make_key(query, filters)
        
        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            self._remove(oldest_key)
        
        self._cache[key] = results
        self._timestamps[key] = datetime.now()
    
    def _remove(self, key: str) -> None:
        """Remove an entry from cache."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._timestamps.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        
        return {
            'enabled': self.enabled,
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'ttl_seconds': self.ttl
        }
    
    def save(self) -> None:
        """Save cache to disk."""
        if not self.enabled:
            return
        
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'cache': dict(self._cache),
                'timestamps': {k: v.isoformat() for k, v in self._timestamps.items()},
                'stats': {'hits': self._hits, 'misses': self._misses}
            }
            
            with open(self._cache_path, 'w') as f:
                json.dump(data, f)
            
            logger.debug(f"Cache saved: {len(self._cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def load(self) -> None:
        """Load cache from disk."""
        if not self.enabled or not self._cache_path.exists():
            return
        
        try:
            with open(self._cache_path, 'r') as f:
                data = json.load(f)
            
            self._cache = OrderedDict(data.get('cache', {}))
            self._timestamps = {
                k: datetime.fromisoformat(v) 
                for k, v in data.get('timestamps', {}).items()
            }
            
            stats = data.get('stats', {})
            self._hits = stats.get('hits', 0)
            self._misses = stats.get('misses', 0)
            
            # Expire old entries
            now = datetime.now()
            expired = [
                k for k, v in self._timestamps.items()
                if (now - v).total_seconds() > self.ttl
            ]
            for k in expired:
                self._remove(k)
            
            logger.info(f"Cache loaded: {len(self._cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
