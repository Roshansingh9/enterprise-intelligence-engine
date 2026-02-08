"""
Query Rewriter
==============
Rewrites user queries for better retrieval using LLM.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Rewrites user queries to improve retrieval quality.
    
    Techniques:
    - Query expansion
    - Synonym injection
    - Intent clarification
    - Typo correction
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config['optimization'].get('query_rewrite_enabled', True)
        self._llm_client = None
    
    @property
    def llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None:
            from src.agents.llm_client import LLMClient
            self._llm_client = LLMClient(self.config)
        return self._llm_client
    
    def rewrite(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Rewrite a query for better retrieval.
        
        Args:
            query: Original user query
            context: Optional context (product, category, etc.)
            
        Returns:
            Rewritten query
        """
        if not self.enabled:
            return query
        
        try:
            context_str = ""
            if context:
                context_str = f"\nContext: Product={context.get('product', 'Unknown')}, Category={context.get('category', 'Unknown')}"
            
            prompt = f"""Rewrite this customer support query to be more specific and searchable.
Fix typos, expand abbreviations, and add relevant keywords.
Keep it concise (under 50 words).

Original query: {query}{context_str}

Rewritten query:"""
            
            rewritten = self.llm_client.generate(prompt, max_tokens=100)
            rewritten = rewritten.strip().strip('"').strip("'")
            
            if len(rewritten) > 10 and len(rewritten) < 200:
                logger.debug(f"Query rewritten: '{query}' → '{rewritten}'")
                return rewritten
            
            return query
            
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return query
    
    def expand(self, query: str) -> List[str]:
        """
        Generate query variations for multi-query retrieval.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations
        """
        variations = [query]
        
        if not self.enabled:
            return variations
        
        try:
            prompt = f"""Generate 3 alternative phrasings of this customer support query.
Each should capture the same intent but use different words.

Original: {query}

Return only the 3 alternatives, one per line:"""
            
            response = self.llm_client.generate(prompt, max_tokens=200)
            
            for line in response.strip().split('\n'):
                line = line.strip().lstrip('0123456789.-) ')
                if len(line) > 10 and line not in variations:
                    variations.append(line)
            
            return variations[:4]  # Max 4 variations
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return variations
