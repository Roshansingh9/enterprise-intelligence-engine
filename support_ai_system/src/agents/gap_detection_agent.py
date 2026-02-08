"""
Gap Detection Agent
===================
Identifies missing knowledge and gaps in the KB.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeGap:
    """A detected knowledge gap."""
    gap_type: str  # missing_article, outdated, incomplete, conflicting
    description: str
    topic: str
    severity: str  # low, medium, high, critical
    confidence: float = 0.0
    source_queries: List[str] = field(default_factory=list)
    suggested_action: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class GapDetectionAgent(BaseAgent):
    """
    Detects gaps in knowledge base coverage.
    
    Gap types:
    - missing_article: No KB article covers this topic
    - outdated: Existing article may be out of date
    - incomplete: Article exists but missing information
    - conflicting: Multiple articles with conflicting info
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, 'gap_detector')
        self.min_confidence = self.agent_config.get('min_gap_confidence', 0.7)
    
    def process(self, input_data: Dict[str, Any]) -> List[KnowledgeGap]:
        """
        Detect knowledge gaps.
        
        Args:
            input_data: Dict with:
                - 'query': User query that wasn't answered well
                - 'retrieval_results': Results from retrieval (if any)
                - 'context': Additional context
                
        Returns:
            List of KnowledgeGap objects
        """
        query = input_data.get('query', '')
        results = input_data.get('retrieval_results', [])
        context = input_data.get('context', {})
        
        # Build gap detection prompt
        prompt = self._build_prompt(query, results, context)
        
        # Call LLM
        response = self.llm.generate_structured(
            prompt,
            schema={
                'gaps': [
                    {
                        'type': 'string',
                        'description': 'string',
                        'topic': 'string',
                        'severity': 'string',
                        'confidence': 'number',
                        'suggested_action': 'string'
                    }
                ]
            }
        )
        
        # Parse response
        gaps = []
        for gap_data in response.get('gaps', []):
            confidence = float(gap_data.get('confidence', 0))
            
            if confidence >= self.min_confidence:
                gap = KnowledgeGap(
                    gap_type=gap_data.get('type', 'missing_article'),
                    description=gap_data.get('description', ''),
                    topic=gap_data.get('topic', query),
                    severity=gap_data.get('severity', 'medium'),
                    confidence=confidence,
                    source_queries=[query],
                    suggested_action=gap_data.get('suggested_action', ''),
                    metadata=context
                )
                gaps.append(gap)
        
        logger.debug(f"Detected {len(gaps)} knowledge gaps for query: {query[:50]}")
        return gaps
    
    def _build_prompt(self, query: str, results: List[Dict], context: Dict) -> str:
        """Build gap detection prompt."""
        
        results_text = ""
        if results:
            for i, r in enumerate(results[:5]):
                results_text += f"\n{i+1}. {r.get('title', 'Untitled')}: {r.get('content', '')[:200]}..."
        else:
            results_text = "No relevant articles found."
        
        if self.prompt_template:
            return self.prompt_template.format(
                query=query,
                results=results_text,
                context=context
            )
        
        return f"""Analyze if this customer query reveals a knowledge gap in our KB.

Customer Query: {query}

Existing KB Articles Found:
{results_text}

Context: Product={context.get('product', 'Unknown')}, Category={context.get('category', 'Unknown')}

Determine if there are gaps:
1. Is there a missing article that should exist?
2. Are existing articles outdated or incomplete?
3. Are there conflicting articles?

For each gap, provide:
- type: missing_article, outdated, incomplete, or conflicting
- description: What's missing or wrong
- topic: The topic that needs coverage
- severity: low, medium, high, or critical
- confidence: 0.0 to 1.0
- suggested_action: What should be done"""
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input."""
        if not isinstance(input_data, dict):
            return False
        
        query = input_data.get('query', '')
        return bool(query and len(query) > 10)
    
    def validate_output(self, output_data: Any) -> bool:
        """Validate output."""
        if not isinstance(output_data, list):
            return False
        
        return all(isinstance(g, KnowledgeGap) for g in output_data)
    
    def analyze_unanswered_queries(
        self,
        queries: List[Dict[str, Any]]
    ) -> List[KnowledgeGap]:
        """
        Batch analyze multiple unanswered queries for gaps.
        
        Args:
            queries: List of query dicts with retrieval results
            
        Returns:
            Deduplicated list of gaps
        """
        all_gaps = []
        seen_topics = set()
        
        for query_data in queries:
            gaps = self.run(query_data)
            
            if gaps:
                for gap in gaps:
                    # Deduplicate by topic
                    topic_key = gap.topic.lower().strip()
                    if topic_key not in seen_topics:
                        seen_topics.add(topic_key)
                        all_gaps.append(gap)
                    else:
                        # Merge source queries
                        for existing in all_gaps:
                            if existing.topic.lower().strip() == topic_key:
                                existing.source_queries.extend(gap.source_queries)
                                break
        
        # Sort by severity and confidence
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_gaps.sort(key=lambda g: (severity_order.get(g.severity, 4), -g.confidence))
        
        return all_gaps
