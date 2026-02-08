"""
Hallucination Checker
=====================
Validates KB articles against source facts.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class HallucinationChecker:
    """
    Checks generated content for hallucinations by comparing
    against source facts and citations.
    """
    
    def __init__(self, config: Dict[str, Any], llm_client=None):
        self.config = config
        self.llm = llm_client
        gov_config = config.get('governance', {})
        self.enabled = gov_config.get('hallucination_check_enabled', True)
        self.citation_required = gov_config.get('citation_required', True)
    
    def check(
        self,
        generated_content: str,
        source_facts: List[Dict],
        source_text: str = ""
    ) -> Dict[str, Any]:
        """
        Check for hallucinations.
        
        Args:
            generated_content: The generated KB article content
            source_facts: List of extracted facts used to generate
            source_text: Original source text (transcript, ticket)
            
        Returns:
            {is_valid, confidence, issues, citations_found}
        """
        if not self.enabled:
            return {'is_valid': True, 'confidence': 1.0, 'issues': [], 'citations_found': []}
        
        result = {
            'is_valid': True,
            'confidence': 0.8,
            'issues': [],
            'citations_found': []
        }
        
        # Basic checks without LLM
        if not generated_content:
            result['is_valid'] = False
            result['issues'].append('Empty content')
            result['confidence'] = 0.0
            return result
        
        # Check content length vs source
        if len(generated_content) > len(source_text) * 3 and source_text:
            result['issues'].append('Content significantly longer than source')
            result['confidence'] -= 0.2
        
        # Check for fact coverage
        fact_contents = [f.get('content', '') if isinstance(f, dict) else str(f) for f in source_facts]
        covered = 0
        for fact in fact_contents:
            # Simple keyword overlap check
            keywords = set(fact.lower().split()[:5])
            content_words = set(generated_content.lower().split())
            if len(keywords & content_words) >= 2:
                covered += 1
                result['citations_found'].append(fact[:100])
        
        if source_facts and covered == 0:
            result['issues'].append('No source facts reflected in content')
            result['confidence'] -= 0.3
        elif source_facts:
            coverage_ratio = covered / len(source_facts)
            if coverage_ratio < 0.5:
                result['issues'].append(f'Only {covered}/{len(source_facts)} facts covered')
                result['confidence'] -= 0.1
        
        # Citation requirement check
        if self.citation_required and not result['citations_found']:
            result['issues'].append('No citations to source material')
            result['confidence'] -= 0.1
        
        # LLM-based deep check (if available and content is suspicious)
        if self.llm and result['confidence'] < 0.7:
            llm_result = self._llm_check(generated_content, source_facts, source_text)
            result.update(llm_result)
        
        # Final validity
        result['is_valid'] = result['confidence'] >= 0.5 and not any(
            'Empty' in i or 'No source facts' in i for i in result['issues']
        )
        result['confidence'] = max(0.0, min(1.0, result['confidence']))
        
        return result
    
    def _llm_check(
        self,
        content: str,
        facts: List[Dict],
        source: str
    ) -> Dict[str, Any]:
        """LLM-based hallucination check."""
        try:
            facts_text = '\n'.join([
                f.get('content', str(f))[:100] if isinstance(f, dict) else str(f)[:100]
                for f in facts[:5]
            ])
            
            prompt = f"""Check if this KB article is grounded in the source facts. Return JSON only.

Source Facts:
{facts_text}

Generated Article:
{content[:500]}

{{"is_grounded": true|false, "confidence": 0.8, "issues": ["issue1"]}}"""

            response = self.llm.generate_structured(prompt, schema={
                'is_grounded': 'boolean',
                'confidence': 'number',
                'issues': ['string']
            })
            
            return {
                'is_valid': response.get('is_grounded', True),
                'confidence': response.get('confidence', 0.5),
                'issues': response.get('issues', [])
            }
        except Exception as e:
            logger.warning(f"LLM hallucination check failed: {e}")
            return {}
