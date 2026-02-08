"""
Compliance Agent
================
Enforces policy compliance and safety checks.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class ComplianceResult:
    """Result of compliance check."""
    is_compliant: bool
    violations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence: float = 1.0
    sanitized_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceAgent(BaseAgent):
    """
    Enforces policy compliance for KB articles.
    
    Checks:
    - PII detection and scrubbing
    - Policy violations
    - Hallucination detection
    - Citation requirements
    - Safety checks
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, 'compliance')
        self.strict_mode = self.agent_config.get('strict_mode', True)
        
        # Governance settings
        gov_config = config.get('governance', {})
        self.pii_enabled = gov_config.get('pii_detection_enabled', True)
        self.pii_entities = gov_config.get('pii_entities', [])
        self.pii_action = gov_config.get('pii_action', 'redact')
        self.citation_required = gov_config.get('citation_required', True)
        self.hallucination_check = gov_config.get('hallucination_check_enabled', True)
        
        # PII patterns (fallback if Presidio unavailable)
        self._pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b',
            'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        }
        
        # Presidio analyzer (lazy loaded)
        self._analyzer = None
        self._anonymizer = None
    
    def process(self, input_data: Dict[str, Any]) -> ComplianceResult:
        """
        Check content for compliance.
        
        Args:
            input_data: Dict with:
                - 'content': Text content to check
                - 'article': Optional full article object
                - 'source_facts': Optional source facts for citation check
                
        Returns:
            ComplianceResult object
        """
        content = input_data.get('content', '')
        article = input_data.get('article')
        source_facts = input_data.get('source_facts', [])
        
        violations = []
        warnings = []
        sanitized = content
        
        # Check 1: PII Detection
        if self.pii_enabled:
            pii_result = self._check_pii(content)
            if pii_result['found']:
                violations.extend([
                    {'type': 'pii', 'entity': e['type'], 'text': e['text']}
                    for e in pii_result['entities']
                ])
                sanitized = pii_result['sanitized']
        
        # Check 2: Policy violations
        policy_violations = self._check_policies(content)
        violations.extend(policy_violations)
        
        # Check 3: Citation requirements
        if self.citation_required and source_facts:
            citation_issues = self._check_citations(content, source_facts)
            if citation_issues:
                warnings.extend(citation_issues)
        
        # Check 4: Hallucination detection
        if self.hallucination_check and source_facts:
            hallucination_result = self._check_hallucinations(content, source_facts)
            if hallucination_result['detected']:
                violations.append({
                    'type': 'hallucination',
                    'description': hallucination_result['description'],
                    'confidence': hallucination_result['confidence']
                })
        
        # Determine overall compliance
        is_compliant = len(violations) == 0
        if self.strict_mode and warnings:
            is_compliant = False
        
        return ComplianceResult(
            is_compliant=is_compliant,
            violations=violations,
            warnings=warnings,
            confidence=1.0 - (len(violations) * 0.2),
            sanitized_content=sanitized if sanitized != content else None,
            metadata={
                'pii_scrubbed': sanitized != content,
                'checks_performed': ['pii', 'policy', 'citation', 'hallucination']
            }
        )
    
    def _check_pii(self, content: str) -> Dict[str, Any]:
        """Check for PII and optionally sanitize."""
        entities = []
        sanitized = content
        
        try:
            # Try to use Presidio
            if self._analyzer is None:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine
                self._analyzer = AnalyzerEngine()
                self._anonymizer = AnonymizerEngine()
            
            results = self._analyzer.analyze(
                text=content,
                language='en',
                entities=self.pii_entities or None
            )
            
            for result in results:
                entities.append({
                    'type': result.entity_type,
                    'text': content[result.start:result.end],
                    'score': result.score
                })
            
            if entities and self.pii_action in ['redact', 'mask']:
                anonymized = self._anonymizer.anonymize(
                    text=content,
                    analyzer_results=results
                )
                sanitized = anonymized.text
                
        except ImportError:
            # Fallback to regex patterns
            logger.debug("Presidio not available, using regex patterns")
            
            for pii_type, pattern in self._pii_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'type': pii_type.upper(),
                        'text': match.group(),
                        'score': 0.9
                    })
                    if self.pii_action == 'redact':
                        sanitized = sanitized.replace(match.group(), f'[{pii_type.upper()}_REDACTED]')
                    elif self.pii_action == 'mask':
                        sanitized = sanitized.replace(match.group(), '*' * len(match.group()))
        
        return {
            'found': len(entities) > 0,
            'entities': entities,
            'sanitized': sanitized
        }
    
    def _check_policies(self, content: str) -> List[Dict[str, Any]]:
        """Check for policy violations."""
        violations = []
        
        # Check for prohibited content patterns
        prohibited_patterns = [
            (r'\b(password|passwd|pwd)\s*[:=]\s*\S+', 'exposed_password'),
            (r'\b(api[_-]?key|secret[_-]?key)\s*[:=]\s*\S+', 'exposed_api_key'),
            (r'\b(guarantee|promise|ensure)\s+(100%|always|never\s+fail)', 'overpromise'),
            (r'\b(competitor|vs\.?\s+\w+brand)', 'competitor_mention'),
        ]
        
        for pattern, violation_type in prohibited_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append({
                    'type': 'policy',
                    'violation': violation_type,
                    'description': f'Content contains prohibited pattern: {violation_type}'
                })
        
        return violations
    
    def _check_citations(self, content: str, source_facts: List) -> List[str]:
        """Check if content properly cites sources."""
        warnings = []
        
        # Check for citation markers
        has_citations = bool(re.search(r'\[(?:source|ref|ticket|conv)[-:]?\d+\]', content, re.IGNORECASE))
        
        if not has_citations and len(source_facts) > 0:
            warnings.append("Article does not contain source citations")
        
        return warnings
    
    def _check_hallucinations(self, content: str, source_facts: List) -> Dict[str, Any]:
        """Check for potential hallucinations using LLM."""
        try:
            # Prepare source content
            source_text = ""
            for fact in source_facts[:10]:
                if hasattr(fact, 'content'):
                    source_text += f"- {fact.content}\n"
                elif isinstance(fact, dict):
                    source_text += f"- {fact.get('content', '')}\n"
            
            prompt = f"""Compare this KB article content with the source facts.
Identify any claims in the article that are NOT supported by the sources.

Source Facts:
{source_text}

Article Content:
{content[:1500]}

Are there any unsupported claims (hallucinations)?
Respond with JSON: {{"detected": true/false, "description": "...", "confidence": 0.0-1.0}}"""
            
            response = self.llm.generate_structured(prompt, schema={
                'detected': 'boolean',
                'description': 'string',
                'confidence': 'number'
            })
            
            return {
                'detected': response.get('detected', False),
                'description': response.get('description', ''),
                'confidence': response.get('confidence', 0.0)
            }
            
        except Exception as e:
            logger.warning(f"Hallucination check failed: {e}")
            return {'detected': False, 'description': '', 'confidence': 0.0}
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input."""
        if not isinstance(input_data, dict):
            return False
        
        content = input_data.get('content', '')
        return bool(content and len(content) > 10)
    
    def validate_output(self, output_data: Any) -> bool:
        """Validate output."""
        return isinstance(output_data, ComplianceResult)
    
    def sanitize_article(self, article: Dict[str, Any]) -> Tuple[Dict[str, Any], ComplianceResult]:
        """
        Check and sanitize an article.
        
        Returns:
            Tuple of (sanitized_article, compliance_result)
        """
        result = self.run({
            'content': article.get('content', ''),
            'article': article,
            'source_facts': article.get('source_facts', [])
        })
        
        if result and result.sanitized_content:
            article = article.copy()
            article['content'] = result.sanitized_content
        
        return article, result
