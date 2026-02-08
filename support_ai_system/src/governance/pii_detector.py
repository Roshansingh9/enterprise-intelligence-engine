"""
PII Detector
============
Detects and redacts personally identifiable information.
"""

import re
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class PIIDetector:
    """
    Detects PII patterns in text and optionally redacts them.
    
    Patterns: emails, phones, SSN, credit cards, names (basic).
    """
    
    PATTERNS = {
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'PHONE': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        'SSN': r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
        'CREDIT_CARD': r'\b(?:\d{4}[-.\s]?){3}\d{4}\b',
        'IP_ADDRESS': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        gov_config = config.get('governance', {})
        self.enabled = gov_config.get('pii_detection_enabled', True)
        self.action = gov_config.get('pii_action', 'redact')
        self.entities = gov_config.get('pii_entities', list(self.PATTERNS.keys()))
    
    def detect(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII in text.
        
        Returns:
            List of {type, value, start, end} dicts
        """
        if not self.enabled or not text:
            return []
        
        findings = []
        for pii_type in self.entities:
            pattern = self.PATTERNS.get(pii_type)
            if not pattern:
                continue
            
            for match in re.finditer(pattern, text, re.IGNORECASE):
                findings.append({
                    'type': pii_type,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        if findings:
            logger.warning(f"Found {len(findings)} PII instances")
        
        return findings
    
    def redact(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Redact PII from text.
        
        Returns:
            (redacted_text, findings)
        """
        findings = self.detect(text)
        if not findings:
            return text, []
        
        # Sort by position descending to preserve indices
        findings.sort(key=lambda x: x['start'], reverse=True)
        
        redacted = text
        for f in findings:
            replacement = f"[{f['type']}]"
            if self.action == 'mask':
                replacement = '*' * len(f['value'])
            redacted = redacted[:f['start']] + replacement + redacted[f['end']:]
        
        return redacted, findings
    
    def is_safe(self, text: str) -> bool:
        """Check if text contains no PII."""
        return len(self.detect(text)) == 0
