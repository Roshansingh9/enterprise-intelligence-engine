"""
Extractor Agent
===============
Extracts structured facts from raw text data.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFact:
    """A single extracted fact."""
    fact_type: str  # problem, solution, procedure, requirement, etc.
    content: str
    confidence: float = 0.0
    source_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExtractorAgent(BaseAgent):
    """
    Extracts structured facts from conversations, tickets, and scripts.
    
    Fact types:
    - problem: Customer issue description
    - solution: Resolution or fix
    - procedure: Step-by-step instructions
    - requirement: Prerequisites or conditions
    - error: Error messages or codes
    - product: Product/feature mentions
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, 'extractor')
        self.max_facts = self.agent_config.get('max_facts_per_ticket', 10)
    
    def process(self, input_data: Dict[str, Any]) -> List[ExtractedFact]:
        """
        Extract facts from input data.
        
        Args:
            input_data: Dict with 'text', 'type' (conversation/ticket/script), 'metadata'
            
        Returns:
            List of ExtractedFact objects
        """
        text = input_data.get('text', '')
        data_type = input_data.get('type', 'conversation')
        metadata = input_data.get('metadata', {})
        
        # Build extraction prompt
        prompt = self._build_prompt(text, data_type)
        
        # Call LLM
        response = self.llm.generate_structured(
            prompt,
            schema={
                'facts': [
                    {
                        'type': 'string',
                        'content': 'string',
                        'confidence': 'number'
                    }
                ]
            }
        )
        
        # Parse response
        facts = []
        for fact_data in response.get('facts', [])[:self.max_facts]:
            fact = ExtractedFact(
                fact_type=fact_data.get('type', 'unknown'),
                content=fact_data.get('content', ''),
                confidence=float(fact_data.get('confidence', 0.5)),
                source_text=text[:500],
                metadata=metadata
            )
            facts.append(fact)
        
        logger.debug(f"Extracted {len(facts)} facts from {data_type}")
        return facts
    
    def _build_prompt(self, text: str, data_type: str) -> str:
        """Build extraction prompt - optimized for speed."""
        
        # Only use template if it actually includes the needed placeholders.
        # (Default init creates a stub file like "# Extractor Prompt Template" which would break extraction.)
        if self.prompt_template and ('{text}' in self.prompt_template):
            return self.prompt_template.format(text=text, type=data_type)
        
        # Truncate text to reduce processing time
        truncated = text[:1500] if len(text) > 1500 else text
        
        return f"""Extract facts from this {data_type}. Return JSON only.

Text: {truncated}

{{"facts":[{{"type":"problem|solution|error","content":"brief fact","confidence":0.8}}]}}"""
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input has required fields."""
        if not isinstance(input_data, dict):
            return False
        
        text = input_data.get('text', '')
        if not text or len(text) < 50:
            return False
        
        return True
    
    def validate_output(self, output_data: Any) -> bool:
        """Validate output is a list of facts."""
        if not isinstance(output_data, list):
            return False
        
        for fact in output_data:
            if not isinstance(fact, ExtractedFact):
                return False
            if not fact.content:
                return False
        
        return True
    
    def extract_from_conversation(self, transcript: str, metadata: Dict[str, Any] = None) -> List[ExtractedFact]:
        """Convenience method for conversation extraction."""
        return self.run({
            'text': transcript,
            'type': 'conversation',
            'metadata': metadata or {}
        })
    
    def extract_from_ticket(self, resolution: str, metadata: Dict[str, Any] = None) -> List[ExtractedFact]:
        """Convenience method for ticket extraction."""
        return self.run({
            'text': resolution,
            'type': 'ticket',
            'metadata': metadata or {}
        })
