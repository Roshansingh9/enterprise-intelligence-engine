"""
KB Generator Agent
==================
Generates structured knowledge base articles.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .base_agent import BaseAgent
from .extractor_agent import ExtractedFact

logger = logging.getLogger(__name__)


@dataclass
class GeneratedArticle:
    """A generated KB article."""
    article_id: str
    title: str
    summary: str
    content: str
    product: str = ""
    category: str = ""
    tags: List[str] = field(default_factory=list)
    source_facts: List[ExtractedFact] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    version: int = 1
    status: str = "draft"
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class KBGeneratorAgent(BaseAgent):
    """
    Generates structured KB articles from extracted facts.
    
    Features:
    - Template-based generation
    - Example-guided few-shot learning
    - Automatic tagging
    - Citation generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, 'kb_generator')
        self.template_version = self.agent_config.get('template_version', 'v1')
        self.include_examples = self.agent_config.get('include_examples', True)
        self.max_examples = self.agent_config.get('max_examples', 3)
        
        # Example memory for few-shot learning
        self.example_articles: List[GeneratedArticle] = []
    
    def process(self, input_data: Dict[str, Any]) -> GeneratedArticle:
        """
        Generate a KB article.
        
        Args:
            input_data: Dict with:
                - 'facts': List of ExtractedFact objects
                - 'topic': Main topic
                - 'product': Product name
                - 'category': Category
                - 'source_ids': Source document IDs
                
        Returns:
            GeneratedArticle object
        """
        facts = input_data.get('facts', [])
        topic = input_data.get('topic', '')
        product = input_data.get('product', '')
        category = input_data.get('category', '')
        source_ids = input_data.get('source_ids', [])
        
        # Build generation prompt
        prompt = self._build_prompt(facts, topic, product, category)
        
        # Call LLM
        response = self.llm.generate_structured(
            prompt,
            schema={
                'title': 'string',
                'summary': 'string',
                'content': 'string',
                'tags': ['string'],
                'confidence': 'number'
            }
        )
        
        # Create article
        article = GeneratedArticle(
            article_id=f"KB-{uuid.uuid4().hex[:8].upper()}",
            title=response.get('title', f"KB: {topic}"),
            summary=response.get('summary', ''),
            content=response.get('content', ''),
            product=product,
            category=category,
            tags=response.get('tags', []),
            source_facts=facts if isinstance(facts, list) else [],
            source_ids=source_ids,
            confidence=float(response.get('confidence', 0.5)),
            version=1,
            status='draft',
            created_at=datetime.now().isoformat(),
            metadata={
                'template_version': self.template_version,
                'facts_count': len(facts)
            }
        )
        
        logger.info(f"Generated article: {article.article_id} - {article.title}")
        return article
    
    def _build_prompt(
        self,
        facts: List[ExtractedFact],
        topic: str,
        product: str,
        category: str
    ) -> str:
        """Build article generation prompt."""
        
        # Format facts
        facts_text = ""
        for i, fact in enumerate(facts):
            if isinstance(fact, ExtractedFact):
                facts_text += f"\n{i+1}. [{fact.fact_type}] {fact.content}"
            elif isinstance(fact, dict):
                facts_text += f"\n{i+1}. [{fact.get('type', 'unknown')}] {fact.get('content', '')}"
        
        # Get examples if enabled
        examples_text = ""
        if self.include_examples and self.example_articles:
            examples_text = "\n\n--- EXAMPLE ARTICLES ---\n"
            for ex in self.example_articles[:self.max_examples]:
                examples_text += f"\nTitle: {ex.title}\nContent: {ex.content[:300]}...\n"
        
        if self.prompt_template:
            return self.prompt_template.format(
                topic=topic,
                product=product,
                category=category,
                facts=facts_text,
                examples=examples_text
            )
        
        # Simplified prompt for faster generation
        return f"""Create KB article from facts. Return JSON only.

Topic: {topic}
Product: {product}
Facts:{facts_text[:800]}

{{"title":"short title","summary":"1 sentence","content":"Problem: X\\nSolution: Y\\nSteps: 1,2,3","tags":["tag1","tag2"],"confidence":0.8}}"""
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input."""
        if not isinstance(input_data, dict):
            return False
        
        facts = input_data.get('facts', [])
        topic = input_data.get('topic', '')
        
        if not facts or not topic:
            return False
        
        return True
    
    def validate_output(self, output_data: Any) -> bool:
        """Validate output."""
        if not isinstance(output_data, GeneratedArticle):
            return False
        
        if not output_data.title or not output_data.content:
            return False
        
        if len(output_data.content) < 100:
            return False
        
        return True
    
    def add_example(self, article: GeneratedArticle) -> None:
        """Add a high-quality article as an example for few-shot learning."""
        if article.confidence >= 0.8:
            self.example_articles.append(article)
            # Keep only top examples
            self.example_articles = sorted(
                self.example_articles,
                key=lambda a: a.confidence,
                reverse=True
            )[:self.max_examples * 2]
    
    def generate_from_conversation(
        self,
        transcript: str,
        facts: List[ExtractedFact],
        metadata: Dict[str, Any]
    ) -> Optional[GeneratedArticle]:
        """
        Convenience method to generate article from conversation.
        
        Args:
            transcript: Conversation transcript
            facts: Pre-extracted facts
            metadata: Conversation metadata
            
        Returns:
            Generated article or None
        """
        # Determine topic from facts or metadata
        topic = metadata.get('issue_summary', '')
        if not topic and facts:
            # Use first problem fact as topic
            for fact in facts:
                if fact.fact_type == 'problem':
                    topic = fact.content[:100]
                    break
        
        if not topic:
            topic = "Support Resolution"
        
        return self.run({
            'facts': facts,
            'topic': topic,
            'product': metadata.get('product', ''),
            'category': metadata.get('category', ''),
            'source_ids': [metadata.get('ticket_number', ''), metadata.get('conversation_id', '')]
        })
