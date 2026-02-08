"""
QA Scoring Agent
================
Scores knowledge base article quality.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class QAScore:
    """Quality assessment score for an article."""
    article_id: str
    accuracy_score: float = 0.0
    completeness_score: float = 0.0
    clarity_score: float = 0.0
    compliance_score: float = 0.0
    overall_score: float = 0.0
    feedback: str = ""
    suggestions: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class QAScorerAgent(BaseAgent):
    """
    Scores KB article quality across multiple dimensions.
    
    Dimensions:
    - Accuracy: Factual correctness
    - Completeness: Coverage of topic
    - Clarity: Readability and structure
    - Compliance: Policy adherence
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, 'qa_scorer')
        
        # Dimension weights
        self.weight_accuracy = self.agent_config.get('weight_accuracy', 0.4)
        self.weight_completeness = self.agent_config.get('weight_completeness', 0.3)
        self.weight_clarity = self.agent_config.get('weight_clarity', 0.2)
        self.weight_compliance = self.agent_config.get('weight_compliance', 0.1)
        
        # Scoring dimensions
        self.dimensions = self.agent_config.get('dimensions', [
            'accuracy', 'completeness', 'clarity', 'compliance'
        ])
    
    def process(self, input_data: Dict[str, Any]) -> QAScore:
        """
        Score an article's quality.
        
        Args:
            input_data: Dict with:
                - 'article': Article object/dict to score
                - 'source_facts': Original facts used to generate
                - 'ground_truth': Optional expected answer
                
        Returns:
            QAScore object
        """
        article = input_data.get('article', {})
        source_facts = input_data.get('source_facts', [])
        ground_truth = input_data.get('ground_truth')
        
        article_id = article.get('article_id', article.get('kb_article_id', 'unknown'))
        title = article.get('title', '')
        content = article.get('content', '')
        
        # Build scoring prompt
        prompt = self._build_prompt(title, content, source_facts, ground_truth)
        
        # Call LLM
        response = self.llm.generate_structured(
            prompt,
            schema={
                'accuracy': 'number',
                'completeness': 'number',
                'clarity': 'number',
                'compliance': 'number',
                'feedback': 'string',
                'suggestions': ['string'],
                'violations': ['string']
            }
        )
        
        # Parse scores (ensure 0-1 range)
        accuracy = min(1.0, max(0.0, float(response.get('accuracy', 0.5))))
        completeness = min(1.0, max(0.0, float(response.get('completeness', 0.5))))
        clarity = min(1.0, max(0.0, float(response.get('clarity', 0.5))))
        compliance = min(1.0, max(0.0, float(response.get('compliance', 0.5))))
        
        # Calculate weighted overall score
        overall = (
            self.weight_accuracy * accuracy +
            self.weight_completeness * completeness +
            self.weight_clarity * clarity +
            self.weight_compliance * compliance
        )
        
        score = QAScore(
            article_id=article_id,
            accuracy_score=accuracy,
            completeness_score=completeness,
            clarity_score=clarity,
            compliance_score=compliance,
            overall_score=overall,
            feedback=response.get('feedback', ''),
            suggestions=response.get('suggestions', []),
            violations=response.get('violations', []),
            metadata={
                'title': title,
                'content_length': len(content),
                'facts_count': len(source_facts),
                'has_ground_truth': ground_truth is not None
            }
        )
        
        logger.debug(f"Scored article {article_id}: {overall:.2f}")
        return score
    
    def _build_prompt(
        self,
        title: str,
        content: str,
        source_facts: List,
        ground_truth: Optional[str]
    ) -> str:
        """Build scoring prompt."""
        
        # Format source facts
        facts_text = ""
        for i, fact in enumerate(source_facts[:10]):
            if hasattr(fact, 'content'):
                facts_text += f"\n{i+1}. {fact.content}"
            elif isinstance(fact, dict):
                facts_text += f"\n{i+1}. {fact.get('content', '')}"
        
        ground_truth_text = ""
        if ground_truth:
            ground_truth_text = f"\n\nExpected Answer:\n{ground_truth}"
        
        if self.prompt_template:
            return self.prompt_template.format(
                title=title,
                content=content,
                facts=facts_text,
                ground_truth=ground_truth_text
            )
        
        return f"""Score this KB article on quality dimensions (0.0 to 1.0 scale).

Article Title: {title}

Article Content:
{content[:2000]}

Source Facts Used:{facts_text}
{ground_truth_text}

Score each dimension:
1. accuracy: Is the information factually correct and supported by sources? (0.0-1.0)
2. completeness: Does it cover all necessary aspects of the topic? (0.0-1.0)
3. clarity: Is it well-written, organized, and easy to follow? (0.0-1.0)
4. compliance: Does it follow professional standards and policies? (0.0-1.0)

Also provide:
- feedback: Brief overall assessment
- suggestions: List of improvement suggestions
- violations: List of any policy/quality violations found"""
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input."""
        if not isinstance(input_data, dict):
            return False
        
        article = input_data.get('article', {})
        if not article:
            return False
        
        content = article.get('content', '')
        return bool(content and len(content) > 50)
    
    def validate_output(self, output_data: Any) -> bool:
        """Validate output."""
        if not isinstance(output_data, QAScore):
            return False
        
        # Scores should be in valid range
        for attr in ['accuracy_score', 'completeness_score', 'clarity_score', 
                     'compliance_score', 'overall_score']:
            score = getattr(output_data, attr)
            if not (0.0 <= score <= 1.0):
                return False
        
        return True
    
    def batch_score(self, articles: List[Dict[str, Any]]) -> List[QAScore]:
        """
        Score multiple articles.
        
        Args:
            articles: List of article dicts
            
        Returns:
            List of QAScore objects
        """
        scores = []
        
        for article in articles:
            score = self.run({
                'article': article,
                'source_facts': article.get('source_facts', [])
            })
            
            if score:
                scores.append(score)
        
        return scores
    
    def get_aggregate_stats(self, scores: List[QAScore]) -> Dict[str, Any]:
        """Calculate aggregate statistics from scores."""
        if not scores:
            return {}
        
        def avg(values):
            return sum(values) / len(values) if values else 0
        
        return {
            'count': len(scores),
            'mean_accuracy': avg([s.accuracy_score for s in scores]),
            'mean_completeness': avg([s.completeness_score for s in scores]),
            'mean_clarity': avg([s.clarity_score for s in scores]),
            'mean_compliance': avg([s.compliance_score for s in scores]),
            'mean_overall': avg([s.overall_score for s in scores]),
            'min_overall': min(s.overall_score for s in scores),
            'max_overall': max(s.overall_score for s in scores),
            'total_violations': sum(len(s.violations) for s in scores)
        }
