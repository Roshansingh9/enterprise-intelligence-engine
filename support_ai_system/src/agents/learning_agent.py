"""
Learning Agent
==============
Optimizes system performance through continuous learning.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .base_agent import BaseAgent

logger = logging.getLogger('learning')


@dataclass
class LearningEvent:
    """A learning event record."""
    event_id: str
    event_type: str  # prompt_update, example_added, threshold_change, index_refresh
    details: Dict[str, Any] = field(default_factory=dict)
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""


class LearningAgent(BaseAgent):
    """
    Manages continuous learning and optimization.
    
    Features:
    - Prompt optimization
    - Example memory management
    - Threshold adaptation
    - Index refresh triggers
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, 'learning')
        
        self.prompt_optimization = self.agent_config.get('prompt_optimization_enabled', True)
        self.example_memory_size = self.agent_config.get('example_memory_size', 50)
        
        # Learning state paths
        learning_config = config['learning']
        self.state_dir = Path(config['paths']['learning_state'])
        self.optimizer_state_file = self.state_dir / 'optimizer_state.json'
        
        # Learning history
        self.events: List[LearningEvent] = []
        self.prompt_versions: Dict[str, int] = {}
        self.example_memory: Dict[str, List[Dict]] = {}  # agent_name -> examples
        
        # Adaptive thresholds
        self.thresholds = {
            'confidence': config['governance'].get('auto_approve_threshold', 0.95),
            'quality': config['evaluation'].get('qa_sample_size', 100),
            'retrieval': config['retrieval'].get('min_confidence', 0.5)
        }
        
        # Load existing state
        self._load_state()
    
    def process(self, input_data: Dict[str, Any]) -> LearningEvent:
        """
        Process a learning trigger.
        
        Args:
            input_data: Dict with:
                - 'trigger': Type of learning trigger
                - 'metrics': Current metrics
                - 'samples': Optional sample data
                
        Returns:
            LearningEvent with results
        """
        trigger = input_data.get('trigger', 'evaluation')
        metrics = input_data.get('metrics', {})
        samples = input_data.get('samples', [])
        
        event = LearningEvent(
            event_id=f"LEARN-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            event_type=trigger,
            metrics_before=metrics.copy(),
            timestamp=datetime.now().isoformat()
        )
        
        # Execute learning based on trigger type
        if trigger == 'prompt_optimization' and self.prompt_optimization:
            self._optimize_prompts(metrics, samples)
            event.details['prompts_updated'] = True
            
        elif trigger == 'example_update':
            count = self._update_examples(samples)
            event.details['examples_added'] = count
            
        elif trigger == 'threshold_adaptation':
            changes = self._adapt_thresholds(metrics)
            event.details['threshold_changes'] = changes
            
        elif trigger == 'index_refresh':
            event.details['index_refreshed'] = True
            
        elif trigger == 'evaluation':
            # Full learning cycle
            if self.prompt_optimization:
                self._optimize_prompts(metrics, samples)
            self._update_examples(samples)
            self._adapt_thresholds(metrics)
        
        # Record event
        self.events.append(event)
        self._save_state()
        
        logger.info(f"Learning event: {event.event_type} - {event.event_id}")
        return event
    
    def _optimize_prompts(self, metrics: Dict[str, float], samples: List) -> None:
        """Optimize prompts based on performance metrics."""
        
        # Identify underperforming areas
        low_accuracy = metrics.get('accuracy', 1.0) < 0.7
        low_completeness = metrics.get('completeness', 1.0) < 0.7
        
        if not (low_accuracy or low_completeness):
            return
        
        # Get worst performing samples
        poor_samples = [s for s in samples if s.get('score', 1.0) < 0.6][:5]
        
        if not poor_samples:
            return
        
        # Generate prompt improvements
        prompt = f"""Analyze these poorly performing KB article generations and suggest prompt improvements.

Poor Samples:
{json.dumps(poor_samples[:3], indent=2, default=str)}

Current Metrics:
- Accuracy: {metrics.get('accuracy', 'N/A')}
- Completeness: {metrics.get('completeness', 'N/A')}

Suggest specific prompt modifications to improve:
1. If accuracy is low: How to make outputs more factually grounded
2. If completeness is low: How to ensure all aspects are covered

Return JSON with prompt_changes list."""
        
        try:
            response = self.llm.generate_structured(prompt, schema={
                'prompt_changes': [{
                    'agent': 'string',
                    'change': 'string'
                }]
            })
            
            # Log suggestions (actual prompt updates would be manual)
            for change in response.get('prompt_changes', []):
                logger.info(f"Prompt suggestion for {change.get('agent')}: {change.get('change')}")
                
        except Exception as e:
            logger.warning(f"Prompt optimization failed: {e}")
    
    def _update_examples(self, samples: List) -> int:
        """Add high-quality samples to example memory."""
        added = 0
        
        for sample in samples:
            score = sample.get('score', 0)
            agent = sample.get('agent', 'kb_generator')
            
            if score >= 0.85:
                if agent not in self.example_memory:
                    self.example_memory[agent] = []
                
                # Add to memory
                self.example_memory[agent].append({
                    'input': sample.get('input'),
                    'output': sample.get('output'),
                    'score': score
                })
                added += 1
                
                # Trim to max size
                self.example_memory[agent] = sorted(
                    self.example_memory[agent],
                    key=lambda x: x['score'],
                    reverse=True
                )[:self.example_memory_size]
        
        if added:
            logger.info(f"Added {added} examples to memory")
        
        return added
    
    def _adapt_thresholds(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Adapt thresholds based on performance."""
        changes = {}
        
        # Confidence threshold
        approval_rate = metrics.get('auto_approval_rate', 0.5)
        if approval_rate > 0.9:
            # Too many auto-approvals, raise threshold
            new_threshold = min(0.99, self.thresholds['confidence'] + 0.02)
            if new_threshold != self.thresholds['confidence']:
                changes['confidence'] = {
                    'old': self.thresholds['confidence'],
                    'new': new_threshold
                }
                self.thresholds['confidence'] = new_threshold
        elif approval_rate < 0.3:
            # Too few auto-approvals, lower threshold
            new_threshold = max(0.7, self.thresholds['confidence'] - 0.02)
            if new_threshold != self.thresholds['confidence']:
                changes['confidence'] = {
                    'old': self.thresholds['confidence'],
                    'new': new_threshold
                }
                self.thresholds['confidence'] = new_threshold
        
        # Retrieval threshold
        hit_rate = metrics.get('hit_at_3', 0.5)
        if hit_rate < 0.5:
            # Poor retrieval, lower minimum confidence
            new_threshold = max(0.3, self.thresholds['retrieval'] - 0.05)
            if new_threshold != self.thresholds['retrieval']:
                changes['retrieval'] = {
                    'old': self.thresholds['retrieval'],
                    'new': new_threshold
                }
                self.thresholds['retrieval'] = new_threshold
        
        if changes:
            logger.info(f"Threshold adaptations: {changes}")
        
        return changes
    
    def _load_state(self) -> None:
        """Load learning state from disk."""
        if self.optimizer_state_file.exists():
            try:
                with open(self.optimizer_state_file, 'r') as f:
                    state = json.load(f)
                
                self.thresholds = state.get('thresholds', self.thresholds)
                self.prompt_versions = state.get('prompt_versions', {})
                self.example_memory = state.get('example_memory', {})
                
                logger.info("Learning state loaded")
            except Exception as e:
                logger.warning(f"Failed to load learning state: {e}")
    
    def _save_state(self) -> None:
        """Save learning state to disk."""
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            
            state = {
                'thresholds': self.thresholds,
                'prompt_versions': self.prompt_versions,
                'example_memory': self.example_memory,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.optimizer_state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to save learning state: {e}")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input."""
        if not isinstance(input_data, dict):
            return False
        
        trigger = input_data.get('trigger')
        return trigger in ['prompt_optimization', 'example_update', 
                          'threshold_adaptation', 'index_refresh', 'evaluation']
    
    def validate_output(self, output_data: Any) -> bool:
        """Validate output."""
        return isinstance(output_data, LearningEvent)
    
    def get_examples(self, agent_name: str) -> List[Dict]:
        """Get examples for an agent."""
        return self.example_memory.get(agent_name, [])
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current adaptive thresholds."""
        return self.thresholds.copy()
