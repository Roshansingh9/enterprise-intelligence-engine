"""
Base Agent
==========
Base class for all AI agents.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for AI agents.
    
    All agents must implement:
    - process(): Main processing method
    - validate_input(): Input validation
    - validate_output(): Output validation
    """
    
    def __init__(self, config: Dict[str, Any], agent_name: str):
        self.config = config
        self.agent_name = agent_name
        self.agent_config = config['agents'].get(agent_name, {})
        
        # LLM client (lazy loaded)
        self._llm_client = None
        
        # Load prompt template
        self.prompt_template = self._load_prompt()
        
        # Statistics
        self.calls = 0
        self.successes = 0
        self.failures = 0
    
    @property
    def llm(self) -> LLMClient:
        """Get LLM client."""
        if self._llm_client is None:
            self._llm_client = LLMClient(self.config)
        return self._llm_client
    
    def _load_prompt(self) -> str:
        """Load prompt template from file."""
        prompt_file = self.agent_config.get('prompt_file')
        
        if not prompt_file:
            return ""
        
        prompt_path = Path(prompt_file)
        
        if not prompt_path.exists():
            logger.warning(f"Prompt file not found: {prompt_file}")
            return ""
        
        return prompt_path.read_text()
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Main processing method.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Processed output
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        pass
    
    @abstractmethod
    def validate_output(self, output_data: Any) -> bool:
        """Validate output data."""
        pass
    
    def run(self, input_data: Any) -> Optional[Any]:
        """
        Execute agent with validation and error handling.
        
        Args:
            input_data: Input data
            
        Returns:
            Output or None if failed
        """
        self.calls += 1
        
        try:
            # Validate input
            if not self.validate_input(input_data):
                logger.error(f"{self.agent_name}: Invalid input")
                self.failures += 1
                return None
            
            # Process
            output = self.process(input_data)
            
            # Validate output
            if not self.validate_output(output):
                logger.error(f"{self.agent_name}: Invalid output")
                self.failures += 1
                return None
            
            self.successes += 1
            return output
            
        except Exception as e:
            logger.error(f"{self.agent_name} failed: {e}")
            self.failures += 1
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        success_rate = self.successes / self.calls if self.calls > 0 else 0
        
        return {
            'agent': self.agent_name,
            'calls': self.calls,
            'successes': self.successes,
            'failures': self.failures,
            'success_rate': success_rate
        }
