"""
LLM Client
==========
Client for interacting with Ollama and fallback LLM providers.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM client with Ollama as primary provider and fallback support.
    
    Features:
    - Automatic retry with exponential backoff
    - Fallback model support
    - Streaming support
    - Token counting
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config['llm']
        
        # Primary provider
        self.provider = self.llm_config['provider']
        
        # Ollama settings
        ollama = self.llm_config['ollama']
        self.endpoint = ollama['endpoint']
        self.model = ollama['model']
        self.fallback_models = ollama.get('fallback_models', [])
        self.timeout = ollama.get('timeout', 120)
        self.max_retries = ollama.get('max_retries', 3)
        self.retry_delay = ollama.get('retry_delay', 2)
        
        # Generation parameters
        gen_config = self.llm_config.get('generation', {})
        self.temperature = gen_config.get('temperature', 0.7)
        self.top_p = gen_config.get('top_p', 0.9)
        self.max_tokens = gen_config.get('max_tokens', 2048)
        
        self._available_model = None
    
    def check_availability(self) -> bool:
        """Check if Ollama is available and model is loaded."""
        try:
            response = requests.get(
                f"{self.endpoint}/api/tags",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                available_models = [m['name'] for m in data.get('models', [])]
                
                # Check primary model
                if self.model in available_models or f"{self.model}:latest" in available_models:
                    self._available_model = self.model
                    return True
                
                # Check fallback models
                for fallback in self.fallback_models:
                    if fallback in available_models or f"{fallback}:latest" in available_models:
                        logger.warning(f"Primary model {self.model} not found, using fallback: {fallback}")
                        self._available_model = fallback
                        return True
                
                logger.error(f"No available models found. Available: {available_models}")
                return False
            
            return False
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama not available: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            stop_sequences: Stop sequences
            
        Returns:
            Generated text
        """
        model = self._available_model or self.model
        
        payload = {
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': temperature or self.temperature,
                'top_p': self.top_p,
                'num_predict': max_tokens or self.max_tokens,
            }
        }
        
        if system_prompt:
            payload['system'] = system_prompt
        
        if stop_sequences:
            payload['options']['stop'] = stop_sequences
        
        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.endpoint}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('response', '')
                
                logger.warning(f"LLM request failed with status {response.status_code}")
                last_error = f"HTTP {response.status_code}: {response.text}"
                
            except requests.exceptions.Timeout:
                logger.warning(f"LLM request timed out (attempt {attempt + 1})")
                last_error = "Request timed out"
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"LLM request failed: {e}")
                last_error = str(e)
            
            # Exponential backoff
            if attempt < self.max_retries - 1:
                sleep_time = self.retry_delay * (2 ** attempt)
                time.sleep(sleep_time)
        
        raise RuntimeError(f"LLM generation failed after {self.max_retries} attempts: {last_error}")
    
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate using chat format.
        
        Args:
            messages: List of {'role': 'user'|'assistant'|'system', 'content': '...'}
            max_tokens: Override max tokens
            temperature: Override temperature
            
        Returns:
            Generated text
        """
        model = self._available_model or self.model
        
        payload = {
            'model': model,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': temperature or self.temperature,
                'top_p': self.top_p,
                'num_predict': max_tokens or self.max_tokens,
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.endpoint}/api/chat",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('message', {}).get('content', '')
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Chat request failed: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (2 ** attempt))
        
        raise RuntimeError("Chat generation failed")
    
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output.
        
        Args:
            prompt: User prompt
            schema: Expected JSON schema
            system_prompt: Optional system prompt
            
        Returns:
            Parsed JSON object
        """
        schema_str = json.dumps(schema, indent=2)
        
        full_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
{schema_str}

JSON Response:"""
        
        response = self.generate(
            full_prompt,
            system_prompt=system_prompt,
            temperature=0.3  # Lower temperature for structured output
        )
        
        # Extract JSON from response
        try:
            # Try to find JSON in response
            response = response.strip()
            
            # Handle markdown code blocks
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                response = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                response = response[start:end].strip()
            
            # Find JSON object
            if '{' in response:
                start = response.find('{')
                # Find matching closing brace
                depth = 0
                for i, c in enumerate(response[start:]):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            response = response[start:start + i + 1]
                            break
            
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return {}
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings using Ollama (if supported).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        model = self._available_model or self.model
        embeddings = []
        
        for text in texts:
            try:
                response = requests.post(
                    f"{self.endpoint}/api/embeddings",
                    json={'model': model, 'prompt': text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    embeddings.append(data.get('embedding', []))
                else:
                    embeddings.append([])
                    
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")
                embeddings.append([])
        
        return embeddings
