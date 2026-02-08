"""
LLM Client
==========
Client for Groq API (primary) with Ollama fallback.
Supports multiple API keys with automatic rotation.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM client with Groq as primary and Ollama as fallback.

    Features:
    - Multiple API key rotation
    - Per-key cooldown when rate limited
    - Optional client-side throttling (min interval between requests)
    - Automatic fallback to Ollama
    - Robust JSON parsing
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config['llm']

        # Primary provider
        self.provider = self.llm_config['provider']

        # Groq settings
        groq = self.llm_config.get('groq', {})
        self.groq_endpoint = groq.get('endpoint', 'https://api.groq.com/openai/v1/chat/completions')
        self.groq_model = groq.get('model', 'llama-3.1-8b-instant')
        self.groq_api_keys = groq.get('api_keys', [])
        self.groq_timeout = groq.get('timeout', 30)
        self.current_key_index = 0

        # Rate limiting / throttling
        # Some accounts have strict RPM/TPM limits. These settings reduce 429 storms.
        self.groq_min_interval_seconds = float(groq.get('min_interval_seconds', 0.0))
        self.groq_default_cooldown_seconds = float(groq.get('default_cooldown_seconds', 10.0))
        self._last_groq_request_ts: float = 0.0
        self._key_cooldown_until: Dict[int, float] = {}

        # Ollama settings (fallback)
        ollama = self.llm_config.get('ollama', {})
        self.ollama_endpoint = ollama.get('endpoint', 'http://localhost:11434')
        self.ollama_model = ollama.get('model', 'tinyllama')
        self.ollama_fallback_models = ollama.get('fallback_models', [])
        self.ollama_timeout = ollama.get('timeout', 120)
        self.num_ctx = ollama.get('num_ctx', 1024)

        # Common settings
        self.max_retries = self.llm_config.get('groq', {}).get('max_retries', 2)
        self.retry_delay = self.llm_config.get('groq', {}).get('retry_delay', 1)

        # Generation parameters
        gen_config = self.llm_config.get('generation', {})
        self.temperature = gen_config.get('temperature', 0.3)
        self.top_p = gen_config.get('top_p', 0.9)
        self.max_tokens = gen_config.get('max_tokens', 512)

        self._ollama_available_model = None

    def _is_valid_key(self, key: str) -> bool:
        """Check if API key looks valid (not placeholder)."""
        return key and not key.startswith('YOUR_') and len(key) > 20

    def _valid_key_indices(self) -> List[int]:
        return [i for i, k in enumerate(self.groq_api_keys) if self._is_valid_key(k)]

    def _select_next_key_index(self) -> Optional[int]:
        """Round-robin select a key index that is not in cooldown."""
        valid = self._valid_key_indices()
        if not valid:
            return None

        now = time.time()
        for _ in range(len(valid)):
            idx = self.current_key_index % len(self.groq_api_keys)
            self.current_key_index = (self.current_key_index + 1) % max(1, len(self.groq_api_keys))

            if idx not in valid:
                continue

            cooldown_until = self._key_cooldown_until.get(idx, 0.0)
            if cooldown_until <= now:
                return idx

        return None

    def _throttle(self) -> None:
        if self.groq_min_interval_seconds <= 0:
            return
        now = time.time()
        delta = now - self._last_groq_request_ts
        if delta < self.groq_min_interval_seconds:
            time.sleep(self.groq_min_interval_seconds - delta)

    def _compute_retry_after_seconds(self, response: requests.Response) -> float:
        """Best-effort parse of Retry-After header (seconds)."""
        ra = response.headers.get('retry-after') or response.headers.get('Retry-After')
        if not ra:
            return self.groq_default_cooldown_seconds
        try:
            return max(0.0, float(ra))
        except Exception:
            return self.groq_default_cooldown_seconds

    def _mark_rate_limited(self, key_index: int, response: requests.Response) -> float:
        cooldown = self._compute_retry_after_seconds(response)
        self._key_cooldown_until[key_index] = time.time() + cooldown
        return cooldown

    def check_availability(self) -> bool:
        """Check if Groq or Ollama is available."""
        if self.provider == 'groq':
            valid_keys = [k for k in self.groq_api_keys if self._is_valid_key(k)]
            if valid_keys:
                logger.info(f"Groq configured with {len(valid_keys)} API keys")
                return True
            logger.warning("No valid Groq API keys found, falling back to Ollama")

        # Check Ollama
        try:
            response = requests.get(f"{self.ollama_endpoint}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                available_models = [m['name'] for m in data.get('models', [])]

                if self.ollama_model in available_models or f"{self.ollama_model}:latest" in available_models:
                    self._ollama_available_model = self.ollama_model
                    logger.info(f"Ollama available with model: {self.ollama_model}")
                    return True

                for fallback in self.ollama_fallback_models:
                    if fallback in available_models or f"{fallback}:latest" in available_models:
                        logger.warning(f"Using fallback model: {fallback}")
                        self._ollama_available_model = fallback
                        return True

                logger.error(f"No models available. Found: {available_models}")
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
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate text using Groq (primary) or Ollama (fallback)."""

        if self.provider == 'groq':
            result = self._generate_groq(prompt, system_prompt, max_tokens, temperature)
            if result is not None:
                return result
            logger.warning("Groq failed, falling back to Ollama")

        return self._generate_ollama(prompt, system_prompt, max_tokens, temperature, stop_sequences)

    def _generate_groq(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        """Generate using Groq API with key rotation + cooldown/backoff."""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        valid_key_indices = self._valid_key_indices()
        if not valid_key_indices:
            return None

        last_status: Optional[int] = None

        # Try up to N attempts across keys (cooldowns may cause immediate no-key availability)
        max_attempts = max(1, len(valid_key_indices) * 2)
        for attempt in range(max_attempts):
            key_index = self._select_next_key_index()
            if key_index is None:
                # All keys currently cooling down
                next_ready = min(self._key_cooldown_until.values(), default=time.time())
                sleep_for = max(0.1, next_ready - time.time())
                logger.warning(f"All Groq keys cooling down; sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)
                continue

            api_key = self.groq_api_keys[key_index]

            try:
                self._throttle()
                response = requests.post(
                    self.groq_endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.groq_model,
                        "messages": messages,
                        "temperature": temperature or self.temperature,
                        "max_tokens": max_tokens or self.max_tokens,
                        "top_p": self.top_p,
                    },
                    timeout=self.groq_timeout,
                )
                self._last_groq_request_ts = time.time()

                if response.status_code == 200:
                    data = response.json()
                    return data['choices'][0]['message']['content']

                last_status = response.status_code

                if response.status_code == 429:
                    cooldown = self._mark_rate_limited(key_index, response)
                    logger.warning(f"Rate limited on key {key_index + 1}; cooling down {cooldown:.1f}s")
                    # Short sleep to avoid tight loop; cooldown is per-key and enforced by selector
                    time.sleep(min(1.0, cooldown))
                    continue

                logger.warning(f"Groq error {response.status_code}: {response.text[:200]}")

            except requests.exceptions.Timeout:
                logger.warning("Groq request timed out")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Groq request failed: {e}")

            # Small backoff on non-429 failures to avoid hammering
            time.sleep(min(2.0, self.retry_delay * (2 ** attempt)))

        if last_status == 429:
            logger.warning("Groq appears rate limited across keys; consider increasing min_interval_seconds")
        return None
    
    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """Generate using Ollama (fallback)."""
        
        model = self._ollama_available_model or self.ollama_model
        
        payload = {
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': temperature or self.temperature,
                'top_p': self.top_p,
                'num_predict': max_tokens or self.max_tokens,
                'num_ctx': self.num_ctx,
            }
        }
        
        if system_prompt:
            payload['system'] = system_prompt
        
        if stop_sequences:
            payload['options']['stop'] = stop_sequences
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_endpoint}/api/generate",
                    json=payload,
                    timeout=self.ollama_timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('response', '')
                
                last_error = f"HTTP {response.status_code}"
                
            except requests.exceptions.Timeout:
                last_error = "Timeout"
            except requests.exceptions.RequestException as e:
                last_error = str(e)
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (2 ** attempt))
        
        raise RuntimeError(f"LLM generation failed: {last_error}")
    
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate using chat format."""

        if self.provider == 'groq':
            result = self._chat_groq(messages, max_tokens, temperature)
            if result is not None:
                return result

        return self._chat_ollama(messages, max_tokens, temperature)

    def _chat_groq(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        """Chat using Groq API with cooldown/backoff."""

        valid_key_indices = self._valid_key_indices()
        if not valid_key_indices:
            return None

        max_attempts = max(1, len(valid_key_indices) * 2)
        for attempt in range(max_attempts):
            key_index = self._select_next_key_index()
            if key_index is None:
                next_ready = min(self._key_cooldown_until.values(), default=time.time())
                sleep_for = max(0.1, next_ready - time.time())
                logger.warning(f"All Groq keys cooling down; sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)
                continue

            api_key = self.groq_api_keys[key_index]

            try:
                self._throttle()
                response = requests.post(
                    self.groq_endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.groq_model,
                        "messages": messages,
                        "temperature": temperature or self.temperature,
                        "max_tokens": max_tokens or self.max_tokens,
                    },
                    timeout=self.groq_timeout,
                )
                self._last_groq_request_ts = time.time()

                if response.status_code == 200:
                    data = response.json()
                    return data['choices'][0]['message']['content']

                if response.status_code == 429:
                    cooldown = self._mark_rate_limited(key_index, response)
                    logger.warning(f"Rate limited on key {key_index + 1}; cooling down {cooldown:.1f}s")
                    time.sleep(min(1.0, cooldown))
                    continue

                logger.warning(f"Groq chat error {response.status_code}: {response.text[:200]}")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Groq chat failed: {e}")

            time.sleep(min(2.0, self.retry_delay * (2 ** attempt)))

        return None
    
    def _chat_ollama(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Chat using Ollama."""
        
        model = self._ollama_available_model or self.ollama_model
        
        payload = {
            'model': model,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': temperature or self.temperature,
                'top_p': self.top_p,
                'num_predict': max_tokens or self.max_tokens,
                'num_ctx': self.num_ctx,
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_endpoint}/api/chat",
                    json=payload,
                    timeout=self.ollama_timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('message', {}).get('content', '')
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Ollama chat failed: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (2 ** attempt))
        
        raise RuntimeError("Chat generation failed")
    
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """Generate structured JSON output with robust parsing."""
        
        full_prompt = f"""{prompt}

Return ONLY valid JSON. No explanations."""
        
        for attempt in range(max_retries):
            response = self.generate(
                full_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=512
            )
            
            parsed = self._extract_json(response)
            if parsed:
                return parsed
            
            logger.warning(f"JSON parse attempt {attempt + 1} failed, retrying...")
        
        logger.error("Failed to parse JSON after retries")
        return {}
    
    def _extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLM response."""
        try:
            response = response.strip()
            
            # Handle markdown code blocks
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end > start:
                    response = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                if end > start:
                    response = response[start:end].strip()
            
            # Find JSON object
            if '{' in response:
                start = response.find('{')
                depth = 0
                end_idx = len(response)
                for i, c in enumerate(response[start:]):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            end_idx = start + i + 1
                            break
                response = response[start:end_idx]
            
            # Fix common JSON issues
            response = self._fix_json(response)
            
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error: {e}")
            return None
    
    def _fix_json(self, text: str) -> str:
        """Fix common JSON formatting issues from LLMs."""
        import re
        
        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)
        
        # Fix unescaped newlines in strings
        lines = text.split('\n')
        fixed_lines = []
        in_string = False
        for line in lines:
            quote_count = line.count('"') - line.count('\\"')
            if in_string:
                fixed_lines[-1] += ' ' + line.strip()
            else:
                fixed_lines.append(line)
            in_string = (quote_count % 2 == 1) != in_string
        
        return '\n'.join(fixed_lines)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Ollama."""
        model = self._ollama_available_model or self.ollama_model
        embeddings = []
        
        for text in texts:
            try:
                response = requests.post(
                    f"{self.ollama_endpoint}/api/embeddings",
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
