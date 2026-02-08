"""
Configuration Loader
====================
Loads and validates system configuration from YAML.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply environment variable overrides
    config = _apply_env_overrides(config)
    
    # Validate configuration
    _validate_config(config)
    
    return config


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration."""
    
    # LLM endpoint override
    if os.getenv('OLLAMA_ENDPOINT'):
        config['llm']['ollama']['endpoint'] = os.getenv('OLLAMA_ENDPOINT')
    
    # Model override
    if os.getenv('LLM_MODEL'):
        config['llm']['ollama']['model'] = os.getenv('LLM_MODEL')
    
    # Database path override
    if os.getenv('DATABASE_PATH'):
        config['database']['path'] = os.getenv('DATABASE_PATH')
    
    # Debug mode override
    if os.getenv('DEBUG'):
        config['system']['debug'] = os.getenv('DEBUG').lower() == 'true'
    
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure and values."""
    
    required_sections = ['system', 'llm', 'database', 'paths', 'learning', 'evaluation']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate LLM configuration
    if config['llm']['provider'] not in ['ollama', 'openai']:
        raise ValueError(f"Invalid LLM provider: {config['llm']['provider']}")
    
    # Validate learning configuration
    if config['learning']['train_ratio'] + config['learning']['validate_ratio'] != 1.0:
        raise ValueError("train_ratio + validate_ratio must equal 1.0")
    
    # Validate retrieval weights
    weights = config['retrieval']['weights']
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        raise ValueError(f"Retrieval weights must sum to 1.0, got {total_weight}")


def save_config(config: Dict[str, Any], config_path: str = "config.yaml") -> None:
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
