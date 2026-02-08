"""
Directory Setup
===============
Creates required directory structure for the system.
"""

import os
from pathlib import Path
from typing import Any, Dict


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Create all required directories for the system.
    
    Args:
        config: System configuration
    """
    directories = [
        # Data directories
        config['paths']['raw_excel'],
        config['paths']['processed'],
        config['paths']['backups'],
        
        # Database directory
        Path(config['database']['path']).parent,
        
        # Embedding directories
        config['paths']['faiss_index'],
        config['paths']['bm25_index'],
        
        # Version storage
        config['paths']['versions'],
        
        # Learning state
        config['paths']['learning_state'],
        
        # Logs
        config['paths']['logs'],
        
        # Metrics
        config['paths']['metrics'],
        f"{config['paths']['metrics']}/plots",
        
        # Prompts
        config['paths']['prompts'],
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def ensure_file_exists(filepath: str, default_content: str = "") -> Path:
    """
    Ensure a file exists, creating it with default content if needed.
    
    Args:
        filepath: Path to the file
        default_content: Content to write if file doesn't exist
        
    Returns:
        Path object for the file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if not path.exists():
        path.write_text(default_content)
    
    return path
