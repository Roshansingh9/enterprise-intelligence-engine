"""
Evaluation Metrics
==================
Comprehensive evaluation metrics for retrieval and KB quality.
"""

import logging
from typing import Any, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    mrr: float = 0.0
    coverage: float = 0.0
    avg_cosine: float = 0.0
    structural_score: float = 0.0
    llm_judge_score: float = 0.0
    mean_qa_score: float = 0.0
    violations: int = 0


def calculate_hit_at_k(predictions: List[List[str]], ground_truth: List[str], k: int) -> float:
    """Calculate Hit@K metric."""
    if not predictions or not ground_truth:
        return 0.0
    hits = sum(1 for pred, truth in zip(predictions, ground_truth) 
               if truth in pred[:k])
    return hits / len(ground_truth)


def calculate_mrr(predictions: List[List[str]], ground_truth: List[str]) -> float:
    """Calculate Mean Reciprocal Rank."""
    if not predictions or not ground_truth:
        return 0.0
    
    reciprocal_ranks = []
    for pred, truth in zip(predictions, ground_truth):
        try:
            rank = pred.index(truth) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def calculate_coverage(answered: int, total: int) -> float:
    """Calculate coverage rate."""
    return answered / total if total > 0 else 0.0
