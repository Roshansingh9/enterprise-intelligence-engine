"""
Approval Manager
================
Manages confidence-based approval workflow.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ApprovalManager:
    """
    Routes KB articles through approval workflow based on confidence.
    
    Thresholds:
    - >= auto_approve: Automatically approved
    - >= human_review: Queued for human review
    - < reject: Rejected outright
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        gov_config = config.get('governance', {})
        self.auto_approve = gov_config.get('auto_approve_threshold', 0.95)
        self.human_review = gov_config.get('human_review_threshold', 0.7)
        self.reject = gov_config.get('reject_threshold', 0.5)
        self.require_human = gov_config.get('require_human_approval', False)
        self.queue_size = gov_config.get('approval_queue_size', 50)
        
        # In-memory queue (would be DB in production)
        self.pending_queue: List[Dict] = []
    
    def evaluate(
        self,
        article_id: str,
        confidence: float,
        pii_safe: bool = True,
        hallucination_check: Dict = None
    ) -> Dict[str, Any]:
        """
        Evaluate article for approval.
        
        Returns:
            {decision, status, reason, requires_action}
        """
        result = {
            'article_id': article_id,
            'confidence': confidence,
            'decision': 'pending',
            'status': 'pending_review',
            'reason': '',
            'requires_action': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # PII check
        if not pii_safe:
            result['decision'] = 'rejected'
            result['status'] = 'rejected'
            result['reason'] = 'Contains PII - must be redacted'
            result['requires_action'] = True
            return result
        
        # Hallucination check
        if hallucination_check and not hallucination_check.get('is_valid', True):
            result['decision'] = 'rejected'
            result['status'] = 'rejected'
            result['reason'] = f"Hallucination detected: {hallucination_check.get('issues', [])}"
            result['requires_action'] = True
            return result
        
        # Confidence-based routing
        if self.require_human:
            result['decision'] = 'human_review'
            result['status'] = 'pending_review'
            result['reason'] = 'Human approval required by policy'
            self._add_to_queue(result)
        elif confidence >= self.auto_approve:
            result['decision'] = 'approved'
            result['status'] = 'approved'
            result['reason'] = f'Auto-approved (confidence {confidence:.2f} >= {self.auto_approve})'
        elif confidence >= self.human_review:
            result['decision'] = 'human_review'
            result['status'] = 'pending_review'
            result['reason'] = f'Needs review (confidence {confidence:.2f})'
            self._add_to_queue(result)
        elif confidence >= self.reject:
            result['decision'] = 'human_review'
            result['status'] = 'pending_review'
            result['reason'] = f'Low confidence ({confidence:.2f}) - needs review'
            result['requires_action'] = True
            self._add_to_queue(result)
        else:
            result['decision'] = 'rejected'
            result['status'] = 'rejected'
            result['reason'] = f'Rejected (confidence {confidence:.2f} < {self.reject})'
        
        logger.info(f"Article {article_id}: {result['decision']} - {result['reason']}")
        return result
    
    def _add_to_queue(self, item: Dict) -> None:
        """Add item to review queue."""
        self.pending_queue.append(item)
        if len(self.pending_queue) > self.queue_size:
            self.pending_queue = self.pending_queue[-self.queue_size:]
    
    def get_pending(self) -> List[Dict]:
        """Get pending review items."""
        return self.pending_queue.copy()
    
    def approve(self, article_id: str, reviewer: str = None) -> bool:
        """Manually approve an article."""
        for item in self.pending_queue:
            if item['article_id'] == article_id:
                item['decision'] = 'approved'
                item['status'] = 'approved'
                item['reason'] = f'Manually approved by {reviewer or "admin"}'
                self.pending_queue.remove(item)
                return True
        return False
    
    def reject(self, article_id: str, reason: str, reviewer: str = None) -> bool:
        """Manually reject an article."""
        for item in self.pending_queue:
            if item['article_id'] == article_id:
                item['decision'] = 'rejected'
                item['status'] = 'rejected'
                item['reason'] = f'Rejected by {reviewer or "admin"}: {reason}'
                self.pending_queue.remove(item)
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get approval statistics."""
        return {
            'pending_count': len(self.pending_queue),
            'auto_approve_threshold': self.auto_approve,
            'human_review_threshold': self.human_review,
            'reject_threshold': self.reject,
            'require_human': self.require_human
        }
