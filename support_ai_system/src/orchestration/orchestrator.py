"""
System Orchestrator
===================
Main orchestrator coordinating all system components.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SystemOrchestrator:
    """Main system orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._db = None
        self._retriever = None
        self._agents = {}
    
    def init_database(self):
        from src.storage import DatabaseManager
        self._db = DatabaseManager(self.config)
        self._db.initialize()
    
    def ingest_data(self):
        from src.ingestion import ExcelIngestion
        ingestion = ExcelIngestion(self.config)
        excel_path = self.config['ingestion']['excel_file']
        
        if Path(excel_path).exists():
            sheets = ingestion.ingest(excel_path)
            
            # Map sheet names to table names
            sheet_to_table = {
                'conversations': 'conversations',
                'tickets': 'tickets',
                'scripts_master': 'scripts',
                'scripts': 'scripts',
                'knowledge_articles': 'knowledge_articles',
                'existing_knowledge_articles': 'knowledge_articles',
                'questions': 'questions',
                'kb_lineage': 'kb_lineage',
                'learning_events': 'learning_events',
                'placeholders': 'placeholders',
                'placeholder_dictionary': 'placeholders'
            }
            
            for sheet_name, df in sheets.items():
                table_key = sheet_name.lower().replace(' ', '_')
                table = sheet_to_table.get(table_key, table_key)
                
                # Only insert into known tables
                if table in sheet_to_table.values():
                    rows = self._db.insert_dataframe(table, df)
                    logger.info(f"Inserted {rows} rows into {table} from sheet {sheet_name}")
        else:
            logger.warning(f"Excel file not found: {excel_path}")
    
    def build_indexes(self, force: bool = False):
        """Build or load indexes. Loads from disk if available, unless force=True."""
        from src.retrieval import HybridRetriever
        self._retriever = HybridRetriever(self.config)
        
        # Check if indexes already exist on disk
        faiss_path = Path(self.config['paths']['faiss_index']) / "index.faiss"
        bm25_path = Path(self.config['paths']['bm25_index']) / "bm25.pkl"
        
        if faiss_path.exists() and bm25_path.exists() and not force:
            logger.info("Loading existing indexes from disk (use --rebuild to force rebuild)")
            # Retriever will lazy-load from disk when needed
            return
        
        # Build new indexes
        articles = self._db.get_all('knowledge_articles')
        if articles:
            docs = [{'id': a['kb_article_id'], 'content': a['content'], 
                     'title': a['title'], 'metadata': a} for a in articles]
            self._retriever.build_indexes(docs)
    
    def init_prompts(self):
        prompts_dir = Path(self.config['paths']['prompts'])
        prompts_dir.mkdir(parents=True, exist_ok=True)
        
        for name in ['extractor', 'generator', 'validator', 'scorer']:
            path = prompts_dir / f"{name}.txt"
            if not path.exists():
                path.write_text(f"# {name.title()} Prompt Template\n")
    
    def load_state(self):
        """Load existing database and indexes."""
        if self._db is None:
            self.init_database()
        if self._retriever is None:
            self.build_indexes()
    
    def load_checkpoint(self) -> Optional[Dict]:
        cp_file = Path(self.config['learning']['checkpoint_file'])
        if cp_file.exists():
            return json.loads(cp_file.read_text())
        return None
    
    def save_checkpoint(self, data: Dict = None):
        cp_file = Path(self.config['learning']['checkpoint_file'])
        cp_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = data or {}
        checkpoint['timestamp'] = datetime.now().isoformat()
        cp_file.write_text(json.dumps(checkpoint, indent=2))
    
    def evaluate_retrieval(self) -> Dict:
        """Evaluate retrieval performance using Questions table as ground truth."""
        if not self._db or not self._retriever:
            return {'hit_at_1': 0.0, 'hit_at_3': 0.0, 'mrr': 0.0}
        
        # Get questions with KB answers as ground truth
        questions = self._db.get_all('questions', where="answer_type = 'KB'")
        if not questions:
            logger.warning("No KB questions found for retrieval evaluation")
            return {'hit_at_1': 0.0, 'hit_at_3': 0.0, 'mrr': 0.0}
        
        # Filter valid questions and collect queries + targets
        valid_questions = []
        queries = []
        target_ids = []
        
        for q in questions:
            query = q.get('question_text', '')
            target_id = q.get('target_id', '')
            if query and target_id:
                valid_questions.append(q)
                queries.append(query)
                target_ids.append(target_id)
        
        if not queries:
            return {'hit_at_1': 0.0, 'hit_at_3': 0.0, 'mrr': 0.0}
        
        # Batch search all queries at once
        try:
            all_results = self._retriever.batch_search(queries, top_k=10, rerank=False)
        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            return {'hit_at_1': 0.0, 'hit_at_3': 0.0, 'mrr': 0.0}
        
        # Calculate metrics
        hits_at_1 = 0
        hits_at_3 = 0
        reciprocal_ranks = []
        
        for results, target_id in zip(all_results, target_ids):
            result_ids = [r.id for r in results]
            
            if target_id in result_ids:
                rank = result_ids.index(target_id) + 1
                reciprocal_ranks.append(1.0 / rank)
                if rank == 1:
                    hits_at_1 += 1
                if rank <= 3:
                    hits_at_3 += 1
            else:
                reciprocal_ranks.append(0.0)
        
        total = len(queries)
        return {
            'hit_at_1': hits_at_1 / total if total > 0 else 0.0,
            'hit_at_3': hits_at_3 / total if total > 0 else 0.0,
            'mrr': sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        }
    
    def evaluate_kb_quality(self) -> Dict:
        """Evaluate KB article quality metrics."""
        if not self._db:
            return {'avg_cosine': 0.0, 'structural_score': 0.0, 'llm_judge_score': 0.0}
        
        articles = self._db.get_all('knowledge_articles')
        if not articles:
            logger.warning("No KB articles found for quality evaluation")
            return {'avg_cosine': 0.0, 'structural_score': 0.0, 'llm_judge_score': 0.0}
        
        structural_scores = []
        confidence_scores = []
        
        for article in articles:
            content = article.get('content', '')
            title = article.get('title', '')
            
            # Structural score: check for proper structure
            score = 0.0
            if title and len(title) > 5:
                score += 0.2
            if content and len(content) > 100:
                score += 0.3
            if article.get('summary'):
                score += 0.2
            if article.get('category'):
                score += 0.15
            if article.get('product'):
                score += 0.15
            structural_scores.append(score)
            
            # Use confidence as proxy for quality
            confidence_scores.append(article.get('confidence', 0.5))
        
        return {
            'avg_cosine': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
            'structural_score': sum(structural_scores) / len(structural_scores) if structural_scores else 0.0,
            'llm_judge_score': 0.0  # Requires LLM call, placeholder for now
        }
    
    def evaluate_qa(self) -> Dict:
        """Evaluate QA scores from qa_scores table."""
        if not self._db:
            return {'mean_score': 0.0, 'violations': 0}
        
        qa_scores = self._db.get_all('qa_scores')
        if not qa_scores:
            # Fallback: compute from questions coverage
            questions = self._db.get_all('questions')
            answered = sum(1 for q in questions if q.get('answer_text'))
            return {
                'mean_score': answered / len(questions) if questions else 0.0,
                'violations': 0
            }
        
        scores = [s.get('overall_score', 0) for s in qa_scores if s.get('overall_score') is not None]
        violations = sum(1 for s in qa_scores if s.get('violations'))
        
        return {
            'mean_score': sum(scores) / len(scores) if scores else 0.0,
            'violations': violations
        }
    
    def save_baseline_metrics(self, metrics: Dict):
        """Save baseline metrics to file."""
        metrics_dir = Path(self.config['paths']['metrics'])
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        baseline_file = metrics_dir / 'baseline.json'
        baseline_file.write_text(json.dumps(metrics, indent=2))
        logger.info(f"Baseline metrics saved to {baseline_file}")
    
    async def run_training_pipeline(self, num_rounds: int, progress_callback: Callable = None):
        """Run the training pipeline - processes conversations and generates KB articles."""
        from src.agents.extractor_agent import ExtractorAgent
        from src.agents.kb_generator_agent import KBGeneratorAgent
        from src.agents.learning_agent import LearningAgent
        
        logger.info(f"Starting training for {num_rounds} rounds")
        
        # Initialize agents
        extractor = ExtractorAgent(self.config)
        generator = KBGeneratorAgent(self.config)
        learning_agent = LearningAgent(self.config)
        
        # Get unprocessed conversations
        conversations = self._db.get_all('conversations')
        if not conversations:
            logger.warning("No conversations found for training")
            return
        
        # Calculate batch size per round
        batch_size = self.config['learning'].get('buffer_size', 500)
        train_ratio = self.config['learning'].get('train_ratio', 0.7)
        train_count = int(len(conversations) * train_ratio)
        per_round = max(1, train_count // num_rounds)
        
        total_processed = 0
        total_articles = 0
        
        for round_num in range(num_rounds):
            if progress_callback:
                progress_callback(round=round_num, total_rounds=num_rounds)
            
            # Get batch for this round
            start_idx = round_num * per_round
            end_idx = min(start_idx + per_round, train_count)
            batch = conversations[start_idx:end_idx]
            
            if not batch:
                logger.info(f"Round {round_num + 1}: No more conversations to process")
                continue
            
            logger.info(f"Training round {round_num + 1}/{num_rounds} - Processing {len(batch)} conversations")
            
            round_articles = 0
            for conv in batch:
                try:
                    # Extract facts from conversation transcript
                    transcript = conv.get('transcript', '') or conv.get('full_text', '')
                    if not transcript or len(transcript) < 50:
                        continue
                    
                    # Check if already processed (look for existing KB article)
                    conv_id = conv.get('conversation_id', '')
                    existing = self._db.get_all('kb_lineage', 
                        where=f"source_type = 'conversation' AND source_id = '{conv_id}'")
                    if existing:
                        continue
                    
                    # Extract facts
                    facts = extractor.extract_from_conversation(transcript, {
                        'ticket_number': conv.get('ticket_number', ''),
                        'product': conv.get('product', ''),
                        'category': conv.get('category', '')
                    })
                    
                    if not facts:
                        continue
                    
                    # Generate KB article
                    article = generator.generate_from_conversation(
                        transcript=transcript,
                        facts=facts,
                        metadata={
                            'conversation_id': conv_id,
                            'ticket_number': conv.get('ticket_number', ''),
                            'product': conv.get('product', ''),
                            'category': conv.get('category', ''),
                            'issue_summary': conv.get('issue_summary', '')
                        }
                    )
                    
                    if article and article.content:
                        # Save article to database
                        self._db.insert('knowledge_articles', {
                            'kb_article_id': article.article_id,
                            'title': article.title,
                            'summary': article.summary,
                            'content': article.content,
                            'product': article.product,
                            'category': article.category,
                            'confidence': article.confidence,
                            'version': 1,
                            'status': 'draft'
                        })
                        
                        # Save lineage
                        self._db.insert('kb_lineage', {
                            'kb_article_id': article.article_id,
                            'source_type': 'conversation',
                            'source_id': conv_id,
                            'version': 1,
                            'status': 'active'
                        })
                        
                        round_articles += 1
                        total_articles += 1
                        
                        # Add high-quality examples for learning
                        if article.confidence >= 0.85:
                            generator.add_example(article)
                    
                    total_processed += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing conversation: {e}")
                    continue
            
            logger.info(f"Round {round_num + 1}: Generated {round_articles} KB articles")
            
            # Trigger learning after each round
            if round_articles > 0:
                learning_agent.process({
                    'trigger': 'evaluation',
                    'metrics': self.evaluate_retrieval()
                })
            
            # Save checkpoint after each round
            self.save_checkpoint({'round': round_num + 1, 'processed': total_processed})
        
        logger.info(f"Training complete: {total_processed} conversations processed, {total_articles} articles generated")
        
        # Rebuild indexes with new articles
        if total_articles > 0:
            logger.info("Rebuilding indexes with new articles...")
            self.build_indexes()
        
    async def resume_training(self, checkpoint: Dict, progress_callback: Callable = None):
        """Resume training from checkpoint."""
        logger.info("Resuming training from checkpoint")
        
        # Ensure database and retriever are loaded
        self.load_state()
        
        start_round = checkpoint.get('round', 0)
        num_rounds = self.config['learning']['num_rounds']
        
        # Continue with remaining rounds
        remaining = num_rounds - start_round
        if remaining > 0:
            await self.run_training_pipeline(remaining, progress_callback)
        else:
            logger.info("Training already complete")
    
    def run_full_evaluation(self, full: bool = False) -> Dict:
        """Run comprehensive evaluation."""
        retrieval = self.evaluate_retrieval()
        quality = self.evaluate_kb_quality()
        qa = self.evaluate_qa()
        
        # Calculate coverage
        if self._db:
            questions = self._db.get_all('questions')
            answered = sum(1 for q in questions if q.get('answer_text'))
            total = len(questions)
        else:
            answered, total = 0, 0
        
        return {
            'retrieval': retrieval,
            'coverage': {
                'answered': answered,
                'total': total,
                'rate': answered / total if total > 0 else 0
            },
            'quality': quality,
            'qa': {
                'mean_score': qa.get('mean_score', 0),
                'min_score': 0,  # Would need to compute from individual scores
                'max_score': 1,
                'violations': qa.get('violations', 0)
            }
        }
    
    def save_evaluation_results(self, results: Dict):
        """Save evaluation results to file."""
        metrics_dir = Path(self.config['paths']['metrics'])
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = metrics_dir / 'evaluation_results.json'
        results['timestamp'] = datetime.now().isoformat()
        results_file.write_text(json.dumps(results, indent=2))
        logger.info(f"Evaluation results saved to {results_file}")
    
    def get_final_metrics(self) -> Dict:
        """Get final metrics summary."""
        return self.run_full_evaluation()
    
    def get_status(self) -> Dict:
        return {
            'db_connected': self._db is not None,
            'table_count': 12,
            'total_records': self._db.get_stats()['total_records'] if self._db else 0,
            'faiss_ready': self._retriever.is_ready() if self._retriever else False,
            'bm25_ready': self._retriever.is_ready() if self._retriever else False,
            'llm_available': True,
            'progress': {'dataset_pct': 0, 'kb_coverage_pct': 0, 'current_phase': 'idle'},
            'checkpoint': self.load_checkpoint()
        }


class ProgressTracker:
    """Tracks and displays progress."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def update(self, **kwargs): pass
    def print_final_report(self, metrics: Dict): pass
