"""
Hybrid Retrieval Engine
=======================
5-stage retrieval pipeline with semantic search, BM25, metadata filtering,
LLM reranking, and score fusion.
"""

import json
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with scores."""
    id: str
    content: str
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    validation_score: float = 0.0
    recency_score: float = 0.0
    final_score: float = 0.0


class HybridRetriever:
    """
    Hybrid retrieval system combining:
    1. Semantic Search (FAISS)
    2. Keyword Search (BM25)
    3. Metadata Filtering
    4. LLM Re-ranking
    5. Score Fusion
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retrieval_config = config['retrieval']
        
        # Score weights
        self.weights = self.retrieval_config['weights']
        
        # Top-K settings
        self.semantic_top_k = self.retrieval_config['semantic_top_k']
        self.bm25_top_k = self.retrieval_config['bm25_top_k']
        self.final_top_k = self.retrieval_config['final_top_k']
        self.rerank_top_k = self.retrieval_config.get('rerank_top_k', 20)
        
        # Components (lazy loaded)
        self._embedding_model = None
        self._faiss_index = None
        self._bm25_index = None
        self._document_store = {}
        
        # LLM client for reranking
        self._llm_client = None
        self.rerank_enabled = self.retrieval_config.get('rerank_enabled', True)
        
        # Paths
        self.faiss_path = Path(config['paths']['faiss_index'])
        self.bm25_path = Path(config['paths']['bm25_index'])
    
    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            model_name = self.config['embeddings']['model']
            logger.info(f"Loading embedding model: {model_name}")
            self._embedding_model = SentenceTransformer(model_name)
        return self._embedding_model
    
    @property
    def faiss_index(self):
        """Get or load FAISS index."""
        if self._faiss_index is None:
            self._load_faiss_index()
        return self._faiss_index
    
    @property
    def bm25_index(self):
        """Get or load BM25 index."""
        if self._bm25_index is None:
            self._load_bm25_index()
        return self._bm25_index
    
    def build_indexes(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build both FAISS and BM25 indexes from documents.
        
        Args:
            documents: List of documents with 'id', 'content', 'title', and 'metadata'
        """
        logger.info(f"Building indexes for {len(documents)} documents")
        
        # Store documents
        self._document_store = {doc['id']: doc for doc in documents}
        
        # Build FAISS index
        self._build_faiss_index(documents)
        
        # Build BM25 index
        self._build_bm25_index(documents)
        
        # Save indexes
        self._save_indexes()
        
        logger.info("Indexes built successfully")
    
    def _build_faiss_index(self, documents: List[Dict[str, Any]]) -> None:
        """Build FAISS semantic search index."""
        import faiss
        
        logger.info("Building FAISS index...")
        
        # Get embeddings
        texts = [f"{doc.get('title', '')} {doc['content']}" for doc in documents]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=self.config['embeddings']['batch_size'],
            normalize_embeddings=self.config['embeddings']['normalize']
        )
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        
        # Use IVF index for larger datasets
        if len(documents) > 1000:
            nlist = min(100, len(documents) // 10)
            quantizer = faiss.IndexFlatIP(dimension)
            self._faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            self._faiss_index.train(embeddings.astype('float32'))
        else:
            self._faiss_index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings
        self._faiss_index.add(embeddings.astype('float32'))
        
        # Store ID mapping
        self._faiss_id_map = [doc['id'] for doc in documents]
        
        logger.info(f"FAISS index built with {self._faiss_index.ntotal} vectors")
    
    def _build_bm25_index(self, documents: List[Dict[str, Any]]) -> None:
        """Build BM25 keyword search index."""
        from rank_bm25 import BM25Okapi
        import re
        
        logger.info("Building BM25 index...")
        
        # Tokenize documents
        def tokenize(text: str) -> List[str]:
            text = text.lower()
            tokens = re.findall(r'\b\w+\b', text)
            return tokens
        
        tokenized_corpus = []
        self._bm25_id_map = []
        
        for doc in documents:
            text = f"{doc.get('title', '')} {doc['content']}"
            tokenized_corpus.append(tokenize(text))
            self._bm25_id_map.append(doc['id'])
        
        self._bm25_index = BM25Okapi(tokenized_corpus)
        
        logger.info(f"BM25 index built with {len(tokenized_corpus)} documents")
    
    def _save_indexes(self) -> None:
        """Save indexes to disk."""
        import faiss
        
        # Save FAISS index
        self.faiss_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss_index, str(self.faiss_path / "index.faiss"))
        
        with open(self.faiss_path / "id_map.json", 'w') as f:
            json.dump(self._faiss_id_map, f)
        
        # Save BM25 index
        self.bm25_path.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_path / "bm25.pkl", 'wb') as f:
            pickle.dump({
                'index': self._bm25_index,
                'id_map': self._bm25_id_map
            }, f)
        
        # Save document store
        with open(self.faiss_path / "documents.json", 'w') as f:
            json.dump(self._document_store, f)
        
        logger.info("Indexes saved to disk")
    
    def _load_faiss_index(self) -> None:
        """Load FAISS index from disk."""
        import faiss
        
        index_path = self.faiss_path / "index.faiss"
        id_map_path = self.faiss_path / "id_map.json"
        docs_path = self.faiss_path / "documents.json"
        
        if not index_path.exists():
            logger.warning("FAISS index not found")
            return
        
        self._faiss_index = faiss.read_index(str(index_path))
        
        with open(id_map_path, 'r') as f:
            self._faiss_id_map = json.load(f)
        
        if docs_path.exists():
            with open(docs_path, 'r') as f:
                self._document_store = json.load(f)
        
        logger.info(f"FAISS index loaded with {self._faiss_index.ntotal} vectors")
    
    def _load_bm25_index(self) -> None:
        """Load BM25 index from disk."""
        bm25_path = self.bm25_path / "bm25.pkl"
        
        if not bm25_path.exists():
            logger.warning("BM25 index not found")
            return
        
        with open(bm25_path, 'rb') as f:
            data = pickle.load(f)
            self._bm25_index = data['index']
            self._bm25_id_map = data['id_map']
        
        logger.info(f"BM25 index loaded with {len(self._bm25_id_map)} documents")
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True
    ) -> List[RetrievalResult]:
        """
        Execute hybrid search with all 5 stages.
        
        Args:
            query: Search query
            top_k: Number of results to return (default: config value)
            filters: Metadata filters to apply
            rerank: Whether to apply LLM reranking
            
        Returns:
            List of RetrievalResult objects
        """
        top_k = top_k or self.final_top_k
        
        # Stage 1: Semantic Search
        semantic_results = self._semantic_search(query)
        
        # Stage 2: BM25 Search
        bm25_results = self._bm25_search(query)
        
        # Merge results
        all_ids = set(semantic_results.keys()) | set(bm25_results.keys())
        
        results = []
        for doc_id in all_ids:
            doc = self._document_store.get(doc_id, {})
            
            result = RetrievalResult(
                id=doc_id,
                content=doc.get('content', ''),
                title=doc.get('title', ''),
                metadata=doc.get('metadata', {}),
                semantic_score=semantic_results.get(doc_id, 0.0),
                bm25_score=bm25_results.get(doc_id, 0.0)
            )
            results.append(result)
        
        # Stage 3: Metadata Filtering
        if filters:
            results = self._apply_filters(results, filters)
        
        # Calculate validation and recency scores
        for result in results:
            result.validation_score = self._get_validation_score(result)
            result.recency_score = self._get_recency_score(result)
        
        # Stage 4: LLM Reranking (optional)
        if rerank and self.rerank_enabled and len(results) > 0:
            results = self._llm_rerank(query, results[:self.rerank_top_k])
        
        # Stage 5: Score Fusion
        for result in results:
            result.final_score = self._calculate_final_score(result)
        
        # Sort by final score and return top-K
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results[:top_k]
    
    def batch_search(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        rerank: bool = False
    ) -> List[List[RetrievalResult]]:
        """
        Execute batch hybrid search for multiple queries at once.
        More efficient than calling search() multiple times.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            rerank: Whether to apply LLM reranking
            
        Returns:
            List of result lists, one per query
        """
        if not queries:
            return []
        
        top_k = top_k or self.final_top_k
        
        # Batch semantic search (single embedding call)
        semantic_results_list = self._batch_semantic_search(queries)
        
        # BM25 search for each query (already fast)
        bm25_results_list = [self._bm25_search(q) for q in queries]
        
        all_results = []
        for i, query in enumerate(queries):
            semantic_results = semantic_results_list[i]
            bm25_results = bm25_results_list[i]
            
            # Merge results
            all_ids = set(semantic_results.keys()) | set(bm25_results.keys())
            
            results = []
            for doc_id in all_ids:
                doc = self._document_store.get(doc_id, {})
                
                result = RetrievalResult(
                    id=doc_id,
                    content=doc.get('content', ''),
                    title=doc.get('title', ''),
                    metadata=doc.get('metadata', {}),
                    semantic_score=semantic_results.get(doc_id, 0.0),
                    bm25_score=bm25_results.get(doc_id, 0.0)
                )
                results.append(result)
            
            # Calculate validation and recency scores
            for result in results:
                result.validation_score = self._get_validation_score(result)
                result.recency_score = self._get_recency_score(result)
            
            # Score fusion
            for result in results:
                result.final_score = self._calculate_final_score(result)
            
            # Sort by final score and return top-K
            results.sort(key=lambda x: x.final_score, reverse=True)
            all_results.append(results[:top_k])
        
        return all_results
    
    def _batch_semantic_search(self, queries: List[str]) -> List[Dict[str, float]]:
        """Execute batch semantic search with FAISS for multiple queries at once."""
        if self._faiss_index is None:
            self._load_faiss_index()
        
        if self._faiss_index is None:
            return [{} for _ in queries]
        
        # Get all query embeddings in one batch call
        query_embeddings = self.embedding_model.encode(
            queries,
            normalize_embeddings=self.config['embeddings']['normalize'],
            show_progress_bar=True,
            batch_size=self.config['embeddings']['batch_size']
        )
        
        # Batch search
        scores_batch, indices_batch = self._faiss_index.search(
            query_embeddings.astype('float32'),
            self.semantic_top_k
        )
        
        all_results = []
        for scores, indices in zip(scores_batch, indices_batch):
            results = {}
            for score, idx in zip(scores, indices):
                if idx >= 0 and idx < len(self._faiss_id_map):
                    doc_id = self._faiss_id_map[idx]
                    # Normalize score to 0-1
                    results[doc_id] = float(max(0, min(1, (score + 1) / 2)))
            all_results.append(results)
        
        return all_results

    def _semantic_search(self, query: str) -> Dict[str, float]:
        """Execute semantic search with FAISS."""
        if self._faiss_index is None:
            self._load_faiss_index()
        
        if self._faiss_index is None:
            return {}
        
        # Get query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=self.config['embeddings']['normalize']
        )
        
        # Search
        scores, indices = self._faiss_index.search(
            query_embedding.astype('float32'),
            self.semantic_top_k
        )
        
        results = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._faiss_id_map):
                doc_id = self._faiss_id_map[idx]
                # Normalize score to 0-1
                results[doc_id] = float(max(0, min(1, (score + 1) / 2)))
        
        return results
    
    def _bm25_search(self, query: str) -> Dict[str, float]:
        """Execute BM25 keyword search."""
        import re
        
        if self._bm25_index is None:
            self._load_bm25_index()
        
        if self._bm25_index is None:
            return {}
        
        # Tokenize query
        tokens = re.findall(r'\b\w+\b', query.lower())
        
        # Get scores
        scores = self._bm25_index.get_scores(tokens)
        
        # Normalize and return top-K
        max_score = max(scores) if max(scores) > 0 else 1
        
        results = {}
        top_indices = np.argsort(scores)[::-1][:self.bm25_top_k]
        
        for idx in top_indices:
            if scores[idx] > 0:
                doc_id = self._bm25_id_map[idx]
                results[doc_id] = float(scores[idx] / max_score)
        
        return results
    
    def _apply_filters(
        self,
        results: List[RetrievalResult],
        filters: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """Apply metadata filters to results."""
        filtered = []
        
        min_confidence = filters.get('min_confidence', self.retrieval_config.get('min_confidence', 0))
        allowed_statuses = filters.get('status', self.retrieval_config.get('allowed_statuses', []))
        
        for result in results:
            metadata = result.metadata
            
            # Check confidence
            if metadata.get('confidence', 1.0) < min_confidence:
                continue
            
            # Check status
            if allowed_statuses and metadata.get('status') not in allowed_statuses:
                continue
            
            # Custom filters
            match = True
            for key, value in filters.items():
                if key in ['min_confidence', 'status']:
                    continue
                if metadata.get(key) != value:
                    match = False
                    break
            
            if match:
                filtered.append(result)
        
        return filtered
    
    def _get_validation_score(self, result: RetrievalResult) -> float:
        """Calculate validation score based on metadata."""
        metadata = result.metadata
        
        # Base score from confidence
        score = metadata.get('confidence', 0.5)
        
        # Boost for approved articles
        if metadata.get('status') == 'approved':
            score *= 1.2
        
        # Boost for verified sources
        if metadata.get('verified', False):
            score *= 1.1
        
        return min(1.0, score)
    
    def _get_recency_score(self, result: RetrievalResult) -> float:
        """Calculate recency score based on update date."""
        metadata = result.metadata
        
        updated_at = metadata.get('updated_at')
        if not updated_at:
            return 0.5
        
        try:
            if isinstance(updated_at, str):
                update_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            else:
                update_date = updated_at
            
            days_old = (datetime.now() - update_date.replace(tzinfo=None)).days
            
            # Decay function: full score for <7 days, 0.5 at 30 days, 0.1 at 90 days
            if days_old < 7:
                return 1.0
            elif days_old < 30:
                return 0.9 - (days_old - 7) * 0.02
            elif days_old < 90:
                return 0.5 - (days_old - 30) * 0.007
            else:
                return 0.1
        except:
            return 0.5
    
    def _llm_rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank results using LLM."""
        if not results:
            return results
        
        try:
            from src.agents.llm_client import LLMClient
            
            if self._llm_client is None:
                self._llm_client = LLMClient(self.config)
            
            # Prepare documents for reranking
            docs_text = []
            for i, result in enumerate(results):
                docs_text.append(f"[{i}] {result.title}: {result.content[:500]}")
            
            prompt = f"""Given the query: "{query}"

Rank the following documents by relevance. Return a comma-separated list of document indices from most to least relevant.

Documents:
{chr(10).join(docs_text)}

Return ONLY the comma-separated indices, e.g.: 2,0,3,1,4"""
            
            response = self._llm_client.generate(prompt, max_tokens=100)
            
            # Parse response
            try:
                indices = [int(x.strip()) for x in response.split(',')]
                reranked = []
                for idx in indices:
                    if 0 <= idx < len(results):
                        # Boost score based on rerank position
                        result = results[idx]
                        position_boost = 1.0 - (len(reranked) * 0.05)
                        result.semantic_score *= position_boost
                        reranked.append(result)
                
                # Add any missing results
                for result in results:
                    if result not in reranked:
                        reranked.append(result)
                
                return reranked
            except:
                return results
                
        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}")
            return results
    
    def _calculate_final_score(self, result: RetrievalResult) -> float:
        """Calculate final fused score."""
        return (
            self.weights['semantic'] * result.semantic_score +
            self.weights['bm25'] * result.bm25_score +
            self.weights['validation'] * result.validation_score +
            self.weights['recency'] * result.recency_score
        )
    
    def is_ready(self) -> bool:
        """Check if retriever is ready to use."""
        faiss_exists = (self.faiss_path / "index.faiss").exists()
        bm25_exists = (self.bm25_path / "bm25.pkl").exists()
        return faiss_exists and bm25_exists
