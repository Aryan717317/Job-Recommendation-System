"""
BERT/Sentence-Transformers Model for Job Recommendation System

This module implements semantic embedding generation using BERT-based models
via the Sentence-Transformers library.

Features:
- Dense semantic embeddings that capture meaning beyond keywords
- Multiple pre-trained model options
- Batch processing for efficiency
- Cosine similarity for semantic matching
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_model import BaseVectorizationModel, cosine_similarity_matrix, normalize_scores
from config import config
from preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)

# Lazy import for optional dependency
SentenceTransformer = None


def _import_sentence_transformers():
    """Lazily import sentence-transformers to handle optional dependency."""
    global SentenceTransformer
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as ST
            SentenceTransformer = ST
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for BERTModel. "
                "Install with: pip install sentence-transformers"
            )
    return SentenceTransformer


class BERTModel(BaseVectorizationModel):
    """
    BERT-based semantic embedding model for job recommendation.
    
    This model uses Sentence-Transformers to generate dense embeddings
    that capture semantic meaning, enabling matching beyond keyword overlap.
    
    Recommended models:
    - all-MiniLM-L6-v2: Fast, good quality (default)
    - all-mpnet-base-v2: Higher quality, slower
    - paraphrase-multilingual-MiniLM-L12-v2: Multilingual support
    
    Usage:
        model = BERTModel()
        
        # No explicit fitting needed (pre-trained)
        candidate_embeddings = model.transform(candidate_texts)
        job_embeddings = model.transform(job_texts)
        
        # Compute similarities
        similarities = model.compute_similarity(
            candidate_embeddings[0], 
            job_embeddings
        )
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_sequence_length: Optional[int] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        use_preprocessing: bool = False,  # BERT handles its own tokenization
        device: Optional[str] = None
    ):
        """
        Initialize the BERT model.
        
        Args:
            model_name: Name of the Sentence-Transformers model to use
            max_sequence_length: Maximum sequence length for tokenization
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to L2-normalize embeddings
            use_preprocessing: Whether to apply custom preprocessing
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        super().__init__(name="bert")
        
        # Use config defaults if not specified
        self.model_name = model_name or config.bert.model_name
        self.max_sequence_length = max_sequence_length or config.bert.max_sequence_length
        self.batch_size = batch_size or config.bert.batch_size
        self.normalize_embeddings = (
            normalize_embeddings 
            if normalize_embeddings is not None 
            else config.bert.normalize_embeddings
        )
        
        # Device selection
        self.device = device
        
        # Initialize model (lazy loading)
        self._model = None
        
        # Preprocessing (optional for BERT)
        self.use_preprocessing = use_preprocessing
        self._preprocessor = TextPreprocessor() if use_preprocessing else None
        
        # Cache
        self._embedding_cache: dict = {}
    
    def _load_model(self):
        """Load the Sentence-Transformer model (lazy loading)."""
        if self._model is None:
            ST = _import_sentence_transformers()
            
            logger.info(f"Loading BERT model: {self.model_name}")
            
            self._model = ST(
                self.model_name,
                device=self.device
            )
            
            # Set max sequence length
            if self.max_sequence_length:
                self._model.max_seq_length = self.max_sequence_length
            
            # Get embedding dimension
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            
            logger.info(
                f"BERT model loaded: embedding_dim={self._embedding_dim}, "
                f"max_seq_length={self._model.max_seq_length}"
            )
        
        return self._model
    
    def _preprocess(self, documents: List[str]) -> List[str]:
        """Optionally preprocess documents."""
        if self._preprocessor:
            return self._preprocessor.process_batch(documents)
        return documents
    
    def fit(self, documents: List[str]) -> 'BERTModel':
        """
        'Fit' the BERT model.
        
        Note: BERT models are pre-trained, so this method just ensures
        the model is loaded and ready. It doesn't perform any training.
        
        Args:
            documents: Documents (not used for training)
        
        Returns:
            Self for method chaining
        """
        # Just load the model
        self._load_model()
        self.is_fitted = True
        
        logger.info("BERT model 'fitted' (pre-trained model loaded)")
        
        return self
    
    def transform(
        self,
        documents: List[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Transform documents to dense semantic embeddings.
        
        Args:
            documents: List of text documents to transform
            show_progress: Whether to show progress bar
        
        Returns:
            Array of embeddings (shape: [n_documents, embedding_dim])
        """
        model = self._load_model()
        self.is_fitted = True
        
        # Preprocess if enabled
        processed_docs = self._preprocess(documents)
        
        # Encode documents
        embeddings = model.encode(
            processed_docs,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def transform_single(self, text: str) -> np.ndarray:
        """
        Transform a single text to embedding.
        
        Uses caching for efficiency.
        
        Args:
            text: Single text document
        
        Returns:
            Embedding vector (shape: [embedding_dim])
        """
        # Check cache
        cache_key = hash(text)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # Transform
        embedding = self.transform([text])[0]
        
        # Cache (limit cache size)
        if len(self._embedding_cache) < 1000:
            self._embedding_cache[cache_key] = embedding
        
        return embedding
    
    def fit_transform(
        self,
        documents: List[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Load model and transform documents.
        
        Args:
            documents: List of text documents
            show_progress: Whether to show progress bar
        
        Returns:
            Array of embeddings
        """
        self.fit(documents)
        return self.transform(documents, show_progress=show_progress)
    
    def compute_similarity(
        self,
        query_vector: np.ndarray,
        document_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings.
        
        Args:
            query_vector: Query embedding
            document_vectors: Matrix of document embeddings
        
        Returns:
            Array of similarity scores
        """
        similarities = cosine_similarity_matrix(query_vector, document_vectors)
        
        # Flatten if single query
        if similarities.shape[0] == 1:
            similarities = similarities.flatten()
        
        return similarities
    
    def encode_and_compare(
        self,
        query_text: str,
        document_texts: List[str],
        top_n: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Encode texts and compute similarities in one step.
        
        Convenience method for quick matching.
        
        Args:
            query_text: Query text
            document_texts: List of documents to compare against
            top_n: Optional number of top results to return
        
        Returns:
            List of (index, score) tuples, sorted by similarity
        """
        # Encode query
        query_embedding = self.transform_single(query_text)
        
        # Encode documents
        doc_embeddings = self.transform(document_texts)
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, doc_embeddings)
        
        # Sort and return
        sorted_indices = np.argsort(similarities)[::-1]
        
        if top_n:
            sorted_indices = sorted_indices[:top_n]
        
        return [
            (int(idx), float(similarities[idx]))
            for idx in sorted_indices
        ]
    
    def semantic_similarity_score(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score between 0 and 1
        """
        emb1 = self.transform_single(text1)
        emb2 = self.transform_single(text2)
        
        return float(self.compute_similarity(emb1, emb2.reshape(1, -1))[0])
    
    def get_similar_skills(
        self,
        skill: str,
        all_skills: List[str],
        threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Find semantically similar skills.
        
        Useful for skill matching beyond exact string matching.
        
        Args:
            skill: Query skill name
            all_skills: List of all skill names to compare against
            threshold: Minimum similarity threshold
        
        Returns:
            List of (similar_skill, score) tuples
        """
        skill_embedding = self.transform_single(skill)
        all_embeddings = self.transform(all_skills)
        
        similarities = self.compute_similarity(skill_embedding, all_embeddings)
        
        similar = []
        for idx, score in enumerate(similarities):
            if score >= threshold and all_skills[idx].lower() != skill.lower():
                similar.append((all_skills[idx], float(score)))
        
        return sorted(similar, key=lambda x: x[1], reverse=True)
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.info("BERT embedding cache cleared")
    
    @property
    def model_info(self) -> dict:
        """Get model information."""
        self._load_model()
        return {
            "model_name": self.model_name,
            "embedding_dim": self._embedding_dim,
            "max_seq_length": self._model.max_seq_length,
            "device": str(self._model.device),
            "normalize_embeddings": self.normalize_embeddings
        }
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        path: str,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Array of embeddings
            path: Path to save
            ids: Optional list of document IDs
        """
        np.savez(
            path,
            embeddings=embeddings,
            ids=np.array(ids) if ids else None,
            model_name=self.model_name
        )
        logger.info(f"Embeddings saved to {path}")
    
    @staticmethod
    def load_embeddings(path: str) -> Tuple[np.ndarray, Optional[List[str]]]:
        """
        Load embeddings from disk.
        
        Args:
            path: Path to load from
        
        Returns:
            Tuple of (embeddings, ids)
        """
        data = np.load(path, allow_pickle=True)
        embeddings = data['embeddings']
        ids = data['ids'].tolist() if data['ids'] is not None else None
        
        return embeddings, ids
