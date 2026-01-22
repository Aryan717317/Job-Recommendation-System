"""
Base Model Interface for Job Recommendation System

This module defines the abstract base class for all vectorization models.
Ensures consistent interface across TF-IDF, BERT, and future models.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseVectorizationModel(ABC):
    """
    Abstract base class for vectorization models.
    
    All vectorization models (TF-IDF, BERT, etc.) must implement this interface
    to ensure consistency and easy swapping in the recommendation engine.
    
    Methods to implement:
    - fit: Train the model on a corpus of documents
    - transform: Convert documents to vectors
    - fit_transform: Fit and transform in one step
    - compute_similarity: Calculate similarity between vectors
    - get_model_name: Return the model identifier
    """
    
    def __init__(self, name: str = "base"):
        """
        Initialize the base model.
        
        Args:
            name: Model identifier name
        """
        self.name = name
        self.is_fitted = False
        self._vocabulary = None
        self._embedding_dim = None
    
    @abstractmethod
    def fit(self, documents: List[str]) -> 'BaseVectorizationModel':
        """
        Fit the model on a corpus of documents.
        
        Args:
            documents: List of text documents for training
        
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to vector representations.
        
        Args:
            documents: List of text documents to transform
        
        Returns:
            Array of document vectors (shape: [n_documents, embedding_dim])
        """
        pass
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fit the model and transform documents in one step.
        
        Args:
            documents: List of text documents
        
        Returns:
            Array of document vectors
        """
        self.fit(documents)
        return self.transform(documents)
    
    @abstractmethod
    def compute_similarity(
        self,
        query_vector: np.ndarray,
        document_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between a query vector and document vectors.
        
        Args:
            query_vector: Vector representation of the query (shape: [1, dim] or [dim])
            document_vectors: Matrix of document vectors (shape: [n_docs, dim])
        
        Returns:
            Array of similarity scores (shape: [n_docs])
        """
        pass
    
    def get_top_n(
        self,
        query_vector: np.ndarray,
        document_vectors: np.ndarray,
        n: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top N most similar documents.
        
        Args:
            query_vector: Query vector
            document_vectors: Document vectors
            n: Number of top results to return
        
        Returns:
            Tuple of (indices, scores) for top N documents
        """
        similarities = self.compute_similarity(query_vector, document_vectors)
        
        # Get indices of top N similarities (descending order)
        top_indices = np.argsort(similarities)[::-1][:n]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def get_model_name(self) -> str:
        """
        Get the model identifier name.
        
        Returns:
            Model name string
        """
        return self.name
    
    @property
    def vocabulary_size(self) -> Optional[int]:
        """Get the vocabulary size (if applicable)."""
        return len(self._vocabulary) if self._vocabulary else None
    
    @property
    def embedding_dimension(self) -> Optional[int]:
        """Get the embedding dimension."""
        return self._embedding_dim
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"is_fitted={self.is_fitted}, "
            f"embedding_dim={self._embedding_dim})"
        )


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity score between 0 and 1
    """
    # Handle sparse matrices
    if hasattr(vec1, 'toarray'):
        vec1 = vec1.toarray().flatten()
    if hasattr(vec2, 'toarray'):
        vec2 = vec2.toarray().flatten()
    
    # Flatten if necessary
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    
    # Compute cosine similarity
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def cosine_similarity_matrix(
    query_vectors: np.ndarray,
    document_vectors: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between query vectors and document vectors.
    
    Args:
        query_vectors: Query vectors (shape: [n_queries, dim])
        document_vectors: Document vectors (shape: [n_docs, dim])
    
    Returns:
        Similarity matrix (shape: [n_queries, n_docs])
    """
    # Handle sparse matrices
    if hasattr(query_vectors, 'toarray'):
        query_vectors = query_vectors.toarray()
    if hasattr(document_vectors, 'toarray'):
        document_vectors = document_vectors.toarray()
    
    # Ensure 2D
    if query_vectors.ndim == 1:
        query_vectors = query_vectors.reshape(1, -1)
    if document_vectors.ndim == 1:
        document_vectors = document_vectors.reshape(1, -1)
    
    # Normalize vectors
    query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    doc_norms = np.linalg.norm(document_vectors, axis=1, keepdims=True)
    
    # Avoid division by zero
    query_norms = np.where(query_norms == 0, 1, query_norms)
    doc_norms = np.where(doc_norms == 0, 1, doc_norms)
    
    query_normalized = query_vectors / query_norms
    doc_normalized = document_vectors / doc_norms
    
    # Compute similarity matrix
    similarities = np.dot(query_normalized, doc_normalized.T)
    
    return similarities


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores to [0, 1] range.
    
    Args:
        scores: Array of scores
    
    Returns:
        Normalized scores
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        return np.ones_like(scores) * 0.5
    
    return (scores - min_score) / (max_score - min_score)
