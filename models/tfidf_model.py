"""
TF-IDF Vectorization Model for Job Recommendation System

This module implements TF-IDF (Term Frequency-Inverse Document Frequency)
vectorization for job-candidate matching.

Features:
- Configurable n-gram range
- Shared vocabulary between candidates and jobs
- Cosine similarity computation
- Sublinear TF scaling option
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_model import BaseVectorizationModel, cosine_similarity_matrix, normalize_scores
from config import config
from preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


class TFIDFModel(BaseVectorizationModel):
    """
    TF-IDF based vectorization model for job recommendation.
    
    This model uses scikit-learn's TfidfVectorizer to convert text documents
    into TF-IDF vectors, then uses cosine similarity for matching.
    
    Usage:
        model = TFIDFModel()
        
        # Fit on all documents (candidates + jobs)
        all_texts = candidate_texts + job_texts
        model.fit(all_texts)
        
        # Transform and compute similarities
        candidate_vecs = model.transform(candidate_texts)
        job_vecs = model.transform(job_texts)
        similarities = model.compute_similarity(candidate_vecs[0], job_vecs)
    """
    
    def __init__(
        self,
        max_features: Optional[int] = None,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        use_preprocessing: bool = True
    ):
        """
        Initialize the TF-IDF model.
        
        Args:
            max_features: Maximum number of vocabulary terms
            ngram_range: N-gram range (min, max)
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            sublinear_tf: Use sublinear TF scaling (1 + log(tf))
            use_preprocessing: Whether to preprocess text before vectorizing
        """
        super().__init__(name="tfidf")
        
        # Use config defaults if not specified
        self.max_features = max_features or config.tfidf.max_features
        self.ngram_range = ngram_range or config.tfidf.ngram_range
        self.min_df = min_df or config.tfidf.min_df
        self.max_df = max_df or config.tfidf.max_df
        self.sublinear_tf = sublinear_tf if sublinear_tf is not None else config.tfidf.sublinear_tf
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            lowercase=True,  # We handle this in preprocessing
            stop_words=None,  # We handle this in preprocessing
            token_pattern=r'(?u)\b\w+\b'  # Include single-character tokens
        )
        
        # Preprocessing
        self.use_preprocessing = use_preprocessing
        self._preprocessor = TextPreprocessor() if use_preprocessing else None
        
        # Cache for transformed document vectors
        self._document_vectors: Optional[csr_matrix] = None
        self._document_ids: Optional[List[str]] = None
    
    def _preprocess(self, documents: List[str]) -> List[str]:
        """Preprocess documents if preprocessing is enabled."""
        if self._preprocessor:
            return self._preprocessor.process_batch(documents)
        return documents
    
    def fit(self, documents: List[str]) -> 'TFIDFModel':
        """
        Fit the TF-IDF vectorizer on a corpus of documents.
        
        This builds the vocabulary from all provided documents.
        For job recommendation, fit on both candidate texts and job texts
        to create a shared vocabulary.
        
        Args:
            documents: List of text documents for vocabulary building
        
        Returns:
            Self for method chaining
        """
        if not documents:
            raise ValueError("Cannot fit on empty document list")
        
        # Preprocess documents
        processed_docs = self._preprocess(documents)
        
        # Fit vectorizer
        self.vectorizer.fit(processed_docs)
        
        # Store vocabulary info
        self._vocabulary = self.vectorizer.vocabulary_
        self._embedding_dim = len(self._vocabulary)
        self.is_fitted = True
        
        logger.info(
            f"TF-IDF model fitted with vocabulary size: {self._embedding_dim}, "
            f"ngram_range: {self.ngram_range}"
        )
        
        return self
    
    def transform(self, documents: List[str]) -> csr_matrix:
        """
        Transform documents to TF-IDF vectors.
        
        Args:
            documents: List of text documents to transform
        
        Returns:
            Sparse matrix of TF-IDF vectors (shape: [n_documents, vocabulary_size])
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform")
        
        # Preprocess documents
        processed_docs = self._preprocess(documents)
        
        # Transform to TF-IDF vectors
        vectors = self.vectorizer.transform(processed_docs)
        
        return vectors
    
    def fit_transform(self, documents: List[str]) -> csr_matrix:
        """
        Fit the model and transform documents in one step.
        
        Args:
            documents: List of text documents
        
        Returns:
            Sparse matrix of TF-IDF vectors
        """
        if not documents:
            raise ValueError("Cannot fit on empty document list")
        
        # Preprocess documents
        processed_docs = self._preprocess(documents)
        
        # Fit and transform
        vectors = self.vectorizer.fit_transform(processed_docs)
        
        # Store vocabulary info
        self._vocabulary = self.vectorizer.vocabulary_
        self._embedding_dim = len(self._vocabulary)
        self.is_fitted = True
        
        logger.info(
            f"TF-IDF model fitted with vocabulary size: {self._embedding_dim}"
        )
        
        return vectors
    
    def compute_similarity(
        self,
        query_vector: np.ndarray,
        document_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and document vectors.
        
        Args:
            query_vector: Query vector (single candidate or job)
            document_vectors: Matrix of document vectors
        
        Returns:
            Array of similarity scores
        """
        similarities = cosine_similarity_matrix(query_vector, document_vectors)
        
        # Flatten if single query
        if similarities.shape[0] == 1:
            similarities = similarities.flatten()
        
        return similarities
    
    def get_top_n_similar(
        self,
        query_text: str,
        document_texts: List[str],
        n: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Get top N most similar documents to a query.
        
        Convenience method that handles vectorization internally.
        
        Args:
            query_text: Query text (candidate or job)
            document_texts: List of document texts to rank
            n: Number of top results to return
        
        Returns:
            List of (index, score) tuples for top N documents
        """
        # Transform query
        query_vec = self.transform([query_text])
        
        # Transform documents
        doc_vecs = self.transform(document_texts)
        
        # Compute similarities
        similarities = self.compute_similarity(query_vec, doc_vecs)
        
        # Get top N indices
        top_indices = np.argsort(similarities)[::-1][:n]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names (vocabulary terms).
        
        Returns:
            List of vocabulary terms
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_top_terms_for_document(
        self,
        document: str,
        n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top N most important terms for a document.
        
        Args:
            document: Text document
            n: Number of top terms to return
        
        Returns:
            List of (term, tfidf_score) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        # Transform document
        vector = self.transform([document])
        
        # Get non-zero entries
        if hasattr(vector, 'toarray'):
            dense_vector = vector.toarray().flatten()
        else:
            dense_vector = vector.flatten()
        
        # Get feature names
        feature_names = self.get_feature_names()
        
        # Get top N indices
        top_indices = np.argsort(dense_vector)[::-1][:n]
        
        return [
            (feature_names[idx], float(dense_vector[idx]))
            for idx in top_indices
            if dense_vector[idx] > 0
        ]
    
    def compute_skill_overlap_score(
        self,
        candidate_skills: List[str],
        job_skills: List[str]
    ) -> float:
        """
        Compute skill overlap score between candidate and job.
        
        This is a supplementary metric that measures direct skill matches,
        which can be combined with TF-IDF similarity.
        
        Args:
            candidate_skills: List of candidate skill names
            job_skills: List of job skill names
        
        Returns:
            Overlap score between 0 and 1
        """
        if not job_skills:
            return 0.0
        
        # Normalize skill names
        candidate_set = {s.lower().strip() for s in candidate_skills}
        job_set = {s.lower().strip() for s in job_skills}
        
        # Compute overlap
        overlap = candidate_set & job_set
        
        # Calculate score (Jaccard similarity)
        union = candidate_set | job_set
        if not union:
            return 0.0
        
        return len(overlap) / len(union)
    
    def save_model(self, path: str) -> None:
        """
        Save the fitted model to disk.
        
        Args:
            path: Path to save the model
        """
        import pickle
        
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'vocabulary': self._vocabulary,
                'embedding_dim': self._embedding_dim,
                'config': {
                    'max_features': self.max_features,
                    'ngram_range': self.ngram_range,
                    'min_df': self.min_df,
                    'max_df': self.max_df,
                    'sublinear_tf': self.sublinear_tf
                }
            }, f)
        
        logger.info(f"TF-IDF model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'TFIDFModel':
        """
        Load a fitted model from disk.
        
        Args:
            path: Path to load the model from
        
        Returns:
            Loaded TFIDFModel instance
        """
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        config_data = data['config']
        model = cls(
            max_features=config_data['max_features'],
            ngram_range=config_data['ngram_range'],
            min_df=config_data['min_df'],
            max_df=config_data['max_df'],
            sublinear_tf=config_data['sublinear_tf']
        )
        
        model.vectorizer = data['vectorizer']
        model._vocabulary = data['vocabulary']
        model._embedding_dim = data['embedding_dim']
        model.is_fitted = True
        
        logger.info(f"TF-IDF model loaded from {path}")
        
        return model
