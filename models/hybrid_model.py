"""
Hybrid Model for Job Recommendation System

This module implements a hybrid scoring approach that combines
TF-IDF (keyword-based) and BERT (semantic-based) similarities.

Features:
- Weighted combination of TF-IDF and BERT scores
- Configurable fusion strategies
- Skill-aware scoring adjustments
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_model import BaseVectorizationModel, normalize_scores
from .tfidf_model import TFIDFModel
from .bert_model import BERTModel
from config import config
from preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


class HybridModel(BaseVectorizationModel):
    """
    Hybrid model combining TF-IDF and BERT for job recommendation.
    
    This model leverages both:
    - TF-IDF: Captures exact keyword matches (important for skills)
    - BERT: Captures semantic similarity (important for job descriptions)
    
    The final score is a weighted combination of both approaches.
    
    Usage:
        model = HybridModel(tfidf_weight=0.4, bert_weight=0.6)
        model.fit(all_documents)
        
        candidate_scores = model.get_recommendation_scores(
            candidate_text,
            job_texts
        )
    """
    
    def __init__(
        self,
        tfidf_weight: float = 0.4,
        bert_weight: float = 0.6,
        skill_match_weight: float = 0.0,
        tfidf_config: Optional[Dict] = None,
        bert_config: Optional[Dict] = None,
        use_preprocessing: bool = True
    ):
        """
        Initialize the hybrid model.
        
        Args:
            tfidf_weight: Weight for TF-IDF similarity (0-1)
            bert_weight: Weight for BERT similarity (0-1)
            skill_match_weight: Additional weight for direct skill matches (0-1)
            tfidf_config: Configuration dict for TF-IDF model
            bert_config: Configuration dict for BERT model
            use_preprocessing: Whether to preprocess text
        """
        super().__init__(name="hybrid")
        
        # Normalize weights
        total = tfidf_weight + bert_weight + skill_match_weight
        if total > 0:
            self.tfidf_weight = tfidf_weight / total
            self.bert_weight = bert_weight / total
            self.skill_match_weight = skill_match_weight / total
        else:
            self.tfidf_weight = 0.4
            self.bert_weight = 0.6
            self.skill_match_weight = 0.0
        
        # Initialize sub-models
        tfidf_params = tfidf_config or {}
        bert_params = bert_config or {}
        
        self.tfidf_model = TFIDFModel(
            use_preprocessing=use_preprocessing,
            **tfidf_params
        )
        
        self.bert_model = BERTModel(
            use_preprocessing=False,  # BERT has its own preprocessing
            **bert_params
        )
        
        # Preprocessing
        self.use_preprocessing = use_preprocessing
        self._preprocessor = TextPreprocessor() if use_preprocessing else None
        
        # Cached vectors
        self._tfidf_vectors = None
        self._bert_vectors = None
        self._document_ids = None
    
    def fit(self, documents: List[str]) -> 'HybridModel':
        """
        Fit both TF-IDF and BERT models on the document corpus.
        
        Args:
            documents: List of text documents
        
        Returns:
            Self for method chaining
        """
        logger.info("Fitting hybrid model...")
        
        # Fit TF-IDF
        self.tfidf_model.fit(documents)
        
        # Fit BERT (just loads the model)
        self.bert_model.fit(documents)
        
        self.is_fitted = True
        self._embedding_dim = (
            self.tfidf_model.embedding_dimension,
            self.bert_model.embedding_dimension
        )
        
        logger.info(
            f"Hybrid model fitted: TF-IDF dim={self.tfidf_model.embedding_dimension}, "
            f"BERT dim={self.bert_model.embedding_dimension}"
        )
        
        return self
    
    def transform(
        self,
        documents: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform documents using both models.
        
        Args:
            documents: List of text documents
        
        Returns:
            Tuple of (tfidf_vectors, bert_vectors)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform")
        
        tfidf_vecs = self.tfidf_model.transform(documents)
        bert_vecs = self.bert_model.transform(documents)
        
        return tfidf_vecs, bert_vecs
    
    def compute_similarity(
        self,
        query_vector: Tuple[np.ndarray, np.ndarray],
        document_vectors: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """
        Compute hybrid similarity scores.
        
        Args:
            query_vector: Tuple of (tfidf_query, bert_query)
            document_vectors: Tuple of (tfidf_docs, bert_docs)
        
        Returns:
            Array of hybrid similarity scores
        """
        tfidf_query, bert_query = query_vector
        tfidf_docs, bert_docs = document_vectors
        
        # Compute individual similarities
        tfidf_sim = self.tfidf_model.compute_similarity(tfidf_query, tfidf_docs)
        bert_sim = self.bert_model.compute_similarity(bert_query, bert_docs)
        
        # Combine with weights
        hybrid_sim = (
            self.tfidf_weight * tfidf_sim + 
            self.bert_weight * bert_sim
        )
        
        return hybrid_sim
    
    def get_recommendation_scores(
        self,
        candidate_text: str,
        job_texts: List[str],
        candidate_skills: Optional[List[str]] = None,
        job_skills_list: Optional[List[List[str]]] = None
    ) -> Dict:
        """
        Get comprehensive recommendation scores for a candidate.
        
        Returns individual scores from each model plus the hybrid score.
        
        Args:
            candidate_text: Candidate's combined text
            job_texts: List of job description texts
            candidate_skills: Optional list of candidate skill names
            job_skills_list: Optional list of skill lists for each job
        
        Returns:
            Dictionary with:
            - hybrid_scores: Combined similarity scores
            - tfidf_scores: TF-IDF similarity scores
            - bert_scores: BERT similarity scores
            - skill_scores: Direct skill match scores (if provided)
            - rankings: Job indices sorted by hybrid score
        """
        if not self.is_fitted:
            self.fit([candidate_text] + job_texts)
        
        # Transform candidate
        cand_tfidf = self.tfidf_model.transform([candidate_text])
        cand_bert = self.bert_model.transform([candidate_text])
        
        # Transform jobs
        job_tfidf = self.tfidf_model.transform(job_texts)
        job_bert = self.bert_model.transform(job_texts)
        
        # Compute similarities
        tfidf_scores = self.tfidf_model.compute_similarity(cand_tfidf, job_tfidf)
        bert_scores = self.bert_model.compute_similarity(cand_bert, job_bert)
        
        # Compute skill match scores if skills provided
        skill_scores = None
        if candidate_skills and job_skills_list:
            skill_scores = np.array([
                self._compute_skill_overlap(candidate_skills, job_skills)
                for job_skills in job_skills_list
            ])
        
        # Compute hybrid scores
        if skill_scores is not None and self.skill_match_weight > 0:
            hybrid_scores = (
                self.tfidf_weight * tfidf_scores +
                self.bert_weight * bert_scores +
                self.skill_match_weight * skill_scores
            )
        else:
            # Renormalize weights
            total = self.tfidf_weight + self.bert_weight
            hybrid_scores = (
                (self.tfidf_weight / total) * tfidf_scores +
                (self.bert_weight / total) * bert_scores
            )
        
        # Get rankings
        rankings = np.argsort(hybrid_scores)[::-1].tolist()
        
        return {
            "hybrid_scores": hybrid_scores.tolist(),
            "tfidf_scores": tfidf_scores.tolist(),
            "bert_scores": bert_scores.tolist(),
            "skill_scores": skill_scores.tolist() if skill_scores is not None else None,
            "rankings": rankings,
            "weights": {
                "tfidf": self.tfidf_weight,
                "bert": self.bert_weight,
                "skill_match": self.skill_match_weight
            }
        }
    
    def _compute_skill_overlap(
        self,
        candidate_skills: List[str],
        job_skills: List[str]
    ) -> float:
        """
        Compute skill overlap between candidate and job.
        
        Args:
            candidate_skills: List of candidate skills
            job_skills: List of job required skills
        
        Returns:
            Overlap score between 0 and 1
        """
        if not job_skills:
            return 0.0
        
        candidate_set = {s.lower().strip() for s in candidate_skills}
        job_set = {s.lower().strip() for s in job_skills}
        
        # Compute recall (what fraction of job skills does candidate have)
        overlap = candidate_set & job_set
        recall = len(overlap) / len(job_set) if job_set else 0.0
        
        return recall
    
    def compare_models(
        self,
        candidate_texts: List[str],
        job_texts: List[str]
    ) -> Dict:
        """
        Compare TF-IDF vs BERT rankings for analysis.
        
        Args:
            candidate_texts: List of candidate texts
            job_texts: List of job texts
        
        Returns:
            Comparison results dictionary
        """
        comparisons = []
        
        for i, cand_text in enumerate(candidate_texts):
            # Get scores from both models
            cand_tfidf = self.tfidf_model.transform([cand_text])
            cand_bert = self.bert_model.transform([cand_text])
            
            job_tfidf = self.tfidf_model.transform(job_texts)
            job_bert = self.bert_model.transform(job_texts)
            
            tfidf_scores = self.tfidf_model.compute_similarity(cand_tfidf, job_tfidf)
            bert_scores = self.bert_model.compute_similarity(cand_bert, job_bert)
            
            # Get rankings
            tfidf_ranking = np.argsort(tfidf_scores)[::-1].tolist()
            bert_ranking = np.argsort(bert_scores)[::-1].tolist()
            
            # Compute ranking correlation
            from scipy.stats import spearmanr
            correlation, _ = spearmanr(tfidf_ranking, bert_ranking)
            
            comparisons.append({
                "candidate_index": i,
                "tfidf_top5": tfidf_ranking[:5],
                "bert_top5": bert_ranking[:5],
                "ranking_correlation": correlation,
                "agreement_top5": len(set(tfidf_ranking[:5]) & set(bert_ranking[:5]))
            })
        
        return {
            "comparisons": comparisons,
            "avg_correlation": np.mean([c["ranking_correlation"] for c in comparisons]),
            "avg_top5_agreement": np.mean([c["agreement_top5"] for c in comparisons])
        }
    
    @property
    def model_config(self) -> Dict:
        """Get model configuration."""
        return {
            "tfidf_weight": self.tfidf_weight,
            "bert_weight": self.bert_weight,
            "skill_match_weight": self.skill_match_weight,
            "tfidf_vocab_size": self.tfidf_model.vocabulary_size,
            "bert_embedding_dim": self.bert_model.embedding_dimension
        }
