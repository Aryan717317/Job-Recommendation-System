"""
Recommendation Engine for Job Recommendation System

This module implements the core recommendation logic that:
- Accepts candidate profiles
- Computes similarity scores against all jobs
- Ranks and returns Top-N recommendations

Features:
- Model-agnostic design
- Easy swapping between TF-IDF, BERT, and Hybrid
- Detailed scoring breakdown
- Experience-aware ranking (optional)
"""

from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
import time
import logging
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (
    CandidateProfile, 
    JobDescription, 
    RecommendationResult,
    RecommendationResponse
)
from models import TFIDFModel, BERTModel, HybridModel, BaseVectorizationModel
from preprocessing import TextPreprocessor, CandidateTextExtractor, JobTextExtractor
from config import config

logger = logging.getLogger(__name__)


@dataclass
class ScoringBreakdown:
    """Detailed breakdown of scoring components."""
    total_score: float
    tfidf_score: Optional[float] = None
    bert_score: Optional[float] = None
    skill_match_score: Optional[float] = None
    experience_score: Optional[float] = None
    matched_skills: List[str] = None
    
    def __post_init__(self):
        if self.matched_skills is None:
            self.matched_skills = []


class RecommendationEngine:
    """
    Core recommendation engine for job-candidate matching.
    
    This engine supports multiple vectorization strategies and provides
    a unified interface for generating job recommendations.
    
    Supported models:
    - "tfidf": TF-IDF based keyword matching
    - "bert": BERT-based semantic matching
    - "hybrid": Weighted combination of TF-IDF and BERT
    
    Usage:
        engine = RecommendationEngine(model_type="hybrid")
        engine.fit(jobs)
        
        recommendations = engine.recommend(candidate, top_n=10)
    """
    
    def __init__(
        self,
        model_type: str = "tfidf",
        top_n: int = 10,
        min_similarity_threshold: float = 0.1,
        hybrid_tfidf_weight: float = 0.4,
        hybrid_bert_weight: float = 0.6,
        use_experience_weighting: bool = False
    ):
        """
        Initialize the recommendation engine.
        
        Args:
            model_type: Type of model to use ("tfidf", "bert", "hybrid")
            top_n: Default number of recommendations to return
            min_similarity_threshold: Minimum similarity score to include
            hybrid_tfidf_weight: TF-IDF weight for hybrid model
            hybrid_bert_weight: BERT weight for hybrid model
            use_experience_weighting: Whether to factor in experience level
        """
        self.model_type = model_type.lower()
        self.default_top_n = top_n or config.recommender.default_top_n
        self.min_similarity_threshold = (
            min_similarity_threshold or config.recommender.min_similarity_threshold
        )
        self.hybrid_tfidf_weight = hybrid_tfidf_weight
        self.hybrid_bert_weight = hybrid_bert_weight
        self.use_experience_weighting = use_experience_weighting
        
        # Initialize model
        self.model = self._create_model()
        
        # Text extractors
        self.preprocessor = TextPreprocessor()
        self.candidate_extractor = CandidateTextExtractor(self.preprocessor)
        self.job_extractor = JobTextExtractor(self.preprocessor)
        
        # Store jobs for recommendations
        self._jobs: List[JobDescription] = []
        self._job_texts: List[str] = []
        self._job_vectors = None
        self._is_fitted = False
    
    def _create_model(self) -> BaseVectorizationModel:
        """Create the appropriate model based on model_type."""
        if self.model_type == "tfidf":
            return TFIDFModel()
        elif self.model_type == "bert":
            return BERTModel()
        elif self.model_type == "hybrid":
            return HybridModel(
                tfidf_weight=self.hybrid_tfidf_weight,
                bert_weight=self.hybrid_bert_weight
            )
        else:
            raise ValueError(
                f"Unknown model type: {self.model_type}. "
                f"Choose from: 'tfidf', 'bert', 'hybrid'"
            )
    
    def fit(self, jobs: List[JobDescription]) -> 'RecommendationEngine':
        """
        Fit the engine on a list of job descriptions.
        
        This prepares the job vectors for efficient recommendation.
        
        Args:
            jobs: List of JobDescription objects
        
        Returns:
            Self for method chaining
        """
        if not jobs:
            raise ValueError("Cannot fit on empty job list")
        
        logger.info(f"Fitting recommendation engine with {len(jobs)} jobs...")
        
        self._jobs = jobs
        self._job_texts = [self.job_extractor.extract_raw_text(job) for job in jobs]
        
        # Fit model and transform job texts
        if self.model_type == "hybrid":
            self.model.fit(self._job_texts)
            tfidf_vecs, bert_vecs = self.model.transform(self._job_texts)
            self._job_vectors = (tfidf_vecs, bert_vecs)
        else:
            self._job_vectors = self.model.fit_transform(self._job_texts)
        
        self._is_fitted = True
        
        logger.info(f"Recommendation engine fitted successfully")
        
        return self
    
    def recommend(
        self,
        candidate: CandidateProfile,
        top_n: Optional[int] = None,
        include_scores: bool = True,
        model_type: Optional[str] = None
    ) -> RecommendationResponse:
        """
        Generate job recommendations for a candidate.
        
        Args:
            candidate: CandidateProfile object
            top_n: Number of recommendations (uses default if None)
            include_scores: Whether to include detailed scores
            model_type: Override the default model type for this request
        
        Returns:
            RecommendationResponse with ranked jobs and scores
        """
        if not self._is_fitted:
            raise RuntimeError("Engine must be fitted before making recommendations")
        
        start_time = time.time()
        
        n = top_n or self.default_top_n
        use_model_type = model_type or self.model_type
        
        # Extract candidate text
        candidate_text = self.candidate_extractor.extract_raw_text(candidate)
        
        # Compute similarities based on model type
        if use_model_type == "hybrid":
            scores_data = self.model.get_recommendation_scores(
                candidate_text,
                self._job_texts,
                candidate_skills=candidate.skill_names,
                job_skills_list=[job.all_skill_names for job in self._jobs]
            )
            similarities = np.array(scores_data["hybrid_scores"])
            tfidf_scores = np.array(scores_data["tfidf_scores"])
            bert_scores = np.array(scores_data["bert_scores"])
        else:
            # Transform candidate
            candidate_vector = self.model.transform([candidate_text])
            
            # Compute similarities
            similarities = self.model.compute_similarity(
                candidate_vector,
                self._job_vectors
            )
            tfidf_scores = similarities if use_model_type == "tfidf" else None
            bert_scores = similarities if use_model_type == "bert" else None
        
        # Apply experience weighting if enabled
        if self.use_experience_weighting:
            experience_weights = self._compute_experience_weights(candidate)
            similarities = similarities * experience_weights
        
        # Get top N indices
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Build recommendations
        recommendations = []
        rank = 1
        
        for idx in sorted_indices:
            score = float(similarities[idx])
            
            # Skip below threshold
            if score < self.min_similarity_threshold:
                continue
            
            if rank > n:
                break
            
            job = self._jobs[idx]
            
            # Calculate skill match details
            matched_skills = self._get_matched_skills(candidate, job)
            skill_ratio = (
                len(matched_skills) / len(job.required_skill_names)
                if job.required_skill_names else 0.0
            )
            # Cap at 1.0 (can exceed if matched skills include preferred skills)
            skill_ratio = min(skill_ratio, 1.0)
            
            rec = RecommendationResult(
                job=job,
                similarity_score=score,
                tfidf_score=float(tfidf_scores[idx]) if tfidf_scores is not None else None,
                bert_score=float(bert_scores[idx]) if bert_scores is not None else None,
                skill_match_ratio=skill_ratio,
                matched_skills=matched_skills,
                rank=rank
            )
            
            recommendations.append(rec)
            rank += 1
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        response = RecommendationResponse(
            candidate_id=candidate.id,
            recommendations=recommendations,
            model_used=use_model_type,
            processing_time_ms=processing_time,
            total_jobs_considered=len(self._jobs)
        )
        
        logger.info(
            f"Generated {len(recommendations)} recommendations for candidate "
            f"{candidate.id} in {processing_time:.2f}ms"
        )
        
        return response
    
    def recommend_batch(
        self,
        candidates: List[CandidateProfile],
        top_n: Optional[int] = None
    ) -> List[RecommendationResponse]:
        """
        Generate recommendations for multiple candidates.
        
        Args:
            candidates: List of CandidateProfile objects
            top_n: Number of recommendations per candidate
        
        Returns:
            List of RecommendationResponse objects
        """
        return [self.recommend(candidate, top_n) for candidate in candidates]
    
    def get_similar_jobs(
        self,
        job: JobDescription,
        top_n: int = 5
    ) -> List[Tuple[JobDescription, float]]:
        """
        Find jobs similar to a given job.
        
        Args:
            job: Reference JobDescription
            top_n: Number of similar jobs to return
        
        Returns:
            List of (job, similarity_score) tuples
        """
        if not self._is_fitted:
            raise RuntimeError("Engine must be fitted first")
        
        job_text = self.job_extractor.extract_raw_text(job)
        job_vector = self.model.transform([job_text])
        
        if self.model_type == "hybrid":
            job_tfidf, job_bert = job_vector
            job_vecs_tfidf, job_vecs_bert = self._job_vectors
            
            tfidf_sim = self.model.tfidf_model.compute_similarity(job_tfidf, job_vecs_tfidf)
            bert_sim = self.model.bert_model.compute_similarity(job_bert, job_vecs_bert)
            
            similarities = (
                self.hybrid_tfidf_weight * tfidf_sim +
                self.hybrid_bert_weight * bert_sim
            )
        else:
            similarities = self.model.compute_similarity(job_vector, self._job_vectors)
        
        # Get top N (excluding the job itself if present)
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in sorted_indices:
            if self._jobs[idx].id != job.id:
                results.append((self._jobs[idx], float(similarities[idx])))
                if len(results) >= top_n:
                    break
        
        return results
    
    def _get_matched_skills(
        self,
        candidate: CandidateProfile,
        job: JobDescription
    ) -> List[str]:
        """Get list of skills that match between candidate and job."""
        candidate_skills = {s.lower() for s in candidate.skill_names}
        job_skills = {s.lower() for s in job.all_skill_names}
        
        matched = candidate_skills & job_skills
        return list(matched)
    
    def _compute_experience_weights(
        self,
        candidate: CandidateProfile
    ) -> np.ndarray:
        """
        Compute experience-based weights for job matching.
        
        Jobs closer to candidate's experience level get higher weights.
        """
        candidate_years = candidate.years_of_experience or 0
        
        weights = []
        for job in self._jobs:
            min_years = job.min_years_experience or 0
            max_years = job.max_years_experience or 20
            
            # Perfect fit
            if min_years <= candidate_years <= max_years:
                weight = 1.0
            # Slightly overqualified
            elif candidate_years > max_years:
                diff = candidate_years - max_years
                weight = max(0.7, 1.0 - (diff * 0.1))
            # Underqualified
            else:
                diff = min_years - candidate_years
                weight = max(0.5, 1.0 - (diff * 0.15))
            
            weights.append(weight)
        
        return np.array(weights)
    
    def switch_model(self, model_type: str) -> None:
        """
        Switch to a different model type.
        
        Note: Requires refitting the engine.
        
        Args:
            model_type: New model type ("tfidf", "bert", "hybrid")
        """
        self.model_type = model_type.lower()
        self.model = self._create_model()
        
        if self._jobs:
            self.fit(self._jobs)
        
        logger.info(f"Switched to model: {model_type}")
    
    def get_scoring_breakdown(
        self,
        candidate: CandidateProfile,
        job: JobDescription
    ) -> ScoringBreakdown:
        """
        Get detailed scoring breakdown for a specific candidate-job pair.
        
        Args:
            candidate: CandidateProfile object
            job: JobDescription object
        
        Returns:
            ScoringBreakdown with all score components
        """
        candidate_text = self.candidate_extractor.extract_raw_text(candidate)
        job_text = self.job_extractor.extract_raw_text(job)
        
        if self.model_type == "hybrid":
            scores = self.model.get_recommendation_scores(
                candidate_text,
                [job_text],
                candidate_skills=candidate.skill_names,
                job_skills_list=[job.all_skill_names]
            )
            
            tfidf_score = scores["tfidf_scores"][0]
            bert_score = scores["bert_scores"][0]
            total_score = scores["hybrid_scores"][0]
        else:
            candidate_vec = self.model.transform([candidate_text])
            job_vec = self.model.transform([job_text])
            
            sim = self.model.compute_similarity(candidate_vec, job_vec)
            total_score = float(sim[0]) if hasattr(sim, '__len__') else float(sim)
            
            tfidf_score = total_score if self.model_type == "tfidf" else None
            bert_score = total_score if self.model_type == "bert" else None
        
        matched_skills = self._get_matched_skills(candidate, job)
        skill_score = (
            len(matched_skills) / len(job.required_skill_names)
            if job.required_skill_names else 0.0
        )
        
        return ScoringBreakdown(
            total_score=total_score,
            tfidf_score=tfidf_score,
            bert_score=bert_score,
            skill_match_score=skill_score,
            matched_skills=matched_skills
        )
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_type": self.model_type,
            "is_fitted": self._is_fitted,
            "num_jobs": len(self._jobs),
            "default_top_n": self.default_top_n,
            "min_similarity_threshold": self.min_similarity_threshold,
            "use_experience_weighting": self.use_experience_weighting
        }


def create_engine(
    model_type: str = "tfidf",
    **kwargs
) -> RecommendationEngine:
    """
    Factory function to create a recommendation engine.
    
    Args:
        model_type: Type of model ("tfidf", "bert", "hybrid")
        **kwargs: Additional arguments for RecommendationEngine
    
    Returns:
        Configured RecommendationEngine instance
    """
    return RecommendationEngine(model_type=model_type, **kwargs)
