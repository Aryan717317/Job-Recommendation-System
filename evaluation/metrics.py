"""
Evaluation Metrics for Job Recommendation System

This module implements evaluation strategies for the recommendation system:
- Precision@K
- Recall@K
- NDCG@K
- MRR (Mean Reciprocal Rank)
- Hit Rate
- Manual relevance validation helpers

Provides comprehensive analysis for comparing TF-IDF vs BERT approaches.
"""

from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
import numpy as np
import logging
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import CandidateProfile, JobDescription, RecommendationResponse
from recommender import RecommendationEngine
from config import config

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics at a specific K value."""
    k: int
    precision: float
    recall: float
    f1_score: float
    ndcg: float
    hit_rate: float
    mrr: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "k": self.k,
            "precision@k": round(self.precision, 4),
            "recall@k": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "ndcg@k": round(self.ndcg, 4),
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4)
        }


@dataclass
class ModelComparisonResult:
    """Results from comparing two models."""
    model1_name: str
    model2_name: str
    model1_metrics: Dict[int, EvaluationMetrics]
    model2_metrics: Dict[int, EvaluationMetrics]
    ranking_correlation: float
    agreement_ratios: Dict[int, float]
    winner: str
    detailed_comparison: List[Dict] = field(default_factory=list)


class RecommendationEvaluator:
    """
    Evaluator for job recommendation quality.
    
    This class provides methods to:
    - Compute standard IR metrics (Precision, Recall, NDCG)
    - Compare different model approaches
    - Generate relevance judgments
    - Analyze recommendation quality
    
    Usage:
        evaluator = RecommendationEvaluator()
        
        # With ground truth labels
        metrics = evaluator.evaluate(
            recommendations=recs,
            ground_truth=relevant_job_ids
        )
        
        # Compare models
        comparison = evaluator.compare_models(
            engine_tfidf,
            engine_bert,
            candidates,
            ground_truth_map
        )
    """
    
    def __init__(self, k_values: Optional[List[int]] = None):
        """
        Initialize the evaluator.
        
        Args:
            k_values: List of K values for evaluation (default: [1, 3, 5, 10])
        """
        self.k_values = k_values or config.evaluation.k_values
    
    def precision_at_k(
        self,
        recommended_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K.
        
        Precision@K = |Relevant ∩ Top-K| / K
        
        Args:
            recommended_ids: Ordered list of recommended job IDs
            relevant_ids: Set of relevant job IDs
            k: Number of top recommendations to consider
        
        Returns:
            Precision score between 0 and 1
        """
        if k <= 0:
            return 0.0
        
        top_k = set(recommended_ids[:k])
        relevant_in_top_k = len(top_k & relevant_ids)
        
        return relevant_in_top_k / k
    
    def recall_at_k(
        self,
        recommended_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """
        Calculate Recall@K.
        
        Recall@K = |Relevant ∩ Top-K| / |Relevant|
        
        Args:
            recommended_ids: Ordered list of recommended job IDs
            relevant_ids: Set of relevant job IDs
            k: Number of top recommendations to consider
        
        Returns:
            Recall score between 0 and 1
        """
        if not relevant_ids:
            return 0.0
        
        top_k = set(recommended_ids[:k])
        relevant_in_top_k = len(top_k & relevant_ids)
        
        return relevant_in_top_k / len(relevant_ids)
    
    def f1_at_k(
        self,
        recommended_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """
        Calculate F1@K (harmonic mean of Precision and Recall).
        
        Args:
            recommended_ids: Ordered list of recommended job IDs
            relevant_ids: Set of relevant job IDs
            k: Number of top recommendations to consider
        
        Returns:
            F1 score between 0 and 1
        """
        precision = self.precision_at_k(recommended_ids, relevant_ids, k)
        recall = self.recall_at_k(recommended_ids, relevant_ids, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg_at_k(
        self,
        recommended_ids: List[str],
        relevant_ids: Set[str],
        k: int,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.
        
        NDCG accounts for the position of relevant items in the ranking.
        
        Args:
            recommended_ids: Ordered list of recommended job IDs
            relevant_ids: Set of relevant job IDs
            k: Number of top recommendations to consider
            relevance_scores: Optional dict of job_id -> relevance score
        
        Returns:
            NDCG score between 0 and 1
        """
        if not relevant_ids or k <= 0:
            return 0.0
        
        # DCG: Discounted Cumulative Gain
        dcg = 0.0
        for i, job_id in enumerate(recommended_ids[:k]):
            if job_id in relevant_ids:
                rel = relevance_scores.get(job_id, 1.0) if relevance_scores else 1.0
                dcg += rel / np.log2(i + 2)  # +2 because i is 0-indexed
        
        # IDCG: Ideal DCG (all relevant items at top)
        ideal_relevances = []
        for job_id in relevant_ids:
            rel = relevance_scores.get(job_id, 1.0) if relevance_scores else 1.0
            ideal_relevances.append(rel)
        
        ideal_relevances.sort(reverse=True)
        
        idcg = 0.0
        for i, rel in enumerate(ideal_relevances[:k]):
            idcg += rel / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def hit_rate(
        self,
        recommended_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """
        Calculate Hit Rate@K (whether any relevant item is in Top-K).
        
        Args:
            recommended_ids: Ordered list of recommended job IDs
            relevant_ids: Set of relevant job IDs
            k: Number of top recommendations to consider
        
        Returns:
            1.0 if hit, 0.0 otherwise
        """
        top_k = set(recommended_ids[:k])
        return 1.0 if top_k & relevant_ids else 0.0
    
    def mean_reciprocal_rank(
        self,
        recommended_ids: List[str],
        relevant_ids: Set[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        MRR = 1 / rank_of_first_relevant_item
        
        Args:
            recommended_ids: Ordered list of recommended job IDs
            relevant_ids: Set of relevant job IDs
        
        Returns:
            MRR score between 0 and 1
        """
        for i, job_id in enumerate(recommended_ids):
            if job_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0
    
    def evaluate(
        self,
        recommendations: RecommendationResponse,
        ground_truth: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> Dict[int, EvaluationMetrics]:
        """
        Evaluate recommendations against ground truth.
        
        Args:
            recommendations: RecommendationResponse object
            ground_truth: Set of relevant job IDs
            relevance_scores: Optional relevance scores per job
        
        Returns:
            Dictionary of K -> EvaluationMetrics
        """
        recommended_ids = [rec.job.id for rec in recommendations.recommendations]
        
        results = {}
        for k in self.k_values:
            metrics = EvaluationMetrics(
                k=k,
                precision=self.precision_at_k(recommended_ids, ground_truth, k),
                recall=self.recall_at_k(recommended_ids, ground_truth, k),
                f1_score=self.f1_at_k(recommended_ids, ground_truth, k),
                ndcg=self.ndcg_at_k(recommended_ids, ground_truth, k, relevance_scores),
                hit_rate=self.hit_rate(recommended_ids, ground_truth, k),
                mrr=self.mean_reciprocal_rank(recommended_ids, ground_truth)
            )
            results[k] = metrics
        
        return results
    
    def evaluate_batch(
        self,
        recommendations_list: List[RecommendationResponse],
        ground_truth_map: Dict[str, Set[str]]
    ) -> Dict[int, EvaluationMetrics]:
        """
        Evaluate multiple recommendations and compute average metrics.
        
        Args:
            recommendations_list: List of RecommendationResponse objects
            ground_truth_map: Dict of candidate_id -> relevant_job_ids
        
        Returns:
            Dictionary of K -> average EvaluationMetrics
        """
        all_metrics = defaultdict(list)
        
        for recommendations in recommendations_list:
            candidate_id = recommendations.candidate_id
            ground_truth = ground_truth_map.get(candidate_id, set())
            
            if ground_truth:
                metrics = self.evaluate(recommendations, ground_truth)
                for k, m in metrics.items():
                    all_metrics[k].append(m)
        
        # Average metrics
        avg_results = {}
        for k, metrics_list in all_metrics.items():
            if metrics_list:
                avg_results[k] = EvaluationMetrics(
                    k=k,
                    precision=np.mean([m.precision for m in metrics_list]),
                    recall=np.mean([m.recall for m in metrics_list]),
                    f1_score=np.mean([m.f1_score for m in metrics_list]),
                    ndcg=np.mean([m.ndcg for m in metrics_list]),
                    hit_rate=np.mean([m.hit_rate for m in metrics_list]),
                    mrr=np.mean([m.mrr for m in metrics_list])
                )
        
        return avg_results
    
    def compare_models(
        self,
        engine1: RecommendationEngine,
        engine2: RecommendationEngine,
        candidates: List[CandidateProfile],
        ground_truth_map: Dict[str, Set[str]],
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ) -> ModelComparisonResult:
        """
        Compare two recommendation engines.
        
        Args:
            engine1: First recommendation engine
            engine2: Second recommendation engine
            candidates: List of candidates to evaluate on
            ground_truth_map: Dict of candidate_id -> relevant_job_ids
            model1_name: Name for first model
            model2_name: Name for second model
        
        Returns:
            ModelComparisonResult with detailed comparison
        """
        logger.info(f"Comparing {model1_name} vs {model2_name}...")
        
        # Generate recommendations from both models
        recs1 = [engine1.recommend(c, top_n=max(self.k_values)) for c in candidates]
        recs2 = [engine2.recommend(c, top_n=max(self.k_values)) for c in candidates]
        
        # Evaluate both
        metrics1 = self.evaluate_batch(recs1, ground_truth_map)
        metrics2 = self.evaluate_batch(recs2, ground_truth_map)
        
        # Calculate ranking agreement
        agreement_ratios = {}
        correlations = []
        detailed = []
        
        for i, (r1, r2) in enumerate(zip(recs1, recs2)):
            ids1 = [rec.job.id for rec in r1.recommendations]
            ids2 = [rec.job.id for rec in r2.recommendations]
            
            for k in self.k_values:
                top_k_1 = set(ids1[:k])
                top_k_2 = set(ids2[:k])
                agreement = len(top_k_1 & top_k_2) / k if k > 0 else 0
                
                if k not in agreement_ratios:
                    agreement_ratios[k] = []
                agreement_ratios[k].append(agreement)
            
            # Compute ranking correlation
            if len(ids1) >= 5 and len(ids2) >= 5:
                # Create rank mappings
                rank1 = {id_: i for i, id_ in enumerate(ids1)}
                rank2 = {id_: i for i, id_ in enumerate(ids2)}
                
                common = set(ids1) & set(ids2)
                if len(common) >= 3:
                    from scipy.stats import spearmanr
                    ranks_1 = [rank1[id_] for id_ in common]
                    ranks_2 = [rank2[id_] for id_ in common]
                    corr, _ = spearmanr(ranks_1, ranks_2)
                    correlations.append(corr if not np.isnan(corr) else 0)
            
            detailed.append({
                "candidate_id": r1.candidate_id,
                "model1_top5": ids1[:5],
                "model2_top5": ids2[:5]
            })
        
        avg_agreement = {k: np.mean(v) for k, v in agreement_ratios.items()}
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        # Determine winner based on NDCG@5
        k5_ndcg1 = metrics1.get(5, EvaluationMetrics(5, 0, 0, 0, 0, 0, 0)).ndcg
        k5_ndcg2 = metrics2.get(5, EvaluationMetrics(5, 0, 0, 0, 0, 0, 0)).ndcg
        
        if k5_ndcg1 > k5_ndcg2 * 1.05:  # 5% threshold
            winner = model1_name
        elif k5_ndcg2 > k5_ndcg1 * 1.05:
            winner = model2_name
        else:
            winner = "Tie"
        
        result = ModelComparisonResult(
            model1_name=model1_name,
            model2_name=model2_name,
            model1_metrics=metrics1,
            model2_metrics=metrics2,
            ranking_correlation=avg_correlation,
            agreement_ratios=avg_agreement,
            winner=winner,
            detailed_comparison=detailed
        )
        
        logger.info(f"Comparison complete. Winner: {winner}")
        
        return result
    
    def generate_synthetic_ground_truth(
        self,
        candidates: List[CandidateProfile],
        jobs: List[JobDescription],
        skill_match_threshold: float = 0.3
    ) -> Dict[str, Set[str]]:
        """
        Generate synthetic ground truth based on skill matching.
        
        This is useful for testing when manual labels aren't available.
        Jobs are considered relevant if skill overlap exceeds threshold.
        
        Args:
            candidates: List of candidates
            jobs: List of jobs
            skill_match_threshold: Minimum skill overlap ratio
        
        Returns:
            Dict of candidate_id -> relevant_job_ids
        """
        ground_truth = {}
        
        for candidate in candidates:
            candidate_skills = set(s.lower() for s in candidate.skill_names)
            relevant_jobs = set()
            
            for job in jobs:
                job_skills = set(s.lower() for s in job.all_skill_names)
                
                if not job_skills:
                    continue
                
                overlap = len(candidate_skills & job_skills)
                ratio = overlap / len(job_skills)
                
                if ratio >= skill_match_threshold:
                    relevant_jobs.add(job.id)
            
            ground_truth[candidate.id] = relevant_jobs
        
        return ground_truth
    
    def print_evaluation_report(
        self,
        metrics: Dict[int, EvaluationMetrics],
        title: str = "Evaluation Report"
    ) -> str:
        """
        Generate a formatted evaluation report.
        
        Args:
            metrics: Dictionary of K -> EvaluationMetrics
            title: Report title
        
        Returns:
            Formatted report string
        """
        lines = [
            f"\n{'='*60}",
            f" {title}",
            f"{'='*60}",
            f"{'K':>5} {'Prec@K':>10} {'Rec@K':>10} {'F1':>10} {'NDCG':>10} {'MRR':>10}",
            f"{'-'*60}"
        ]
        
        for k in sorted(metrics.keys()):
            m = metrics[k]
            lines.append(
                f"{k:>5} {m.precision:>10.4f} {m.recall:>10.4f} "
                f"{m.f1_score:>10.4f} {m.ndcg:>10.4f} {m.mrr:>10.4f}"
            )
        
        lines.append(f"{'='*60}\n")
        
        report = "\n".join(lines)
        print(report)
        return report
    
    def print_comparison_report(
        self,
        comparison: ModelComparisonResult
    ) -> str:
        """
        Generate a formatted model comparison report.
        
        Args:
            comparison: ModelComparisonResult object
        
        Returns:
            Formatted report string
        """
        lines = [
            f"\n{'='*70}",
            f" Model Comparison: {comparison.model1_name} vs {comparison.model2_name}",
            f"{'='*70}",
            "",
            f"Ranking Correlation: {comparison.ranking_correlation:.4f}",
            f"Winner: {comparison.winner}",
            "",
            "Top-K Agreement Ratios:",
        ]
        
        for k, ratio in sorted(comparison.agreement_ratios.items()):
            lines.append(f"  K={k}: {ratio:.2%}")
        
        lines.append("")
        lines.append(f"{'K':>5} | {comparison.model1_name:^25} | {comparison.model2_name:^25}")
        lines.append(f"      | {'Prec':>8} {'NDCG':>8} {'MRR':>7} | {'Prec':>8} {'NDCG':>8} {'MRR':>7}")
        lines.append(f"{'-'*70}")
        
        for k in sorted(comparison.model1_metrics.keys()):
            m1 = comparison.model1_metrics[k]
            m2 = comparison.model2_metrics[k]
            lines.append(
                f"{k:>5} | {m1.precision:>8.4f} {m1.ndcg:>8.4f} {m1.mrr:>7.4f} | "
                f"{m2.precision:>8.4f} {m2.ndcg:>8.4f} {m2.mrr:>7.4f}"
            )
        
        lines.append(f"{'='*70}\n")
        
        report = "\n".join(lines)
        print(report)
        return report


def evaluate_recommendations(
    engine: RecommendationEngine,
    candidates: List[CandidateProfile],
    ground_truth_map: Dict[str, Set[str]],
    k_values: Optional[List[int]] = None
) -> Dict[int, EvaluationMetrics]:
    """
    Convenience function to evaluate recommendations.
    
    Args:
        engine: Fitted recommendation engine
        candidates: List of candidates
        ground_truth_map: Ground truth labels
        k_values: K values for evaluation
    
    Returns:
        Evaluation metrics
    """
    evaluator = RecommendationEvaluator(k_values=k_values)
    recommendations = [engine.recommend(c, top_n=20) for c in candidates]
    return evaluator.evaluate_batch(recommendations, ground_truth_map)
