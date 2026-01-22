"""
Tests for Job Recommendation System

This module contains unit tests for all system components:
- Data schemas and loading
- Text preprocessing
- TF-IDF and BERT models
- Recommendation engine
- Evaluation metrics
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataSchemas:
    """Tests for data schemas."""
    
    def test_candidate_profile_creation(self):
        """Test creating a candidate profile."""
        from data import CandidateProfile, Skill
        
        skills = [
            Skill(name="python", proficiency=4),
            Skill(name="machine learning", proficiency=3)
        ]
        
        candidate = CandidateProfile(
            name="Test User",
            skills=skills,
            summary="Experienced developer",
            years_of_experience=5.0
        )
        
        assert candidate.name == "Test User"
        assert len(candidate.skills) == 2
        assert candidate.years_of_experience == 5.0
        assert "python" in candidate.skill_names
    
    def test_candidate_combined_text(self):
        """Test combined text extraction from candidate."""
        from data import CandidateProfile, Skill
        
        candidate = CandidateProfile(
            name="Test User",
            skills=[Skill(name="python"), Skill(name="sql")],
            summary="Data scientist with ML experience",
            preferred_roles=["Data Scientist"]
        )
        
        text = candidate.get_combined_text()
        
        assert "python" in text.lower()
        assert "sql" in text.lower()
        assert "data scientist" in text.lower()
    
    def test_job_description_creation(self):
        """Test creating a job description."""
        from data import JobDescription, SkillRequirement, ExperienceLevel
        
        job = JobDescription(
            title="Senior Developer",
            company="TechCorp",
            description="Build amazing products",
            required_skills=[
                SkillRequirement(name="python", required=True),
                SkillRequirement(name="django", required=False)
            ],
            experience_level=ExperienceLevel.SENIOR
        )
        
        assert job.title == "Senior Developer"
        assert len(job.required_skills) == 2
        assert "python" in job.required_skill_names
    
    def test_skill_normalization(self):
        """Test skill name normalization."""
        from data import Skill
        
        skill = Skill(name="  PyThOn  ")
        assert skill.name == "python"


class TestMockData:
    """Tests for mock data generation."""
    
    def test_generate_candidates(self):
        """Test generating candidate profiles."""
        from data import generate_candidate_profiles
        
        candidates = generate_candidate_profiles(10)
        
        assert len(candidates) == 10
        for candidate in candidates:
            assert candidate.name
            assert len(candidate.skills) > 0
    
    def test_generate_jobs(self):
        """Test generating job descriptions."""
        from data import generate_job_descriptions
        
        jobs = generate_job_descriptions(20)
        
        assert len(jobs) == 20
        for job in jobs:
            assert job.title
            assert job.company
            assert job.description
    
    def test_load_mock_data(self):
        """Test loading mock data."""
        from data import load_mock_data
        
        candidates, jobs = load_mock_data(5, 10)
        
        assert len(candidates) == 5
        assert len(jobs) == 10


class TestPreprocessing:
    """Tests for text preprocessing."""
    
    def test_text_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        from preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        assert preprocessor.lowercase is True
        assert preprocessor.remove_stopwords is True
    
    def test_skill_normalization(self):
        """Test skill abbreviation normalization."""
        from preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        normalized = preprocessor.normalize_skills("Experience with ML and NLP")
        
        assert "machine learning" in normalized.lower()
        assert "natural language processing" in normalized.lower()
    
    def test_text_processing(self):
        """Test full text processing pipeline."""
        from preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        text = "Python developer with 5+ years of experience in ML"
        processed = preprocessor.process(text)
        
        assert "python" in processed.lower()
        # Stop words should be removed
        assert len(processed.split()) < len(text.split())
    
    def test_tokenization(self):
        """Test tokenization."""
        from preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        tokens = preprocessor.tokenize("Python and machine learning")
        
        assert isinstance(tokens, list)
        assert "python" in [t.lower() for t in tokens]


class TestTFIDFModel:
    """Tests for TF-IDF model."""
    
    def test_tfidf_initialization(self):
        """Test TF-IDF model initialization."""
        from models import TFIDFModel
        
        model = TFIDFModel()
        assert model.name == "tfidf"
        assert not model.is_fitted
    
    def test_tfidf_fit(self):
        """Test fitting TF-IDF model."""
        from models import TFIDFModel
        
        model = TFIDFModel()
        documents = [
            "python developer machine learning",
            "java backend developer",
            "frontend react javascript"
        ]
        
        model.fit(documents)
        
        assert model.is_fitted
        assert model.vocabulary_size > 0
    
    def test_tfidf_transform(self):
        """Test TF-IDF transformation."""
        from models import TFIDFModel
        
        model = TFIDFModel()
        documents = ["python machine learning", "java backend"]
        
        vectors = model.fit_transform(documents)
        
        assert vectors.shape[0] == 2
        assert vectors.shape[1] > 0
    
    def test_tfidf_similarity(self):
        """Test TF-IDF similarity computation."""
        from models import TFIDFModel
        
        model = TFIDFModel()
        documents = [
            "python machine learning data science",
            "python ml deep learning",
            "java spring boot backend"
        ]
        
        vectors = model.fit_transform(documents)
        
        # First two documents should be more similar
        sim_0_1 = model.compute_similarity(vectors[0], vectors[1].reshape(1, -1))[0]
        sim_0_2 = model.compute_similarity(vectors[0], vectors[2].reshape(1, -1))[0]
        
        assert sim_0_1 > sim_0_2


class TestRecommendationEngine:
    """Tests for recommendation engine."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        from recommender import RecommendationEngine
        
        engine = RecommendationEngine(model_type="tfidf")
        assert engine.model_type == "tfidf"
    
    def test_engine_fit(self):
        """Test fitting the engine."""
        from recommender import RecommendationEngine
        from data import generate_job_descriptions
        
        engine = RecommendationEngine(model_type="tfidf")
        jobs = generate_job_descriptions(10)
        
        engine.fit(jobs)
        
        assert engine._is_fitted
    
    def test_engine_recommend(self):
        """Test generating recommendations."""
        from recommender import RecommendationEngine
        from data import generate_candidate_profiles, generate_job_descriptions
        
        engine = RecommendationEngine(model_type="tfidf")
        jobs = generate_job_descriptions(10)
        candidates = generate_candidate_profiles(3)
        
        engine.fit(jobs)
        
        recommendations = engine.recommend(candidates[0], top_n=5)
        
        assert len(recommendations.recommendations) <= 5
        assert recommendations.model_used == "tfidf"
        
        # Check ranking order
        for i, rec in enumerate(recommendations.recommendations):
            assert rec.rank == i + 1
    
    def test_recommendation_scores(self):
        """Test that recommendation scores are valid."""
        from recommender import RecommendationEngine
        from data import generate_candidate_profiles, generate_job_descriptions
        
        engine = RecommendationEngine(model_type="tfidf")
        jobs = generate_job_descriptions(10)
        candidates = generate_candidate_profiles(1)
        
        engine.fit(jobs)
        recommendations = engine.recommend(candidates[0], top_n=5)
        
        for rec in recommendations.recommendations:
            assert 0 <= rec.similarity_score <= 1


class TestEvaluation:
    """Tests for evaluation metrics."""
    
    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        from evaluation import RecommendationEvaluator
        
        evaluator = RecommendationEvaluator()
        
        recommended = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "f"}
        
        # At K=3: {a, b, c} ∩ {a, c, f} = {a, c} -> 2/3
        precision = evaluator.precision_at_k(recommended, relevant, k=3)
        
        assert abs(precision - 2/3) < 0.01
    
    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        from evaluation import RecommendationEvaluator
        
        evaluator = RecommendationEvaluator()
        
        recommended = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "f"}
        
        # At K=3: {a, b, c} ∩ {a, c, f} = {a, c} -> 2/3
        recall = evaluator.recall_at_k(recommended, relevant, k=3)
        
        assert abs(recall - 2/3) < 0.01
    
    def test_ndcg_at_k(self):
        """Test NDCG@K calculation."""
        from evaluation import RecommendationEvaluator
        
        evaluator = RecommendationEvaluator()
        
        # Perfect ranking
        recommended = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        
        ndcg = evaluator.ndcg_at_k(recommended, relevant, k=3)
        
        assert abs(ndcg - 1.0) < 0.01
    
    def test_mrr(self):
        """Test Mean Reciprocal Rank calculation."""
        from evaluation import RecommendationEvaluator
        
        evaluator = RecommendationEvaluator()
        
        # First relevant at position 2
        recommended = ["x", "a", "b"]
        relevant = {"a", "b"}
        
        mrr = evaluator.mean_reciprocal_rank(recommended, relevant)
        
        assert abs(mrr - 0.5) < 0.01  # 1/2
    
    def test_generate_synthetic_ground_truth(self):
        """Test synthetic ground truth generation."""
        from evaluation import RecommendationEvaluator
        from data import generate_candidate_profiles, generate_job_descriptions
        
        evaluator = RecommendationEvaluator()
        candidates = generate_candidate_profiles(5)
        jobs = generate_job_descriptions(10)
        
        ground_truth = evaluator.generate_synthetic_ground_truth(
            candidates, jobs, skill_match_threshold=0.2
        )
        
        assert len(ground_truth) == 5
        for cid, relevant_jobs in ground_truth.items():
            assert isinstance(relevant_jobs, set)


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data to recommendations."""
        from data import load_mock_data
        from recommender import create_engine
        from evaluation import RecommendationEvaluator
        
        # Load data
        candidates, jobs = load_mock_data(5, 20)
        
        # Create and fit engine
        engine = create_engine(model_type="tfidf")
        engine.fit(jobs)
        
        # Generate recommendations
        recommendations = engine.recommend(candidates[0], top_n=5)
        
        # Evaluate
        evaluator = RecommendationEvaluator()
        ground_truth = evaluator.generate_synthetic_ground_truth(
            [candidates[0]], jobs
        )
        
        metrics = evaluator.evaluate(
            recommendations, 
            ground_truth.get(candidates[0].id, set())
        )
        
        assert 5 in metrics
        assert 0 <= metrics[5].precision <= 1


def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    run_tests()
