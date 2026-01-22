"""
Configuration Module for Job Recommendation System

This module centralizes all configuration parameters for the system,
including model settings, preprocessing options, and API configurations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing pipeline."""
    lowercase: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    remove_punctuation: bool = True
    min_token_length: int = 2
    
    # Skill normalization mappings
    skill_mappings: Dict[str, str] = field(default_factory=lambda: {
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "dl": "deep learning",
        "nlp": "natural language processing",
        "cv": "computer vision",
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "sql": "structured query language",
        "nosql": "non-relational database",
        "aws": "amazon web services",
        "gcp": "google cloud platform",
        "k8s": "kubernetes",
        "ci/cd": "continuous integration continuous deployment",
        "oop": "object oriented programming",
        "api": "application programming interface",
        "rest": "representational state transfer",
        "db": "database",
        "ui": "user interface",
        "ux": "user experience",
        "qa": "quality assurance",
        "pm": "project management",
        "ml ops": "machine learning operations",
        "devops": "development operations",
    })


@dataclass
class TFIDFConfig:
    """Configuration for TF-IDF vectorization."""
    max_features: int = 5000
    ngram_range: tuple = (1, 2)
    min_df: int = 1
    max_df: float = 0.95
    sublinear_tf: bool = True


@dataclass
class BERTConfig:
    """Configuration for BERT/Sentence-Transformers."""
    model_name: str = "all-MiniLM-L6-v2"  # Lightweight, fast model
    max_sequence_length: int = 512
    batch_size: int = 32
    normalize_embeddings: bool = True


@dataclass
class RecommenderConfig:
    """Configuration for the recommendation engine."""
    default_top_n: int = 10
    min_similarity_threshold: float = 0.1
    default_model: str = "tfidf"  # Options: "tfidf", "bert", "hybrid"
    hybrid_tfidf_weight: float = 0.4
    hybrid_bert_weight: float = 0.6


@dataclass
class APIConfig:
    """Configuration for Flask API."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    relevance_threshold: float = 0.5


class Config:
    """
    Main configuration class that aggregates all sub-configurations.
    
    Usage:
        config = Config()
        print(config.preprocessing.lowercase)
        print(config.tfidf.max_features)
    """
    
    def __init__(self):
        self.preprocessing = PreprocessingConfig()
        self.tfidf = TFIDFConfig()
        self.bert = BERTConfig()
        self.recommender = RecommenderConfig()
        self.api = APIConfig()
        self.evaluation = EvaluationConfig()
    
    def __repr__(self):
        return (
            f"Config(\n"
            f"  preprocessing={self.preprocessing},\n"
            f"  tfidf={self.tfidf},\n"
            f"  bert={self.bert},\n"
            f"  recommender={self.recommender},\n"
            f"  api={self.api},\n"
            f"  evaluation={self.evaluation}\n"
            f")"
        )


# Global config instance
config = Config()
