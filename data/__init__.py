"""Data package for Job Recommendation System."""

from .schemas import (
    CandidateProfile,
    JobDescription,
    Skill,
    SkillRequirement,
    WorkExperience,
    Education,
    RecommendationResult,
    RecommendationResponse,
    ExperienceLevel,
    EmploymentType,
    WorkLocation
)
from .loader import DataLoader, load_mock_data
from .mock_data import (
    generate_mock_data,
    generate_candidate_profiles,
    generate_job_descriptions,
    get_sample_candidates,
    get_sample_jobs
)

__all__ = [
    # Schemas
    "CandidateProfile",
    "JobDescription",
    "Skill",
    "SkillRequirement",
    "WorkExperience",
    "Education",
    "RecommendationResult",
    "RecommendationResponse",
    "ExperienceLevel",
    "EmploymentType",
    "WorkLocation",
    # Loader
    "DataLoader",
    "load_mock_data",
    # Mock data
    "generate_mock_data",
    "generate_candidate_profiles",
    "generate_job_descriptions",
    "get_sample_candidates",
    "get_sample_jobs"
]
