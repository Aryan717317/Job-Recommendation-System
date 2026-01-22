"""
Data Schemas for Job Recommendation System

This module defines the core data structures for:
- Candidate profiles (skills, experience, role preferences)
- Job descriptions (title, responsibilities, required skills)

Uses Pydantic for validation and type safety.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class ExperienceLevel(str, Enum):
    """Enumeration of experience levels."""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"
    EXECUTIVE = "executive"


class EmploymentType(str, Enum):
    """Enumeration of employment types."""
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    FREELANCE = "freelance"
    INTERNSHIP = "internship"


class WorkLocation(str, Enum):
    """Enumeration of work location preferences."""
    REMOTE = "remote"
    ONSITE = "onsite"
    HYBRID = "hybrid"


class Skill(BaseModel):
    """
    Represents a skill with optional proficiency level.
    
    Attributes:
        name: The skill name (e.g., "Python", "Machine Learning")
        proficiency: Optional proficiency level (1-5 scale)
        years_experience: Optional years of experience with this skill
    """
    name: str = Field(..., min_length=1, description="Skill name")
    proficiency: Optional[int] = Field(None, ge=1, le=5, description="Proficiency level (1-5)")
    years_experience: Optional[float] = Field(None, ge=0, description="Years of experience")
    
    @field_validator('name')
    @classmethod
    def normalize_skill_name(cls, v: str) -> str:
        """Normalize skill name to lowercase and strip whitespace."""
        return v.strip().lower()
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Skill):
            return self.name == other.name
        return False


class WorkExperience(BaseModel):
    """
    Represents a work experience entry.
    
    Attributes:
        job_title: Title of the position
        company: Company name
        start_date: Start date of employment
        end_date: End date (None if current position)
        description: Job description and responsibilities
        skills_used: Skills utilized in this role
    """
    job_title: str = Field(..., min_length=1)
    company: str = Field(..., min_length=1)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    description: Optional[str] = None
    skills_used: List[str] = Field(default_factory=list)
    
    @property
    def is_current(self) -> bool:
        """Check if this is the current position."""
        return self.end_date is None
    
    @property
    def duration_years(self) -> Optional[float]:
        """Calculate duration in years."""
        if self.start_date is None:
            return None
        end = self.end_date or datetime.now()
        return (end - self.start_date).days / 365.25


class Education(BaseModel):
    """
    Represents educational background.
    
    Attributes:
        degree: Degree type (e.g., "Bachelor's", "Master's")
        field: Field of study (e.g., "Computer Science")
        institution: Educational institution name
        graduation_year: Year of graduation
    """
    degree: str = Field(..., min_length=1)
    field: str = Field(..., min_length=1)
    institution: Optional[str] = None
    graduation_year: Optional[int] = Field(None, ge=1900, le=2100)


class CandidateProfile(BaseModel):
    """
    Complete candidate profile for job matching.
    
    Attributes:
        id: Unique identifier for the candidate
        name: Candidate's full name
        email: Contact email
        skills: List of skills with proficiency levels
        experience: List of work experiences
        education: List of educational qualifications
        summary: Professional summary or objective
        years_of_experience: Total years of professional experience
        preferred_roles: List of desired job titles
        preferred_locations: Preferred work locations
        preferred_employment_types: Preferred employment types
        expected_salary_min: Minimum expected salary
        expected_salary_max: Maximum expected salary
        created_at: Profile creation timestamp
        updated_at: Profile last update timestamp
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1)
    email: Optional[str] = None
    skills: List[Skill] = Field(default_factory=list)
    experience: List[WorkExperience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    summary: Optional[str] = Field(None, description="Professional summary")
    years_of_experience: Optional[float] = Field(None, ge=0)
    preferred_roles: List[str] = Field(default_factory=list)
    preferred_locations: List[WorkLocation] = Field(default_factory=list)
    preferred_employment_types: List[EmploymentType] = Field(default_factory=list)
    expected_salary_min: Optional[float] = Field(None, ge=0)
    expected_salary_max: Optional[float] = Field(None, ge=0)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def skill_names(self) -> List[str]:
        """Get list of skill names."""
        return [skill.name for skill in self.skills]
    
    def get_combined_text(self) -> str:
        """
        Generate combined text representation for NLP processing.
        
        Returns:
            Combined string of all relevant text fields
        """
        text_parts = []
        
        # Add summary
        if self.summary:
            text_parts.append(self.summary)
        
        # Add skills
        skill_text = " ".join(self.skill_names)
        if skill_text:
            text_parts.append(skill_text)
        
        # Add preferred roles
        if self.preferred_roles:
            text_parts.append(" ".join(self.preferred_roles))
        
        # Add experience descriptions
        for exp in self.experience:
            if exp.description:
                text_parts.append(exp.description)
            text_parts.append(exp.job_title)
            if exp.skills_used:
                text_parts.append(" ".join(exp.skills_used))
        
        # Add education
        for edu in self.education:
            text_parts.append(f"{edu.degree} {edu.field}")
        
        return " ".join(text_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SkillRequirement(BaseModel):
    """
    Represents a skill requirement for a job.
    
    Attributes:
        name: Skill name
        required: Whether this skill is mandatory
        min_years: Minimum years of experience required
    """
    name: str = Field(..., min_length=1)
    required: bool = True
    min_years: Optional[float] = Field(None, ge=0)
    
    @field_validator('name')
    @classmethod
    def normalize_skill_name(cls, v: str) -> str:
        """Normalize skill name to lowercase and strip whitespace."""
        return v.strip().lower()


class JobDescription(BaseModel):
    """
    Complete job description for candidate matching.
    
    Attributes:
        id: Unique identifier for the job
        title: Job title
        company: Company name
        description: Full job description
        responsibilities: List of job responsibilities
        required_skills: Required skills with specifications
        preferred_skills: Nice-to-have skills
        experience_level: Required experience level
        min_years_experience: Minimum years of experience
        max_years_experience: Maximum years of experience
        employment_type: Type of employment
        location: Work location preference
        salary_min: Minimum salary offered
        salary_max: Maximum salary offered
        department: Department name
        posted_date: Date when job was posted
        is_active: Whether the job is still active
        tags: Additional tags for categorization
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., min_length=1)
    company: str = Field(..., min_length=1)
    description: str = Field(..., min_length=10)
    responsibilities: List[str] = Field(default_factory=list)
    required_skills: List[SkillRequirement] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    experience_level: Optional[ExperienceLevel] = None
    min_years_experience: Optional[float] = Field(None, ge=0)
    max_years_experience: Optional[float] = Field(None, ge=0)
    employment_type: EmploymentType = EmploymentType.FULL_TIME
    location: WorkLocation = WorkLocation.HYBRID
    salary_min: Optional[float] = Field(None, ge=0)
    salary_max: Optional[float] = Field(None, ge=0)
    department: Optional[str] = None
    posted_date: datetime = Field(default_factory=datetime.now)
    is_active: bool = True
    tags: List[str] = Field(default_factory=list)
    
    @property
    def required_skill_names(self) -> List[str]:
        """Get list of required skill names."""
        return [skill.name for skill in self.required_skills]
    
    @property
    def all_skill_names(self) -> List[str]:
        """Get all skill names (required + preferred)."""
        skills = self.required_skill_names.copy()
        skills.extend([s.lower().strip() for s in self.preferred_skills])
        return list(set(skills))
    
    def get_combined_text(self) -> str:
        """
        Generate combined text representation for NLP processing.
        
        Returns:
            Combined string of all relevant text fields
        """
        text_parts = []
        
        # Add title and company
        text_parts.append(self.title)
        text_parts.append(self.company)
        
        # Add description
        text_parts.append(self.description)
        
        # Add responsibilities
        if self.responsibilities:
            text_parts.extend(self.responsibilities)
        
        # Add all skills
        text_parts.extend(self.all_skill_names)
        
        # Add tags
        if self.tags:
            text_parts.extend(self.tags)
        
        # Add department
        if self.department:
            text_parts.append(self.department)
        
        return " ".join(text_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RecommendationResult(BaseModel):
    """
    Represents a single job recommendation result.
    
    Attributes:
        job: The recommended job description
        similarity_score: Overall similarity score
        tfidf_score: TF-IDF based similarity score
        bert_score: BERT embedding based similarity score
        skill_match_ratio: Ratio of matched skills
        matched_skills: List of skills that matched
        rank: Ranking position
    """
    job: JobDescription
    similarity_score: float = Field(..., ge=0, le=1)
    tfidf_score: Optional[float] = Field(None, ge=0, le=1)
    bert_score: Optional[float] = Field(None, ge=0, le=1)
    skill_match_ratio: Optional[float] = Field(None, ge=0, le=1)
    matched_skills: List[str] = Field(default_factory=list)
    rank: int = Field(..., ge=1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job.id,
            "job_title": self.job.title,
            "company": self.job.company,
            "similarity_score": round(self.similarity_score, 4),
            "tfidf_score": round(self.tfidf_score, 4) if self.tfidf_score else None,
            "bert_score": round(self.bert_score, 4) if self.bert_score else None,
            "skill_match_ratio": round(self.skill_match_ratio, 4) if self.skill_match_ratio else None,
            "matched_skills": self.matched_skills,
            "rank": self.rank
        }


class RecommendationResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    """
    Complete recommendation response.
    
    Attributes:
        candidate_id: ID of the candidate
        recommendations: List of recommendation results
        model_used: Name of the model used for recommendations
        processing_time_ms: Time taken to generate recommendations
        total_jobs_considered: Total number of jobs in the pool
    """
    candidate_id: str
    recommendations: List[RecommendationResult]
    model_used: str
    processing_time_ms: float
    total_jobs_considered: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "candidate_id": self.candidate_id,
            "recommendations": [rec.to_dict() for rec in self.recommendations],
            "model_used": self.model_used,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "total_jobs_considered": self.total_jobs_considered
        }
