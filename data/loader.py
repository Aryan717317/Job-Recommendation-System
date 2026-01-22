"""
Data Loader Module for Job Recommendation System

This module provides utilities for loading data from various sources:
- CSV files
- JSON files
- Mock data generation

Supports batch loading and data validation.
"""

import json
import csv
from pathlib import Path
from typing import List, Optional, Union, Tuple
import logging

from .schemas import CandidateProfile, JobDescription, Skill, SkillRequirement
from .mock_data import generate_mock_data

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loader for candidates and job descriptions.
    
    Supports loading from:
    - CSV files
    - JSON files
    - Mock data generation
    
    Usage:
        loader = DataLoader()
        candidates = loader.load_candidates_from_json("candidates.json")
        jobs = loader.load_jobs_from_csv("jobs.csv")
        candidates, jobs = loader.load_mock_data(50, 100)
    """
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize the data loader.
        
        Args:
            base_path: Base directory path for data files (optional)
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
    
    def _resolve_path(self, file_path: Union[str, Path]) -> Path:
        """Resolve file path relative to base path if not absolute."""
        path = Path(file_path)
        if path.is_absolute():
            return path
        return self.base_path / path
    
    # ===== JSON Loading =====
    
    def load_candidates_from_json(
        self,
        file_path: Union[str, Path]
    ) -> List[CandidateProfile]:
        """
        Load candidate profiles from a JSON file.
        
        Expected format:
        [
            {
                "name": "John Doe",
                "skills": [{"name": "python", "proficiency": 4}],
                ...
            }
        ]
        
        Args:
            file_path: Path to the JSON file
        
        Returns:
            List of CandidateProfile objects
        """
        path = self._resolve_path(file_path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            candidates = []
            for item in data:
                try:
                    candidate = CandidateProfile(**item)
                    candidates.append(candidate)
                except Exception as e:
                    logger.warning(f"Failed to parse candidate: {e}")
                    continue
            
            logger.info(f"Loaded {len(candidates)} candidates from {path}")
            return candidates
            
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {path}: {e}")
            raise
    
    def load_jobs_from_json(
        self,
        file_path: Union[str, Path]
    ) -> List[JobDescription]:
        """
        Load job descriptions from a JSON file.
        
        Args:
            file_path: Path to the JSON file
        
        Returns:
            List of JobDescription objects
        """
        path = self._resolve_path(file_path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            jobs = []
            for item in data:
                try:
                    job = JobDescription(**item)
                    jobs.append(job)
                except Exception as e:
                    logger.warning(f"Failed to parse job: {e}")
                    continue
            
            logger.info(f"Loaded {len(jobs)} jobs from {path}")
            return jobs
            
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {path}: {e}")
            raise
    
    # ===== CSV Loading =====
    
    def load_candidates_from_csv(
        self,
        file_path: Union[str, Path],
        skills_delimiter: str = "|"
    ) -> List[CandidateProfile]:
        """
        Load candidate profiles from a CSV file.
        
        Expected columns:
        - name (required)
        - email (optional)
        - skills (pipe-delimited string)
        - summary (optional)
        - years_of_experience (optional)
        - preferred_roles (pipe-delimited string)
        
        Args:
            file_path: Path to the CSV file
            skills_delimiter: Delimiter for skill lists (default: "|")
        
        Returns:
            List of CandidateProfile objects
        """
        path = self._resolve_path(file_path)
        candidates = []
        
        try:
            with open(path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        # Parse skills
                        skills = []
                        if 'skills' in row and row['skills']:
                            skill_names = row['skills'].split(skills_delimiter)
                            skills = [Skill(name=s.strip()) for s in skill_names if s.strip()]
                        
                        # Parse preferred roles
                        preferred_roles = []
                        if 'preferred_roles' in row and row['preferred_roles']:
                            preferred_roles = [
                                r.strip() for r in row['preferred_roles'].split(skills_delimiter)
                                if r.strip()
                            ]
                        
                        # Parse years of experience
                        years_exp = None
                        if 'years_of_experience' in row and row['years_of_experience']:
                            try:
                                years_exp = float(row['years_of_experience'])
                            except ValueError:
                                pass
                        
                        candidate = CandidateProfile(
                            name=row.get('name', 'Unknown'),
                            email=row.get('email'),
                            skills=skills,
                            summary=row.get('summary'),
                            years_of_experience=years_exp,
                            preferred_roles=preferred_roles
                        )
                        candidates.append(candidate)
                    
                    except Exception as e:
                        logger.warning(f"Failed to parse candidate row: {e}")
                        continue
            
            logger.info(f"Loaded {len(candidates)} candidates from {path}")
            return candidates
            
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise
    
    def load_jobs_from_csv(
        self,
        file_path: Union[str, Path],
        skills_delimiter: str = "|"
    ) -> List[JobDescription]:
        """
        Load job descriptions from a CSV file.
        
        Expected columns:
        - title (required)
        - company (required)
        - description (required)
        - required_skills (pipe-delimited string)
        - preferred_skills (pipe-delimited string)
        - min_years_experience (optional)
        - location (optional)
        
        Args:
            file_path: Path to the CSV file
            skills_delimiter: Delimiter for skill lists (default: "|")
        
        Returns:
            List of JobDescription objects
        """
        path = self._resolve_path(file_path)
        jobs = []
        
        try:
            with open(path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        # Parse required skills
                        required_skills = []
                        if 'required_skills' in row and row['required_skills']:
                            skill_names = row['required_skills'].split(skills_delimiter)
                            required_skills = [
                                SkillRequirement(name=s.strip())
                                for s in skill_names if s.strip()
                            ]
                        
                        # Parse preferred skills
                        preferred_skills = []
                        if 'preferred_skills' in row and row['preferred_skills']:
                            preferred_skills = [
                                s.strip() for s in row['preferred_skills'].split(skills_delimiter)
                                if s.strip()
                            ]
                        
                        # Parse min years experience
                        min_years = None
                        if 'min_years_experience' in row and row['min_years_experience']:
                            try:
                                min_years = float(row['min_years_experience'])
                            except ValueError:
                                pass
                        
                        job = JobDescription(
                            title=row.get('title', 'Unknown'),
                            company=row.get('company', 'Unknown'),
                            description=row.get('description', ''),
                            required_skills=required_skills,
                            preferred_skills=preferred_skills,
                            min_years_experience=min_years
                        )
                        jobs.append(job)
                    
                    except Exception as e:
                        logger.warning(f"Failed to parse job row: {e}")
                        continue
            
            logger.info(f"Loaded {len(jobs)} jobs from {path}")
            return jobs
            
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise
    
    # ===== Mock Data Loading =====
    
    def load_mock_data(
        self,
        num_candidates: int = 20,
        num_jobs: int = 50
    ) -> Tuple[List[CandidateProfile], List[JobDescription]]:
        """
        Generate mock data for testing.
        
        Args:
            num_candidates: Number of candidate profiles to generate
            num_jobs: Number of job descriptions to generate
        
        Returns:
            Tuple of (candidates, jobs)
        """
        candidates, jobs = generate_mock_data(num_candidates, num_jobs)
        logger.info(f"Generated {len(candidates)} mock candidates and {len(jobs)} mock jobs")
        return candidates, jobs
    
    # ===== Data Saving =====
    
    def save_candidates_to_json(
        self,
        candidates: List[CandidateProfile],
        file_path: Union[str, Path]
    ) -> None:
        """
        Save candidate profiles to a JSON file.
        
        Args:
            candidates: List of CandidateProfile objects
            file_path: Path to save the JSON file
        """
        path = self._resolve_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [c.model_dump(mode='json') for c in candidates]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(candidates)} candidates to {path}")
    
    def save_jobs_to_json(
        self,
        jobs: List[JobDescription],
        file_path: Union[str, Path]
    ) -> None:
        """
        Save job descriptions to a JSON file.
        
        Args:
            jobs: List of JobDescription objects
            file_path: Path to save the JSON file
        """
        path = self._resolve_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [j.model_dump(mode='json') for j in jobs]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(jobs)} jobs to {path}")


# Convenience functions for quick access
def load_mock_data(
    num_candidates: int = 20,
    num_jobs: int = 50
) -> Tuple[List[CandidateProfile], List[JobDescription]]:
    """
    Quick function to load mock data.
    
    Args:
        num_candidates: Number of candidates to generate
        num_jobs: Number of jobs to generate
    
    Returns:
        Tuple of (candidates, jobs)
    """
    loader = DataLoader()
    return loader.load_mock_data(num_candidates, num_jobs)
