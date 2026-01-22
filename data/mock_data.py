"""
Mock Data Generator for Job Recommendation System

This module provides realistic mock data for testing and development:
- Candidate profiles with varied skills and experience
- Job descriptions across different industries and roles

The mock data is designed to be comprehensive enough for meaningful
testing of the recommendation algorithms.
"""

from typing import List, Tuple
from datetime import datetime, timedelta
import random

from .schemas import (
    CandidateProfile, Skill, WorkExperience, Education,
    JobDescription, SkillRequirement,
    ExperienceLevel, EmploymentType, WorkLocation
)


# Skill pools for realistic data generation
TECH_SKILLS = [
    "python", "javascript", "typescript", "java", "c++", "go", "rust",
    "react", "angular", "vue.js", "node.js", "django", "flask", "fastapi",
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "tensorflow", "pytorch", "scikit-learn",
    "sql", "postgresql", "mongodb", "redis", "elasticsearch",
    "aws", "google cloud platform", "azure", "kubernetes", "docker",
    "git", "linux", "agile", "scrum", "ci/cd",
    "data analysis", "data visualization", "pandas", "numpy",
    "spark", "hadoop", "kafka", "airflow",
    "rest api", "graphql", "microservices", "system design",
    "html", "css", "sass", "tailwind css"
]

DATA_SCIENCE_SKILLS = [
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "statistics", "data modeling", "feature engineering",
    "a/b testing", "experiment design", "python", "r", "sql",
    "tensorflow", "pytorch", "scikit-learn", "xgboost", "lightgbm",
    "pandas", "numpy", "matplotlib", "seaborn", "tableau", "power bi",
    "spark", "hadoop", "hive", "big data", "data pipeline",
    "mlops", "model deployment", "recommendation systems",
    "time series analysis", "clustering", "classification", "regression"
]

SOFT_SKILLS = [
    "communication", "leadership", "problem solving", "teamwork",
    "project management", "analytical thinking", "attention to detail",
    "time management", "adaptability", "creativity"
]

COMPANIES = [
    "TechCorp Inc.", "DataDriven Labs", "AI Solutions Ltd.",
    "CloudFirst Technologies", "InnovateTech", "Digital Dynamics",
    "SmartSystems Co.", "FutureTech Industries", "CodeCraft Solutions",
    "Quantum Computing Corp.", "Neural Networks Inc.", "ByteForce",
    "DataStream Analytics", "AlgorithmX", "TechVentures",
    "MetaLogic Systems", "CyberCore Technologies", "Infinity Software"
]

JOB_TITLES = [
    "Software Engineer", "Senior Software Engineer", "Staff Engineer",
    "Data Scientist", "Senior Data Scientist", "Machine Learning Engineer",
    "Backend Developer", "Frontend Developer", "Full Stack Developer",
    "DevOps Engineer", "Site Reliability Engineer", "Cloud Architect",
    "Data Engineer", "Analytics Engineer", "AI Research Scientist",
    "Technical Lead", "Engineering Manager", "Principal Engineer",
    "Product Manager", "Technical Product Manager",
    "Solutions Architect", "System Administrator"
]

UNIVERSITIES = [
    "MIT", "Stanford University", "Carnegie Mellon University",
    "UC Berkeley", "Georgia Tech", "University of Michigan",
    "University of Washington", "Cornell University", "UCLA",
    "University of Illinois", "Purdue University", "NYU"
]

DEGREES = ["Bachelor's", "Master's", "PhD"]
FIELDS = [
    "Computer Science", "Software Engineering", "Data Science",
    "Mathematics", "Statistics", "Electrical Engineering",
    "Information Technology", "Physics", "Applied Mathematics"
]


def generate_random_skills(
    skill_pool: List[str],
    min_skills: int = 3,
    max_skills: int = 10
) -> List[Skill]:
    """
    Generate a random list of skills from a skill pool.
    
    Args:
        skill_pool: List of potential skills to choose from
        min_skills: Minimum number of skills to generate
        max_skills: Maximum number of skills to generate
    
    Returns:
        List of Skill objects with random proficiency levels
    """
    num_skills = random.randint(min_skills, max_skills)
    selected_skills = random.sample(skill_pool, min(num_skills, len(skill_pool)))
    
    skills = []
    for skill_name in selected_skills:
        skills.append(Skill(
            name=skill_name,
            proficiency=random.randint(2, 5),
            years_experience=round(random.uniform(0.5, 8), 1)
        ))
    
    return skills


def generate_work_experience(
    num_positions: int = 2,
    skills: List[str] = None
) -> List[WorkExperience]:
    """
    Generate realistic work experience entries.
    
    Args:
        num_positions: Number of positions to generate
        skills: Skills to potentially include in experience descriptions
    
    Returns:
        List of WorkExperience objects
    """
    experiences = []
    current_date = datetime.now()
    
    for i in range(num_positions):
        is_current = (i == 0)
        duration_months = random.randint(12, 48)
        
        if is_current:
            end_date = None
            start_date = current_date - timedelta(days=duration_months * 30)
        else:
            end_date = current_date - timedelta(days=i * 365 + random.randint(0, 180))
            start_date = end_date - timedelta(days=duration_months * 30)
        
        job_title = random.choice(JOB_TITLES)
        company = random.choice(COMPANIES)
        
        # Generate description
        responsibilities = [
            f"Developed and maintained {random.choice(['web applications', 'APIs', 'data pipelines', 'ML models'])}",
            f"Collaborated with {random.choice(['product', 'design', 'data science', 'infrastructure'])} teams",
            f"Improved {random.choice(['performance', 'reliability', 'code quality', 'user experience'])} by {random.randint(20, 60)}%",
            f"Led {random.choice(['technical initiatives', 'code reviews', 'architecture decisions', 'team meetings'])}"
        ]
        description = ". ".join(random.sample(responsibilities, random.randint(2, 4)))
        
        # Skills used
        skills_used = random.sample(skills or TECH_SKILLS, random.randint(3, 6))
        
        experiences.append(WorkExperience(
            job_title=job_title,
            company=company,
            start_date=start_date,
            end_date=end_date,
            description=description,
            skills_used=skills_used
        ))
        
        current_date = start_date - timedelta(days=random.randint(30, 180))
    
    return experiences


def generate_education(num_degrees: int = 1) -> List[Education]:
    """
    Generate educational background entries.
    
    Args:
        num_degrees: Number of degrees to generate
    
    Returns:
        List of Education objects
    """
    educations = []
    current_year = datetime.now().year
    
    for i in range(num_degrees):
        degree = DEGREES[min(i, len(DEGREES) - 1)]
        grad_year = current_year - random.randint(1, 15) - (i * 3)
        
        educations.append(Education(
            degree=degree,
            field=random.choice(FIELDS),
            institution=random.choice(UNIVERSITIES),
            graduation_year=grad_year
        ))
    
    return educations


def generate_candidate_profiles(num_candidates: int = 20) -> List[CandidateProfile]:
    """
    Generate diverse candidate profiles for testing.
    
    Args:
        num_candidates: Number of candidate profiles to generate
    
    Returns:
        List of CandidateProfile objects
    """
    first_names = [
        "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley",
        "Jamie", "Quinn", "Avery", "Drew", "Blake", "Cameron",
        "Peyton", "Skyler", "Dakota", "Reese", "Hayden", "Emery",
        "Finley", "Phoenix"
    ]
    
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
        "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson",
        "Martin", "Lee"
    ]
    
    candidate_types = [
        ("data_scientist", DATA_SCIENCE_SKILLS, 
         ["Data Scientist", "Machine Learning Engineer", "AI Researcher"]),
        ("software_engineer", TECH_SKILLS,
         ["Software Engineer", "Backend Developer", "Full Stack Developer"]),
        ("frontend_dev", ["javascript", "typescript", "react", "vue.js", "angular", "html", "css", "sass", "tailwind css", "node.js", "graphql"],
         ["Frontend Developer", "UI Engineer", "React Developer"]),
        ("devops", ["kubernetes", "docker", "aws", "gcp", "azure", "terraform", "linux", "ci/cd", "python", "bash", "monitoring", "prometheus", "grafana"],
         ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer"]),
        ("data_engineer", ["python", "sql", "spark", "hadoop", "kafka", "airflow", "data pipeline", "etl", "aws", "gcp", "data modeling"],
         ["Data Engineer", "Analytics Engineer", "Platform Engineer"])
    ]
    
    candidates = []
    
    for i in range(num_candidates):
        # Select candidate type
        candidate_type = candidate_types[i % len(candidate_types)]
        type_name, skill_pool, preferred_roles = candidate_type
        
        # Generate basic info
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        email = f"{name.lower().replace(' ', '.')}@email.com"
        
        # Generate skills (mix technical and soft skills)
        tech_skills = generate_random_skills(skill_pool, min_skills=5, max_skills=12)
        soft_skill_list = random.sample(SOFT_SKILLS, random.randint(2, 4))
        for soft_skill in soft_skill_list:
            tech_skills.append(Skill(name=soft_skill, proficiency=random.randint(3, 5)))
        
        # Generate experience
        years_exp = random.uniform(1, 12)
        num_positions = max(1, int(years_exp / 3))
        experience = generate_work_experience(num_positions, [s.name for s in tech_skills])
        
        # Generate education
        num_degrees = 1 if years_exp < 5 else (2 if random.random() > 0.5 else 1)
        education = generate_education(num_degrees)
        
        # Generate summary
        summaries = [
            f"Experienced {preferred_roles[0]} with {years_exp:.1f}+ years of experience in building scalable solutions.",
            f"Passionate technologist specializing in {', '.join([s.name for s in tech_skills[:3]])}.",
            f"Results-driven professional with expertise in {skill_pool[0]} and {skill_pool[1]}.",
            f"Dedicated {preferred_roles[0]} focused on delivering high-quality, innovative solutions."
        ]
        summary = random.choice(summaries)
        
        # Create candidate profile
        candidate = CandidateProfile(
            name=name,
            email=email,
            skills=tech_skills,
            experience=experience,
            education=education,
            summary=summary,
            years_of_experience=round(years_exp, 1),
            preferred_roles=preferred_roles,
            preferred_locations=random.sample(list(WorkLocation), random.randint(1, 3)),
            preferred_employment_types=[EmploymentType.FULL_TIME]
        )
        
        candidates.append(candidate)
    
    return candidates


def generate_job_descriptions(num_jobs: int = 50) -> List[JobDescription]:
    """
    Generate diverse job descriptions for testing.
    
    Args:
        num_jobs: Number of job descriptions to generate
    
    Returns:
        List of JobDescription objects
    """
    job_templates = [
        {
            "title": "Senior Data Scientist",
            "skills": ["machine learning", "python", "sql", "statistics", "deep learning"],
            "preferred": ["tensorflow", "pytorch", "spark", "nlp"],
            "level": ExperienceLevel.SENIOR,
            "min_years": 5,
            "description": "We are looking for a Senior Data Scientist to join our team and help build ML-powered features. You will work on challenging problems in recommendation systems, NLP, and predictive modeling."
        },
        {
            "title": "Machine Learning Engineer",
            "skills": ["python", "machine learning", "tensorflow", "pytorch", "mlops"],
            "preferred": ["kubernetes", "docker", "aws", "spark"],
            "level": ExperienceLevel.MID,
            "min_years": 3,
            "description": "Join our ML Engineering team to build and deploy production machine learning systems. You'll work on the full ML lifecycle from data processing to model deployment."
        },
        {
            "title": "Backend Software Engineer",
            "skills": ["python", "java", "sql", "rest api", "microservices"],
            "preferred": ["kubernetes", "aws", "redis", "kafka"],
            "level": ExperienceLevel.MID,
            "min_years": 3,
            "description": "We're seeking a Backend Engineer to design and implement scalable backend services. You'll work on high-throughput systems processing millions of requests daily."
        },
        {
            "title": "Full Stack Developer",
            "skills": ["javascript", "react", "node.js", "sql", "rest api"],
            "preferred": ["typescript", "graphql", "aws", "docker"],
            "level": ExperienceLevel.MID,
            "min_years": 2,
            "description": "Join our product team as a Full Stack Developer. You'll build user-facing features from database to UI, working closely with design and product teams."
        },
        {
            "title": "DevOps Engineer",
            "skills": ["kubernetes", "docker", "aws", "ci/cd", "linux"],
            "preferred": ["terraform", "python", "monitoring", "security"],
            "level": ExperienceLevel.MID,
            "min_years": 3,
            "description": "We're looking for a DevOps Engineer to help scale our infrastructure. You'll work on Kubernetes clusters, CI/CD pipelines, and infrastructure automation."
        },
        {
            "title": "Data Engineer",
            "skills": ["python", "sql", "spark", "airflow", "data pipeline"],
            "preferred": ["kafka", "aws", "dbt", "data modeling"],
            "level": ExperienceLevel.MID,
            "min_years": 3,
            "description": "Join our data platform team to build robust data pipelines. You'll work on ETL processes, data warehousing, and real-time streaming systems."
        },
        {
            "title": "Frontend Engineer",
            "skills": ["javascript", "react", "typescript", "css", "html"],
            "preferred": ["vue.js", "graphql", "testing", "accessibility"],
            "level": ExperienceLevel.MID,
            "min_years": 2,
            "description": "We're seeking a Frontend Engineer passionate about building beautiful, responsive web applications. You'll work on our customer-facing products."
        },
        {
            "title": "AI Research Scientist",
            "skills": ["python", "deep learning", "pytorch", "mathematics", "research"],
            "preferred": ["nlp", "computer vision", "reinforcement learning", "publications"],
            "level": ExperienceLevel.SENIOR,
            "min_years": 4,
            "description": "Join our AI Research team to push the boundaries of machine learning. You'll conduct original research and publish at top conferences."
        },
        {
            "title": "Site Reliability Engineer",
            "skills": ["linux", "python", "kubernetes", "monitoring", "incident response"],
            "preferred": ["aws", "terraform", "prometheus", "grafana"],
            "level": ExperienceLevel.MID,
            "min_years": 3,
            "description": "We're looking for an SRE to ensure our systems are reliable and scalable. You'll work on monitoring, automation, and incident response."
        },
        {
            "title": "NLP Engineer",
            "skills": ["python", "natural language processing", "transformers", "pytorch", "machine learning"],
            "preferred": ["bert", "gpt", "spacy", "information extraction"],
            "level": ExperienceLevel.MID,
            "min_years": 2,
            "description": "Join our NLP team to build language understanding systems. You'll work on text classification, entity extraction, and conversational AI."
        }
    ]
    
    jobs = []
    
    for i in range(num_jobs):
        template = job_templates[i % len(job_templates)]
        company = random.choice(COMPANIES)
        
        # Vary the job slightly
        required_skills = [
            SkillRequirement(name=skill, required=True, min_years=random.randint(1, 3))
            for skill in template["skills"]
        ]
        
        # Add some random additional skills
        additional_skills = random.sample(TECH_SKILLS, random.randint(0, 3))
        for skill in additional_skills:
            if skill not in [s.name for s in required_skills]:
                required_skills.append(SkillRequirement(name=skill, required=False))
        
        preferred_skills = template["preferred"] + random.sample(SOFT_SKILLS, 2)
        
        responsibilities = [
            "Design, develop, and maintain high-quality software solutions",
            "Collaborate with cross-functional teams to define and implement features",
            "Participate in code reviews and maintain code quality standards",
            "Mentor junior team members and contribute to technical decisions",
            "Write technical documentation and share knowledge with the team",
            "Identify and resolve performance bottlenecks",
            "Contribute to system architecture and design decisions"
        ]
        
        job = JobDescription(
            title=f"{template['title']}" + (f" - {company[:3].upper()}" if random.random() > 0.7 else ""),
            company=company,
            description=template["description"],
            responsibilities=random.sample(responsibilities, random.randint(3, 5)),
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            experience_level=template["level"],
            min_years_experience=template["min_years"],
            max_years_experience=template["min_years"] + random.randint(2, 5),
            employment_type=EmploymentType.FULL_TIME,
            location=random.choice(list(WorkLocation)),
            salary_min=60000 + (template["min_years"] * 15000),
            salary_max=80000 + (template["min_years"] * 20000),
            department=random.choice(["Engineering", "Data Science", "Platform", "Product"]),
            tags=[template["title"].split()[0], template["skills"][0], company.split()[0]]
        )
        
        jobs.append(job)
    
    return jobs


def generate_mock_data(
    num_candidates: int = 20,
    num_jobs: int = 50
) -> Tuple[List[CandidateProfile], List[JobDescription]]:
    """
    Generate complete mock dataset for testing.
    
    Args:
        num_candidates: Number of candidate profiles to generate
        num_jobs: Number of job descriptions to generate
    
    Returns:
        Tuple of (candidates, jobs)
    """
    candidates = generate_candidate_profiles(num_candidates)
    jobs = generate_job_descriptions(num_jobs)
    
    return candidates, jobs


# Pre-generated datasets for quick access
def get_sample_candidates() -> List[CandidateProfile]:
    """Get a pre-generated sample of candidate profiles."""
    return generate_candidate_profiles(20)


def get_sample_jobs() -> List[JobDescription]:
    """Get a pre-generated sample of job descriptions."""
    return generate_job_descriptions(50)
