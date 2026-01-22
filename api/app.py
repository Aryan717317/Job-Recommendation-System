"""
Flask REST API for Job Recommendation System

This module implements the backend API exposing:
- POST /recommend: Get job recommendations for a candidate
- GET /jobs: List all jobs
- GET /jobs/<id>: Get a specific job
- GET /candidates: List all candidates
- GET /candidates/<id>: Get a specific candidate
- POST /compare: Compare TF-IDF vs BERT models
- GET /health: Health check endpoint

Features:
- JSON request/response
- Clear error handling
- Similarity scores in responses
- Model selection support
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from typing import Dict, Any, Optional
import logging
import traceback
import time
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (
    CandidateProfile, 
    JobDescription, 
    Skill,
    SkillRequirement,
    load_mock_data,
    DataLoader
)
from recommender import RecommendationEngine, create_engine
from evaluation import RecommendationEvaluator
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global state
_engine: Optional[RecommendationEngine] = None
_candidates: list = []
_jobs: list = []
_initialized: bool = False


def get_engine() -> RecommendationEngine:
    """Get or create the recommendation engine."""
    global _engine, _initialized
    
    if not _initialized:
        initialize_system()
    
    return _engine


def initialize_system(
    num_candidates: int = 20,
    num_jobs: int = 50,
    model_type: str = "tfidf"
) -> None:
    """
    Initialize the recommendation system with data.
    
    Args:
        num_candidates: Number of mock candidates to generate
        num_jobs: Number of mock jobs to generate
        model_type: Model type to use ("tfidf", "bert", "hybrid")
    """
    global _engine, _candidates, _jobs, _initialized
    
    logger.info(f"Initializing recommendation system with {model_type} model...")
    
    # Load mock data
    _candidates, _jobs = load_mock_data(num_candidates, num_jobs)
    
    # Create and fit engine
    _engine = create_engine(model_type=model_type)
    _engine.fit(_jobs)
    
    _initialized = True
    
    logger.info(
        f"System initialized with {len(_candidates)} candidates "
        f"and {len(_jobs)} jobs"
    )


def candidate_to_dict(candidate: CandidateProfile) -> Dict[str, Any]:
    """Convert CandidateProfile to API response dict."""
    return {
        "id": candidate.id,
        "name": candidate.name,
        "email": candidate.email,
        "skills": [
            {"name": s.name, "proficiency": s.proficiency, "years": s.years_experience}
            for s in candidate.skills
        ],
        "summary": candidate.summary,
        "years_of_experience": candidate.years_of_experience,
        "preferred_roles": candidate.preferred_roles,
        "experience": [
            {
                "job_title": exp.job_title,
                "company": exp.company,
                "description": exp.description
            }
            for exp in candidate.experience
        ],
        "education": [
            {
                "degree": edu.degree,
                "field": edu.field,
                "institution": edu.institution
            }
            for edu in candidate.education
        ]
    }


def job_to_dict(job: JobDescription) -> Dict[str, Any]:
    """Convert JobDescription to API response dict."""
    return {
        "id": job.id,
        "title": job.title,
        "company": job.company,
        "description": job.description,
        "responsibilities": job.responsibilities,
        "required_skills": [
            {"name": s.name, "required": s.required, "min_years": s.min_years}
            for s in job.required_skills
        ],
        "preferred_skills": job.preferred_skills,
        "experience_level": job.experience_level.value if job.experience_level else None,
        "min_years_experience": job.min_years_experience,
        "max_years_experience": job.max_years_experience,
        "employment_type": job.employment_type.value,
        "location": job.location.value,
        "salary_min": job.salary_min,
        "salary_max": job.salary_max,
        "department": job.department,
        "is_active": job.is_active
    }


# ============== API Endpoints ==============

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "initialized": _initialized,
        "model_type": _engine.model_type if _engine else None,
        "num_candidates": len(_candidates),
        "num_jobs": len(_jobs)
    })


@app.route('/jobs', methods=['GET'])
def list_jobs():
    """
    List all available jobs.
    
    Query parameters:
    - limit: Maximum number of jobs to return (default: 50)
    - offset: Number of jobs to skip (default: 0)
    - search: Search term to filter jobs by title or company
    """
    if not _initialized:
        initialize_system()
    
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    search = request.args.get('search', '', type=str).lower()
    
    # Filter jobs
    filtered_jobs = _jobs
    if search:
        filtered_jobs = [
            job for job in _jobs
            if search in job.title.lower() or search in job.company.lower()
        ]
    
    # Paginate
    paginated = filtered_jobs[offset:offset + limit]
    
    return jsonify({
        "total": len(filtered_jobs),
        "limit": limit,
        "offset": offset,
        "jobs": [job_to_dict(job) for job in paginated]
    })


@app.route('/jobs/<job_id>', methods=['GET'])
def get_job(job_id: str):
    """Get a specific job by ID."""
    if not _initialized:
        initialize_system()
    
    job = next((j for j in _jobs if j.id == job_id), None)
    
    if not job:
        return jsonify({"error": f"Job not found: {job_id}"}), 404
    
    return jsonify(job_to_dict(job))


@app.route('/candidates', methods=['GET'])
def list_candidates():
    """
    List all candidates.
    
    Query parameters:
    - limit: Maximum number to return (default: 20)
    - offset: Number to skip (default: 0)
    """
    if not _initialized:
        initialize_system()
    
    limit = request.args.get('limit', 20, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    paginated = _candidates[offset:offset + limit]
    
    return jsonify({
        "total": len(_candidates),
        "limit": limit,
        "offset": offset,
        "candidates": [candidate_to_dict(c) for c in paginated]
    })


@app.route('/candidates/<candidate_id>', methods=['GET'])
def get_candidate(candidate_id: str):
    """Get a specific candidate by ID."""
    if not _initialized:
        initialize_system()
    
    candidate = next((c for c in _candidates if c.id == candidate_id), None)
    
    if not candidate:
        return jsonify({"error": f"Candidate not found: {candidate_id}"}), 404
    
    return jsonify(candidate_to_dict(candidate))


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Get job recommendations for a candidate.
    
    Request body:
    {
        "candidate_id": "existing-candidate-id",
        // OR provide full candidate profile:
        "candidate": {
            "name": "John Doe",
            "skills": [{"name": "python"}, {"name": "machine learning"}],
            "summary": "Experienced data scientist...",
            "preferred_roles": ["Data Scientist", "ML Engineer"]
        },
        "top_n": 10,
        "model_type": "tfidf"  // Optional: "tfidf", "bert", or "hybrid"
    }
    
    Response:
    {
        "candidate_id": "...",
        "model_used": "tfidf",
        "processing_time_ms": 45.2,
        "total_jobs_considered": 50,
        "recommendations": [
            {
                "rank": 1,
                "job_id": "...",
                "job_title": "Senior Data Scientist",
                "company": "TechCorp",
                "similarity_score": 0.87,
                "tfidf_score": 0.87,
                "skill_match_ratio": 0.75,
                "matched_skills": ["python", "machine learning"]
            },
            ...
        ]
    }
    """
    if not _initialized:
        initialize_system()
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Request body required"}), 400
        
        # Get or create candidate
        candidate = None
        
        if 'candidate_id' in data:
            candidate = next(
                (c for c in _candidates if c.id == data['candidate_id']),
                None
            )
            if not candidate:
                return jsonify({
                    "error": f"Candidate not found: {data['candidate_id']}"
                }), 404
        
        elif 'candidate' in data:
            # Create candidate from request data
            cand_data = data['candidate']
            
            skills = []
            for skill_data in cand_data.get('skills', []):
                if isinstance(skill_data, str):
                    skills.append(Skill(name=skill_data))
                else:
                    skills.append(Skill(
                        name=skill_data.get('name', ''),
                        proficiency=skill_data.get('proficiency'),
                        years_experience=skill_data.get('years_experience')
                    ))
            
            candidate = CandidateProfile(
                name=cand_data.get('name', 'Anonymous'),
                email=cand_data.get('email'),
                skills=skills,
                summary=cand_data.get('summary'),
                years_of_experience=cand_data.get('years_of_experience'),
                preferred_roles=cand_data.get('preferred_roles', [])
            )
        
        else:
            return jsonify({
                "error": "Either 'candidate_id' or 'candidate' object required"
            }), 400
        
        # Get parameters
        top_n = data.get('top_n', 10)
        model_type = data.get('model_type')
        
        # Generate recommendations
        engine = get_engine()
        response = engine.recommend(
            candidate=candidate,
            top_n=top_n,
            model_type=model_type
        )
        
        # Format response
        result = {
            "candidate_id": response.candidate_id,
            "model_used": response.model_used,
            "processing_time_ms": round(response.processing_time_ms, 2),
            "total_jobs_considered": response.total_jobs_considered,
            "recommendations": [
                {
                    "rank": rec.rank,
                    "job_id": rec.job.id,
                    "job_title": rec.job.title,
                    "company": rec.job.company,
                    "similarity_score": round(rec.similarity_score, 4),
                    "tfidf_score": round(rec.tfidf_score, 4) if rec.tfidf_score else None,
                    "bert_score": round(rec.bert_score, 4) if rec.bert_score else None,
                    "skill_match_ratio": round(rec.skill_match_ratio, 4) if rec.skill_match_ratio else None,
                    "matched_skills": rec.matched_skills,
                    "job_details": {
                        "description": rec.job.description[:200] + "..." if len(rec.job.description) > 200 else rec.job.description,
                        "required_skills": rec.job.required_skill_names,
                        "experience_level": rec.job.experience_level.value if rec.job.experience_level else None,
                        "location": rec.job.location.value
                    }
                }
                for rec in response.recommendations
            ]
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in /recommend: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/recommend/batch', methods=['POST'])
def recommend_batch():
    """
    Get recommendations for multiple candidates.
    
    Request body:
    {
        "candidate_ids": ["id1", "id2", ...],
        "top_n": 5,
        "model_type": "tfidf"
    }
    """
    if not _initialized:
        initialize_system()
    
    try:
        data = request.get_json()
        
        if not data or 'candidate_ids' not in data:
            return jsonify({"error": "'candidate_ids' required"}), 400
        
        candidate_ids = data['candidate_ids']
        top_n = data.get('top_n', 5)
        model_type = data.get('model_type')
        
        results = []
        engine = get_engine()
        
        for cid in candidate_ids:
            candidate = next((c for c in _candidates if c.id == cid), None)
            if candidate:
                response = engine.recommend(
                    candidate=candidate, 
                    top_n=top_n,
                    model_type=model_type
                )
                results.append({
                    "candidate_id": cid,
                    "recommendations": [
                        {
                            "rank": rec.rank,
                            "job_id": rec.job.id,
                            "job_title": rec.job.title,
                            "similarity_score": round(rec.similarity_score, 4)
                        }
                        for rec in response.recommendations
                    ]
                })
        
        return jsonify({
            "total_candidates": len(results),
            "results": results
        })
    
    except Exception as e:
        logger.error(f"Error in /recommend/batch: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/compare', methods=['POST'])
def compare_models():
    """
    Compare TF-IDF vs BERT model performance.
    
    Request body:
    {
        "candidate_ids": ["id1", "id2"],  // Optional, uses all if not provided
        "k_values": [1, 3, 5, 10]
    }
    """
    if not _initialized:
        initialize_system()
    
    try:
        data = request.get_json() or {}
        
        candidate_ids = data.get('candidate_ids')
        k_values = data.get('k_values', [1, 3, 5, 10])
        
        # Select candidates
        if candidate_ids:
            candidates = [c for c in _candidates if c.id in candidate_ids]
        else:
            candidates = _candidates[:5]  # Use first 5 for demo
        
        # Create engines
        engine_tfidf = create_engine(model_type="tfidf")
        engine_tfidf.fit(_jobs)
        
        # For demo, generate synthetic ground truth
        evaluator = RecommendationEvaluator(k_values=k_values)
        ground_truth = evaluator.generate_synthetic_ground_truth(
            candidates, _jobs, skill_match_threshold=0.3
        )
        
        # Get recommendations from both models
        recs_tfidf = [engine_tfidf.recommend(c, top_n=max(k_values)) for c in candidates]
        
        # Evaluate TF-IDF
        metrics_tfidf = evaluator.evaluate_batch(recs_tfidf, ground_truth)
        
        # Format response
        result = {
            "comparison_summary": {
                "num_candidates": len(candidates),
                "num_jobs": len(_jobs),
                "k_values": k_values
            },
            "tfidf_metrics": {
                k: m.to_dict() for k, m in metrics_tfidf.items()
            },
            "interpretation": (
                "TF-IDF excels at exact keyword matching. For semantic understanding, "
                "try the BERT model by setting model_type='bert' in /recommend."
            )
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in /compare: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/model/switch', methods=['POST'])
def switch_model():
    """
    Switch the recommendation model.
    
    Request body:
    {
        "model_type": "bert"  // "tfidf", "bert", or "hybrid"
    }
    """
    global _engine
    
    if not _initialized:
        initialize_system()
    
    try:
        data = request.get_json()
        
        if not data or 'model_type' not in data:
            return jsonify({"error": "'model_type' required"}), 400
        
        model_type = data['model_type'].lower()
        
        if model_type not in ['tfidf', 'bert', 'hybrid']:
            return jsonify({
                "error": f"Invalid model_type: {model_type}. Choose from: tfidf, bert, hybrid"
            }), 400
        
        # Switch model
        _engine.switch_model(model_type)
        
        return jsonify({
            "message": f"Switched to {model_type} model",
            "current_model": _engine.model_type,
            "stats": _engine.stats
        })
    
    except Exception as e:
        logger.error(f"Error in /model/switch: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    if not _initialized:
        initialize_system()
    
    return jsonify({
        "engine_stats": _engine.stats,
        "data_stats": {
            "num_candidates": len(_candidates),
            "num_jobs": len(_jobs),
            "avg_skills_per_candidate": sum(len(c.skills) for c in _candidates) / len(_candidates) if _candidates else 0,
            "avg_skills_per_job": sum(len(j.all_skill_names) for j in _jobs) / len(_jobs) if _jobs else 0
        }
    })


@app.route('/initialize', methods=['POST'])
def reinitialize():
    """
    Reinitialize the system with new parameters.
    
    Request body:
    {
        "num_candidates": 30,
        "num_jobs": 100,
        "model_type": "hybrid"
    }
    """
    global _initialized
    
    try:
        data = request.get_json() or {}
        
        _initialized = False
        
        initialize_system(
            num_candidates=data.get('num_candidates', 20),
            num_jobs=data.get('num_jobs', 50),
            model_type=data.get('model_type', 'tfidf')
        )
        
        return jsonify({
            "message": "System reinitialized",
            "status": "success",
            "stats": {
                "num_candidates": len(_candidates),
                "num_jobs": len(_jobs),
                "model_type": _engine.model_type
            }
        })
    
    except Exception as e:
        logger.error(f"Error in /initialize: {e}")
        return jsonify({"error": str(e)}), 500


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


def create_app(model_type: str = "tfidf") -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        model_type: Initial model type
    
    Returns:
        Configured Flask app
    """
    initialize_system(model_type=model_type)
    return app


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Job Recommendation API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--model', default='tfidf', choices=['tfidf', 'bert', 'hybrid'],
                       help='Initial model type')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize system
    initialize_system(model_type=args.model)
    
    # Run server
    print(f"\n{'='*60}")
    print(f" Job Recommendation API")
    print(f"{'='*60}")
    print(f" Server running at: http://{args.host}:{args.port}")
    print(f" Model: {args.model}")
    print(f" Endpoints:")
    print(f"   GET  /health       - Health check")
    print(f"   GET  /jobs         - List jobs")
    print(f"   GET  /candidates   - List candidates")
    print(f"   POST /recommend    - Get recommendations")
    print(f"   POST /compare      - Compare models")
    print(f"{'='*60}\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
