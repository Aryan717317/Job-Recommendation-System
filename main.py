"""
AI-Powered Job Recommendation System
=====================================

Main entry point for the recommendation system.
Provides CLI interface and demonstration capabilities.

Usage:
    # Run the API server
    python main.py serve --port 5000 --model tfidf
    
    # Run demo/evaluation
    python main.py demo
    
    # Evaluate models
    python main.py evaluate --model tfidf
    
    # Compare TF-IDF vs BERT
    python main.py compare
"""

import argparse
import logging
import sys
import time
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_demo():
    """Run a demonstration of the recommendation system."""
    print("\n" + "="*70)
    print(" AI-Powered Job Recommendation System - Demo")
    print("="*70 + "\n")
    
    from data import load_mock_data, CandidateProfile, Skill
    from recommender import create_engine
    from preprocessing import TextPreprocessor
    
    # Generate mock data
    print("1. Generating mock data...")
    candidates, jobs = load_mock_data(num_candidates=10, num_jobs=30)
    print(f"   [OK] Generated {len(candidates)} candidates and {len(jobs)} jobs\n")
    
    # Show sample candidate
    print("2. Sample Candidate Profile:")
    sample_candidate = candidates[0]
    print(f"   Name: {sample_candidate.name}")
    print(f"   Skills: {', '.join(sample_candidate.skill_names[:5])}...")
    print(f"   Years of Experience: {sample_candidate.years_of_experience}")
    print(f"   Preferred Roles: {', '.join(sample_candidate.preferred_roles)}\n")
    
    # Create and fit engine
    print("3. Creating TF-IDF recommendation engine...")
    engine = create_engine(model_type="tfidf")
    engine.fit(jobs)
    print(f"   [OK] Engine fitted with {len(jobs)} jobs\n")
    
    # Generate recommendations
    print("4. Generating recommendations for sample candidate...")
    start_time = time.time()
    recommendations = engine.recommend(sample_candidate, top_n=5)
    elapsed = (time.time() - start_time) * 1000
    
    print(f"   [OK] Generated {len(recommendations.recommendations)} recommendations in {elapsed:.1f}ms\n")
    
    # Display recommendations
    print("5. Top 5 Job Recommendations:")
    print("-" * 70)
    
    for rec in recommendations.recommendations:
        print(f"\n   Rank #{rec.rank}: {rec.job.title}")
        print(f"   Company: {rec.job.company}")
        print(f"   Similarity Score: {rec.similarity_score:.4f}")
        print(f"   Skill Match: {rec.skill_match_ratio:.1%}")
        print(f"   Matched Skills: {', '.join(rec.matched_skills[:5])}")
    
    print("\n" + "-" * 70)
    print("\n6. Scoring Breakdown for Top Match:")
    breakdown = engine.get_scoring_breakdown(sample_candidate, recommendations.recommendations[0].job)
    print(f"   Total Score: {breakdown.total_score:.4f}")
    print(f"   Skill Match Score: {breakdown.skill_match_score:.4f}")
    print(f"   Matched Skills: {', '.join(breakdown.matched_skills[:5])}")
    
    print("\n" + "="*70)
    print(" Demo Complete!")
    print("="*70 + "\n")


def run_evaluation(model_type: str = "tfidf"):
    """Run evaluation of the recommendation model."""
    print("\n" + "="*70)
    print(f" Evaluating {model_type.upper()} Model")
    print("="*70 + "\n")
    
    from data import load_mock_data
    from recommender import create_engine
    from evaluation import RecommendationEvaluator
    
    # Load data
    print("Loading data...")
    candidates, jobs = load_mock_data(num_candidates=20, num_jobs=50)
    
    # Create engine
    print(f"Creating {model_type} engine...")
    engine = create_engine(model_type=model_type)
    engine.fit(jobs)
    
    # Generate synthetic ground truth
    evaluator = RecommendationEvaluator(k_values=[1, 3, 5, 10])
    ground_truth = evaluator.generate_synthetic_ground_truth(
        candidates, jobs, skill_match_threshold=0.3
    )
    
    print(f"Generated ground truth for {len(ground_truth)} candidates\n")
    
    # Evaluate
    print("Evaluating recommendations...")
    recommendations = [engine.recommend(c, top_n=20) for c in candidates]
    metrics = evaluator.evaluate_batch(recommendations, ground_truth)
    
    # Print report
    evaluator.print_evaluation_report(metrics, f"{model_type.upper()} Model Evaluation")


def run_comparison():
    """Compare TF-IDF vs BERT models."""
    print("\n" + "="*70)
    print(" Model Comparison: TF-IDF vs BERT")
    print("="*70 + "\n")
    
    from data import load_mock_data
    from recommender import create_engine
    from evaluation import RecommendationEvaluator
    
    # Load data
    print("Loading data...")
    candidates, jobs = load_mock_data(num_candidates=10, num_jobs=30)
    
    # Create engines
    print("Creating TF-IDF engine...")
    engine_tfidf = create_engine(model_type="tfidf")
    engine_tfidf.fit(jobs)
    
    print("Creating BERT engine (this may take a moment)...")
    try:
        engine_bert = create_engine(model_type="bert")
        engine_bert.fit(jobs)
        has_bert = True
    except ImportError as e:
        print(f"   [!] BERT not available: {e}")
        print("   Skipping BERT comparison\n")
        has_bert = False
    
    # Generate ground truth
    evaluator = RecommendationEvaluator(k_values=[1, 3, 5, 10])
    ground_truth = evaluator.generate_synthetic_ground_truth(
        candidates, jobs, skill_match_threshold=0.3
    )
    
    if has_bert:
        # Compare models
        print("\nComparing models...")
        comparison = evaluator.compare_models(
            engine_tfidf,
            engine_bert,
            candidates,
            ground_truth,
            model1_name="TF-IDF",
            model2_name="BERT"
        )
        
        evaluator.print_comparison_report(comparison)
    else:
        # Just evaluate TF-IDF
        print("\nEvaluating TF-IDF only...")
        recommendations = [engine_tfidf.recommend(c, top_n=20) for c in candidates]
        metrics = evaluator.evaluate_batch(recommendations, ground_truth)
        evaluator.print_evaluation_report(metrics, "TF-IDF Model Evaluation")


def run_server(host: str, port: int, model_type: str, debug: bool):
    """Run the Flask API server."""
    from api.app import app, initialize_system
    
    initialize_system(model_type=model_type)
    
    print(f"\n{'='*60}")
    print(f" Job Recommendation API Server")
    print(f"{'='*60}")
    print(f" URL: http://{host}:{port}")
    print(f" Model: {model_type}")
    print(f" Debug: {debug}")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=debug)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AI-Powered Job Recommendation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py demo              Run a demonstration
  python main.py serve             Start the API server
  python main.py evaluate          Evaluate model performance
  python main.py compare           Compare TF-IDF vs BERT
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run a demonstration')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start the API server')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    serve_parser.add_argument('--model', default='tfidf', 
                             choices=['tfidf', 'bert', 'hybrid'],
                             help='Model type to use')
    serve_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--model', default='tfidf',
                            choices=['tfidf', 'bert', 'hybrid'],
                            help='Model to evaluate')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare TF-IDF vs BERT')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        run_demo()
    elif args.command == 'serve':
        run_server(args.host, args.port, args.model, args.debug)
    elif args.command == 'evaluate':
        run_evaluation(args.model)
    elif args.command == 'compare':
        run_comparison()
    else:
        # Default to demo
        parser.print_help()
        print("\nRunning demo by default...\n")
        run_demo()


if __name__ == '__main__':
    main()
