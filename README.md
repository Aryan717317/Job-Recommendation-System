# AI-Powered Job Recommendation System

A modular, scalable job recommendation system that matches candidate profiles to relevant job postings using NLP-based content similarity techniques.

## 🎯 Features

- **Multi-Model Architecture**: Switch between TF-IDF (keyword-based), BERT (semantic), and Hybrid approaches
- **Robust NLP Pipeline**: Tokenization, lemmatization, stop-word removal, and skill normalization
- **REST API**: Flask-based backend with comprehensive endpoints
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG, MRR metrics
- **Model Comparison**: Built-in tools to compare TF-IDF vs BERT performance

## 📁 Project Structure

```
job-recommendation-system/
├── api/                    # Flask REST API
│   ├── __init__.py
│   └── app.py              # API endpoints
├── config/                 # Configuration
│   ├── __init__.py
│   └── config.py           # Centralized settings
├── data/                   # Data models and loading
│   ├── __init__.py
│   ├── schemas.py          # Pydantic models
│   ├── mock_data.py        # Mock data generation
│   └── loader.py           # Data loading utilities
├── evaluation/             # Evaluation metrics
│   ├── __init__.py
│   └── metrics.py          # Precision, Recall, NDCG, etc.
├── models/                 # Vectorization models
│   ├── __init__.py
│   ├── base_model.py       # Abstract base class
│   ├── tfidf_model.py      # TF-IDF implementation
│   ├── bert_model.py       # BERT/Sentence-Transformers
│   └── hybrid_model.py     # Weighted combination
├── preprocessing/          # NLP preprocessing
│   ├── __init__.py
│   └── text_processor.py   # Text cleaning, tokenization
├── recommender/            # Recommendation engine
│   ├── __init__.py
│   └── engine.py           # Core recommendation logic
├── tests/                  # Unit tests
│   ├── __init__.py
│   └── test_system.py
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm
```

### 2. Run Demo

```bash
python main.py demo
```

This will:
- Generate mock candidate and job data
- Create a TF-IDF recommendation engine
- Generate and display top 5 job recommendations

### 3. Start API Server

```bash
python main.py serve --port 5000 --model tfidf
```

Or run directly:
```bash
python api/app.py
```

## 📡 API Endpoints

### Health Check
```
GET /health
```

### List Jobs
```
GET /jobs?limit=10&offset=0&search=python
```

### Get Job Details
```
GET /jobs/<job_id>
```

### List Candidates
```
GET /candidates?limit=10&offset=0
```

### Get Candidate Details
```
GET /candidates/<candidate_id>
```

### Get Recommendations
```
POST /recommend
Content-Type: application/json

{
  "candidate_id": "existing-id",
  "top_n": 10,
  "model_type": "tfidf"
}
```

Or provide a new candidate profile:
```json
{
  "candidate": {
    "name": "John Doe",
    "skills": [{"name": "python"}, {"name": "machine learning"}],
    "summary": "Experienced data scientist...",
    "preferred_roles": ["Data Scientist"]
  },
  "top_n": 10
}
```

### Compare Models
```
POST /compare
```

### Switch Model
```
POST /model/switch
Content-Type: application/json

{
  "model_type": "bert"
}
```

## 🔧 Model Types

### 1. TF-IDF (`tfidf`)
- **Best for**: Exact keyword matching
- **Speed**: Fast
- **Use case**: When skill names and job titles are most important

### 2. BERT (`bert`)
- **Best for**: Semantic understanding
- **Speed**: Slower (requires GPU for best performance)
- **Use case**: When understanding context and meaning matters

### 3. Hybrid (`hybrid`)
- **Best for**: Balanced approach
- **Speed**: Moderate
- **Use case**: Production systems needing both keyword and semantic matching

## 📊 Evaluation

Run model evaluation:
```bash
python main.py evaluate --model tfidf
```

Compare TF-IDF vs BERT:
```bash
python main.py compare
```

### Metrics Explained

| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of recommended items that are relevant |
| **Recall@K** | Fraction of relevant items that are recommended |
| **NDCG@K** | Normalized Discounted Cumulative Gain (position-aware) |
| **MRR** | Mean Reciprocal Rank of first relevant item |
| **Hit Rate** | Whether any relevant item appears in top K |

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_system.py::TestTFIDFModel -v
```

## 💡 Example Usage

```python
from data import load_mock_data, CandidateProfile, Skill
from recommender import create_engine

# Load data
candidates, jobs = load_mock_data(20, 50)

# Create engine
engine = create_engine(model_type="tfidf")
engine.fit(jobs)

# Get recommendations
recommendations = engine.recommend(candidates[0], top_n=5)

for rec in recommendations.recommendations:
    print(f"#{rec.rank}: {rec.job.title} ({rec.similarity_score:.2%})")
```

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Data Layer    │────▶│  Preprocessing   │────▶│     Models      │
│  (Schemas,      │     │  (Tokenization,  │     │  (TF-IDF, BERT, │
│   Mock Data)    │     │   Lemmatization) │     │   Hybrid)       │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              ▼
                        │   Evaluation     │◀────┌─────────────────┐
                        │   (Metrics)      │     │   Recommender   │
                        └──────────────────┘     │   Engine        │
                                                 └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │    Flask API    │
                                                 └─────────────────┘
```

## ⚙️ Configuration

Edit `config/config.py` to customize:

- **Preprocessing**: Tokenization settings, skill mappings
- **TF-IDF**: Max features, n-gram range, TF scaling
- **BERT**: Model name, max sequence length, batch size
- **Recommender**: Default top N, similarity threshold
- **Evaluation**: K values for metrics

## 🔮 Future Enhancements

- [ ] Resume parsing (PDF/DOCX upload)
- [ ] Experience-aware ranking
- [ ] Skill importance weighting
- [ ] Real-time model updates
- [ ] A/B testing framework

## 📄 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request
