"""
Text Preprocessing Pipeline for Job Recommendation System

This module implements a robust NLP preprocessing pipeline including:
- Lowercasing
- Tokenization
- Stop-word removal
- Lemmatization
- Skill normalization

Uses SpaCy and NLTK for NLP operations.
Designed for reusability across TF-IDF and BERT models.
"""

import re
import string
from typing import List, Dict, Optional, Set, Union
import logging

# NLP Libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for NLP-based job matching.
    
    Features:
    - Configurable preprocessing steps
    - SpaCy-based processing (with NLTK fallback)
    - Skill normalization (e.g., "ML" → "Machine Learning")
    - Reusable across different vectorization strategies
    
    Usage:
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.process("Python, ML, and deep learning skills")
        tokens = preprocessor.tokenize("Experience with NLP and AI")
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        remove_punctuation: bool = True,
        min_token_length: int = 2,
        skill_mappings: Optional[Dict[str, str]] = None,
        use_spacy: bool = True
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            lowercase: Whether to convert text to lowercase
            remove_stopwords: Whether to remove stop words
            lemmatize: Whether to lemmatize tokens
            remove_punctuation: Whether to remove punctuation
            min_token_length: Minimum token length to keep
            skill_mappings: Custom skill abbreviation mappings
            use_spacy: Whether to use SpaCy (falls back to NLTK if unavailable)
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_punctuation = remove_punctuation
        self.min_token_length = min_token_length
        
        # Load skill mappings from config or use provided ones
        self.skill_mappings = skill_mappings or config.preprocessing.skill_mappings
        
        # Initialize NLP components
        self._nlp = None
        self._stopwords = None
        self._lemmatizer = None
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        
        self._initialize_nlp()
    
    def _initialize_nlp(self) -> None:
        """Initialize NLP libraries and download required data."""
        if self.use_spacy:
            try:
                # Try to load SpaCy model
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("SpaCy model loaded successfully")
            except OSError:
                logger.warning("SpaCy model not found. Attempting to download...")
                try:
                    import subprocess
                    subprocess.run([
                        sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                    ], check=True, capture_output=True)
                    self._nlp = spacy.load("en_core_web_sm")
                    logger.info("SpaCy model downloaded and loaded")
                except Exception as e:
                    logger.warning(f"Failed to download SpaCy model: {e}. Falling back to NLTK.")
                    self.use_spacy = False
        
        if not self.use_spacy and NLTK_AVAILABLE:
            try:
                # Initialize NLTK components
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                
                self._stopwords = set(stopwords.words('english'))
                self._lemmatizer = WordNetLemmatizer()
                logger.info("NLTK components initialized")
            except Exception as e:
                logger.error(f"Failed to initialize NLTK: {e}")
        
        # Additional custom stop words for job matching
        self._custom_stopwords = {
            'experience', 'required', 'preferred', 'must', 'strong',
            'ability', 'working', 'knowledge', 'skills', 'proficiency',
            'years', 'year', 'plus', 'etc', 'including', 'related'
        }
    
    @property
    def stopwords(self) -> Set[str]:
        """Get the combined set of stop words."""
        if self.use_spacy:
            base_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        elif self._stopwords:
            base_stopwords = self._stopwords
        else:
            base_stopwords = set()
        
        return base_stopwords | self._custom_stopwords
    
    def normalize_skills(self, text: str) -> str:
        """
        Normalize skill abbreviations to full forms.
        
        Args:
            text: Input text containing skill abbreviations
        
        Returns:
            Text with normalized skill names
        """
        # Sort by length (longest first) to avoid partial replacements
        sorted_mappings = sorted(
            self.skill_mappings.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        normalized = text.lower()
        
        for abbrev, full_form in sorted_mappings:
            # Use word boundary matching to avoid partial replacements
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            normalized = re.sub(pattern, full_form, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize raw text.
        
        Args:
            text: Raw input text
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Normalize skills first (before lowercasing affects abbreviations)
        text = self.normalize_skills(text)
        
        # Lowercase if configured
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep hyphens for compound words
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Remove numbers (optional, keep for version numbers)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text to tokenize
        
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Clean text first
        cleaned = self.clean_text(text)
        
        if self.use_spacy and self._nlp:
            # Use SpaCy tokenization
            doc = self._nlp(cleaned)
            tokens = [token.text for token in doc]
        elif NLTK_AVAILABLE:
            # Use NLTK tokenization
            tokens = word_tokenize(cleaned)
        else:
            # Fallback to simple whitespace tokenization
            tokens = cleaned.split()
        
        # Filter tokens
        tokens = self._filter_tokens(tokens)
        
        return tokens
    
    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens based on configuration.
        
        Args:
            tokens: List of tokens to filter
        
        Returns:
            Filtered list of tokens
        """
        filtered = []
        
        for token in tokens:
            # Skip empty tokens
            if not token or not token.strip():
                continue
            
            token = token.strip()
            
            # Skip short tokens
            if len(token) < self.min_token_length:
                continue
            
            # Remove punctuation if configured
            if self.remove_punctuation:
                token = token.translate(str.maketrans('', '', string.punctuation))
                if not token:
                    continue
            
            # Skip stopwords if configured
            if self.remove_stopwords and token.lower() in self.stopwords:
                continue
            
            filtered.append(token)
        
        return filtered
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize a list of tokens.
        
        Args:
            tokens: List of tokens to lemmatize
        
        Returns:
            List of lemmatized tokens
        """
        if not self.lemmatize:
            return tokens
        
        lemmatized = []
        
        if self.use_spacy and self._nlp:
            # Use SpaCy lemmatization
            text = ' '.join(tokens)
            doc = self._nlp(text)
            lemmatized = [token.lemma_ for token in doc if token.text in tokens]
        elif self._lemmatizer:
            # Use NLTK lemmatization
            lemmatized = [self._lemmatizer.lemmatize(token) for token in tokens]
        else:
            lemmatized = tokens
        
        return lemmatized
    
    def process(self, text: str) -> str:
        """
        Full preprocessing pipeline: clean, tokenize, lemmatize, and join.
        
        Args:
            text: Raw input text
        
        Returns:
            Fully processed text as a single string
        """
        if not text:
            return ""
        
        # Tokenize (includes cleaning)
        tokens = self.tokenize(text)
        
        # Lemmatize
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Join back into string
        return ' '.join(tokens)
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of texts to process
        
        Returns:
            List of processed texts
        """
        return [self.process(text) for text in texts]
    
    def get_tokens(self, text: str) -> List[str]:
        """
        Get fully processed tokens from text.
        
        Args:
            text: Raw input text
        
        Returns:
            List of processed tokens
        """
        tokens = self.tokenize(text)
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        return tokens
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract potential skills from text.
        
        This is a simple heuristic-based extraction that looks for:
        - Known skill keywords
        - Technical terms
        - Multi-word skill phrases
        
        Args:
            text: Input text to extract skills from
        
        Returns:
            List of extracted skill names
        """
        # Clean and normalize text
        cleaned = self.clean_text(text)
        
        extracted_skills = []
        
        # Check for known skill mappings (both abbreviated and full forms)
        all_skills = set(self.skill_mappings.keys()) | set(self.skill_mappings.values())
        
        for skill in all_skills:
            if skill.lower() in cleaned.lower():
                extracted_skills.append(skill)
        
        # Also include processed tokens that could be skills
        tokens = self.tokenize(cleaned)
        extracted_skills.extend(tokens)
        
        # Remove duplicates and return
        return list(set(extracted_skills))


class CandidateTextExtractor:
    """
    Utility class to extract and combine text from candidate profiles.
    
    Provides consistent text representation for candidates across
    different vectorization strategies.
    """
    
    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize the extractor.
        
        Args:
            preprocessor: Optional preprocessor instance (creates default if None)
        """
        self.preprocessor = preprocessor or TextPreprocessor()
    
    def extract_raw_text(self, candidate) -> str:
        """
        Extract raw combined text from a candidate profile.
        
        Args:
            candidate: CandidateProfile object
        
        Returns:
            Combined raw text
        """
        return candidate.get_combined_text()
    
    def extract_processed_text(self, candidate) -> str:
        """
        Extract and preprocess text from a candidate profile.
        
        Args:
            candidate: CandidateProfile object
        
        Returns:
            Processed text ready for vectorization
        """
        raw_text = self.extract_raw_text(candidate)
        return self.preprocessor.process(raw_text)


class JobTextExtractor:
    """
    Utility class to extract and combine text from job descriptions.
    
    Provides consistent text representation for jobs across
    different vectorization strategies.
    """
    
    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize the extractor.
        
        Args:
            preprocessor: Optional preprocessor instance (creates default if None)
        """
        self.preprocessor = preprocessor or TextPreprocessor()
    
    def extract_raw_text(self, job) -> str:
        """
        Extract raw combined text from a job description.
        
        Args:
            job: JobDescription object
        
        Returns:
            Combined raw text
        """
        return job.get_combined_text()
    
    def extract_processed_text(self, job) -> str:
        """
        Extract and preprocess text from a job description.
        
        Args:
            job: JobDescription object
        
        Returns:
            Processed text ready for vectorization
        """
        raw_text = self.extract_raw_text(job)
        return self.preprocessor.process(raw_text)


# Create default preprocessor instance
default_preprocessor = TextPreprocessor()


def preprocess_text(text: str) -> str:
    """
    Convenience function to preprocess text using default settings.
    
    Args:
        text: Raw input text
    
    Returns:
        Processed text
    """
    return default_preprocessor.process(text)


def tokenize_text(text: str) -> List[str]:
    """
    Convenience function to tokenize text using default settings.
    
    Args:
        text: Raw input text
    
    Returns:
        List of tokens
    """
    return default_preprocessor.get_tokens(text)
