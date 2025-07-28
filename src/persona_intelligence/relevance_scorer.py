"""
Relevance Scorer - Advanced multi-criteria decision analysis for section relevance
Enhanced with semantic understanding, adaptive weighting, and performance optimization
"""

import re
import math
import logging
import numpy as np
import time
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import yaml

# ML and NLP imports with fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("sklearn not available, falling back to basic similarity")

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("transformers not available, using fallback methods")

from persona_intelligence.persona_manager import PersonaProfile, PersonaCluster

@dataclass
class ScoringMetrics:
    """Container for scoring metrics and breakdown"""
    final_score: float
    semantic_score: float
    keyword_score: float
    readability_score: float
    section_importance_score: float
    recency_score: float
    authority_score: float
    persona_alignment_score: float
    confidence_level: str
    processing_time: float

class RelevanceScorer:
    """Advanced multi-criteria relevance scoring with adaptive weighting"""
    
    def __init__(self, config_path: Optional[str] = None, model_name: str = "huawei-noah/TinyBERT_General_4L_312D"):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize models with fallbacks
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.tfidf_vectorizer = None
        
        self._initialize_models()
        
        # Enhanced MCDA weights with persona-specific adaptations
        self.base_mcda_weights = self.config.get('base_weights', {
            'semantic_relevance': 0.35,
            'persona_alignment': 0.25,
            'keyword_relevance': 0.15,
            'readability': 0.10,
            'section_importance': 0.10,
            'authority': 0.03,
            'recency': 0.02
        })
        
        # Persona-specific weight adjustments
        self.persona_weight_adjustments = self._get_persona_weight_adjustments()
        
        # Performance optimization
        self._similarity_cache = {}
        self._max_cache_size = 1000
        
        # Advanced features
        self.enable_semantic_expansion = self.config.get('enable_semantic_expansion', True)
        self.enable_adaptive_weighting = self.config.get('enable_adaptive_weighting', True)
        
        self.logger.info("Enhanced RelevanceScorer initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration with fallback"""
        if not config_path:
            config_path = Path(__file__).parent.parent.parent / "configs" / "model_config_1b.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('relevance_scoring', {})
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'enable_semantic_expansion': True,
            'enable_adaptive_weighting': True,
            'similarity_threshold': 0.3,
            'max_cache_size': 1000
        }
    
    def _initialize_models(self):
        """Initialize ML models with graceful fallbacks"""
        # Initialize transformer model if available
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.eval()
                self.logger.info(f"Loaded transformer model: {self.model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load transformer model: {e}")
                self.tokenizer = None
                self.model = None
        
        # Initialize TF-IDF vectorizer if sklearn available
        if HAS_SKLEARN:
            try:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    lowercase=True,
                    token_pattern=r'\b[a-zA-Z]{3,}\b',
                    ngram_range=(1, 2)  # Include bigrams for better context
                )
                self.logger.info("Initialized TF-IDF vectorizer")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TF-IDF: {e}")
                self.tfidf_vectorizer = None
    
    def _get_persona_weight_adjustments(self) -> Dict[PersonaCluster, Dict[str, float]]:
        """Get persona-specific weight adjustments"""
        return {
            PersonaCluster.ANALYTICAL: {
                'semantic_relevance': 1.2,
                'authority': 1.5,
                'readability': 0.8
            },
            PersonaCluster.BUSINESS: {
                'semantic_relevance': 1.1,
                'recency': 1.8,
                'persona_alignment': 1.3
            },
            PersonaCluster.TECHNICAL: {
                'semantic_relevance': 1.3,
                'section_importance': 1.2,
                'readability': 0.7
            },
            PersonaCluster.EDUCATIONAL: {
                'readability': 1.5,
                'section_importance': 1.1,
                'semantic_relevance': 0.9
            },
            PersonaCluster.COMMUNICATION: {
                'readability': 1.4,
                'persona_alignment': 1.2,
                'authority': 0.8
            },
            PersonaCluster.PROFESSIONAL: {
                'authority': 1.6,
                'semantic_relevance': 1.1,
                'recency': 1.2
            }
        }
    
    def calculate_comprehensive_relevance_score(self, 
                                           text: str, 
                                           persona: PersonaProfile, 
                                           job_keywords: List[str],
                                           section_title: str = "",
                                           section_level: str = "h2",
                                           metadata: Dict[str, Any] = None) -> ScoringMetrics:
        """Calculate comprehensive relevance score with detailed breakdown"""
        start_time = time.time()
        
        # Get persona-adjusted weights
        weights = self._get_adjusted_weights(persona)
        
        # Calculate individual component scores
        semantic_score = self._calculate_semantic_relevance(text, persona, job_keywords)
        persona_alignment_score = self._calculate_persona_alignment(text, persona)
        keyword_score = self._calculate_enhanced_keyword_relevance(text, job_keywords, persona)
        readability_score = self._calculate_enhanced_readability(text, persona)
        section_importance_score = self._calculate_section_importance(section_title, section_level, persona)
        authority_score = self._calculate_enhanced_authority(text, metadata, persona)
        recency_score = self._calculate_recency_score(text, metadata)
        
        # Calculate weighted final score
        final_score = (
            semantic_score * weights['semantic_relevance'] +
            persona_alignment_score * weights['persona_alignment'] +
            keyword_score * weights['keyword_relevance'] +
            readability_score * weights['readability'] +
            section_importance_score * weights['section_importance'] +
            authority_score * weights['authority'] +
            recency_score * weights['recency']
        )
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(final_score, semantic_score, keyword_score)
        
        processing_time = time.time() - start_time
        
        return ScoringMetrics(
            final_score=min(final_score, 1.0),
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            readability_score=readability_score,
            section_importance_score=section_importance_score,
            recency_score=recency_score,
            authority_score=authority_score,
            persona_alignment_score=persona_alignment_score,
            confidence_level=confidence_level,
            processing_time=processing_time
        )
    
    def _get_adjusted_weights(self, persona: PersonaProfile) -> Dict[str, float]:
        """Get persona-adjusted weights"""
        if not self.enable_adaptive_weighting:
            return self.base_mcda_weights
        
        weights = self.base_mcda_weights.copy()
        
        # Apply persona cluster adjustments
        if persona.persona_cluster in self.persona_weight_adjustments:
            adjustments = self.persona_weight_adjustments[persona.persona_cluster]
            for weight_key, adjustment in adjustments.items():
                if weight_key in weights:
                    weights[weight_key] *= adjustment
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_semantic_relevance(self, text: str, persona: PersonaProfile, job_keywords: List[str]) -> float:
        """Enhanced semantic relevance calculation"""
        # Create context from persona and job
        persona_context = f"{persona.persona_type.value} {' '.join(persona.expertise_areas)} {' '.join(job_keywords)}"
        
        # Try transformer-based similarity first
        if self.model and self.tokenizer:
            try:
                similarity = self._calculate_transformer_similarity(text, persona_context)
                if similarity > 0:
                    return similarity
            except Exception as e:
                self.logger.debug(f"Transformer similarity failed: {e}")
        
        # Fallback to TF-IDF similarity
        if self.tfidf_vectorizer:
            try:
                similarity = self._calculate_tfidf_similarity(text, persona_context)
                if similarity > 0:
                    return similarity
            except Exception as e:
                self.logger.debug(f"TF-IDF similarity failed: {e}")
        
        # Final fallback to keyword overlap
        return self._calculate_keyword_overlap(text, persona_context)
    
    def _calculate_persona_alignment(self, text: str, persona: PersonaProfile) -> float:
        """Calculate how well text aligns with persona preferences"""
        alignment_score = 0.0
        text_lower = text.lower()
        
        # Check alignment with expertise areas
        expertise_matches = sum(1 for area in persona.expertise_areas 
                               if any(word in text_lower for word in area.lower().split()))
        if persona.expertise_areas:
            alignment_score += (expertise_matches / len(persona.expertise_areas)) * 0.4
        
        # Check alignment with preferred sections
        section_matches = sum(1 for section in persona.preferred_sections 
                             if section.lower() in text_lower)
        if persona.preferred_sections:
            alignment_score += (section_matches / len(persona.preferred_sections)) * 0.3
        
        # Check complexity alignment
        complexity_score = self._assess_text_complexity(text)
        if persona.complexity_preference == "simple" and complexity_score < 0.3:
            alignment_score += 0.2
        elif persona.complexity_preference == "complex" and complexity_score > 0.7:
            alignment_score += 0.2
        elif persona.complexity_preference == "balanced" and 0.3 <= complexity_score <= 0.7:
            alignment_score += 0.2
        
        # Check length alignment with detail level
        text_length = len(text.split())
        if persona.detail_level == "summary" and text_length < 100:
            alignment_score += 0.1
        elif persona.detail_level == "comprehensive" and text_length > 200:
            alignment_score += 0.1
        elif persona.detail_level == "detailed" and 100 <= text_length <= 200:
            alignment_score += 0.1
        
        return min(alignment_score, 1.0)
    
    def _calculate_enhanced_keyword_relevance(self, text: str, keywords: List[str], persona: PersonaProfile) -> float:
        """Enhanced keyword relevance with semantic expansion"""
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        total_score = 0.0
        
        # Expand keywords with persona-specific terms
        expanded_keywords = set(keywords)
        if self.enable_semantic_expansion:
            expanded_keywords.update(persona.semantic_keywords)
        
        # Calculate weighted keyword matches
        for keyword in expanded_keywords:
            keyword_lower = keyword.lower()
            
            # Exact match (highest weight)
            exact_matches = text_lower.count(keyword_lower)
            if exact_matches > 0:
                total_score += exact_matches * 1.0
            
            # Partial matches (medium weight)
            words = keyword_lower.split()
            if len(words) > 1:
                partial_matches = sum(1 for word in words if word in text_lower)
                total_score += (partial_matches / len(words)) * 0.5
        
        # Normalize by text length and keyword count
        normalization_factor = max(len(text.split()) / 100, 1.0) * len(expanded_keywords)
        normalized_score = total_score / normalization_factor if normalization_factor > 0 else 0.0
        
        return min(normalized_score, 1.0)
    
    def _calculate_enhanced_readability(self, text: str, persona: PersonaProfile) -> float:
        """Enhanced readability calculation with persona preferences"""
        if not text:
            return 0.0
        
        base_readability = self._calculate_flesch_reading_ease(text)
        
        # Adjust based on persona preferences
        if persona.complexity_preference == "simple":
            # Reward higher readability for simple preference
            adjusted_score = base_readability * 1.2
        elif persona.complexity_preference == "complex":
            # Tolerate lower readability for complex preference
            adjusted_score = max(base_readability, 0.5)
        else:
            adjusted_score = base_readability
        
        return min(adjusted_score, 1.0)
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using best available method"""
        # Check cache first
        cache_key = hash(text1 + text2)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        similarity = 0.0
        
        # Try transformer-based similarity
        if self.model and self.tokenizer:
            try:
                similarity = self._calculate_transformer_similarity(text1, text2)
            except Exception as e:
                self.logger.debug(f"Transformer similarity failed: {e}")
        
        # Fallback to TF-IDF if transformer failed
        if similarity == 0.0 and self.tfidf_vectorizer:
            try:
                similarity = self._calculate_tfidf_similarity(text1, text2)
            except Exception as e:
                self.logger.debug(f"TF-IDF similarity failed: {e}")
        
        # Final fallback to keyword overlap
        if similarity == 0.0:
            similarity = self._calculate_keyword_overlap(text1, text2)
        
        # Cache result
        if len(self._similarity_cache) < self._max_cache_size:
            self._similarity_cache[cache_key] = similarity
        
        return similarity
    
    def _calculate_transformer_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using transformer embeddings"""
        if not self.model or not self.tokenizer:
            return 0.0
        
        try:
            # Tokenize and encode
            inputs1 = self.tokenizer(text1, return_tensors="pt", max_length=128, truncation=True, padding=True)
            inputs2 = self.tokenizer(text2, return_tensors="pt", max_length=128, truncation=True, padding=True)
            
            with torch.no_grad():
                outputs1 = self.model(**inputs1)
                outputs2 = self.model(**inputs2)
            
            # Get embeddings (CLS token)
            embedding1 = outputs1.last_hidden_state[:, 0, :].numpy()
            embedding2 = outputs2.last_hidden_state[:, 0, :].numpy()
            
            # Calculate cosine similarity
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Semantic similarity calculation failed: {e}")
            return self._fallback_similarity(text1, text2)
    
    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """Fallback similarity using TF-IDF cosine similarity"""
        try:
            # Fit TF-IDF on both texts
            texts = [text1, text2]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Fallback similarity failed: {e}")
            return self._keyword_similarity(text1, text2)
    
    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Basic keyword overlap similarity"""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_keyword_relevance(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword-based relevance score"""
        if not keywords or not text:
            return 0.0
        
        text_lower = text.lower()
        keyword_scores = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Count occurrences
            count = text_lower.count(keyword_lower)
            
            # Calculate TF-IDF like score
            tf = count / len(text_lower.split()) if text_lower.split() else 0
            
            # Boost for exact matches
            if keyword_lower in text_lower:
                tf *= 1.5
            
            keyword_scores.append(tf)
        
        # Return average keyword relevance
        return sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0
    
    def calculate_readability_score(self, text: str) -> float:
        """Calculate readability score (0-1, higher = more readable)"""
        if not text:
            return 0.0
        
        # Basic readability metrics
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        syllables = self._count_syllables(text)
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Flesch Reading Ease approximation
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1 range
        normalized_score = max(0, min(1, flesch_score / 100))
        
        return normalized_score
    
    def _count_syllables(self, text: str) -> int:
        """Approximate syllable count"""
        vowels = "aeiouy"
        syllables = 0
        prev_was_vowel = False
        
        for char in text.lower():
            if char in vowels:
                if not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent e
        if text.endswith('e'):
            syllables -= 1
        
        return max(1, syllables)
    
    def calculate_section_importance(self, section_title: str, section_level: str) -> float:
        """Calculate importance based on section title and level"""
        importance_weights = {
            'h1': 1.0,
            'h2': 0.8,
            'h3': 0.6,
            'title': 1.0,
            'abstract': 0.9,
            'summary': 0.9,
            'conclusion': 0.8,
            'introduction': 0.7,
            'methodology': 0.6,
            'results': 0.7,
            'discussion': 0.6,
            'references': 0.3
        }
        
        # Base importance from level
        base_importance = importance_weights.get(section_level.lower(), 0.5)
        
        # Adjust based on title keywords
        title_lower = section_title.lower()
        title_boost = 0.0
        
        important_keywords = [
            'summary', 'conclusion', 'results', 'findings', 'analysis',
            'key', 'important', 'main', 'primary', 'executive', 'overview'
        ]
        
        for keyword in important_keywords:
            if keyword in title_lower:
                title_boost += 0.1
        
        return min(1.0, base_importance + title_boost)
    
    def calculate_recency_score(self, text: str) -> float:
        """Calculate recency score based on temporal indicators"""
        recency_keywords = {
            'recent': 0.9,
            'latest': 0.9,
            'current': 0.8,
            'new': 0.7,
            'updated': 0.7,
            '2024': 0.9,
            '2023': 0.8,
            '2022': 0.6,
            'today': 1.0,
            'now': 0.9
        }
        
        text_lower = text.lower()
        max_score = 0.0
        
        for keyword, score in recency_keywords.items():
            if keyword in text_lower:
                max_score = max(max_score, score)
        
        return max_score
    
    def calculate_authority_score(self, text: str, metadata: Dict[str, Any] = None) -> float:
        """Calculate authority score based on source credibility indicators"""
        authority_indicators = {
            'peer-reviewed': 0.9,
            'journal': 0.8,
            'university': 0.8,
            'research': 0.7,
            'study': 0.7,
            'analysis': 0.6,
            'report': 0.6,
            'official': 0.8,
            'published': 0.7
        }
        
        text_lower = text.lower()
        scores = []
        
        for indicator, score in authority_indicators.items():
            if indicator in text_lower:
                scores.append(score)
        
        # Also check metadata if available
        if metadata:
            source = metadata.get('source', '').lower()
            if any(domain in source for domain in ['.edu', '.gov', '.org']):
                scores.append(0.8)
        
        return max(scores) if scores else 0.5  # Default neutral score
    
    def calculate_mcda_score(self, 
                           text: str, 
                           persona: PersonaProfile, 
                           job_keywords: List[str],
                           section_title: str = "",
                           section_level: str = "h2",
                           metadata: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate multi-criteria decision analysis score"""
        
        # Create persona context for semantic similarity
        persona_context = f"{persona.persona_type.value} {' '.join(persona.expertise_areas)} {' '.join(persona.keywords)}"
        
        # Calculate individual criteria scores
        criteria_scores = {
            'semantic_relevance': self.calculate_semantic_similarity(text, persona_context),
            'readability': self.calculate_readability_score(text),
            'section_importance': self.calculate_section_importance(section_title, section_level),
            'recency': self.calculate_recency_score(text),
            'authority': self.calculate_authority_score(text, metadata)
        }
        
        # Add keyword relevance to semantic score
        keyword_relevance = self.calculate_keyword_relevance(text, job_keywords)
        criteria_scores['semantic_relevance'] = (criteria_scores['semantic_relevance'] + keyword_relevance) / 2
        
        # Calculate weighted MCDA score
        mcda_score = sum(
            criteria_scores[criterion] * self.base_mcda_weights[criterion]
            for criterion in criteria_scores
        )
        
        return {
            'mcda_score': mcda_score,
            'criteria_breakdown': criteria_scores,
            'weights_used': self.base_mcda_weights
        }
    
    def rank_sections(self, sections: List[Dict[str, Any]], 
                     persona: PersonaProfile, 
                     job_keywords: List[str]) -> List[Dict[str, Any]]:
        """Rank sections based on relevance scores"""
        
        scored_sections = []
        
        for section in sections:
            score_result = self.calculate_mcda_score(
                text=section.get('text', ''),
                persona=persona,
                job_keywords=job_keywords,
                section_title=section.get('title', ''),
                section_level=section.get('level', 'h2'),
                metadata=section.get('metadata', {})
            )
            
            section_with_score = section.copy()
            section_with_score['relevance_score'] = score_result['mcda_score']
            section_with_score['score_breakdown'] = score_result['criteria_breakdown']
            
            scored_sections.append(section_with_score)
        
        # Sort by relevance score (descending)
        scored_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        self.logger.info(f"Ranked {len(scored_sections)} sections")
        return scored_sections
    
    def filter_by_confidence(self, sections: List[Dict[str, Any]], 
                           min_confidence: float = 0.3) -> List[Dict[str, Any]]:
        """Filter sections by minimum confidence threshold"""
        filtered = [s for s in sections if s.get('relevance_score', 0) >= min_confidence]
        
        self.logger.info(f"Filtered {len(sections)} sections to {len(filtered)} with confidence >= {min_confidence}")
        return filtered