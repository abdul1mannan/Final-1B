#!/usr/bin/env python3
"""
Adobe Hackathon Round 1A - Heading Detection Engine
Hybrid cascading pipeline: Rules → ML → Confidence scoring
"""

import re
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import yaml
from pathlib import Path

from .utils import text_processor, font_analyzer, position_analyzer, validation_helper

@dataclass
class HeadingCandidate:
    """Data class for heading candidates"""
    text: str
    page: int
    bbox: List[float]
    font_size: float
    font_name: str
    flags: int
    rule_confidence: float
    ml_confidence: float
    final_confidence: float
    predicted_level: int
    source: str  # 'rules', 'ml', 'hybrid'

class RuleBasedDetector:
    """Fast rule-based heading detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.font_thresholds = config.get('heading_detection', {}).get('font_thresholds', {})
        self.position_rules = config.get('heading_detection', {}).get('position_rules', {})
        
    def detect_candidates(self, text_elements: List[Dict], font_stats: Dict) -> List[HeadingCandidate]:
        """Detect heading candidates using rules"""
        candidates = []
        
        for element in text_elements:
            text = element.get('text', '').strip()
            if not text or len(text) < 2:
                continue
            
            # Calculate rule-based confidence
            confidence, level = self._calculate_rule_confidence(element, text, font_stats)
            
            if confidence >= 0.3:  # Minimum threshold for consideration
                candidate = HeadingCandidate(
                    text=text,
                    page=element.get('page', 1),
                    bbox=element.get('bbox', [0, 0, 0, 0]),
                    font_size=element.get('font_size', 0),
                    font_name=element.get('font_name', ''),
                    flags=element.get('flags', 0),
                    rule_confidence=confidence,
                    ml_confidence=0.0,
                    final_confidence=confidence,
                    predicted_level=level,
                    source='rules'
                )
                candidates.append(candidate)
        
        return candidates
    
    def _calculate_rule_confidence(self, element: Dict, text: str, font_stats: Dict) -> Tuple[float, int]:
        """Calculate confidence score based on rules"""
        confidence = 0.0
        level = 1
        
        # 1. Numbering pattern detection (high confidence)
        numbering = text_processor.extract_numbering(text)
        if numbering:
            confidence += numbering['confidence']
            level = numbering['level_hint']
        
        # 2. Font size analysis (when available)
        if font_stats:
            is_heading, font_conf, font_level = font_analyzer.is_heading_by_font(element, font_stats)
            if is_heading:
                confidence += font_conf * 0.7  # Weight font evidence
                if font_level > 0:
                    level = font_level
        
        # 3. Heading keywords
        keyword_conf = text_processor.is_likely_heading_keyword(text)
        confidence += keyword_conf * 0.5
        
        # 4. Text statistics
        stats = text_processor.calculate_text_statistics(text)
        
        # Ideal heading length (5-50 characters)
        if 5 <= stats['length'] <= 50:
            confidence += 0.2
        elif stats['length'] > 100:
            confidence -= 0.3  # Too long for heading
        
        # Capitalization patterns
        if stats['title_case'] or stats['all_caps']:
            confidence += 0.2
        
        # No ending punctuation (typical for headings)
        if not stats['has_punctuation']:
            confidence += 0.1
        
        # 5. Position analysis
        position = position_analyzer.analyze_position(element)
        if position.get('is_left_aligned', False):
            confidence += 0.1
        
        return min(confidence, 1.0), level

class MLClassifier:
    """TinyBERT-based heading classifier"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_path = Path(config['tinybert']['model_path'])
        self.confidence_threshold = config['tinybert']['confidence_threshold']
        self.max_length = config['tinybert']['max_length']
        
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load TinyBERT model"""
        try:
            if self.model_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    local_files_only=True
                )
                self.model = AutoModel.from_pretrained(
                    str(self.model_path),
                    local_files_only=True
                )
                self.model.eval()
                print(f"✅ TinyBERT loaded from {self.model_path}")
            else:
                print(f"⚠️ TinyBERT model not found at {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load TinyBERT: {e}")
            self.tokenizer = None
            self.model = None
    
    def classify_candidates(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Classify candidates using TinyBERT"""
        if not self.model or not self.tokenizer:
            print("⚠️ TinyBERT not available, skipping ML classification")
            return candidates
        
        # Only classify uncertain candidates (rule confidence < 0.7)
        uncertain_candidates = [c for c in candidates if c.rule_confidence < 0.7]
        certain_candidates = [c for c in candidates if c.rule_confidence >= 0.7]
        
        if not uncertain_candidates:
            return candidates
        
        print(f"🤖 TinyBERT classifying {len(uncertain_candidates)} uncertain candidates...")
        
        # Batch process for efficiency
        try:
            for candidate in uncertain_candidates:
                ml_conf, ml_level = self._classify_single(candidate.text)
                candidate.ml_confidence = ml_conf
                
                # Update level if ML is more confident
                if ml_conf > candidate.rule_confidence and ml_level > 0:
                    candidate.predicted_level = ml_level
                
                # Combine scores
                candidate.final_confidence = self._combine_scores(
                    candidate.rule_confidence, 
                    candidate.ml_confidence
                )
                candidate.source = 'hybrid'
        
        except Exception as e:
            print(f"❌ ML classification failed: {e}")
            # Fallback to rule-based scores
            for candidate in uncertain_candidates:
                candidate.ml_confidence = 0.0
                candidate.final_confidence = candidate.rule_confidence
                candidate.source = 'rules_fallback'
        
        return certain_candidates + uncertain_candidates
    
    def _classify_single(self, text: str) -> Tuple[float, int]:
        """Classify single text as heading"""
        if not self.model or not self.tokenizer:
            return 0.0, 0
        
        try:
            # Prepare input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
            
            # Simple classification based on embedding properties
            # This is a heuristic approach - in production you'd train a classifier
            embedding_norm = torch.norm(embeddings).item()
            embedding_mean = torch.mean(embeddings).item()
            
            # Heuristic scoring (can be improved with training data)
            confidence = self._heuristic_classification(text, embedding_norm, embedding_mean)
            level = self._predict_level_from_text(text)
            
            return confidence, level
            
        except Exception as e:
            print(f"❌ Single classification failed: {e}")
            return 0.0, 0
    
    def _heuristic_classification(self, text: str, norm: float, mean: float) -> float:
        """Heuristic classification based on embeddings and text features"""
        confidence = 0.0
        
        # Text-based features
        words = text.split()
        
        # Length features
        if 2 <= len(words) <= 8:  # Ideal heading length
            confidence += 0.3
        
        # Embedding-based features (heuristic)
        if norm > 10.0:  # Strong activation
            confidence += 0.2
        
        if abs(mean) < 0.1:  # Balanced activation
            confidence += 0.1
        
        # Text pattern features
        if text[0].isupper():  # Starts with capital
            confidence += 0.1
        
        if not text.endswith('.'):  # No ending period
            confidence += 0.1
        
        # Common heading patterns
        if re.match(r'^(CHAPTER|SECTION|PART|CONCLUSION|INTRODUCTION)', text, re.IGNORECASE):
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def _predict_level_from_text(self, text: str) -> int:
        """Predict heading level from text content"""
        text_lower = text.lower()
        
        # H1 indicators
        if any(keyword in text_lower for keyword in ['chapter', 'introduction', 'conclusion', 'abstract']):
            return 1
        
        # H2 indicators  
        if any(keyword in text_lower for keyword in ['section', 'background', 'methodology', 'results']):
            return 2
        
        # H3 indicators
        if any(keyword in text_lower for keyword in ['subsection', 'example', 'case study']):
            return 3
        
        # Default based on length
        word_count = len(text.split())
        if word_count <= 3:
            return 1
        elif word_count <= 6:
            return 2
        else:
            return 3
    
    def _combine_scores(self, rule_score: float, ml_score: float) -> float:
        """Combine rule and ML confidence scores"""
        # Weighted average favoring rules for speed
        combined = 0.6 * rule_score + 0.4 * ml_score
        return min(combined, 1.0)

class HierarchyValidator:
    """Ensure logical heading hierarchy"""
    
    def validate_and_adjust_hierarchy(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Validate and adjust heading hierarchy"""
        if not candidates:
            return candidates
        
        # Sort by page and position
        sorted_candidates = sorted(candidates, key=lambda x: (x.page, x.bbox[1] if x.bbox else 0))
        
        # Adjust levels to ensure logical progression
        adjusted_candidates = []
        last_level = 0
        
        for candidate in sorted_candidates:
            # Ensure no skipping levels (H1 -> H3 not allowed)
            if candidate.predicted_level > last_level + 1:
                candidate.predicted_level = last_level + 1
            
            # Update tracking
            last_level = candidate.predicted_level
            adjusted_candidates.append(candidate)
        
        return adjusted_candidates

class HeadingDetector:
    """Main heading detection coordinator"""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.rule_detector = RuleBasedDetector(self.config)
        self.ml_classifier = MLClassifier(self.config)
        self.hierarchy_validator = HierarchyValidator()
        
        self.min_confidence = 0.5  # Minimum confidence for final selection
    
    def detect_headings(self, text_elements: List[Dict]) -> List[Dict]:
        """Main heading detection pipeline"""
        start_time = time.time()
        
        # Step 1: Analyze font properties
        font_stats = font_analyzer.analyze_font_properties(text_elements)
        
        # Step 2: Rule-based candidate detection
        candidates = self.rule_detector.detect_candidates(text_elements, font_stats)
        print(f"📋 Found {len(candidates)} candidates from rules")
        
        # Step 3: ML classification for uncertain candidates
        candidates = self.ml_classifier.classify_candidates(candidates)
        
        # Step 4: Filter by confidence threshold
        filtered_candidates = [c for c in candidates if c.final_confidence >= self.min_confidence]
        print(f"🎯 {len(filtered_candidates)} candidates above confidence threshold")
        
        # Step 5: Validate and adjust hierarchy
        final_candidates = self.hierarchy_validator.validate_and_adjust_hierarchy(filtered_candidates)
        
        # Step 6: Convert to output format
        headings = []
        for candidate in final_candidates:
            heading = validation_helper.create_outline_entry(
                candidate.predicted_level,
                candidate.text,
                candidate.page
            )
            headings.append(heading)
        
        processing_time = time.time() - start_time
        print(f"⚡ Heading detection completed in {processing_time:.3f}s")
        
        return headings
    
    def get_detection_stats(self, candidates: List[HeadingCandidate]) -> Dict[str, Any]:
        """Get statistics about detection process"""
        if not candidates:
            return {}
        
        sources = [c.source for c in candidates]
        levels = [c.predicted_level for c in candidates]
        confidences = [c.final_confidence for c in candidates]
        
        return {
            'total_candidates': len(candidates),
            'sources': {
                'rules': sources.count('rules'),
                'ml': sources.count('ml'),
                'hybrid': sources.count('hybrid'),
                'fallback': sources.count('rules_fallback')
            },
            'levels': {
                'h1': levels.count(1),
                'h2': levels.count(2),
                'h3': levels.count(3)
            },
            'confidence_stats': {
                'mean': np.mean(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'std': np.std(confidences)
            }
        }