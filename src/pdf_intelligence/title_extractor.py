#!/usr/bin/env python3
"""
Adobe Hackathon Round 1A - Title Extraction
Multi-strategy approach: metadata → font analysis → position analysis
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import fitz  # PyMuPDF

from .utils import text_processor, font_analyzer, validation_helper

class TitleExtractor:
    """Multi-strategy title extraction"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.title_config = config.get('title_extraction', {})
        self.strategies = self.title_config.get('strategies', {})
        self.validation = self.title_config.get('validation', {})
        
        # Title validation parameters
        self.min_length = self.validation.get('min_length', 5)
        self.max_length = self.validation.get('max_length', 200)
        self.max_words = self.validation.get('max_words', 20)
    
    def extract_title(self, pdf_path: str, text_elements: Optional[List[Dict]] = None) -> str:
        """Extract title using multiple strategies"""
        
        title_candidates = []
        
        # Strategy 1: PDF Metadata (highest priority)
        metadata_title = self._extract_from_metadata(pdf_path)
        if metadata_title:
            title_candidates.append({
                'title': metadata_title,
                'confidence': 0.9,
                'source': 'metadata'
            })
        
        # Strategy 2: First page font analysis
        if text_elements:
            font_title = self._extract_from_font_analysis(text_elements)
            if font_title:
                title_candidates.append({
                    'title': font_title,
                    'confidence': 0.8,
                    'source': 'font_analysis'
                })
            
            # Strategy 3: First page position analysis
            position_title = self._extract_from_position_analysis(text_elements)
            if position_title:
                title_candidates.append({
                    'title': position_title,
                    'confidence': 0.7,
                    'source': 'position_analysis'
                })
            
            # Strategy 4: First significant text
            first_text_title = self._extract_first_significant_text(text_elements)
            if first_text_title:
                title_candidates.append({
                    'title': first_text_title,
                    'confidence': 0.6,
                    'source': 'first_text'
                })
        
        # Strategy 5: Filename fallback
        filename_title = self._extract_from_filename(pdf_path)
        if filename_title:
            title_candidates.append({
                'title': filename_title,
                'confidence': 0.3,
                'source': 'filename'
            })
        
        # Select best candidate
        best_title = self._select_best_title(title_candidates)
        return best_title if best_title else "Untitled Document"
    
    def _extract_from_metadata(self, pdf_path: str) -> Optional[str]:
        """Extract title from PDF metadata"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            doc.close()
            
            # Try different metadata fields
            title_fields = ['title', 'Title', 'subject', 'Subject']
            
            if metadata:
                for field in title_fields:
                    if field in metadata and metadata[field]:
                        title = metadata[field].strip()
                        if self._validate_title(title):
                            return text_processor.normalize_text(title)
            
        except Exception as e:
            print(f"⚠️ Metadata extraction failed: {e}")
        
        return None
    
    def _extract_from_font_analysis(self, text_elements: List[Dict]) -> Optional[str]:
        """Extract title based on font size analysis"""
        if not text_elements:
            return None
        
        # Get font statistics
        font_stats = font_analyzer.analyze_font_properties(text_elements)
        if not font_stats or 'size_distribution' not in font_stats:
            return None
        
        # Find largest font elements on first page
        first_page_elements = [e for e in text_elements if e.get('page', 1) == 1]
        
        if not first_page_elements:
            return None
        
        # Get 90th percentile font size (likely headings/titles)
        large_font_threshold = font_stats['size_distribution']['p90']
        
        # Find elements with large fonts
        large_font_elements = [
            e for e in first_page_elements 
            if e.get('font_size', 0) >= large_font_threshold
        ]
        
        if not large_font_elements:
            return None
        
        # Sort by font size (largest first) then by position (top first)
        large_font_elements.sort(key=lambda x: (-x.get('font_size', 0), x.get('bbox', [0, 0, 0, 0])[1]))
        
        # Check top candidates
        for element in large_font_elements[:3]:  # Check top 3
            text = element.get('text', '').strip()
            if self._validate_title(text) and self._is_likely_title(text):
                return text_processor.normalize_text(text)
        
        return None
    
    def _extract_from_position_analysis(self, text_elements: List[Dict]) -> Optional[str]:
        """Extract title based on position (top of first page)"""
        if not text_elements:
            return None
        
        # Get first page elements
        first_page_elements = [e for e in text_elements if e.get('page', 1) == 1]
        
        if not first_page_elements:
            return None
        
        # Sort by vertical position (top to bottom)
        first_page_elements.sort(key=lambda x: x.get('bbox', [0, 0, 0, 0])[1])
        
        # Check elements in top 20% of page
        page_height = max([e.get('bbox', [0, 0, 0, 0])[3] for e in first_page_elements], default=792)
        top_threshold = page_height * 0.2
        
        top_elements = [
            e for e in first_page_elements 
            if e.get('bbox', [0, 0, 0, 0])[1] <= top_threshold
        ]
        
        # Find longest meaningful text in top area
        for element in top_elements:
            text = element.get('text', '').strip()
            if self._validate_title(text) and len(text) > 10:  # Reasonable title length
                # Check if it looks like a title
                if self._is_likely_title(text):
                    return text_processor.normalize_text(text)
        
        return None
    
    def _extract_first_significant_text(self, text_elements: List[Dict]) -> Optional[str]:
        """Extract first significant text block as title fallback"""
        if not text_elements:
            return None
        
        # Get first page elements
        first_page_elements = [e for e in text_elements if e.get('page', 1) == 1]
        
        # Sort by position
        first_page_elements.sort(key=lambda x: (x.get('bbox', [0, 0, 0, 0])[1], x.get('bbox', [0, 0, 0, 0])[0]))
        
        # Find first significant text (not just a few characters)
        for element in first_page_elements:
            text = element.get('text', '').strip()
            if len(text) >= 10 and self._validate_title(text):
                # Avoid obvious non-titles
                if not self._is_obviously_not_title(text):
                    return text_processor.normalize_text(text)
        
        return None
    
    def _extract_from_filename(self, pdf_path: str) -> Optional[str]:
        """Extract title from filename as last resort"""
        try:
            filename = Path(pdf_path).stem  # Remove extension
            
            # Clean filename
            title = filename.replace('_', ' ').replace('-', ' ')
            title = re.sub(r'[^\w\s]', ' ', title)  # Remove special chars
            title = re.sub(r'\s+', ' ', title).strip()  # Normalize spaces
            title = title.title()  # Title case
            
            if self._validate_title(title):
                return title
                
        except Exception:
            pass
        
        return None
    
    def _validate_title(self, title: str) -> bool:
        """Validate if text can be a title"""
        if not title or not isinstance(title, str):
            return False
        
        title = title.strip()
        
        # Length checks
        if len(title) < self.min_length or len(title) > self.max_length:
            return False
        
        # Word count check
        word_count = len(title.split())
        if word_count > self.max_words:
            return False
        
        # Must contain letters
        if not re.search(r'[a-zA-Z]', title):
            return False
        
        # Avoid obvious non-titles
        if self._is_obviously_not_title(title):
            return False
        
        return True
    
    def _is_likely_title(self, text: str) -> bool:
        """Check if text looks like a title"""
        text_lower = text.lower().strip()
        
        # Positive indicators
        positive_score = 0
        
        # Title case or all caps
        if text.istitle() or text.isupper():
            positive_score += 1
        
        # Reasonable length for title
        if 10 <= len(text) <= 100:
            positive_score += 1
        
        # Contains meaningful words
        meaningful_words = ['analysis', 'study', 'report', 'guide', 'manual', 'review', 'research']
        if any(word in text_lower for word in meaningful_words):
            positive_score += 1
        
        # Negative indicators
        negative_score = 0
        
        # Too many numbers (might be a code or reference)
        if len(re.findall(r'\d', text)) > len(text) * 0.3:
            negative_score += 1
        
        # Contains common non-title phrases
        non_title_phrases = ['page', 'figure', 'table', 'appendix', 'section', 'chapter']
        if any(phrase in text_lower for phrase in non_title_phrases):
            negative_score += 1
        
        return positive_score > negative_score
    
    def _is_obviously_not_title(self, text: str) -> bool:
        """Check if text is obviously not a title"""
        text_lower = text.lower().strip()
        
        # Common non-title patterns
        non_title_patterns = [
            r'^page\s+\d+',      # "Page 1"
            r'^figure\s+\d+',    # "Figure 1"  
            r'^table\s+\d+',     # "Table 1"
            r'^\d+\.\d+\.\d+',   # Version numbers
            r'^copyright',       # Copyright notices
            r'^all rights',      # Rights notices
            r'^\w+@\w+',         # Email addresses
            r'^http[s]?://',     # URLs
        ]
        
        for pattern in non_title_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Too short meaningful content
        if len(text.strip()) < 5:
            return True
        
        # Mostly numbers or symbols
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars < len(text) * 0.5:
            return True
        
        return False
    
    def _select_best_title(self, candidates: List[Dict]) -> Optional[str]:
        """Select best title from candidates"""
        if not candidates:
            return None
        
        # Sort by confidence (highest first)
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Additional validation for top candidate
        best_candidate = candidates[0]
        
        # If confidence is very low, try to improve by combining strategies
        if best_candidate['confidence'] < 0.5 and len(candidates) > 1:
            # Try to find consensus between strategies
            titles = [c['title'] for c in candidates]
            if len(set(titles)) < len(titles):  # Some agreement
                # Find most common title
                title_counts = {}
                for title in titles:
                    title_counts[title] = title_counts.get(title, 0) + 1
                
                most_common = max(title_counts.keys(), key=lambda x: title_counts[x])
                return most_common
        
        return best_candidate['title']