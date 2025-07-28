#!/usr/bin/env python3
"""
Adobe Hackathon Round 1A - Core Utilities
Text processing, font analysis, and validation helpers
"""

import re
import unicodedata
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path



class TextProcessor:
    """Advanced text processing for PDF content"""
    
    def __init__(self):
        # Multilingual numbering patterns (bonus points)
        self.numbering_patterns = [
            r'^\d+\.\s+',                    # "1. ", "2. "
            r'^\d+\.\d+\s+',                 # "1.1 ", "2.3 "
            r'^\d+\.\d+\.\d+\s+',            # "1.1.1 "
            r'^Chapter\s+\d+',               # "Chapter 1"
            r'^Section\s+\d+',               # "Section 2"
            r'^Part\s+[IVX]+',               # "Part I", "Part IV"
            r'^第\d+章',                     # Japanese chapters
            r'^الفصل\s+\d+',                 # Arabic chapters
            r'^Глава\s+\d+',                 # Russian chapters
            r'^[A-Z]\.\s+',                  # "A. ", "B. "
            r'^\([0-9]+\)\s+',               # "(1) ", "(2) "
            r'^\w+\)\s+',                    # "a) ", "i) "
        ]
        
        # Heading keywords (language agnostic where possible)
        self.heading_keywords = {
            'introduction', 'conclusion', 'background', 'methodology',
            'results', 'discussion', 'references', 'appendix', 'abstract',
            'summary', 'overview', 'objectives', 'scope', 'findings',
            'recommendations', 'acknowledgments', 'bibliography',
            # Multilingual keywords
            'はじめに', '結論', '要約',  # Japanese
            'مقدمة', 'خلاصة', 'ملخص',   # Arabic
            'введение', 'заключение'    # Russian
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing"""
        if not text:
            return ""
        
        # Unicode normalization (handles accents, ligatures)
        text = unicodedata.normalize('NFKC', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove soft hyphens and other control characters
        text = re.sub(r'[\u00AD\u200B-\u200D\uFEFF]', '', text)
        
        return text
    
    def extract_numbering(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract numbering information from text"""
        text = text.strip()
        
        for pattern in self.numbering_patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                numbering = match.group(0).strip()
                remaining_text = text[len(numbering):].strip()
                
                # Determine heading level from numbering
                level = self._determine_level_from_numbering(numbering)
                
                return {
                    'numbering': numbering,
                    'text': remaining_text,
                    'level_hint': level,
                    'pattern': pattern,
                    'confidence': 0.9  # High confidence for numbered headings
                }
        
        return None
    
    def _determine_level_from_numbering(self, numbering: str) -> int:
        """Determine heading level from numbering pattern"""
        # Count dots for hierarchical numbering
        dot_count = numbering.count('.')
        if dot_count > 0:
            return min(dot_count, 3)  # Max H3
        
        # Chapter/Section indicators are usually H1
        if re.search(r'(chapter|section|part|第.*章|الفصل|глава)', numbering, re.IGNORECASE):
            return 1
        
        # Single letters/numbers are usually H2
        if re.match(r'^[A-Za-z0-9]\s*[\.\)]\s*$', numbering):
            return 2
        
        return 1  # Default to H1
    
    def is_likely_heading_keyword(self, text: str) -> float:
        """Check if text contains heading keywords - return confidence"""
        text_lower = text.lower().strip()
        
        # Exact matches
        if text_lower in self.heading_keywords:
            return 0.95
        
        # Partial matches
        for keyword in self.heading_keywords:
            if keyword in text_lower:
                return 0.7
        
        # Pattern-based detection
        if re.match(r'^(introduction|conclusion|summary|overview|abstract)$', text_lower):
            return 0.9
        
        return 0.0
    
    def calculate_text_statistics(self, text: str) -> Dict[str, Any]:
        """Calculate useful statistics for heading detection"""
        if not text:
            return {'length': 0, 'word_count': 0, 'avg_word_length': 0}
        
        words = text.split()
        
        return {
            'length': len(text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'has_punctuation': bool(re.search(r'[.!?]$', text.strip())),
            'capitalization_ratio': sum(1 for c in text if c.isupper()) / len(text),
            'all_caps': text.isupper() and len(text) > 2,
            'title_case': text.istitle(),
            'starts_with_capital': text[0].isupper() if text else False
        }

class FontAnalyzer:
    """Analyze font properties for heading detection"""
    
    def __init__(self):
        self.font_size_threshold_ratios = {
            'h1': 1.4,  # 40% larger than body
            'h2': 1.25, # 25% larger than body  
            'h3': 1.1   # 10% larger than body
        }
    
    def analyze_font_properties(self, text_elements: List[Dict]) -> Dict[str, Any]:
        """Analyze font properties across document"""
        if not text_elements:
            return {}
        
        font_sizes = []
        font_names = []
        
        for element in text_elements:
            if 'font_size' in element and element['font_size'] > 0:
                font_sizes.append(element['font_size'])
            if 'font_name' in element:
                font_names.append(element['font_name'])
        
        if not font_sizes:
            return {}
        
        font_sizes = np.array(font_sizes)
        
        return {
            'body_font_size': np.percentile(font_sizes, 50),  # Median
            'font_size_std': np.std(font_sizes),
            'font_size_min': np.min(font_sizes),
            'font_size_max': np.max(font_sizes),
            'font_size_range': np.max(font_sizes) - np.min(font_sizes),
            'common_fonts': self._get_common_fonts(font_names),
            'size_distribution': {
                'p25': np.percentile(font_sizes, 25),
                'p50': np.percentile(font_sizes, 50),
                'p75': np.percentile(font_sizes, 75),
                'p90': np.percentile(font_sizes, 90)
            }
        }
    
    def _get_common_fonts(self, font_names: List[str]) -> List[str]:
        """Get most common font names"""
        if not font_names:
            return []
        
        font_counts = {}
        for font in font_names:
            font_counts[font] = font_counts.get(font, 0) + 1
        
        # Return top 3 most common fonts
        return sorted(font_counts.keys(), key=lambda x: font_counts[x], reverse=True)[:3]
    
    def is_heading_by_font(self, element: Dict, font_stats: Dict) -> Tuple[bool, float, int]:
        """Determine if element is heading based on font properties"""
        if not font_stats or 'body_font_size' not in font_stats:
            return False, 0.0, 0
        
        element_size = element.get('font_size', 0)
        body_size = font_stats['body_font_size']
        
        if element_size <= 0 or body_size <= 0:
            return False, 0.0, 0
        
        size_ratio = element_size / body_size
        
        # Check if font indicates bold/italic
        font_name = element.get('font_name', '').lower()
        is_bold = 'bold' in font_name or element.get('flags', 0) & 16  # Bold flag
        is_italic = 'italic' in font_name or element.get('flags', 0) & 2  # Italic flag
        
        confidence = 0.0
        level = 0
        
        # Font size based detection
        if size_ratio >= self.font_size_threshold_ratios['h1']:
            confidence += 0.6
            level = 1
        elif size_ratio >= self.font_size_threshold_ratios['h2']:
            confidence += 0.5
            level = 2
        elif size_ratio >= self.font_size_threshold_ratios['h3']:
            confidence += 0.4
            level = 3
        
        # Style based boost
        if is_bold:
            confidence += 0.2
        if is_italic:
            confidence += 0.1
        
        # Font name based detection
        if any(keyword in font_name for keyword in ['heading', 'title', 'header']):
            confidence += 0.3
        
        is_heading = confidence >= 0.4
        return is_heading, min(confidence, 1.0), level


class PositionAnalyzer:
    """Analyze text positioning for heading detection"""
    
    def __init__(self):
        self.left_margin_threshold = 100  # pixels
        self.spacing_threshold = 20       # minimum spacing around headings
    
    def analyze_position(self, element: Dict, context_elements: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Analyze position characteristics of text element"""
        bbox = element.get('bbox', [0, 0, 0, 0])
        if len(bbox) < 4:
            return {}
        
        x0, y0, x1, y1 = bbox[:4]
        
        analysis = {
            'left_margin': x0,
            'top_position': y0,
            'width': x1 - x0,
            'height': y1 - y0,
            'is_left_aligned': x0 <= self.left_margin_threshold,
            'aspect_ratio': (x1 - x0) / (y1 - y0) if (y1 - y0) > 0 else 0
        }
        
        # Analyze spacing context if provided
        if context_elements:
            spacing = self._analyze_spacing(element, context_elements)
            analysis.update(spacing)
        
        return analysis
    
    def _analyze_spacing(self, element: Dict, context_elements: List[Dict]) -> Dict[str, Any]:
        """Analyze spacing around element"""
        element_bbox = element.get('bbox', [0, 0, 0, 0])
        if len(element_bbox) < 4:
            return {}
        
        _, ey0, _, ey1 = element_bbox
        
        # Find elements above and below
        above_elements = [e for e in context_elements 
                         if e.get('bbox', [0, 0, 0, 0])[3] < ey0]  # y1 < element y0
        below_elements = [e for e in context_elements 
                         if e.get('bbox', [0, 0, 0, 0])[1] > ey1]  # y0 > element y1
        
        spacing_above = min([ey0 - e.get('bbox', [0, 0, 0, 0])[3] 
                            for e in above_elements], default=float('inf'))
        spacing_below = min([e.get('bbox', [0, 0, 0, 0])[1] - ey1 
                            for e in below_elements], default=float('inf'))
        
        return {
            'spacing_above': spacing_above if spacing_above != float('inf') else 0,
            'spacing_below': spacing_below if spacing_below != float('inf') else 0,
            'has_adequate_spacing': (spacing_above >= self.spacing_threshold or 
                                   spacing_below >= self.spacing_threshold)
        }

class ValidationHelper:
    """Validation utilities for JSON output"""
    
    @staticmethod
    def validate_heading_level(level: str) -> bool:
        """Validate heading level format"""
        return level in ['H1', 'H2', 'H3']
    
    @staticmethod
    def validate_page_number(page: int, max_pages: int = 50) -> bool:
        """Validate page number"""
        return isinstance(page, int) and 1 <= page <= max_pages
    
    @staticmethod
    def validate_heading_text(text: str) -> bool:
        """Validate heading text"""
        if not isinstance(text, str):
            return False
        text = text.strip()
        return 2 <= len(text) <= 200  # Reasonable heading length
    
    @staticmethod
    def clean_heading_text(text: str) -> str:
        """Clean and normalize heading text"""
        if not text:
            return ""
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove trailing punctuation that's not meaningful
        text = re.sub(r'[\.,:;]+$', '', text)
        
        # Normalize internal whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle common OCR artifacts
        text = re.sub(r'[^\w\s\-\.,:;!?()\[\]{}""''`~@#$%^&*+=<>/\\|]', '', text)
        
        return text
    
    @staticmethod
    def create_outline_entry(level: int, text: str, page: int) -> Dict[str, Any]:
        """Create standardized outline entry"""
        # Convert level number to string format
        level_map = {1: 'H1', 2: 'H2', 3: 'H3'}
        level_str = level_map.get(level, 'H1')
        
        # Clean text
        clean_text = ValidationHelper.clean_heading_text(text)
        
        return {
            'level': level_str,
            'text': clean_text,
            'page': page
        }
    
    @staticmethod
    def validate_outline_structure(outline: List[Dict]) -> Tuple[bool, str]:
        """Validate complete outline structure"""
        if not isinstance(outline, list):
            return False, "Outline must be a list"
        
        for i, entry in enumerate(outline):
            # Check required fields
            required_fields = ['level', 'text', 'page']
            for field in required_fields:
                if field not in entry:
                    return False, f"Entry {i} missing required field: {field}"
            
            # Validate individual fields
            if not ValidationHelper.validate_heading_level(entry['level']):
                return False, f"Entry {i} has invalid level: {entry['level']}"
            
            if not ValidationHelper.validate_heading_text(entry['text']):
                return False, f"Entry {i} has invalid text: {entry['text']}"
            
            if not ValidationHelper.validate_page_number(entry['page']):
                return False, f"Entry {i} has invalid page: {entry['page']}"
        
        return True, "Valid outline structure"

# Global utility instances (singleton pattern for performance)
text_processor = TextProcessor()
font_analyzer = FontAnalyzer()
position_analyzer = PositionAnalyzer()
validation_helper = ValidationHelper()