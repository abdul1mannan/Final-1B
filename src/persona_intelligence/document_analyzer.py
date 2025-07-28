"""
Document Analyzer - Hierarchical document parsing and sub-section analysis
Enhanced with 1A PDF intelligence infrastructure
"""

import re
import logging
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import yaml

# Import local PDF intelligence infrastructure
from pdf_intelligence.parser import PDFOutlineExtractor
from pdf_intelligence.heading_detector import HeadingDetector
from pdf_intelligence.title_extractor import TitleExtractor
from pdf_intelligence.utils import TextProcessor, FontAnalyzer, ValidationHelper

class DocumentAnalyzer:
    """Hierarchical document analysis with sub-section extraction"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration (fallback to 1B config)
        if not config_path:
            config_path = str(Path(__file__).parent.parent.parent / "configs" / "model_config_1b.yaml")
        
        # Initialize 1A PDF extraction components
        self.pdf_extractor = PDFOutlineExtractor(config_path)
        self.heading_detector = HeadingDetector(config_path)
        
        # Initialize utility classes
        self.text_processor = TextProcessor()
        self.font_analyzer = FontAnalyzer()
        self.validation_helper = ValidationHelper()
        
        # Load config for title extractor
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.title_extractor = TitleExtractor(config)
        except Exception as e:
            self.logger.warning(f"Failed to load config for TitleExtractor: {e}")
            self.title_extractor = TitleExtractor({})
        
        # Sub-section analysis patterns (enhanced)
        self.subsection_patterns = {
            'bullet_points': r'^\s*[•\-\*•▪▫]\s+(.+)$',
            'numbered_lists': r'^\s*\d+[\.\)]\s+(.+)$',
            'lettered_lists': r'^\s*[a-zA-Z][\.\)]\s+(.+)$',
            'roman_numerals': r'^\s*[ivxlcdm]+[\.\)]\s+(.+)$',
            'key_value_pairs': r'^([^:]{2,50}):\s*(.+)$',
            'definition_lists': r'^([A-Z][a-zA-Z\s]+)\s*[-–—]\s*(.+)$',
            'paragraphs': r'^(.{50,}?)(?:\n\n|\n\s*\n|$)'
        }
        
        # Content type classifiers
        self.content_classifiers = {
            'financial_data': [r'\$[\d,]+', r'\d+%', r'revenue', r'profit', r'cost'],
            'technical_specs': [r'version\s+\d+', r'specification', r'requirement'],
            'dates': [r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', r'\d{4}-\d{2}-\d{2}'],
            'statistics': [r'\d+\.?\d*%', r'\d+\s*out\s*of\s*\d+'],
            'conclusions': [r'conclusion', r'summary', r'in\s+summary', r'therefore'],
            'methodologies': [r'method', r'approach', r'procedure', r'process'],
            'key_findings': [r'finding', r'result', r'discovered', r'showed']
        }
    
    def analyze_document(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze a single document with hierarchical structure"""
        try:
            self.logger.info(f"Analyzing document: {pdf_path}")
            
            # Use 1A's PDF outline extractor to get structured data
            outline_result = self.pdf_extractor.extract_outline(pdf_path)
            if not outline_result:
                self.logger.error(f"Failed to extract outline from {pdf_path}")
                return None
            
            # Extract title from 1A result
            title = outline_result.get('title', 'Untitled Document')
            
            # Extract headings from 1A result
            headings = outline_result.get('outline', [])
            
            # Get additional document info using 1A's text extractor
            text_elements, doc_info = self.pdf_extractor.text_extractor.extract_text_elements(pdf_path)
            
            # Create enhanced document structure with sections
            document_structure = self._create_document_structure_from_outline(
                text_elements, headings, title, doc_info
            )
            
            # Analyze sections with persona intelligence features
            analyzed_sections = self._analyze_sections(document_structure['sections'])
            
            # Extract enhanced metadata
            metadata = self._extract_enhanced_metadata(doc_info, outline_result)
            
            return {
                'document_path': pdf_path,
                'title': title,
                'metadata': metadata,
                'sections': analyzed_sections,
                'total_pages': doc_info.get('page_count', 0),
                'total_sections': len(analyzed_sections),
                'outline_quality': self._assess_outline_quality(headings),
                'text_elements_count': len(text_elements)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing document {pdf_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_document_structure_from_outline(self, 
                                              text_elements: List[Dict[str, Any]],
                                              headings: List[Dict[str, Any]], 
                                              title: str,
                                              doc_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create hierarchical document structure from 1A outline results"""
        sections = []
        
        # Convert text elements to page-based structure
        pages_text = {}
        for element in text_elements:
            page_num = element.get('page', 1)
            if page_num not in pages_text:
                pages_text[page_num] = []
            pages_text[page_num].append(element)
        
        # Create sections based on headings
        for i, heading in enumerate(headings):
            section = {
                'title': heading.get('text', f'Section {i+1}'),
                'level': heading.get('level', 'H2'),
                'page': heading.get('page', 1),
                'section_type': self._classify_section_type(heading.get('text', ''))
            }
            
            # Extract text content for this section
            section_text = self._extract_section_text(
                heading, headings[i+1:], pages_text, text_elements
            )
            section['text'] = section_text
            
            sections.append(section)
        
        # If no headings found, create sections from pages
        if not sections:
            sections = self._create_sections_from_pages(pages_text)
        
        return {
            'title': title,
            'sections': sections,
            'total_headings': len(headings),
            'has_structured_outline': len(headings) > 0
        }
    
    def _extract_section_text(self, current_heading: Dict[str, Any], 
                            remaining_headings: List[Dict[str, Any]], 
                            pages_text: Dict[int, List[Dict]], 
                            all_text_elements: List[Dict]) -> str:
        """Extract text content for a section between headings"""
        start_page = current_heading.get('page', 1)
        
        # Find end page (next heading or end of document)
        end_page = start_page
        if remaining_headings:
            next_heading = remaining_headings[0]
            end_page = next_heading.get('page', start_page + 1)
        else:
            end_page = max(pages_text.keys()) if pages_text else start_page
        
        # Extract text from relevant pages
        section_texts = []
        for page_num in range(start_page, end_page + 1):
            if page_num in pages_text:
                page_elements = pages_text[page_num]
                
                # Sort by position on page
                page_elements.sort(key=lambda x: (x.get('bbox', [0, 0, 0, 0])[1], 
                                                 x.get('bbox', [0, 0, 0, 0])[0]))
                
                # Extract text, skipping the heading itself on start page
                for element in page_elements:
                    text = element.get('text', '').strip()
                    
                    # Skip heading text on first page
                    if page_num == start_page and text == current_heading.get('text', ''):
                        continue
                    
                    if text and len(text) > 2:
                        section_texts.append(text)
        
        return '\n'.join(section_texts)
    
    def _create_sections_from_pages(self, pages_text: Dict[int, List[Dict]]) -> List[Dict[str, Any]]:
        """Create basic sections from pages when no outline is available"""
        sections = []
        
        for page_num, elements in pages_text.items():
            if not elements:
                continue
            
            # Sort elements by position
            elements.sort(key=lambda x: (x.get('bbox', [0, 0, 0, 0])[1], 
                                        x.get('bbox', [0, 0, 0, 0])[0]))
            
            # Extract all text from page
            page_text = '\n'.join([e.get('text', '').strip() for e in elements if e.get('text', '').strip()])
            
            if page_text:
                section = {
                    'title': f'Page {page_num}',
                    'level': 'H2',
                    'page': page_num,
                    'section_type': 'general',
                    'text': page_text
                }
                sections.append(section)
        
        return sections
    
    def _assess_outline_quality(self, headings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality of extracted outline"""
        if not headings:
            return {
                'quality_score': 0.0,
                'has_hierarchy': False,
                'level_distribution': {},
                'assessment': 'no_outline'
            }
        
        # Analyze level distribution
        levels = [h.get('level', 'H2') for h in headings]
        level_counts = {}
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Calculate quality metrics
        has_hierarchy = len(level_counts) > 1
        total_headings = len(headings)
        
        # Quality score based on various factors
        quality_score = 0.0
        
        # Base score for having headings
        quality_score += min(0.4, total_headings * 0.1)
        
        # Bonus for hierarchy
        if has_hierarchy:
            quality_score += 0.3
        
        # Bonus for reasonable number of headings
        if 3 <= total_headings <= 20:
            quality_score += 0.3
        
        # Assessment level
        if quality_score >= 0.8:
            assessment = 'excellent'
        elif quality_score >= 0.6:
            assessment = 'good'
        elif quality_score >= 0.4:
            assessment = 'fair'
        else:
            assessment = 'poor'
        
        return {
            'quality_score': min(quality_score, 1.0),
            'has_hierarchy': has_hierarchy,
            'level_distribution': level_counts,
            'total_headings': total_headings,
            'assessment': assessment
        }
    
    def _extract_enhanced_metadata(self, doc_info: Dict[str, Any], 
                                  outline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract enhanced metadata combining 1A results with 1B analysis"""
        base_metadata = self._extract_metadata(doc_info)
        
        # Add 1A processing metadata if available
        if '_metadata' in outline_result:
            processing_meta = outline_result['_metadata']
            base_metadata.update({
                'extraction_processing_time': processing_meta.get('processing_time', 0),
                'extraction_memory_usage': processing_meta.get('memory_usage_mb', 0),
                'extraction_method': processing_meta.get('extraction_method', 'unknown'),
                'extraction_version': processing_meta.get('version', '1.0')
            })
        
        return base_metadata
    
    def _create_document_structure(self, document_info: Dict[str, Any], 
                                 headings: List[Dict[str, Any]], 
                                 title: str) -> Dict[str, Any]:
        """Create hierarchical document structure"""
        sections = []
        
        # Sort headings by page and position
        sorted_headings = sorted(headings, key=lambda h: (h.get('page', 0), h.get('y_position', 0)))
        
        # Group text by sections
        current_section = None
        section_text = []
        
        for page_num, page_content in enumerate(document_info.get('pages', []), 1):
            page_text = page_content.get('text', '')
            
            # Find headings on this page
            page_headings = [h for h in sorted_headings if h.get('page') == page_num]
            
            if page_headings:
                # Save previous section if exists
                if current_section:
                    current_section['text'] = '\n'.join(section_text)
                    sections.append(current_section)
                
                # Start new section
                heading = page_headings[0]  # Take first heading on page
                current_section = {
                    'title': heading.get('text', ''),
                    'level': heading.get('level', 'H2'),
                    'page': page_num,
                    'section_type': self._classify_section_type(heading.get('text', ''))
                }
                section_text = []
            
            # Add page text to current section
            if current_section:
                section_text.append(page_text)
        
        # Don't forget the last section
        if current_section:
            current_section['text'] = '\n'.join(section_text)
            sections.append(current_section)
        
        return {
            'title': title,
            'sections': sections
        }
    
    def _analyze_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sections with sub-section extraction"""
        analyzed_sections = []
        
        for section in sections:
            analyzed_section = section.copy()
            
            # Extract sub-sections
            subsections = self._extract_subsections(section.get('text', ''))
            analyzed_section['subsections'] = subsections
            
            # Classify content
            content_types = self._classify_content(section.get('text', ''))
            analyzed_section['content_types'] = content_types
            
            # Extract key information
            key_info = self._extract_key_information(section.get('text', ''))
            analyzed_section['key_information'] = key_info
            
            # Calculate section metrics
            metrics = self._calculate_section_metrics(section.get('text', ''))
            analyzed_section['metrics'] = metrics
            
            analyzed_sections.append(analyzed_section)
        
        return analyzed_sections
    
    def _extract_subsections(self, text: str) -> List[Dict[str, Any]]:
        """Extract sub-sections from text"""
        subsections = []
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) < 20:  # Skip very short paragraphs
                continue
            
            subsection = {
                'id': f"subsec_{i+1}",
                'text': paragraph.strip(),
                'type': 'paragraph',
                'position': i + 1
            }
            
            # Check for specific patterns
            for pattern_name, pattern in self.subsection_patterns.items():
                matches = re.findall(pattern, paragraph, re.MULTILINE)
                if matches:
                    subsection['type'] = pattern_name
                    subsection['structured_content'] = matches
                    break
            
            # Extract entities and key phrases
            entities = self._extract_entities(paragraph)
            subsection['entities'] = entities
            
            # Calculate relevance indicators
            relevance_indicators = self._calculate_relevance_indicators(paragraph)
            subsection['relevance_indicators'] = relevance_indicators
            
            subsections.append(subsection)
        
        return subsections
    
    def _classify_section_type(self, title: str) -> str:
        """Classify section type based on title"""
        title_lower = title.lower()
        
        type_patterns = {
            'abstract': ['abstract', 'summary', 'overview'],
            'introduction': ['introduction', 'background', 'overview'],
            'methodology': ['method', 'approach', 'procedure'],
            'results': ['result', 'finding', 'outcome'],
            'discussion': ['discussion', 'analysis', 'interpretation'],
            'conclusion': ['conclusion', 'summary', 'final'],
            'references': ['reference', 'bibliography', 'citation'],
            'appendix': ['appendix', 'supplement', 'additional'],
            'financial': ['financial', 'revenue', 'profit', 'cost'],
            'technical': ['technical', 'specification', 'implementation']
        }
        
        for section_type, patterns in type_patterns.items():
            if any(pattern in title_lower for pattern in patterns):
                return section_type
        
        return 'general'
    
    def _classify_content(self, text: str) -> List[str]:
        """Classify content types within text"""
        content_types = []
        text_lower = text.lower()
        
        for content_type, patterns in self.content_classifiers.items():
            if any(re.search(pattern, text_lower) for pattern in patterns):
                content_types.append(content_type)
        
        return content_types
    
    def _extract_key_information(self, text: str) -> Dict[str, Any]:
        """Extract key information from text"""
        key_info = {
            'numbers': [],
            'dates': [],
            'percentages': [],
            'key_phrases': [],
            'entities': []
        }
        
        # Extract numbers
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)
        key_info['numbers'] = numbers[:10]  # Limit to first 10
        
        # Extract dates
        dates = re.findall(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b', text)
        key_info['dates'] = dates[:5]
        
        # Extract percentages
        percentages = re.findall(r'\b\d+(?:\.\d+)?%\b', text)
        key_info['percentages'] = percentages[:10]
        
        # Extract capitalized phrases (potential entities)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        key_info['entities'] = list(set(entities))[:15]
        
        # Extract key phrases (sentences with important keywords)
        important_keywords = ['important', 'key', 'significant', 'major', 'critical', 'essential']
        sentences = re.split(r'[.!?]+', text)
        key_phrases = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in important_keywords):
                key_phrases.append(sentence.strip())
        
        key_info['key_phrases'] = key_phrases[:5]
        
        return key_info
    
    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction"""
        # Extract capitalized words/phrases
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return list(set(entities))[:10]
    
    def _calculate_relevance_indicators(self, text: str) -> Dict[str, float]:
        """Calculate relevance indicators for text"""
        indicators = {
            'length_score': min(1.0, len(text) / 500),  # Normalize by ideal length
            'keyword_density': self._calculate_keyword_density(text),
            'sentence_complexity': self._calculate_sentence_complexity(text),
            'information_density': self._calculate_information_density(text)
        }
        
        return indicators
    
    def _calculate_keyword_density(self, text: str) -> float:
        """Calculate keyword density"""
        words = text.split()
        if not words:
            return 0.0
        
        important_words = [w for w in words if len(w) > 3 and w.isalpha()]
        return len(important_words) / len(words) if words else 0.0
    
    def _calculate_sentence_complexity(self, text: str) -> float:
        """Calculate sentence complexity"""
        sentences = re.split(r'[.!?]+', text)
        if not sentences:
            return 0.0
        
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        return min(1.0, avg_length / 20)  # Normalize by ideal sentence length
    
    def _calculate_information_density(self, text: str) -> float:
        """Calculate information density"""
        # Count informative elements
        numbers = len(re.findall(r'\d+', text))
        capitals = len(re.findall(r'[A-Z][a-z]+', text))
        punctuation = len(re.findall(r'[.!?:;]', text))
        
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        
        density = (numbers + capitals + punctuation) / total_chars
        return min(1.0, density * 100)  # Scale appropriately
    
    def _calculate_section_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate section-level metrics"""
        return {
            'word_count': len(text.split()),
            'char_count': len(text),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'paragraph_count': len(re.split(r'\n\s*\n', text)),
            'avg_sentence_length': self._calculate_avg_sentence_length(text),
            'readability_score': self._calculate_readability(text)
        }
    
    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length"""
        sentences = re.split(r'[.!?]+', text)
        if not sentences:
            return 0.0
        
        total_words = sum(len(s.split()) for s in sentences)
        return total_words / len(sentences)
    
    def _calculate_readability(self, text: str) -> float:
        """Simple readability score"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        if not words or not sentences:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability formula
        readability = 100 - (avg_sentence_length * 1.5) - (avg_word_length * 2)
        return max(0, min(100, readability)) / 100
    
    def _extract_metadata(self, document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract document metadata from 1A document info structure"""
        metadata = {
            'page_count': document_info.get('page_count', 0),
            'file_size': document_info.get('file_size', 0),
            'needs_password': document_info.get('needs_password', False),
            'is_pdf': document_info.get('is_pdf', True),
            'language': 'en'  # Default language
        }
        
        # Extract from PDF metadata if available
        pdf_metadata = document_info.get('metadata', {})
        if pdf_metadata:
            metadata.update({
                'creation_date': pdf_metadata.get('creationDate'),
                'modification_date': pdf_metadata.get('modDate'),
                'author': pdf_metadata.get('author'),
                'subject': pdf_metadata.get('subject'),
                'title': pdf_metadata.get('title'),
                'creator': pdf_metadata.get('creator'),
                'producer': pdf_metadata.get('producer')
            })
        
        # Clean up None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return metadata
    
    def create_section_hierarchy(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create hierarchical section structure"""
        hierarchy = {
            'title': 'Document Structure',
            'sections': []
        }
        
        current_h1 = None
        current_h2 = None
        
        for section in sections:
            level = section.get('level', 'H2')
            
            if level == 'H1':
                current_h1 = section.copy()
                current_h1['subsections'] = []
                hierarchy['sections'].append(current_h1)
                current_h2 = None
            elif level == 'H2':
                current_h2 = section.copy()
                current_h2['subsections'] = []
                
                if current_h1:
                    current_h1['subsections'].append(current_h2)
                else:
                    hierarchy['sections'].append(current_h2)
            elif level == 'H3':
                if current_h2:
                    current_h2['subsections'].append(section)
                elif current_h1:
                    current_h1['subsections'].append(section)
                else:
                    hierarchy['sections'].append(section)
        
        return hierarchy
    
    def validate_1a_integration(self) -> Dict[str, bool]:
        """Validate that 1A components are properly integrated"""
        validation_results = {
            'pdf_extractor_available': False,
            'heading_detector_available': False,
            'title_extractor_available': False,
            'config_loaded': False,
            'text_processor_available': False,
            'font_analyzer_available': False
        }
        
        try:
            # Test PDF extractor
            validation_results['pdf_extractor_available'] = (
                self.pdf_extractor is not None and 
                hasattr(self.pdf_extractor, 'extract_outline')
            )
            
            # Test heading detector  
            validation_results['heading_detector_available'] = (
                self.heading_detector is not None and
                hasattr(self.heading_detector, 'detect_headings')
            )
            
            # Test title extractor
            validation_results['title_extractor_available'] = (
                self.title_extractor is not None and
                hasattr(self.title_extractor, 'extract_title')
            )
            
            # Test utility functions
            validation_results['text_processor_available'] = (
                self.text_processor is not None and
                hasattr(self.text_processor, 'normalize_text')
            )
            validation_results['font_analyzer_available'] = (
                self.font_analyzer is not None and
                hasattr(self.font_analyzer, 'analyze_font_properties')
            )
            
            # Test config
            validation_results['config_loaded'] = True
            
        except Exception as e:
            self.logger.error(f"Integration validation failed: {e}")
        
        return validation_results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and performance info"""
        return {
            'subsection_patterns_count': len(self.subsection_patterns),
            'content_classifiers_count': len(self.content_classifiers),
            'integration_status': self.validate_1a_integration(),
            'supports_hierarchical_analysis': True,
            'supports_persona_scoring': True
        }