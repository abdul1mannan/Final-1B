#!/usr/bin/env python3
"""
Adobe Hackathon Round 1A - Output Validator
JSON compliance, error handling, and quality assurance
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import time

@dataclass
class ValidationResult:
    """Result of validation process"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_output: Optional[Dict]
    validation_time: float

class OutputValidator:
    """Comprehensive output validation and cleaning"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.validation_rules = self.config.get('validation', {})
        
        # Validation parameters
        self.max_title_length = self.validation_rules.get('max_title_length', 200)
        self.max_heading_length = self.validation_rules.get('max_heading_length', 200)
        self.min_heading_length = self.validation_rules.get('min_heading_length', 2)
        self.max_outline_entries = self.validation_rules.get('max_outline_entries', 500)
        self.max_pages = self.validation_rules.get('max_pages', 50)
        
        # Valid heading levels
        self.valid_levels = ['H1', 'H2', 'H3']
        
        # Quality thresholds
        self.min_quality_score = 0.7
    
    def validate_and_clean(self, raw_output: Dict) -> ValidationResult:
        """Main validation and cleaning pipeline"""
        start_time = time.time()
        
        errors = []
        warnings = []
        cleaned_output = None
        
        try:
            # Step 1: Basic structure validation
            structure_valid, structure_errors = self._validate_basic_structure(raw_output)
            errors.extend(structure_errors)
            
            if not structure_valid:
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    cleaned_output=None,
                    validation_time=time.time() - start_time
                )
            
            # Step 2: Content validation (validate original data before cleaning)
            content_valid, content_errors, content_warnings = self._validate_content(raw_output)
            errors.extend(content_errors)
            warnings.extend(content_warnings)
            
            # Step 3: Clean and normalize data (only if no critical errors)
            cleaned_output = self._clean_output(raw_output)
            
            # Step 4: Hierarchy validation (on cleaned data for structure check)
            hierarchy_valid, hierarchy_errors, hierarchy_warnings = self._validate_hierarchy(cleaned_output)
            errors.extend(hierarchy_errors)
            warnings.extend(hierarchy_warnings)
            
            # Step 5: Quality assessment
            quality_warnings = self._assess_quality(cleaned_output)
            warnings.extend(quality_warnings)
            
            # Step 6: Final cleanup
            if not errors:
                cleaned_output = self._final_cleanup(cleaned_output)
            
            is_valid = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Validation failed with exception: {str(e)}")
            is_valid = False
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            cleaned_output=cleaned_output if is_valid else None,
            validation_time=time.time() - start_time
        )
    
    def _validate_basic_structure(self, output: Dict) -> Tuple[bool, List[str]]:
        """Validate basic JSON structure"""
        errors = []
        
        # Check if it's a dictionary
        if not isinstance(output, dict):
            errors.append("Output must be a JSON object (dictionary)")
            return False, errors
        
        # Check required top-level fields
        required_fields = ['title', 'outline']
        for field in required_fields:
            if field not in output:
                errors.append(f"Missing required field: '{field}'")
        
        # Check title type
        if 'title' in output and not isinstance(output['title'], str):
            errors.append("'title' field must be a string")
        
        # Check outline type
        if 'outline' in output:
            if not isinstance(output['outline'], list):
                errors.append("'outline' field must be a list")
            else:
                # Check outline entries
                for i, entry in enumerate(output['outline']):
                    if not isinstance(entry, dict):
                        errors.append(f"Outline entry {i} must be an object")
                        continue
                    
                    # Check required entry fields
                    entry_required = ['level', 'text', 'page']
                    for field in entry_required:
                        if field not in entry:
                            errors.append(f"Outline entry {i} missing required field: '{field}'")
        
        return len(errors) == 0, errors
    
    def _validate_content(self, output: Dict) -> Tuple[bool, List[str], List[str]]:
        """Validate content quality and constraints"""
        errors = []
        warnings = []
        
        # Validate title
        title = output.get('title', '')
        if not title.strip():
            errors.append("Title cannot be empty")
        elif len(title) > self.max_title_length:
            errors.append(f"Title too long ({len(title)} > {self.max_title_length} characters)")
        elif len(title.strip()) < 3:
            warnings.append("Title is very short (< 3 characters)")
        
        # Validate outline
        outline = output.get('outline', [])
        
        if len(outline) > self.max_outline_entries:
            errors.append(f"Too many outline entries ({len(outline)} > {self.max_outline_entries})")
        
        if len(outline) == 0:
            warnings.append("No outline entries found")
        
        # Validate individual entries
        page_numbers = []
        for i, entry in enumerate(outline):
            # Validate level
            level = entry.get('level', '')
            if level not in self.valid_levels:
                errors.append(f"Entry {i}: Invalid level '{level}'. Must be one of {self.valid_levels}")
            
            # Validate text
            text = entry.get('text', '')
            if not text.strip():
                errors.append(f"Entry {i}: Text cannot be empty")
            elif len(text) > self.max_heading_length:
                errors.append(f"Entry {i}: Text too long ({len(text)} > {self.max_heading_length} characters)")
            elif len(text.strip()) < self.min_heading_length:
                errors.append(f"Entry {i}: Text too short ({len(text.strip())} < {self.min_heading_length} characters)")
            
            # Validate page
            page = entry.get('page', 0)
            if not isinstance(page, int):
                errors.append(f"Entry {i}: Page must be an integer")
            elif page < 1:
                errors.append(f"Entry {i}: Page must be >= 1")
            elif page > self.max_pages:
                errors.append(f"Entry {i}: Page {page} exceeds maximum ({self.max_pages})")
            else:
                page_numbers.append(page)
        
        # Check page number consistency
        if page_numbers:
            if max(page_numbers) - min(page_numbers) > self.max_pages:
                warnings.append(f"Large page range detected: {min(page_numbers)}-{max(page_numbers)}")
        
        return len(errors) == 0, errors, warnings
    
    def _validate_hierarchy(self, output: Dict) -> Tuple[bool, List[str], List[str]]:
        """Validate heading hierarchy logic"""
        errors = []
        warnings = []
        
        outline = output.get('outline', [])
        if not outline:
            return True, errors, warnings
        
        # Sort by page and assumed position
        sorted_outline = sorted(outline, key=lambda x: x.get('page', 1))
        
        # Check hierarchy progression
        level_map = {'H1': 1, 'H2': 2, 'H3': 3}
        prev_level = 0
        
        for i, entry in enumerate(sorted_outline):
            level_str = entry.get('level', 'H1')
            current_level = level_map.get(level_str, 1)
            
            # Check for level skipping
            if current_level > prev_level + 1:
                warnings.append(f"Possible level skip at entry {i}: jumped from H{prev_level} to {level_str}")
            
            prev_level = current_level
        
        # Check for reasonable distribution
        level_counts = {'H1': 0, 'H2': 0, 'H3': 0}
        for entry in outline:
            level = entry.get('level', 'H1')
            level_counts[level] += 1
        
        # Warn about unusual distributions
        total_entries = len(outline)
        if total_entries > 5:
            h1_ratio = level_counts['H1'] / total_entries
            if h1_ratio > 0.8:
                warnings.append("Very high H1 ratio - check heading level detection")
            elif h1_ratio < 0.1:
                warnings.append("Very low H1 ratio - document might be missing main sections")
        
        return True, errors, warnings
    
    def _assess_quality(self, output: Dict) -> List[str]:
        """Assess output quality and provide warnings"""
        warnings = []
        
        title = output.get('title', '')
        outline = output.get('outline', [])
        
        # Title quality checks
        if title:
            if title.lower() in ['untitled document', 'untitled', 'document']:
                warnings.append("Generic title detected - consider manual review")
            
            if re.match(r'^\w+\.\w+$', title):  # filename-like
                warnings.append("Title appears to be filename-based")
        
        # Outline quality checks
        if outline:
            # Check for very short headings
            short_headings = [e for e in outline if len(e.get('text', '').strip()) < 5]
            if len(short_headings) > len(outline) * 0.3:
                warnings.append("Many very short headings detected")
            
            # Check for duplicates
            texts = [e.get('text', '').strip().lower() for e in outline]
            unique_texts = set(texts)
            if len(unique_texts) < len(texts):
                duplicate_count = len(texts) - len(unique_texts)
                warnings.append(f"{duplicate_count} duplicate heading(s) detected")
            
            # Check for numbering consistency
            numbered_count = sum(1 for e in outline if re.match(r'^\d+\.', e.get('text', '')))
            if 0 < numbered_count < len(outline) * 0.5:
                warnings.append("Inconsistent numbering pattern detected")
        
        return warnings
    
    def _clean_output(self, output: Dict) -> Dict:
        """Clean and normalize output data"""
        cleaned = {
            'title': self._clean_title(output.get('title', '')),
            'outline': []
        }
        
        outline = output.get('outline', [])
        for entry in outline:
            cleaned_entry = self._clean_outline_entry(entry)
            if cleaned_entry:  # Only add valid entries
                cleaned['outline'].append(cleaned_entry)
        
        return cleaned
    
    def _clean_title(self, title: str) -> str:
        """Clean and normalize title"""
        if not isinstance(title, str):
            return "Untitled Document"
        
        # Basic cleaning
        title = title.strip()
        
        # Remove multiple whitespace
        title = re.sub(r'\s+', ' ', title)
        
        # Remove leading/trailing punctuation
        title = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', title)
        
        # Normalize quotes
        title = re.sub(r'[""''`]', '"', title)
        
        # Ensure reasonable length
        if len(title) > self.max_title_length:
            title = title[:self.max_title_length].rsplit(' ', 1)[0] + '...'
        
        return title if title else "Untitled Document"
    
    def _clean_outline_entry(self, entry: Dict) -> Optional[Dict]:
        """Clean and normalize outline entry"""
        if not isinstance(entry, dict):
            return None
        
        # Extract and clean fields
        level = entry.get('level', 'H1')
        text = entry.get('text', '')
        page = entry.get('page', 1)
        
        # Validate and clean level
        if level not in self.valid_levels:
            level = 'H1'  # Default fallback
        
        # Clean text
        if isinstance(text, str):
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = re.sub(r'^[^\w\s]+|[^\w\s]*$', '', text)  # Remove leading/trailing punct
            
            # Ensure reasonable length
            if len(text) > self.max_heading_length:
                text = text[:self.max_heading_length].rsplit(' ', 1)[0] + '...'
        else:
            text = str(text).strip() if text else ''
        
        # Validate text length
        if len(text) < self.min_heading_length:
            return None  # Skip invalid entries
        
        # Clean page number
        try:
            page = int(page)
            if page < 1:
                page = 1
            elif page > self.max_pages:
                page = self.max_pages
        except (ValueError, TypeError):
            page = 1
        
        return {
            'level': level,
            'text': text,
            'page': page
        }
    
    def _final_cleanup(self, output: Dict) -> Dict:
        """Final cleanup and optimization"""
        # Remove duplicate entries
        outline = output.get('outline', [])
        seen = set()
        unique_outline = []
        
        for entry in outline:
            # Create signature for deduplication
            signature = (entry['level'], entry['text'].lower().strip(), entry['page'])
            if signature not in seen:
                seen.add(signature)
                unique_outline.append(entry)
        
        # Sort by page then by level priority
        level_priority = {'H1': 1, 'H2': 2, 'H3': 3}
        unique_outline.sort(key=lambda x: (x['page'], level_priority.get(x['level'], 1)))
        
        return {
            'title': output['title'],
            'outline': unique_outline
        }
    
    def create_fallback_output(self, title: Optional[str] = None, error_message: Optional[str] = None) -> Dict:
        """Create fallback output when processing fails"""
        fallback_title = title or "Document Processing Failed"
        
        fallback_outline = []
        if error_message:
            fallback_outline.append({
                'level': 'H1',
                'text': f"Error: {error_message}",
                'page': 1
            })
        
        return {
            'title': fallback_title,
            'outline': fallback_outline
        }
    
    def save_validation_report(self, result: ValidationResult, output_path: str) -> None:
        """Save detailed validation report"""
        report = {
            'validation_result': {
                'is_valid': result.is_valid,
                'validation_time': result.validation_time,
                'error_count': len(result.errors),
                'warning_count': len(result.warnings)
            },
            'errors': result.errors,
            'warnings': result.warnings,
            'timestamp': time.time()
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save validation report: {e}")

class JSONFormatter:
    """Format JSON output for submission"""
    
    @staticmethod
    def format_for_submission(output: Dict, pretty: bool = True) -> str:
        """Format JSON for submission"""
        try:
            if pretty:
                return json.dumps(output, indent=2, ensure_ascii=False)
            else:
                return json.dumps(output, ensure_ascii=False)
        except Exception as e:
            # Fallback to basic formatting
            return json.dumps({
                'title': 'JSON Formatting Error',
                'outline': [{'level': 'H1', 'text': f'Error: {str(e)}', 'page': 1}]
            }, indent=2)
    
    @staticmethod
    def validate_json_syntax(json_string: str) -> Tuple[bool, str]:
        """Validate JSON syntax"""
        try:
            json.loads(json_string)
            return True, "Valid JSON"
        except json.JSONDecodeError as e:
            return False, f"JSON syntax error: {str(e)}"
    
    @staticmethod
    def minify_json(json_string: str) -> str:
        """Minify JSON string"""
        try:
            obj = json.loads(json_string)
            return json.dumps(obj, separators=(',', ':'), ensure_ascii=False)
        except Exception:
            return json_string

# Global validator instance
output_validator = OutputValidator()
json_formatter = JSONFormatter()