#!/usr/bin/env python3
"""
Adobe Hackathon Round 1B - PDF Intelligence Module
Enhanced PDF processing for multi-document persona intelligence
"""
from typing import Optional
# Version info
__version__ = "1.1.0"
__author__ = "Adopted"
__description__ = "PDF Intelligence for Persona-Driven Document Analysis (1B Enhanced)"

# Core components - Import main classes only
from .parser import PDFOutlineExtractor
from .heading_detector import HeadingDetector
from .title_extractor import TitleExtractor
from .validator import OutputValidator, JSONFormatter

# Utility functions - Import commonly used utilities
from .utils import (
    TextProcessor,
    FontAnalyzer, 
    PositionAnalyzer,
    ValidationHelper
)

# Performance monitoring
from .parser import PerformanceMonitor

# Main entry points for easy access
__all__ = [
    # Primary classes
    'PDFOutlineExtractor',
    'HeadingDetector', 
    'TitleExtractor',
    'OutputValidator',
    'JSONFormatter',
    
    # Utility classes
    'TextProcessor',
    'FontAnalyzer',
    'PositionAnalyzer', 
    'ValidationHelper',
    'PerformanceMonitor',
    
    # Module info
    '__version__',
    '__author__',
    '__description__'
]

# Convenience functions for quick access
def extract_pdf_outline(pdf_path: str, config_path: Optional[str] = None) -> dict:
    """
    Quick function to extract outline from PDF
    
    Args:
        pdf_path: Path to PDF file
        config_path: Optional config file path
        
    Returns:
        Dictionary with title and outline
    """
    extractor = PDFOutlineExtractor(config_path)
    return extractor.extract_outline(pdf_path)

def validate_output(output: dict) -> bool:
    """
    Quick function to validate output format
    
    Args:
        output: Output dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    validator = OutputValidator()
    result = validator.validate_and_clean(output)
    return result.is_valid

# Add convenience functions to __all__
__all__.extend(['extract_pdf_outline', 'validate_output'])