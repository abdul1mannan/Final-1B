"""
Adobe Hackathon Round 1B - Persona Intelligence Package
Persona-driven document intelligence for multi-document analysis
"""

from .persona_manager import PersonaManager
from .relevance_scorer import RelevanceScorer
from .document_analyzer import DocumentAnalyzer
from .multi_doc_processor import MultiDocumentProcessor

__version__ = "1.0.0"
__author__ = "Adopted"

# Main exports for easy import
__all__ = [
    'PersonaManager',
    'RelevanceScorer', 
    'DocumentAnalyzer',
    'MultiDocumentProcessor'
]

# Convenience function for quick processing
def process_documents_with_persona(pdf_paths, persona, job_to_be_done):
    """
    Convenience function for quick persona-driven document processing
    
    Args:
        pdf_paths: List of PDF file paths
        persona: Persona dictionary with role, expertise_areas, etc.
        job_to_be_done: String describing the task
        
    Returns:
        Dict with metadata, extracted_sections, and subsection_analysis
    """
    # Initialize components
    persona_manager = PersonaManager()
    relevance_scorer = RelevanceScorer()
    document_analyzer = DocumentAnalyzer()
    processor = MultiDocumentProcessor()
    
    # Process documents
    return processor.process_documents(
        pdf_paths=pdf_paths,
        persona=persona,
        job_to_be_done=job_to_be_done,
        persona_manager=persona_manager,
        relevance_scorer=relevance_scorer,
        document_analyzer=document_analyzer
    )