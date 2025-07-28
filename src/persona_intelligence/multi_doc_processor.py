"""
Multi-Document Processor - Orchestrates persona-driven document intelligence
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from persona_intelligence.persona_manager import PersonaManager, PersonaProfile
from persona_intelligence.relevance_scorer import RelevanceScorer
from persona_intelligence.document_analyzer import DocumentAnalyzer

class MultiDocumentProcessor:
    """Orchestrates multi-document processing with persona-driven intelligence"""
    
    def __init__(self, max_workers: int = None):
        self.logger = logging.getLogger(__name__)
        
        # Set max workers based on CPU count
        if max_workers is None:
            max_workers = min(4, multiprocessing.cpu_count())
        self.max_workers = max_workers
        
        self.logger.info(f"Initialized MultiDocumentProcessor with {max_workers} workers")
    
    def process_documents(self, 
                         pdf_paths: List[str],
                         persona: Dict[str, Any],
                         job_to_be_done: str,
                         persona_manager: PersonaManager,
                         relevance_scorer: RelevanceScorer,
                         document_analyzer: DocumentAnalyzer) -> Optional[Dict[str, Any]]:
        """
        Process multiple documents with persona-driven intelligence
        
        Args:
            pdf_paths: List of PDF file paths
            persona: Persona definition
            job_to_be_done: Job description
            persona_manager: PersonaManager instance
            relevance_scorer: RelevanceScorer instance
            document_analyzer: DocumentAnalyzer instance
            
        Returns:
            Dict containing the structured output
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing {len(pdf_paths)} documents for persona-driven intelligence")
            
            # Parse persona and job
            persona_profile = persona_manager.parse_persona(persona)
            job_analysis = persona_manager.analyze_job_to_be_done(job_to_be_done, original_persona=persona)
            
            self.logger.info(f"Persona: {persona_profile.persona_type.value}")
            self.logger.info(f"Job categories: {job_analysis.get('job_categories', [])}")
            
            # Process documents in parallel
            document_analyses = self._process_documents_parallel(
                pdf_paths, document_analyzer
            )
            
            if not document_analyses:
                self.logger.error("No documents were successfully processed")
                return None
            
            # Extract and score sections
            all_sections = self._extract_all_sections(document_analyses)
            
            # Score sections based on persona and job
            scored_sections = relevance_scorer.rank_sections(
                sections=all_sections,
                persona=persona_profile,
                job_keywords=job_analysis.get('priority_keywords', [])
            )
            
            # Filter by confidence threshold
            relevant_sections = relevance_scorer.filter_by_confidence(
                sections=scored_sections,
                min_confidence=0.3
            )
            
            # Generate sub-section analysis
            subsection_analysis = self._generate_subsection_analysis(
                relevant_sections, persona_profile, job_analysis
            )
            
            # Create output structure
            output = self._create_output_structure(
                documents=document_analyses,
                persona=persona_profile,
                job_analysis=job_analysis,
                relevant_sections=relevant_sections,
                subsection_analysis=subsection_analysis,
                processing_time=time.time() - start_time
            )
            
            self.logger.info(f"Successfully processed {len(pdf_paths)} documents in {time.time() - start_time:.2f}s")
            return output
            
        except Exception as e:
            self.logger.error(f"Error in multi-document processing: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_documents_parallel(self, 
                                   pdf_paths: List[str],
                                   document_analyzer: DocumentAnalyzer) -> List[Dict[str, Any]]:
        """Process documents in parallel"""
        document_analyses = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(document_analyzer.analyze_document, path): path
                for path in pdf_paths
            }
            
            # Collect results
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    if result:
                        document_analyses.append(result)
                        self.logger.info(f"✅ Processed: {os.path.basename(path)}")
                    else:
                        self.logger.warning(f"❌ Failed to process: {os.path.basename(path)}")
                except Exception as e:
                    self.logger.error(f"❌ Error processing {path}: {e}")
        
        return document_analyses
    
    def _extract_all_sections(self, document_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract all sections from document analyses"""
        all_sections = []
        
        for doc_analysis in document_analyses:
            doc_path = doc_analysis.get('document_path', '')
            doc_title = doc_analysis.get('title', '')
            
            for section in doc_analysis.get('sections', []):
                section_with_doc = section.copy()
                section_with_doc['document_path'] = doc_path
                section_with_doc['document_title'] = doc_title
                section_with_doc['section_id'] = f"{os.path.basename(doc_path)}_{section.get('title', 'untitled')}"
                
                all_sections.append(section_with_doc)
        
        self.logger.info(f"Extracted {len(all_sections)} sections from {len(document_analyses)} documents")
        return all_sections
    
    def _generate_subsection_analysis(self, 
                                    relevant_sections: List[Dict[str, Any]],
                                    persona_profile: PersonaProfile,
                                    job_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed sub-section analysis"""
        subsection_analysis = []
        
        for section in relevant_sections:
            subsections = section.get('subsections', [])
            
            for subsection in subsections:
                # Score subsection relevance
                subsection_score = self._score_subsection_relevance(
                    subsection, persona_profile, job_analysis
                )
                
                if subsection_score > 0.3:  # Threshold for inclusion
                    refined_text = self._refine_subsection_text(
                        subsection.get('text', ''), persona_profile
                    )
                    
                    subsection_analysis.append({
                        'id': f"{section.get('section_id', 'unknown')}_{subsection.get('id', 'subsec')}",
                        'document': os.path.basename(section.get('document_path', '')),
                        'section_title': section.get('title', ''),
                        'content_snippet': refined_text,
                        'relevance_score': subsection_score,
                        'page_number': section.get('page', 0),
                        'content_type': subsection.get('type', 'paragraph'),
                        'entities': subsection.get('entities', []),
                        'key_information': subsection.get('relevance_indicators', {})
                    })
        
        # Sort by relevance score
        subsection_analysis.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Limit to top results based on persona reading time
        max_subsections = min(50, len(subsection_analysis))  # Reasonable limit
        subsection_analysis = subsection_analysis[:max_subsections]
        
        self.logger.info(f"Generated analysis for {len(subsection_analysis)} sub-sections")
        return subsection_analysis
    
    def _score_subsection_relevance(self, 
                                   subsection: Dict[str, Any],
                                   persona_profile: PersonaProfile,
                                   job_analysis: Dict[str, Any]) -> float:
        """Score subsection relevance"""
        text = subsection.get('text', '')
        if not text:
            return 0.0
        
        # Basic keyword matching
        job_keywords = job_analysis.get('priority_keywords', [])
        persona_keywords = persona_profile.keywords
        
        all_keywords = job_keywords + persona_keywords
        text_lower = text.lower()
        
        keyword_matches = sum(1 for keyword in all_keywords if keyword.lower() in text_lower)
        keyword_score = min(1.0, keyword_matches / len(all_keywords)) if all_keywords else 0.0
        
        # Content type bonus
        content_type = subsection.get('type', 'paragraph')
        type_bonus = {
            'key_value_pairs': 0.3,
            'numbered_lists': 0.2,
            'bullet_points': 0.1,
            'paragraph': 0.0
        }.get(content_type, 0.0)
        
        # Length penalty for very short or very long text
        length_score = 1.0
        if len(text) < 50:
            length_score = 0.5
        elif len(text) > 1000:
            length_score = 0.8
        
        # Information density bonus
        info_density = subsection.get('relevance_indicators', {}).get('information_density', 0.0)
        
        final_score = (keyword_score * 0.6 + type_bonus * 0.2 + 
                      length_score * 0.1 + info_density * 0.1)
        
        return min(1.0, final_score)
    
    def _refine_subsection_text(self, text: str, persona_profile: PersonaProfile) -> str:
        """Refine subsection text based on persona preferences"""
        if not text:
            return text
        
        # Truncate based on persona detail level
        max_length = {
            'summary': 200,
            'detailed': 500,
            'comprehensive': 1000
        }.get(persona_profile.detail_level, 500)
        
        if len(text) <= max_length:
            return text
        
        # Smart truncation - try to end at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        
        if last_period > max_length * 0.8:  # If we can find a good cut point
            return truncated[:last_period + 1]
        else:
            return truncated + "..."
    
    def _create_output_structure(self, 
                               documents: List[Dict[str, Any]],
                               persona: PersonaProfile,
                               job_analysis: Dict[str, Any],
                               relevant_sections: List[Dict[str, Any]],
                               subsection_analysis: List[Dict[str, Any]],
                               processing_time: float) -> Dict[str, Any]:
        """Create the final output structure matching sample format"""
        
        # Get original persona from job analysis
        original_persona = job_analysis.get('original_persona', '')
        
        # Create metadata exactly like sample
        metadata = {
            'input_documents': [os.path.basename(doc.get('document_path', '')) for doc in documents],
            'persona': original_persona,  # Use original persona string
            'job_to_be_done': job_analysis.get('original_description', ''),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Extract sections with simple format
        extracted_sections = []
        for i, section in enumerate(relevant_sections[:20]):  # Limit to top 20
            extracted_sections.append({
                'document': os.path.basename(section.get('document_path', '')),
                'section_title': section.get('title', ''),
                'importance_rank': i + 1,
                'page_number': section.get('page', 0)
            })
        
        # Create subsection analysis with simple format
        simple_subsection_analysis = []
        for subsection in subsection_analysis[:20]:  # Limit to top 20
            simple_subsection_analysis.append({
                'document': subsection.get('document', ''),
                'refined_text': subsection.get('content_snippet', ''),
                'page_number': subsection.get('page_number', 0)
            })
        
        # Create final output structure matching sample exactly
        output = {
            'metadata': metadata,
            'extracted_sections': extracted_sections,
            'subsection_analysis': simple_subsection_analysis
        }
        
        return output
    
    def validate_output_format(self, output: Dict[str, Any]) -> bool:
        """Validate output format meets Round 1B requirements"""
        required_fields = ['metadata', 'extracted_sections', 'subsection_analysis']
        
        for field in required_fields:
            if field not in output:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        # Validate metadata format
        metadata = output.get('metadata', {})
        required_metadata_fields = ['input_documents', 'persona', 'job_to_be_done', 'processing_timestamp']
        for field in required_metadata_fields:
            if field not in metadata:
                self.logger.error(f"Missing required metadata field: {field}")
                return False
        
        # Validate extracted sections format
        for section in output.get('extracted_sections', []):
            required_section_fields = ['document', 'section_title', 'importance_rank', 'page_number']
            for field in required_section_fields:
                if field not in section:
                    self.logger.error(f"Missing required section field: {field}")
                    return False
        
        # Validate subsection analysis format
        for subsection in output.get('subsection_analysis', []):
            required_subsection_fields = ['document', 'refined_text', 'page_number']
            for field in required_subsection_fields:
                if field not in subsection:
                    self.logger.error(f"Missing required subsection field: {field}")
                    return False
        
        self.logger.info("Output format validation passed")
        return True