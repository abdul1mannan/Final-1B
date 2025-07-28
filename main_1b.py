#!/usr/bin/env python3
"""
Adobe Hackathon Round 1B - Persona-Driven Document Intelligence
Main entry point for multi-document processing with persona-based relevance scoring
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import local modules
from src.pdf_intelligence.parser import PDFOutlineExtractor
from src.persona_intelligence.persona_manager import PersonaManager
from src.persona_intelligence.relevance_scorer import RelevanceScorer
from src.persona_intelligence.document_analyzer import DocumentAnalyzer
from src.persona_intelligence.multi_doc_processor import MultiDocumentProcessor

def setup_logging(level: int = logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def validate_constraints(processing_time: float, max_time: float, memory_usage: Optional[float], max_memory: float) -> bool:
    """Validate performance constraints"""
    time_ok = processing_time <= max_time
    memory_ok = True
    if memory_usage is not None:
        memory_ok = memory_usage <= max_memory
    
    if not time_ok:
        logging.warning(f"Time constraint violated: {processing_time:.2f}s > {max_time}s")
    if not memory_ok:
        logging.warning(f"Memory constraint violated: {memory_usage:.2f}MB > {max_memory}MB")
        
    return time_ok and memory_ok

def load_test_case(test_case_path: str) -> Dict[str, Any]:
    """Load test case configuration"""
    try:
        with open(test_case_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load test case {test_case_path}: {e}")
        return None

def process_documents(input_dir: str, output_dir: str, test_case_file: str = None) -> bool:
    """
    Process multiple documents with persona-driven intelligence
    
    Args:
        input_dir: Directory containing PDFs and test case JSON
        output_dir: Directory for output JSON files
        test_case_file: Optional specific test case file
    
    Returns:
        bool: Success status
    """
    start_time = time.time()
    
    try:
        # Find test case file
        if test_case_file:
            test_case_path = os.path.join(input_dir, test_case_file)
        else:
            # Look for test case JSON files recursively
            test_case_files = [str(p) for p in Path(input_dir).rglob('*.json')]
            if not test_case_files:
                logging.error("No test case JSON found in input directory")
                return False
            test_case_path = test_case_files[0]
            logging.info(f"Found test case: {test_case_path}")
        
        # Load test case
        test_case = load_test_case(test_case_path)
        if not test_case:
            return False
        
        logging.info(f"Processing test case: {test_case.get('job_to_be_done', 'Unknown')}")
        
        # Find PDF files recursively
        pdf_files = [str(p) for p in Path(input_dir).rglob('*.pdf')]
        if not pdf_files:
            logging.error("No PDF files found in input directory")
            return False
        
        pdf_paths = pdf_files  # These are already absolute paths
        logging.info(f"Found {len(pdf_paths)} PDF files to process")
        
        # Initialize components
        persona_manager = PersonaManager()
        relevance_scorer = RelevanceScorer()
        document_analyzer = DocumentAnalyzer()
        multi_processor = MultiDocumentProcessor()
        
        # Process documents
        result = multi_processor.process_documents(
            pdf_paths=pdf_paths,
            persona=test_case.get('persona', {}),
            job_to_be_done=test_case.get('job_to_be_done', ''),
            persona_manager=persona_manager,
            relevance_scorer=relevance_scorer,
            document_analyzer=document_analyzer
        )
        
        if not result:
            logging.error("Document processing failed")
            return False
        
        # Generate output filename
        # The JSON input will only contain persona and job_to_be_done, so we'll use a default filename.
        output_filename = 'challenge1b_output.json'
        output_path = os.path.join(output_dir, output_filename)
        
        # Save result
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Performance validation
        processing_time = time.time() - start_time
        logging.info(f"Processing completed in {processing_time:.2f} seconds")
        
        # Validate constraints
        constraints_met = validate_constraints(
            processing_time=processing_time,
            max_time=60.0,  # 60 seconds for 1B
            memory_usage=None,  # Will be monitored separately
            max_memory=1024  # 1GB for 1B
        )
        
        if not constraints_met:
            logging.warning("Performance constraints not met")
            return False
        
        logging.info(f"✅ Successfully processed {len(pdf_paths)} documents")
        logging.info(f"📄 Output saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error in document processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for Round 1B"""
    parser = argparse.ArgumentParser(description='Adobe Hackathon Round 1B - Persona-Driven Document Intelligence')
    
    # Get the current script directory and set default paths relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, 'input')
    default_output = os.path.join(script_dir, 'output')
    
    parser.add_argument('--input', default=default_input, help='Input directory path')
    parser.add_argument('--output', default=default_output, help='Output directory path')
    parser.add_argument('--test-case', help='Specific test case file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    logging.info("🚀 Starting Adobe Hackathon Round 1B - Persona-Driven Document Intelligence")
    logging.info(f"📂 Input directory: {args.input}")
    logging.info(f"📁 Output directory: {args.output}")
    
    # Validate input directory
    if not os.path.exists(args.input):
        logging.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)
    
    # Process documents
    success = process_documents(
        input_dir=args.input,
        output_dir=args.output,
        test_case_file=args.test_case
    )
    
    if success:
        logging.info("✅ Round 1B processing completed successfully")
        sys.exit(0)
    else:
        logging.error("❌ Round 1B processing failed")
        sys.exit(1)

if __name__ == "__main__":
    main()