#!/usr/bin/env python3
"""
Adobe Hackathon Round 1A - Main PDF Processing Engine
Orchestrates the complete pipeline: PDF → Text → Headings → Validation → JSON
"""

import os
import sys
import time
import json
import traceback
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yaml
import fitz  # PyMuPDF
import psutil

# Import our components
from .utils import text_processor, font_analyzer, validation_helper
from .heading_detector import HeadingDetector
from .title_extractor import TitleExtractor
from .validator import OutputValidator, JSONFormatter

class PerformanceMonitor:
    """Monitor processing performance and constraints"""
    
    def __init__(self, max_time: float = 10.0, max_memory_mb: float = 8192):
        self.max_time = max_time
        self.max_memory_mb = max_memory_mb
        self.start_time = None
        self.start_memory = None
        
        self.process = psutil.Process()
    
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
    
    def check_constraints(self) -> Tuple[bool, str]:
        """Check if constraints are still met"""
        if self.start_time is None:
            return True, "Monitoring not started"
        
        # Check time constraint
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_time:
            return False, f"Time constraint exceeded: {elapsed_time:.2f}s > {self.max_time}s"
        
        # Check memory constraint
        current_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        if current_memory > self.max_memory_mb:
            return False, f"Memory constraint exceeded: {current_memory:.1f}MB > {self.max_memory_mb}MB"
        
        return True, "Within constraints"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if self.start_time is None:
            return {}
        
        elapsed_time = time.time() - self.start_time
        current_memory = self.process.memory_info().rss / (1024 * 1024)
        memory_delta = current_memory - (self.start_memory or 0)
        
        return {
            'elapsed_time': elapsed_time,
            'current_memory_mb': current_memory,
            'memory_delta_mb': memory_delta,
            'time_remaining': max(0, self.max_time - elapsed_time),
            'memory_usage_percent': (current_memory / self.max_memory_mb) * 100
        }

class PDFTextExtractor:
    """Extract text and metadata from PDF using PyMuPDF"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.extraction_config = config.get('text_extraction', {})
        
        # Extraction parameters
        self.extract_fonts = self.extraction_config.get('extract_fonts', True)
        self.extract_positions = self.extraction_config.get('extract_positions', True)
        self.min_text_length = self.extraction_config.get('min_text_length', 1)
        self.max_pages = self.extraction_config.get('max_pages', 50)
    
    def extract_text_elements(self, pdf_path: str) -> Tuple[List[Dict], Dict]:
        """Extract text elements with metadata from PDF"""
        text_elements = []
        document_info = {}
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            # Get document metadata
            document_info = {
                'page_count': len(doc),
                'metadata': doc.metadata,
                'needs_password': doc.needs_pass,
                'is_pdf': doc.is_pdf,
                'file_size': os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0
            }
            
            # Process pages (limit to max_pages)
            pages_to_process = min(len(doc), self.max_pages)
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                
                # Extract text with detailed information
                if self.extract_fonts and self.extract_positions:
                    # Get text with font and position information
                    text_dict = page.get_text("dict")  # type: ignore
                    elements = self._extract_from_dict(text_dict, page_num + 1)
                else:
                    # Fallback to basic text extraction
                    elements = self._extract_basic_text(page, page_num + 1)
                
                text_elements.extend(elements)
            
            doc.close()
            
        except Exception as e:
            print(f"❌ PDF extraction failed: {e}")
            # Create minimal fallback info
            document_info = {
                'page_count': 0,
                'metadata': {},
                'error': str(e)
            }
        
        return text_elements, document_info
    
    def _extract_from_dict(self, text_dict: Dict, page_num: int) -> List[Dict]:
        """Extract text elements from PyMuPDF text dict"""
        elements = []
        
        try:
            blocks = text_dict.get("blocks", [])
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    if "spans" not in line:
                        continue
                    
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        
                        if len(text) >= self.min_text_length:
                            element = {
                                'text': text,
                                'page': page_num,
                                'bbox': span.get("bbox", [0, 0, 0, 0]),
                                'font_name': span.get("font", "unknown"),
                                'font_size': span.get("size", 0),
                                'flags': span.get("flags", 0),
                                'color': span.get("color", 0),
                                'block_id': block.get("number", -1),
                                'line_id': line.get("number", -1)
                            }
                            elements.append(element)
        
        except Exception as e:
            print(f"⚠️ Dict extraction failed for page {page_num}: {e}")
        
        return elements
    
    def _extract_basic_text(self, page, page_num: int) -> List[Dict]:
        """Fallback basic text extraction"""
        elements = []
        
        try:
            # Get text blocks
            text_blocks = page.get_text("blocks")
            
            for i, block in enumerate(text_blocks):
                if len(block) >= 5:  # Ensure block has enough elements
                    text = block[4].strip()  # Text is at index 4
                    bbox = block[:4]  # Bbox is first 4 elements
                    
                    if len(text) >= self.min_text_length:
                        element = {
                            'text': text,
                            'page': page_num,
                            'bbox': list(bbox),
                            'font_name': 'unknown',
                            'font_size': 0,
                            'flags': 0,
                            'color': 0,
                            'block_id': i,
                            'line_id': 0
                        }
                        elements.append(element)
        
        except Exception as e:
            print(f"⚠️ Basic extraction failed for page {page_num}: {e}")
        
        return elements

class PDFOutlineExtractor:
    """Main PDF outline extraction class"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with configuration"""
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = self._get_default_config()
        
        # Initialize components
        self.text_extractor = PDFTextExtractor(self.config)
        self.heading_detector = HeadingDetector(config_path or "configs/model_config.yaml")
        self.title_extractor = TitleExtractor(self.config)
        self.validator = OutputValidator(self.config)
        self.formatter = JSONFormatter()
        
        # Performance monitoring
        perf_config = self.config.get('performance', {})
        self.monitor = PerformanceMonitor(
            max_time=perf_config.get('max_processing_time', 8.0),  # 2s buffer from 10s limit
            max_memory_mb=perf_config.get('max_memory_usage', 6144)  # 2GB buffer from 8GB limit
        )
        
        # Processing flags
        self.debug_mode = self.config.get('debug', {}).get('enabled', False)
        self.save_intermediate = self.config.get('debug', {}).get('save_intermediate', False)
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if config file not found"""
        return {
            'performance': {
                'max_processing_time': 8.0,
                'max_memory_usage': 6144
            },
            'text_extraction': {
                'extract_fonts': True,
                'extract_positions': True,
                'min_text_length': 1,
                'max_pages': 50
            },
            'heading_detection': {
                'min_confidence': 0.5,
                'font_thresholds': {
                    'h1': 1.4,
                    'h2': 1.25,
                    'h3': 1.1
                }
            },
            'title_extraction': {
                'strategies': ['metadata', 'font_analysis', 'position_analysis', 'filename'],
                'validation': {
                    'min_length': 5,
                    'max_length': 200,
                    'max_words': 20
                }
            },
            'validation': {
                'max_title_length': 200,
                'max_heading_length': 200,
                'min_heading_length': 2,
                'max_outline_entries': 500,
                'max_pages': 50
            },
            'debug': {
                'enabled': False,
                'save_intermediate': False
            }
        }
    
    def extract_outline(self, pdf_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Main extraction pipeline
        
        Args:
            pdf_path: Path to input PDF file
            output_path: Optional path to save output JSON
            
        Returns:
            Dictionary with title and outline
        """
        
        # Start performance monitoring
        self.monitor.start()
        
        if self.debug_mode:
            print(f"🚀 Starting PDF outline extraction: {pdf_path}")
        
        try:
            # Validate input
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Check file size (basic validation)
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            if file_size_mb > 100:  # 100MB limit
                print(f"⚠️ Large file detected: {file_size_mb:.1f}MB")
            
            # Step 1: Extract text elements
            if self.debug_mode:
                print("📄 Extracting text elements...")
            
            text_elements, doc_info = self.text_extractor.extract_text_elements(pdf_path)
            
            # Check constraints after text extraction
            constraint_ok, constraint_msg = self.monitor.check_constraints()
            if not constraint_ok:
                raise RuntimeError(f"Constraint violation during text extraction: {constraint_msg}")
            
            if self.debug_mode:
                print(f"📊 Extracted {len(text_elements)} text elements from {doc_info.get('page_count', 0)} pages")
            
            # Step 2: Extract title
            if self.debug_mode:
                print("🏷️ Extracting document title...")
            
            title = self.title_extractor.extract_title(pdf_path, text_elements)
            
            # Step 3: Detect headings
            if self.debug_mode:
                print("🔍 Detecting headings...")
            
            headings = self.heading_detector.detect_headings(text_elements)
            
            # Check constraints after heading detection
            constraint_ok, constraint_msg = self.monitor.check_constraints()
            if not constraint_ok:
                raise RuntimeError(f"Constraint violation during heading detection: {constraint_msg}")
            
            if self.debug_mode:
                print(f"📋 Detected {len(headings)} headings")
            
            # Step 4: Create raw output
            raw_output = {
                'title': title,
                'outline': headings
            }
            
            # Save intermediate results if debug mode
            if self.save_intermediate and output_path:
                intermediate_path = output_path.replace('.json', '_intermediate.json')
                self._save_intermediate_results(intermediate_path, {
                    'raw_output': raw_output,
                    'text_elements_count': len(text_elements),
                    'document_info': doc_info,
                    'performance_stats': self.monitor.get_stats()
                })
            
            # Step 5: Validate and clean output
            if self.debug_mode:
                print("✅ Validating and cleaning output...")
            
            validation_result = self.validator.validate_and_clean(raw_output)
            
            if not validation_result.is_valid:
                print(f"❌ Validation failed: {validation_result.errors}")
                # Create fallback output
                final_output = self.validator.create_fallback_output(
                    title=title,
                    error_message="; ".join(validation_result.errors[:3])
                )
            else:
                final_output = validation_result.cleaned_output
                if validation_result.warnings and self.debug_mode:
                    print(f"⚠️ Validation warnings: {validation_result.warnings}")
            
            # Ensure we have a valid output (should never be None, but safety check)
            if final_output is None:
                final_output = self.validator.create_fallback_output(
                    title=title or "Unknown Document",
                    error_message="Validation returned None"
                )
            
            # Step 6: Save output if path provided
            if output_path:
                self._save_output(final_output, output_path)
            
            # Final performance check
            final_stats = self.monitor.get_stats()
            if self.debug_mode:
                print(f"⚡ Processing completed in {final_stats['elapsed_time']:.3f}s")
                print(f"💾 Memory usage: {final_stats['current_memory_mb']:.1f}MB")
            
            # Add metadata to output
            final_output['_metadata'] = {
                'processing_time': final_stats['elapsed_time'],
                'memory_usage_mb': final_stats['current_memory_mb'],
                'source_pages': doc_info.get('page_count', 0),
                'extraction_method': 'hybrid_cascading_pipeline',
                'version': '1.0'
            }
            
            return final_output
        
        except Exception as e:
            error_msg = f"PDF processing failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            if self.debug_mode:
                print("📊 Full traceback:")
                traceback.print_exc()
            
            # Create error fallback
            fallback_output = self.validator.create_fallback_output(
                title=f"Error Processing {Path(pdf_path).stem}",
                error_message=str(e)
            )
            
            if output_path:
                self._save_output(fallback_output, output_path)
            
            return fallback_output
    
    def extract_batch(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Process multiple PDFs in batch
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save JSON outputs
            
        Returns:
            Batch processing summary
        """
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all PDF files
        pdf_files = list(input_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️ No PDF files found in {input_dir}")
            return {'processed': 0, 'failed': 0, 'files': []}
        
        print(f"📁 Processing {len(pdf_files)} PDF files...")
        
        results = {
            'processed': 0,
            'failed': 0,
            'files': [],
            'total_time': 0,
            'start_time': time.time()
        }
        
        for pdf_file in pdf_files:
            file_start_time = time.time()
            
            try:
                # Determine output filename
                output_file = output_path / f"{pdf_file.stem}.json"
                
                print(f"📄 Processing: {pdf_file.name}")
                
                # Process single file
                result = self.extract_outline(str(pdf_file), str(output_file))
                
                file_time = time.time() - file_start_time
                
                # Record result
                file_result = {
                    'filename': pdf_file.name,
                    'success': True,
                    'processing_time': file_time,
                    'output_file': output_file.name,
                    'headings_count': len(result.get('outline', [])),
                    'title': result.get('title', 'Unknown')
                }
                
                results['files'].append(file_result)
                results['processed'] += 1
                
                print(f"✅ Completed: {pdf_file.name} ({file_time:.2f}s)")
                
            except Exception as e:
                print(f"❌ Failed: {pdf_file.name} - {str(e)}")
                
                file_result = {
                    'filename': pdf_file.name,
                    'success': False,
                    'error': str(e),
                    'processing_time': time.time() - file_start_time
                }
                
                results['files'].append(file_result)
                results['failed'] += 1
        
        results['total_time'] = time.time() - results['start_time']
        
        # Save batch summary
        summary_file = output_path / "batch_summary.json"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save batch summary: {e}")
        
        print(f"🎯 Batch processing complete: {results['processed']} succeeded, {results['failed']} failed")
        
        return results
    
    def _save_output(self, output: Dict, output_path: str) -> None:
        """Save output to JSON file"""
        try:
            # Remove metadata for submission (if present)
            submission_output = {k: v for k, v in output.items() if not k.startswith('_')}
            
            formatted_json = self.formatter.format_for_submission(submission_output)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            
            if self.debug_mode:
                print(f"💾 Output saved to: {output_path}")
                
        except Exception as e:
            print(f"❌ Failed to save output: {e}")
    
    def _save_intermediate_results(self, path: str, data: Dict) -> None:
        """Save intermediate results for debugging"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"⚠️ Failed to save intermediate results: {e}")

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Adobe Hackathon Round 1A - PDF Outline Extractor')
    parser.add_argument('input', help='Input PDF file or directory')
    parser.add_argument('-o', '--output', help='Output JSON file or directory')
    parser.add_argument('-c', '--config', help='Configuration file path')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-b', '--batch', action='store_true', help='Batch process directory')
    
    args = parser.parse_args()
    
    # Initialize extractor
    config_path = args.config or "configs/model_config.yaml"
    extractor = PDFOutlineExtractor(config_path)
    
    # Enable debug mode if requested
    if args.debug:
        extractor.debug_mode = True
        extractor.save_intermediate = True
    
    try:
        if args.batch:
            # Batch processing
            input_dir = args.input
            output_dir = args.output or f"{input_dir}_output"
            
            extractor.extract_batch(input_dir, output_dir)
        
        else:
            # Single file processing
            input_file = args.input
            
            if args.output:
                output_file = args.output
            else:
                # Generate output filename
                input_path = Path(input_file)
                output_file = f"{input_path.stem}.json"
            
            result = extractor.extract_outline(input_file, output_file)
            
            # Print summary
            print(f"\n📊 Processing Summary:")
            print(f"   Title: {result.get('title', 'Unknown')}")
            print(f"   Headings: {len(result.get('outline', []))}")
            if '_metadata' in result:
                metadata = result['_metadata']
                print(f"   Time: {metadata.get('processing_time', 0):.3f}s")
                print(f"   Memory: {metadata.get('memory_usage_mb', 0):.1f}MB")
    
    except KeyboardInterrupt:
        print("\n⛔ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()