"""
Legal Document Processor
Handles PDF processing, entity extraction, and document structuring for Bengali legal documents
"""

import os
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
import PyPDF2
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
import json
from collections import defaultdict
import hashlib
from datetime import datetime

class LegalDocumentProcessor:
    """Advanced processor for Bengali legal documents"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_docs = {}
        self.legal_hierarchy = {}
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for document processor"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def process_legal_pdfs(self, output_dir: str = "legal_advocate/training_data") -> Dict[str, Any]:
        """
        Process all legal PDFs and extract structured information
        
        Args:
            output_dir: Directory to save processed data
            
        Returns:
            Dictionary containing processed documents metadata
        """
        processed_data = {
            'documents': {},
            'total_processed': 0,
            'extraction_stats': {},
            'errors': []
        }
        
        try:
            pdf_files = list(self.data_dir.glob("*.pdf"))
            self.logger.info(f"Found {len(pdf_files)} PDF files to process")
            
            for pdf_file in pdf_files:
                try:            
                    self.logger.info(f"Processing: {pdf_file.name}")
                    
                    # Extract text using multiple methods for robustness
                    text_content = self._extract_pdf_text(pdf_file)
                    
                    if not text_content.strip():
                        self.logger.warning(f"No text extracted from {pdf_file.name}")
                        continue
                    
                    # Identify document type
                    doc_type = self._identify_document_type(pdf_file.name, text_content)
                    
                    # Extract legal entities and structure
                    legal_entities = self.extract_legal_entities(text_content)
                    structured_content = self.structure_legal_text(text_content, doc_type)
                    
                    # Create document metadata
                    doc_metadata = {
                        'filename': pdf_file.name,
                        'doc_type': doc_type,
                        'text_length': len(text_content),
                        'entities': legal_entities,
                        'structured_content': structured_content,
                        'processing_date': str(datetime.now()),
                        'file_hash': self._calculate_file_hash(pdf_file)
                    }
                    
                    processed_data['documents'][pdf_file.stem] = doc_metadata
                    processed_data['total_processed'] += 1
                    
                    # Save individual processed document
                    self._save_processed_document(doc_metadata, output_dir)
                    
                except Exception as e:
                    error_msg = f"Error processing {pdf_file.name}: {str(e)}"
                    self.logger.error(error_msg)
                    processed_data['errors'].append(error_msg)
            
            # Save processing summary
            self._save_processing_summary(processed_data, output_dir)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error in process_legal_pdfs: {e}")
            return processed_data
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using multiple methods for maximum reliability
        """
        text_content = ""
        
        # Method 1: Try pdfplumber (best for tables and complex layouts)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                if text_parts:
                    text_content = '\n'.join(text_parts)
                    self.logger.debug(f"pdfplumber extracted {len(text_content)} characters")
        except Exception as e:
            self.logger.warning(f"pdfplumber failed for {pdf_path.name}: {e}")
        
        # Method 2: Try PyMuPDF if pdfplumber failed or gave poor results
        if len(text_content) < 100:
            try:
                doc = fitz.open(pdf_path)
                text_parts = []
                for page in doc:
                    page_text = page.get_text()
                    if page_text:
                        text_parts.append(page_text)
                doc.close()
                
                if text_parts:
                    fitz_text = '\n'.join(text_parts)
                    if len(fitz_text) > len(text_content):
                        text_content = fitz_text
                        self.logger.debug(f"PyMuPDF extracted {len(text_content)} characters")
            except Exception as e:
                self.logger.warning(f"PyMuPDF failed for {pdf_path.name}: {e}")
        
        # Method 3: Try PyPDF2 as fallback
        if len(text_content) < 100:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_parts = []
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    
                    if text_parts:
                        pypdf2_text = '\n'.join(text_parts)
                        if len(pypdf2_text) > len(text_content):
                            text_content = pypdf2_text
                            self.logger.debug(f"PyPDF2 extracted {len(text_content)} characters")
            except Exception as e:
                self.logger.warning(f"PyPDF2 failed for {pdf_path.name}: {e}")
        
        return text_content
    
    def _identify_document_type(self, filename: str, content: str) -> str:
        """
        Identify the type of legal document based on filename and content
        """
        filename_lower = filename.lower()
        content_sample = content[:2000].lower()  # First 2000 chars for analysis
        
        # Document type patterns
        type_patterns = {
            'constitution': ['সংবিধান', 'constitution'],
            'family_law': ['পারিবারিক', 'মুসলিম পরিবার', 'তালাক', 'family'],
            'rent_control': ['ভাড়া', 'rent', 'বাড়ী ভাড়া'],
            'court_procedure': ['আদালত', 'প্রক্রিয়া', 'মামলা দায়ের', 'court', 'procedure'],
            'ordinance': ['অধ্যাদেশ', 'ordinance'],
            'legal_notice': ['নোটিশ', 'notice'],
            'property_law': ['সম্পত্তি', 'property'],
            'criminal_law': ['দণ্ড', 'ফৌজদারি', 'criminal'],
            'civil_law': ['দেওয়ানী', 'civil']
        }
        
        # Check filename first
        for doc_type, patterns in type_patterns.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    return doc_type
        
        # Check content
        for doc_type, patterns in type_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in content_sample)
            if matches >= 2:  # Require at least 2 pattern matches
                return doc_type
        
        return 'general_law'
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract legal entities from Bengali text using pattern matching
        
        Args:
            text: Legal document text
            
        Returns:
            Dictionary of extracted entities by category
        """
        entities = defaultdict(list)
        
        try:
            # Section patterns
            section_patterns = [
                r'ধারা\s*(\d+(?:\([ক-৯]+\))?)',
                r'অনুচ্ছেদ\s*(\d+(?:\([ক-৯]+\))?)',
                r'উপধারা\s*\(([ক-৯]+)\)'
            ]
            
            for pattern in section_patterns:
                matches = re.findall(pattern, text, re.UNICODE)
                entities['sections'].extend(matches)
            
            # Law and ordinance patterns
            law_patterns = [
                r'(\d{4})\s*সালের\s*(.+?)\s*আইন',
                r'(\d{4})\s*সালের\s*(.+?)\s*অধ্যাদেশ'
            ]
            
            for pattern in law_patterns:
                matches = re.findall(pattern, text, re.UNICODE)
                entities['laws'].extend([f"{match[1]} ({match[0]})" for match in matches])
            
            # Legal terms
            legal_terms = [
                'আদালত', 'বিচারক', 'মামলা', 'রায়', 'আদেশ', 'নোটিশ',
                'আইনজীবী', 'উকিল', 'বাদী', 'বিবাদী', 'সাক্ষী'
            ]
            
            for term in legal_terms:
                if term in text:
                    entities['legal_terms'].append(term)
            
            # Remove duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            return dict(entities)
            
        except Exception as e:
            self.logger.error(f"Error extracting legal entities: {e}")
            return {}
    
    def structure_legal_text(self, text: str, doc_type: str) -> Dict[str, List[Dict]]:
        """
        Structure legal text into hierarchical format
        
        Args:
            text: Legal document text
            doc_type: Type of legal document
            
        Returns:
            Structured content with sections and paragraphs
        """
        structured_content = defaultdict(list)
        
        try:
            # Split into sections based on document type
            if doc_type in ['constitution', 'ordinance']:
                sections = self._split_constitutional_sections(text)
            elif doc_type in ['family_law', 'property_law']:
                sections = self._split_law_sections(text)
            else:
                sections = self._split_general_sections(text)
            
            for section in sections:
                if section.strip():
                    structured_section = {
                        'section_number': self._extract_section_number(section),
                        'content': section.strip(),
                        'paragraphs': self._split_paragraphs(section),
                        'entities': self.extract_legal_entities(section)
                    }
                    structured_content[doc_type].append(structured_section)
            
            return dict(structured_content)
            
        except Exception as e:
            self.logger.error(f"Error structuring legal text: {e}")
            return {}
    
    def _split_constitutional_sections(self, text: str) -> List[str]:
        """Split constitutional text into articles/sections"""
        # Split by article numbers
        sections = re.split(r'(?=অনুচ্ছেদ\s*\d+)', text)
        return [section.strip() for section in sections if section.strip()]
    
    def _split_law_sections(self, text: str) -> List[str]:
        """Split law text into sections"""
        # Split by section numbers
        sections = re.split(r'(?=ধারা\s*\d+)', text)
        return [section.strip() for section in sections if section.strip()]
    
    def _split_general_sections(self, text: str) -> List[str]:
        """Split general text into meaningful sections"""
        # Split by major headings or numbered sections
        sections = re.split(r'(?=\d+\.\s*|\n\n\s*[A-Z])', text)
        return [section.strip() for section in sections if len(section.strip()) > 100]
    
    def _extract_section_number(self, section: str) -> str:
        """Extract section/article number from text"""
        # Try to find section number
        section_match = re.search(r'ধারা\s*(\d+(?:\([ক-৯]+\))?)', section)
        if section_match:
            return section_match.group(1)
        
        # Try to find article number
        article_match = re.search(r'অনুচ্ছেদ\s*(\d+(?:\([ক-৯]+\))?)', section)
        if article_match:
            return article_match.group(1)
        
        return ""
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into meaningful paragraphs"""
        # Split by double newlines or sentence endings
        paragraphs = re.split(r'\n\n+|।\s*\n', text)
        return [p.strip() for p in paragraphs if len(p.strip()) > 50]
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _save_processed_document(self, doc_metadata: Dict, output_dir: str):
        """Save processed document to JSON file"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            filename = doc_metadata['filename'].replace('.pdf', '.json')
            file_path = output_path / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(doc_metadata, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving processed document: {e}")
    
    def _save_processing_summary(self, processed_data: Dict, output_dir: str):
        """Save processing summary"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            summary_path = output_path / "processing_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving processing summary: {e}")
    
    def normalize_bengali_legal_terms(self, text: str) -> str:
        """
        Normalize Bengali legal terminology for consistency
        
        Args:
            text: Text with Bengali legal terms
            
        Returns:
            Normalized text
        """
        try:
            # Common legal term variations
            normalizations = {
                'আদালত': ['আদালত', 'আদালতে', 'আদালতের', 'কোর্ট'],
                'ধারা': ['ধারা', 'ধারায়', 'ধারার', 'ধারাটি'],
                'আইন': ['আইন', 'আইনি', 'আইনের', 'আইনে'],
                'মামলা': ['মামলা', 'মামলায়', 'মামলার', 'কেস'],
            }
            
            for standard, variations in normalizations.items():
                for variation in variations:
                    if variation != standard:
                        text = text.replace(variation, standard)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error normalizing Bengali legal terms: {e}")
            return text 