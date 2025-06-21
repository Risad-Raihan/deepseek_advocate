"""
Bengali Legal Advocate AI System
A high-performance legal AI combining RAG + Fine-tuning for Bengali legal documents
"""

__version__ = "1.0.0"
__author__ = "Legal AI Team"
__description__ = "Advanced Bengali Legal Advocate using Hybrid RAG + Fine-tuning"

from .document_processor import LegalDocumentProcessor
from .bengali_processor import BengaliLegalProcessor
from .vector_store import LegalVectorStore
from .legal_rag import LegalRAGEngine
from .query_processor import BengaliLegalQueryProcessor
from .response_generator import BengaliLegalResponseGenerator

__all__ = [
    "LegalDocumentProcessor",
    "BengaliLegalProcessor", 
    "LegalVectorStore",
    "LegalRAGEngine",
    "BengaliLegalQueryProcessor",
    "BengaliLegalResponseGenerator"
] 