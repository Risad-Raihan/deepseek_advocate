"""
Legal Vector Store
Multi-level FAISS indexing system with hybrid search for Bengali legal documents
"""

import os
import numpy as np
import faiss
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import json
from pathlib import Path
from collections import defaultdict
import sqlite3

class LegalVectorStore:
    """Advanced multi-level vector store for legal documents"""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 vector_db_path: str = "legal_advocate/vector_db"):
        
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = 384  # Dimension for multilingual MiniLM
        
        # Multi-level indexes
        self.indexes = {
            'document': None,
            'section': None, 
            'paragraph': None,
            'entity': None
        }
        
        # BM25 indexes for hybrid search
        self.bm25_indexes = {}
        
        # Metadata storage
        self.metadata_db = {}
        self.document_mapping = {}
        
        self.setup_logging()
        self._initialize_storage()
    
    def setup_logging(self):
        """Setup logging for vector store"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _initialize_storage(self):
        """Initialize SQLite database for metadata storage"""
        try:
            self.db_path = self.vector_db_path / "metadata.db"
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            
            # Create tables for different levels
            self._create_tables()
            
        except Exception as e:
            self.logger.error(f"Error initializing storage: {e}")
    
    def _create_tables(self):
        """Create database tables for metadata"""
        cursor = self.conn.cursor()
        
        # Document level table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                filename TEXT,
                doc_type TEXT,
                content_hash TEXT,
                embedding_id INTEGER,
                metadata TEXT
            )
        """)
        
        # Section level table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY,
                document_id INTEGER,
                section_type TEXT,
                section_number TEXT,
                content TEXT,
                embedding_id INTEGER,
                metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        """)
        
        # Paragraph level table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paragraphs (
                id INTEGER PRIMARY KEY,
                section_id INTEGER,
                paragraph_text TEXT,
                embedding_id INTEGER,
                legal_entities TEXT,
                FOREIGN KEY (section_id) REFERENCES sections (id)
            )
        """)
        
        # Entity level table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                entity_type TEXT,
                entity_value TEXT,
                context TEXT,
                document_id INTEGER,
                embedding_id INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        """)
        
        self.conn.commit()
    
    def create_multi_level_index(self, processed_documents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create multi-level FAISS indexes from processed documents
        
        Args:
            processed_documents: Dictionary of processed legal documents
            
        Returns:
            Indexing statistics and metadata
        """
        stats = {
            'total_documents': 0,
            'total_sections': 0,
            'total_paragraphs': 0,
            'total_entities': 0,
            'index_sizes': {}
        }
        
        try:
            # Initialize FAISS indexes
            for level in self.indexes.keys():
                self.indexes[level] = faiss.IndexFlatIP(self.embedding_dim)
            
            # Collections for BM25
            bm25_collections = {
                'document': [],
                'section': [],
                'paragraph': [],
                'entity': []
            }
            
            # Process each document
            for doc_id, doc_data in processed_documents.get('documents', {}).items():
                self._index_document(doc_data, stats, bm25_collections)
            
            # Create BM25 indexes
            self._create_bm25_indexes(bm25_collections)
            
            # Save indexes
            self._save_indexes()
            
            # Update stats
            for level, index in self.indexes.items():
                stats['index_sizes'][level] = index.ntotal
            
            self.logger.info(f"Created multi-level indexes: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error creating multi-level index: {e}")
            return stats
    
    def _index_document(self, doc_data: Dict, stats: Dict, bm25_collections: Dict):
        """Index a single document at all levels"""
        try:
            # Document level indexing
            doc_embedding = self.embedding_model.encode([doc_data.get('filename', '')])
            doc_embedding = doc_embedding.astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(doc_embedding)
            
            # Add to document index
            self.indexes['document'].add(doc_embedding)
            doc_embedding_id = self.indexes['document'].ntotal - 1
            
            # Store in database
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO documents (filename, doc_type, content_hash, embedding_id, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                doc_data.get('filename'),
                doc_data.get('doc_type'),
                doc_data.get('file_hash'),
                doc_embedding_id,
                json.dumps(doc_data)
            ))
            
            document_id = cursor.lastrowid
            stats['total_documents'] += 1
            
            # Add to BM25 collection
            bm25_collections['document'].append(doc_data.get('filename', '').split())
            
            # Section level indexing
            structured_content = doc_data.get('structured_content', {})
            for section_type, sections in structured_content.items():
                for section in sections:
                    self._index_section(section, document_id, section_type, stats, bm25_collections)
            
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error indexing document: {e}")
    
    def _index_section(self, section: Dict, document_id: int, section_type: str, 
                      stats: Dict, bm25_collections: Dict):
        """Index a section and its components"""
        try:
            section_text = section.get('content', '')
            if not section_text:
                return
            
            # Section level embedding
            section_embedding = self.embedding_model.encode([section_text])
            section_embedding = section_embedding.astype('float32')
            faiss.normalize_L2(section_embedding)
            
            self.indexes['section'].add(section_embedding)
            section_embedding_id = self.indexes['section'].ntotal - 1
            
            # Store section
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO sections (document_id, section_type, section_number, content, embedding_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                document_id,
                section_type,
                section.get('section_number', ''),
                section_text,
                section_embedding_id,
                json.dumps(section)
            ))
            
            section_id = cursor.lastrowid
            stats['total_sections'] += 1
            
            # Add to BM25
            bm25_collections['section'].append(section_text.split())
            
            # Paragraph level indexing
            paragraphs = section.get('paragraphs', [])
            for paragraph in paragraphs:
                self._index_paragraph(paragraph, section_id, stats, bm25_collections)
            
            # Entity level indexing
            entities = section.get('entities', {})
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    self._index_entity(entity, entity_type, document_id, stats, bm25_collections)
                    
        except Exception as e:
            self.logger.error(f"Error indexing section: {e}")
    
    def _index_paragraph(self, paragraph: str, section_id: int, stats: Dict, bm25_collections: Dict):
        """Index individual paragraphs"""
        try:
            if not paragraph.strip():
                return
            
            # Paragraph embedding
            para_embedding = self.embedding_model.encode([paragraph])
            para_embedding = para_embedding.astype('float32')
            faiss.normalize_L2(para_embedding)
            
            self.indexes['paragraph'].add(para_embedding)
            para_embedding_id = self.indexes['paragraph'].ntotal - 1
            
            # Store paragraph
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO paragraphs (section_id, paragraph_text, embedding_id, legal_entities)
                VALUES (?, ?, ?, ?)
            """, (
                section_id,
                paragraph,
                para_embedding_id,
                json.dumps({})  # Will be populated later
            ))
            
            stats['total_paragraphs'] += 1
            bm25_collections['paragraph'].append(paragraph.split())
            
        except Exception as e:
            self.logger.error(f"Error indexing paragraph: {e}")
    
    def _index_entity(self, entity: Any, entity_type: str, document_id: int, 
                     stats: Dict, bm25_collections: Dict):
        """Index legal entities"""
        try:
            if isinstance(entity, dict):
                entity_text = entity.get('term', '') + ' ' + entity.get('context', '')
                entity_value = entity.get('term', '')
                context = entity.get('context', '')
            else:
                entity_text = str(entity)
                entity_value = str(entity)
                context = ''
            
            if not entity_text.strip():
                return
            
            # Entity embedding
            entity_embedding = self.embedding_model.encode([entity_text])
            entity_embedding = entity_embedding.astype('float32')
            faiss.normalize_L2(entity_embedding)
            
            self.indexes['entity'].add(entity_embedding)
            entity_embedding_id = self.indexes['entity'].ntotal - 1
            
            # Store entity
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO entities (entity_type, entity_value, context, document_id, embedding_id)
                VALUES (?, ?, ?, ?, ?)
            """, (
                entity_type,
                entity_value,
                context,
                document_id,
                entity_embedding_id
            ))
            
            stats['total_entities'] += 1
            bm25_collections['entity'].append(entity_text.split())
            
        except Exception as e:
            self.logger.error(f"Error indexing entity: {e}")
    
    def _create_bm25_indexes(self, bm25_collections: Dict):
        """Create BM25 indexes for hybrid search"""
        try:
            for level, tokenized_docs in bm25_collections.items():
                if tokenized_docs:
                    self.bm25_indexes[level] = BM25Okapi(tokenized_docs)
                    self.logger.info(f"Created BM25 index for {level} level with {len(tokenized_docs)} documents")
        except Exception as e:
            self.logger.error(f"Error creating BM25 indexes: {e}")
    
    def _save_indexes(self):
        """Save FAISS indexes to disk"""
        try:
            for level, index in self.indexes.items():
                if index is not None and index.ntotal > 0:
                    index_path = self.vector_db_path / f"{level}_index.faiss"
                    faiss.write_index(index, str(index_path))
                    self.logger.info(f"Saved {level} index with {index.ntotal} vectors")
            
            # Save BM25 indexes
            bm25_path = self.vector_db_path / "bm25_indexes.pkl"
            with open(bm25_path, 'wb') as f:
                pickle.dump(self.bm25_indexes, f)
                        
        except Exception as e:
            self.logger.error(f"Error saving indexes: {e}")
    
    def load_indexes(self):
        """Load existing indexes from disk"""
        try:
            for level in self.indexes.keys():
                index_path = self.vector_db_path / f"{level}_index.faiss"
                if index_path.exists():
                    self.indexes[level] = faiss.read_index(str(index_path))
                    self.logger.info(f"Loaded {level} index with {self.indexes[level].ntotal} vectors")
            
            # Load BM25 indexes
            bm25_path = self.vector_db_path / "bm25_indexes.pkl"
            if bm25_path.exists():
                with open(bm25_path, 'rb') as f:
                    self.bm25_indexes = pickle.load(f)
                    
        except Exception as e:
            self.logger.error(f"Error loading indexes: {e}")
    
    def hybrid_search(self, query: str, level: str = 'paragraph', 
                     top_k: int = 10, alpha: float = 0.7) -> List[Dict]:
        """
        Perform hybrid search combining dense embeddings and BM25
        
        Args:
            query: Search query
            level: Index level to search ('document', 'section', 'paragraph', 'entity')
            top_k: Number of results to return
            alpha: Weight for dense search (1-alpha for BM25)
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            if level not in self.indexes or self.indexes[level] is None:
                self.logger.warning(f"Index for level {level} not found")
                return []
            
            # Dense search
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            dense_scores, dense_indices = self.indexes[level].search(query_embedding, top_k * 2)
            dense_scores = dense_scores[0]
            dense_indices = dense_indices[0]
            
            # BM25 search
            bm25_scores = []
            if level in self.bm25_indexes:
                tokenized_query = query.split()
                bm25_scores = self.bm25_indexes[level].get_scores(tokenized_query)
            
            # Combine scores
            combined_results = []
            for i, idx in enumerate(dense_indices):
                if idx == -1:  # No more results
                    break
                
                dense_score = dense_scores[i]
                bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0.0
                
                # Normalize scores
                combined_score = alpha * dense_score + (1 - alpha) * (bm25_score / 10.0)
                
                # Get metadata
                metadata = self._get_metadata(level, idx)
                
                combined_results.append({
                    'index': idx,
                    'dense_score': float(dense_score),
                    'bm25_score': float(bm25_score),
                    'combined_score': float(combined_score),
                    'metadata': metadata
                })
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            return combined_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {e}")
            return []
    
    def _get_metadata(self, level: str, index: int) -> Dict:
        """Retrieve metadata for a specific index"""
        try:
            cursor = self.conn.cursor()
            
            if level == 'document':
                cursor.execute("SELECT * FROM documents WHERE embedding_id = ?", (index,))
            elif level == 'section':
                cursor.execute("SELECT * FROM sections WHERE embedding_id = ?", (index,))
            elif level == 'paragraph':
                cursor.execute("SELECT * FROM paragraphs WHERE embedding_id = ?", (index,))
            elif level == 'entity':
                cursor.execute("SELECT * FROM entities WHERE embedding_id = ?", (index,))
            
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error retrieving metadata: {e}")
            return {} 