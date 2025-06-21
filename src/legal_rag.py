"""
Legal RAG Engine - Core Retrieval Augmented Generation for Bengali Legal Documents
Orchestrates query processing, multi-strategy retrieval, and context building
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import json
from collections import defaultdict
import re

class LegalRAGEngine:
    """Core RAG engine for Bengali legal document processing and retrieval"""
    
    def __init__(self, vector_store, bengali_processor, query_processor=None):
        self.vector_store = vector_store
        self.bengali_processor = bengali_processor
        self.query_processor = query_processor
        self.setup_logging()
        
        # Initialize retrieval strategies
        self._initialize_retrieval_strategies()
        
        # Legal context building parameters
        self.max_context_length = 2048
        self.max_citations = 5
        self.context_overlap_threshold = 0.7
        
    def setup_logging(self):
        """Setup logging for RAG engine"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _initialize_retrieval_strategies(self):
        """Initialize different retrieval strategies"""
        try:
            from .retrieval_strategies import RetrievalStrategyFactory
            
            self.retrieval_strategies = {
                'direct_legal': RetrievalStrategyFactory.create_strategy(
                    'direct_legal_retrieval', self.vector_store, self.bengali_processor
                ),
                'conceptual': RetrievalStrategyFactory.create_strategy(
                    'conceptual_retrieval', self.vector_store, self.bengali_processor
                ),
                'multi_hop': RetrievalStrategyFactory.create_strategy(
                    'multi_hop_retrieval', self.vector_store, self.bengali_processor
                ),
                'precedence': RetrievalStrategyFactory.create_strategy(
                    'precedence_retrieval', self.vector_store, self.bengali_processor
                )
            }
            
            self.logger.info("Retrieval strategies initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing retrieval strategies: {e}")
            self.retrieval_strategies = {}
    
    def process_legal_query(self, query: str) -> Dict[str, Any]:
        """
        Main function to process legal query and generate comprehensive response
        
        Args:
            query: Bengali legal query string
            
        Returns:
            Complete legal analysis with context and citations
        """
        try:
            self.logger.info(f"Processing legal query: {query[:50]}...")
            
            # Step 1: Process and understand query
            if self.query_processor:
                processed_query = self.query_processor.process_legal_query(query)
            else:
                # Fallback to basic processing
                processed_query = self._basic_query_processing(query)
            
            # Step 2: Retrieve relevant legal context
            retrieved_context = self.retrieve_legal_context(processed_query)
            
            # Step 3: Build response context
            response_context = self.build_legal_response_context(
                retrieved_context['documents'],
                processed_query
            )
            
            # Step 4: Format legal citations
            formatted_citations = self.format_legal_citations(
                retrieved_context['documents']
            )
            
            # Step 5: Compile complete response
            complete_response = {
                'query_analysis': processed_query,
                'retrieved_context': retrieved_context,
                'response_context': response_context,
                'citations': formatted_citations,
                'legal_domain': processed_query.get('domain', {}).get('domain', 'general'),
                'confidence_score': self._calculate_overall_confidence(
                    processed_query, retrieved_context
                ),
                'processing_metadata': {
                    'retrieval_strategy': retrieved_context.get('strategy_used'),
                    'documents_found': len(retrieved_context.get('documents', [])),
                    'context_length': len(response_context),
                    'citations_count': len(formatted_citations)
                }
            }
            
            self.logger.info("Legal query processed successfully")
            return complete_response
            
        except Exception as e:
            self.logger.error(f"Error processing legal query: {e}")
            return {
                'error': str(e),
                'query': query,
                'fallback_response': "দুঃখিত, আপনার প্রশ্নটি প্রক্রিয়া করতে সমস্যা হয়েছে। দয়া করে আবার চেষ্টা করুন।"
            }
    
    def _basic_query_processing(self, query: str) -> Dict[str, Any]:
        """Basic query processing fallback when query processor is not available"""
        try:
            # Basic Bengali text preprocessing
            clean_query = self.bengali_processor.preprocess_bengali_legal_text(query)
            
            # Basic entity extraction
            entities = self.bengali_processor.extract_legal_entities(clean_query)
            
            # Basic intent classification
            intent_info = self.bengali_processor.extract_legal_intent(clean_query)
            
            return {
                'original_query': query,
                'clean_query': clean_query,
                'entities': entities,
                'domain': {
                    'domain': intent_info.get('legal_domain', 'general'),
                    'confidence': intent_info.get('confidence', 0.5)
                },
                'complexity': 'medium',
                'suggested_retrieval_strategy': 'conceptual'
            }
            
        except Exception as e:
            self.logger.error(f"Error in basic query processing: {e}")
            return {
                'original_query': query,
                'clean_query': query,
                'domain': {'domain': 'general', 'confidence': 0.0}
            }
    
    def retrieve_legal_context(self, processed_query: Dict) -> Dict[str, Any]:
        """
        Retrieve relevant legal context using appropriate strategy
        
        Args:
            processed_query: Processed query information
            
        Returns:
            Retrieved documents with metadata
        """
        try:
            # Determine retrieval strategy
            suggested_strategy = processed_query.get('suggested_retrieval_strategy', 'conceptual')
            
            # Map strategy names
            strategy_mapping = {
                'direct_legal_retrieval': 'direct_legal',
                'conceptual_retrieval': 'conceptual', 
                'multi_hop_retrieval': 'multi_hop',
                'precedence_retrieval': 'precedence',
                'hybrid_retrieval': 'conceptual'
            }
            
            strategy_key = strategy_mapping.get(suggested_strategy, 'conceptual')
            
            # Execute retrieval
            if strategy_key in self.retrieval_strategies:
                retrieved_docs = self.retrieval_strategies[strategy_key].retrieve(
                    processed_query, top_k=10
                )
            else:
                # Fallback to direct vector store search
                query = processed_query.get('clean_query', '')
                retrieved_docs = self.vector_store.hybrid_search(
                    query=query, level='paragraph', top_k=10
                )
            
            # Multi-hop retrieval for complex queries
            if processed_query.get('complexity') == 'high' and strategy_key != 'multi_hop':
                additional_docs = self._perform_additional_retrieval(processed_query, retrieved_docs)
                retrieved_docs.extend(additional_docs)
            
            # Rank and filter results
            ranked_docs = self._rank_and_filter_results(retrieved_docs, processed_query)
            
            # Build cross-references
            cross_references = self._build_cross_references(ranked_docs)
            
            return {
                'documents': ranked_docs,
                'strategy_used': strategy_key,
                'total_retrieved': len(retrieved_docs),
                'final_count': len(ranked_docs),
                'cross_references': cross_references,
                'retrieval_metadata': {
                    'query_complexity': processed_query.get('complexity', 'medium'),
                    'domain': processed_query.get('domain', {}).get('domain', 'general'),
                    'strategy_confidence': self._assess_strategy_confidence(strategy_key, ranked_docs)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving legal context: {e}")
            return {'documents': [], 'strategy_used': 'fallback', 'error': str(e)}
    
    def _perform_additional_retrieval(self, processed_query: Dict, initial_docs: List[Dict]) -> List[Dict]:
        """Perform additional retrieval for complex queries"""
        try:
            additional_docs = []
            
            # Extract key concepts from initial results
            key_concepts = self._extract_key_concepts(initial_docs)
            
            # Expand search with key concepts
            for concept in key_concepts[:3]:
                concept_query = f"{processed_query.get('clean_query', '')} {concept}"
                concept_docs = self.vector_store.hybrid_search(
                    query=concept_query, level='paragraph', top_k=3
                )
                additional_docs.extend(concept_docs)
            
            return additional_docs
            
        except Exception as e:
            self.logger.error(f"Error in additional retrieval: {e}")
            return []
    
    def _extract_key_concepts(self, documents: List[Dict]) -> List[str]:
        """Extract key legal concepts from retrieved documents"""
        try:
            concepts = []
            
            for doc in documents[:5]:  # Analyze top 5 documents
                metadata = doc.get('metadata', {})
                
                # Extract from paragraph text
                if 'paragraph_text' in metadata:
                    text = metadata['paragraph_text']
                    # Extract legal terms
                    legal_patterns = [
                        r'ধারা\s*\d+',
                        r'অনুচ্ছেদ\s*\d+',
                        r'\d{4}\s*সালের\s*\w+\s*আইন',
                        r'আদালত',
                        r'বিচার',
                        r'মামলা'
                    ]
                    
                    for pattern in legal_patterns:
                        matches = re.findall(pattern, text)
                        concepts.extend(matches)
            
            # Return unique concepts
            return list(set(concepts))[:5]
            
        except Exception as e:
            self.logger.error(f"Error extracting key concepts: {e}")
            return []
    
    def _rank_and_filter_results(self, documents: List[Dict], processed_query: Dict) -> List[Dict]:
        """Rank and filter retrieved documents based on legal relevance"""
        try:
            if not documents:
                return []
            
            # Calculate enhanced relevance scores
            for doc in documents:
                base_score = doc.get('combined_score', 0)
                
                # Domain relevance boost
                domain = processed_query.get('domain', {}).get('domain', 'general')
                domain_boost = self._get_domain_boost(doc, domain)
                
                # Entity matching boost
                entity_boost = self._get_entity_boost(doc, processed_query.get('entities', {}))
                
                # Legal specificity boost
                specificity_boost = self._get_specificity_boost(doc)
                
                # Calculate final relevance score
                doc['legal_relevance'] = base_score * domain_boost * entity_boost * specificity_boost
            
            # Sort by legal relevance
            documents.sort(key=lambda x: x.get('legal_relevance', 0), reverse=True)
            
            # Filter for quality and diversity
            filtered_docs = self._filter_for_quality_and_diversity(documents)
            
            return filtered_docs[:8]  # Return top 8 most relevant
            
        except Exception as e:
            self.logger.error(f"Error ranking and filtering results: {e}")
            return documents[:5]  # Return first 5 as fallback
    
    def _get_domain_boost(self, document: Dict, query_domain: str) -> float:
        """Calculate domain relevance boost"""
        try:
            metadata = document.get('metadata', {})
            
            # Check document type
            if 'document_id' in metadata:
                # This would require mapping document IDs to types
                # For now, return neutral boost
                return 1.0
            
            # Check content for domain indicators
            content = str(metadata.get('paragraph_text', '') or metadata.get('content', ''))
            
            domain_keywords = {
                'family_law': ['তালাক', 'বিবাহ', 'খোরপোশ', 'পারিবারিক'],
                'property_law': ['সম্পত্তি', 'জমি', 'দলিল', 'মালিকানা'],
                'constitutional_law': ['সংবিধান', 'মৌলিক অধিকার', 'নাগরিক'],
                'rent_control': ['ভাড়া', 'ইজারা', 'বাড়িওয়ালা'],
                'court_procedure': ['আদালত', 'মামলা', 'বিচার', 'প্রক্রিয়া']
            }
            
            if query_domain in domain_keywords:
                keyword_matches = sum(1 for kw in domain_keywords[query_domain] if kw in content)
                return 1.0 + (keyword_matches * 0.1)
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating domain boost: {e}")
            return 1.0
    
    def _get_entity_boost(self, document: Dict, query_entities: Dict) -> float:
        """Calculate entity matching boost"""
        try:
            if not query_entities:
                return 1.0
            
            metadata = document.get('metadata', {})
            content = str(metadata.get('paragraph_text', '') or metadata.get('content', ''))
            
            entity_matches = 0
            total_entities = 0
            
            for entity_type, entity_list in query_entities.items():
                total_entities += len(entity_list)
                
                for entity in entity_list:
                    if str(entity).lower() in content.lower():
                        entity_matches += 1
            
            if total_entities > 0:
                match_ratio = entity_matches / total_entities
                return 1.0 + (match_ratio * 0.3)
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating entity boost: {e}")
            return 1.0
    
    def _get_specificity_boost(self, document: Dict) -> float:
        """Calculate legal specificity boost"""
        try:
            metadata = document.get('metadata', {})
            content = str(metadata.get('paragraph_text', '') or metadata.get('content', ''))
            
            # Look for specific legal references
            specific_patterns = [
                r'ধারা\s*\d+',
                r'অনুচ্ছেদ\s*\d+',
                r'\d{4}\s*সালের\s*\w+\s*আইন',
                r'অধ্যাদেশ',
                r'বিধি',
                r'নিয়ম'
            ]
            
            specificity_score = 0
            for pattern in specific_patterns:
                matches = len(re.findall(pattern, content))
                specificity_score += matches
            
            return 1.0 + min(specificity_score * 0.05, 0.3)
            
        except Exception as e:
            self.logger.error(f"Error calculating specificity boost: {e}")
            return 1.0
    
    def _filter_for_quality_and_diversity(self, documents: List[Dict]) -> List[Dict]:
        """Filter documents for quality and diversity"""
        try:
            filtered = []
            seen_content_hashes = set()
            
            for doc in documents:
                metadata = doc.get('metadata', {})
                content = str(metadata.get('paragraph_text', '') or metadata.get('content', ''))
                
                # Skip if content is too short
                if len(content) < 50:
                    continue
                
                # Check for content diversity
                content_hash = hash(content[:200])
                if content_hash in seen_content_hashes:
                    continue
                
                seen_content_hashes.add(content_hash)
                filtered.append(doc)
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Error filtering for quality and diversity: {e}")
            return documents
    
    def _build_cross_references(self, documents: List[Dict]) -> List[Dict]:
        """Build cross-references between legal documents"""
        try:
            cross_refs = []
            
            # Extract legal references from documents
            for doc in documents:
                metadata = doc.get('metadata', {})
                content = str(metadata.get('paragraph_text', '') or metadata.get('content', ''))
                
                # Find section/article references
                section_refs = re.findall(r'ধারা\s*(\d+)', content)
                article_refs = re.findall(r'অনুচ্ছেদ\s*(\d+)', content)
                law_refs = re.findall(r'(\d{4})\s*সালের\s*(.+?)\s*আইন', content)
                
                if section_refs or article_refs or law_refs:
                    cross_refs.append({
                        'document_index': documents.index(doc),
                        'section_references': section_refs,
                        'article_references': article_refs,
                        'law_references': law_refs
                    })
            
            return cross_refs
            
        except Exception as e:
            self.logger.error(f"Error building cross-references: {e}")
            return []
    
    def _assess_strategy_confidence(self, strategy: str, documents: List[Dict]) -> float:
        """Assess confidence in the chosen retrieval strategy"""
        try:
            if not documents:
                return 0.0
            
            # Strategy-specific confidence assessment
            avg_score = sum(doc.get('combined_score', 0) for doc in documents) / len(documents)
            
            strategy_multipliers = {
                'direct_legal': 1.2,
                'multi_hop': 1.1,
                'conceptual': 1.0,
                'precedence': 1.05
            }
            
            multiplier = strategy_multipliers.get(strategy, 1.0)
            return min(avg_score * multiplier, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error assessing strategy confidence: {e}")
            return 0.5
    
    def _calculate_overall_confidence(self, processed_query: Dict, retrieved_context: Dict) -> float:
        """Calculate overall confidence in the response"""
        try:
            # Query understanding confidence
            query_confidence = processed_query.get('domain', {}).get('confidence', 0.0)
            
            # Retrieval confidence
            retrieval_confidence = retrieved_context.get('retrieval_metadata', {}).get('strategy_confidence', 0.0)
            
            # Document count factor
            doc_count = len(retrieved_context.get('documents', []))
            doc_factor = min(doc_count / 5.0, 1.0)  # Normalize to 5 documents
            
            # Combined confidence
            overall_confidence = (query_confidence * 0.4 + retrieval_confidence * 0.4 + doc_factor * 0.2)
            
            return min(overall_confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall confidence: {e}")
            return 0.5 