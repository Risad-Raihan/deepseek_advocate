"""
Advanced Retrieval Strategies for Bengali Legal Documents
Multiple retrieval strategies optimized for different types of legal queries
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

class BaseRetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies"""
    
    def __init__(self, vector_store, bengali_processor):
        self.vector_store = vector_store
        self.bengali_processor = bengali_processor
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def retrieve(self, processed_query: Dict, top_k: int = 10) -> List[Dict]:
        """Execute retrieval strategy"""
        pass
    
    def _rank_results_by_legal_relevance(self, results: List[Dict], query_domain: str) -> List[Dict]:
        """Apply legal-specific ranking to results"""
        try:
            # Domain-specific boosting
            domain_boost = {
                'constitutional_law': 1.2,
                'family_law': 1.1,
                'property_law': 1.0,
                'rent_control': 1.05,
                'court_procedure': 0.9
            }
            
            for result in results:
                # Apply domain boost
                if query_domain in domain_boost:
                    result['legal_relevance_score'] = result.get('combined_score', 0) * domain_boost[query_domain]
                else:
                    result['legal_relevance_score'] = result.get('combined_score', 0)
                
                # Boost for exact legal entity matches
                metadata = result.get('metadata', {})
                if 'legal_entities' in metadata:
                    result['legal_relevance_score'] *= 1.15
            
            # Sort by legal relevance
            results.sort(key=lambda x: x.get('legal_relevance_score', 0), reverse=True)
            return results
            
        except Exception as e:
            self.logger.error(f"Error ranking results by legal relevance: {e}")
            return results

class DirectLegalRetrieval(BaseRetrievalStrategy):
    """Direct retrieval for specific law/section queries"""
    
    def retrieve(self, processed_query: Dict, top_k: int = 10) -> List[Dict]:
        """
        Direct retrieval for queries asking about specific laws, sections, or articles
        """
        try:
            query = processed_query.get('clean_query', '')
            entities = processed_query.get('entities', {})
            domain = processed_query.get('domain', {}).get('domain', 'general')
            
            # Check if this is a direct legal reference query
            has_section = 'section' in entities or 'article' in entities
            has_law = 'law_with_year' in entities
            
            if has_section or has_law:
                # Use section-level search for direct references
                results = self.vector_store.hybrid_search(
                    query=query,
                    level='section',
                    top_k=top_k,
                    alpha=0.8  # Favor dense search for direct matches
                )
            else:
                # Use paragraph-level search for general legal terms
                results = self.vector_store.hybrid_search(
                    query=query,
                    level='paragraph',
                    top_k=top_k,
                    alpha=0.7
                )
            
            # Apply legal relevance ranking
            ranked_results = self._rank_results_by_legal_relevance(results, domain)
            
            # Add retrieval metadata
            for result in ranked_results:
                result['retrieval_strategy'] = 'direct_legal'
                result['confidence'] = 'high' if has_section or has_law else 'medium'
            
            return ranked_results
            
        except Exception as e:
            self.logger.error(f"Error in direct legal retrieval: {e}")
            return []

class ConceptualRetrieval(BaseRetrievalStrategy):
    """Conceptual retrieval for broad legal concept questions"""
    
    def retrieve(self, processed_query: Dict, top_k: int = 10) -> List[Dict]:
        """
        Conceptual retrieval for queries about broad legal concepts
        """
        try:
            expanded_query = processed_query.get('expanded_query', '')
            original_query = processed_query.get('clean_query', '')
            domain = processed_query.get('domain', {}).get('domain', 'general')
            
            # Multi-level retrieval for conceptual understanding
            results = []
            
            # Document-level search for domain classification
            doc_results = self.vector_store.hybrid_search(
                query=expanded_query,
                level='document',
                top_k=3,
                alpha=0.6
            )
            
            # Paragraph-level search for detailed content
            para_results = self.vector_store.hybrid_search(
                query=expanded_query,
                level='paragraph',
                top_k=top_k,
                alpha=0.7
            )
            
            # Entity-level search for related terms
            entity_results = self.vector_store.hybrid_search(
                query=original_query,
                level='entity',
                top_k=5,
                alpha=0.5
            )
            
            # Combine results with different weights
            all_results = []
            
            # Add document results with high relevance
            for result in doc_results:
                result['source_level'] = 'document'
                result['conceptual_weight'] = 1.3
                all_results.append(result)
            
            # Add paragraph results
            for result in para_results:
                result['source_level'] = 'paragraph'
                result['conceptual_weight'] = 1.0
                all_results.append(result)
            
            # Add entity results with lower weight
            for result in entity_results:
                result['source_level'] = 'entity'
                result['conceptual_weight'] = 0.8
                all_results.append(result)
            
            # Apply conceptual ranking
            for result in all_results:
                base_score = result.get('combined_score', 0)
                conceptual_weight = result.get('conceptual_weight', 1.0)
                result['conceptual_score'] = base_score * conceptual_weight
            
            # Sort and deduplicate
            all_results.sort(key=lambda x: x.get('conceptual_score', 0), reverse=True)
            unique_results = self._deduplicate_results(all_results, top_k)
            
            # Apply legal relevance ranking
            ranked_results = self._rank_results_by_legal_relevance(unique_results, domain)
            
            # Add retrieval metadata
            for result in ranked_results:
                result['retrieval_strategy'] = 'conceptual'
                result['confidence'] = 'medium'
            
            return ranked_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error in conceptual retrieval: {e}")
            return []
    
    def _deduplicate_results(self, results: List[Dict], max_results: int) -> List[Dict]:
        """Remove duplicate results based on content similarity"""
        try:
            unique_results = []
            seen_content = set()
            
            for result in results:
                metadata = result.get('metadata', {})
                
                # Create content identifier
                if 'paragraph_text' in metadata:
                    content_id = hash(metadata['paragraph_text'][:100])
                elif 'content' in metadata:
                    content_id = hash(str(metadata['content'])[:100])
                else:
                    content_id = hash(str(metadata))
                
                if content_id not in seen_content:
                    seen_content.add(content_id)
                    unique_results.append(result)
                    
                    if len(unique_results) >= max_results:
                        break
            
            return unique_results
            
        except Exception as e:
            self.logger.error(f"Error deduplicating results: {e}")
            return results[:max_results]

class MultiHopRetrieval(BaseRetrievalStrategy):
    """Multi-hop retrieval for complex legal reasoning chains"""
    
    def retrieve(self, processed_query: Dict, top_k: int = 10) -> List[Dict]:
        """
        Multi-hop retrieval for complex queries requiring legal reasoning chains
        """
        try:
            query = processed_query.get('clean_query', '')
            domain = processed_query.get('domain', {}).get('domain', 'general')
            entities = processed_query.get('entities', {})
            
            # First hop: Initial retrieval
            initial_results = self.vector_store.hybrid_search(
                query=query,
                level='paragraph',
                top_k=8,
                alpha=0.7
            )
            
            # Extract related legal concepts from initial results
            related_concepts = self._extract_related_concepts(initial_results)
            
            # Second hop: Expand search with related concepts
            expanded_queries = self._build_expansion_queries(query, related_concepts)
            
            second_hop_results = []
            for exp_query in expanded_queries:
                hop_results = self.vector_store.hybrid_search(
                    query=exp_query,
                    level='paragraph',
                    top_k=5,
                    alpha=0.6
                )
                second_hop_results.extend(hop_results)
            
            # Third hop: Cross-reference search
            cross_ref_results = self._search_cross_references(initial_results, entities)
            
            # Combine all hops
            all_results = []
            
            # Weight results by hop
            for result in initial_results:
                result['hop_level'] = 1
                result['hop_weight'] = 1.0
                all_results.append(result)
            
            for result in second_hop_results:
                result['hop_level'] = 2
                result['hop_weight'] = 0.8
                all_results.append(result)
            
            for result in cross_ref_results:
                result['hop_level'] = 3
                result['hop_weight'] = 0.9
                all_results.append(result)
            
            # Calculate multi-hop scores
            for result in all_results:
                base_score = result.get('combined_score', 0)
                hop_weight = result.get('hop_weight', 1.0)
                result['multi_hop_score'] = base_score * hop_weight
            
            # Sort and deduplicate
            all_results.sort(key=lambda x: x.get('multi_hop_score', 0), reverse=True)
            unique_results = self._deduplicate_results(all_results, top_k * 2)
            
            # Apply legal relevance ranking
            ranked_results = self._rank_results_by_legal_relevance(unique_results, domain)
            
            # Add retrieval metadata
            for result in ranked_results:
                result['retrieval_strategy'] = 'multi_hop'
                result['confidence'] = 'high'
            
            return ranked_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error in multi-hop retrieval: {e}")
            return []
    
    def _extract_related_concepts(self, results: List[Dict]) -> List[str]:
        """Extract related legal concepts from retrieval results"""
        try:
            concepts = []
            
            for result in results:
                metadata = result.get('metadata', {})
                
                # Extract from paragraph text
                if 'paragraph_text' in metadata:
                    text = metadata['paragraph_text']
                    # Extract legal terms using regex
                    legal_terms = re.findall(r'(ধারা\s*\d+|অনুচ্ছেদ\s*\d+|আইন|অধ্যাদেশ)', text)
                    concepts.extend(legal_terms)
                
                # Extract from entities
                if 'legal_entities' in metadata:
                    entities_data = metadata['legal_entities']
                    if isinstance(entities_data, str):
                        import json
                        try:
                            entities_data = json.loads(entities_data)
                        except:
                            pass
                    
                    if isinstance(entities_data, dict):
                        for entity_list in entities_data.values():
                            if isinstance(entity_list, list):
                                concepts.extend(entity_list)
            
            # Remove duplicates and return top concepts
            unique_concepts = list(set(concepts))
            return unique_concepts[:5]
            
        except Exception as e:
            self.logger.error(f"Error extracting related concepts: {e}")
            return []
    
    def _build_expansion_queries(self, original_query: str, concepts: List[str]) -> List[str]:
        """Build expansion queries using related concepts"""
        try:
            expansion_queries = []
            
            for concept in concepts:
                # Combine original query with related concept
                expanded = f"{original_query} {concept}"
                expansion_queries.append(expanded)
            
            return expansion_queries[:3]  # Limit to top 3 expansions
            
        except Exception as e:
            self.logger.error(f"Error building expansion queries: {e}")
            return [original_query]
    
    def _search_cross_references(self, initial_results: List[Dict], entities: Dict) -> List[Dict]:
        """Search for cross-references mentioned in initial results"""
        try:
            cross_ref_results = []
            
            # Extract specific legal references
            references = []
            
            # Add entity-based references
            for entity_type, entity_list in entities.items():
                if entity_type in ['section', 'article', 'law_with_year']:
                    references.extend(entity_list)
            
            # Search for each reference
            for ref in references[:3]:  # Limit cross-reference searches
                ref_query = str(ref)
                ref_results = self.vector_store.hybrid_search(
                    query=ref_query,
                    level='section',
                    top_k=3,
                    alpha=0.8
                )
                cross_ref_results.extend(ref_results)
            
            return cross_ref_results
            
        except Exception as e:
            self.logger.error(f"Error searching cross-references: {e}")
            return []
    
    def _deduplicate_results(self, results: List[Dict], max_results: int) -> List[Dict]:
        """Remove duplicate results based on content similarity"""
        try:
            unique_results = []
            seen_content = set()
            
            for result in results:
                metadata = result.get('metadata', {})
                
                # Create content identifier
                if 'paragraph_text' in metadata:
                    content_id = hash(metadata['paragraph_text'][:100])
                elif 'content' in metadata:
                    content_id = hash(str(metadata['content'])[:100])
                else:
                    content_id = hash(str(metadata))
                
                if content_id not in seen_content:
                    seen_content.add(content_id)
                    unique_results.append(result)
                    
                    if len(unique_results) >= max_results:
                        break
            
            return unique_results
            
        except Exception as e:
            self.logger.error(f"Error deduplicating results: {e}")
            return results[:max_results]

class PrecedenceRetrieval(BaseRetrievalStrategy):
    """Retrieval for legal precedence and procedural queries"""
    
    def retrieve(self, processed_query: Dict, top_k: int = 10) -> List[Dict]:
        """
        Retrieval for queries requiring legal precedence understanding
        """
        try:
            query = processed_query.get('clean_query', '')
            precedence_info = processed_query.get('precedence_requirements', {})
            domain = processed_query.get('domain', {}).get('domain', 'general')
            
            retrieval_depth = precedence_info.get('suggested_retrieval_depth', 2)
            
            # Procedural search strategy
            results = []
            
            # Search for procedural sections
            procedural_results = self.vector_store.hybrid_search(
                query=f"{query} প্রক্রিয়া পদ্ধতি",
                level='section',
                top_k=top_k // 2,
                alpha=0.75
            )
            
            # Search for requirement paragraphs
            requirement_results = self.vector_store.hybrid_search(
                query=f"{query} শর্ত প্রয়োজন",
                level='paragraph',
                top_k=top_k // 2,
                alpha=0.7
            )
            
            # Combine results
            results.extend(procedural_results)
            results.extend(requirement_results)
            
            # Apply precedence-specific ranking
            for result in results:
                base_score = result.get('combined_score', 0)
                
                # Boost for procedural content
                metadata = result.get('metadata', {})
                content = str(metadata.get('paragraph_text', '') or metadata.get('content', ''))
                
                procedural_boost = 1.0
                if any(word in content for word in ['প্রক্রিয়া', 'পদ্ধতি', 'শর্ত', 'প্রয়োজন']):
                    procedural_boost = 1.2
                
                result['precedence_score'] = base_score * procedural_boost
            
            # Sort by precedence score
            results.sort(key=lambda x: x.get('precedence_score', 0), reverse=True)
            
            # Apply legal relevance ranking
            ranked_results = self._rank_results_by_legal_relevance(results, domain)
            
            # Add retrieval metadata
            for result in ranked_results:
                result['retrieval_strategy'] = 'precedence'
                result['confidence'] = 'high' if precedence_info.get('requires_precedence') else 'medium'
            
            return ranked_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error in precedence retrieval: {e}")
            return []

class RetrievalStrategyFactory:
    """Factory class to create appropriate retrieval strategy"""
    
    @staticmethod
    def create_strategy(strategy_name: str, vector_store, bengali_processor) -> BaseRetrievalStrategy:
        """Create retrieval strategy based on name"""
        strategies = {
            'direct_legal_retrieval': DirectLegalRetrieval,
            'conceptual_retrieval': ConceptualRetrieval,
            'multi_hop_retrieval': MultiHopRetrieval,
            'precedence_retrieval': PrecedenceRetrieval,
            'hybrid_retrieval': ConceptualRetrieval  # Default to conceptual for hybrid
        }
        
        strategy_class = strategies.get(strategy_name, ConceptualRetrieval)
        return strategy_class(vector_store, bengali_processor) 