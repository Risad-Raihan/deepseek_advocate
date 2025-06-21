"""
Legal Context Builder
Intelligent legal context construction optimized for language model consumption
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict

class LegalContextBuilder:
    """Advanced context builder for legal document retrieval and language model input"""
    
    def __init__(self, max_context_length: int = 2048):
        self.max_context_length = max_context_length
        self.setup_logging()
        self.legal_hierarchy_levels = self._define_hierarchy_levels()
        
    def setup_logging(self):
        """Setup logging for context builder"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _define_hierarchy_levels(self) -> Dict[str, int]:
        """Define legal document hierarchy importance levels"""
        return {
            'constitution': 5,      # Highest priority
            'law': 4,
            'ordinance': 4,
            'section': 3,
            'article': 3,
            'subsection': 2,
            'paragraph': 1,
            'general': 0
        }
    
    def build_hierarchical_context(self, documents: List[Dict], processed_query: Dict) -> str:
        """
        Build hierarchical legal context from retrieved documents
        
        Args:
            documents: Retrieved legal documents
            processed_query: Processed query information
            
        Returns:
            Structured legal context string
        """
        try:
            self.logger.info(f"Building hierarchical context from {len(documents)} documents")
            
            # Prioritize and organize documents
            prioritized_docs = self.prioritize_legal_sections(documents, processed_query)
            
            # Link related legal concepts
            linked_concepts = self.link_related_legal_concepts(prioritized_docs)
            
            # Build context sections
            context_sections = self._build_context_sections(prioritized_docs, linked_concepts)
            
            # Optimize context length
            optimized_context = self.optimize_context_length(context_sections, processed_query)
            
            # Create citation chain
            citation_chain = self.create_citation_chain(prioritized_docs)
            
            # Format final context
            final_context = self._format_final_context(
                optimized_context, 
                citation_chain, 
                processed_query
            )
            
            self.logger.info(f"Built context of {len(final_context)} characters")
            return final_context
            
        except Exception as e:
            self.logger.error(f"Error building hierarchical context: {e}")
            return self._build_fallback_context(documents)
    
    def prioritize_legal_sections(self, documents: List[Dict], processed_query: Dict) -> List[Dict]:
        """
        Prioritize legal sections based on hierarchy and relevance
        
        Args:
            documents: Retrieved documents
            processed_query: Query information
            
        Returns:
            Prioritized list of documents
        """
        try:
            # Calculate priority scores
            for doc in documents:
                priority_score = self._calculate_priority_score(doc, processed_query)
                doc['priority_score'] = priority_score
            
            # Sort by priority
            prioritized = sorted(documents, key=lambda x: x.get('priority_score', 0), reverse=True)
            
            # Group by legal hierarchy level
            grouped_docs = self._group_by_hierarchy(prioritized)
            
            # Reorder within groups for optimal flow
            reordered_docs = self._reorder_within_groups(grouped_docs)
            
            return reordered_docs
            
        except Exception as e:
            self.logger.error(f"Error prioritizing legal sections: {e}")
            return documents
    
    def _calculate_priority_score(self, document: Dict, processed_query: Dict) -> float:
        """Calculate priority score for a document"""
        try:
            base_score = document.get('legal_relevance', document.get('combined_score', 0))
            
            # Hierarchy level boost
            hierarchy_boost = self._get_hierarchy_boost(document)
            
            # Query domain alignment boost
            domain_boost = self._get_domain_alignment_boost(document, processed_query)
            
            # Entity matching boost
            entity_boost = self._get_entity_matching_boost(document, processed_query)
            
            # Legal specificity boost
            specificity_boost = self._get_legal_specificity_boost(document)
            
            return base_score * hierarchy_boost * domain_boost * entity_boost * specificity_boost
            
        except Exception as e:
            self.logger.error(f"Error calculating priority score: {e}")
            return 0.0
    
    def _get_hierarchy_boost(self, document: Dict) -> float:
        """Get boost based on legal document hierarchy"""
        try:
            metadata = document.get('metadata', {})
            content = str(metadata.get('paragraph_text', '') or metadata.get('content', ''))
            
            # Check for constitutional content
            if 'সংবিধান' in content or 'অনুচ্ছেদ' in content:
                return 1.5
            
            # Check for specific law sections
            if re.search(r'ধারা\s*\d+', content):
                return 1.3
            
            # Check for ordinances
            if 'অধ্যাদেশ' in content:
                return 1.2
            
            # Check for general legal content
            if any(term in content for term in ['আইন', 'বিধি', 'নিয়ম']):
                return 1.1
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error getting hierarchy boost: {e}")
            return 1.0
    
    def _get_domain_alignment_boost(self, document: Dict, processed_query: Dict) -> float:
        """Get boost based on domain alignment"""
        try:
            query_domain = processed_query.get('domain', {}).get('domain', 'general')
            metadata = document.get('metadata', {})
            content = str(metadata.get('paragraph_text', '') or metadata.get('content', ''))
            
            domain_keywords = {
                'family_law': ['তালাক', 'বিবাহ', 'খোরপোশ', 'পারিবারিক', 'উত্তরাধিকার'],
                'property_law': ['সম্পত্তি', 'জমি', 'দলিল', 'মালিকানা', 'রেজিস্ট্রেশন'],
                'constitutional_law': ['সংবিধান', 'মৌলিক অধিকার', 'নাগরিক', 'স্বাধীনতা'],
                'rent_control': ['ভাড়া', 'ইজারা', 'বাড়িওয়ালা', 'ভাড়াটিয়া'],
                'court_procedure': ['আদালত', 'মামলা', 'বিচার', 'প্রক্রিয়া', 'আপিল']
            }
            
            if query_domain in domain_keywords:
                keyword_matches = sum(1 for kw in domain_keywords[query_domain] if kw in content)
                return 1.0 + (keyword_matches * 0.05)
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error getting domain alignment boost: {e}")
            return 1.0
    
    def _get_entity_matching_boost(self, document: Dict, processed_query: Dict) -> float:
        """Get boost based on entity matching"""
        try:
            query_entities = processed_query.get('entities', {})
            if not query_entities:
                return 1.0
            
            metadata = document.get('metadata', {})
            content = str(metadata.get('paragraph_text', '') or metadata.get('content', ''))
            
            matches = 0
            total = 0
            
            for entity_type, entity_list in query_entities.items():
                for entity in entity_list:
                    total += 1
                    if str(entity).lower() in content.lower():
                        matches += 1
            
            if total > 0:
                match_ratio = matches / total
                return 1.0 + (match_ratio * 0.2)
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error getting entity matching boost: {e}")
            return 1.0
    
    def _get_legal_specificity_boost(self, document: Dict) -> float:
        """Get boost based on legal specificity"""
        try:
            metadata = document.get('metadata', {})
            content = str(metadata.get('paragraph_text', '') or metadata.get('content', ''))
            
            # Count specific legal references
            specific_refs = 0
            specific_refs += len(re.findall(r'ধারা\s*\d+', content))
            specific_refs += len(re.findall(r'অনুচ্ছেদ\s*\d+', content))
            specific_refs += len(re.findall(r'\d{4}\s*সালের', content))
            
            return 1.0 + min(specific_refs * 0.03, 0.15)
            
        except Exception as e:
            self.logger.error(f"Error getting legal specificity boost: {e}")
            return 1.0
    
    def _group_by_hierarchy(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """Group documents by legal hierarchy level"""
        try:
            groups = defaultdict(list)
            
            for doc in documents:
                hierarchy_level = self._determine_hierarchy_level(doc)
                groups[hierarchy_level].append(doc)
            
            return dict(groups)
            
        except Exception as e:
            self.logger.error(f"Error grouping by hierarchy: {e}")
            return {'general': documents}
    
    def _determine_hierarchy_level(self, document: Dict) -> str:
        """Determine the hierarchy level of a document"""
        try:
            metadata = document.get('metadata', {})
            content = str(metadata.get('paragraph_text', '') or metadata.get('content', ''))
            
            # Check content for hierarchy indicators
            if 'সংবিধান' in content or 'অনুচ্ছেদ' in content:
                return 'constitution'
            elif re.search(r'\d{4}\s*সালের.*আইন', content):
                return 'law'
            elif 'অধ্যাদেশ' in content:
                return 'ordinance'
            elif re.search(r'ধারা\s*\d+', content):
                return 'section'
            elif re.search(r'উপধারা', content):
                return 'subsection'
            else:
                return 'paragraph'
                
        except Exception as e:
            self.logger.error(f"Error determining hierarchy level: {e}")
            return 'general'
    
    def _reorder_within_groups(self, grouped_docs: Dict[str, List[Dict]]) -> List[Dict]:
        """Reorder documents within hierarchy groups for optimal flow"""
        try:
            reordered = []
            
            # Define processing order by hierarchy importance
            hierarchy_order = [
                'constitution', 'law', 'ordinance', 'section', 
                'subsection', 'paragraph', 'general'
            ]
            
            for hierarchy_level in hierarchy_order:
                if hierarchy_level in grouped_docs:
                    # Sort within group by priority score
                    group_docs = sorted(
                        grouped_docs[hierarchy_level],
                        key=lambda x: x.get('priority_score', 0),
                        reverse=True
                    )
                    reordered.extend(group_docs)
            
            return reordered
            
        except Exception as e:
            self.logger.error(f"Error reordering within groups: {e}")
            return sum(grouped_docs.values(), [])  # Flatten all groups
    
    def link_related_legal_concepts(self, documents: List[Dict]) -> Dict[str, List[str]]:
        """
        Identify and link related legal concepts across documents
        
        Args:
            documents: Prioritized documents
            
        Returns:
            Dictionary of concept relationships
        """
        try:
            concept_links = defaultdict(list)
            
            # Extract all legal concepts
            all_concepts = self._extract_all_concepts(documents)
            
            # Find concept relationships
            for i, doc1 in enumerate(documents):
                for j, doc2 in enumerate(documents):
                    if i != j:
                        relationships = self._find_concept_relationships(doc1, doc2)
                        if relationships:
                            doc_id1 = f"doc_{i}"
                            doc_id2 = f"doc_{j}"
                            concept_links[doc_id1].extend(relationships)
            
            return dict(concept_links)
            
        except Exception as e:
            self.logger.error(f"Error linking related legal concepts: {e}")
            return {}
    
    def _extract_all_concepts(self, documents: List[Dict]) -> Set[str]:
        """Extract all legal concepts from documents"""
        try:
            concepts = set()
            
            for doc in documents:
                metadata = doc.get('metadata', {})
                content = str(metadata.get('paragraph_text', '') or metadata.get('content', ''))
                
                # Extract legal concepts using patterns
                legal_patterns = [
                    r'ধারা\s*\d+',
                    r'অনুচ্ছেদ\s*\d+',
                    r'\d{4}\s*সালের\s*\w+',
                    r'আদালত',
                    r'মামলা',
                    r'বিচার',
                    r'আইন',
                    r'অধ্যাদেশ'
                ]
                
                for pattern in legal_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    concepts.update(matches)
            
            return concepts
            
        except Exception as e:
            self.logger.error(f"Error extracting concepts: {e}")
            return set()
    
    def _find_concept_relationships(self, doc1: Dict, doc2: Dict) -> List[str]:
        """Find relationships between concepts in two documents"""
        try:
            relationships = []
            
            metadata1 = doc1.get('metadata', {})
            metadata2 = doc2.get('metadata', {})
            
            content1 = str(metadata1.get('paragraph_text', '') or metadata1.get('content', ''))
            content2 = str(metadata2.get('paragraph_text', '') or metadata2.get('content', ''))
            
            # Find common legal references
            refs1 = re.findall(r'ধারা\s*\d+|অনুচ্ছেদ\s*\d+', content1)
            refs2 = re.findall(r'ধারা\s*\d+|অনুচ্ছেদ\s*\d+', content2)
            
            common_refs = set(refs1) & set(refs2)
            if common_refs:
                relationships.append(f"Common references: {', '.join(common_refs)}")
            
            # Find related terminology
            legal_terms1 = set(re.findall(r'আইন|অধ্যাদেশ|বিধি|নিয়ম', content1))
            legal_terms2 = set(re.findall(r'আইন|অধ্যাদেশ|বিধি|নিয়ম', content2))
            
            common_terms = legal_terms1 & legal_terms2
            if common_terms:
                relationships.append(f"Related terms: {', '.join(common_terms)}")
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error finding concept relationships: {e}")
            return []
    
    def _build_context_sections(self, documents: List[Dict], linked_concepts: Dict) -> List[Dict]:
        """Build structured context sections from documents"""
        try:
            sections = []
            
            for i, doc in enumerate(documents):
                metadata = doc.get('metadata', {})
                content = str(metadata.get('paragraph_text', '') or metadata.get('content', ''))
                
                # Determine section type
                section_type = self._determine_section_type(doc)
                
                # Build section
                section = {
                    'type': section_type,
                    'content': content,
                    'priority': doc.get('priority_score', 0),
                    'legal_relevance': doc.get('legal_relevance', 0),
                    'hierarchy_level': self._determine_hierarchy_level(doc),
                    'related_concepts': linked_concepts.get(f"doc_{i}", []),
                    'citations': self._extract_section_citations(content),
                    'length': len(content)
                }
                
                sections.append(section)
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error building context sections: {e}")
            return []
    
    def _determine_section_type(self, document: Dict) -> str:
        """Determine the type of legal section"""
        try:
            metadata = document.get('metadata', {})
            
            # Check source level from retrieval
            if 'source_level' in document:
                return document['source_level']
            
            # Determine from content
            content = str(metadata.get('paragraph_text', '') or metadata.get('content', ''))
            
            if 'সংবিধান' in content:
                return 'constitutional'
            elif re.search(r'ধারা\s*\d+', content):
                return 'legal_section'
            elif 'আদালত' in content or 'বিচার' in content:
                return 'procedural'
            else:
                return 'general_legal'
                
        except Exception as e:
            self.logger.error(f"Error determining section type: {e}")
            return 'general'
    
    def _extract_section_citations(self, content: str) -> List[str]:
        """Extract legal citations from section content"""
        try:
            citations = []
            
            # Extract various citation types
            section_refs = re.findall(r'ধারা\s*\d+(?:\([ক-৯]+\))?', content)
            article_refs = re.findall(r'অনুচ্ছেদ\s*\d+(?:\([ক-৯]+\))?', content)
            law_refs = re.findall(r'\d{4}\s*সালের\s*[^।]+?(?:আইন|অধ্যাদেশ)', content)
            
            citations.extend([f"ধারা {ref}" for ref in section_refs])
            citations.extend([f"অনুচ্ছেদ {ref}" for ref in article_refs])
            citations.extend(law_refs)
            
            return list(set(citations))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error extracting section citations: {e}")
            return [] 