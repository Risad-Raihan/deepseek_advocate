"""
Bengali Legal Query Processor
Advanced query understanding for Bengali legal documents with domain classification and entity extraction
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import json

class BengaliLegalQueryProcessor:
    """Advanced processor for Bengali legal queries with domain expertise"""
    
    def __init__(self):
        self.setup_logging()
        self.legal_domains = self._initialize_legal_domains()
        self.legal_entities_patterns = self._setup_entity_patterns()
        self.query_expansion_terms = self._load_expansion_terms()
        self.legal_precedence_indicators = self._setup_precedence_indicators()
        
    def setup_logging(self):
        """Setup logging for query processor"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _initialize_legal_domains(self) -> Dict[str, Dict[str, Any]]:
        """Initialize legal domain classifications with Bengali keywords"""
        return {
            'family_law': {
                'name_bn': 'পারিবারিক আইন',
                'keywords': [
                    'তালাক', 'বিবাহবিচ্ছেদ', 'খোরপোশ', 'ভরণপোষণ', 'দেনমোহর',
                    'বিবাহ', 'বিয়ে', 'স্ত্রী', 'স্বামী', 'সন্তান', 'অভিভাবকত্ব',
                    'উত্তরাধিকার', 'মিরাস', 'ওয়ারিশ', 'পিতৃত্ব', 'মাতৃত্ব'
                ],
                'laws': [
                    'মুসলিম পারিবারিক আইন অধ্যাদেশ ১৯৬১',
                    'পারিবারিক আদালত অধ্যাদেশ ১৯৮৫',
                    'তালাক ও খোরপোশ আইন'
                ],
                'priority': 0.9
            },
            'property_law': {
                'name_bn': 'সম্পত্তি আইন',
                'keywords': [
                    'সম্পত্তি', 'জমি', 'জমিজমা', 'বাড়ি', 'ভূমি', 'দখল',
                    'মালিকানা', 'স্বত্ব', 'দলিল', 'রেজিস্ট্রেশন', 'খতিয়ান',
                    'পর্চা', 'মৌজা', 'দাগ', 'ক্রয়', 'বিক্রয়', 'হস্তান্তর'
                ],
                'laws': [
                    'সম্পত্তি আইন', 'রেজিস্ট্রেশন আইন', 'ভূমি আইন'
                ],
                'priority': 0.8
            },
            'rent_control': {
                'name_bn': 'বাড়ি ভাড়া আইন',
                'keywords': [
                    'ভাড়া', 'বাড়িভাড়া', 'ভাড়াটিয়া', 'বাড়িওয়ালা', 'মালিক',
                    'ইজারা', 'ভাড়া বৃদ্ধি', 'উচ্ছেদ', 'খালি', 'দখল', 'টেন্যান্ট'
                ],
                'laws': [
                    'বাড়ী ভাড়া নিয়ন্ত্রণ আইন ১৯৯১'
                ],
                'priority': 0.85
            },
            'constitutional_law': {
                'name_bn': 'সাংবিধানিক আইন',
                'keywords': [
                    'সংবিধান', 'মৌলিক অধিকার', 'নাগরিক অধিকার', 'স্বাধীনতা',
                    'সমতা', 'ন্যায়বিচার', 'রাষ্ট্র', 'সরকার', 'আইনের শাসন',
                    'জীবনের অধিকার', 'বাকস্বাধীনতা', 'চলাফেরার স্বাধীনতা'
                ],
                'laws': [
                    'বাংলাদেশের সংবিধান'
                ],
                'priority': 0.95
            },
            'court_procedure': {
                'name_bn': 'আদালতি প্রক্রিয়া',
                'keywords': [
                    'আদালত', 'মামলা', 'মোকদ্দমা', 'দায়ের', 'আর্জি', 'আবেদন',
                    'আপিল', 'রিভিশন', 'জামিন', 'নোটিশ', 'সমন', 'ওয়ারেন্ট',
                    'বিচার', 'রায়', 'আদেশ', 'ডিক্রি', 'সাক্ষী', 'প্রমাণ'
                ],
                'laws': [
                    'মামলা দায়ের, আদালতের রীতি ও কার্যপদ্ধতি',
                    'দেওয়ানি কার্যবিধি', 'ফৌজদারি কার্যবিধি'
                ],
                'priority': 0.7
            },
            'criminal_law': {
                'name_bn': 'ফৌজদারি আইন',
                'keywords': [
                    'অপরাধ', 'দণ্ড', 'সাজা', 'জরিমানা', 'কারাদণ্ড', 'গ্রেফতার',
                    'চুরি', 'ডাকাতি', 'হত্যা', 'আঘাত', 'ধর্ষণ', 'জালিয়াতি'
                ],
                'laws': [
                    'দণ্ডবিধি', 'ফৌজদারি কার্যবিধি'
                ],
                'priority': 0.75
            }
        }
    
    def _setup_entity_patterns(self) -> Dict[str, str]:
        """Setup regex patterns for Bengali legal entity extraction"""
        return {
            'section': r'ধারা\s*(\d+(?:\([ক-৯]+\))?(?:\s*উপধারা\s*\([ক-৯]+\))?)',
            'article': r'অনুচ্ছেদ\s*(\d+(?:\([ক-৯]+\))?)',
            'law_with_year': r'(\d{4})\s*সালের\s*(.+?)\s*(?:আইন|অধ্যাদেশ)',
            'court_type': r'(সুপ্রিম\s*কোর্ট|হাইকোর্ট|জেলা\s*জজ\s*আদালত|মেট্রোপলিটন\s*ম্যাজিস্ট্রেট|পারিবারিক\s*আদালত)',
            'legal_action': r'(মামলা\s*দায়ের|আপিল|রিভিশন|জামিন|আর্জি|আবেদন)',
            'time_reference': r'(\d+)\s*(দিন|মাস|বছর)\s*(পূর্বে|পরে|মধ্যে)',
            'money_amount': r'(\d+(?:,\d+)*)\s*টাকা',
            'legal_document': r'(দলিল|খতিয়ান|পর্চা|নোটিশ|সমন|ওয়ারেন্ট|ডিক্রি)'
        }
    
    def _load_expansion_terms(self) -> Dict[str, List[str]]:
        """Load query expansion terms for better retrieval"""
        return {
            'তালাক': ['বিবাহবিচ্ছেদ', 'স্ত্রী ত্যাগ', 'পরিত্যাগ', 'খুলা', 'মুবারাত'],
            'সম্পত্তি': ['জমিজমা', 'ভূমি', 'বাড়িঘর', 'স্থাবর সম্পত্তি', 'অস্থাবর সম্পত্তি'],
            'ভাড়া': ['ইজারা', 'টেন্যান্সি', 'লিজ', 'ভাড়াটিয়া'],
            'আদালত': ['কোর্ট', 'ট্রাইব্যুনাল', 'বিচারালয়', 'ন্যায়ালয়'],
            'মামলা': ['মোকদ্দমা', 'কেস', 'মামলা-মোকদ্দমা', 'বিচার'],
            'অধিকার': ['অধিকার', 'স্বাধীনতা', 'সুবিধা', 'ক্ষমতা'],
            'আইন': ['বিধি', 'নিয়ম', 'প্রবিধান', 'অধ্যাদেশ'],
            'খোরপোশ': ['ভরণপোষণ', 'নফকা', 'ভরণপোষণ খরচ'],
            'উত্তরাধিকার': ['মিরাস', 'ওয়ারিশ', 'উত্তরাধিকার সূত্রে প্রাপ্ত']
        }
    
    def _setup_precedence_indicators(self) -> List[str]:
        """Setup indicators for legal precedence requirements"""
        return [
            'পূর্ববর্তী', 'আগের', 'প্রথমে', 'শর্ত', 'প্রয়োজন', 'আবশ্যক',
            'অবশ্যই', 'লাগবে', 'করতে হবে', 'পূরণ', 'সাপেক্ষে', 'নির্ভর'
        ]
    
    def process_legal_query(self, query: str) -> Dict[str, Any]:
        """
        Main function to process Bengali legal queries
        
        Args:
            query: Bengali legal query string
            
        Returns:
            Comprehensive query analysis dictionary
        """
        try:
            self.logger.info(f"Processing legal query: {query[:50]}...")
            
            # Clean and preprocess query
            clean_query = self._preprocess_query(query)
            
            # Classify legal domain
            domain_info = self.classify_legal_domain(clean_query)
            
            # Extract legal entities
            entities = self.extract_legal_entities_from_query(clean_query)
            
            # Expand query with related terms
            expanded_query = self.expand_query_with_legal_terms(clean_query, domain_info['domain'])
            
            # Identify precedence requirements
            precedence_info = self.identify_legal_precedence_requirements(clean_query)
            
            # Handle multi-part questions
            multi_part_info = self.handle_multi_part_legal_questions(clean_query)
            
            # Determine query complexity and retrieval strategy
            complexity = self._assess_query_complexity(clean_query, entities, multi_part_info)
            
            processed_query = {
                'original_query': query,
                'clean_query': clean_query,
                'domain': domain_info,
                'entities': entities,
                'expanded_query': expanded_query,
                'precedence_requirements': precedence_info,
                'multi_part_analysis': multi_part_info,
                'complexity': complexity,
                'suggested_retrieval_strategy': self._suggest_retrieval_strategy(domain_info, complexity),
                'processing_metadata': {
                    'query_length': len(query),
                    'clean_query_length': len(clean_query),
                    'entity_count': sum(len(v) for v in entities.values()),
                    'expansion_terms_added': len(expanded_query.split()) - len(clean_query.split())
                }
            }
            
            self.logger.info(f"Query processed successfully: Domain={domain_info['domain']}, Complexity={complexity}")
            return processed_query
            
        except Exception as e:
            self.logger.error(f"Error processing legal query: {e}")
            return {
                'original_query': query,
                'error': str(e),
                'domain': {'domain': 'general', 'confidence': 0.0}
            }
    
    def _preprocess_query(self, query: str) -> str:
        """Clean and preprocess Bengali legal query"""
        try:
            # Remove extra whitespace
            query = re.sub(r'\s+', ' ', query.strip())
            
            # Normalize Bengali punctuation
            query = re.sub(r'।+', '।', query)
            query = re.sub(r'\?+', '?', query)
            
            # Fix common OCR errors in Bengali text
            ocr_fixes = {
                'ব্যি': 'বি', 'ত্বি': 'তি', 'ক্ষি': 'ক্ষ', 'জ্ঞি': 'জ্ঞ'
            }
            
            for error, correction in ocr_fixes.items():
                query = query.replace(error, correction)
            
            # Normalize Bengali numbers
            bengali_to_english = {
                '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
                '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'
            }
            
            for bengali, english in bengali_to_english.items():
                query = query.replace(bengali, english)
            
            return query.strip()
            
        except Exception as e:
            self.logger.error(f"Error preprocessing query: {e}")
            return query
    
    def classify_legal_domain(self, query: str) -> Dict[str, Any]:
        """
        Classify legal query into appropriate domain
        
        Args:
            query: Preprocessed Bengali legal query
            
        Returns:
            Domain classification with confidence scores
        """
        try:
            query_lower = query.lower()
            domain_scores = {}
            
            # Calculate scores for each domain
            for domain_key, domain_info in self.legal_domains.items():
                score = 0
                matched_keywords = []
                
                # Score based on keyword matches
                for keyword in domain_info['keywords']:
                    if keyword.lower() in query_lower:
                        score += domain_info['priority']
                        matched_keywords.append(keyword)
                
                # Bonus for exact law mentions
                for law in domain_info['laws']:
                    if any(part.lower() in query_lower for part in law.split() if len(part) > 3):
                        score += 2.0
                        matched_keywords.append(f"Law: {law}")
                
                if score > 0:
                    domain_scores[domain_key] = {
                        'score': score,
                        'matched_keywords': matched_keywords,
                        'domain_info': domain_info
                    }
            
            # Determine primary domain
            if domain_scores:
                primary_domain = max(domain_scores.keys(), key=lambda x: domain_scores[x]['score'])
                confidence = min(domain_scores[primary_domain]['score'] / 3.0, 1.0)
                
                # Sort domains by score for secondary domains
                sorted_domains = sorted(domain_scores.items(), 
                                      key=lambda x: x[1]['score'], reverse=True)
                
                return {
                    'domain': primary_domain,
                    'domain_name_bn': self.legal_domains[primary_domain]['name_bn'],
                    'confidence': confidence,
                    'matched_keywords': domain_scores[primary_domain]['matched_keywords'],
                    'all_domain_scores': {k: v['score'] for k, v in sorted_domains},
                    'secondary_domains': [k for k, v in sorted_domains[1:3]]  # Top 2 secondary domains
                }
            else:
                return {
                    'domain': 'general',
                    'domain_name_bn': 'সাধারণ আইন',
                    'confidence': 0.0,
                    'matched_keywords': [],
                    'all_domain_scores': {},
                    'secondary_domains': []
                }
                
        except Exception as e:
            self.logger.error(f"Error classifying legal domain: {e}")
            return {'domain': 'general', 'confidence': 0.0}
    
    def extract_legal_entities_from_query(self, query: str) -> Dict[str, List[Any]]:
        """
        Extract legal entities from Bengali query
        
        Args:
            query: Bengali legal query
            
        Returns:
            Dictionary of extracted legal entities
        """
        entities = defaultdict(list)
        
        try:
            # Extract using predefined patterns
            for entity_type, pattern in self.legal_entities_patterns.items():
                matches = re.findall(pattern, query, re.IGNORECASE | re.UNICODE)
                if matches:
                    entities[entity_type].extend(matches)
            
            # Extract legal terminology
            legal_terms = []
            for domain_info in self.legal_domains.values():
                for keyword in domain_info['keywords']:
                    if keyword.lower() in query.lower():
                        legal_terms.append(keyword)
            
            if legal_terms:
                entities['legal_terms'] = list(set(legal_terms))
            
            # Extract question indicators
            question_indicators = []
            question_words = ['কী', 'কি', 'কীভাবে', 'কোন', 'কার', 'কাকে', 'কখন', 'কোথায়', 'কেন']
            for word in question_words:
                if word in query:
                    question_indicators.append(word)
            
            if question_indicators:
                entities['question_indicators'] = question_indicators
            
            # Convert defaultdict to regular dict
            return dict(entities)
            
        except Exception as e:
            self.logger.error(f"Error extracting legal entities: {e}")
            return {}
    
    def expand_query_with_legal_terms(self, query: str, domain: str) -> str:
        """
        Expand query with related legal terminology for better retrieval
        
        Args:
            query: Original query
            domain: Identified legal domain
            
        Returns:
            Expanded query with additional terms
        """
        try:
            expanded_terms = []
            query_words = query.split()
            
            # Add domain-specific expansion terms
            for word in query_words:
                if word.lower() in self.query_expansion_terms:
                    expanded_terms.extend(self.query_expansion_terms[word.lower()][:2])  # Add top 2 related terms
            
            # Add domain-specific keywords
            if domain in self.legal_domains:
                domain_keywords = self.legal_domains[domain]['keywords'][:3]  # Top 3 domain keywords
                expanded_terms.extend(domain_keywords)
            
            # Remove duplicates and create expanded query
            unique_expansions = list(set(expanded_terms))
            expanded_query = query + " " + " ".join(unique_expansions)
            
            return expanded_query.strip()
            
        except Exception as e:
            self.logger.error(f"Error expanding query: {e}")
            return query
    
    def identify_legal_precedence_requirements(self, query: str) -> Dict[str, Any]:
        """
        Identify if query requires understanding of legal precedence or procedures
        
        Args:
            query: Bengali legal query
            
        Returns:
            Precedence requirement analysis
        """
        try:
            precedence_score = 0
            matched_indicators = []
            
            # Check for precedence indicators
            for indicator in self.legal_precedence_indicators:
                if indicator in query:
                    precedence_score += 1
                    matched_indicators.append(indicator)
            
            # Check for procedural questions
            procedural_patterns = [
                r'কীভাবে\s+(.+)\s+করবো?',
                r'কী\s+করতে\s+হবে',
                r'প্রক্রিয়া\s+কী',
                r'পদ্ধতি\s+কী',
                r'শর্ত\s+কী'
            ]
            
            procedural_matches = []
            for pattern in procedural_patterns:
                matches = re.findall(pattern, query, re.IGNORECASE | re.UNICODE)
                procedural_matches.extend(matches)
            
            requires_precedence = precedence_score > 0 or len(procedural_matches) > 0
            
            return {
                'requires_precedence': requires_precedence,
                'precedence_score': precedence_score,
                'matched_indicators': matched_indicators,
                'procedural_matches': procedural_matches,
                'suggested_retrieval_depth': min(3, max(1, precedence_score + len(procedural_matches)))
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying precedence requirements: {e}")
            return {'requires_precedence': False, 'precedence_score': 0}
    
    def handle_multi_part_legal_questions(self, query: str) -> Dict[str, Any]:
        """
        Handle queries with multiple legal questions or aspects
        
        Args:
            query: Bengali legal query
            
        Returns:
            Multi-part question analysis
        """
        try:
            # Split by common conjunctions and question markers
            split_patterns = [
                r'এবং\s*',
                r'ও\s*',
                r'আর\s*',
                r'তাছাড়া\s*',
                r'এছাড়া\s*',
                r'কিন্তু\s*',
                r'তবে\s*'
            ]
            
            parts = [query]
            for pattern in split_patterns:
                new_parts = []
                for part in parts:
                    new_parts.extend(re.split(pattern, part))
                parts = new_parts
            
            # Clean and filter parts
            meaningful_parts = []
            for part in parts:
                part = part.strip()
                if len(part) > 10 and any(qw in part for qw in ['কী', 'কি', 'কীভাবে', 'কোন']):
                    meaningful_parts.append(part)
            
            is_multi_part = len(meaningful_parts) > 1
            
            return {
                'is_multi_part': is_multi_part,
                'question_parts': meaningful_parts,
                'part_count': len(meaningful_parts),
                'complexity_multiplier': 1.5 if is_multi_part else 1.0
            }
            
        except Exception as e:
            self.logger.error(f"Error handling multi-part questions: {e}")
            return {'is_multi_part': False, 'question_parts': [query], 'part_count': 1}
    
    def _assess_query_complexity(self, query: str, entities: Dict, multi_part_info: Dict) -> str:
        """Assess overall query complexity"""
        try:
            complexity_score = 0
            
            # Length-based scoring
            if len(query) > 100:
                complexity_score += 2
            elif len(query) > 50:
                complexity_score += 1
            
            # Entity-based scoring
            entity_count = sum(len(v) for v in entities.values())
            complexity_score += min(entity_count, 3)
            
            # Multi-part scoring
            if multi_part_info.get('is_multi_part', False):
                complexity_score += 2
            
            # Legal terminology density
            legal_term_count = len(entities.get('legal_terms', []))
            if legal_term_count > 3:
                complexity_score += 2
            elif legal_term_count > 1:
                complexity_score += 1
            
            # Determine complexity level
            if complexity_score >= 7:
                return 'high'
            elif complexity_score >= 4:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            self.logger.error(f"Error assessing query complexity: {e}")
            return 'medium'
    
    def _suggest_retrieval_strategy(self, domain_info: Dict, complexity: str) -> str:
        """Suggest optimal retrieval strategy based on query analysis"""
        try:
            domain = domain_info.get('domain', 'general')
            confidence = domain_info.get('confidence', 0.0)
            
            # High-confidence, specific domain queries
            if confidence > 0.8 and domain != 'general':
                if complexity == 'high':
                    return 'multi_hop_retrieval'
                else:
                    return 'direct_legal_retrieval'
            
            # Medium confidence or complex queries
            elif confidence > 0.5 or complexity in ['medium', 'high']:
                return 'conceptual_retrieval'
            
            # Low confidence or simple queries
            else:
                return 'hybrid_retrieval'
                
        except Exception as e:
            self.logger.error(f"Error suggesting retrieval strategy: {e}")
            return 'hybrid_retrieval' 