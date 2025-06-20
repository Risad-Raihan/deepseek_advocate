"""
Bengali Legal Text Processor
Handles Bengali text preprocessing, legal entity recognition, and response formatting
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
import nltk
from collections import defaultdict

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BengaliLegalProcessor:
    """Advanced Bengali text processor for legal documents"""
    
    def __init__(self):
        self.setup_logging()
        self.legal_entities = self._load_legal_entities()
        self.legal_terms = self._load_legal_terms()
        self.citation_patterns = self._setup_citation_patterns()
        
    def setup_logging(self):
        """Setup logging for Bengali processor"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _load_legal_entities(self) -> Dict[str, List[str]]:
        """Load Bengali legal entities and their patterns"""
        return {
            'laws': [
                'সংবিধান', 'আইন', 'অধ্যাদেশ', 'বিধি', 'নিয়ম', 'প্রবিধান',
                'দণ্ডবিধি', 'ফৌজদারি', 'দেওয়ানী', 'পারিবারিক আইন'
            ],
            'sections': [
                'ধারা', 'উপধারা', 'অনুচ্ছেদ', 'খণ্ড', 'ভাগ', 'পরিচ্ছেদ',
                'তফসিল', 'তালিকা', 'সূচী'
            ],
            'legal_terms': [
                'আদালত', 'ট্রাইব্যুনাল', 'বিচারক', 'ম্যাজিস্ট্রেট',
                'আইনজীবী', 'উকিল', 'ব্যারিস্টার', 'মামলা', 'মোকদ্দমা',
                'রায়', 'আদেশ', 'ডিক্রি', 'নোটিশ', 'সমন', 'ওয়ারেন্ট'
            ],
            'rights': [
                'অধিকার', 'কর্তব্য', 'দায়িত্ব', 'ক্ষমতা', 'স্বাধীনতা',
                'সুবিধা', 'সম্পত্তি', 'উত্তরাধিকার', 'ভরণপোষণ'
            ]
        }
    
    def _load_legal_terms(self) -> Dict[str, str]:
        """Load legal term mappings for normalization"""
        return {
            'আইন': ['আইন', 'আইনি', 'আইনের'],
            'ধারা': ['ধারা', 'ধারায়', 'ধারার', 'ধারাটি'],
            'আদালত': ['আদালত', 'আদালতে', 'আদালতের', 'কোর্ট'],
            'মামলা': ['মামলা', 'মামলায়', 'মামলার', 'কেস'],
            'বিচার': ['বিচার', 'বিচারে', 'বিচারের', 'বিচারক'],
            'অধ্যাদেশ': ['অধ্যাদেশ', 'অধ্যাদেশে', 'অধ্যাদেশের', 'অর্ডিন্যান্স']
        }
    
    def _setup_citation_patterns(self) -> Dict[str, str]:
        """Setup regex patterns for legal citations"""
        return {
            'section': r'ধারা\s*(\d+(?:\([ক-৯]+\))?(?:\s*উপধারা\s*\([ক-৯]+\))?)',
            'article': r'অনুচ্ছেদ\s*(\d+(?:\([ক-৯]+\))?)',
            'law_year': r'(\d{4})\s*সালের\s*(.+?)\s*আইন',
            'ordinance_year': r'(\d{4})\s*সালের\s*(.+?)\s*অধ্যাদেশ',
            'case_reference': r'([A-Z][A-Z\s]+)\s*বনাম\s*([A-Z][A-Z\s]+)',
            'legal_notice': r'আইনি\s*নোটিশ|লিগ্যাল\s*নোটিশ'
        }
    
    def preprocess_bengali_legal_text(self, text: str) -> str:
        """
        Preprocess Bengali legal text with legal-specific cleaning
        
        Args:
            text: Raw Bengali legal text
            
        Returns:
            Preprocessed and normalized text
        """
        try:
            # Remove extra whitespace and normalize
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Normalize Bengali numbers to English for processing
            bengali_to_english = {
                '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
                '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'
            }
            
            for bengali, english in bengali_to_english.items():
                text = text.replace(bengali, english)
            
            # Fix common OCR errors in Bengali legal documents
            ocr_fixes = {
                'ব্যি': 'বি', 'ত্বি': 'তি', 'ক্ষি': 'ক্ষ', 'জ্ঞি': 'জ্ঞ',
                'হবে।': 'হবে।', 'করা': 'করা', 'হয়': 'হয়'
            }
            
            for error, correction in ocr_fixes.items():
                text = text.replace(error, correction)
            
            # Standardize legal punctuation
            text = re.sub(r'।\s*', '। ', text)  # Fix period spacing
            text = re.sub(r':\s*', ': ', text)  # Fix colon spacing
            text = re.sub(r';\s*', '; ', text)  # Fix semicolon spacing
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Error preprocessing Bengali text: {e}")
            return text
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract legal entities from Bengali text
        
        Args:
            text: Preprocessed Bengali legal text
            
        Returns:
            Dictionary of extracted legal entities
        """
        entities = defaultdict(list)
        
        try:
            # Extract sections and articles
            for pattern_name, pattern in self.citation_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE | re.UNICODE)
                if matches:
                    entities[pattern_name].extend([match if isinstance(match, str) else ' '.join(match) for match in matches])
            
            # Extract legal terms
            words = text.split()
            for i, word in enumerate(words):
                # Check for legal entity matches
                for entity_type, terms in self.legal_entities.items():
                    for term in terms:
                        if term in word or word.startswith(term):
                            context = ' '.join(words[max(0, i-2):i+3])  # Get context
                            entities[entity_type].append({
                                'term': word,
                                'context': context,
                                'position': i
                            })
            
            # Remove duplicates
            for key in entities:
                if isinstance(entities[key][0], dict):
                    # For complex entities, remove duplicates based on term and context
                    seen = set()
                    unique_entities = []
                    for entity in entities[key]:
                        identifier = f"{entity['term']}_{entity['context']}"
                        if identifier not in seen:
                            seen.add(identifier)
                            unique_entities.append(entity)
                    entities[key] = unique_entities
                else:
                    # For simple entities, remove direct duplicates
                    entities[key] = list(set(entities[key]))
            
            return dict(entities)
            
        except Exception as e:
            self.logger.error(f"Error extracting legal entities: {e}")
            return {}
    
    def extract_legal_intent(self, query: str) -> Dict[str, any]:
        """
        Classify legal query intent and extract key information
        
        Args:
            query: User's legal query in Bengali
            
        Returns:
            Intent classification and extracted information
        """
        try:
            query = query.strip()
            intent_info = {
                'intent': 'general',
                'confidence': 0.0,
                'entities': {},
                'legal_domain': 'general',
                'urgency': 'normal'
            }
            
            # Define intent patterns
            intent_patterns = {
                'divorce': [
                    'তালাক', 'বিবাহবিচ্ছেদ', 'স্ত্রী ত্যাগ', 'পরিত্যাগ',
                    'খোরপোশ', 'ভরণপোষণ', 'দেনমোহর'
                ],
                'property': [
                    'সম্পত্তি', 'জমি', 'বাড়ি', 'উত্তরাধিকার', 'দখল',
                    'রেজিস্ট্রেশন', 'দলিল', 'মালিকানা'
                ],
                'rent': [
                    'ভাড়া', 'বাড়িভাড়া', 'ভাড়াটিয়া', 'বাড়িওয়ালা',
                    'ইজারা', 'ভাড়া বৃদ্ধি'
                ],
                'family': [
                    'পারিবারিক', 'বিয়ে', 'বিবাহ', 'সন্তান', 'অভিভাবক',
                    'পিতৃত্ব', 'মাতৃত্ব'
                ],
                'constitutional': [
                    'সংবিধান', 'মৌলিক অধিকার', 'নাগরিক অধিকার',
                    'জীবনের অধিকার', 'স্বাধীনতা'
                ],
                'procedure': [
                    'মামলা দায়ের', 'আদালত', 'প্রক্রিয়া', 'নোটিশ',
                    'আপিল', 'রিভিশন', 'জামিন'
                ]
            }
            
            # Calculate intent scores
            max_score = 0
            best_intent = 'general'
            
            for intent, keywords in intent_patterns.items():
                score = sum(1 for keyword in keywords if keyword in query)
                if score > max_score:
                    max_score = score
                    best_intent = intent
            
            intent_info['intent'] = best_intent
            intent_info['confidence'] = min(max_score / 3.0, 1.0)  # Normalize confidence
            
            # Extract entities from query
            intent_info['entities'] = self.extract_legal_entities(query)
            
            # Determine legal domain
            domain_mapping = {
                'divorce': 'family_law',
                'family': 'family_law',
                'property': 'property_law',
                'rent': 'property_law',
                'constitutional': 'constitutional_law',
                'procedure': 'procedural_law'
            }
            intent_info['legal_domain'] = domain_mapping.get(best_intent, 'general_law')
            
            # Determine urgency
            urgent_keywords = ['জরুরি', 'তাৎক্ষণিক', 'দ্রুত', 'এখনই', 'আজই']
            if any(keyword in query for keyword in urgent_keywords):
                intent_info['urgency'] = 'high'
            
            return intent_info
            
        except Exception as e:
            self.logger.error(f"Error extracting legal intent: {e}")
            return {'intent': 'general', 'confidence': 0.0, 'entities': {}}
    
    def format_bengali_legal_response(self, response: str, citations: List[Dict], 
                                    intent_info: Dict) -> str:
        """
        Format legal response with proper Bengali legal citations and structure
        
        Args:
            response: Generated legal response
            citations: Legal citations to include
            intent_info: Query intent information
            
        Returns:
            Properly formatted Bengali legal response
        """
        try:
            formatted_response = []
            
            # Add response header based on intent
            intent_headers = {
                'divorce': '🏛️ তালাক ও পারিবারিক আইন সংক্রান্ত পরামর্শ:',
                'property': '🏡 সম্পত্তি আইন সংক্রান্ত পরামর্শ:',
                'rent': '🏠 বাড়ি ভাড়া আইন সংক্রান্ত পরামর্শ:',
                'constitutional': '⚖️ সাংবিধানিক আইন সংক্রান্ত পরামর্শ:',
                'procedure': '📋 আদালতি প্রক্রিয়া সংক্রান্ত পরামর্শ:',
                'general': '⚖️ আইনি পরামর্শ:'
            }
            
            header = intent_headers.get(intent_info.get('intent', 'general'), intent_headers['general'])
            formatted_response.append(header)
            formatted_response.append("")
            
            # Format main response
            paragraphs = response.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    formatted_response.append(para.strip())
                    formatted_response.append("")
            
            # Add legal citations section
            if citations:
                formatted_response.append("📚 সংশ্লিষ্ট আইনি ধারা ও রেফারেন্স:")
                formatted_response.append("")
                
                for i, citation in enumerate(citations, 1):
                    citation_text = f"{i}. "
                    
                    if citation.get('law_name'):
                        citation_text += f"**{citation['law_name']}** "
                    
                    if citation.get('section'):
                        citation_text += f"ধারা {citation['section']} "
                    
                    if citation.get('subsection'):
                        citation_text += f"উপধারা ({citation['subsection']}) "
                    
                    if citation.get('description'):
                        citation_text += f"- {citation['description']}"
                    
                    formatted_response.append(citation_text)
                
                formatted_response.append("")
            
            # Add legal disclaimer
            formatted_response.append("⚠️ **আইনি দাবিত্যাগ:**")
            formatted_response.append("এই পরামর্শটি সাধারণ তথ্যের জন্য প্রদান করা হয়েছে। নির্দিষ্ট আইনি পরামর্শের জন্য অভিজ্ঞ আইনজীবীর সাথে পরামর্শ করুন।")
            
            return '\n'.join(formatted_response)
            
        except Exception as e:
            self.logger.error(f"Error formatting Bengali legal response: {e}")
            return response  # Return original response if formatting fails
    
    def validate_legal_citations(self, citations: List[Dict]) -> List[Dict]:
        """
        Validate and clean legal citations
        
        Args:
            citations: List of citation dictionaries
            
        Returns:
            Validated and cleaned citations
        """
        validated_citations = []
        
        try:
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                
                # Required fields check
                if not citation.get('law_name') and not citation.get('section'):
                    continue
                
                # Clean and validate citation
                clean_citation = {}
                
                # Clean law name
                if citation.get('law_name'):
                    law_name = citation['law_name'].strip()
                    # Ensure proper Bengali formatting
                    clean_citation['law_name'] = law_name
                
                # Validate section number
                if citation.get('section'):
                    section = str(citation['section']).strip()
                    if re.match(r'^\d+(\([ক-৯]+\))?$', section):
                        clean_citation['section'] = section
                
                # Clean description
                if citation.get('description'):
                    clean_citation['description'] = citation['description'].strip()
                
                # Add relevance score if available
                if citation.get('relevance_score'):
                    clean_citation['relevance_score'] = float(citation['relevance_score'])
                
                if clean_citation:  # Only add if we have valid content
                    validated_citations.append(clean_citation)
            
            # Sort by relevance score if available
            validated_citations.sort(
                key=lambda x: x.get('relevance_score', 0.0), 
                reverse=True
            )
            
            return validated_citations[:5]  # Return top 5 citations
            
        except Exception as e:
            self.logger.error(f"Error validating legal citations: {e}")
            return citations  # Return original if validation fails

    def normalize_bengali_legal_terms(self, text: str) -> str:
        """
        Normalize Bengali legal terminology for consistency
        
        Args:
            text: Text containing Bengali legal terms
            
        Returns:
            Text with normalized legal terms
        """
        try:
            # Normalize common legal term variations
            for standard_term, variations in self.legal_terms.items():
                for variation in variations:
                    if variation != standard_term:
                        text = text.replace(variation, standard_term)
            
            # Fix common spacing and punctuation issues
            text = re.sub(r'।\s*', '। ', text)
            text = re.sub(r':\s*', ': ', text)
            text = re.sub(r'(\d+)\s*\.\s*', r'\1. ', text)
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Error normalizing Bengali legal terms: {e}")
            return text 