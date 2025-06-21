"""
Legal Q&A Generation Engine - Phase 3
Generates high-quality Bengali legal Q&A pairs from processed documents
"""

import logging
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime

class LegalQAGenerator:
    """Advanced Q&A generation engine for Bengali legal documents"""
    
    def __init__(self, legal_rag, bengali_processor, vector_store, config=None):
        """
        Initialize Q&A generator with Phase 2 components
        
        Args:
            legal_rag: LegalRAGEngine instance from Phase 2
            bengali_processor: BengaliLegalProcessor instance
            vector_store: LegalVectorStore instance
            config: Configuration dictionary
        """
        self.legal_rag = legal_rag
        self.bengali_processor = bengali_processor
        self.vector_store = vector_store
        self.config = config or {}
        
        self.setup_logging()
        self.initialize_question_templates()
        self.initialize_legal_domains()
        
        self.generated_qa_pairs = []
        self.quality_stats = defaultdict(int)
        
    def setup_logging(self):
        """Setup logging for QA generator"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
    def initialize_question_templates(self):
        """Initialize Bengali question templates for different legal scenarios"""
        self.question_templates = {
            'factual': [
                "{legal_concept} কী?",
                "{legal_concept} সম্পর্কে বিস্তারিত বলুন।",
                "{legal_concept} এর সংজ্ঞা কী?",
                "{legal_concept} কিভাবে কাজ করে?",
                "বাংলাদেশের আইনে {legal_concept} কী অর্থ রাখে?",
                "{legal_concept} এর আইনি ভিত্তি কী?",
                "{legal_concept} বলতে কী বোঝায়?",
                "{legal_concept} সম্পর্কে আইনি বিধান কী?",
            ],
            
            'procedural': [
                "{legal_process} এর পদ্ধতি কী?",
                "কিভাবে {legal_process} করতে হয়?",
                "{legal_process} এর জন্য কি কি কাগজপত্র লাগে?",
                "{legal_process} এর সময়সীমা কত?",
                "{legal_process} এর খরচ কত?",
                "{legal_process} এর প্রক্রিয়া ধাপে ধাপে বর্ণনা করুন।",
                "{legal_process} করার জন্য কোথায় যেতে হবে?",
                "{legal_process} সম্পর্কিত আইনি পদক্ষেপ কী?",
            ],
            
            'rights_duties': [
                "{situation} এ আমার অধিকার কী?",
                "{situation} এ আমার দায়িত্ব কী?",
                "{situation} এ আমি কী করতে পারি?",
                "{situation} এ আমার আইনি সুরক্ষা কী?",
                "{situation} এ আমি কোন আইনি সহায়তা পেতে পারি?",
                "{situation} এ আমার কর্তব্য কী?",
                "{situation} এ আমি কী ধরনের সহায়তা পেতে পারি?",
            ],
            
            'consequences': [
                "{violation} এর শাস্তি কী?",
                "{violation} করলে কী হবে?",
                "{violation} এর আইনি পরিণতি কী?",
                "{violation} এর জন্য কী ধরনের মামলা হতে পারে?",
                "{violation} এর জরিমানা কত?",
                "{violation} সম্পর্কিত আইনি ব্যবস্থা কী?",
            ],
            
            'comparative': [
                "{concept1} এবং {concept2} এর মধ্যে পার্থক্য কী?",
                "{concept1} কি {concept2} এর চেয়ে ভালো?",
                "কোন ক্ষেত্রে {concept1} প্রযোজ্য আর কোন ক্ষেত্রে {concept2}?",
                "{concept1} বনাম {concept2} - কোনটি বেছে নেব?",
            ],
            
            'case_based': [
                "যদি {scenario} হয়, তাহলে আইনি সমাধান কী?",
                "{scenario} এর ক্ষেত্রে আদালত কী রায় দিতে পারে?",
                "{scenario} এর মতো পরিস্থিতিতে কী করণীয়?",
                "{scenario} এর জন্য কোন আইনি পদক্ষেপ নেওয়া যায়?",
            ]
        }
        
    def initialize_legal_domains(self):
        """Initialize legal domain-specific concepts and terminology"""
        self.legal_domains = {
            'family_law': {
                'concepts': [
                    'তালাক', 'খোরপোশ', 'দেনমোহর', 'বিবাহ', 'বিবাহবিচ্ছেদ',
                    'সন্তানের হেফাজত', 'উত্তরাধিকার', 'পারিবারিক সম্পত্তি', 'মিরাস'
                ],
                'processes': [
                    'তালাক দেওয়া', 'খোরপোশ দাবি করা', 'বিবাহ নিবন্ধন',
                    'হেফাজতের মামলা করা', 'উত্তরাধিকার বণ্টন'
                ],
                'scenarios': [
                    'স্বামী-স্ত্রীর বিবাদ', 'সন্তানের হেফাজত নিয়ে বিরোধ',
                    'সম্পত্তি বণ্টন নিয়ে সমস্যা', 'পুনরায় বিবাহের আইনি জটিলতা'
                ]
            },
            
            'property_law': {
                'concepts': [
                    'জমির দলিল', 'মালিকানা', 'রেজিস্ট্রেশন', 'সম্পত্তি হস্তান্তর',
                    'ইজারা', 'বন্ধক', 'দখল', 'অধিগ্রহণ'
                ],
                'processes': [
                    'জমি কেনা', 'দলিল রেজিস্ট্রেশন করা', 'সম্পত্তি বিক্রি করা',
                    'বন্ধক দেওয়া', 'ইজারা দেওয়া'
                ],
                'scenarios': [
                    'জমি দখল নিয়ে বিরোধ', 'দলিল জাল হওয়ার সন্দেহ',
                    'সরকারি জমি অধিগ্রহণ', 'সম্পত্তির উপর একাধিক দাবি'
                ]
            },
            
            'constitutional_law': {
                'concepts': [
                    'মৌলিক অধিকার', 'নাগরিক স্বাধীনতা', 'সমতার অধিকার',
                    'ধর্মীয় স্বাধীনতা', 'বাক্‌স্বাধীনতা', 'সংবিধানিক প্রতিকার'
                ],
                'processes': [
                    'মৌলিক অধিকার রক্ষা করা', 'সাংবিধানিক আদালতে মামলা করা',
                    'জনস্বার্থে মামলা দায়ের করা'
                ],
                'scenarios': [
                    'মৌলিক অধিকার লঙ্ঘন', 'সরকারি নিপীড়ন',
                    'ধর্মীয় স্বাধীনতায় হস্তক্ষেপ', 'বাক্‌স্বাধীনতায় বাধা'
                ]
            },
            
            'procedural_law': {
                'concepts': [
                    'মামলা দায়ের', 'আদালতের প্রক্রিয়া', 'আপিল', 'জামিন',
                    'সাক্ষী', 'প্রমাণ', 'রায়', 'কার্যকর করা'
                ],
                'processes': [
                    'মামলা দায়ের করা', 'জামিনের জন্য আবেদন করা',
                    'আপিল করা', 'রায় কার্যকর করা'
                ],
                'scenarios': [
                    'মামলার বিলম্ব', 'আদালতে হাজির না হওয়া',
                    'সাক্ষীর অনুপস্থিতি', 'প্রমাণ গোপন করা'
                ]
            }
        }
        
    def generate_qa_dataset(self, num_pairs: int = 10000, 
                           quality_threshold: float = 0.8) -> List[Dict]:
        """
        Generate comprehensive Bengali legal Q&A dataset
        
        Args:
            num_pairs: Number of Q&A pairs to generate
            quality_threshold: Minimum quality score for inclusion
            
        Returns:
            List of high-quality Q&A pairs
        """
        self.logger.info(f"Starting Q&A generation for {num_pairs} pairs...")
        
        generated_pairs = []
        generation_attempts = 0
        max_attempts = num_pairs * 2  # Allow more attempts for quality filtering
        
        # Get all available documents for context
        all_documents = self._get_all_documents()
        
        while len(generated_pairs) < num_pairs and generation_attempts < max_attempts:
            try:
                # Select random document and domain
                if not all_documents:
                    self.logger.error("No documents available for Q&A generation")
                    break
                    
                document = random.choice(all_documents)
                domain = self._identify_document_domain(document)
                
                # Generate question-answer pair
                qa_pair = self._generate_single_qa_pair(document, domain)
                
                if qa_pair:
                    # Quality assessment
                    quality_score = self._assess_qa_quality(qa_pair)
                    
                    if quality_score >= quality_threshold:
                        qa_pair['quality_score'] = quality_score
                        qa_pair['generation_metadata'] = {
                            'domain': domain,
                            'source_document': document.get('source', 'unknown'),
                            'generation_time': datetime.now().isoformat(),
                            'attempt_number': generation_attempts + 1
                        }
                        
                        generated_pairs.append(qa_pair)
                        self.quality_stats['accepted'] += 1
                        
                        if len(generated_pairs) % 100 == 0:
                            self.logger.info(f"Generated {len(generated_pairs)} Q&A pairs")
                    else:
                        self.quality_stats['rejected_quality'] += 1
                        
                generation_attempts += 1
                
            except Exception as e:
                self.logger.error(f"Error generating Q&A pair: {e}")
                self.quality_stats['errors'] += 1
                generation_attempts += 1
                
        self.logger.info(f"Q&A generation completed. Generated {len(generated_pairs)} pairs from {generation_attempts} attempts")
        self.logger.info(f"Quality stats: {dict(self.quality_stats)}")
        
        self.generated_qa_pairs = generated_pairs
        return generated_pairs
    
    def _get_all_documents(self) -> List[Dict]:
        """Retrieve all processed documents from vector store"""
        try:
            # Try to get documents from vector store metadata
            if hasattr(self.vector_store, 'get_all_documents'):
                return self.vector_store.get_all_documents()
            
            # Fallback: search with broad query to get diverse documents
            broad_queries = [
                'আইন', 'ধারা', 'অনুচ্ছেদ', 'বিধি', 'নিয়ম',
                'অধিকার', 'কর্তব্য', 'আদালত', 'বিচার'
            ]
            
            all_docs = []
            for query in broad_queries:
                results = self.vector_store.hybrid_search(
                    query=query, level='section', top_k=20
                )
                all_docs.extend(results)
            
            # Remove duplicates based on content
            unique_docs = []
            seen_content = set()
            
            for doc in all_docs:
                content_hash = hash(doc.get('content', '')[:200])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
            
            return unique_docs
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _identify_document_domain(self, document: Dict) -> str:
        """Identify the legal domain of a document"""
        content = document.get('content', '').lower()
        metadata = document.get('metadata', {})
        
        # Check metadata first
        if 'domain' in metadata:
            return metadata['domain']
        
        # Use content keywords to identify domain
        domain_scores = {}
        
        for domain, domain_info in self.legal_domains.items():
            score = 0
            for concept in domain_info.get('concepts', []):
                if concept.lower() in content:
                    score += 2
            for process in domain_info.get('processes', []):
                if process.lower() in content:
                    score += 1
                    
            domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return 'general'
    
    def _generate_single_qa_pair(self, document: Dict, domain: str) -> Optional[Dict]:
        """Generate a single Q&A pair from a document"""
        try:
            content = document.get('content', '')
            
            # Select question type and template
            question_type = random.choice(list(self.question_templates.keys()))
            template = random.choice(self.question_templates[question_type])
            
            # Generate question based on document content and domain
            question = self._create_question_from_template(
                template, content, domain, question_type
            )
            
            if not question:
                return None
            
            # Generate answer using RAG system
            answer_data = self.legal_rag.process_legal_query(question)
            
            if 'error' in answer_data:
                return None
            
            # Extract relevant context and format answer
            context = self._extract_relevant_context(answer_data)
            answer = self._format_answer(answer_data, question_type)
            
            # Create Q&A pair
            qa_pair = {
                'system': self._get_system_prompt(domain),
                'instruction': question,
                'context': context,
                'response': answer,
                'question_type': question_type,
                'domain': domain,
                'source_content': content[:500],  # First 500 chars for reference
                'citations': answer_data.get('citations', [])
            }
            
            return qa_pair
            
        except Exception as e:
            self.logger.error(f"Error generating single Q&A pair: {e}")
            return None
    
    def _create_question_from_template(self, template: str, content: str, 
                                     domain: str, question_type: str) -> Optional[str]:
        """Create a specific question from template using document content"""
        try:
            # Extract relevant entities from content
            entities = self.bengali_processor.extract_legal_entities(content)
            
            # Get domain-specific concepts
            domain_concepts = self.legal_domains.get(domain, {})
            
            # Fill template based on question type
            if question_type == 'factual':
                # Use legal concepts from content or domain
                legal_concepts = []
                
                # From extracted entities
                for entity_type, entity_list in entities.items():
                    if entity_type in ['laws', 'sections', 'legal_terms']:
                        if isinstance(entity_list, list) and entity_list:
                            legal_concepts.extend([
                                e['term'] if isinstance(e, dict) else str(e) 
                                for e in entity_list[:3]
                            ])
                
                # From domain concepts
                legal_concepts.extend(domain_concepts.get('concepts', [])[:3])
                
                if legal_concepts:
                    concept = random.choice(legal_concepts)
                    return template.format(legal_concept=concept)
                    
            elif question_type == 'procedural':
                processes = domain_concepts.get('processes', [])
                if processes:
                    process = random.choice(processes)
                    return template.format(legal_process=process)
                    
            elif question_type == 'rights_duties':
                scenarios = domain_concepts.get('scenarios', [])
                if scenarios:
                    situation = random.choice(scenarios)
                    return template.format(situation=situation)
                    
            elif question_type == 'consequences':
                # Create violation scenarios
                violations = [
                    'আইন লঙ্ঘন', 'নিয়ম ভাঙা', 'আদালতের আদেশ অমান্য',
                    'চুক্তি ভঙ্গ', 'কর্তব্য পালনে ব্যর্থতা'
                ]
                violation = random.choice(violations)
                return template.format(violation=violation)
                
            elif question_type == 'comparative':
                concepts = domain_concepts.get('concepts', [])
                if len(concepts) >= 2:
                    concept1, concept2 = random.sample(concepts, 2)
                    return template.format(concept1=concept1, concept2=concept2)
                    
            elif question_type == 'case_based':
                scenarios = domain_concepts.get('scenarios', [])
                if scenarios:
                    scenario = random.choice(scenarios)
                    return template.format(scenario=scenario)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating question from template: {e}")
            return None
    
    def _extract_relevant_context(self, answer_data: Dict) -> str:
        """Extract and format relevant context from RAG response"""
        try:
            retrieved_context = answer_data.get('retrieved_context', {})
            documents = retrieved_context.get('documents', [])
            
            context_parts = []
            
            for doc in documents[:3]:  # Top 3 most relevant documents
                content = doc.get('content', '')
                source = doc.get('source', 'Unknown')
                
                # Clean and truncate content
                clean_content = self.bengali_processor.preprocess_bengali_legal_text(content)
                if len(clean_content) > 300:
                    clean_content = clean_content[:300] + "..."
                
                context_parts.append(f"[{source}] {clean_content}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            self.logger.error(f"Error extracting context: {e}")
            return ""
    
    def _format_answer(self, answer_data: Dict, question_type: str) -> str:
        """Format the answer based on question type and retrieved information"""
        try:
            # Extract key information
            retrieved_context = answer_data.get('retrieved_context', {})
            documents = retrieved_context.get('documents', [])
            citations = answer_data.get('citations', [])
            
            if not documents:
                return "দুঃখিত, এই প্রশ্নের উত্তর দিতে পর্যাপ্ত তথ্য পাওয়া যায়নি।"
            
            # Build comprehensive answer
            answer_parts = []
            
            # Main answer content
            main_content = []
            for doc in documents[:2]:  # Use top 2 documents
                content = doc.get('content', '')
                if content:
                    clean_content = self.bengali_processor.preprocess_bengali_legal_text(content)
                    main_content.append(clean_content[:400])
            
            if main_content:
                answer_parts.append(" ".join(main_content))
            
            # Add citations if available
            if citations:
                citation_text = self._format_citations(citations)
                if citation_text:
                    answer_parts.append(f"\n\nআইনি তথ্যসূত্র:\n{citation_text}")
            
            # Add legal disclaimer
            answer_parts.append(
                "\n\nদ্রষ্টব্য: এই তথ্যগুলো শুধুমাত্র সাধারণ জ্ঞানের জন্য। "
                "নির্দিষ্ট আইনি পরামর্শের জন্য অভিজ্ঞ আইনজীবীর সাথে যোগাযোগ করুন।"
            )
            
            return "\n".join(answer_parts)
            
        except Exception as e:
            self.logger.error(f"Error formatting answer: {e}")
            return "দুঃখিত, উত্তর তৈরি করতে সমস্যা হয়েছে।"
    
    def _format_citations(self, citations: List[Dict]) -> str:
        """Format legal citations for Bengali text"""
        try:
            formatted_citations = []
            
            for citation in citations[:3]:  # Top 3 citations
                source = citation.get('source', '')
                section = citation.get('section', '')
                relevance = citation.get('relevance_score', 0)
                
                if source and relevance > 0.3:
                    citation_text = f"• {source}"
                    if section:
                        citation_text += f" - {section}"
                    formatted_citations.append(citation_text)
            
            return "\n".join(formatted_citations)
            
        except Exception as e:
            self.logger.error(f"Error formatting citations: {e}")
            return ""
    
    def _get_system_prompt(self, domain: str) -> str:
        """Get appropriate system prompt for the domain"""
        domain_prompts = {
            'family_law': "আপনি একজন পারিবারিক আইনের বিশেষজ্ঞ আইনজীবী। বাংলাদেশের মুসলিম পারিবারিক আইন, তালাক, খোরপোশ, এবং পারিবারিক বিষয়ে আপনার বিশেষত্ব রয়েছে।",
            'property_law': "আপনি একজন সম্পত্তি আইনের বিশেষজ্ঞ। জমি, বাড়ি, উত্তরাধিকার, এবং সম্পত্তি সংক্রান্ত আইনি বিষয়ে আপনার গভীর জ্ঞান রয়েছে।",
            'constitutional_law': "আপনি একজন সাংবিধানিক আইনের বিশেষজ্ঞ। বাংলাদেশের সংবিধান, মৌলিক অধিকার, এবং সাংবিধানিক বিষয়ে আপনার বিশেষত্ব রয়েছে।",
            'procedural_law': "আপনি একজন আদালতি প্রক্রিয়া ও পদ্ধতির বিশেষজ্ঞ। মামলা দায়ের, আদালতের নিয়ম, এবং আইনি প্রক্রিয়া সম্পর্কে আপনার গভীর জ্ঞান রয়েছে।",
            'general': "আপনি একজন দক্ষ বাংলাদেশী আইনজীবী। বাংলাদেশের আইন সম্পর্কে আপনার গভীর জ্ঞান রয়েছে এবং আপনি সঠিক আইনি পরামর্শ প্রদান করেন।"
        }
        
        return domain_prompts.get(domain, domain_prompts['general'])
    
    def _assess_qa_quality(self, qa_pair: Dict) -> float:
        """Assess the quality of a generated Q&A pair"""
        quality_score = 0.0
        total_checks = 0
        
        try:
            question = qa_pair.get('instruction', '')
            answer = qa_pair.get('response', '')
            context = qa_pair.get('context', '')
            
            # Question quality checks
            if question:
                total_checks += 1
                if len(question.split()) >= 3:  # Minimum word count
                    quality_score += 0.2
                    
                if '?' in question or any(qword in question for qword in ['কী', 'কি', 'কেন', 'কিভাবে', 'কোথায়', 'কখন']):
                    quality_score += 0.2
                    
                # Check for Bengali legal terminology
                legal_terms = ['আইন', 'ধারা', 'অধিকার', 'কর্তব্য', 'আদালত', 'বিচার']
                if any(term in question for term in legal_terms):
                    quality_score += 0.1
            
            # Answer quality checks
            if answer:
                total_checks += 1
                if len(answer.split()) >= 10:  # Minimum substantive answer
                    quality_score += 0.2
                    
                if not answer.startswith('দুঃখিত'):  # Not an error message
                    quality_score += 0.1
                    
                # Check for legal references
                if any(ref in answer for ref in ['ধারা', 'অনুচ্ছেদ', 'আইন', 'অধ্যাদেশ']):
                    quality_score += 0.1
            
            # Context quality checks
            if context:
                total_checks += 1
                if len(context.split()) >= 20:  # Substantial context
                    quality_score += 0.1
                    
                if '[' in context and ']' in context:  # Proper source attribution
                    quality_score += 0.1
            
            # Overall coherence check
            if question and answer:
                # Simple check: answer should be longer than question
                if len(answer.split()) > len(question.split()):
                    quality_score += 0.1
                    
                # Check for question-answer relevance (basic keyword matching)
                q_words = set(question.lower().split())
                a_words = set(answer.lower().split())
                overlap = len(q_words.intersection(a_words))
                if overlap >= 2:
                    quality_score += 0.1
            
            return min(quality_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error assessing Q&A quality: {e}")
            return 0.0
    
    def export_qa_dataset(self, filename: str, format: str = 'json') -> bool:
        """Export generated Q&A dataset to file"""
        try:
            if not self.generated_qa_pairs:
                self.logger.warning("No Q&A pairs to export")
                return False
            
            export_path = Path(filename)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'json':
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(self.generated_qa_pairs, f, ensure_ascii=False, indent=2)
                    
            elif format.lower() == 'jsonl':
                with open(export_path, 'w', encoding='utf-8') as f:
                    for qa_pair in self.generated_qa_pairs:
                        f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
            
            self.logger.info(f"Exported {len(self.generated_qa_pairs)} Q&A pairs to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting Q&A dataset: {e}")
            return False
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the generation process"""
        return {
            'total_generated': len(self.generated_qa_pairs),
            'quality_stats': dict(self.quality_stats),
            'domain_distribution': self._get_domain_distribution(),
            'question_type_distribution': self._get_question_type_distribution(),
            'average_quality_score': self._get_average_quality_score()
        }
    
    def _get_domain_distribution(self) -> Dict[str, int]:
        """Get distribution of Q&A pairs by domain"""
        domain_counts = defaultdict(int)
        for qa_pair in self.generated_qa_pairs:
            domain = qa_pair.get('domain', 'unknown')
            domain_counts[domain] += 1
        return dict(domain_counts)
    
    def _get_question_type_distribution(self) -> Dict[str, int]:
        """Get distribution of question types"""
        type_counts = defaultdict(int)
        for qa_pair in self.generated_qa_pairs:
            q_type = qa_pair.get('question_type', 'unknown')
            type_counts[q_type] += 1
        return dict(type_counts)
    
    def _get_average_quality_score(self) -> float:
        """Calculate average quality score"""
        if not self.generated_qa_pairs:
            return 0.0
        
        total_score = sum(qa_pair.get('quality_score', 0) for qa_pair in self.generated_qa_pairs)
        return total_score / len(self.generated_qa_pairs)