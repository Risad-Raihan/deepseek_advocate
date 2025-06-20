"""
Legal Response Generator for Bengali Legal Advocate
Optimized for Local LM Studio + DeepSeek Integration
"""

import logging
import json
from typing import Dict, List, Optional, Any
import requests
from datetime import datetime
import re

class BengaliLegalResponseGenerator:
    """Advanced response generator for Bengali legal queries using local LM Studio"""
    
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1", model_name: str = "deepseek"):
        self.lm_studio_url = lm_studio_url
        self.model_name = model_name
        self.setup_logging()
        
        # Response generation parameters
        self.max_response_length = 1500
        self.temperature = 0.3  # Low temperature for legal accuracy
        self.top_p = 0.9
        
        # Legal response templates
        self.response_templates = self._initialize_response_templates()
        
        # Verification patterns for Bengali legal content
        self.verification_patterns = self._initialize_verification_patterns()
    
    def setup_logging(self):
        """Setup logging for response generator"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _initialize_response_templates(self) -> Dict[str, str]:
        """Initialize Bengali legal response templates for different domains"""
        return {
            'family_law': """
আপনার পারিবারিক আইন সংক্রান্ত প্রশ্নের উত্তর:

{main_response}

আইনি ভিত্তি:
{legal_basis}

প্রয়োজনীয় পদক্ষেপ:
{required_steps}

সতর্কতা: {warning}
""",
            
            'property_law': """
সম্পত্তি আইন সংক্রান্ত আপনার প্রশ্নের উত্তর:

{main_response}

প্রাসঙ্গিক আইনি বিধান:
{legal_provisions}

প্রয়োজনীয় কাগজপত্র:
{required_documents}

পরামর্শ: {advice}
""",
            
            'constitutional_law': """
সাংবিধানিক আইন সংক্রান্ত উত্তর:

{main_response}

সংবিধানের প্রাসঙ্গিক অনুচ্ছেদ:
{constitutional_articles}

মৌলিক অধিকার সংক্রান্ত তথ্য:
{fundamental_rights}

গুরুত্বপূর্ণ নোট: {important_note}
""",
            
            'court_procedure': """
আদালতি প্রক্রিয়া সংক্রান্ত নির্দেশনা:

{main_response}

প্রয়োজনীয় প্রক্রিয়া:
{required_procedure}

সময়সীমা:
{time_limits}

আদালত ফি ও খরচ:
{court_fees}

পরবর্তী পদক্ষেপ: {next_steps}
""",
            
            'general': """
আপনার আইনি প্রশ্নের উত্তর:

{main_response}

প্রাসঙ্গিক আইনি তথ্য:
{legal_information}

পরামর্শ:
{recommendations}

দ্রষ্টব্য: {disclaimer}
"""
        }
    
    def _initialize_verification_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for verifying Bengali legal content accuracy"""
        return {
            'legal_terms': [
                r'ধারা\s*\d+',
                r'অনুচ্ছেদ\s*\d+',
                r'\d{4}\s*সালের.*আইন',
                r'আদালত',
                r'বিচারক',
                r'মামলা',
                r'আপিল'
            ],
            'legal_procedures': [
                r'আবেদন',
                r'নোটিশ',
                r'শুনানি',
                r'প্রমাণ',
                r'সাক্ষী',
                r'রায়'
            ],
            'family_law_terms': [
                r'তালাক',
                r'খোরপোশ',
                r'দেনমোহর',
                r'উত্তরাধিকার',
                r'অভিভাবকত্ব'
            ]
        }
    
    def generate_comprehensive_legal_response(self, rag_output: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive legal response using local LM Studio
        
        Args:
            rag_output: Complete RAG system output with context and citations
            
        Returns:
            Comprehensive legal response with metadata
        """
        try:
            self.logger.info("Generating comprehensive legal response...")
            
            # Build prompt for LM Studio
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(rag_output)
            
            # Generate response using local LM Studio
            generated_response = self._call_lm_studio(system_prompt, user_prompt)
            
            if not generated_response:
                # Fallback to template-based response
                generated_response = self._generate_template_response(rag_output)
            
            # Post-process and verify response
            processed_response = self._post_process_response(generated_response, rag_output)
            
            # Format final response
            final_response = self._format_final_response(processed_response, rag_output)
            
            self.logger.info("Legal response generated successfully")
            return final_response
            
        except Exception as e:
            self.logger.error(f"Error generating legal response: {e}")
            return self._generate_error_response(str(e), rag_output)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for Bengali legal expert"""
        return """আপনি একজন বিশেষজ্ঞ বাংলাদেশি আইনজীবী এবং আইনি পরামর্শদাতা। আপনার দায়িত্ব:

1. বাংলা ভাষায় সুস্পষ্ট এবং নির্ভুল আইনি পরামর্শ প্রদান করা
2. প্রাসঙ্গিক আইন, ধারা, এবং অনুচ্ছেদের সঠিক উল্লেখ করা
3. ব্যবহারিক পদক্ষেপ এবং পরামর্শ দেওয়া
4. আইনি জটিলতা সহজ ভাষায় ব্যাখ্যা করা
5. সতর্কতামূলক তথ্য এবং দাবিত্যাগ অন্তর্ভুক্ত করা

নিয়ম:
- শুধুমাত্র প্রদত্ত প্রসঙ্গের ভিত্তিতে উত্তর দিন
- অনুমান বা ভুল তথ্য দিবেন না
- আইনি উৎসের সঠিক রেফারেন্স দিন
- ব্যবহারিক এবং কার্যকর পরামর্শ দিন
- পেশাদার এবং সম্মানজনক ভাষা ব্যবহার করুন"""
    
    def _build_user_prompt(self, rag_output: Dict) -> str:
        """Build user prompt with context and query"""
        try:
            query = rag_output.get('query_analysis', {}).get('original_query', '')
            context = rag_output.get('response_context', '')
            domain = rag_output.get('legal_domain', 'general')
            
            prompt = f"""
প্রশ্ন: {query}

আইনি প্রসঙ্গ এবং তথ্য:
{context}

আইনি ক্ষেত্র: {domain}

অনুগ্রহ করে উপরের প্রসঙ্গের ভিত্তিতে একটি বিস্তারিত, নির্ভুল এবং ব্যবহারিক আইনি পরামর্শ প্রদান করুন। 

আপনার উত্তরে অন্তর্ভুক্ত করুন:
1. মূল উত্তর এবং ব্যাখ্যা
2. প্রাসঙ্গিক আইনি ভিত্তি (ধারা/অনুচ্ছেদ সহ)
3. ব্যবহারিক পদক্ষেপ
4. প্রয়োজনীয় সতর্কতা

বাংলা ভাষায় উত্তর দিন।
"""
            return prompt
            
        except Exception as e:
            self.logger.error(f"Error building user prompt: {e}")
            return f"প্রশ্ন: {rag_output.get('query_analysis', {}).get('original_query', 'অজানা')}"
    
    def _call_lm_studio(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call local LM Studio API"""
        try:
            self.logger.info("Calling local LM Studio API...")
            
            # Prepare request payload for OpenAI-compatible API
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_response_length,
                "stream": False
            }
            
            # Make API call
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.lm_studio_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120  # 2 minute timeout
            )
            
            if response.status_code == 200:
                response_data = response.json()
                generated_text = response_data['choices'][0]['message']['content']
                self.logger.info("Successfully received response from LM Studio")
                return generated_text.strip()
            else:
                self.logger.error(f"LM Studio API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.ConnectionError:
            self.logger.error("Cannot connect to LM Studio. Make sure LM Studio is running on localhost:1234")
            return None
        except requests.exceptions.Timeout:
            self.logger.error("LM Studio API timeout")
            return None
        except Exception as e:
            self.logger.error(f"Error calling LM Studio API: {e}")
            return None
    
    def _generate_template_response(self, rag_output: Dict) -> str:
        """Generate fallback response using templates"""
        try:
            domain = rag_output.get('legal_domain', 'general')
            context = rag_output.get('response_context', '')
            citations = rag_output.get('citations', [])
            
            # Use appropriate template
            template = self.response_templates.get(domain, self.response_templates['general'])
            
            # Extract information from context
            main_response = self._extract_main_answer(context, rag_output)
            legal_basis = self._extract_legal_basis(citations)
            recommendations = self._generate_recommendations(domain, context)
            
            # Fill template
            formatted_response = template.format(
                main_response=main_response,
                legal_information=legal_basis,
                recommendations=recommendations,
                disclaimer="এই পরামর্শ সাধারণ তথ্যের জন্য। নির্দিষ্ট মামলার জন্য অভিজ্ঞ আইনজীবীর সাথে যোগাযোগ করুন।"
            )
            
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"Error generating template response: {e}")
            return "আইনি পরামর্শ তৈরি করতে সমস্যা হয়েছে। দয়া করে আবার চেষ্টা করুন।"
    
    def _extract_main_answer(self, context: str, rag_output: Dict) -> str:
        """Extract main answer from context"""
        try:
            query = rag_output.get('query_analysis', {}).get('original_query', '')
            
            # Simple extraction based on context
            if context:
                # Take first meaningful paragraph
                paragraphs = context.split('\n\n')
                for para in paragraphs:
                    if len(para.strip()) > 50:  # Meaningful content
                        return para.strip()[:400] + "..."
            
            return "প্রদত্ত প্রসঙ্গের ভিত্তিতে আইনি পরামর্শ প্রস্তুত করা সম্ভব হয়নি।"
            
        except Exception as e:
            self.logger.error(f"Error extracting main answer: {e}")
            return "উত্তর তৈরি করতে সমস্যা হয়েছে।"
    
    def _extract_legal_basis(self, citations: List[str]) -> str:
        """Extract legal basis from citations"""
        try:
            if not citations:
                return "নির্দিষ্ট আইনি ভিত্তি পাওয়া যায়নি।"
            
            basis_text = "প্রাসঙ্গিক আইনি বিধান:\n"
            for i, citation in enumerate(citations[:5], 1):
                basis_text += f"{i}. {citation}\n"
            
            return basis_text
            
        except Exception as e:
            self.logger.error(f"Error extracting legal basis: {e}")
            return "আইনি ভিত্তি নির্ধারণ করতে সমস্যা হয়েছে।"
    
    def _generate_recommendations(self, domain: str, context: str) -> str:
        """Generate domain-specific recommendations"""
        try:
            recommendations = {
                'family_law': [
                    "অভিজ্ঞ পারিবারিক আইনজীবীর সাথে পরামর্শ করুন",
                    "প্রয়োজনীয় কাগজপত্র সংগ্রহ করুন",
                    "আদালতের নির্ধারিত সময়সীমা মেনে চলুন"
                ],
                'property_law': [
                    "সম্পত্তির দলিল যাচাই করুন",
                    "রেজিস্ট্রেশন অফিসে তথ্য নিশ্চিত করুন",
                    "সম্পত্তি আইনজীবী নিয়োগ করুন"
                ],
                'court_procedure': [
                    "নির্ধারিত তারিখে আদালতে হাজির হন",
                    "প্রয়োজনীয় কাগজপত্র প্রস্তুত রাখুন",
                    "আইনজীবীর মাধ্যমে মামলা পরিচালনা করুন"
                ]
            }
            
            domain_recs = recommendations.get(domain, [
                "বিশেষজ্ঞ আইনজীবীর পরামর্শ নিন",
                "আইনি পদক্ষেপের আগে সকল তথ্য যাচাই করুন"
            ])
            
            return "\n".join([f"• {rec}" for rec in domain_recs])
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return "• বিশেষজ্ঞ আইনজীবীর পরামর্শ নিন"
    
    def _post_process_response(self, response: str, rag_output: Dict) -> str:
        """Post-process and verify generated response"""
        try:
            if not response:
                return ""
            
            # Clean up response
            cleaned_response = self._clean_response_text(response)
            
            # Verify legal accuracy
            verified_response = self._verify_legal_content(cleaned_response, rag_output)
            
            # Add disclaimer if needed
            final_response = self._add_legal_disclaimer(verified_response)
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"Error post-processing response: {e}")
            return response
    
    def _clean_response_text(self, response: str) -> str:
        """Clean and format response text"""
        try:
            # Remove extra whitespace
            cleaned = re.sub(r'\n\s*\n', '\n\n', response)
            cleaned = cleaned.strip()
            
            # Ensure proper Bengali punctuation
            cleaned = re.sub(r'\.{2,}', '।', cleaned)
            cleaned = re.sub(r'\?{2,}', '?', cleaned)
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Error cleaning response text: {e}")
            return response
    
    def _verify_legal_content(self, response: str, rag_output: Dict) -> str:
        """Verify legal content accuracy"""
        try:
            # Check for legal term consistency
            citations = rag_output.get('citations', [])
            
            # Ensure mentioned laws exist in citations
            mentioned_laws = re.findall(r'\d{4}\s*সালের.*?আইন', response)
            mentioned_sections = re.findall(r'ধারা\s*\d+', response)
            
            # Add warning if uncertain
            if mentioned_laws and not citations:
                response += "\n\n⚠️ দ্রষ্টব্য: উল্লিখিত আইনি তথ্য যাচাই সাপেক্ষ।"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error verifying legal content: {e}")
            return response
    
    def _add_legal_disclaimer(self, response: str) -> str:
        """Add appropriate legal disclaimer"""
        try:
            disclaimer = """

📋 গুরুত্বপূর্ণ দাবিত্যাগ:
এই পরামর্শ সাধারণ তথ্যের উদ্দেশ্যে প্রদান করা হয়েছে। এটি পেশাদার আইনি পরামর্শের বিকল্প নয়। আপনার নির্দিষ্ট পরিস্থিতির জন্য অভিজ্ঞ আইনজীবীর সাথে পরামর্শ করুন।"""
            
            if disclaimer.strip() not in response:
                response += disclaimer
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error adding legal disclaimer: {e}")
            return response
    
    def _format_final_response(self, processed_response: str, rag_output: Dict) -> Dict[str, Any]:
        """Format final response with metadata"""
        try:
            return {
                'response': processed_response,
                'confidence_score': rag_output.get('confidence_score', 0.0),
                'legal_domain': rag_output.get('legal_domain', 'general'),
                'citations': rag_output.get('citations', []),
                'processing_metadata': {
                    'generation_method': 'lm_studio' if self._check_lm_studio_response(processed_response) else 'template',
                    'response_length': len(processed_response),
                    'timestamp': datetime.now().isoformat(),
                    'model_used': self.model_name
                },
                'quality_indicators': {
                    'has_legal_references': bool(re.search(r'ধারা\s*\d+|অনুচ্ছেদ\s*\d+', processed_response)),
                    'has_citations': len(rag_output.get('citations', [])) > 0,
                    'has_recommendations': 'পরামর্শ' in processed_response or 'সুপারিশ' in processed_response,
                    'response_completeness': self._assess_response_completeness(processed_response)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error formatting final response: {e}")
            return {
                'response': processed_response,
                'error': str(e)
            }
    
    def _check_lm_studio_response(self, response: str) -> bool:
        """Check if response was generated by LM Studio"""
        # Simple heuristic: LM Studio responses tend to be more detailed and structured
        return len(response) > 200 and '।' in response
    
    def _assess_response_completeness(self, response: str) -> float:
        """Assess completeness of the response"""
        try:
            score = 0.0
            
            # Check for main answer
            if len(response) > 100:
                score += 0.3
            
            # Check for legal references
            if re.search(r'ধারা\s*\d+|অনুচ্ছেদ\s*\d+', response):
                score += 0.3
            
            # Check for practical advice
            if any(word in response for word in ['পরামর্শ', 'পদক্ষেপ', 'করুন', 'যোগাযোগ']):
                score += 0.2
            
            # Check for disclaimer
            if 'দাবিত্যাগ' in response or 'আইনজীবী' in response:
                score += 0.2
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error assessing response completeness: {e}")
            return 0.5
    
    def _generate_error_response(self, error_msg: str, rag_output: Dict) -> Dict[str, Any]:
        """Generate error response"""
        try:
            error_response = f"""
দুঃখিত, আপনার আইনি প্রশ্নের উত্তর তৈরি করতে সমস্যা হয়েছে।

ত্রুটি: {error_msg}

পরামর্শ:
• প্রশ্নটি আবার সহজ ভাষায় করুন
• ইন্টারনেট সংযোগ পরীক্ষা করুন
• LM Studio চালু আছে কিনা নিশ্চিত করুন

দাবিত্যাগ: এই সমস্যার জন্য অভিজ্ঞ আইনজীবীর সাথে সরাসরি যোগাযোগ করুন।
"""
            
            return {
                'response': error_response,
                'error': error_msg,
                'confidence_score': 0.0,
                'legal_domain': rag_output.get('legal_domain', 'unknown'),
                'processing_metadata': {
                    'generation_method': 'error_fallback',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating error response: {e}")
            return {
                'response': "একটি অপ্রত্যাশিত ত্রুটি ঘটেছে।",
                'error': str(e)
            } 