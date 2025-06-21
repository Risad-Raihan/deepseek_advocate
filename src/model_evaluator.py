"""
Model Evaluator - Phase 3
Comprehensive evaluation system for Bengali legal fine-tuned models
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import re

# Import evaluation libraries
try:
    import evaluate
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from transformers import pipeline
except ImportError as e:
    print(f"Warning: Some evaluation libraries not installed: {e}")

class LegalModelEvaluator:
    """Comprehensive evaluation system for Bengali legal models"""
    
    def __init__(self, config, bengali_processor=None):
        """
        Initialize model evaluator
        
        Args:
            config: Configuration object
            bengali_processor: Bengali text processor for language-specific evaluation
        """
        self.config = config
        self.bengali_processor = bengali_processor
        
        self.setup_logging()
        self.initialize_metrics()
        
        # Evaluation results storage
        self.evaluation_results = {}
        self.benchmark_results = {}
        
        # Bengali legal terminology for evaluation
        self.legal_terms = self._load_legal_terms()
        self.citation_patterns = self._load_citation_patterns()
        
    def setup_logging(self):
        """Setup logging for model evaluator"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def initialize_metrics(self):
        """Initialize evaluation metrics"""
        try:
            # Load standard metrics
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=False
            )
            
            # Initialize NLTK for BLEU
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            self.smoothing_function = SmoothingFunction().method1
            
            self.logger.info("Evaluation metrics initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing metrics: {e}")
    
    def _load_legal_terms(self) -> List[str]:
        """Load Bengali legal terminology for evaluation"""
        return [
            # Basic legal terms
            'আইন', 'ধারা', 'অনুচ্ছেদ', 'বিধি', 'নিয়ম', 'প্রবিধান',
            'আদালত', 'বিচার', 'মামলা', 'রায়', 'আদেশ', 'ডিক্রি',
            'অধিকার', 'কর্তব্য', 'দায়িত্ব', 'ক্ষমতা', 'স্বাধীনতা',
            
            # Constitutional terms
            'সংবিধান', 'মৌলিক অধিকার', 'নাগরিক', 'রাষ্ট্র', 'সরকার',
            'জাতীয় সংসদ', 'প্রধানমন্ত্রী', 'রাষ্ট্রপতি', 'বিচার বিভাগ',
            
            # Family law terms
            'তালাক', 'বিবাহ', 'খোরপোশ', 'দেনমোহর', 'পারিবারিক',
            'উত্তরাধিকার', 'মিরাস', 'হেফাজত', 'বিবাহবিচ্ছেদ',
            
            # Property law terms
            'সম্পত্তি', 'জমি', 'দলিল', 'মালিকানা', 'রেজিস্ট্রেশন',
            'ইজারা', 'বন্ধক', 'দখল', 'অধিগ্রহণ', 'হস্তান্তর',
            
            # Procedural terms
            'আপিল', 'জামিন', 'সাক্ষী', 'প্রমাণ', 'নোটিশ', 'সমন',
            'ওয়ারেন্ট', 'হলফনামা', 'দরখাস্ত', 'আবেদন'
        ]
    
    def _load_citation_patterns(self) -> Dict[str, str]:
        """Load regex patterns for legal citations"""
        return {
            'section': r'ধারা\s*(\d+(?:\([ক-৯]+\))?)',
            'article': r'অনুচ্ছেদ\s*(\d+(?:\([ক-৯]+\))?)',
            'law_year': r'(\d{4})\s*সালের\s*(.+?)\s*আইন',
            'ordinance': r'(\d{4})\s*সালের\s*(.+?)\s*অধ্যাদেশ',
            'case_reference': r'([A-Za-z\s]+)\s*বনাম\s*([A-Za-z\s]+)',
            'court': r'(সুপ্রিম কোর্ট|হাইকোর্ট|জেলা জজ কোর্ট|পারিবারিক আদালত)'
        }
    
    def evaluate_model_comprehensive(self, model, tokenizer, test_dataset, 
                                   reference_answers: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with multiple metrics
        
        Args:
            model: Fine-tuned model to evaluate
            tokenizer: Model tokenizer
            test_dataset: Test dataset
            reference_answers: Optional reference answers for comparison
            
        Returns:
            Complete evaluation results
        """
        self.logger.info("Starting comprehensive model evaluation...")
        
        evaluation_results = {
            'evaluation_date': datetime.now().isoformat(),
            'model_info': self._get_model_info(model),
            'dataset_info': self._get_dataset_info(test_dataset),
            'metrics': {}
        }
        
        try:
            # Generate predictions
            predictions = self._generate_predictions(model, tokenizer, test_dataset)
            
            if reference_answers is None:
                reference_answers = [item.get('response', '') for item in test_dataset]
            
            # Standard NLP metrics
            evaluation_results['metrics']['nlp_metrics'] = self._evaluate_nlp_metrics(
                predictions, reference_answers
            )
            
            # Bengali language-specific metrics
            evaluation_results['metrics']['bengali_metrics'] = self._evaluate_bengali_metrics(
                predictions, reference_answers
            )
            
            # Legal domain-specific metrics
            evaluation_results['metrics']['legal_metrics'] = self._evaluate_legal_metrics(
                predictions, reference_answers, test_dataset
            )
            
            # Citation accuracy
            evaluation_results['metrics']['citation_metrics'] = self._evaluate_citation_accuracy(
                predictions, reference_answers
            )
            
            # Semantic similarity
            evaluation_results['metrics']['semantic_metrics'] = self._evaluate_semantic_similarity(
                predictions, reference_answers
            )
            
            # Overall quality assessment
            evaluation_results['metrics']['quality_metrics'] = self._evaluate_overall_quality(
                predictions, reference_answers, test_dataset
            )
            
            # Calculate composite scores
            evaluation_results['composite_scores'] = self._calculate_composite_scores(
                evaluation_results['metrics']
            )
            
            self.evaluation_results = evaluation_results
            self.logger.info("Comprehensive evaluation completed successfully")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error during comprehensive evaluation: {e}")
            raise
    
    def _generate_predictions(self, model, tokenizer, test_dataset) -> List[str]:
        """Generate model predictions for test dataset"""
        self.logger.info(f"Generating predictions for {len(test_dataset)} test samples...")
        
        predictions = []
        
        for i, item in enumerate(test_dataset):
            try:
                # Format input prompt
                prompt = self._format_evaluation_prompt(item)
                
                # Tokenize input
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.DATA_CONFIG.get("max_seq_length", 2048) - 512
                )
                
                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                
                # Decode prediction
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = full_response[len(prompt):].strip()
                
                predictions.append(prediction)
                
                if (i + 1) % 50 == 0:
                    self.logger.info(f"Generated {i + 1}/{len(test_dataset)} predictions")
                    
            except Exception as e:
                self.logger.error(f"Error generating prediction for item {i}: {e}")
                predictions.append("দুঃখিত, উত্তর তৈরি করতে সমস্যা হয়েছে।")
        
        return predictions
    
    def _format_evaluation_prompt(self, item: Dict) -> str:
        """Format evaluation prompt from test item"""
        template = self.config.DATA_CONFIG.get("inference_template", """### System:
{system}

### Instruction:
{instruction}

### Context:
{context}

### Response:
""")
        
        return template.format(
            system=item.get('system', ''),
            instruction=item.get('instruction', ''),
            context=item.get('context', '')
        )
    
    def _evaluate_nlp_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate standard NLP metrics"""
        self.logger.info("Evaluating NLP metrics...")
        
        metrics = {}
        
        try:
            # ROUGE scores
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for pred, ref in zip(predictions, references):
                rouge_result = self.rouge_scorer.score(ref, pred)
                rouge_scores['rouge1'].append(rouge_result['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(rouge_result['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(rouge_result['rougeL'].fmeasure)
            
            metrics['rouge1'] = np.mean(rouge_scores['rouge1'])
            metrics['rouge2'] = np.mean(rouge_scores['rouge2'])
            metrics['rougeL'] = np.mean(rouge_scores['rougeL'])
            
            # BLEU scores
            bleu_scores = []
            for pred, ref in zip(predictions, references):
                pred_tokens = nltk.word_tokenize(pred.lower())
                ref_tokens = [nltk.word_tokenize(ref.lower())]
                
                if len(pred_tokens) > 0 and len(ref_tokens[0]) > 0:
                    bleu = sentence_bleu(
                        ref_tokens, pred_tokens, 
                        smoothing_function=self.smoothing_function
                    )
                    bleu_scores.append(bleu)
            
            metrics['bleu'] = np.mean(bleu_scores) if bleu_scores else 0.0
            
            # Length statistics
            pred_lengths = [len(pred.split()) for pred in predictions]
            ref_lengths = [len(ref.split()) for ref in references]
            
            metrics['avg_prediction_length'] = np.mean(pred_lengths)
            metrics['avg_reference_length'] = np.mean(ref_lengths)
            metrics['length_ratio'] = np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error evaluating NLP metrics: {e}")
            metrics = {key: 0.0 for key in ['rouge1', 'rouge2', 'rougeL', 'bleu']}
        
        return metrics
    
    def _evaluate_bengali_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate Bengali language-specific metrics"""
        self.logger.info("Evaluating Bengali language metrics...")
        
        metrics = {}
        
        try:
            # Bengali character ratio
            bengali_ratios = []
            for pred in predictions:
                bengali_chars = sum(1 for char in pred if '\u0980' <= char <= '\u09FF')
                total_chars = len([char for char in pred if char.isalnum()])
                ratio = bengali_chars / total_chars if total_chars > 0 else 0
                bengali_ratios.append(ratio)
            
            metrics['bengali_character_ratio'] = np.mean(bengali_ratios)
            
            # Bengali fluency (basic check)
            fluency_scores = []
            for pred in predictions:
                if self.bengali_processor:
                    # Use processor to check fluency
                    processed = self.bengali_processor.preprocess_bengali_legal_text(pred)
                    fluency = len(processed) / len(pred) if len(pred) > 0 else 0
                else:
                    # Basic fluency check
                    words = pred.split()
                    valid_words = sum(1 for word in words if any('\u0980' <= char <= '\u09FF' for char in word))
                    fluency = valid_words / len(words) if len(words) > 0 else 0
                
                fluency_scores.append(fluency)
            
            metrics['bengali_fluency'] = np.mean(fluency_scores)
            
            # Unicode normalization check
            normalized_count = 0
            for pred in predictions:
                try:
                    normalized = pred.encode('utf-8').decode('utf-8')
                    if normalized == pred:
                        normalized_count += 1
                except:
                    pass
            
            metrics['unicode_normalization'] = normalized_count / len(predictions) if predictions else 0
            
        except Exception as e:
            self.logger.error(f"Error evaluating Bengali metrics: {e}")
            metrics = {'bengali_character_ratio': 0.0, 'bengali_fluency': 0.0, 'unicode_normalization': 0.0}
        
        return metrics
    
    def _evaluate_legal_metrics(self, predictions: List[str], references: List[str], 
                               test_dataset: List[Dict]) -> Dict[str, float]:
        """Evaluate legal domain-specific metrics"""
        self.logger.info("Evaluating legal domain metrics...")
        
        metrics = {}
        
        try:
            # Legal terminology usage
            legal_term_scores = []
            for pred in predictions:
                pred_lower = pred.lower()
                legal_terms_found = sum(1 for term in self.legal_terms if term in pred_lower)
                score = legal_terms_found / len(self.legal_terms)
                legal_term_scores.append(score)
            
            metrics['legal_terminology_usage'] = np.mean(legal_term_scores)
            
            # Domain-specific accuracy
            domain_accuracies = defaultdict(list)
            for i, (pred, ref, item) in enumerate(zip(predictions, references, test_dataset)):
                domain = item.get('domain', 'general')
                
                # Simple accuracy based on keyword overlap
                pred_words = set(pred.lower().split())
                ref_words = set(ref.lower().split())
                
                if ref_words:
                    overlap = len(pred_words.intersection(ref_words))
                    accuracy = overlap / len(ref_words)
                    domain_accuracies[domain].append(accuracy)
            
            # Calculate per-domain accuracy
            for domain, accuracies in domain_accuracies.items():
                metrics[f'{domain}_accuracy'] = np.mean(accuracies)
            
            metrics['overall_domain_accuracy'] = np.mean([
                score for scores in domain_accuracies.values() 
                for score in scores
            ]) if domain_accuracies else 0.0
            
            # Legal reasoning indicators
            reasoning_indicators = [
                'কারণ', 'যেহেতু', 'সুতরাং', 'অতএব', 'তাই', 'সেইজন্য',
                'অনুযায়ী', 'মতে', 'ভিত্তিতে', 'প্রযোজ্য'
            ]
            
            reasoning_scores = []
            for pred in predictions:
                pred_lower = pred.lower()
                reasoning_found = sum(1 for indicator in reasoning_indicators if indicator in pred_lower)
                score = min(reasoning_found / 3, 1.0)  # Normalize to max 1.0
                reasoning_scores.append(score)
            
            metrics['legal_reasoning_indicators'] = np.mean(reasoning_scores)
            
        except Exception as e:
            self.logger.error(f"Error evaluating legal metrics: {e}")
            metrics = {'legal_terminology_usage': 0.0, 'overall_domain_accuracy': 0.0}
        
        return metrics
    
    def _evaluate_citation_accuracy(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate citation accuracy"""
        self.logger.info("Evaluating citation accuracy...")
        
        metrics = {}
        
        try:
            citation_scores = []
            
            for pred, ref in zip(predictions, references):
                # Extract citations from prediction and reference
                pred_citations = self._extract_citations(pred)
                ref_citations = self._extract_citations(ref)
                
                if not ref_citations:
                    # No citations in reference, perfect score if no citations in prediction
                    score = 1.0 if not pred_citations else 0.8
                else:
                    # Calculate citation overlap
                    if pred_citations:
                        correct_citations = len(set(pred_citations).intersection(set(ref_citations)))
                        precision = correct_citations / len(pred_citations)
                        recall = correct_citations / len(ref_citations)
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        score = f1
                    else:
                        score = 0.0
                
                citation_scores.append(score)
            
            metrics['citation_accuracy'] = np.mean(citation_scores)
            
            # Citation format accuracy
            format_scores = []
            for pred in predictions:
                citations = self._extract_citations(pred)
                if citations:
                    # Check if citations follow proper format
                    properly_formatted = sum(1 for citation in citations if self._is_properly_formatted_citation(citation))
                    score = properly_formatted / len(citations)
                else:
                    score = 1.0  # No citations, assume perfect format
                
                format_scores.append(score)
            
            metrics['citation_format_accuracy'] = np.mean(format_scores)
            
        except Exception as e:
            self.logger.error(f"Error evaluating citation accuracy: {e}")
            metrics = {'citation_accuracy': 0.0, 'citation_format_accuracy': 0.0}
        
        return metrics
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract legal citations from text"""
        citations = []
        
        for pattern_name, pattern in self.citation_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    citations.append(' '.join(match))
                else:
                    citations.append(match)
        
        return citations
    
    def _is_properly_formatted_citation(self, citation: str) -> bool:
        """Check if citation follows proper Bengali legal format"""
        # Basic format checks
        if re.search(r'ধারা\s*\d+', citation):
            return True
        if re.search(r'অনুচ্ছেদ\s*\d+', citation):
            return True
        if re.search(r'\d{4}\s*সালের.*আইন', citation):
            return True
        if re.search(r'\d{4}\s*সালের.*অধ্যাদেশ', citation):
            return True
        
        return False
    
    def _evaluate_semantic_similarity(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate semantic similarity using embeddings"""
        self.logger.info("Evaluating semantic similarity...")
        
        metrics = {}
        
        try:
            # Try to use BERTScore if available
            try:
                P, R, F1 = bert_score(predictions, references, lang='bn', verbose=False)
                metrics['bert_score_precision'] = float(P.mean())
                metrics['bert_score_recall'] = float(R.mean())
                metrics['bert_score_f1'] = float(F1.mean())
            except Exception as e:
                self.logger.warning(f"BERTScore not available: {e}")
                metrics['bert_score_f1'] = 0.0
            
            # Simple word embedding similarity (if available)
            try:
                from sentence_transformers import SentenceTransformer
                
                # Use multilingual model
                model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                
                pred_embeddings = model.encode(predictions)
                ref_embeddings = model.encode(references)
                
                # Calculate cosine similarity
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = []
                
                for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                    sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
                    similarities.append(sim)
                
                metrics['semantic_similarity'] = np.mean(similarities)
                
            except ImportError:
                self.logger.warning("Sentence transformers not available for semantic similarity")
                metrics['semantic_similarity'] = 0.0
            
        except Exception as e:
            self.logger.error(f"Error evaluating semantic similarity: {e}")
            metrics = {'bert_score_f1': 0.0, 'semantic_similarity': 0.0}
        
        return metrics
    
    def _evaluate_overall_quality(self, predictions: List[str], references: List[str], 
                                 test_dataset: List[Dict]) -> Dict[str, float]:
        """Evaluate overall quality metrics"""
        self.logger.info("Evaluating overall quality...")
        
        metrics = {}
        
        try:
            # Completeness - are responses complete?
            completeness_scores = []
            for pred, ref in zip(predictions, references):
                # Simple completeness check based on length and content
                pred_words = len(pred.split())
                ref_words = len(ref.split())
                
                if pred_words == 0:
                    score = 0.0
                elif pred.strip().startswith('দুঃখিত'):
                    score = 0.2  # Error response
                else:
                    # Completeness based on relative length and presence of key information
                    length_score = min(pred_words / max(ref_words, 50), 1.0)  # Normalize to expected length
                    score = length_score
                
                completeness_scores.append(score)
            
            metrics['completeness'] = np.mean(completeness_scores)
            
            # Coherence - basic coherence check
            coherence_scores = []
            for pred in predictions:
                sentences = pred.split('।')  # Bengali sentence delimiter
                
                if len(sentences) <= 1:
                    score = 0.5  # Single sentence, moderate coherence
                else:
                    # Check for proper sentence structure
                    proper_sentences = sum(1 for sent in sentences if len(sent.strip().split()) >= 3)
                    score = proper_sentences / len(sentences)
                
                coherence_scores.append(score)
            
            metrics['coherence'] = np.mean(coherence_scores)
            
            # Relevance - how relevant are responses to questions?
            relevance_scores = []
            for pred, item in zip(predictions, test_dataset):
                instruction = item.get('instruction', '')
                
                # Simple keyword-based relevance
                instruction_words = set(instruction.lower().split())
                pred_words = set(pred.lower().split())
                
                if instruction_words:
                    overlap = len(instruction_words.intersection(pred_words))
                    relevance = overlap / len(instruction_words)
                else:
                    relevance = 0.0
                
                relevance_scores.append(relevance)
            
            metrics['relevance'] = np.mean(relevance_scores)
            
            # Error rate - how many responses are error messages?
            error_responses = sum(1 for pred in predictions if pred.strip().startswith('দুঃখিত'))
            metrics['error_rate'] = error_responses / len(predictions) if predictions else 0.0
            
        except Exception as e:
            self.logger.error(f"Error evaluating overall quality: {e}")
            metrics = {'completeness': 0.0, 'coherence': 0.0, 'relevance': 0.0, 'error_rate': 1.0}
        
        return metrics
    
    def _calculate_composite_scores(self, metrics: Dict) -> Dict[str, float]:
        """Calculate composite evaluation scores"""
        composite_scores = {}
        
        try:
            # Overall NLP score
            nlp_metrics = metrics.get('nlp_metrics', {})
            nlp_score = np.mean([
                nlp_metrics.get('rouge1', 0) * 0.3,
                nlp_metrics.get('rouge2', 0) * 0.2,
                nlp_metrics.get('rougeL', 0) * 0.3,
                nlp_metrics.get('bleu', 0) * 0.2
            ])
            composite_scores['nlp_score'] = nlp_score
            
            # Bengali language score
            bengali_metrics = metrics.get('bengali_metrics', {})
            bengali_score = np.mean([
                bengali_metrics.get('bengali_character_ratio', 0) * 0.3,
                bengali_metrics.get('bengali_fluency', 0) * 0.5,
                bengali_metrics.get('unicode_normalization', 0) * 0.2
            ])
            composite_scores['bengali_score'] = bengali_score
            
            # Legal domain score
            legal_metrics = metrics.get('legal_metrics', {})
            legal_score = np.mean([
                legal_metrics.get('legal_terminology_usage', 0) * 0.3,
                legal_metrics.get('overall_domain_accuracy', 0) * 0.4,
                legal_metrics.get('legal_reasoning_indicators', 0) * 0.3
            ])
            composite_scores['legal_score'] = legal_score
            
            # Citation score
            citation_metrics = metrics.get('citation_metrics', {})
            citation_score = np.mean([
                citation_metrics.get('citation_accuracy', 0) * 0.7,
                citation_metrics.get('citation_format_accuracy', 0) * 0.3
            ])
            composite_scores['citation_score'] = citation_score
            
            # Quality score
            quality_metrics = metrics.get('quality_metrics', {})
            quality_score = np.mean([
                quality_metrics.get('completeness', 0) * 0.3,
                quality_metrics.get('coherence', 0) * 0.25,
                quality_metrics.get('relevance', 0) * 0.35,
                (1 - quality_metrics.get('error_rate', 1)) * 0.1  # Invert error rate
            ])
            composite_scores['quality_score'] = quality_score
            
            # Overall composite score
            composite_scores['overall_score'] = np.mean([
                nlp_score * 0.2,
                bengali_score * 0.2,
                legal_score * 0.3,
                citation_score * 0.15,
                quality_score * 0.15
            ])
            
        except Exception as e:
            self.logger.error(f"Error calculating composite scores: {e}")
            composite_scores = {
                'nlp_score': 0.0,
                'bengali_score': 0.0,
                'legal_score': 0.0,
                'citation_score': 0.0,
                'quality_score': 0.0,
                'overall_score': 0.0
            }
        
        return composite_scores
    
    def _get_model_info(self, model) -> Dict[str, Any]:
        """Get model information"""
        try:
            model_info = {
                'model_type': type(model).__name__,
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            }
            
            if hasattr(model, 'peft_config'):
                model_info['peft_type'] = str(type(model.peft_config))
                model_info['is_peft_model'] = True
            else:
                model_info['is_peft_model'] = False
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return {'model_type': 'unknown'}
    
    def _get_dataset_info(self, dataset) -> Dict[str, Any]:
        """Get dataset information"""
        try:
            dataset_info = {
                'size': len(dataset),
                'domains': {},
                'question_types': {}
            }
            
            # Count domains and question types
            for item in dataset:
                domain = item.get('domain', 'unknown')
                question_type = item.get('question_type', 'unknown')
                
                dataset_info['domains'][domain] = dataset_info['domains'].get(domain, 0) + 1
                dataset_info['question_types'][question_type] = dataset_info['question_types'].get(question_type, 0) + 1
            
            return dataset_info
            
        except Exception as e:
            self.logger.error(f"Error getting dataset info: {e}")
            return {'size': 0}
    
    def save_evaluation_results(self, filepath: Path):
        """Save evaluation results to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Evaluation results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation results: {e}")
    
    def generate_evaluation_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        if not self.evaluation_results:
            return "No evaluation results available. Run evaluation first."
        
        report_lines = []
        
        # Header
        report_lines.append("# Bengali Legal Advocate - Model Evaluation Report")
        report_lines.append(f"**Evaluation Date:** {self.evaluation_results.get('evaluation_date', 'Unknown')}")
        report_lines.append("")
        
        # Model Information
        model_info = self.evaluation_results.get('model_info', {})
        report_lines.append("## Model Information")
        report_lines.append(f"- **Model Type:** {model_info.get('model_type', 'Unknown')}")
        report_lines.append(f"- **Total Parameters:** {model_info.get('total_parameters', 0):,}")
        report_lines.append(f"- **Trainable Parameters:** {model_info.get('trainable_parameters', 0):,}")
        report_lines.append(f"- **PEFT Model:** {'Yes' if model_info.get('is_peft_model', False) else 'No'}")
        report_lines.append("")
        
        # Dataset Information
        dataset_info = self.evaluation_results.get('dataset_info', {})
        report_lines.append("## Dataset Information")
        report_lines.append(f"- **Test Samples:** {dataset_info.get('size', 0)}")
        
        domains = dataset_info.get('domains', {})
        if domains:
            report_lines.append("- **Domain Distribution:**")
            for domain, count in domains.items():
                percentage = (count / dataset_info['size']) * 100
                report_lines.append(f"  - {domain}: {count} ({percentage:.1f}%)")
        
        report_lines.append("")
        
        # Composite Scores
        composite_scores = self.evaluation_results.get('composite_scores', {})
        report_lines.append("## Overall Performance")
        report_lines.append(f"- **Overall Score:** {composite_scores.get('overall_score', 0):.3f}")
        report_lines.append(f"- **NLP Score:** {composite_scores.get('nlp_score', 0):.3f}")
        report_lines.append(f"- **Bengali Language Score:** {composite_scores.get('bengali_score', 0):.3f}")
        report_lines.append(f"- **Legal Domain Score:** {composite_scores.get('legal_score', 0):.3f}")
        report_lines.append(f"- **Citation Score:** {composite_scores.get('citation_score', 0):.3f}")
        report_lines.append(f"- **Quality Score:** {composite_scores.get('quality_score', 0):.3f}")
        report_lines.append("")
        
        # Detailed Metrics
        metrics = self.evaluation_results.get('metrics', {})
        
        # NLP Metrics
        nlp_metrics = metrics.get('nlp_metrics', {})
        if nlp_metrics:
            report_lines.append("## NLP Metrics")
            report_lines.append(f"- **ROUGE-1:** {nlp_metrics.get('rouge1', 0):.3f}")
            report_lines.append(f"- **ROUGE-2:** {nlp_metrics.get('rouge2', 0):.3f}")
            report_lines.append(f"- **ROUGE-L:** {nlp_metrics.get('rougeL', 0):.3f}")
            report_lines.append(f"- **BLEU:** {nlp_metrics.get('bleu', 0):.3f}")
            report_lines.append("")
        
        # Bengali Metrics
        bengali_metrics = metrics.get('bengali_metrics', {})
        if bengali_metrics:
            report_lines.append("## Bengali Language Metrics")
            report_lines.append(f"- **Bengali Character Ratio:** {bengali_metrics.get('bengali_character_ratio', 0):.3f}")
            report_lines.append(f"- **Bengali Fluency:** {bengali_metrics.get('bengali_fluency', 0):.3f}")
            report_lines.append(f"- **Unicode Normalization:** {bengali_metrics.get('unicode_normalization', 0):.3f}")
            report_lines.append("")
        
        # Legal Metrics
        legal_metrics = metrics.get('legal_metrics', {})
        if legal_metrics:
            report_lines.append("## Legal Domain Metrics")
            report_lines.append(f"- **Legal Terminology Usage:** {legal_metrics.get('legal_terminology_usage', 0):.3f}")
            report_lines.append(f"- **Overall Domain Accuracy:** {legal_metrics.get('overall_domain_accuracy', 0):.3f}")
            report_lines.append(f"- **Legal Reasoning Indicators:** {legal_metrics.get('legal_reasoning_indicators', 0):.3f}")
            report_lines.append("")
        
        # Citation Metrics
        citation_metrics = metrics.get('citation_metrics', {})
        if citation_metrics:
            report_lines.append("## Citation Metrics")
            report_lines.append(f"- **Citation Accuracy:** {citation_metrics.get('citation_accuracy', 0):.3f}")
            report_lines.append(f"- **Citation Format Accuracy:** {citation_metrics.get('citation_format_accuracy', 0):.3f}")
            report_lines.append("")
        
        # Quality Metrics
        quality_metrics = metrics.get('quality_metrics', {})
        if quality_metrics:
            report_lines.append("## Quality Metrics")
            report_lines.append(f"- **Completeness:** {quality_metrics.get('completeness', 0):.3f}")
            report_lines.append(f"- **Coherence:** {quality_metrics.get('coherence', 0):.3f}")
            report_lines.append(f"- **Relevance:** {quality_metrics.get('relevance', 0):.3f}")
            report_lines.append(f"- **Error Rate:** {quality_metrics.get('error_rate', 0):.3f}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        recommendations = self._generate_recommendations(composite_scores, metrics)
        for rec in recommendations:
            report_lines.append(f"- {rec}")
        
        return "\n".join(report_lines)
    
    def _generate_recommendations(self, composite_scores: Dict, metrics: Dict) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        overall_score = composite_scores.get('overall_score', 0)
        
        if overall_score < 0.5:
            recommendations.append("Overall performance is low. Consider extending training duration or improving data quality.")
        elif overall_score < 0.7:
            recommendations.append("Performance is moderate. Focus on specific weak areas identified below.")
        else:
            recommendations.append("Good overall performance. Fine-tune specific areas for optimization.")
        
        # Specific recommendations
        bengali_score = composite_scores.get('bengali_score', 0)
        if bengali_score < 0.6:
            recommendations.append("Bengali language fluency needs improvement. Consider more Bengali-specific training data.")
        
        legal_score = composite_scores.get('legal_score', 0)
        if legal_score < 0.6:
            recommendations.append("Legal domain knowledge is weak. Increase legal terminology and case-based training.")
        
        citation_score = composite_scores.get('citation_score', 0)
        if citation_score < 0.6:
            recommendations.append("Citation accuracy needs work. Focus on proper legal reference formatting.")
        
        quality_metrics = metrics.get('quality_metrics', {})
        error_rate = quality_metrics.get('error_rate', 0)
        if error_rate > 0.1:
            recommendations.append("High error rate detected. Review model stability and input handling.")
        
        return recommendations 