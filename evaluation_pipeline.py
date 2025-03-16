import json
import logging
from typing import List, Dict
from dataclasses import dataclass, asdict
from collections import Counter
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer



class QAEvaluator:
    def __init__(self, model_outputs: List[Dict], annotated_data: List[Dict]):
        self.model_outputs = model_outputs
        self.annotated_data = self._preprocess_annotations(annotated_data)
        self.metrics = None
        self.question_logs = []
        self._setup_logger()

    @classmethod
    def from_json_files(cls, model_qa_path: str, annotated_qa_path: str):
        """Initialize from JSON files"""
        with open(model_qa_path) as f:
            model_answers = json.load(f)
        
        with open(annotated_qa_path) as f:
            annotated_answers = json.load(f)
        
        return cls(model_answers, annotated_answers)
        
    def _setup_logger(self):
        """Set up logger without basic config"""
        self.logger = logging.getLogger('QAEvaluator')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def _preprocess_annotations(self, data: List[Dict]) -> List[Dict]:
        """Ensure answers are in list format"""
        processed = []
        for entry in data:
            processed_entry = entry.copy()
            if isinstance(processed_entry['Answer'], str):
                processed_entry['Answer'] = [processed_entry['Answer']]
            processed.append(processed_entry)
        return processed

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        NLTK-based tokenizer that lowercases text, tokenizes it,
        removes punctuation, and strips trailing 'ing' and plural 's'.
        """
        # Lowercase text
        text = text.lower()
        # Tokenize using NLTK
        tokens = word_tokenize(text)
        wnl = WordNetLemmatizer()
        processed_tokens = []
        for token in tokens:
            if token in string.punctuation:
                continue
            token = wnl.lemmatize(token)
            processed_tokens.append(token)
        return processed_tokens

    def _exact_match(self, pred: str, truths: List[str]) -> bool:
        """Check if prediction matches any ground truth exactly"""
        return any(pred == truth for truth in truths)

    def _token_metrics(self, pred: str, truths: List[str]) -> tuple:
        """Calculate max recall and F1 across all ground truths"""
        tokens_pred = self.tokenize(pred)
        max_recall = 0.0
        max_f1 = 0.0

        for truth in truths:
            tokens_gt = self.tokenize(truth)
            # Count token frequencies
            pred_counts = Counter(tokens_pred)
            gt_counts = Counter(tokens_gt)
        # Calculate true positives
            common = 0
            for token in pred_counts:
                common += min(pred_counts[token], gt_counts.get(token, 0))

            # Precision denominator
            pred_total = len(tokens_pred)
            # Recall denominator
            gt_total = len(tokens_gt)
            pred_len = len(tokens_pred)
            gt_len = len(tokens_gt)

            precision = common / pred_total if pred_total > 0 else 0.0
            recall = common / gt_total if gt_total > 0 else 0.0
            
            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            if recall > max_recall:
                max_recall = recall
            if f1 > max_f1:
                max_f1 = f1

        return max_recall, max_f1

    def _log_question_stats(self, question: str, pred: str, truths: List[str], 
                           em: bool, recall: float, f1: float):
        """Simplified logging without timestamps/levels"""
        self.logger.info(f"\nQuestion: {question}")
        self.logger.info(f"Model Answer: {pred}")
        self.logger.info(f"True Answers: {truths}")
        self.logger.info(f"Exact Match: {em}")
        self.logger.info(f"Max Recall: {recall:.2f}")
        self.logger.info(f"Max F1: {f1:.2f}")
        self.logger.info("-" * 50)


        # JSON-serializable log storage
        self.question_logs.append({
            "question": question,
            "model_answer": pred,
            "true_answers": truths,
            "exact_match": em,
            "max_recall": float(f"{recall:.2f}"),
            "max_f1": float(f"{f1:.2f}")
        })

    def evaluate(self) -> Dict[str, float]:
        """Run full evaluation pipeline"""
        total = 0
        exact_matches = 0
        total_recall = 0.0
        total_f1 = 0.0

        self.logger.info("Starting QA Evaluation Pipeline")
        self.logger.info(f"Evaluating {len(self.model_outputs)} questions\n")

        for model_entry, annotated_entry in zip(self.model_outputs, self.annotated_data):
            # Validate question matching
            if model_entry['Question'] != annotated_entry['Question']:
                raise ValueError("Question mismatch between model output and annotations")

            question = model_entry['Question']
            pred_answer = model_entry['Answer']
            true_answers = annotated_entry['Answer']

            # Calculate metrics
            em = self._exact_match(pred_answer, true_answers)
            recall, f1 = self._token_metrics(pred_answer, true_answers)

            # Update aggregates
            exact_matches += int(em)
            total_recall += recall
            total_f1 += f1
            total += 1

            # Log individual question results
            self._log_question_stats(
                question=question,
                pred=pred_answer,
                truths=true_answers,
                em=em,
                recall=recall,
                f1=f1
            )

        # Calculate final metrics
        self.metrics = {
            'exact_match': (exact_matches / total) * 100,
            'answer_recall': (total_recall / total) * 100,
            'macro_f1': (total_f1 / total) * 100
        }

        self.logger.info("\nEvaluation Complete")
        self.logger.info(f"Exact Match: {self.metrics['exact_match']:.2f}%")
        self.logger.info(f"Answer Recall: {self.metrics['answer_recall']:.2f}%")
        self.logger.info(f"Macro F1: {self.metrics['macro_f1']:.2f}%")

        return self.metrics
    
    def get_metrics(self) -> Dict[str, float]:
        """Return calculated metrics"""
        return self.metrics.copy()
    
    def get_logger(self) -> logging.Logger:
        """Get the logger instance"""
        return self.logger

    def save_logs_to_json(self, output_path: str):
        """Save question logs to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.question_logs, f, indent=2)
        self.logger.info(f"\nSaved detailed logs to {output_path}")