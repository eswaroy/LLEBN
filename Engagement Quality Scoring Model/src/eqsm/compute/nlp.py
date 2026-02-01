from typing import List, Set
import re
import math

class NLPProcessor:
    """
    Statistical NLP Engine.
    Computes Semantic Depth and Novelty without heavy DL models.
    """
    
    def __init__(self):
        self.common_words = {"the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def calculate_semantic_depth(self, text: str) -> float:
        """
        Estimates depth based on vocabulary richness, structure, and length.
        """
        if not text:
            return 0.0
            
        tokens = self._tokenize(text)
        if len(tokens) == 0:
            return 0.0

        # Feature 1: Unique Word Ratio (Lexical Diversity)
        unique_tokens = set(tokens)
        diversity = len(unique_tokens) / len(tokens)
        
        # Feature 2: Content Word Density (ignoring stops)
        content_words = [t for t in tokens if t not in self.common_words]
        density = len(content_words) / len(tokens) if tokens else 0.0
        
        # Feature 3: Structure (Sentence count proxy via punctuation)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if len(s.strip()) > 5])
        structure_score = min(1.0, sentence_count / 5.0) # Cap at 5 sentences

        # Feature 4: Length Log-Scale (Saturates at ~200 words)
        length_score = min(1.0, math.log(len(tokens) + 1) / math.log(200))
        
        # Weighted Check
        raw_score = (diversity * 0.2) + (density * 0.2) + (structure_score * 0.3) + (length_score * 0.3)
        
        return raw_score

    def calculate_novelty(self, text: str, context_texts: List[str]) -> float:
        """
        Computes Jaccard Dissimilarity against context (parent/thread).
        Returns 1.0 (Unique) to 0.0 (Copy).
        """
        if not context_texts:
            return 1.0 # First post is novel
            
        target_tokens = set(self._tokenize(text))
        if not target_tokens:
            return 0.0
            
        max_similarity = 0.0
        
        for ctx in context_texts:
            ctx_tokens = set(self._tokenize(ctx))
            if not ctx_tokens:
                continue
                
            intersection = len(target_tokens.intersection(ctx_tokens))
            union = len(target_tokens.union(ctx_tokens))
            
            similarity = intersection / union if union > 0 else 0.0
            max_similarity = max(max_similarity, similarity)
            
        return 1.0 - max_similarity
