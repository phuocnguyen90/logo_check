from typing import Dict, Any, List
from .scoring import CompositeScorer
from ..text.extractor import TextExtractor
from ..text.phonetic import PhoneticMatcher
from ..utils.logging import logger

class CompositeScoringPipeline:
    """
    Combines visual (Stage 1+2) and text (Stages 5) similarity scores.
    """
    
    def __init__(self):
        self.scorer = CompositeScorer()
        self.text_extractor = TextExtractor()
        self.phonetic_matcher = PhoneticMatcher()
        logger.info("Initialized CompositeScoringPipeline.")

    def score_results(
        self, 
        query_metadata: Dict[str, Any], 
        query_img: Any,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculates final weighted scores for a list of candidates.
        """
        query_text = self.text_extractor.extract(query_metadata, query_img)
        
        for cand in candidates:
            # 1. Visual Scores (Assumed already set by ReRanker)
            v_global = cand.get("global_score", 0.0)
            v_spatial = cand.get("spatial_score", 0.0)
            
            # 2. Text Score
            cand_text = self.text_extractor.extract(cand.get("metadata", {}))
            t_sim = self.phonetic_matcher.compute_similarity(query_text, cand_text)
            
            # 3. Composite Score
            final_score = self.scorer.calculate_final_score(
                global_sim=v_global,
                spatial_sim=v_spatial,
                text_sim=t_sim,
                color_sim=cand.get("color_score", 0.0)
            )
            
            cand["text_similarity"] = t_sim
            cand["final_score"] = final_score
            
        # Sort by final score
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates
