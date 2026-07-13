from sentence_transformers import SentenceTransformer, util
import numpy as np

# Cache for sentence-transformers model
_SBERT_CACHE = {}

def get_similarity_model(model_name="all-MiniLM-L6-v2"):
    """Loads and caches the SentenceTransformer model."""
    if model_name not in _SBERT_CACHE:
        _SBERT_CACHE[model_name] = SentenceTransformer(model_name)
    return _SBERT_CACHE[model_name]

def calculate_semantic_similarity(student_transcript, reference_concept, model_name="all-MiniLM-L6-v2"):
    """
    Computes cosine similarity between the student's transcript and the reference concept.
    
    Parameters:
        student_transcript (str): Transcribed student response.
        reference_concept (str): Reference explanation or target concept summary.
        model_name (str): Sentence-BERT model name.
        
    Returns:
        dict: A dictionary containing:
            - 'score': Cosine similarity score (float between 0 and 1).
            - 'classification': Verbal grade ('Excellent', 'Good', 'Moderate', 'Weak').
            - 'explanation': Brief evaluation text.
    """
    if not student_transcript.strip() or not reference_concept.strip():
        return {
            "score": 0.0,
            "classification": "Weak",
            "explanation": "No text provided for comparison."
        }
        
    model = get_similarity_model(model_name)
    
    # Generate embeddings
    embeddings = model.encode([student_transcript, reference_concept], convert_to_tensor=True)
    
    # Compute similarity
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    
    # Normalize score (cosine similarity can be negative, but let's clamp it between 0 and 1)
    normalized_score = max(0.0, min(1.0, float(similarity)))
    
    # Classify the score
    if normalized_score >= 0.80:
        classification = "Excellent"
        explanation = "The response demonstrates a deep, highly aligned understanding of the target concept."
    elif normalized_score >= 0.65:
        classification = "Good"
        explanation = "The response captures the core concept well, showing solid understanding."
    elif normalized_score >= 0.45:
        classification = "Moderate"
        explanation = "The response shows partial understanding but lacks key details or has minor inaccuracies."
    else:
        classification = "Weak"
        explanation = "The response shows significant divergence or a lack of understanding of the concept."
        
    return {
        "score": normalized_score,
        "classification": classification,
        "explanation": explanation
    }
