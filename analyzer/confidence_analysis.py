import numpy as np

def calculate_confidence(semantic_score, fluency_score, f0, rms):
    """
    Computes a synthesized Confidence Score combining semantic understanding, 
    fluency metrics, and acoustic qualities (pitch stability, volume range).
    
    Parameters:
        semantic_score (float): Cosine similarity score, 0.0 to 1.0.
        fluency_score (float): Fluency rating, 0.0 to 100.0.
        f0 (numpy.ndarray): Pitch array.
        rms (numpy.ndarray): RMS energy array.
        
    Returns:
        dict: A dictionary containing:
            - 'confidence_score': Score from 0 to 100 (float).
            - 'acoustic_confidence': Acoustic components sub-score (float).
            - 'lexical_confidence': Lexical components sub-score (float).
            - 'classification': Verbal classification ('High', 'Moderate', 'Low').
            - 'feedback': Textual explanation of the confidence level.
    """
    # 1. Acoustic Confidence Calculation
    # Pitch (F0) Analysis
    active_f0 = f0[f0 > 0.0]
    pitch_score = 70.0  # Default baseline
    
    if len(active_f0) > 5:
        f0_std = np.std(active_f0)
        # Optimal expressive pitch variation is standard deviation between 20Hz and 55Hz
        if 20.0 <= f0_std <= 55.0:
            pitch_score = 90.0 + (55.0 - f0_std) * 0.2
        elif f0_std < 20.0:
            # Monotone or reading speech
            pitch_score = max(40.0, 70.0 - (20.0 - f0_std) * 2.0)
        else:
            # Unstable or high jitter
            pitch_score = max(40.0, 80.0 - (f0_std - 55.0) * 0.8)
            
    # Volume/Energy Stability (RMS Coefficient of Variation)
    volume_score = 70.0
    if len(rms) > 5 and np.mean(rms) > 0.001:
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        cv = rms_std / rms_mean  # Coefficient of variation
        
        # Expressive speech has a CV between 0.3 and 0.6
        if 0.3 <= cv <= 0.6:
            volume_score = 95.0
        elif cv < 0.3:
            # Mumbled or entirely flat volume
            volume_score = max(50.0, 80.0 - (0.3 - cv) * 100.0)
        else:
            # High volume instability (whispering to shouting)
            volume_score = max(40.0, 80.0 - (cv - 0.6) * 50.0)
            
    acoustic_confidence = (pitch_score * 0.5) + (volume_score * 0.5)
    
    # 2. Lexical/Semantic Confidence (alignment + fluency)
    # High semantic similarity indicates correct material grasp, which correlates with confidence
    # Convert semantic score from 0.0-1.0 to 0-100 scale
    semantic_scale = semantic_score * 100.0
    lexical_confidence = (semantic_scale * 0.6) + (fluency_score * 0.4)
    
    # 3. Overall Synthesis
    # 40% Semantic/Understanding, 30% Fluency/Tempo, 30% Acoustic (Pitch & Volume)
    overall_score = (semantic_scale * 0.4) + (fluency_score * 0.3) + (acoustic_confidence * 0.3)
    overall_score = max(0.0, min(100.0, overall_score))
    
    # Classification
    if overall_score >= 80.0:
        classification = "High"
        feedback = "The speaker demonstrates strong vocal stability, continuous flow, and highly accurate concept articulation, indicating high confidence."
    elif overall_score >= 60.0:
        classification = "Moderate"
        feedback = "The speaker shows moderate confidence, with minor vocal hesitations, slightly flat expression, or minor conceptual inaccuracies."
    else:
        classification = "Low"
        feedback = "Low confidence is indicated by high hesitation rates, conceptual misalignment, or unstable volume and monotonous pitch."
        
    return {
        "confidence_score": round(overall_score, 1),
        "acoustic_confidence": round(acoustic_confidence, 1),
        "lexical_confidence": round(lexical_confidence, 1),
        "classification": classification,
        "feedback": feedback
    }
