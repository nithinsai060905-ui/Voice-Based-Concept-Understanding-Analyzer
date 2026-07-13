import re

def analyze_fluency(transcript, duration, silent_intervals, filler_words_list=None):
    """
    Analyzes speech fluency, calculation of words per minute (WPM), 
    pauses, filler word usage, and computes a fluency score.
    
    Parameters:
        transcript (str): The transcribed text.
        duration (float): Total audio duration in seconds.
        silent_intervals (list): List of [start, end] times of silent regions.
        filler_words_list (list): Custom list of filler words (strings).
        
    Returns:
        dict: A dictionary containing:
            - 'wpm': Words per minute (float).
            - 'filler_count': Total filler words found (int).
            - 'filler_percentage': Percentage of filler words in text (float).
            - 'pause_count': Total number of pauses (int).
            - 'total_pause_duration': Total pause duration in seconds (float).
            - 'pause_ratio': Ratio of silence to total duration (float).
            - 'fluency_score': Computed fluency rating from 0 to 100 (float).
            - 'feedback': Detailed textual feedback on fluency.
    """
    if filler_words_list is None:
        filler_words_list = ["um", "uh", "like", "you know", "basically", "actually", "so", "well", "mean"]
        
    # Clean and split transcript
    words = [w.lower().strip(".,?!;:-") for w in transcript.split() if w.strip()]
    word_count = len(words)
    
    # 1. Words per Minute (WPM)
    duration_min = duration / 60.0 if duration > 0 else 0
    wpm = word_count / duration_min if duration_min > 0 else 0.0
    
    # 2. Filler words detection
    filler_count = 0
    detected_fillers = {}
    
    # Use regex for whole-word matching to avoid sub-word false positives (e.g. "some" matches "um" if not careful)
    for word in words:
        # Match against compound fillers like "you know"
        pass
        
    # Simple search of text for compound and single filler words
    text_lower = " " + " ".join(words) + " "
    for filler in filler_words_list:
        filler_clean = filler.lower().strip()
        # Find matches with boundaries
        pattern = r'\b' + re.escape(filler_clean) + r'\b'
        matches = re.findall(pattern, text_lower)
        count = len(matches)
        if count > 0:
            filler_count += count
            detected_fillers[filler_clean] = count
            
    filler_percentage = (filler_count / word_count * 100.0) if word_count > 0 else 0.0
    
    # 3. Pauses Analysis
    pause_count = len(silent_intervals)
    total_pause_duration = sum(end - start for start, end in silent_intervals)
    pause_ratio = total_pause_duration / duration if duration > 0 else 0.0
    
    # 4. Fluency Score Calculation
    # Normal speech rate: 110 - 150 WPM
    # Normal silence ratio: 10% - 25%
    # Normal filler rate: < 5% of words
    
    score = 100.0
    
    # Deductions:
    # WPM penalties
    if wpm > 0:
        if wpm < 90:
            # Too slow
            score -= (90 - wpm) * 0.8
        elif wpm > 170:
            # Too fast
            score -= (wpm - 170) * 0.8
            
    # Filler word penalties
    # Deduct 5 points per 2% filler rate
    score -= (filler_percentage / 2.0) * 5.0
    
    # Pause/Silence penalties
    # Deduct points if silence ratio is too high (e.g., > 30% of speech) or average pause is very long
    if pause_ratio > 0.30:
        score -= (pause_ratio - 0.30) * 100.0
    elif pause_ratio < 0.05 and word_count > 10:
        # Too rushed (no breathing room)
        score -= 10.0
        
    # Enforce bounds
    fluency_score = max(0.0, min(100.0, score))
    
    # Generate detailed fluency feedback
    feedback_parts = []
    if wpm == 0:
        feedback_parts.append("No speech was detected. Please try recording again.")
    else:
        if wpm < 100:
            feedback_parts.append("The pace is relatively slow. Try to speak more continuously.")
        elif wpm > 160:
            feedback_parts.append("The pace is fast. Slowing down slightly may improve clarity.")
        else:
            feedback_parts.append("The speaking pace is at a natural, standard speed.")
            
        if filler_percentage > 8.0:
            feedback_parts.append(f"A high frequency of filler words ({filler_count} detected, {filler_percentage:.1f}%) was noted. Focus on pausing silently instead of using verbal fillers like 'um' or 'like'.")
        elif filler_percentage > 3.0:
            feedback_parts.append(f"Some filler words were used ({filler_percentage:.1f}%). Minor improvements in fluency are possible.")
        else:
            feedback_parts.append("Minimal filler word usage was detected. Great job!")
            
        if pause_ratio > 0.35:
            feedback_parts.append("There are extensive silent pauses in the recording, which might suggest hesitation or difficulty recalling the material.")
        elif pause_ratio < 0.08:
            feedback_parts.append("Very few pauses detected. Remember to breathe and pause between major thoughts.")
            
    return {
        "wpm": round(wpm, 1),
        "filler_count": filler_count,
        "filler_percentage": round(filler_percentage, 1),
        "detected_fillers": detected_fillers,
        "pause_count": pause_count,
        "total_pause_duration": round(total_pause_duration, 2),
        "pause_ratio": round(pause_ratio, 2),
        "fluency_score": round(fluency_score, 1),
        "feedback": " ".join(feedback_parts)
    }
