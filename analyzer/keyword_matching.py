import re

def clean_word(word):
    """Utility to lowercase and strip punctuation from a word."""
    return re.sub(r'[^\w\s]', '', word.lower()).strip()

def match_keywords(transcript, target_keywords):
    """
    Performs robust keyword and key phrase matching on the student transcript.
    Supports single words, compound phrases, and basic suffix matching (plurals/tenses).
    
    Parameters:
        transcript (str): The student's transcribed text.
        target_keywords (list): List of strings representing target terms.
        
    Returns:
        dict: A dictionary containing:
            - 'matched': List of matched keywords/phrases.
            - 'missed': List of missed keywords/phrases.
            - 'coverage': Percentage of keywords matched (float).
    """
    if not target_keywords:
        return {"matched": [], "missed": [], "coverage": 100.0}
        
    # Standardize transcript text
    cleaned_transcript = " " + clean_word(transcript) + " "
    
    matched = []
    missed = []
    
    for kw in target_keywords:
        kw_clean = kw.strip()
        if not kw_clean:
            continue
            
        kw_normalized = clean_word(kw_clean)
        
        # Scenario 1: Exact phrase or word match with word boundaries
        pattern = r'\b' + re.escape(kw_normalized) + r'\b'
        if re.search(pattern, cleaned_transcript):
            matched.append(kw_clean)
            continue
            
        # Scenario 2: Simple stemming / plural checks for single-word keywords
        # e.g., matching "force" against "forces" or "accelerate" against "acceleration"
        words_in_kw = kw_normalized.split()
        if len(words_in_kw) == 1:
            word = words_in_kw[0]
            # Create regex pattern for variations: force -> forces, force's; gravity -> gravitation, gravitational
            # Let's check if the word is a prefix of any word in the transcript (minimum length 4 to avoid short false positives)
            matched_stem = False
            transcript_words = cleaned_transcript.split()
            for tw in transcript_words:
                if len(word) >= 4:
                    # Check if they share a common root (e.g. prefix or substring)
                    if tw.startswith(word) or word.startswith(tw):
                        matched_stem = True
                        break
                else:
                    if tw == word:
                        matched_stem = True
                        break
            if matched_stem:
                matched.append(kw_clean)
                continue
                
        # If we reached here, the keyword was not found
        missed.append(kw_clean)
        
    total_kws = len(matched) + len(missed)
    coverage = (len(matched) / total_kws * 100.0) if total_kws > 0 else 100.0
    
    return {
        "matched": matched,
        "missed": missed,
        "coverage": round(coverage, 1)
    }
