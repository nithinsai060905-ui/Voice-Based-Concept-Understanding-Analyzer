import os
import json
import requests

def generate_gemini_feedback(concept_desc: str, student_transcript: str, 
                             matched_concepts: list, missed_concepts: list, 
                             similarity_score: float, fluency_score: float, 
                             confidence_score: float) -> dict:
    """
    Generates intelligent feedback using Google Gemini AI API.
    If the API key is not present or an error occurs, falls back to an intelligent mock feedback.
    
    Returns:
        dict: A dictionary containing:
            - 'strengths': list of strings
            - 'weaknesses': list of strings
            - 'missing_concepts': list of strings
            - 'learning_suggestions': list of strings
            - 'improvement_tips': list of strings
            - 'overall_evaluation': string
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # 1. Fallback / Mock Generator if key is not found
    if not api_key:
        return _generate_mock_feedback(
            concept_desc, student_transcript, matched_concepts, 
            missed_concepts, similarity_score, fluency_score, confidence_score,
            "Google Gemini API Key is missing. Active fallback generation is running."
        )
        
    # 2. Prompt construction
    prompt = f"""
    You are an expert academic evaluator. Your job is to analyze a student's voice response describing a concept, compare it with the reference concept description, and provide detailed, professional feedback.
    
    Reference Concept:
    "{concept_desc}"
    
    Student's Transcript:
    "{student_transcript}"
    
    Metrics Computed:
    - Matched Keywords/Phrases: {matched_concepts}
    - Missed Keywords/Phrases: {missed_concepts}
    - Semantic Similarity: {similarity_score * 100:.1f}%
    - Speech Fluency Score: {fluency_score:.1f}/100
    - Overall Confidence Score: {confidence_score:.1f}/100
    
    Analyze the inputs carefully and provide the following points in JSON format:
    1. strengths: List of 2-3 specific concepts the student explained well or aspects of their delivery that were strong.
    2. weaknesses: List of 2-3 specific misconceptions, errors, or issues in their speaking delivery (e.g. hesitation, filler words).
    3. missing_concepts: List of key details from the reference concept that they omitted.
    4. learning_suggestions: List of 2-3 concepts or resources they should review.
    5. improvement_tips: List of 2-3 practical tips (e.g. speak slower, use less fillers, structure answer with examples).
    6. overall_evaluation: A concise, professional summary (3-4 sentences) evaluating their overall level of conceptual understanding.
    """
    
    # Standard Gemini API endpoint for 2.5 Flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Request configuration with responseSchema for guaranteed JSON formatting
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "strengths": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "weaknesses": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "missing_concepts": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "learning_suggestions": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "improvement_tips": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "overall_evaluation": {"type": "STRING"}
                },
                "required": ["strengths", "weaknesses", "missing_concepts", "learning_suggestions", "improvement_tips", "overall_evaluation"]
            }
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        
        resp_data = response.json()
        text_content = resp_data["candidates"][0]["content"]["parts"][0]["text"]
        
        # Parse the structured JSON feedback
        feedback_dict = json.loads(text_content)
        return feedback_dict
        
    except Exception as e:
        print(f"Gemini API Exception: {e}")
        return _generate_mock_feedback(
            concept_desc, student_transcript, matched_concepts, 
            missed_concepts, similarity_score, fluency_score, confidence_score,
            f"An error occurred while calling the Gemini API ({str(e)}). Local fallback generation is running."
        )

def _generate_mock_feedback(concept_desc, student_transcript, matched_concepts, 
                             missed_concepts, similarity_score, fluency_score, 
                             confidence_score, notice) -> dict:
    """Generates standard engineering academic feedback when API is offline or missing."""
    
    # Strengths
    strengths = []
    if len(matched_concepts) > 0:
        strengths.append(f"Successfully integrated key terminology such as: {', '.join(matched_concepts[:3])}.")
    if similarity_score >= 0.7:
        strengths.append("Demonstrated high conceptual alignment with the target explanation.")
    if fluency_score >= 75:
        strengths.append("Showed good verbal flow and pacing with minimal vocal filler drag.")
    if not strengths:
        strengths.append("Provided a verbal response for evaluation.")
        strengths.append("Attempted to cover the target concept keywords.")
        
    # Weaknesses
    weaknesses = []
    if len(missed_concepts) > 0:
        weaknesses.append(f"Omitted or did not clearly articulate: {', '.join(missed_concepts[:3])}.")
    if similarity_score < 0.6:
        weaknesses.append("The response deviated significantly from the reference conceptual definition.")
    if fluency_score < 60:
        weaknesses.append("Pacing issues detected, including extended pauses or excessive filler words (um/like).")
    if not weaknesses:
        weaknesses.append("No major structural flaws found. Minor delivery polishing can be done.")
        
    # Missing Concepts
    missing = missed_concepts if missed_concepts else ["None - all keywords matched!"]
    
    # Suggestions
    suggestions = []
    if len(missed_concepts) > 0:
        suggestions.append(f"Review core definitions relating to: {', '.join(missed_concepts[:2])}.")
    suggestions.append("Re-read the reference concept summary and focus on logical sequence.")
    suggestions.append("Practice summarizing technical explanations in under 60 seconds.")
    
    # Tips
    tips = []
    if fluency_score < 70:
        tips.append("Slow down your rate of speech to allow natural breathing and reduce verbal pauses.")
        tips.append("Practice silent pauses instead of vocal fillers like 'um' or 'like'.")
    else:
        tips.append("Maintain the current steady pace and try to add expressive vocal variance.")
    tips.append("Draft a quick mental bullet-point list before starting your voice recording.")
    
    # Overall Evaluation
    if similarity_score >= 0.8:
        eval_text = "The response is excellent, showing a clear, comprehensive, and highly accurate understanding of the concept. The delivery is structured and correctly frames all key principles."
    elif similarity_score >= 0.6:
        eval_text = "The response is solid, displaying a good understanding of the core concept. Most central ideas were communicated, though minor details or precise terminologies were omitted."
    elif similarity_score >= 0.4:
        eval_text = "The response shows moderate understanding. The student is familiar with some vocabulary but failed to explain the core mechanics, relations, or definitions accurately."
    else:
        eval_text = "The response indicates a weak understanding of the target concept. Key ideas were missed, and the speech showed high deviation from the academic standard definitions."
        
    eval_text += f" (Note: {notice})"
    
    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "missing_concepts": missing,
        "learning_suggestions": suggestions,
        "improvement_tips": tips,
        "overall_evaluation": eval_text
    }
