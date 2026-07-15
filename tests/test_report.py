import os
import pytest
from reports.pdf_generator import generate_pdf_report

def test_generate_pdf_report(tmp_path):
    output_pdf = os.path.join(tmp_path, "test_report.pdf")
    
    report_data = {
        "session_name": "Test Speech PDF Session",
        "date": "2026-07-15 15:00:00",
        "audio_filename": "test_speech.wav",
        "duration": 12.5,
        "wpm": 120.0,
        "similarity_score": 0.85,
        "fluency_score": 90.0,
        "confidence_score": 88.0,
        "concept_desc": "Newton's second law: Force equals mass times acceleration.",
        "transcript": "Well, force is mass multiplied by acceleration, basically.",
        "concepts_matched": ["force", "mass", "acceleration"],
        "concepts_missed": [],
        "coverage": 100.0,
        "pause_count": 2,
        "total_pause_duration": 1.2,
        "pause_ratio": 0.1,
        "filler_count": 1,
        "filler_percentage": 5.0,
        "confidence_class": "High",
        "confidence_feedback": "Expressive pitch variation.",
        "fluency_feedback": "Minor hesitation.",
        "gemini_feedback": {
            "strengths": ["Clear mapping of force and mass", "Excellent speaking rate"],
            "weaknesses": ["Minor grammatical structure deviation"],
            "missing_concepts": [],
            "learning_suggestions": ["Explore vector representations of forces"],
            "improvement_tips": ["Speak directly into the microphone"],
            "overall_evaluation": "A solid explanation matching physics."
        }
    }
    
    generate_pdf_report(report_data, output_pdf, chart_image_path=None)
    
    assert os.path.exists(output_pdf)
    assert os.path.getsize(output_pdf) > 0
