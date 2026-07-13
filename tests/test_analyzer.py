import pytest
import numpy as np
from analyzer.keyword_matching import match_keywords
from analyzer.semantic_similarity import calculate_semantic_similarity
from analyzer.fluency_analysis import analyze_fluency
from analyzer.confidence_analysis import calculate_confidence

def test_keyword_matching_exact():
    transcript = "The universal gravitational constant is key to understanding mass attraction."
    keywords = ["gravitational constant", "mass", "attraction", "missing_kw"]
    
    result = match_keywords(transcript, keywords)
    
    assert "gravitational constant" in result["matched"]
    assert "mass" in result["matched"]
    assert "attraction" in result["matched"]
    assert "missing_kw" in result["missed"]
    assert result["coverage"] == 75.0  # 3 matched out of 4

def test_keyword_matching_stemming():
    transcript = "We are studying forces and accelerations in Newton's laws."
    keywords = ["force", "acceleration"]
    
    result = match_keywords(transcript, keywords)
    
    assert "force" in result["matched"]
    assert "acceleration" in result["matched"]
    assert len(result["missed"]) == 0
    assert result["coverage"] == 100.0

def test_fluency_analysis_scoring():
    transcript = "well um the gravity pulls objects down like you know basically towards the earth so yes"
    silent_intervals = [[0.0, 0.5], [2.0, 3.0]]  # 1.5 seconds silence total
    duration = 10.0
    fillers = ["um", "like", "you know", "basically", "well", "so"]
    
    result = analyze_fluency(transcript, duration, silent_intervals, filler_words_list=fillers)
    
    assert result["wpm"] == 96.0  # 16 words / 10s * 60s
    assert result["filler_count"] == 6
    assert result["pause_count"] == 2
    assert result["total_pause_duration"] == 1.5
    assert result["pause_ratio"] == 0.15
    assert 0.0 <= result["fluency_score"] <= 100.0

def test_confidence_analysis_synthesis():
    f0 = np.array([120.0, 125.0, 130.0, 128.0, 0.0, 122.0])
    rms = np.array([0.02, 0.022, 0.021, 0.025, 0.001, 0.02])
    
    result = calculate_confidence(
        semantic_score=0.85,
        fluency_score=90.0,
        f0=f0,
        rms=rms
    )
    
    assert result["confidence_score"] > 50.0
    assert result["classification"] in ["High", "Moderate", "Low"]
    assert len(result["feedback"]) > 0
