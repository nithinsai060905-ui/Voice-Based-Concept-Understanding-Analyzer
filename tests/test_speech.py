import pytest
from unittest.mock import patch, MagicMock
from analyzer.speech_to_text import transcribe_audio

def test_transcribe_audio_not_found():
    with pytest.raises(FileNotFoundError):
        transcribe_audio("nonexistent_file.wav")

@patch('os.path.exists')
@patch('whisper.load_model')
def test_transcribe_audio_mock(mock_load, mock_exists):
    mock_exists.return_value = True
    
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "Hello world from speech analyzer",
        "segments": [
            {
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.99},
                    {"word": "world", "start": 0.5, "end": 1.0, "probability": 0.95}
                ]
            }
        ]
    }
    mock_load.return_value = mock_model
    
    res = transcribe_audio("fake_audio.wav", model_name="tiny")
    
    assert res["text"] == "Hello world from speech analyzer"
    assert len(res["words"]) == 2
    assert res["words"][0]["word"] == "Hello"
    assert res["words"][1]["start"] == 0.5
