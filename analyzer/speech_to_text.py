import whisper
import os
import torch

# Cache for loaded models to avoid reloading on each analysis
_MODEL_CACHE = {}

def get_whisper_model(model_name="tiny"):
    """Loads and caches the specified OpenAI Whisper model on the CPU."""
    if model_name not in _MODEL_CACHE:
        # Explicitly force CPU loading to avoid torch cuda warnings in this environment
        _MODEL_CACHE[model_name] = whisper.load_model(model_name, device="cpu")
    return _MODEL_CACHE[model_name]

def transcribe_audio(audio_path, model_name="tiny"):
    """
    Transcribes an audio file using OpenAI Whisper.
    
    Parameters:
        audio_path (str): Path to the audio file.
        model_name (str): Whisper model type ('tiny', 'base', 'small', 'medium', 'large').
        
    Returns:
        dict: A dictionary containing:
            - 'text': The full transcribed text.
            - 'segments': List of transcription segments containing start, end, and text.
            - 'words': List of words with timestamps (if word_timestamps is supported and enabled).
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        
    model = get_whisper_model(model_name)
    
    # Transcribe the audio
    # word_timestamps=True provides precise word segment times
    result = model.transcribe(audio_path, word_timestamps=True, fp16=False)
    
    # Extract word timestamps if available
    words_data = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            words_data.append({
                "word": word_info.get("word", "").strip(),
                "start": word_info.get("start", 0.0),
                "end": word_info.get("end", 0.0),
                "probability": word_info.get("probability", 0.0)
            })
            
    return {
        "text": result.get("text", "").strip(),
        "segments": result.get("segments", []),
        "words": words_data
    }
