import librosa
import numpy as np
import os

def extract_audio_features(audio_path, silence_threshold_db=-35):
    """
    Extracts core acoustic features including duration, pitch (F0), 
    RMS energy, and silence intervals.
    
    Parameters:
        audio_path (str): Path to the audio file.
        silence_threshold_db (int): Threshold below reference (dB) to consider as silence.
        
    Returns:
        dict: A dictionary containing:
            - 'y': Audio signal (numpy array).
            - 'sr': Sample rate (int).
            - 'duration': Total duration in seconds (float).
            - 'rms': Frame-level root-mean-square energy (numpy array).
            - 'rms_times': Times corresponding to RMS frames (numpy array).
            - 'f0': Pitch tracking contour (numpy array, in Hz).
            - 'f0_times': Times corresponding to pitch values (numpy array).
            - 'non_silent_intervals': List of [start, end] times (seconds) of non-silent regions.
            - 'silent_intervals': List of [start, end] times (seconds) of silent regions.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    # Load audio (downsample/resample as necessary; let librosa handle sr)
    # sr=16000 is standard for audio analysis and Whisper
    y, sr = librosa.load(audio_path, sr=16000)
    
    duration = librosa.get_duration(y=y, sr=sr)
    
    if len(y) == 0:
        return {
            "y": y, "sr": sr, "duration": 0.0,
            "rms": np.array([]), "rms_times": np.array([]),
            "f0": np.array([]), "f0_times": np.array([]),
            "non_silent_intervals": [], "silent_intervals": []
        }
        
    # 1. Compute RMS Energy
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
    
    # 2. Pitch Tracking (F0) using Yin
    # Yin requires reasonable fmin and fmax
    fmin = 50.0
    fmax = 400.0
    try:
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
        # Yin might return very high values where it's unvoiced, or we can filter it
        # Replace extreme or nan values with 0
        f0 = np.nan_to_num(f0, nan=0.0)
        # Filter unvoiced frames by energy: if energy is very low, pitch is unvoiced (0)
        threshold_energy = np.max(rms) * 0.05
        for i in range(min(len(f0), len(rms))):
            if rms[i] < threshold_energy:
                f0[i] = 0.0
        f0_times = librosa.frames_to_time(range(len(f0)), sr=sr, hop_length=hop_length)
    except Exception as e:
        print(f"Pitch extraction warning: {e}")
        f0 = np.zeros_like(rms)
        f0_times = rms_times
        
    # 3. Detect non-silent intervals using librosa.effects.split
    # top_db is the threshold below reference (max db) for silence
    # So if silence_threshold_db is -35, top_db is 35
    top_db = abs(silence_threshold_db)
    non_silent_frames = librosa.effects.split(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    
    # Convert non-silent intervals to start-end seconds (librosa.effects.split returns sample indices)
    non_silent_sec = []
    for start, end in non_silent_frames:
        non_silent_sec.append([round(float(start) / sr, 3), round(float(end) / sr, 3)])
        
    # 4. Compute silent intervals as gaps between non-silent intervals
    silent_intervals = []
    if not non_silent_sec:
        # Whole audio is silent
        silent_intervals.append([0.0, round(duration, 3)])
    else:
        # Check gap at beginning
        if non_silent_sec[0][0] > 0.05:
            silent_intervals.append([0.0, non_silent_sec[0][0]])
            
        # Gaps between non-silent intervals
        for i in range(len(non_silent_sec) - 1):
            start = non_silent_sec[i][1]
            end = non_silent_sec[i+1][0]
            if end - start > 0.05:  # silence longer than 50ms
                silent_intervals.append([start, end])
                
        # Check gap at the end
        if duration - non_silent_sec[-1][1] > 0.05:
            silent_intervals.append([non_silent_sec[-1][1], round(duration, 3)])
            
    return {
        "y": y,
        "sr": sr,
        "duration": duration,
        "rms": rms,
        "rms_times": rms_times,
        "f0": f0,
        "f0_times": f0_times,
        "non_silent_intervals": non_silent_sec,
        "silent_intervals": silent_intervals
    }
