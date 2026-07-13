import sys
import subprocess
import os

def run_verification():
    print("=" * 60)
    print("ENVIRONMENT VERIFICATION SCRIPT")
    print("=" * 60)
    
    # 1. Python Version
    print(f"Python Version: {sys.version}")
    assert sys.version_info >= (3, 12), "Python version must be 3.12+"
    
    # 2. Pip Version
    try:
        pip_version = subprocess.check_output([sys.executable, "-m", "pip", "--version"]).decode("utf-8").strip()
        print(f"Pip Version: {pip_version}")
    except Exception as e:
        print(f"Failed to check pip version: {e}")

    # 3. Import Libraries
    libraries = [
        ("streamlit", "streamlit"),
        ("whisper (OpenAI Whisper)", "whisper"),
        ("sentence_transformers", "sentence_transformers"),
        ("torch (PyTorch)", "torch"),
        ("torchaudio", "torchaudio"),
        ("transformers", "transformers"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("plotly", "plotly"),
        ("sklearn (scikit-learn)", "sklearn"),
        ("nltk", "nltk"),
        ("reportlab", "reportlab"),
        ("sqlite3", "sqlite3"),
        ("dotenv (python-dotenv)", "dotenv"),
        ("cv2 (opencv-python)", "cv2"),
        ("PIL (pillow)", "PIL"),
        ("scipy", "scipy"),
        ("pytest", "pytest"),
        ("tqdm", "tqdm"),
        ("joblib", "joblib"),
    ]
    
    print("\nChecking Package Imports:")
    print("-" * 30)
    failed_imports = []
    for name, import_name in libraries:
        try:
            __import__(import_name)
            print(f"  [PASS] {name}")
        except ImportError as e:
            print(f"  [FAIL] {name}: {e}")
            failed_imports.append(name)
            
    # 4. PyTorch & CUDA Check
    try:
        import torch
        print("\nPyTorch Verification:")
        print("-" * 30)
        print(f"  PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"  CUDA Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print("  Running in CPU-only mode (expected for this environment).")
    except Exception as e:
        print(f"  Failed PyTorch diagnostics: {e}")

    # 5. Sentence-BERT Loading Check
    try:
        from sentence_transformers import SentenceTransformer
        print("\nSentence-BERT Verification:")
        print("-" * 30)
        print("  Loading 'all-MiniLM-L6-v2' model...")
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        test_emb = sbert_model.encode(["Verification of semantic similarity works!"])
        print(f"  [PASS] Model loaded successfully. Embedding size: {test_emb.shape}")
    except Exception as e:
        print(f"  [FAIL] Sentence-BERT model loading failed: {e}")
        failed_imports.append("sentence-transformers-model")

    # 6. Whisper Loading Check
    try:
        import whisper
        print("\nWhisper Verification:")
        print("-" * 30)
        print("  Loading Whisper 'tiny' model (CPU)...")
        # Load whisper onto CPU explicitly
        whisper_model = whisper.load_model('tiny', device='cpu')
        print(f"  [PASS] Whisper model loaded successfully. Type: {type(whisper_model)}")
    except Exception as e:
        print(f"  [FAIL] Whisper model loading failed: {e}")
        failed_imports.append("whisper-model")

    print("\n" + "=" * 60)
    if failed_imports:
        print(f"VERIFICATION FAILED! Failures: {failed_imports}")
        sys.exit(1)
    else:
        print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)

if __name__ == "__main__":
    run_verification()
