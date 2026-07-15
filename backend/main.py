import os
import sys
import uuid
import shutil
import tempfile
import datetime
from typing import List, Optional

# Add project root directory to path to resolve modular imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import db_helper
from analyzer import speech_to_text, audio_features, fluency_analysis, keyword_matching, semantic_similarity, confidence_analysis, gemini_feedback
from reports import pdf_generator
from utils import auth_helper, visualization

app = FastAPI(
    title="Voice-Based Concept Understanding Analyzer (VBCUA) API",
    description="REST backend for acoustic feature extraction, speech transcription, semantic evaluation, and Gemini AI feedback.",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.environ.get("SECRET_KEY", "vbcua_secret_key_default_9981273")
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generated_reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Pydantic schemas
class UserRegister(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    created_at: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class AnalyzeRequest(BaseModel):
    audio_path: str
    concept_desc: str
    keywords: str
    session_name: str
    silence_threshold_db: Optional[int] = -35
    whisper_model: Optional[str] = "tiny"
    similarity_model: Optional[str] = "all-MiniLM-L6-v2"

# Security dependency
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = auth_helper.decode_jwt_token(token, SECRET_KEY)
        username = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: missing subject",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user = db_helper.get_user_by_username(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_data: UserRegister):
    """Registers a new user."""
    if not user_data.username.strip() or not user_data.password.strip():
        raise HTTPException(status_code=400, detail="Username and password cannot be empty")
        
    existing_user = db_helper.get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
        
    hashed = auth_helper.hash_password(user_data.password)
    user_id = db_helper.create_user(user_data.username, hashed)
    if not user_id:
        raise HTTPException(status_code=500, detail="Failed to register user")
        
    user = db_helper.get_user_by_id(user_id)
    return {
        "id": user["id"],
        "username": user["username"],
        "created_at": user["created_at"]
    }

@app.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Logs in an existing user and issues an access token."""
    user = db_helper.get_user_by_username(form_data.username)
    if not user or not auth_helper.verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    token = auth_helper.create_jwt_token({"sub": user["username"]}, SECRET_KEY)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/upload")
def upload_audio(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """Uploads an audio file and saves it temporarily to the server. Returns the file path."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".wav", ".mp3", ".m4a"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format. Use WAV, MP3, or M4A.")
        
    filename = f"{uuid.uuid4()}{ext}"
    dest_path = os.path.join(UPLOAD_DIR, filename)
    
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {
        "filename": file.filename,
        "saved_path": dest_path,
        "message": "File uploaded successfully"
    }

@app.post("/transcribe")
def transcribe_audio_file(audio_path: str = Query(...), whisper_model: str = "tiny", current_user: dict = Depends(get_current_user)):
    """Transcribes an uploaded audio file using OpenAI Whisper."""
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found on server")
        
    try:
        result = speech_to_text.transcribe_audio(audio_path, model_name=whisper_model)
        return {
            "transcript": result["text"],
            "words_count": len(result["text"].split()),
            "words": result["words"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@app.post("/analyze")
def run_analysis(request: AnalyzeRequest, current_user: dict = Depends(get_current_user)):
    """Runs a complete audio and semantic analysis, fetches Gemini AI feedback, and saves to database."""
    if not os.path.exists(request.audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found on server")
        
    try:
        # 1. Transcribe
        transcription_results = speech_to_text.transcribe_audio(
            request.audio_path, 
            model_name=request.whisper_model
        )
        transcript = transcription_results["text"]
        
        # 2. Extract Acoustic Features
        acoustic_results = audio_features.extract_audio_features(
            request.audio_path,
            silence_threshold_db=request.silence_threshold_db
        )
        
        # 3. Analyze Fluency
        db_settings = db_helper.get_settings()
        default_fillers = db_settings.get("filler_words", "um,uh,like,you know").split(",")
        fluency_results = fluency_analysis.analyze_fluency(
            transcript,
            acoustic_results["duration"],
            acoustic_results["silent_intervals"],
            filler_words_list=default_fillers
        )
        
        # 4. Keyword Coverage
        keyword_list = [k.strip() for k in request.keywords.split(",") if k.strip()]
        keyword_results = keyword_matching.match_keywords(transcript, keyword_list)
        
        # 5. Semantic Similarity
        semantic_results = semantic_similarity.calculate_semantic_similarity(
            transcript,
            request.concept_desc,
            model_name=request.similarity_model
        )
        
        # 6. Confidence Scoring
        confidence_results = confidence_analysis.calculate_confidence(
            semantic_results["score"],
            fluency_results["fluency_score"],
            acoustic_results["f0"],
            acoustic_results["rms"]
        )
        
        # 7. Gemini feedback
        gemini_res = gemini_feedback.generate_gemini_feedback(
            request.concept_desc,
            transcript,
            keyword_results["matched"],
            keyword_results["missed"],
            semantic_results["score"],
            fluency_results["fluency_score"],
            confidence_results["confidence_score"]
        )
        
        # 8. Generate chart and PDF
        chart_png_path = os.path.join(REPORTS_DIR, f"temp_plot_{int(datetime.datetime.now().timestamp())}.png")
        visualization.save_acoustic_plots_matplotlib(
            acoustic_results["y"],
            acoustic_results["sr"],
            acoustic_results["rms"],
            acoustic_results["f0"],
            acoustic_results["rms_times"],
            acoustic_results["f0_times"],
            chart_png_path
        )
        
        pdf_filename = f"report_{int(datetime.datetime.now().timestamp())}.pdf"
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
        
        report_data = {
            "session_name": request.session_name,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "audio_filename": os.path.basename(request.audio_path),
            "duration": acoustic_results["duration"],
            "wpm": fluency_results["wpm"],
            "similarity_score": semantic_results["score"],
            "fluency_score": fluency_results["fluency_score"],
            "confidence_score": confidence_results["confidence_score"],
            "concept_desc": request.concept_desc,
            "transcript": transcript,
            "concepts_matched": keyword_results["matched"],
            "concepts_missed": keyword_results["missed"],
            "coverage": keyword_results["coverage"],
            "pause_count": fluency_results["pause_count"],
            "total_pause_duration": fluency_results["total_pause_duration"],
            "pause_ratio": fluency_results["pause_ratio"],
            "filler_count": fluency_results["filler_count"],
            "filler_percentage": fluency_results["filler_percentage"],
            "confidence_class": confidence_results["classification"],
            "confidence_feedback": confidence_results["feedback"],
            "fluency_feedback": fluency_results["feedback"],
            "gemini_feedback": gemini_res
        }
        
        pdf_generator.generate_pdf_report(report_data, pdf_path, chart_image_path=chart_png_path)
        
        # Clean up temp plot
        if os.path.exists(chart_png_path):
            try:
                os.remove(chart_png_path)
            except Exception:
                pass
                
        # Clean up original audio file after analysis is done
        try:
            os.remove(request.audio_path)
        except Exception:
            pass
            
        # 9. Save in SQLite Database
        db_id = db_helper.save_analysis(
            session_name=request.session_name,
            audio_filename=report_data["audio_filename"],
            transcript=transcript,
            concept_desc=request.concept_desc,
            similarity_score=report_data["similarity_score"],
            fluency_score=report_data["fluency_score"],
            confidence_score=report_data["confidence_score"],
            words_per_minute=report_data["wpm"],
            filler_count=report_data["filler_count"],
            pause_count=report_data["pause_count"],
            duration=report_data["duration"],
            concepts_matched=report_data["concepts_matched"],
            concepts_missed=report_data["concepts_missed"],
            pdf_report_path=pdf_path,
            user_id=current_user["id"],
            gemini_feedback=gemini_res
        )
        
        report_data["id"] = db_id
        return {
            "message": "Analysis successfully executed and saved",
            "analysis_id": db_id,
            "results": report_data
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis pipeline failure: {str(e)}")

@app.post("/generate-report")
def generate_report_pdf(analysis_id: int = Query(...), current_user: dict = Depends(get_current_user)):
    """Triggers PDF generation manually for a past analysis. Returns the download path."""
    history = db_helper.get_analyses_history(user_id=current_user["id"])
    target_record = None
    for rec in history:
        if rec["id"] == analysis_id:
            target_record = rec
            break
            
    if not target_record:
        raise HTTPException(status_code=404, detail="Analysis record not found or access denied")
        
    # Reconstruct report structure
    try:
        pdf_filename = f"report_{analysis_id}_{int(datetime.datetime.now().timestamp())}.pdf"
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
        
        # Adapt database row to PDF schema
        report_data = {
            "session_name": target_record["session_name"],
            "date": target_record["timestamp"],
            "audio_filename": target_record["audio_filename"],
            "duration": target_record["duration"],
            "wpm": target_record["words_per_minute"],
            "similarity_score": target_record["similarity_score"],
            "fluency_score": target_record["fluency_score"],
            "confidence_score": target_record["confidence_score"],
            "concept_desc": target_record["concept_desc"],
            "transcript": target_record["transcript"],
            "concepts_matched": target_record["concepts_matched"],
            "concepts_missed": target_record["concepts_missed"],
            "coverage": (len(target_record["concepts_matched"]) / max(1, len(target_record["concepts_matched"]) + len(target_record["concepts_missed"]))) * 100,
            "pause_count": target_record["pause_count"],
            "total_pause_duration": 0.0,  # default placeholder for reconstructed row
            "pause_ratio": 0.0,
            "filler_count": target_record["filler_count"],
            "filler_percentage": 0.0,
            "confidence_class": "N/A",
            "confidence_feedback": "Historical record summary.",
            "fluency_feedback": "Historical record summary.",
            "gemini_feedback": target_record.get("gemini_feedback")
        }
        
        pdf_generator.generate_pdf_report(report_data, pdf_path, chart_image_path=None)
        
        # Save pdf path back to DB
        # Retrieve settings or create a simple update function.
        # For simplicity, we just return the generated path
        return {
            "message": "Report generated",
            "pdf_report_path": pdf_path,
            "download_endpoint": f"/download-report?analysis_id={analysis_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Generation failure: {str(e)}")

@app.get("/history", response_model=List[dict])
def get_user_history(current_user: dict = Depends(get_current_user)):
    """Retrieves the history of past analyses for the authenticated user."""
    return db_helper.get_analyses_history(user_id=current_user["id"])

@app.get("/download-report")
def download_pdf_report(analysis_id: int = Query(...), current_user: dict = Depends(get_current_user)):
    """Downloads the generated PDF report for an analysis record."""
    history = db_helper.get_analyses_history(user_id=current_user["id"])
    target_record = None
    for rec in history:
        if rec["id"] == analysis_id:
            target_record = rec
            break
            
    if not target_record:
        raise HTTPException(status_code=404, detail="Analysis record not found or access denied")
        
    pdf_path = target_record.get("pdf_report_path")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF report file does not exist on server")
        
    return FileResponse(
        pdf_path, 
        media_type="application/pdf", 
        filename=f"Report_{target_record['session_name'].replace(' ', '_')}.pdf"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
