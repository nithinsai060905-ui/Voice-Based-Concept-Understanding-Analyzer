import streamlit as st
import os
import datetime
import numpy as np
import tempfile
from database import db_helper
from analyzer import speech_to_text, semantic_similarity, audio_features, fluency_analysis, confidence_analysis, keyword_matching
from reports import pdf_generator
from utils import visualization

# Page Config
st.set_page_config(
    page_title="Voice Concept Understanding Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Dark Theme Core overrides */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Card panel styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #3B82F6;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Header and Subtitles styling */
    h1, h2, h3 {
        color: #F1F5F9 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Highlight box style */
    .info-panel {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #3B82F6;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    
    /* Sidebar styling adjustment */
    section[data-testid="stSidebar"] {
        background-color: #1E293B !important;
    }
</style>
""", unsafe_allow_html=True)

# Create folders for reports if they don't exist
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Fetch settings from DB
app_settings = db_helper.get_settings()
default_fillers = app_settings.get("filler_words", "um,uh,like,you know,so,actually").split(",")
default_silence_db = int(app_settings.get("silence_threshold_db", "-35"))
default_whisper_model = app_settings.get("whisper_model", "tiny")
default_similarity_model = app_settings.get("similarity_model", "all-MiniLM-L6-v2")

# Main Page Header
st.title("🎙️ Voice Based Concept Understanding Analyzer")
st.markdown("Analyze how well someone understands a concept using voice recordings, acoustic measurements, and semantic similarity AI.")

# Initialize state
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# Sidebar - Settings & History
with st.sidebar:
    st.header("⚙️ Settings & Configuration")
    
    whisper_model_choice = st.selectbox(
        "Whisper Model (Transcription)",
        ["tiny", "base", "small"],
        index=["tiny", "base", "small"].index(default_whisper_model)
    )
    
    similarity_model_choice = st.selectbox(
        "Similarity Model",
        ["all-MiniLM-L6-v2"],
        index=0
    )
    
    silence_threshold = st.slider(
        "Silence Threshold (dB)",
        min_value=-50,
        max_value=-15,
        value=default_silence_db,
        step=5
    )
    
    custom_fillers = st.text_input(
        "Filler Words (comma-separated)",
        value=",".join(default_fillers)
    )
    
    # Save settings to DB if changed
    if (whisper_model_choice != default_whisper_model or 
        silence_threshold != default_silence_db or 
        custom_fillers != ",".join(default_fillers)):
        db_helper.save_setting("whisper_model", whisper_model_choice)
        db_helper.save_setting("silence_threshold_db", silence_threshold)
        db_helper.save_setting("filler_words", custom_fillers)
        st.toast("Settings updated and saved!", icon="💾")
        
    st.markdown("---")
    st.header("📜 Session History")
    
    # Session History Listing
    history = db_helper.get_analyses_history()
    
    if not history:
        st.info("No past analyses found.")
    else:
        for idx, item in enumerate(history):
            col1, col2 = st.columns([4, 1])
            with col1:
                # Format timestamp
                dt = datetime.datetime.fromisoformat(item["timestamp"].replace(" ", "T"))
                date_str = dt.strftime("%b %d, %H:%M")
                # Show session button
                if st.button(f"{item['session_name']} ({date_str})", key=f"hist_{item['id']}"):
                    st.session_state.analysis_results = item
            with col2:
                # Delete item
                if st.button("🗑️", key=f"del_{item['id']}", help="Delete from history"):
                    db_helper.delete_analysis(item["id"])
                    if st.session_state.analysis_results and st.session_state.analysis_results.get("id") == item["id"]:
                        st.session_state.analysis_results = None
                    st.rerun()

# Layout Columns: Target Concept Input (Left) & Audio Recorder/Uploader (Right)
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("📚 Concept Configuration")
    
    session_name = st.text_input(
        "Session / Speaker Name", 
        value=st.session_state.analysis_results.get("session_name", "Concept Test Session") if st.session_state.analysis_results else "Concept Test Session",
        placeholder="e.g. John Doe - Physics Gravity"
    )
    
    concept_description = st.text_area(
        "Reference Concept / Target Explanation",
        value=st.session_state.analysis_results.get("concept_desc", "") if st.session_state.analysis_results else "",
        placeholder="Explain Newton's Law of Universal Gravitation: The force of attraction between two masses is directly proportional to the product of their masses and inversely proportional to the square of the distance between them...",
        height=150
    )
    
    target_keywords_input = st.text_input(
        "Target Keywords & Phrases (comma-separated)",
        value=",".join(st.session_state.analysis_results.get("concepts_matched", []) + st.session_state.analysis_results.get("concepts_missed", [])) if st.session_state.analysis_results else "gravity, masses, proportional, distance, square, attraction",
        placeholder="e.g., gravity, force, proportional, inverse square"
    )

with right_col:
    st.subheader("🔊 Audio Input")
    
    audio_source = st.radio(
        "Choose Audio Source",
        ["Upload Audio File", "Record Live Audio"]
    )
    
    audio_file = None
    if audio_source == "Upload Audio File":
        audio_file = st.file_uploader(
            "Upload WAV, MP3, or M4A file", 
            type=["wav", "mp3", "m4a"]
        )
    else:
        # Use Streamlit's native audio_input
        audio_file = st.audio_input("Record your answer (Click microphone to start/stop)")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Analyze Trigger
    if st.button("🔥 Run Comprehensive Analysis", type="primary", use_container_width=True):
        if not concept_description.strip():
            st.error("Please provide a reference concept description first.")
        elif not audio_file:
            st.error("Please upload or record an audio response first.")
        else:
            with st.spinner("Analyzing Speech & Acoustics... (Please wait)"):
                # Save input audio to a temporary file
                suffix = ".wav"
                if hasattr(audio_file, "name"):
                    suffix = os.path.splitext(audio_file.name)[1]
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                    temp_audio.write(audio_file.read())
                    temp_audio_path = temp_audio.name
                
                try:
                    # 1. Speech-to-Text Transcription
                    transcription_results = speech_to_text.transcribe_audio(
                        temp_audio_path, 
                        model_name=whisper_model_choice
                    )
                    transcript = transcription_results["text"]
                    
                    # 2. Extract Acoustic Features
                    acoustic_results = audio_features.extract_audio_features(
                        temp_audio_path,
                        silence_threshold_db=silence_threshold
                    )
                    
                    # 3. Analyze Fluency
                    filler_words_list = [w.strip() for w in custom_fillers.split(",") if w.strip()]
                    fluency_results = fluency_analysis.analyze_fluency(
                        transcript,
                        acoustic_results["duration"],
                        acoustic_results["silent_intervals"],
                        filler_words_list=filler_words_list
                    )
                    
                    # 4. Keyword Coverage
                    keyword_list = [k.strip() for k in target_keywords_input.split(",") if k.strip()]
                    keyword_results = keyword_matching.match_keywords(transcript, keyword_list)
                    
                    # 5. Semantic Similarity
                    semantic_results = semantic_similarity.calculate_semantic_similarity(
                        transcript,
                        concept_description,
                        model_name=similarity_model_choice
                    )
                    
                    # 6. Confidence Scoring
                    confidence_results = confidence_analysis.calculate_confidence(
                        semantic_results["score"],
                        fluency_results["fluency_score"],
                        acoustic_results["f0"],
                        acoustic_results["rms"]
                    )
                    
                    # 7. Generate visualizations for PDF
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
                    
                    # 8. Compile PDF Report
                    pdf_filename = f"report_{int(datetime.datetime.now().timestamp())}.pdf"
                    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
                    
                    report_data = {
                        "session_name": session_name,
                        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "audio_filename": audio_file.name if hasattr(audio_file, "name") else "Live Recording",
                        "duration": acoustic_results["duration"],
                        "wpm": fluency_results["wpm"],
                        "similarity_score": semantic_results["score"],
                        "fluency_score": fluency_results["fluency_score"],
                        "confidence_score": confidence_results["confidence_score"],
                        "concept_desc": concept_description,
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
                        "fluency_feedback": fluency_results["feedback"]
                    }
                    
                    pdf_generator.generate_pdf_report(report_data, pdf_path, chart_image_path=chart_png_path)
                    
                    # Clean up temporary chart image
                    if os.path.exists(chart_png_path):
                        try:
                            os.remove(chart_png_path)
                        except Exception:
                            pass
                            
                    # 9. Save in SQLite Database
                    db_id = db_helper.save_analysis(
                        session_name=session_name,
                        audio_filename=report_data["audio_filename"],
                        transcript=transcript,
                        concept_desc=concept_description,
                        similarity_score=report_data["similarity_score"],
                        fluency_score=report_data["fluency_score"],
                        confidence_score=report_data["confidence_score"],
                        words_per_minute=report_data["wpm"],
                        filler_count=report_data["filler_count"],
                        pause_count=report_data["pause_count"],
                        duration=report_data["duration"],
                        concepts_matched=report_data["concepts_matched"],
                        concepts_missed=report_data["concepts_missed"],
                        pdf_report_path=pdf_path
                    )
                    
                    # Store variables in session state for displaying
                    report_data["id"] = db_id
                    report_data["timestamp"] = report_data["date"]
                    # Retain acoustic arrays for interactive plotly (session state only, not database)
                    report_data["acoustic_y"] = acoustic_results["y"]
                    report_data["acoustic_sr"] = acoustic_results["sr"]
                    report_data["acoustic_rms"] = acoustic_results["rms"]
                    report_data["acoustic_f0"] = acoustic_results["f0"]
                    report_data["acoustic_rms_times"] = acoustic_results["rms_times"]
                    report_data["acoustic_f0_times"] = acoustic_results["f0_times"]
                    report_data["silent_intervals"] = acoustic_results["silent_intervals"]
                    
                    st.session_state.analysis_results = report_data
                    st.success("Analysis Complete!")
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    import traceback
                    st.error(traceback.format_exc())
                finally:
                    # Clean up temporary audio file
                    if os.path.exists(temp_audio_path):
                        try:
                            os.remove(temp_audio_path)
                        except Exception:
                            pass

# --- Display Results Section ---
if st.session_state.analysis_results:
    res = st.session_state.analysis_results
    st.markdown("---")
    st.header(f"📊 Analysis Results: {res['session_name']}")
    
    # 3-Column Metric Cards Row
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        sim_val = int(res['similarity_score'] * 100)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Semantic similarity</div>
            <div class="metric-value">{sim_val}%</div>
            <div style="font-size:0.9rem; color:#A7F3D0;">Understanding Align</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Speech Fluency</div>
            <div class="metric-value">{int(res['fluency_score'])}/100</div>
            <div style="font-size:0.9rem; color:#F59E0B;">{res['wpm']:.1f} WPM Pace</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confidence Score</div>
            <div class="metric-value">{int(res['confidence_score'])}/100</div>
            <div style="font-size:0.9rem; color:#60A5FA;">{res.get('confidence_class', 'N/A')} Class</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tab_overview, tab_acoustics, tab_fluency, tab_keywords, tab_pdf = st.tabs([
        "👁️ Overview", 
        "📈 Acoustic Waves & Pitch", 
        "🗣️ Fluency & Pauses", 
        "🔑 Keywords & Concepts", 
        "📄 PDF Report & Download"
    ])
    
    with tab_overview:
        col_ov_left, col_ov_right = st.columns([3, 2])
        with col_ov_left:
            st.subheader("Speech Transcription")
            st.info(f"\"{res['transcript']}\"")
            
            st.subheader("Analysis Feedback Summary")
            st.markdown(f"**Confidence Analysis:** {res.get('confidence_feedback', '')}")
            st.markdown(f"**Fluency Analysis:** {res.get('fluency_feedback', '')}")
            
        with col_ov_right:
            st.subheader("Session Details")
            st.markdown(f"**Date:** {res.get('timestamp', 'N/A')}")
            st.markdown(f"**Duration:** {res.get('duration', 0.0):.2f} seconds")
            st.markdown(f"**Target Concept Word Length:** {len(res.get('concept_desc', '').split())} words")
            st.markdown(f"**Transcribed Speech Word Length:** {len(res.get('transcript', '').split())} words")
            
    with tab_acoustics:
        st.subheader("Interactive Waveform, RMS Energy & Pitch (F0) Tracking")
        st.markdown("This visualization charts the speech volume dynamics and pitch inflection (contour) over the duration of the recording. Shaded red zones indicate detected pauses/silence.")
        
        # We need the raw audio arrays. If loading from history, we might not have them.
        # Let's handle loading raw acoustic features from audio file if not in state.
        if "acoustic_y" in res:
            fig = visualization.plot_waveform_and_features_plotly(
                res["acoustic_y"],
                res["acoustic_sr"],
                res["acoustic_rms"],
                res["acoustic_f0"],
                res["acoustic_rms_times"],
                res["acoustic_f0_times"],
                res.get("silent_intervals", [])
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Try to reconstruct from saved report path if it exists
            # In history loads, we can either re-extract or show a warning.
            # Let's extract on the fly if the report file exists
            audio_f = res.get("audio_filename")
            # If it's a live recording, it wasn't saved, but we can display a note.
            st.warning("Waveform visualization is only available immediately after recording/analyzing. Reloading from history does not persist the raw signal.")
            
    with tab_fluency:
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            st.subheader("Pacing & Flow Metrics")
            st.metric(label="Words Per Minute (WPM)", value=f"{res['wpm']:.1f}")
            st.metric(label="Total Silent Pauses", value=f"{res['pause_count']}")
            st.metric(label="Total Pause Duration", value=f"{res['total_pause_duration']:.2f} seconds")
            st.metric(label="Pause Ratio (% of speech)", value=f"{int(res['pause_ratio'] * 100)}%")
        with col_f2:
            st.subheader("Detected Filler Words")
            st.markdown("Verbal fillers (e.g., 'like', 'um', 'uh') add lexical drag to speech delivery and lower confidence.")
            
            # Reconstruct detected fillers list
            # We can run a mini-analyzer on the transcript using the current settings filler words
            transcript_words = [w.lower().strip(".,?!;:-") for w in res['transcript'].split() if w.strip()]
            fillers_found = {}
            for filler in custom_fillers.split(","):
                f_clean = filler.strip().lower()
                if f_clean:
                    cnt = transcript_words.count(f_clean)
                    if cnt > 0:
                        fillers_found[f_clean] = cnt
                        
            if not fillers_found:
                st.success("🎉 Amazing! No filler words were detected in your speech.")
            else:
                for f_word, count in fillers_found.items():
                    st.write(f"• **\"{f_word}\"**: used {count} times")
                    
    with tab_keywords:
        st.subheader("Concept Coverage Checklist")
        st.write(f"Keyword match percentage: **{res.get('coverage', 0.0):.1f}%**")
        
        col_kw1, col_kw2 = st.columns(2)
        with col_kw1:
            st.markdown("### ✅ Matched Concepts")
            matched = res.get("concepts_matched", [])
            if not matched:
                st.write("*No concepts matched.*")
            else:
                for kw in matched:
                    st.markdown(f"- <span style='color:#34D399;'>{kw}</span>", unsafe_allow_html=True)
        with col_kw2:
            st.markdown("### ❌ Missed Concepts")
            missed = res.get("concepts_missed", [])
            if not missed:
                st.write("*All target concepts were mentioned! Excellent job.*")
            else:
                for kw in missed:
                    st.markdown(f"- <span style='color:#F87171;'>{kw}</span>", unsafe_allow_html=True)
                    
    with tab_pdf:
        st.subheader("Download Printable PDF Report")
        st.markdown("Get a comprehensive, publication-ready PDF document including all semantic, acoustic, fluency, and delivery metrics.")
        
        pdf_path = res.get("pdf_report_path")
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()
                
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_data,
                file_name=f"Concept_Analysis_{res['session_name'].replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            # If the PDF is missing or deleted, we can regenerate it on the fly
            st.error("Report PDF not found. It might have been deleted or moved.")
            if st.button("Regenerate PDF Report"):
                try:
                    # Save a temp matplotlib plot
                    chart_png_path = os.path.join(REPORTS_DIR, f"temp_plot_{int(datetime.datetime.now().timestamp())}.png")
                    # If we don't have raw arrays in this session state, create a dummy or skip
                    if "acoustic_y" in res:
                        visualization.save_acoustic_plots_matplotlib(
                            res["acoustic_y"], res["acoustic_sr"], res["acoustic_rms"],
                            res["acoustic_f0"], res["acoustic_rms_times"], res["acoustic_f0_times"],
                            chart_png_path
                        )
                    else:
                        chart_png_path = None
                        
                    regen_pdf_path = os.path.join(REPORTS_DIR, f"report_{int(datetime.datetime.now().timestamp())}.pdf")
                    
                    pdf_generator.generate_pdf_report(res, regen_pdf_path, chart_image_path=chart_png_path)
                    
                    # Save in DB
                    db_helper.save_setting("pdf_report_path", regen_pdf_path) # or update query
                    res["pdf_report_path"] = regen_pdf_path
                    st.success("Report successfully regenerated! Please refresh this tab.")
                except Exception as ex:
                    st.error(f"Failed to regenerate PDF: {ex}")
