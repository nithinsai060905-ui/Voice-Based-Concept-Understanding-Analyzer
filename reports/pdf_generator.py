from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import os

def generate_pdf_report(analysis_data, output_path, chart_image_path=None):
    """
    Generates a beautiful, professionally-styled PDF report for the voice concept analysis.
    
    Parameters:
        analysis_data (dict): Dict of analysis metrics (transcript, similarity_score, fluency_score, etc.)
        output_path (str): Filepath where the PDF will be saved.
        chart_image_path (str, optional): Filepath to a saved PNG plot to embed in the PDF.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles to look premium
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1E3A8A"), # Deep Navy
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#4B5563"), # Slate Gray
        spaceAfter=20
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1F2937"),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#374151")
    )
    
    body_bold = ParagraphStyle(
        'Body_Bold_Custom',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    metric_title_style = ParagraphStyle(
        'MetricTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        alignment=1, # Center
        textColor=colors.HexColor("#4B5563")
    )
    
    metric_value_style = ParagraphStyle(
        'MetricValue',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        alignment=1, # Center
        textColor=colors.HexColor("#2563EB") # Brand Blue
    )
    
    # Build Flowables
    story = []
    
    # 1. Header Section
    story.append(Paragraph("Voice Concept Understanding Analysis", title_style))
    story.append(Paragraph(f"Session Name: {analysis_data.get('session_name', 'Unnamed Session')}", subtitle_style))
    
    # Metadata Table
    meta_data = [
        [Paragraph("<b>Date:</b>", body_style), Paragraph(analysis_data.get("date", "N/A"), body_style),
         Paragraph("<b>Audio File:</b>", body_style), Paragraph(analysis_data.get("audio_filename", "N/A"), body_style)],
        [Paragraph("<b>Duration:</b>", body_style), Paragraph(f"{analysis_data.get('duration', 0.0):.2f} seconds", body_style),
         Paragraph("<b>Speaking Rate:</b>", body_style), Paragraph(f"{analysis_data.get('wpm', 0.0):.1f} WPM", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[80, 180, 80, 180])
    meta_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 20))
    
    # 2. Key Metrics Callout Boxes (Scores)
    sim_percent = int(analysis_data.get("similarity_score", 0.0) * 100)
    fluency_score = int(analysis_data.get("fluency_score", 0.0))
    conf_score = int(analysis_data.get("confidence_score", 0.0))
    
    metrics_data = [
        [
            Paragraph("Semantic Similarity", metric_title_style),
            Paragraph("Speech Fluency", metric_title_style),
            Paragraph("Confidence Score", metric_title_style)
        ],
        [
            Paragraph(f"{sim_percent}%", metric_value_style),
            Paragraph(f"{fluency_score}/100", metric_value_style),
            Paragraph(f"{conf_score}/100", metric_value_style)
        ]
    ]
    metrics_table = Table(metrics_data, colWidths=[175, 175, 175])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # 3. Reference Concept & Student Transcript
    story.append(Paragraph("Target Concept Description", h1_style))
    story.append(Paragraph(analysis_data.get("concept_desc", "N/A"), body_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("Speech Transcript", h1_style))
    transcript_text = analysis_data.get("transcript", "No transcript available.")
    story.append(Paragraph(f"\"{transcript_text}\"", ParagraphStyle('TranscriptItalic', parent=body_style, fontName='Helvetica-Oblique')))
    story.append(Spacer(1, 20))
    
    # 4. Detailed Concept & Keywords Analysis
    story.append(Paragraph("Keyword & Concept Coverage", h1_style))
    matched_kws = analysis_data.get("concepts_matched", [])
    missed_kws = analysis_data.get("concepts_missed", [])
    
    coverage_text = f"<b>Keyword Match Coverage:</b> {len(matched_kws)} / {len(matched_kws) + len(missed_kws)} terms ({analysis_data.get('coverage', 0.0):.1f}%)"
    story.append(Paragraph(coverage_text, body_style))
    story.append(Spacer(1, 8))
    
    kw_data = [
        [Paragraph("<b>Matched Concepts:</b>", body_bold), Paragraph(", ".join(matched_kws) if matched_kws else "None", body_style)],
        [Paragraph("<b>Missed Concepts:</b>", body_bold), Paragraph(", ".join(missed_kws) if missed_kws else "None", body_style)]
    ]
    kw_table = Table(kw_data, colWidths=[120, 400])
    kw_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#F1F5F9")),
    ]))
    story.append(kw_table)
    story.append(Spacer(1, 20))
    
    # 5. Acoustic & Delivery Feedback
    story.append(Paragraph("Speech Delivery & Acoustic Feedback", h1_style))
    delivery_data = [
        [Paragraph("<b>Silent Pauses:</b>", body_bold), Paragraph(f"{analysis_data.get('pause_count', 0)} pauses (Total: {analysis_data.get('total_pause_duration', 0.0)} seconds, {int(analysis_data.get('pause_ratio', 0.0)*100)}% of duration)", body_style)],
        [Paragraph("<b>Filler Words:</b>", body_bold), Paragraph(f"{analysis_data.get('filler_count', 0)} detected ({analysis_data.get('filler_percentage', 0.0)}% of words)", body_style)],
        [Paragraph("<b>Confidence Level:</b>", body_bold), Paragraph(f"<b>{analysis_data.get('confidence_class', 'N/A')}</b> - {analysis_data.get('confidence_feedback', '')}", body_style)],
        [Paragraph("<b>Fluency Feedback:</b>", body_bold), Paragraph(analysis_data.get('fluency_feedback', ''), body_style)],
    ]
    delivery_table = Table(delivery_data, colWidths=[120, 400])
    delivery_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#F1F5F9")),
    ]))
    story.append(delivery_table)
    
    # 5b. Google Gemini AI Analysis Feedback
    gemini_data = analysis_data.get("gemini_feedback")
    if gemini_data:
        story.append(Spacer(1, 15))
        story.append(Paragraph("Google Gemini AI Feedback & Evaluation", h1_style))
        
        strengths_list = gemini_data.get("strengths", [])
        weaknesses_list = gemini_data.get("weaknesses", [])
        missing_list = gemini_data.get("missing_concepts", [])
        suggestions_list = gemini_data.get("learning_suggestions", [])
        tips_list = gemini_data.get("improvement_tips", [])
        evaluation_text = gemini_data.get("overall_evaluation", "")
        
        def format_list(lst):
            if not lst:
                return "None"
            if isinstance(lst, str):
                return lst
            return "<br/>".join(f"• {item}" for item in lst)
            
        gemini_table_data = [
            [Paragraph("<b>Overall Evaluation:</b>", body_bold), Paragraph(evaluation_text or "N/A", body_style)],
            [Paragraph("<b>Strengths:</b>", body_bold), Paragraph(format_list(strengths_list), body_style)],
            [Paragraph("<b>Weaknesses:</b>", body_bold), Paragraph(format_list(weaknesses_list), body_style)],
            [Paragraph("<b>Omitted Concepts:</b>", body_bold), Paragraph(format_list(missing_list), body_style)],
            [Paragraph("<b>Learning Suggestions:</b>", body_bold), Paragraph(format_list(suggestions_list), body_style)],
            [Paragraph("<b>Improvement Tips:</b>", body_bold), Paragraph(format_list(tips_list), body_style)]
        ]
        gemini_table = Table(gemini_table_data, colWidths=[130, 390])
        gemini_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#F1F5F9")),
        ]))
        story.append(KeepTogether([gemini_table]))
        
    # 6. Visualization Chart (if provided)
    if chart_image_path and os.path.exists(chart_image_path):
        story.append(Spacer(1, 20))
        # Keep chart on next page if it doesn't fit, or wrap in KeepTogether
        chart_flowables = [
            Paragraph("Acoustic Visualizations", h1_style),
            Image(chart_image_path, width=500, height=220),
            Spacer(1, 10)
        ]
        story.append(KeepTogether(chart_flowables))
        
    # Build the document
    doc.build(story)
