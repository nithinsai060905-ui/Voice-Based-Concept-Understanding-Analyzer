import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def plot_waveform_and_features_plotly(y, sr, rms, f0, rms_times, f0_times, silent_intervals=None):
    """
    Creates an interactive, beautiful Plotly visualization of the audio waveform, 
    pitch tracking (F0), and RMS energy for the Streamlit UI.
    """
    # Create subplots: 2 rows (1 for waveform + RMS, 1 for Pitch)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=("Waveform & RMS Energy", "Pitch Tracking (F0 Contour)")
    )
    
    # 1. Waveform (downsampled for performance)
    duration = len(y) / sr if sr > 0 else 0
    step = max(1, len(y) // 2000) # limit to 2000 points
    y_downsampled = y[::step]
    x_downsampled = np.linspace(0, duration, len(y_downsampled))
    
    fig.add_trace(
        go.Scatter(x=x_downsampled, y=y_downsampled, name="Waveform",
                   line=dict(color="#93C5FD", width=1), opacity=0.7),
        row=1, col=1
    )
    
    # 2. RMS Energy Overlay
    if len(rms) > 0:
        fig.add_trace(
            go.Scatter(x=rms_times, y=rms, name="RMS Energy",
                       line=dict(color="#2563EB", width=2)),
            row=1, col=1
        )
        
    # 3. Pitch Tracking
    if len(f0) > 0:
        # Only plot voiced frames (F0 > 0) to avoid drawing zero-lines in unvoiced regions
        voiced_mask = f0 > 0
        f0_voiced = f0[voiced_mask]
        f0_times_voiced = f0_times[voiced_mask]
        
        fig.add_trace(
            go.Scatter(x=f0_times_voiced, y=f0_voiced, name="Pitch (F0)",
                       mode="markers+lines", marker=dict(size=3, color="#D97706"),
                       line=dict(color="#F59E0B", width=1.5)),
            row=2, col=1
        )
        
    # Shading silence intervals
    if silent_intervals:
        for start, end in silent_intervals:
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="#FEE2E2", opacity=0.3,
                layer="below", line_width=0,
                row=1, col=1
            )
            
    fig.update_layout(
        height=450,
        showlegend=True,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        paper_bgcolor="rgba(15, 23, 42, 0.8)"
    )
    
    fig.update_yaxes(title_text="Amplitude / Energy", row=1, col=1)
    fig.update_yaxes(title_text="Pitch (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="Time (Seconds)", row=2, col=1)
    
    return fig

def save_acoustic_plots_matplotlib(y, sr, rms, f0, rms_times, f0_times, output_path):
    """
    Generates a high-quality static multi-subplot figure using Matplotlib 
    and saves it to the disk for inclusion in PDF reports.
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 3 rows of subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 5.5), sharex=True)
    
    # 1. Waveform
    duration = len(y) / sr if sr > 0 else 0
    time_axis = np.linspace(0, duration, len(y))
    # Downsample slightly to speed up plotting
    ds_factor = max(1, len(y) // 5000)
    ax1.plot(time_axis[::ds_factor], y[::ds_factor], color='#A5F3FC', alpha=0.8, linewidth=0.5)
    ax1.set_title("Audio Waveform", fontsize=10, fontweight='bold', color='#1E293B')
    ax1.set_ylabel("Amplitude", fontsize=8)
    ax1.tick_params(labelsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. RMS Energy
    if len(rms) > 0:
        ax2.plot(rms_times, rms, color='#2563EB', linewidth=1.5)
    ax2.set_title("RMS Energy (Volume)", fontsize=10, fontweight='bold', color='#1E293B')
    ax2.set_ylabel("Energy", fontsize=8)
    ax2.tick_params(labelsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Pitch Contour
    if len(f0) > 0:
        voiced_mask = f0 > 0
        if np.any(voiced_mask):
            ax3.scatter(f0_times[voiced_mask], f0[voiced_mask], color='#D97706', s=2, alpha=0.7)
            ax3.plot(f0_times[voiced_mask], f0[voiced_mask], color='#F59E0B', linewidth=1, alpha=0.5)
    ax3.set_title("Pitch Tracking (F0 Contour)", fontsize=10, fontweight='bold', color='#1E293B')
    ax3.set_ylabel("Pitch (Hz)", fontsize=8)
    ax3.set_xlabel("Time (Seconds)", fontsize=9)
    ax3.tick_params(labelsize=8)
    ax3.grid(True, linestyle='--', alpha=0.5)
    if len(f0) > 0 and np.any(voiced_mask):
        ax3.set_ylim(max(0, np.min(f0[voiced_mask]) - 20), np.max(f0[voiced_mask]) + 20)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
