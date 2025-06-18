"""
ARES GUI Interface.

This module implements a graphical user interface for ARES using tkinter.
"""

import os
import sys
import time
import threading
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pyaudio
import wave
import tempfile

# Import ARES modules
from ..core.engine import ARESEngine

logger = logging.getLogger(__name__)

class ARESApp:
    """
    Main application class for the ARES GUI.
    """
    
    def __init__(self, engine=None):
        """
        Initialize the ARES GUI application.
        
        Args:
            engine: An initialized ARESEngine instance (optional)
        """
        self.engine = engine if engine else ARESEngine()
        self.current_file = None
        self.recording = False
        self.is_analyzing = False
        self.analysis_result = None
        self.audio_stream = None
        self.frames = []
        
        # Audio recording parameters
        self.sample_rate = 22050
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        
        logger.info("ARES GUI application initialized")
    
    def run(self):
        """
        Start the GUI application.
        """
        self.root = tk.Tk()
        self.root.title("ARES Voice Verification System")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Set theme and style
        self.setup_style()
        
        # Create the main layout
        self.create_layout()
        
        # Start the main event loop
        self.root.mainloop()
    
    def setup_style(self):
        """
        Set up the visual theme and styles.
        """
        # Configure the ttk style
        self.style = ttk.Style()
        
        # Try to use a modern theme if available
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass  # Use default theme if 'clam' is not available
        
        # Configure colors
        bg_color = "#f5f5f5"
        accent_color = "#3a7ca5"
        
        self.root.configure(bg=bg_color)
        
        # Configure styles
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TButton", padding=5, background=accent_color)
        self.style.configure("TLabel", background=bg_color)
        self.style.configure("Header.TLabel", font=('Helvetica', 16, 'bold'))
        self.style.configure("SubHeader.TLabel", font=('Helvetica', 12, 'bold'))
        self.style.configure("Authentic.TLabel", foreground="green", font=('Helvetica', 14, 'bold'))
        self.style.configure("Cloned.TLabel", foreground="red", font=('Helvetica', 14, 'bold'))
        
        # Configure progress bar style
        self.style.configure("TProgressbar", thickness=10, background=accent_color)
    
    def create_layout(self):
        """
        Create the main application layout.
        """
        # Create main container
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Application title
        title_label = ttk.Label(
            header_frame, 
            text="ARES Voice Verification System", 
            style="Header.TLabel"
        )
        title_label.pack(side=tk.LEFT)
        
        # Version info
        version_label = ttk.Label(
            header_frame, 
            text="v0.1.0", 
            style="TLabel"
        )
        version_label.pack(side=tk.RIGHT)
        
        # Create main content with left and right panels
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        left_panel = ttk.Frame(content_frame, padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        # Input section
        input_frame = ttk.LabelFrame(left_panel, text="Input", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 20))
        
        # File input
        file_frame = ttk.Frame(input_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=30)
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        browse_button = ttk.Button(
            file_frame, 
            text="Browse", 
            command=self.browse_file
        )
        browse_button.pack(side=tk.RIGHT)
        
        # Recording section
        record_frame = ttk.Frame(input_frame)
        record_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.record_button = ttk.Button(
            record_frame,
            text="Start Recording",
            command=self.toggle_recording
        )
        self.record_button.pack(fill=tk.X)
        
        # Analysis button
        analyze_frame = ttk.Frame(left_panel)
        analyze_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.analyze_button = ttk.Button(
            analyze_frame,
            text="Analyze Audio",
            command=self.analyze_audio
        )
        self.analyze_button.pack(fill=tk.X)
        
        # Analysis progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            left_panel, 
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 20))
        
        # Results summary section
        results_frame = ttk.LabelFrame(left_panel, text="Results Summary", padding=10)
        results_frame.pack(fill=tk.X)
        
        # Classification result
        classification_frame = ttk.Frame(results_frame)
        classification_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(classification_frame, text="Classification:", style="SubHeader.TLabel").pack(side=tk.LEFT)
        
        self.classification_label = ttk.Label(
            classification_frame, 
            text="N/A", 
            style="TLabel"
        )
        self.classification_label.pack(side=tk.RIGHT)
        
        # Confidence score
        confidence_frame = ttk.Frame(results_frame)
        confidence_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(confidence_frame, text="Confidence:", style="SubHeader.TLabel").pack(side=tk.LEFT)
        
        self.confidence_label = ttk.Label(
            confidence_frame, 
            text="N/A", 
            style="TLabel"
        )
        self.confidence_label.pack(side=tk.RIGHT)
        
        # Analysis time
        time_frame = ttk.Frame(results_frame)
        time_frame.pack(fill=tk.X)
        
        ttk.Label(time_frame, text="Analysis Time:", style="SubHeader.TLabel").pack(side=tk.LEFT)
        
        self.time_label = ttk.Label(
            time_frame, 
            text="N/A", 
            style="TLabel"
        )
        self.time_label.pack(side=tk.RIGHT)
        
        # Right panel for detailed results and visualizations
        right_panel = ttk.Frame(content_frame, padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create notebook for detailed results
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results overview tab
        self.overview_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.overview_tab, text="Overview")
        
        # Create a text area for findings
        findings_frame = ttk.LabelFrame(self.overview_tab, text="Key Findings", padding=10)
        findings_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.findings_text = tk.Text(findings_frame, wrap=tk.WORD, height=10)
        self.findings_text.pack(fill=tk.BOTH, expand=True)
        self.findings_text.configure(state='disabled')
        
        # Create a frame for the analysis stream results
        streams_frame = ttk.LabelFrame(self.overview_tab, text="Analysis Streams", padding=10)
        streams_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for visualization
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=streams_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Stream-specific tabs
        self.spectral_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.spectral_tab, text="Spectro-Temporal")
        
        self.acoustic_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.acoustic_tab, text="Acoustic & Biometric")
        
        self.linguistic_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.linguistic_tab, text="Linguistic & Prosodic")
    
    def browse_file(self):
        """
        Open file browser dialog to select an audio file.
        """
        filetypes = [
            ('Audio files', '*.wav *.mp3 *.flac *.ogg'),
            ('WAV files', '*.wav'),
            ('MP3 files', '*.mp3'),
            ('FLAC files', '*.flac'),
            ('All files', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=filetypes
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            self.current_file = file_path
    
    def toggle_recording(self):
        """
        Toggle audio recording on/off.
        """
        if self.recording:
            # Stop recording
            self.recording = False
            self.record_button.configure(text="Start Recording")
            self.stop_recording()
        else:
            # Start recording
            self.recording = True
            self.record_button.configure(text="Stop Recording")
            self.start_recording()
    
    def start_recording(self):
        """
        Start recording audio.
        """
        try:
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            
            # Open audio stream
            self.frames = []
            self.audio_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            # Start the stream
            self.audio_stream.start_stream()
            
            logger.info("Recording started")
            
        except Exception as e:
            logger.error(f"Error starting recording: {str(e)}")
            messagebox.showerror("Recording Error", f"Could not start recording: {str(e)}")
            self.recording = False
            self.record_button.configure(text="Start Recording")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback for audio recording.
        """
        if self.recording:
            self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def stop_recording(self):
        """
        Stop recording and save the audio file.
        """
        if self.audio_stream:
            # Stop and close the audio stream
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
            
            # Close PyAudio
            self.audio.terminate()
            
            # Save the recorded audio to a temporary file
            if self.frames:
                try:
                    # Create a temporary file
                    temp_file = tempfile.mktemp(suffix=".wav")
                    
                    # Save the audio data
                    wf = wave.open(temp_file, 'wb')
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.audio.get_sample_size(self.format))
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(b''.join(self.frames))
                    wf.close()
                    
                    # Set as current file
                    self.current_file = temp_file
                    self.file_path_var.set("Recording: " + os.path.basename(temp_file))
                    
                    logger.info(f"Recording saved to: {temp_file}")
                    
                except Exception as e:
                    logger.error(f"Error saving recording: {str(e)}")
                    messagebox.showerror("Recording Error", f"Could not save recording: {str(e)}")
    
    def analyze_audio(self):
        """
        Analyze the current audio file.
        """
        # Check if we have a file to analyze
        if not self.current_file:
            messagebox.showinfo("No File", "Please select an audio file or record audio first.")
            return
        
        # Check if file exists
        if not os.path.exists(self.current_file):
            messagebox.showerror("File Error", f"File not found: {self.current_file}")
            return
        
        # Disable UI during analysis
        self.set_ui_analyzing_state(True)
        
        # Reset progress bar
        self.progress_var.set(0)
        
        # Start analysis in a separate thread
        threading.Thread(target=self.run_analysis, daemon=True).start()
    
    def run_analysis(self):
        """
        Run the analysis in a background thread.
        """
        try:
            # Show initial progress
            self.update_progress(10)
            
            # Perform analysis
            start_time = time.time()
            self.analysis_result = self.engine.analyze_file(self.current_file)
            
            # Update progress
            self.update_progress(100)
            
            # Update UI with results
            self.root.after(0, self.update_ui_with_results)
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {str(e)}")
            self.root.after(0, lambda: self.show_analysis_error(str(e)))
            
        finally:
            # Re-enable UI
            self.root.after(0, lambda: self.set_ui_analyzing_state(False))
    
    def update_progress(self, value):
        """
        Update the progress bar.
        """
        self.root.after(0, lambda: self.progress_var.set(value))
    
    def set_ui_analyzing_state(self, analyzing):
        """
        Set the UI state during analysis.
        
        Args:
            analyzing: True if analysis is running, False otherwise
        """
        self.is_analyzing = analyzing
        state = 'disabled' if analyzing else 'normal'
        
        self.analyze_button.configure(state=state)
        self.record_button.configure(state=state)
        
        if analyzing:
            self.analyze_button.configure(text="Analyzing...")
        else:
            self.analyze_button.configure(text="Analyze Audio")
    
    def show_analysis_error(self, error_msg):
        """
        Show an error message after failed analysis.
        """
        messagebox.showerror("Analysis Error", f"Could not analyze audio: {error_msg}")
    
    def update_ui_with_results(self):
        """
        Update the UI with analysis results.
        """
        if not self.analysis_result:
            return
            
        # Extract results
        classification = self.analysis_result['classification']
        confidence = self.analysis_result['confidence_score']
        evidence = self.analysis_result['evidence']
        analysis_time = self.analysis_result.get('analysis_time', 0)
        
        # Update classification with appropriate style
        if classification == 'Authentic':
            self.classification_label.configure(text=classification, style="Authentic.TLabel")
        else:
            self.classification_label.configure(text=classification, style="Cloned.TLabel")
        
        # Update confidence and time
        self.confidence_label.configure(text=f"{confidence:.2f}%")
        self.time_label.configure(text=f"{analysis_time:.2f} seconds")
        
        # Update findings text
        findings = []
        
        # Add decision factors
        if 'fusion' in evidence and 'decision_factors' in evidence['fusion']:
            findings.extend(evidence['fusion']['decision_factors'])
            
        # Add individual stream findings
        stream_keys = ['spectro_temporal', 'acoustic_biometric', 'linguistic_prosodic']
        for key in stream_keys:
            if key in evidence and 'findings' in evidence[key]:
                findings.extend(evidence[key]['findings'])
        
        # Update the findings text widget
        self.findings_text.configure(state='normal')
        self.findings_text.delete(1.0, tk.END)
        self.findings_text.insert(tk.END, "\n".join(f"• {finding}" for finding in findings))
        self.findings_text.configure(state='disabled')
        
        # Update the overall visualization
        self.update_visualization()
    
    def update_visualization(self):
        """
        Update the visualization with analysis results.
        """
        if not self.analysis_result or 'evidence' not in self.analysis_result:
            return
            
        evidence = self.analysis_result['evidence']
        
        # Extract scores
        scores = []
        labels = []
        
        if 'spectro_temporal' in evidence:
            scores.append(evidence['spectro_temporal']['score'])
            labels.append('Spectro-Temporal')
            
        if 'acoustic_biometric' in evidence:
            scores.append(evidence['acoustic_biometric']['score'])
            labels.append('Acoustic & Biometric')
            
        if 'linguistic_prosodic' in evidence:
            scores.append(evidence['linguistic_prosodic']['score'])
            labels.append('Linguistic & Prosodic')
            
        if 'fusion' in evidence:
            scores.append(evidence['fusion']['score'])
            labels.append('Final Score')
        
        # Plot the scores
        self.ax.clear()
        
        # Set colors based on scores (green for authentic, red for cloned)
        colors = ['green' if s < 0.5 else 'red' for s in scores]
        
        # Create the bar plot
        bars = self.ax.bar(labels, scores, color=colors)
        
        # Add a horizontal line at 0.5 (threshold)
        self.ax.axhline(y=0.5, color='gray', linestyle='--')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            self.ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.05,
                f"{height:.2f}",
                ha='center', va='bottom'
            )
        
        # Set labels and title
        self.ax.set_ylabel('Score')
        self.ax.set_title('Analysis Scores by Stream')
        self.ax.set_ylim(0, 1.1)  # Ensure there's room for text
        
        # Draw the canvas
        self.canvas.draw()
    
    def on_close(self):
        """
        Handle window close event.
        """
        # Clean up any resources
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        # Close the window
        self.root.destroy()
