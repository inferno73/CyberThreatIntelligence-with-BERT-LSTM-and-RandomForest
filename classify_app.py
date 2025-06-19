import tkinter as tk
import nlpaug
from tkinter import ttk, filedialog, messagebox
import threading
import os
from pathlib import Path
import pickle


# Model loading imports
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax


class CyberThreatClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🛡️ CyberGuard AI - Threat Classification System")
        self.root.geometry("1000x800")
        
        # Modern dark cybersecurity theme
        self.colors = {
            'bg_primary': '#0a0e1a',      # Deep dark blue
            'bg_secondary': '#1a1f2e',    # Slightly lighter dark blue
            'bg_tertiary': '#252b3d',     # Card backgrounds
            'accent_cyan': '#00ffff',     # Neon cyan
            'accent_green': '#00ff41',    # Matrix green
            'accent_red': '#ff073a',      # Alert red
            'accent_orange': '#ff8c00',   # Warning orange
            'text_primary': '#ffffff',    # White text
            'text_secondary': '#b0c4de',  # Light blue text
            'text_muted': '#718096',      # Muted text
            'border': '#2d3748',          # Border color
            'success': '#00ff41',         # Success green
            'warning': '#ffab00',         # Warning amber
            'error': '#ff1744'            # Error red
        }
        
        self.setup_theme()
        
        # Model storage
        self.lstm_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.rf_model = None
        self.augmenter = None
        self.lstm_tokenizer = None
        self.tfidf_vectorizer = None
        
        # Model paths
        self.lstm_path  = ""
        self.bert_path  = ""
        self.rf_path    = "Modeli\\random_forest_model.joblib"
        self.aug_path   = "Modeli\\augmenter.pkl"
        
        self.setup_ui()
        
    def setup_theme(self):
        """Configure the modern cybersecurity theme"""
        self.root.configure(bg=self.colors['bg_primary'])
        
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure main styles
        style.configure('Cyber.TFrame', 
                       background=self.colors['bg_primary'],
                       borderwidth=0)
        
        style.configure('Card.TFrame',
                       background=self.colors['bg_tertiary'],
                       relief='flat',
                       borderwidth=1)
        
        style.configure('Header.TLabel',
                       background=self.colors['bg_primary'],
                       foreground=self.colors['accent_cyan'],
                       font=('Consolas', 18, 'bold'))
        
        style.configure('Subheader.TLabel',
                       background=self.colors['bg_tertiary'],
                       foreground=self.colors['text_primary'],
                       font=('Segoe UI', 12, 'bold'))
        
        style.configure('Cyber.TLabel',
                       background=self.colors['bg_tertiary'],
                       foreground=self.colors['text_secondary'],
                       font=('Segoe UI', 10))
        
        style.configure('Status.TLabel',
                       background=self.colors['bg_tertiary'],
                       foreground=self.colors['accent_green'],
                       font=('Consolas', 9))
        
        style.configure('Cyber.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       borderwidth=1,
                       focuscolor='none')
        
        style.map('Cyber.TButton',
                 background=[('active', self.colors['accent_cyan']),
                           ('!active', self.colors['bg_secondary'])],
                 foreground=[('active', self.colors['bg_primary']),
                           ('!active', self.colors['text_primary'])],
                 bordercolor=[('active', self.colors['accent_cyan']),
                            ('!active', self.colors['border'])])
        
        style.configure('Action.TButton',
                       font=('Segoe UI', 11, 'bold'),
                       borderwidth=2,
                       focuscolor='none')
        
        style.map('Action.TButton',
                 background=[('active', self.colors['accent_green']),
                           ('!active', self.colors['bg_secondary'])],
                 foreground=[('active', self.colors['bg_primary']),
                           ('!active', self.colors['accent_green'])],
                 bordercolor=[('active', self.colors['accent_green']),
                            ('!active', self.colors['accent_green'])])
        
        style.configure('Cyber.TEntry',
                       fieldbackground=self.colors['bg_secondary'],
                       foreground=self.colors['text_primary'],
                       bordercolor=self.colors['border'],
                       insertcolor=self.colors['accent_cyan'],
                       font=('Consolas', 9))
        
        style.configure('Cyber.TCheckbutton',
                       background=self.colors['bg_tertiary'],
                       foreground=self.colors['text_secondary'],
                       font=('Segoe UI', 10),
                       focuscolor='none')
        
        style.map('Cyber.TCheckbutton',
                 background=[('active', self.colors['bg_tertiary'])],
                 foreground=[('active', self.colors['accent_cyan'])])
        
    def create_glowing_frame(self, parent, **kwargs):
        """Create a frame with a subtle glow effect"""
        outer_frame = tk.Frame(parent, 
                              bg=self.colors['accent_cyan'], 
                              bd=1, 
                              relief='solid')
        inner_frame = tk.Frame(outer_frame, 
                              bg=self.colors['bg_tertiary'], 
                              bd=8,
                              **kwargs)
        inner_frame.pack(fill='both', expand=True)
        return outer_frame, inner_frame
        
    def setup_ui(self):
        # Main container with padding
        main_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header section with cyber theme
        header_frame = tk.Frame(main_container, bg=self.colors['bg_primary'])
        header_frame.pack(fill='x', pady=(0, 20))
        
        # Title with cyber styling
        title_frame = tk.Frame(header_frame, bg=self.colors['bg_primary'])
        title_frame.pack()
        
        title_label = tk.Label(title_frame,
                              text="CYBERGUARD AI",  # Removed shield emoji
                              bg=self.colors['bg_primary'],
                              fg=self.colors['accent_cyan'],
                              font=('Consolas', 24, 'bold'))
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame,
                                 text="Advanced Threat Classification System",
                                 bg=self.colors['bg_primary'],
                                 fg=self.colors['text_secondary'],
                                 font=('Segoe UI', 12))
        subtitle_label.pack(pady=(0, 5))
        
        # Status indicator
        self.status_indicator = tk.Label(title_frame,
                                        text="SYSTEM OFFLINE",  # Removed dot emoji
                                        bg=self.colors['bg_primary'],
                                        fg=self.colors['accent_red'],
                                        font=('Consolas', 10, 'bold'))
        self.status_indicator.pack()
        
        # Main content area with scrollable canvas
        canvas_frame = tk.Frame(main_container, bg=self.colors['bg_primary'])
        canvas_frame.pack(fill='both', expand=True)
        
        canvas = tk.Canvas(canvas_frame, bg=self.colors['bg_primary'], highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_primary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Model loading section
        model_outer, model_frame = self.create_glowing_frame(scrollable_frame)
        model_outer.pack(fill='x', pady=(0, 20))
        
        # Section header
        header_frame = tk.Frame(model_frame, bg=self.colors['bg_tertiary'])
        header_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(header_frame,
                text="AI MODEL CONFIGURATION",  # Removed wrench emoji
                bg=self.colors['bg_tertiary'],
                fg=self.colors['accent_cyan'],
                font=('Consolas', 14, 'bold')).pack(side='left')
        
        tk.Label(header_frame,
                text="Load your trained models for threat analysis",
                bg=self.colors['bg_tertiary'],
                fg=self.colors['text_muted'],
                font=('Segoe UI', 9)).pack(side='right')
        
        # Model selection grid
        models_info = [
            ("LSTM Neural Network", "Deep learning model for sequence analysis", "lstm"),
            ("BERT Transformer", "Bidirectional encoder for context understanding", "bert"),
            ("Random Forest", "Ensemble classifier for robust predictions", "rf"),
            ("Text Augmenter", "Optional data augmentation module", "augmenter")
        ]
        
        self.path_vars = {}
        
        for i, (name, desc, model_type) in enumerate(models_info):
            row_frame = tk.Frame(model_frame, bg=self.colors['bg_tertiary'])
            row_frame.pack(fill='x', pady=8)
            row_frame.columnconfigure(1, weight=1)
            
            # Model info
            info_frame = tk.Frame(row_frame, bg=self.colors['bg_tertiary'])
            info_frame.grid(row=0, column=0, sticky='w', padx=(0, 15))
            
            tk.Label(info_frame,
                    text=name,
                    bg=self.colors['bg_tertiary'],
                    fg=self.colors['text_primary'],
                    font=('Segoe UI', 11, 'bold')).pack(anchor='w')
            
            tk.Label(info_frame,
                    text=desc,
                    bg=self.colors['bg_tertiary'],
                    fg=self.colors['text_muted'],
                    font=('Segoe UI', 8)).pack(anchor='w')
            
            # Path entry
            self.path_vars[model_type] = tk.StringVar()
            path_entry = tk.Entry(row_frame,
                                 textvariable=self.path_vars[model_type],
                                 bg=self.colors['bg_primary'],  # Changed from bg_secondary for better contrast
                                 fg=self.colors['text_primary'],
                                 font=('Consolas', 9),
                                 bd=1,
                                 relief='solid')
            path_entry.grid(row=0, column=1, sticky='ew', padx=(0, 10))
            
            # Browse button
            browse_btn = tk.Button(row_frame,
                                  text="Browse",  # Removed folder emoji
                                  bg=self.colors['bg_secondary'],
                                  fg=self.colors['accent_cyan'],
                                  font=('Segoe UI', 10),
                                  bd=1,
                                  relief='solid',
                                  width=8,  # Increased width for text
                                  command=lambda mt=model_type: self.browse_model(mt))
            browse_btn.grid(row=0, column=2)
        
        # Load button and status
        control_frame = tk.Frame(model_frame, bg=self.colors['bg_tertiary'])
        control_frame.pack(fill='x', pady=(15, 0))
        
        self.load_button = tk.Button(control_frame,
                                    text="INITIALIZE AI MODELS",  # Removed rocket emoji
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['accent_green'],
                                    font=('Consolas', 12, 'bold'),
                                    bd=2,
                                    relief='solid',
                                    pady=8,
                                    command=self.load_models)
        self.load_button.pack(pady=(0, 10))
        
        self.status_var = tk.StringVar(value="Awaiting model configuration...") # Removed red circle emoji
        self.status_label = tk.Label(control_frame,
                                    textvariable=self.status_var,
                                    bg=self.colors['bg_tertiary'],
                                    fg=self.colors['text_muted'],
                                    font=('Consolas', 9))
        self.status_label.pack()
        
        # Input section
        input_outer, input_frame = self.create_glowing_frame(scrollable_frame)
        input_outer.pack(fill='x', pady=(0, 20))
        
        # Input header
        input_header = tk.Frame(input_frame, bg=self.colors['bg_tertiary'])
        input_header.pack(fill='x', pady=(0, 15))
        
        tk.Label(input_header,
                text="THREAT ANALYSIS INPUT",  # Removed memo emoji
                bg=self.colors['bg_tertiary'],
                fg=self.colors['accent_cyan'],
                font=('Consolas', 14, 'bold')).pack(side='left')
        
        tk.Label(input_header,
                text="Enter suspicious text for classification",
                bg=self.colors['bg_tertiary'],
                fg=self.colors['text_muted'],
                font=('Segoe UI', 9)).pack(side='right')
        
        # Text input with cyber styling
        text_frame = tk.Frame(input_frame, bg=self.colors['bg_secondary'], bd=1, relief='solid')
        text_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(text_frame,
                text="INPUT TEXT:",
                bg=self.colors['bg_secondary'],
                fg=self.colors['accent_cyan'],
                font=('Consolas', 9, 'bold')).pack(anchor='w', padx=8, pady=(8, 0))
        
        self.text_input = tk.Text(text_frame,
                                 height=6,
                                 bg=self.colors['bg_primary'],  # Changed to primary bg for better contrast
                                 fg=self.colors['text_primary'],
                                 font=('Consolas', 10),
                                 bd=1,  # Added border
                                 relief='solid',
                                 wrap=tk.WORD,
                                 insertbackground=self.colors['accent_cyan'])
        self.text_input.pack(fill='x', padx=8, pady=(0, 8))
        
        # Options and controls
        options_frame = tk.Frame(input_frame, bg=self.colors['bg_tertiary'])
        options_frame.pack(fill='x', pady=(0, 15))
        
        self.use_augmentation = tk.BooleanVar(value=False)
        aug_check = tk.Checkbutton(options_frame,
                                  text="Enable text augmentation (if available)",  # Removed lightning emoji
                                  variable=self.use_augmentation,
                                  bg=self.colors['bg_tertiary'],
                                  fg=self.colors['text_secondary'],
                                  selectcolor=self.colors['bg_secondary'],
                                  activebackground=self.colors['bg_tertiary'],
                                  activeforeground=self.colors['accent_cyan'],
                                  font=('Segoe UI', 10))
        aug_check.pack(side='left')
        
        # Action buttons
        action_frame = tk.Frame(input_frame, bg=self.colors['bg_tertiary'])
        action_frame.pack(fill='x')
        
        self.predict_button = tk.Button(action_frame,
                                       text="ANALYZE THREAT",  # Removed magnifying glass emoji
                                       bg=self.colors['bg_secondary'],
                                       fg=self.colors['accent_green'],
                                       font=('Consolas', 12, 'bold'),
                                       bd=2,
                                       relief='solid',
                                       pady=8,
                                       state='disabled',
                                       command=self.predict)
        self.predict_button.pack(side='left', padx=(0, 10))
        
        clear_button = tk.Button(action_frame,
                                text="CLEAR",  # Removed trash emoji
                                bg=self.colors['bg_secondary'],
                                fg=self.colors['accent_orange'],
                                font=('Consolas', 10, 'bold'),
                                bd=2,
                                relief='solid',
                                pady=8,
                                command=self.clear_results)
        clear_button.pack(side='left')
        
        # Results section
        results_outer, results_frame = self.create_glowing_frame(scrollable_frame)
        results_outer.pack(fill='both', expand=True)
        
        # Results header
        results_header = tk.Frame(results_frame, bg=self.colors['bg_tertiary'])
        results_header.pack(fill='x', pady=(0, 20))
        
        tk.Label(results_header,
                text="THREAT CLASSIFICATION RESULTS",  # Removed chart emoji
                bg=self.colors['bg_tertiary'],
                fg=self.colors['accent_cyan'],
                font=('Consolas', 14, 'bold')).pack(side='left')
        
        tk.Label(results_header,
                text="AI model predictions and confidence scores",
                bg=self.colors['bg_tertiary'],
                fg=self.colors['text_muted'],
                font=('Segoe UI', 9)).pack(side='right')
        
        # Results grid with modern cards
        results_grid = tk.Frame(results_frame, bg=self.colors['bg_tertiary'])
        results_grid.pack(fill='both', expand=True)
        
        models = [
            ("LSTM", "Neural Network", "LSTM"),
            ("BERT", "Transformer", "BERT"), 
            ("Random Forest", "Ensemble", "Random Forest")
        ]
        
        self.result_vars = {}
        self.confidence_vars = {}
        self.result_frames = {}
        
        for i, (icon_name, subtitle, model_key) in enumerate(models):
            # Create result card
            card_frame = tk.Frame(results_grid, 
                                 bg=self.colors['bg_secondary'], 
                                 bd=1, 
                                 relief='solid')
            card_frame.pack(fill='x', pady=8)
            
            self.result_frames[model_key] = card_frame
            
            # Card header
            header = tk.Frame(card_frame, bg=self.colors['bg_secondary'])
            header.pack(fill='x', padx=15, pady=(12, 8))
            
            tk.Label(header,
                    text=icon_name,
                    bg=self.colors['bg_secondary'],
                    fg=self.colors['text_primary'],
                    font=('Consolas', 12, 'bold')).pack(side='left')
            
            tk.Label(header,
                    text=subtitle,
                    bg=self.colors['bg_secondary'],
                    fg=self.colors['text_muted'],
                    font=('Segoe UI', 9)).pack(side='left', padx=(8, 0))
            
            # Status indicator for this model
            status_dot = tk.Label(header,
                                 text="OFF",  # Replaced circle emoji with text
                                 bg=self.colors['bg_secondary'],
                                 fg=self.colors['text_muted'],
                                 font=('Segoe UI', 10, 'bold'))
            status_dot.pack(side='right')
            
            # Prediction display
            pred_frame = tk.Frame(card_frame, bg=self.colors['bg_secondary'])
            pred_frame.pack(fill='x', padx=15, pady=(0, 12))
            
            # Prediction result
            self.result_vars[model_key] = tk.StringVar(value="Awaiting analysis...") # Removed hourglass emoji
            pred_label = tk.Label(pred_frame,
                                 textvariable=self.result_vars[model_key],
                                 bg=self.colors['bg_secondary'],
                                 fg=self.colors['text_secondary'],
                                 font=('Consolas', 11),
                                 justify='left')
            pred_label.pack(anchor='w')
            
            # Confidence score
            self.confidence_vars[model_key] = tk.StringVar(value="")
            conf_label = tk.Label(pred_frame,
                                 textvariable=self.confidence_vars[model_key],
                                 bg=self.colors['bg_secondary'],
                                 fg=self.colors['text_muted'],
                                 font=('Consolas', 9))
            conf_label.pack(anchor='w', pady=(4, 0))
        
        # Bind mouse wheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def browse_model(self, model_type):
        if model_type == "lstm":
            file_path = filedialog.askopenfilename(
                title="Select LSTM Model File",
                filetypes=[("Keras files", "*.keras"), ("All files", "*.*")]
            )
            if file_path:
                self.path_vars['lstm'].set(file_path)
                self.lstm_path = file_path
                
        elif model_type == "bert":
            folder_path = filedialog.askdirectory(
                title="Select BERT Model Folder"
            )
            if folder_path:
                self.path_vars['bert'].set(folder_path)
                self.bert_path = folder_path
                
        elif model_type == "rf":
            file_path = filedialog.askopenfilename(
                title="Select Random Forest Model File",
                filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
            )
            if file_path:
                self.path_vars['rf'].set(file_path)
                self.rf_path = file_path
                
        elif model_type == "augmenter":
            file_path = filedialog.askopenfilename(
                title="Select Augmenter File (Optional)",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if file_path:
                self.path_vars['augmenter'].set(file_path)
                self.aug_path = file_path
    
    def update_system_status(self, status, color):
        """Update the main system status indicator"""
        status_texts = {
            'offline': "SYSTEM OFFLINE",
            'loading': "INITIALIZING...",
            'online': "SYSTEM ONLINE",
            'analyzing': "ANALYZING THREAT..."
        }
        
        colors = {
            'offline': self.colors['accent_red'],
            'loading': self.colors['accent_orange'],
            'online': self.colors['accent_green'],
            'analyzing': self.colors['accent_cyan']
        }
        
        self.status_indicator.config(
            text=status_texts.get(status, status),
            fg=colors.get(color, self.colors['text_muted'])
        )
    
    def load_models(self):
        """Load all models in a separate thread to prevent UI freezing"""
        def load_thread():
            try:
                self.root.after(0, lambda: self.update_system_status('loading', 'loading'))
                self.root.after(0, lambda: self.status_var.set("Initializing AI systems..."))
                self.root.after(0, lambda: self.load_button.config(state="disabled"))
                
                models_loaded = 0
                total_models = 3
                
                # Load LSTM
                if self.lstm_path and os.path.exists(self.lstm_path):
                    self.root.after(0, lambda: self.status_var.set("Loading LSTM Neural Network..."))
                    self.lstm_model = tf.keras.models.load_model(self.lstm_path)
                    models_loaded += 1
                    self.root.after(0, lambda: self.status_var.set(f"LSTM loaded ({models_loaded}/{total_models})"))
                
                # Load BERT
                if self.bert_path and os.path.exists(self.bert_path):
                    self.root.after(0, lambda: self.status_var.set("Loading BERT Transformer..."))
                    self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
                    self.bert_model = AutoModelForSequenceClassification.from_pretrained(self.bert_path)
                    self.bert_model.eval()
                    models_loaded += 1
                    self.root.after(0, lambda: self.status_var.set(f"BERT loaded ({models_loaded}/{total_models})"))
                
                # Load Random Forest
                if self.rf_path and os.path.exists(self.rf_path):
                    self.root.after(0, lambda: self.status_var.set("Loading Random Forest..."))
                    self.rf_model = joblib.load(self.rf_path)
                    models_loaded += 1
                    self.root.after(0, lambda: self.status_var.set(f"Random Forest loaded ({models_loaded}/{total_models})"))
                
                # Load Augmenter (Optional)
                if hasattr(self, 'aug_path') and self.aug_path and os.path.exists(self.aug_path):
                    try:
                        self.root.after(0, lambda: self.status_var.set("Loading text augmenter..."))
                        with open(self.aug_path, 'rb') as f:
                            self.augmenter = pickle.load(f)
                        self.root.after(0, lambda: self.status_var.set("Augmenter loaded successfully!"))
                    except Exception as e:
                        self.root.after(0, lambda: self.status_var.set(f"Augmenter failed: {str(e)}"))
                        print(f"Augmenter loading failed: {e}")
                
                if models_loaded > 0:
                    self.root.after(0, lambda: self.update_system_status('online', 'online'))
                    self.root.after(0, lambda: self.status_var.set(f"System ready! {models_loaded}/{total_models} AI models active"))
                    self.root.after(0, lambda: self.predict_button.config(state="normal"))
                    
                    # Update predict button appearance
                    self.root.after(0, lambda: self.predict_button.config(
                        bg=self.colors['accent_green'],
                        fg=self.colors['bg_primary']
                    ))
                else:
                    self.root.after(0, lambda: self.update_system_status('offline', 'offline'))
                    self.root.after(0, lambda: self.status_var.set("No models loaded. Check file paths."))
                    
            except Exception as e:
                error_msg = f"Initialization failed: {str(e)}"
                self.root.after(0, lambda: self.status_var.set(error_msg))
                self.root.after(0, lambda: self.update_system_status('offline', 'offline'))
                self.root.after(0, lambda: messagebox.showerror("System Error", f"Failed to initialize AI models:\n{str(e)}"))
            finally:
                self.root.after(0, lambda: self.load_button.config(state="normal"))
        
        # Start loading in separate thread
        thread = threading.Thread(target=load_thread)
        thread.daemon = True
        thread.start()
    
    def clear_results(self):
        """Clear all prediction results"""
        for model_name in ["LSTM", "BERT", "Random Forest"]:
            self.result_vars[model_name].set("Awaiting analysis...")
            self.confidence_vars[model_name].set("")
            # Reset card colors
            if model_name in self.result_frames:
                self.result_frames[model_name].config(bg=self.colors['bg_secondary'])
        
        self.text_input.delete("1.0", tk.END)
    
    def update_result_display(self, model_name, prediction_result):
        """Update the display with prediction results"""
        if 'error' in prediction_result:
            self.result_vars[model_name].set(f"ERROR: {prediction_result.get('error', 'Unknown error')}")
            self.confidence_vars[model_name].set("")
            # Set error color
            if model_name in self.result_frames:
                self.result_frames[model_name].config(bg='#2d1b1b')  # Dark red tint
        else:
            confidence = prediction_result['confidence']
            
            # Determine threat level color and styling
            if prediction_result['class'] == 1:  # Threat detected
                if confidence > 0.8:
                    threat_color = self.colors['accent_red']
                    threat_text = "HIGH THREAT"
                    card_bg = '#2d1b1b'  # Dark red background
                elif confidence > 0.6:
                    threat_color = self.colors['accent_orange']
                    threat_text = "MEDIUM THREAT"
                    card_bg = '#2d251b'  # Dark orange background
                else:
                    threat_color = self.colors['warning']
                    threat_text = "LOW THREAT"
                    card_bg = '#2d2b1b'  # Dark yellow background
            else:  # Benign
                threat_color = self.colors['accent_green']
                threat_text = "BENIGN"
                card_bg = '#1b2d1b'  # Dark green background
            
            self.result_vars[model_name].set(f"{threat_text} (Class {prediction_result['class']})")
            
            # Format confidence with visual indicator
            conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            self.confidence_vars[model_name].set(f"Confidence: {confidence:.3f} [{conf_bar}]")
            
            # Update card background color
            if model_name in self.result_frames:
                self.result_frames[model_name].config(bg=card_bg)
            
            # Show augmentation status if it was applied
            if prediction_result.get('augmented', False):
                current_text = self.result_vars[model_name].get()
                self.result_vars[model_name].set(f"{current_text} [AUG]")
    
    def apply_augmentation(self, text):
        """Apply augmentation to input text if augmenter is loaded and enabled"""
        if self.use_augmentation.get() and self.augmenter is not None:
            try:
                # Apply augmentation - this depends on your augmenter type
                # Common nlpaug usage patterns:
                augmented_text = self.augmenter.augment(text)
                
                # If augmenter returns a list, take the first element
                if isinstance(augmented_text, list):
                    augmented_text = augmented_text[0]
                
                print(f"Original text: {text}")
                print(f"Augmented text: {augmented_text}")
                
                return augmented_text, True  # Return text and augmentation flag
            except Exception as e:
                print(f"Augmentation failed: {e}")
                return text, False  # Return original text if augmentation fails
        else:
            return text, False  # Return original text if augmentation not enabled
    
    def predict(self):
        """Make predictions with all loaded models"""
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Input Required", "Please enter text to analyze for threats.")
            return
        
        def predict_thread():
            try:
                # Update system status
                self.root.after(0, lambda: self.update_system_status('analyzing', 'analyzing'))
                
                # Clear previous results
                self.root.after(0, self.clear_results)
                
                # LSTM Prediction
                if self.lstm_model is not None:
                    try:
                        self.root.after(0, lambda: self.result_vars["LSTM"].set("Analyzing with LSTM..."))
                        lstm_pred = self.predict_lstm(text)
                        self.root.after(0, lambda: self.update_result_display("LSTM", lstm_pred))
                        print(f"LSTM prediction: {lstm_pred}")
                    except Exception as e:
                        error_result = {'error': str(e)}
                        self.root.after(0, lambda: self.update_result_display("LSTM", error_result))
                        print(f"LSTM error: {e}")
                else:
                    self.root.after(0, lambda: self.result_vars["LSTM"].set("Model not loaded"))
                
                # BERT Prediction
                if self.bert_model is not None and self.bert_tokenizer is not None:
                    try:
                        self.root.after(0, lambda: self.result_vars["BERT"].set("Analyzing with BERT..."))
                        bert_pred = self.predict_bert(text)
                        self.root.after(0, lambda: self.update_result_display("BERT", bert_pred))
                        print(f"BERT prediction: {bert_pred}")
                    except Exception as e:
                        error_result = {'error': str(e)}
                        self.root.after(0, lambda: self.update_result_display("BERT", error_result))
                        print(f"BERT error: {e}")
                else:
                    self.root.after(0, lambda: self.result_vars["BERT"].set("Model not loaded"))
                
                # Random Forest Prediction
                if self.rf_model is not None:
                    try:
                        self.root.after(0, lambda: self.result_vars["Random Forest"].set("Analyzing with Random Forest..."))
                        rf_pred = self.predict_rf(text)
                        self.root.after(0, lambda: self.update_result_display("Random Forest", rf_pred))
                        print(f"Random Forest prediction: {rf_pred}")
                    except Exception as e:
                        error_result = {'error': str(e)}
                        self.root.after(0, lambda: self.update_result_display("Random Forest", error_result))
                        print(f"Random Forest error: {e}")
                else:
                    self.root.after(0, lambda: self.result_vars["Random Forest"].set("Model not loaded"))
                
                # Update system status back to online
                self.root.after(0, lambda: self.update_system_status('online', 'online'))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Analysis Error", f"Threat analysis failed:\n{str(e)}"))
                self.root.after(0, lambda: self.update_system_status('online', 'online'))
        
        # Start prediction in separate thread
        thread = threading.Thread(target=predict_thread)
        thread.daemon = True
        thread.start()
    
    def predict_lstm(self, text):
        """Predict using LSTM model with proper tokenization and padding"""
        try:
            # Apply augmentation if enabled
            processed_text, was_augmented = self.apply_augmentation(text)
            
            if not hasattr(self, 'lstm_tokenizer') or self.lstm_tokenizer is None:
                # Try to load the tokenizer
                tokenizer_path = 'Tokenizers/lstm_tokenizer_bezaug.pkl'
                if os.path.exists(tokenizer_path):
                    with open(tokenizer_path, 'rb') as file:
                        self.lstm_tokenizer = pickle.load(file)
                    print(f"LSTM tokenizer loaded from '{tokenizer_path}'")
                else:
                    raise FileNotFoundError(f"LSTM tokenizer not found at '{tokenizer_path}'")
            
            # Tokenize and pad the input text
            sequences = self.lstm_tokenizer.texts_to_sequences([processed_text])
            
            # Use the same max_length as during training
            max_length = 100  # ADJUST THIS TO YOUR TRAINING MAX_LENGTH
            padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
                sequences, 
                maxlen=max_length
            )
            
            # Make prediction
            prediction = self.lstm_model.predict(padded_sequences, verbose=0)
            prob = float(prediction[0][0]) if prediction.shape[1] == 1 else float(np.max(prediction[0]))
            predicted_class = 1 if prob > 0.5 else 0
            
            return {
                'class': predicted_class,
                'confidence': prob if predicted_class == 1 else 1 - prob,
                'augmented': was_augmented
            }
        except Exception as e:
            return {'error': f"LSTM analysis failed: {str(e)}"}
    
    def predict_bert(self, text):
        """Predict using BERT model"""
        try:
            # Apply augmentation if enabled
            processed_text, was_augmented = self.apply_augmentation(text)
            
            # Tokenize input
            inputs = self.bert_tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Make prediction
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                probabilities = softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = float(torch.max(probabilities).item())
            
            return {
                'class': predicted_class,
                'confidence': confidence,
                'augmented': was_augmented
            }
        except Exception as e:
            return {'error': f"BERT analysis failed: {str(e)}"}
    
    def predict_rf(self, text):
        """Predict using Random Forest with TF-IDF vectorization"""
        try:
            # Apply augmentation if enabled
            processed_text, was_augmented = self.apply_augmentation(text)
            
            if not hasattr(self, 'tfidf_vectorizer') or self.tfidf_vectorizer is None:
                # Try to load the vectorizer
                vectorizer_path = 'Tokenizers/random_forest_vectorizer.joblib'
                if os.path.exists(vectorizer_path):
                    self.tfidf_vectorizer = joblib.load(vectorizer_path)
                    print(f"TF-IDF vectorizer loaded from '{vectorizer_path}'")
                else:
                    raise FileNotFoundError(f"TF-IDF vectorizer not found at '{vectorizer_path}'")
            
            # Transform the input text using TF-IDF
            text_tfidf = self.tfidf_vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = self.rf_model.predict(text_tfidf)
            probabilities = self.rf_model.predict_proba(text_tfidf)
            
            predicted_class = int(prediction[0])
            confidence = float(np.max(probabilities[0]))
            
            return {
                'class': predicted_class,
                'confidence': confidence,
                'augmented': was_augmented
            }
        except Exception as e:
            return {'error': f"Random Forest analysis failed: {str(e)}"}


def main():
    root = tk.Tk()
    app = CyberThreatClassifierApp(root)
    
    # Add window icon if available (optional)
    try:
        # You can add an icon file here
        # root.iconbitmap('cyber_icon.ico')
        pass
    except:
        pass
    
    # Make window resizable but set minimum size
    root.minsize(900, 700)
    
    # Center window on screen - FIXED VERSION
    root.update_idletasks()
    width = 1000  # Use the actual geometry width
    height = 800  # Use the actual geometry height
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()