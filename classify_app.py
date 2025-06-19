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


class ModelComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Binary Classification Model Comparison")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
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
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Binary Classification Model Comparison", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Model loading section
        model_frame = ttk.LabelFrame(main_frame, text="Load Models", padding="10")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        model_frame.columnconfigure(1, weight=1)
        
        # LSTM Model
        ttk.Label(model_frame, text="LSTM Model (.keras):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.lstm_path_var = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.lstm_path_var, state="readonly").grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=5)
        ttk.Button(model_frame, text="Browse", command=lambda: self.browse_model("lstm")).grid(row=0, column=2, padx=(5, 0), pady=5)
        
        # BERT Model
        ttk.Label(model_frame, text="BERT Model (folder):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.bert_path_var = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.bert_path_var, state="readonly").grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=5)
        ttk.Button(model_frame, text="Browse", command=lambda: self.browse_model("bert")).grid(row=1, column=2, padx=(5, 0), pady=5)
        
        # Random Forest Model
        ttk.Label(model_frame, text="Random Forest (.joblib):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.rf_path_var = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.rf_path_var, state="readonly").grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=5)
        ttk.Button(model_frame, text="Browse", command=lambda: self.browse_model("rf")).grid(row=2, column=2, padx=(5, 0), pady=5)
        
        # Augmenter (Optional)
        ttk.Label(model_frame, text="Augmenter (optional .pkl):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.aug_path_var = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.aug_path_var, state="readonly").grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=5)
        ttk.Button(model_frame, text="Browse", command=lambda: self.browse_model("augmenter")).grid(row=3, column=2, padx=(5, 0), pady=5)
        
        # Load button
        self.load_button = ttk.Button(model_frame, text="Load All Models", command=self.load_models)
        self.load_button.grid(row=4, column=0, columnspan=3, pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Select model files and click 'Load All Models'")
        self.status_label = ttk.Label(model_frame, textvariable=self.status_var, foreground="blue")
        self.status_label.grid(row=5, column=0, columnspan=3, pady=5)
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Text Input", padding="10")
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        input_frame.columnconfigure(0, weight=1)
        
        # Text input
        self.text_input = tk.Text(input_frame, height=5, width=70, wrap=tk.WORD)
        self.text_input.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Option to use augmentation
        self.use_augmentation = tk.BooleanVar(value=False)
        aug_checkbox = ttk.Checkbutton(input_frame, text="Apply augmentation to input text (if augmenter loaded)", 
                                      variable=self.use_augmentation)
        aug_checkbox.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        
        # Predict button
        self.predict_button = ttk.Button(input_frame, text="Predict", command=self.predict, state="disabled")
        self.predict_button.grid(row=2, column=0, pady=5)
        
        # Clear results button
        self.clear_button = ttk.Button(input_frame, text="Clear Results", command=self.clear_results)
        self.clear_button.grid(row=2, column=1, padx=(10, 0), pady=5)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Predictions", padding="15")
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        results_frame.columnconfigure(1, weight=1)
        results_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Results display with better formatting
        models = ["LSTM", "BERT", "Random Forest"]
        self.result_vars = {}
        self.confidence_vars = {}
        
        # Header row
        ttk.Label(results_frame, text="Model", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Label(results_frame, text="Prediction", font=("Arial", 12, "bold")).grid(row=0, column=1, sticky=tk.W, padx=(20, 0), pady=(0, 10))
        ttk.Label(results_frame, text="Confidence", font=("Arial", 12, "bold")).grid(row=0, column=2, sticky=tk.W, padx=(20, 0), pady=(0, 10))
        
        # Separator
        ttk.Separator(results_frame, orient='horizontal').grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        for i, model_name in enumerate(models):
            row = i + 2
            
            # Model name
            model_label = ttk.Label(results_frame, text=f"{model_name}:", font=("Arial", 11, "bold"))
            model_label.grid(row=row, column=0, sticky=tk.W, pady=8)
            
            # Prediction
            self.result_vars[model_name] = tk.StringVar(value="Not predicted")
            pred_label = ttk.Label(results_frame, textvariable=self.result_vars[model_name], 
                                  font=("Arial", 11), background="white", relief="sunken", padding=(5, 2))
            pred_label.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(20, 5), pady=8)
            
            # Confidence
            self.confidence_vars[model_name] = tk.StringVar(value="")
            conf_label = ttk.Label(results_frame, textvariable=self.confidence_vars[model_name], 
                                  font=("Arial", 10), foreground="gray", background="white", relief="sunken", padding=(5, 2))
            conf_label.grid(row=row, column=2, sticky=(tk.W, tk.E), padx=(5, 0), pady=8)
    
    def browse_model(self, model_type):
        if model_type == "lstm":
            file_path = filedialog.askopenfilename(
                title="Select LSTM Model File",
                filetypes=[("Keras files", "*.keras"), ("All files", "*.*")]
            )
            if file_path:
                self.lstm_path_var.set(file_path)
                self.lstm_path = file_path
                
        elif model_type == "bert":
            folder_path = filedialog.askdirectory(
                title="Select BERT Model Folder"
            )
            if folder_path:
                self.bert_path_var.set(folder_path)
                self.bert_path = folder_path
                
        elif model_type == "rf":
            file_path = filedialog.askopenfilename(
                title="Select Random Forest Model File",
                filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
            )
            if file_path:
                self.rf_path_var.set(file_path)
                self.rf_path = file_path
                
        elif model_type == "augmenter":
            file_path = filedialog.askopenfilename(
                title="Select Augmenter File (Optional)",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if file_path:
                self.aug_path_var.set(file_path)
                self.aug_path = file_path
    
    def load_models(self):
        """Load all models in a separate thread to prevent UI freezing"""
        def load_thread():
            try:
                self.root.after(0, lambda: self.status_var.set("Loading models..."))
                self.root.after(0, lambda: self.load_button.config(state="disabled"))
                
                models_loaded = 0
                total_models = 3
                
                # Load LSTM
                if self.lstm_path and os.path.exists(self.lstm_path):
                    self.root.after(0, lambda: self.status_var.set("Loading LSTM model..."))
                    self.lstm_model = tf.keras.models.load_model(self.lstm_path)
                    models_loaded += 1
                    self.root.after(0, lambda: self.status_var.set(f"Loaded {models_loaded}/{total_models} models - LSTM loaded"))
                
                # Load BERT
                if self.bert_path and os.path.exists(self.bert_path):
                    self.root.after(0, lambda: self.status_var.set("Loading BERT model..."))
                    self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
                    self.bert_model = AutoModelForSequenceClassification.from_pretrained(self.bert_path)
                    self.bert_model.eval()
                    models_loaded += 1
                    self.root.after(0, lambda: self.status_var.set(f"Loaded {models_loaded}/{total_models} models - BERT loaded"))
                
                # Load Random Forest
                if self.rf_path and os.path.exists(self.rf_path):
                    self.root.after(0, lambda: self.status_var.set("Loading Random Forest model..."))
                    self.rf_model = joblib.load(self.rf_path)
                    models_loaded += 1
                    self.root.after(0, lambda: self.status_var.set(f"Loaded {models_loaded}/{total_models} models - Random Forest loaded"))
                
                # Load Augmenter (Optional)
                if hasattr(self, 'aug_path') and self.aug_path and os.path.exists(self.aug_path):
                    try:
                        self.root.after(0, lambda: self.status_var.set("Loading Augmenter..."))
                        with open(self.aug_path, 'rb') as f:
                            self.augmenter = pickle.load(f)
                        self.root.after(0, lambda: self.status_var.set("Augmenter loaded successfully!"))
                    except Exception as e:
                        self.root.after(0, lambda: self.status_var.set(f"Warning: Could not load augmenter - {str(e)}"))
                        print(f"Augmenter loading failed: {e}")
                
                if models_loaded > 0:
                    self.root.after(0, lambda: self.status_var.set(f"Successfully loaded {models_loaded}/{total_models} models!"))
                    self.root.after(0, lambda: self.predict_button.config(state="normal"))
                else:
                    self.root.after(0, lambda: self.status_var.set("No models loaded. Please check file paths."))
                    
            except Exception as e:
                error_msg = f"Error loading models: {str(e)}"
                self.root.after(0, lambda: self.status_var.set(error_msg))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load models:\n{str(e)}"))
            finally:
                self.root.after(0, lambda: self.load_button.config(state="normal"))
        
        # Start loading in separate thread
        thread = threading.Thread(target=load_thread)
        thread.daemon = True
        thread.start()
    
    def clear_results(self):
        """Clear all prediction results"""
        for model_name in ["LSTM", "BERT", "Random Forest"]:
            self.result_vars[model_name].set("Not predicted")
            self.confidence_vars[model_name].set("")
    
    def update_result_display(self, model_name, prediction_result):
        """Update the display with prediction results"""
        if 'error' in prediction_result:
            self.result_vars[model_name].set(f"Error: {prediction_result.get('error', 'Unknown error')}")
            self.confidence_vars[model_name].set("")
        else:
            class_name = "Positive" if prediction_result['class'] == 1 else "Negative"
            self.result_vars[model_name].set(f"{class_name} (Class {prediction_result['class']})")
            self.confidence_vars[model_name].set(f"Confidence: {prediction_result['confidence']:.3f}")
            
            # Show augmentation status if it was applied
            if prediction_result.get('augmented', False):
                current_text = self.result_vars[model_name].get()
                self.result_vars[model_name].set(f"{current_text} [Augmented]")
    
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
            messagebox.showwarning("Warning", "Please enter some text to predict.")
            return
        
        def predict_thread():
            try:
                # Clear previous results
                self.root.after(0, self.clear_results)
                
                # LSTM Prediction
                if self.lstm_model is not None:
                    try:
                        self.root.after(0, lambda: self.result_vars["LSTM"].set("Predicting..."))
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
                        self.root.after(0, lambda: self.result_vars["BERT"].set("Predicting..."))
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
                        self.root.after(0, lambda: self.result_vars["Random Forest"].set("Predicting..."))
                        rf_pred = self.predict_rf(text)
                        self.root.after(0, lambda: self.update_result_display("Random Forest", rf_pred))
                        print(f"Random Forest prediction: {rf_pred}")
                    except Exception as e:
                        error_result = {'error': str(e)}
                        self.root.after(0, lambda: self.update_result_display("Random Forest", error_result))
                        print(f"Random Forest error: {e}")
                else:
                    self.root.after(0, lambda: self.result_vars["Random Forest"].set("Model not loaded"))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{str(e)}"))
        
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
            return {'error': f"LSTM prediction failed: {str(e)}"}
    
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
            return {'error': f"BERT prediction failed: {str(e)}"}
    
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
            return {'error': f"Random Forest prediction failed: {str(e)}"}


def main():
    root = tk.Tk()
    app = ModelComparisonApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()