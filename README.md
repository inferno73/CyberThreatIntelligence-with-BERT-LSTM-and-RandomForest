# AI Cyber Attack Pattern Detection — NLP Classification Project

This repository contains the implementation of a project focused on **automated detection of cyber attack patterns in textual content** using NLP and machine learning techniques.

> This is a part of my university VI (final year) project on AI for Cyber Attack Pattern Detection.

## 🔍 Objective

The goal of the project is to **detect cyber attack patterns in textual content** from various sources, particularly social media posts. Such patterns often serve as early warning signals for potential cyber threats and attacks. The classifier distinguishes between:
- `0`: Text without attack patterns (benign content)
- `1`: Text containing attack patterns (malicious indicators)

## 📁 Dataset

The project uses a **merged dataset based on the LADDER dataset**, which has been expanded with additional social media content (tweets) that don't contain attack patterns to create a more comprehensive dataset.

- Total instances: ~5000
- Class labels: `0` (no attack patterns), `1` (contains attack patterns)
- Format: CSV with two columns: `text`, `label`
- Sources: LADDER dataset + curated social media content

To address class imbalance, **data augmentation** was applied using contextual BERT-based word substitution on attack pattern examples (class 1) for some models.

## ⚙️ Models Implemented

The following models were trained, tuned, and evaluated:

1. **BERT fine-tuned (HuggingFace Transformers)**
2. **LSTM neural networks** (4 variants with different architecture tweaks)
3. **Random Forest classifier**


## 🚀 Getting Started with the Classification App

### Prerequisites

- Python 3.11 (recommended to use a virtual environment)
- Git

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd viproject
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r app_req.txt
   ```

4. **Run the classification app:**
   ```bash
   python classify_app.py
   ```

### Using the Application

1. **Launch the app** - The GUI will open with a cybersecurity-themed interface
2. **Load models** - Browse and select model files from the `Modeli` folder:
   - **Random Forest**: Select `.joblib` model file
   - **BERT**: Select the model directory containing the fine-tuned BERT model
   - **LSTM**: Select `.keras` model file (multiple variants available)
3. **Initialize models** - Click "INITIALIZE AI MODELS" to load the selected models
4. **Input text** - Enter the text you want to analyze for attack patterns
5. **Analyze** - Click "ANALYZE THREAT" to get predictions from all loaded models
6. **Review results** - Each model will provide:
   - Classification (Attack Pattern / Benign)
   - Confidence score
   - Visual threat level indicators

### Available Pre-trained Models

The `Modeli` folder contains:
- **Random Forest**: Trained with TF-IDF features and SMOTE oversampling
- **BERT**: Fine-tuned `bert-base-uncased` model for sequence classification
- **LSTM variants**: Multiple architectures including bidirectional and stacked LSTM models

## Preprocessing

### For LSTM & Random Forest:
- Text normalization (URL standardization, quote normalization, punctuation handling)
- Tokenization (Keras tokenizer for LSTM, TF-IDF for Random Forest)
- Padding (for LSTM)
- Optional: class weighting to handle imbalance
- Augmentation via [`nlpaug`](https://github.com/makcedward/nlpaug): BERT-based substitutions

### For BERT:
- Preprocessing handled via HuggingFace tokenizer
- Fine-tuned using `bert-base-uncased`
- Max sequence length: 128 tokens

## 📊 Evaluation Metrics

- **Accuracy**
- **F1-score**
- **Precision & Recall**
- **Confusion Matrix**

The focus was placed on improving F1-score for the **attack pattern class** (`1`), as correctly identifying potential attack indicators is more critical than false positives.

## 📈 Results Summary

| Model        | Accuracy | F1 (Attack) | F1 (Macro) |
|--------------|----------|-------------|------------|
| LSTM (aug)   | 0.90     | 0.90        | 0.90       |
| LSTM (no aug)| 0.89     | 0.84        | 0.88       |
| BERT         | 0.90     | 0.88        | 0.90       |
| RF           | 0.87     | 0.82        | 0.86       |


**Best balance** between precision and recall for attack pattern detection was achieved with **BERT** and **LSTM with augmentation**.
