# AI Cyber Threat Intelligence — NLP Classification Project

This repository contains the implementation of a project focused on **automated detection of cyber threats in social media content** using NLP and machine learning techniques.

> This is a part of my university VI (final year) project on AI for Cyber Threat Intelligence.

## 🔍 Objective

The goal of the project is to **detect threatening textual content** in posts from social media platforms. Such posts often serve as early warning signals in cybersecurity. The classifier distinguishes between:
- `0`: Non-threatening (neutral) posts
- `1`: Threatening posts

## 📁 Dataset

The project uses a **merged dataset from the LADDER Social Media Threat Corpus**, which includes labeled social media posts.

- Total posts: ~5000
- Class labels: `0` (non-threat), `1` (threat)
- Format: CSV with two columns: `text`, `label`

To address class imbalance, **data augmentation** was applied using contextual BERT-based word substitution on threat-class examples.

## ⚙️ Models Implemented

The following models were trained, tuned, and evaluated:

1. **BERT fine-tuned (HuggingFace Transformers)**
2. **LSTM neural networks** (4 variants with different architecture tweaks)
3. **Random Forest classifier**

Each model was evaluated in two iterations:
- **Iteration 1**: Using **augmented** dataset (threat class expanded)
- **Iteration 2**: Using **original** dataset without augmentation

## Preprocessing

### For LSTM & Random Forest:
- Text normalization
- Tokenization (Keras tokenizer)
- Padding (for LSTM)
- Optional: class weighting to handle imbalance
- Augmentation via [`nlpaug`](https://github.com/makcedward/nlpaug): BERT-based substitutions

### For BERT:
- Preprocessing handled via HuggingFace tokenizer
- Fine-tuned using `bert-base-uncased`

## 📊 Evaluation Metrics

- **Accuracy**
- **F1-score**
- **Precision & Recall**
- **Confusion Matrix**

The focus was placed on improving F1-score for the **threat class** (`1`), as correctly identifying potential threats is more critical than false positives.

## 📈 Results Summary

| Model        | Accuracy | F1 (Threat) | F1 (Macro) |
|--------------|----------|-------------|------------|
| LSTM (aug)   | 0.90     | 0.90        | 0.90       |
| LSTM (no aug)| 0.89     | 0.84        | 0.88       |
| BERT         | 0.91     | 0.91        | 0.91       |
| RF           | 0.83     | 0.74        | 0.80       |

**Best balance** between precision and recall on threat detection was achieved with **BERT** and **LSTM with augmentation**.

##  Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
