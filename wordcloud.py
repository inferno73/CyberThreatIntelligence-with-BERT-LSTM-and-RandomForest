import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("your_dataset.csv")  # Adjust filename
text_col = "text"
label_col = "label"  # 1 = Relevant, 0 = Not Relevant

# Split the data
relevant_texts = df[df[label_col] == 1][text_col]
not_relevant_texts = df[df[label_col] == 0][text_col]

# Use the same vocabulary for both
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
vectorizer.fit(df[text_col])

# Transform separately
tfidf_rel = vectorizer.transform(relevant_texts)
tfidf_nonrel = vectorizer.transform(not_relevant_texts)

# Get average TF-IDF per word in each class
avg_tfidf_rel = tfidf_rel.mean(axis=0).A1
avg_tfidf_nonrel = tfidf_nonrel.mean(axis=0).A1
vocab = np.array(vectorizer.get_feature_names_out())

# Compare scores
diff = avg_tfidf_rel - avg_tfidf_nonrel

# Create class-specific word dictionaries
rel_words = {word: diff[i] for i, word in enumerate(vocab) if diff[i] > 0}
nonrel_words = {word: -diff[i] for i, word in enumerate(vocab) if diff[i] < 0}

# Optional: Keep only top 100
rel_words = dict(sorted(rel_words.items(), key=lambda x: x[1], reverse=True)[:100])
nonrel_words = dict(sorted(nonrel_words.items(), key=lambda x: x[1], reverse=True)[:100])

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

wc_rel = WordCloud(width=800, height=800, background_color='white', colormap='Greens')
wc_rel.generate_from_frequencies(rel_words)
axes[0].imshow(wc_rel, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('Relevant Words', fontsize=22)

wc_nonrel = WordCloud(width=800, height=800, background_color='white', colormap='Reds')
wc_nonrel.generate_from_frequencies(nonrel_words)
axes[1].imshow(wc_nonrel, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Not Relevant Words', fontsize=22)

plt.tight_layout()
plt.show()