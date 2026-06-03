# balance_topics.py

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Hardcoded environment tracking for the DePaul University Master's Project root workspace
BASE_DIR  = "C:/Users/sabal/OneDrive - DePaul University/Courses/Spr 2026/Project/fake-news-detection"
PROC_DIR  = os.path.join(BASE_DIR, "data/processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

print("Executing Technique 2: Stratification & Topic Balancing...")

# ==============================================================================
# 1. LOAD CORPUS AND LDA DOMINANT TOPIC MAPPINGS
# Syncing primary data structures with unsupervised probabilistic assignments
# ==============================================================================
df = pd.read_csv(os.path.join(PROC_DIR, "cleaned_news.csv"))
topics_df = pd.read_csv("results/document_topic_assignments.csv")

# Inject the latent topic categorical index into the primary text dataframe
df["dominant_topic"] = topics_df["dominant_topic"]

print("\nOriginal Distribution of Dominant Topics in the FAKE News Pool:")
print(df[df["label"] == 0]["dominant_topic"].value_counts().sort_index())

# ==============================================================================
# 2. STRATEGIC DOWNSAMPLING TO MITIGATE TOPICAL CONFOUNDING (SHORTCUT LEARNING)
# Normalizing conversational commentary overrepresentation (Topic 1) in Fake news
# to force the classifiers to study stylometric structure instead of macro-themes.
# ==============================================================================
# Establish a baseline sample constraint count using Topic 0 (Civil Unrest & Crimes)
fake_topic_0_count = sum((df["label"] == 0) & (df["dominant_topic"] == 0))

# Segment the Fake class into non-confounded topics and the confounded campaign topic
fake_non_topic1 = df[(df["label"] == 0) & (df["dominant_topic"] != 1)]
fake_topic1 = df[(df["label"] == 0) & (df["dominant_topic"] == 1)]

# Sub-sample the overrepresented Topic 1 entries down to match the Topic 0 baseline ceiling
fake_topic1_downsampled = fake_topic1.sample(n=fake_topic_0_count, random_state=42)

# Recombine the adjusted, un-confounded fake category dataframes
df_fake_balanced = pd.concat([fake_non_topic1, fake_topic1_downsampled])

# Extract the true/verified article partition pool
df_real = df[df["label"] == 1]

# Balance overall target class representation to enforce an absolute 50/50 balance split
min_class_size = min(len(df_fake_balanced), len(df_real))
df_final = pd.concat([
    df_fake_balanced.sample(n=min_class_size, random_state=42),
    df_real.sample(n=min_class_size, random_state=42)
]).reset_index(drop=True) # Reset the index array to fix continuity after structural slicing

print("\n--- Adjusted Unbiased Dataset Summary ---")
print(f"Total entries: {len(df_final):,}")
print("Class Distribution:")
print(df_final["label"].value_counts().rename({0: "Fake", 1: "Real"}))
print("Topic distribution across remaining Fake pool:")
print(df_final[df_final["label"] == 0]["dominant_topic"].value_counts().sort_index())

# ==============================================================================
# 3. GENERATE CROSS-TOPIC STRATIFIED TRAIN/TEST SPLITS
# Combines label maps and topic keys to eliminate structural cross-leakage
# ==============================================================================
# Formulate a multi-variable stratification key array string (e.g. '0_1', '1_4')
df_final["stratify_key"] = df_final["label"].astype(str) + "_" + df_final["dominant_topic"].astype(str)

X = df_final["content"].astype(str)
y = df_final["label"]

# Execute a highly rigorous split, guaranteeing uniform label/topic distributions in train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=df_final["stratify_key"]
)

# ==============================================================================
# 4. TF-IDF VECTORIZATION VIA UNBIASED N-GRAM FEATURES
# Initializing feature space extraction over training bounds with zero look-ahead bias
# ==============================================================================
print("\nVectorizing balanced text features via TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=20000,   # Set vocabulary limits at top tokens to bound sparse array footprints
    ngram_range=(1, 2),   # Extract both isolated unigrams and sequential bigram framing metrics
    min_df=5,             # Suppress hyper-rare vocabulary tokens appearing in fewer than 5 docs
    max_df=0.9            # Drop redundant dictionary filler appearing across more than 90% of corpus
)

# Fit vocabulary configurations over train data only; transform test data to maintain leakage barriers
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# ==============================================================================
# 5. SERIALIZE OUTPUT DATA MATRICES FOR MACHINE LEARNING TIERS
# Overwrites processed files to seamlessly update train_models.py and evaluate.py
# ==============================================================================
pickle.dump(vectorizer,    open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb"))
pickle.dump(X_train_tfidf, open(os.path.join(PROC_DIR,  "X_train.pkl"),          "wb"))
pickle.dump(X_test_tfidf,  open(os.path.join(PROC_DIR,  "X_test.pkl"),           "wb"))
pickle.dump(y_train,       open(os.path.join(PROC_DIR,  "y_train.pkl"),          "wb"))
pickle.dump(y_test,        open(os.path.join(PROC_DIR,  "y_test.pkl"),           "wb"))
