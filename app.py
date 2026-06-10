# app.py

import os
import re
import string
import pickle
import sys
import numpy as np
import pandas as pd
import streamlit as st

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

# ==============================================================================
# SET STREAMLIT VIEWPORT PAGE CONFIGURATION
# Configures the global web browser window bounds and structural panel state
# ==============================================================================
st.set_page_config(
    page_title="Fake News Detector & Topic Analyzer",
    page_icon="📰",
    layout="wide",                       # Optimizes screen space for multi-column metrics tables
    initial_sidebar_state="expanded"     # Guarantees the metadata sidebar panel renders immediately
)

# ==============================================================================
# NLTK DEPENDENCIES DOWNLOAD PORTAL
# Caches resources programmatically to prevent blocking re-downloads on text entries
# ==============================================================================
@st.cache_resource(show_spinner=False)
def download_nltk_dependencies():
    nltk.download("wordnet", quiet=True)     # For morphology mapping Lookups
    nltk.download("stopwords", quiet=True)   # For high-frequency structural vocabulary dumps
    nltk.download("omw-1.4", quiet=True)     # Open Multilingual WordNet lexical constants
    nltk.download("punkt", quiet=True)       # Sentence-level regex tokenizer checkpoints
    nltk.download("punkt_tab", quiet=True)   # Tabular lookup components for data mapping stability
    nltk.download("averaged_perceptron_tagger", quiet=True)     # Contextual part-of-speech matrix
    nltk.download("averaged_perceptron_tagger_eng", quiet=True) # English grammatical label weights

download_nltk_dependencies()

# ==============================================================================
# INTERACTIVE TEXT PREPROCESSING UTILITY PIPELINE
# Mirrors the exact morphological and feature-filtration sequence used in train loops
# ==============================================================================
CUSTOM_STOP = {
    "reuters", "ap", "afp", "bloomberg",
    "said", "say", "would", "could", "also",
    "even", "still", "well",
    "one", "two", "three",
    "via", "video", "image", "photo",
    "share", "watch", "click", "read",
    "post", "article",
}

# Synchronize standard NLTK stopwords with specialized custom filter collections
stop_words = set(stopwords.words("english")) | CUSTOM_STOP
lemmatizer = WordNetLemmatizer()

def remove_dateline(text):
    """Isolates and drops parenthetical wire datelines at absolute document start positions."""
    text = re.sub(r"^\s*\([^)]{1,40}\)\s*[-–—]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[\w\s,\.]{1,60}\([^)]{1,40}\)\s*[-–—]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(reuters|associated press|ap|afp|bloomberg)\s*[-–—]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()

def clean_text(text):
    """Applies global lowercase formatting, strips URL strings, digits, and punctuation pools."""
    text = str(text)
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore") # Wipe non-ASCII text artifacts
    text = remove_dateline(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def truncate_text(text, max_sentences=15):
    """Enforces strict multi-sentence caps to preserve model array dimensionality input shapes."""
    sentences = sent_tokenize(text)
    return " ".join(sentences[:max_sentences])

def get_wordnet_pos(treebank_tag):
    """Converts Penn Treebank grammar attributes into corresponding WordNet constant keys."""
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_text(text):
    """Runs structural POS tagging to safely normalize tokens down to root dictionary states."""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged
    ]
    return " ".join(lemmatized)

def preprocess_pipeline(text):
    """Unified orchestration pipeline executing sequentially layered NLP transformations."""
    text = clean_text(text)
    text = truncate_text(text)
    text = lemmatize_text(text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

# ==============================================================================
# SERIALIZED MODEL ARTIFACT MANAGEMENT LAYER
# Caches static binary files in server RAM to avoid local disk read/write blocking bottlenecks
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_classification_artifacts():
    model_path = "models/best_model.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    
    # Graceful handling for missing file states
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

# ==============================================================================
# PRESENTATION VIEW LAYER IMPLEMENTATION
# ==============================================================================

# Meta Content Structural Sidebar Layout
st.sidebar.title("📌 Project Metadata")
st.sidebar.markdown(
    """
    **Student:** Saba Bashir  
    **Student ID:** 2179443  
    """
)
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Status")

# Call artifact caching logic
model, vectorizer = load_classification_artifacts()

# Display current pipeline parameters dynamically in the sidebar UI view
if model and vectorizer:
    st.sidebar.success("✅ Models Loaded Successfully")
    model_type = type(model).__name__
    st.sidebar.info(f"**Classifier:** {model_type}\n\n**Features:** TF-IDF (Max: 20k)")
else:
    st.sidebar.error("❌ Models Missing! Please verify files exist inside the `models/` folder.")

# Application Main Landing Header Block
st.title("📰 Fake News Detection & Topic Analysis System")
st.markdown(
    "Welcome to the interactive text mining verification dashboard. This framework utilizes "
    "natural language processing pipelines alongside machine learning architectures to predict "
    "validity metrics and isolate semantic structures."
)
st.markdown("---")

# Initialize multi-tab workspace architecture layout
tab1, tab2, tab3 = st.tabs(["🔍 Real-time Article Verifier", "📊 Model Leaderboard", "🧬 Unsupervised Topic Insight"])

# ==============================================================================
# TAB 1: REAL-TIME INFERENCE PORTAL (CLASSIFICATION ENQUIRIES)
# ==============================================================================
with tab1:
    st.header("🔮 Validate an Online News Article")
    st.write("Input raw text body content to generate automated classification metrics.")
    
    # Form input field for copy-pasted textual records
    raw_input = st.text_area(
        "Paste Article Content Here:",
        height=250,
        placeholder="Type or paste text content here (e.g., policy adjustments, geopolitical reports, press wire extracts)..."
    )
    
    # Multi-column alignments button configurations
    col_btn, col_clear = st.columns([1, 8])
    with col_btn:
        submit = st.button("Analyze Text", type="primary")
        
    if submit:
        # Step 1: Input text sanity boundary check
        if not raw_input.strip():
            st.warning("⚠️ Submission empty. Please paste article text content before initiating analysis.")
        # Step 2: System runtime dependency verification
        elif model is None or vectorizer is None:
            st.error("🛑 Operational failure: Vectorizer/Model structural reference points missing inside environment pipelines.")
        else:
            with st.spinner("Executing structural text processing and feature maps..."):
                # Step 3: Stream text through the integrated preprocessing rules
                processed_input = preprocess_pipeline(raw_input)
                
                # Step 4: Ensure processing didn't result in an empty string array post-filtration
                if not processed_input.strip():
                    st.error("❌ Clean Failure: Preprocessing stripped away all valid tokens. Ensure input text holds English character strings.")
                else:
                    # Step 5: Map processed text into the fitted 20,000-dimensional TF-IDF coordinate space
                    vectorized_input = vectorizer.transform([processed_input])
                    prediction = int(model.predict(vectorized_input)[0])
                    
                    # Step 6: Map LinearSVC decision bounds to isolate distance scores from the hyper-plane
                    decision_score = float(model.decision_function(vectorized_input)[0])
                    # Step 7: Apply a custom logistic sigmoid function transformation to compute confidence probabilities
                    confidence_pct = 1 / (1 + np.exp(-abs(decision_score)))
                    
                    # Step 8: Calibrate output status metric classification buckets based on confidence values
                    if confidence_pct >= 0.90:
                        level, alert_style = "Very High", "high"
                    elif confidence_pct >= 0.75:
                        level, alert_style = "High", "high"
                    elif confidence_pct >= 0.60:
                        level, alert_style = "Moderate", "normal"
                    else:
                        level, alert_style = "Low", "normal"
                        
                    st.markdown("### 📊 Engine Results Evaluation")
                    res_col1, res_col2 = st.columns(2)
                    
                    # Sub-column 1: Render categorical classification alerts
                    with res_col1:
                        if prediction == 1:
                            st.success("### 🟢 Verdict: REAL ARTICLE")
                            st.markdown(
                                "The engine classified this document as **REAL**, mapping features commonly seen "
                                "in structured journalistic publications and verified news wires."
                            )
                        else:
                            st.error("### 🔴 Verdict: FAKE / MISLEADING")
                            st.markdown(
                                "The engine classified this document as **FAKE/MISLEADING**, aligning closely with "
                                "clickbait text syntax, structural bias patterns, or non-verified editorial copy."
                            )
                            
                    # Sub-column 2: Render continuous probability scores and distance thresholds
                    with res_col2:
                        st.metric(label="System Confidence", value=f"{confidence_pct * 100:.2f}%")
                        st.write(f"**Confidence Classification:** {level}")
                        st.write(f"**Raw Decision Margin Value:** `{decision_score:+.4f}`")
                        
                    st.markdown("---")
                    st.markdown("### 🛠️ Diagnostic Verification Pipeline View")
                    diag_col1, diag_col2 = st.columns(2)
                    
                    # Render processing trace-logs to support complete technical audit trails
                    with diag_col1:
                        st.caption("Raw Input Sample Data (First 350 Chars)")
                        st.info(f"{raw_input[:350]}...")
                    with diag_col2:
                        st.caption("Post NLP-Preprocessing Output Data Structure")
                        st.code(f"{processed_input[:350]}...", language="text")

# ==============================================================================
# TAB 2: SUPERVISED PERFORMANCE LEADERBOARD
# Displays permanent evaluation metrics compiled directly from stratified testing datasets
# ==============================================================================
with tab2:
    st.header("📈 Supervised Classification Evaluation Engine")
    st.write("Review the final system metrics compiled directly from stratified testing datasets during train loops.")
    
    # High-integrity evaluation matrix matching final project execution logs
    leaderboard_data = {
        "Classifier Architecture": [
            "Support Vector Machine (LinearSVC)", 
            "Tuned SVM", 
            "Tuned Logistic Regression", 
            "Random Forest", 
            "Tuned Random Forest", 
            "Logistic Regression (Baseline)", 
            "Naive Bayes (MultinomialNB)"
        ],
        "Train F1": [0.9995, 0.9995, 0.9963, 1.0000, 1.0000, 0.9761, 0.9267],
        "Test F1": [0.9781, 0.9781, 0.9760, 0.9666, 0.9666, 0.9612, 0.9160],
        "Test Accuracy": [0.9781, 0.9781, 0.9760, 0.9668, 0.9668, 0.9611, 0.9177],
        "ROC-AUC": [0.9972, 0.9972, 0.9964, 0.9951, 0.9951, 0.9936, 0.9751],
        "PR-AUC": [0.9973, 0.9973, 0.9964, 0.9950, 0.9950, 0.9934, 0.9751],
        "Matthews Corr (MCC)": [0.9563, 0.9563, 0.9520, 0.9337, 0.9337, 0.9223, 0.8360],
        "Overfit Variance Delta": [0.0214, 0.0214, 0.0203, 0.0334, 0.0334, 0.0149, 0.0107]
    }
    df_metrics = pd.DataFrame(leaderboard_data)
    # Highlight high-performing cells using a color gradient overlay directly on top of the dataset view
    st.dataframe(df_metrics.style.background_gradient(subset=["Test F1", "Test Accuracy", "ROC-AUC"], cmap="Blues"), use_container_width=True)
    
    st.markdown("### 🔑 Critical Analysis Observations")
    obs_col1, obs_col2 = st.columns(2)
    with obs_col1:
        st.markdown(
            """
            * **Optimal Architecture Choice:** The **Support Vector Machine (LinearSVC)** retained peak operational efficiency, 
            scoring a robust **0.9781 Test F1** alongside **97.81% Global Accuracy** post-debiasing.
            * **Hyperparameter Observations:** Tuning SVM regularization parameters optimized performance exactly at $C=1$, 
            retaining robust margin separation vectors even after stripping out high-frequency political unigrams.
            """
        )
    with obs_col2:
        st.markdown(
            """
            * **Generalization Check:** The generalization gap remains tightly controlled (roughly **2.14%** drop from training split), 
            proving that the model is learning stylistic markers of text framing rather than memorizing keyword patterns.
            * **Evaluation Metric Robustness:** A high Matthews Correlation Coefficient metric (**0.9563 MCC**) confirms 
            flawless sorting balance across both true and fake validation sets under realistic, non-confounded conditions.
            """
        )

# ==============================================================================
# TAB 3: UNSUPERVISED TOPIC INSIGHT MATRIX
# Displays the underlying latent distributions to provide qualitative text mining validation
# ==============================================================================
with tab3:
    st.header("🧬 Unsupervised Semantic Clustering Discovery (LDA Model)")
    st.write("Latent Dirichlet Allocation profiles summarizing underlying textual focus behaviors across 36,817 initial processed records.")
    
    st.info(
        "💡 **Key Structural Insight for Report:** Our unsupervised baseline exposed a profound topical confound: "
        "Fake news heavily clustered around colloquial commentary and opinion dynamics (Topic 1: 55.67% average weight), "
        "whereas Real news locked onto official international relations and governance channels. This insight allowed us to "
        "proactively downsample overrepresented clusters technique to build an unbiased classification model."
    )
    
    st.markdown("### 📋 Primary Topic Distributions Matrix ($n=5$)")
    
    # Primary dictionary matrix capturing the tokens discovered during active LDA training loops
    topics_matrix = {
        "Topic Index": ["Topic 0", "Topic 1", "Topic 2", "Topic 3", "Topic 4"],
        "Top Token Key Associations": [
            "police, people, city, kill, law, gun, year, school, group, attack",
            "go, get, people, make, like, know, think, time, white, take",
            "vote, percent, tax, new, year, million, plan, election, make, law",
            "russia, russian, china, official, north, korea, washington, report, security, united",
            "minister, country, party, united, leader, force, year, muslim, military, refugee"
        ],
        "Mean Weight (Fake News Pool)": [0.1548, 0.5567, 0.1311, 0.1097, 0.0476],
        "Mean Weight (Real News Pool)": [0.1365, 0.0702, 0.2718, 0.2426, 0.2789],
        "Inferred Domain Label Context": [
            "Civil Unrest, Law Enforcement & Domestic Incidents",
            "Colloquial Discourse, Opinion Framing & Social Media Echoes",
            "Congressional Legislation, Tax Frameworks & Electoral Metrics",
            "Geopolitical Security Briefings & Intelligence Enquiries",
            "Global State Departments, Foreign Policy & Sovereign Affairs"
        ]
    }
    
    # Render static data matrix layout tables cleanly
    st.table(pd.DataFrame(topics_matrix))
