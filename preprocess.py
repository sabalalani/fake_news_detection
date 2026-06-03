# preprocess.py

import os
import re
import string
import pickle
import pandas as pd

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

# ==============================================================================
# NLTK CORE DEPENDENCIES SETUP
# Silent initialization to prevent runtime console logging clutter during loops
# ==============================================================================
nltk.download("wordnet",     quiet=True) # Downloads the WordNet lexical database semantic network
nltk.download("stopwords",   quiet=True) # Downloads standard high-frequency grammatical filler words
nltk.download("omw-1.4",     quiet=True) # Open Multilingual WordNet maps to prevent reduction errors
nltk.download("punkt",       quiet=True) # Fundamental unsupervised point-distribution sentence tokenizer
nltk.download("punkt_tab",   quiet=True) # Tabular formatting lookups for sentence tokenization maps
nltk.download("averaged_perceptron_tagger",     quiet=True) # Part-of-Speech tagging predictive matrix
nltk.download("averaged_perceptron_tagger_eng", quiet=True) # Language-specific English mapping constants

# ==============================================================================
# TECHNIQUE 1: EXPANDED DOMAIN-SPECIFIC STOPWORDS
# Strategic mitigation to strip out structural confounding "shortcuts"[cite: 5].
# Forces classifiers to evaluate text framing/style rather than memorizing names[cite: 5].
# ==============================================================================
POLITICAL_SHORTCUT_STOPS = {
    "trump", "donald", "clinton", "hillary", "obama", "barack", 
    "republican", "republicans", "democrat", "democrats", "democratic",
    "president", "presidents", "presidential", "campaign", "campaigns",
    "government", "state", "states", "house", "senate", "court", "bill"
}

# Source identity exclusions to eliminate raw news-wire fingerprint confounding
NEWS_WIRE_STOPS = {"reuters", "ap", "afp", "bloomberg", "said", "say", "would", "could", "also"}
# Domain numerical literals that evade standard digit strip regex passes
FILLER_STOPS = {"even", "still", "well", "one", "two", "three"}
# Social platform clickbait framing markers that artificially weight class spaces
CLICKBAIT_STOPS = {"via", "video", "image", "photo", "share", "watch", "click", "read", "post", "article"}

# Compile unified set matching standard English stopwords with all custom tokens
stop_words = set(stopwords.words("english")) | POLITICAL_SHORTCUT_STOPS | NEWS_WIRE_STOPS | FILLER_STOPS | CLICKBAIT_STOPS
lemmatizer = WordNetLemmatizer()

# Hardcoded absolute path directory tracking for DePaul University master's capstone repository[cite: 5]
BASE_DIR  = "C:/Users/sabal/OneDrive - DePaul University/Courses/Spr 2026/Project/fake-news-detection"
RAW_DIR   = os.path.join(BASE_DIR, "data/raw")
PROC_DIR  = os.path.join(BASE_DIR, "data/processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Guarantee target folder architecture exists prior to saving output pkl matrices[cite: 5]
os.makedirs(PROC_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================================================================
# NOISE SEPARATION FILTERS (DATA INTEGRITY CLEANING UTILITIES)
# ==============================================================================

def is_url_only(text):
    """Flags short text bodies containing raw web address scraping artifacts."""
    text = str(text).strip()
    return text.startswith("http") and len(text.split()) < 5

def is_image_row(text):
    """Identifies broken database records that point only to graphics filenames."""
    return ".jpg" in text or ".png" in text or ".jpeg" in text

def is_html_junk(text):
    """Detects residual WordPress shortcodes, page-builders, or raw code blocks."""
    patterns = [r"\[vc_row", r"\[td_block", r"\[\/vc_", r"wpengine", r"td_block"]
    return any(re.search(p, text) for p in patterns)

def is_navigation_junk(text):
    """Isolates web rendering page errors like short broken site home links."""
    return len(text.split()) < 5 and "homepage" in text.lower()

def is_entity_dump(text):
    """Filters massive metadata blocks or multi-name directory tables."""
    words = text.split()
    if len(words) > 600: return True
    org_keywords = ["foundation", "organization", "center", "institute", "committee", "project"]
    return sum(text.count(k) for k in org_keywords) > 15

def remove_dateline(text):
    """
    CRITICAL CLEAN: Strips starting parenthetical publisher markers (e.g. 'WASHINGTON (Reuters) -').
    Prevents classifiers from trivially memorizing real agency tags instead of linguistic signals[cite: 5].
    """
    # Pattern 1: Starting agency inside parentheses
    text = re.sub(r"^\s*\([^)]{1,40}\)\s*[-–—]\s*", "", text, flags=re.IGNORECASE)
    # Pattern 2: Location prefix alongside starting agency marker
    text = re.sub(r"^[\w\s,\.]{1,60}\([^)]{1,40}\)\s*[-–—]\s*", "", text, flags=re.IGNORECASE)
    # Pattern 3: Unbracketed starter news wire names
    text = re.sub(r"^(reuters|associated press|ap|afp|bloomberg)\s*[-–—]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()

def clean_text(text):
    """Main normalization pipeline: handles encoding, datelines, cases, and characters."""
    text = str(text)
    # Enforce safe utf-8 formatting strings to eliminate invalid scraped symbols
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    text = remove_dateline(text)  # Must run BEFORE lowercasing to preserve regex bounds
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text) # Expunge mid-text hyperlinks
    text = re.sub(r"\d+", "", text)            # Strip numeric values to clean alpha space
    # Remove punctuation maps using character mapping translations
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Standardize whitespace variations to uniform single blanks
    text = re.sub(r"\s+", " ", text).strip()
    return text

def truncate_text(text, max_sentences=15):
    """Bounds maximum document length to prevent long outliers from skewing vectors[cite: 5]."""
    sentences = sent_tokenize(text)
    return " ".join(sentences[:max_sentences])

def get_wordnet_pos(treebank_tag):
    """Maps dynamic Penn Treebank POS markers into stationary WordNet constant categories."""
    if treebank_tag.startswith("V"): return wordnet.VERB
    elif treebank_tag.startswith("J"): return wordnet.ADJ
    elif treebank_tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN

def lemmatize_text(text):
    """
    POS-AWARE LEMMATIZATION ENGINE: Prevents verbal truncation issues (e.g., 'was' -> 'wa').
    Guarantees verbs reduce cleanly to base forms ('was'->'be'), enabling stopword capture[cite: 5].
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens) # Returns list of tuples matching [("word", "POS_TAG")]
    # Apply context reduction loop mapping tag variables explicitly
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]
    return " ".join(lemmatized)

# ==============================================================================
# PIPELINE STREAM EXECUTION SEQUENCE
# ==============================================================================

print("Loading raw data...")
fake = pd.read_csv(os.path.join(RAW_DIR, "Fake.csv"))
true = pd.read_csv(os.path.join(RAW_DIR, "True.csv"))

# Assign explicit ground truth target labels (0 = Fraudulent, 1 = Verified)[cite: 5]
fake["label"] = 0
true["label"] = 1
# Combine text sets structurally
df = pd.concat([fake, true], ignore_index=True)
# Concat titles with body elements to analyze structural headlines syntax[cite: 5]
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

# Filter operational noise elements across features
df["is_bad"] = df["content"].apply(lambda x: is_url_only(x) or is_image_row(x) or is_html_junk(x) or is_navigation_junk(x))
df = df[~df["is_bad"]].drop(columns=["is_bad"])

print("Executing cleaning pipeline with expanded debiased stopwords...")
df["content"] = df["content"].apply(clean_text)        # Step 1: Text normalization passes
df = df[~df["content"].apply(is_entity_dump)]          # Step 2: Clear entity structural dumps[cite: 5]
df["content"] = df["content"].apply(truncate_text)     # Step 3: Sentence window constraints[cite: 5]
df["content"] = df["content"].apply(lemmatize_text)    # Step 4: POS-aware vocabulary morphs[cite: 5]

# Step 5: Final stopword filter loop targeting the custom political unigrams[cite: 5]
df["content"] = df["content"].apply(lambda x: " ".join([w for w in x.split() if w not in stop_words]))
# Remove any residual documents that became entirely blank post-filtration loop[cite: 5]
df = df[df["content"].str.strip().str.len() > 0]

print(f"Final unbiased dataset size: {len(df):,}")

# Save the primary balanced, filtered file out to disk partition paths[cite: 5]
df.to_csv(os.path.join(PROC_DIR, "cleaned_news.csv"), index=False, encoding="utf-8-sig")
print("Saved: cleaned_news.csv")