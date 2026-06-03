# predict.py

import pickle
import re
import string
import argparse
import os
import sys

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

CUSTOM_STOP = {
    "reuters", "ap", "afp", "bloomberg",
    "said", "say", "would", "could", "also",
    "even", "still", "well",
    "one", "two", "three",
    "via", "video", "image", "photo",
    "share", "watch", "click", "read",
    "post", "article",
}

stop_words = set(stopwords.words("english")) | CUSTOM_STOP

lemmatizer = WordNetLemmatizer()

LABEL_MAP = {0: "FAKE", 1: "REAL"}
LABEL_COLOR = {0: "🔴 FAKE", 1: "🟢 REAL"}


# =====================================================
# TEXT PREPROCESSING  (mirrors preprocess.py exactly)
# Any change to preprocess.py must be reflected here
# =====================================================

def remove_dateline(text):
    text = re.sub(r"^\s*\([^)]{1,40}\)\s*[-–—]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[\w\s,\.]{1,60}\([^)]{1,40}\)\s*[-–—]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(reuters|associated press|ap|afp|bloomberg)\s*[-–—]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def clean_text(text):
    text = str(text)
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    text = remove_dateline(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_text(text, max_sentences=15):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:max_sentences])

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith("V"):
        return wordnet.VERB

    elif treebank_tag.startswith("J"):
        return wordnet.ADJ

    elif treebank_tag.startswith("R"):
        return wordnet.ADV

    else:
        return wordnet.NOUN


def lemmatize_text(text):

    tokens = word_tokenize(text)

    tagged = pos_tag(tokens)

    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged
    ]

    return " ".join(lemmatized)


def preprocess(text):
    text = clean_text(text)
    text = truncate_text(text)
    text = lemmatize_text(text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text


# =====================================================
# LOAD ARTIFACTS
# =====================================================

def load_artifacts(model_path, vectorizer_path):
    if not os.path.exists(model_path):
        sys.exit(f"[ERROR] Model file not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        sys.exit(f"[ERROR] Vectorizer file not found: {vectorizer_path}")

    print(f"Loading model       : {model_path}")
    model = pickle.load(open(model_path, "rb"))

    print(f"Loading vectorizer  : {vectorizer_path}")
    vectorizer = pickle.load(open(vectorizer_path, "rb"))

    return model, vectorizer


# =====================================================
# CONFIDENCE LABEL
# =====================================================

def confidence_label(prob):
    if prob >= 0.90:
        return "Very High"
    elif prob >= 0.75:
        return "High"
    elif prob >= 0.60:
        return "Moderate"
    else:
        return "Low"


# =====================================================
# SINGLE PREDICTION
# =====================================================

def predict_single(model, vectorizer, raw_text, verbose=True):
    cleaned = preprocess(raw_text)

    if not cleaned.strip():
        return {"error": "Text is empty after preprocessing."}

    X    = vectorizer.transform([cleaned])
    pred = int(model.predict(X)[0])

    result = {
        "prediction"      : pred,
        "label"           : LABEL_MAP[pred],
        "cleaned_text"    : cleaned,
    }

    # probability / confidence
    if hasattr(model, "predict_proba"):
        probs             = model.predict_proba(X)[0]
        result["prob_fake"]    = round(float(probs[0]), 4)
        result["prob_real"]    = round(float(probs[1]), 4)
        result["confidence"]   = round(float(max(probs)), 4)
        result["confidence_level"] = confidence_label(result["confidence"])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        result["decision_score"] = round(score, 4)
        # convert to rough confidence via sigmoid
        sig  = 1 / (1 + np.exp(-abs(score)))
        result["confidence"]       = round(sig, 4)
        result["confidence_level"] = confidence_label(sig)
    else:
        result["confidence"]       = None
        result["confidence_level"] = "Unknown"

    if verbose:
        print("\n" + "="*55)
        print("  PREDICTION RESULT")
        print("="*55)
        print(f"  Verdict     : {LABEL_COLOR[pred]}")
        print(f"  Prediction  : {pred}  ({LABEL_MAP[pred]})")
        if "prob_fake" in result:
            print(f"  Prob Fake   : {result['prob_fake']:.4f}  ({result['prob_fake']*100:.1f}%)")
            print(f"  Prob Real   : {result['prob_real']:.4f}  ({result['prob_real']*100:.1f}%)")
        elif "decision_score" in result:
            print(f"  Decision    : {result['decision_score']:.4f}  (>0 = Real, <0 = Fake)")
        print(f"  Confidence  : {result['confidence']:.4f}  [{result['confidence_level']}]")
        print(f"\n  Input (first 120 chars):")
        print(f"    {raw_text[:120]}{'...' if len(raw_text) > 120 else ''}")
        print(f"\n  After preprocessing (first 120 chars):")
        print(f"    {cleaned[:120]}{'...' if len(cleaned) > 120 else ''}")
        print("="*55)

    return result


# =====================================================
# BATCH PREDICTION
# =====================================================

def predict_batch(model, vectorizer, input_csv, text_column, output_path):
    if not os.path.exists(input_csv):
        sys.exit(f"[ERROR] Input CSV not found: {input_csv}")

    print(f"Reading CSV : {input_csv}")
    df = pd.read_csv(input_csv)

    if text_column not in df.columns:
        available = ", ".join(df.columns.tolist())
        sys.exit(f"[ERROR] Column '{text_column}' not found.\nAvailable columns: {available}")

    total = len(df)
    print(f"Articles to predict : {total:,}")

    raw_texts  = df[text_column].astype(str).tolist()
    cleaned    = [preprocess(t) for t in raw_texts]

    # filter out empty after cleaning
    empty_mask = [c.strip() == "" for c in cleaned]
    n_empty    = sum(empty_mask)
    if n_empty:
        print(f"[WARNING] {n_empty} rows became empty after preprocessing — will be labelled -1")

    X    = vectorizer.transform(cleaned)
    preds = model.predict(X).astype(int)
    preds[empty_mask] = -1   # flag empties

    df["prediction"]   = preds
    df["label"]        = df["prediction"].map({0: "FAKE", 1: "REAL", -1: "EMPTY"})
    df["cleaned_text"] = cleaned

    # probabilities / confidence
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        df["prob_fake"]   = np.round(probs[:, 0], 4)
        df["prob_real"]   = np.round(probs[:, 1], 4)
        df["confidence"]  = np.round(probs.max(axis=1), 4)
        df["confidence_level"] = df["confidence"].apply(confidence_label)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        df["decision_score"] = np.round(scores, 4)
        df["confidence"]     = np.round(1 / (1 + np.exp(-np.abs(scores))), 4)
        df["confidence_level"] = df["confidence"].apply(confidence_label)

    # summary
    counts = df["label"].value_counts()
    print("\nBatch prediction summary:")
    print(f"  {'FAKE':<8}: {counts.get('FAKE', 0):,}  ({counts.get('FAKE', 0)/total*100:.1f}%)")
    print(f"  {'REAL':<8}: {counts.get('REAL', 0):,}  ({counts.get('REAL', 0)/total*100:.1f}%)")
    if n_empty:
        print(f"  {'EMPTY':<8}: {n_empty:,}")
    if "confidence" in df.columns:
        print(f"\n  Avg confidence : {df['confidence'].mean():.4f}")
        print(f"  Low-conf rows  : {(df['confidence'] < 0.60).sum():,}  (confidence < 0.60)")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to : {output_path}")

    return df


# =====================================================
# CLI
# =====================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Fake News Prediction",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "batch"],
        required=True,
        help="single  : predict one article from --text\n"
             "batch   : predict a CSV file from --input_csv"
    )
    parser.add_argument("--text",         type=str, help="Raw article text (single mode)")
    parser.add_argument("--input_csv",    type=str, help="Path to CSV file (batch mode)")
    parser.add_argument("--text_column",  type=str, default="text",
                        help="Column name for article text in CSV (default: 'text')")
    parser.add_argument("--model_path",   type=str, default="models/best_model.pkl",
                        help="Path to trained model (default: models/best_model.pkl)")
    parser.add_argument(
                        "--vectorizer_path",
                        type=str,
                        default="models/tfidf_vectorizer.pkl",
                        help="Path to TF-IDF vectorizer (default: models/tfidf_vectorizer.pkl)"
                    )
    parser.add_argument("--output_path",  type=str, default="results/predictions.csv",
                        help="Output CSV path for batch mode (default: results/predictions.csv)")
    parser.add_argument("--quiet",        action="store_true",
                        help="Suppress verbose output in single mode")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    model, vectorizer = load_artifacts(args.model_path, args.vectorizer_path)

    if args.mode == "single":
        if not args.text:
            parser.error("--text is required for single mode")
        predict_single(model, vectorizer, args.text, verbose=not args.quiet)

    elif args.mode == "batch":
        if not args.input_csv:
            parser.error("--input_csv is required for batch mode")
        predict_batch(
            model, vectorizer,
            args.input_csv,
            args.text_column,
            args.output_path
        )


if __name__ == "__main__":
    main()