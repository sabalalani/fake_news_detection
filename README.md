# 📰 Unbiased Fake News Detection & Latent Semantic Topic Analysis Pipeline

An end-to-end Natural Language Processing (NLP) and Machine Learning Knowledge Discovery in Databases (KDD) pipeline that programmatically detects fraudulent news copy while exposing and neutralizing systemic topical confounding (shortcut learning). This data science project features structural text debiasing, unsupervised topic profiling (LDA), high-dimensional latent clustering ($K$-Means), multi-model hyperparameter optimization, and a functional deployment interface layer built with Streamlit.

---

## 📌 Project Architecture & Metadata
* **Author:** Saba Bashir
* **Student ID:** 2179443
* **Course:** Master's Project (Data Science Capstone)
* **Institution:** School of Computing, DePaul University
* **Primary Champion Architecture:** Linear Support Vector Machine (`LinearSVC`)
* **Core Core Results Performance:** **0.9781 Test F1-Score** | **0.9563 Matthews Correlation Coefficient (MCC)**

---

## ⚙️ System Requirements & Environment Setup
This repository relies strictly on open-source scientific computing libraries. Follow these sequential steps to establish a clean environment block:

### 1. Platform Prerequisites
* Python SDK Plattform Interface `>= 3.9`
* Virtual Environment shell tool (`venv`)

### 2. Dependency Installation
Clone this repository to your local path workspace, open a terminal instance inside the root directory, and initialize the system environment:

```bash
# Initialize localized virtual environment shell
python -m venv venv

# Activate the virtual environment path
# For Windows PowerShell/CMD:
.\venv\Scripts\activate
# For macOS/Linux Terminal:
source venv/bin/activate

# Upgrade pip and ingest production dependencies
pip install --upgrade pip
pip install -r requirements.txt

#run code files
python preprocess.py          # Stage 1: Noise filtering, POS-lemmatization, & Technique 1 Stopwords
python topic_modeling.py       # Stage 2: Latent Dirichlet Allocation (LDA) Topic Profiling Discovery
python balance_topics.py       # Stage 3: Technique 2 Dataset Topic Stratification & Vectorization
python train_models.py         # Stage 4: Cross-Validation, GridSearchCV, & Model Training Loops
python evaluate.py             # Stage 5: Advanced Metric Calculations & Visualizations Compilation

#Repository Directory Index Map
.
├── data/
│   ├── raw/                           # Directory placeholder for input raw data CSV files
│   └── processed/                     # High-integrity pkl sparse matrices and balanced splits
├── models/
│   ├── best_model.pkl                 # Serialization of the optimized LinearSVC champion model
│   ├── tfidf_vectorizer.pkl           # Serialization of the 20,000-feature text vocabulary vectorizer
│   └── lda_n5.pkl                     # Caches the fixed 5-topic LDA allocations model
├── results/                           # Generated evaluation curve outputs and analytics sheets
│   ├── final_model_comparison.csv     # Compiled master metrics spreadsheet tracker
│   ├── roc_curves_all_models.png      # Comparative threshold curves plot
│   ├── overfitting_gap_heatmap.png    # Divergent color overfitting delta check plot
│   └── lda_fake_vs_real_topics.png    # Grouped bar chart tracking topic mixtures bias
├── preprocess.py                      # Tier-1 preprocessing script utility
├── topic_modeling.py                  # Tier-2 LDA text exploration script
├── balance_topics.py                  # Tier-3 structural data downsampling & stratification script
├── train_models.py                    # Tier-4 base/tuned architecture execution training loop
├── evaluate.py                        # Tier-5 diagnostic metrics visualization compilation script
├── app.py                             # Interactive application deployment presentation layer
└── requirements.txt                   # Ecosystem packages setup guidelines sheet

#Live Dashboard Interface Deployment
streamlit run app.py