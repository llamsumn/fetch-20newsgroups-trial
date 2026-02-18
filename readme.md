# NLPipe: Text Classification Pipeline

A modular Python implementation for comparing the effects of **Stemming** vs. **Lemmatization** on text classification performance. This notebook utilizes the `nltk` library for preprocessing and `scikit-learn` for logistic regression modeling.

---

## 🚀 Overview
This project builds a custom `NLP` class that handles text normalization and preprocessing. It specifically targets four technical and recreational categories from the **20 Newsgroups** dataset to evaluate how different text reduction techniques impact model accuracy.

## 🛠 The Pipeline
The `NLP` class processes raw text through several modular stages:
* **Normalization:** Converts all text to lowercase.
* **Punctuation & Noise Removal:** Removes punctuation, tabs, newlines, and hidden characters.
* **Tokenization:** Breaks text into individual words using `nltk.word_tokenize`.
* **Stopword Removal:** Filters out common English stopwords.
* **Numeric & Short Token Filtering:** Removes words containing digits and tokens shorter than two characters.
* **Text Reduction:**
    * **Porter Stemmer:** Aggressively chops word suffixes (e.g., "running" to "run").
    * **WordNet Lemmatizer:** Uses Parts of Speech (POS) tagging to reduce words to their dictionary root.

## 📊 Dataset & Features
The model focuses on the following newsgroups:
* `rec.motorcycles`
* `rec.sport.hockey`
* `sci.electronics`
* `sci.space`

The processed text is transformed into a **Binary Feature Vector** based on the vocabulary built from the training data.

## 📈 Performance Results
The pipeline compares the Stemmed and Lemmatised models using Logistic Regression:

| Metric | Stemmed Model | Lemmatised Model |
| :--- | :--- | :--- |
| **Vocabulary Size** | 24,336 | 27,455 |
| **Test Accuracy** | 0.82 | 0.83 |
| **CV Accuracy (Mean)** | 83.29% | 83.04% |

**Observation:** Lemmatization results in a larger feature dimension (difference of 3,119 words), but Stemming showed a slightly higher cross-validation accuracy by **0.25%**.

## ⚙️ Requirements
To run this notebook, ensure you have the following installed:
* **Python 3.x**
* **Libraries:** `numpy`, `nltk`, `scikit-learn`
* **NLTK Data:** `punkt`, `wordnet`, `omw-1.4`, `averaged_perceptron_tagger`
    
