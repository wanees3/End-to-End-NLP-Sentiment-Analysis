# End-to-End NLP Sentiment Analysis: TF-IDF vs GloVe vs Transformer Models

An end-to-end Natural Language Processing project for binary sentiment classification using IMDb movie reviews. The project compares traditional TF-IDF features, pre-trained GloVe word embeddings, and a transformer-based DistilBERT model.

## Project Overview

This project investigates how different text representation techniques affect sentiment classification performance.

The following approaches were evaluated:

1. **TF-IDF + Logistic Regression**
2. **GloVe Embeddings + Logistic Regression**
3. **DistilBERT Transformer**

The IMDb movie review dataset was used for binary sentiment classification:

* `0` → Negative
* `1` → Positive

## Research Question

> How does the choice of text representation affect the performance of sentiment classification models?

## Pipeline

```text
IMDb Reviews
     ↓
Text Preprocessing
     ↓
 ┌───────────────┬──────────────────┬────────────────┐
 │               │                  │                │
TF-IDF          GloVe            DistilBERT
 │               │                  │
Logistic        Logistic          Pre-trained
Regression      Regression        Transformer
 │               │                  │
 └───────────────┴──────────────────┴────────────────┘
                       ↓
                 Model Evaluation
```

## Dataset

The project uses the IMDb Movie Reviews dataset containing:

* 25,000 training reviews
* 25,000 testing reviews
* Binary sentiment labels
* Balanced positive and negative classes

## Text Preprocessing

The preprocessing pipeline includes:

* Lowercasing
* HTML tag removal
* URL removal
* Punctuation removal
* Tokenization
* Stop-word removal
* Lemmatization

Negation words such as `not`, `no`, `nor`, and `never` were retained because they can be important for sentiment classification.

### Example

**Before:**

```text
This movie was NOT good! I expected much more from it.
```

**After preprocessing:**

```text
movie not good expected much
```

## Feature Engineering

### TF-IDF

TF-IDF was used with:

* Maximum 10,000 features
* Unigrams and bigrams

The resulting representation was:

```text
Training: (25000, 10000)
Testing:  (25000, 10000)
```

### GloVe

Pre-trained 100-dimensional GloVe embeddings were used.

Each review was converted into a single 100-dimensional vector by averaging the word embeddings present in the review.

```text
Training: (25000, 100)
Testing:  (25000, 100)
```

## Models

### 1. TF-IDF + Logistic Regression

A traditional machine learning baseline using sparse TF-IDF features.

### 2. GloVe + Logistic Regression

A dense semantic representation using pre-trained GloVe embeddings followed by Logistic Regression.

### 3. DistilBERT

The pre-trained:

`distilbert-base-uncased-finetuned-sst-2-english`

model was used for sentiment classification.

The transformer was evaluated directly on the IMDb test set without additional fine-tuning on IMDb.

## Results

| Model                        |   Accuracy |  Precision |     Recall |   F1-Score |
| ---------------------------- | ---------: | ---------: | ---------: | ---------: |
| TF-IDF + Logistic Regression |     88.71% |     88.51% | **88.98%** |     88.74% |
| GloVe + Logistic Regression  |     79.90% |     80.25% |     79.30% |     79.78% |
| DistilBERT                   | **89.07%** | **91.46%** |     86.19% | **88.75%** |

## Key Findings

* **DistilBERT achieved the best overall accuracy:** 89.07%.
* **DistilBERT achieved the highest precision:** 91.46%.
* **TF-IDF achieved the highest recall:** 88.98%.
* **TF-IDF and DistilBERT had almost identical F1-scores.**
* **GloVe performed substantially lower** in this experiment.
* Mean-pooling GloVe embeddings removes word-order and contextual information, which can be important for sentiment classification.
* A well-designed traditional TF-IDF representation can remain highly competitive with transformer-based models.

## Conclusion

The experiment demonstrates that the choice of text representation has a significant impact on sentiment classification performance.

DistilBERT produced the strongest overall results, but TF-IDF with Logistic Regression achieved nearly the same F1-score with a much simpler approach. This highlights an important practical consideration in NLP: more complex models do not always provide a large performance improvement over strong traditional baselines.

## Technologies

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* GloVe
* Hugging Face Transformers
* PyTorch
* Matplotlib
* Seaborn
* Google Colab

## Project Structure

```text
End-to-End-NLP-Sentiment-Analysis/
│
├── README.md
├── End_to_End_NLP_Sentiment_Analysis.ipynb
├── requirements.txt
├── .gitignore
│
└── images/
    ├── class_distribution.png
    ├── tfidf_confusion_matrix.png
    ├── glove_confusion_matrix.png
    ├── distilbert_confusion_matrix.png
    └── model_comparison.png
```

## How to Run

The notebook can be run using Google Colab or a local Python environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open:

```text
End_to_End_NLP_Sentiment_Analysis.ipynb
```

The notebook downloads the required dataset and GloVe embeddings during execution.

## Author

**Warda Anees**

MS Data Science | NUST

Areas of Interest: Natural Language Processing, Large Language Models, Artificial Intelligence
