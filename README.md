# Naive Bayes for Sentiment Classification (Portfolio Sample)

This repository demonstrates **text classification** using **Naive Bayes**.
It contains both a clean `scikit-learn` pipeline and a concise **from-scratch**
implementation of **Multinomial Naive Bayes** with Laplace smoothing.

## What this shows
- End-to-end **NLP workflow**: tokenization → vectorization → model training → evaluation
- **From-scratch Multinomial NB** (transparent math; Laplace smoothing)
- `scikit-learn` **Pipeline** baseline for comparison
- Works with a tiny built-in toy dataset *or* any CSV with columns `text,label`

## Structure
```
.
├── notebooks
│   └── nb_sentiment.ipynb         # End-to-end demo (scratch + sklearn)
├── src
│   └── nb_scratch.py              # Minimal Multinomial NB from scratch
├── data
│   └── sample_tiny.csv            # Tiny demo dataset
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/nb_sentiment.ipynb
```

### Using your own data
Provide a CSV with columns `text,label` (labels like `pos`/`neg` or `1`/`0`). In the notebook, set the file path and run all cells.

## License
MIT
