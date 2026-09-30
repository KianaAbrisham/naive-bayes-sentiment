# Multinomial Naive Bayes for Text Classification

[![Checks](https://github.com/KianaAbrisham/naive-bayes-sentiment/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/KianaAbrisham/naive-bayes-sentiment/actions/workflows/checks.yml)

A NumPy implementation of multinomial Naive Bayes, compared numerically with a
scikit-learn reference using the same tokenization and smoothing.

## What the example demonstrates

- Lowercase word tokens of at least two characters, matching CountVectorizer's default behavior.
- A vocabulary learned from training documents only.
- Additive smoothing and empirical class priors.
- Numerically stable, normalized log probabilities.
- An explicit check that the scratch and reference probabilities agree.

The included dataset contains **six hand-written sentences**. The fixed split uses four for training
and two for testing; both implementations classify one of those two test examples correctly.
This is a small implementation check, not evidence of useful sentiment-model accuracy.

## Files and limitations

| Path | Purpose |
|---|---|
| [notebooks/nb_sentiment.ipynb](notebooks/nb_sentiment.ipynb) | Executed comparison with real outputs |
| [src/nb_scratch.py](src/nb_scratch.py) | Vocabulary, counts, smoothing and prediction |
| [data/sample_tiny.csv](data/sample_tiny.csv) | Six illustrative `text,label` rows |
| [tests/test_nb.py](tests/test_nb.py) | Reference agreement, probability normalization and invalid-input checks |

Unknown words are ignored; a document with no known words receives the learned priors.
The dense count matrix is intended for small examples, not large corpora. For another CSV,
update the notebook's data path and provide nonmissing text and labels, with enough examples
per class to support the stratified split. Meaningful evaluation needs a substantially larger
dataset and independent test observations.

## Run locally

Use Python 3.12 and a separate environment for this project. From the repository folder:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` in Windows Command Prompt or
`source .venv/bin/activate` on Linux/macOS, then run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
jupyter notebook notebooks/nb_sentiment.ipynb
```

The notebook finds the repository from either its root folder or `notebooks/`.
The saved outputs come from CPU execution with the included data; see
[validation](docs/VALIDATION.md) for the checks and limits.

[Development notes](https://github.com/KianaAbrisham/KianaAbrisham/blob/main/docs/DEVELOPMENT.md)

## License

MIT — see [LICENSE](LICENSE).
