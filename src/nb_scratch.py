import numpy as np
from collections import defaultdict

class MultinomialNBScratch:
    """Multinomial Naive Bayes for bag-of-words with Laplace smoothing."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.vocab_ = {}
        self.inv_vocab_ = []

    def _build_vocab(self, texts):
        vocab = {}
        for txt in texts:
            for tok in txt.split():
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self.vocab_ = vocab
        self.inv_vocab_ = [None]*len(vocab)
        for t, i in vocab.items():
            self.inv_vocab_[i] = t

    def _vectorize(self, texts):
        X = np.zeros((len(texts), len(self.vocab_)), dtype=np.int64)
        for i, txt in enumerate(texts):
            for tok in txt.split():
                j = self.vocab_.get(tok)
                if j is not None:
                    X[i, j] += 1
        return X

    def fit(self, texts, y):
        # build vocab then count
        self._build_vocab(texts)
        X = self._vectorize(texts)
        y = np.asarray(y)
        classes = np.unique(y)
        self.classes_ = classes
        n_classes = len(classes)
        n_features = X.shape[1]

        class_count = np.zeros(n_classes, dtype=np.float64)
        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, c in enumerate(classes):
            Xc = X[y == c]
            class_count[idx] = Xc.shape[0]
            feature_count[idx] = Xc.sum(axis=0)

        # priors
        self.class_log_prior_ = np.log(class_count / class_count.sum())

        # likelihood with Laplace smoothing
        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)
        return self

    def predict_log_proba(self, texts):
        X = self._vectorize(texts)
        # log P(y) + X log P(x|y)
        jll = self.class_log_prior_ + X @ self.feature_log_prob_.T
        return jll

    def predict(self, texts):
        jll = self.predict_log_proba(texts)
        return self.classes_[np.argmax(jll, axis=1)]
