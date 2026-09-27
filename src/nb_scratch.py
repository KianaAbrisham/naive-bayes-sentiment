"""Small, dense bag-of-words implementation for teaching and comparison."""
import re
import numpy as np

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")

def tokenize(text):
    """Match CountVectorizer's default lowercase, two-character word tokens."""
    return TOKEN_PATTERN.findall(text.lower())

class MultinomialNBScratch:
    """Multinomial Naive Bayes with additive smoothing and fitted vocabulary."""

    def __init__(self, alpha=1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('alpha must be finite and strictly positive.')
        self.alpha = float(alpha)
        self.classes_ = None

    @staticmethod
    def _texts(texts):
        if isinstance(texts, str):
            raise ValueError('Pass a sequence of documents, not one string.')
        texts = list(texts)
        if not all(isinstance(text, str) for text in texts):
            raise ValueError('Every document must be a string.')
        return texts

    def _vectorize(self, texts):
        counts = np.zeros((len(texts), len(self.vocab_)), dtype=np.float64)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                column = self.vocab_.get(token)
                if column is not None:
                    counts[row, column] += 1
        return counts

    def fit(self, texts, y):
        texts = self._texts(texts)
        y = np.asarray(y)
        if not texts or y.ndim != 1 or len(y) != len(texts):
            raise ValueError('Provide equally sized, nonempty documents and labels.')
        if any(label is None or (isinstance(label, (float, np.floating))
                                and not np.isfinite(label)) for label in y):
            raise ValueError('Labels must not be missing or infinite.')
        vocabulary = sorted({token for text in texts for token in tokenize(text)})
        if not vocabulary:
            raise ValueError('Training documents contain no usable tokens.')
        self.vocab_ = {word: index for index, word in enumerate(vocabulary)}
        self.inv_vocab_ = vocabulary
        counts = self._vectorize(texts)
        self.classes_, inverse = np.unique(y, return_inverse=True)
        self.class_count_ = np.bincount(inverse).astype(float)
        self.feature_count_ = np.stack([
            counts[inverse == index].sum(axis=0)
            for index in range(len(self.classes_))
        ])
        self.class_log_prior_ = np.log(self.class_count_ / len(y))
        smoothed = self.feature_count_ + self.alpha
        self.feature_log_prob_ = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
        return self

    def _joint_log_likelihood(self, texts):
        if self.classes_ is None:
            raise RuntimeError('Call fit before prediction.')
        texts = self._texts(texts)
        return self._vectorize(texts) @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_log_proba(self, texts):
        joint = self._joint_log_likelihood(texts)
        # Stable log-sum-exp normalization; each row then sums to one in probability space.
        maximum = joint.max(axis=1, keepdims=True)
        log_normalizer = maximum + np.log(np.exp(joint - maximum).sum(axis=1, keepdims=True))
        return joint - log_normalizer

    def predict_proba(self, texts):
        return np.exp(self.predict_log_proba(texts))

    def predict(self, texts):
        joint = self._joint_log_likelihood(texts)
        return self.classes_[np.argmax(joint, axis=1)]
