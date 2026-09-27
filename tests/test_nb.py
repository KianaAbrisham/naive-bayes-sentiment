import unittest
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from src.nb_scratch import MultinomialNBScratch

class NaiveBayesTests(unittest.TestCase):
    def test_reference_probabilities_and_unknown_words(self):
        texts = ['Great great FILM!', 'A fine café film', 'bad dull film', 'awful film']
        labels = ['pos', 'pos', 'neg', 'neg']
        vectorizer = CountVectorizer()
        reference = MultinomialNB(alpha=0.7).fit(vectorizer.fit_transform(texts), labels)
        scratch = MultinomialNBScratch(alpha=0.7).fit(texts, labels)
        unseen = ['GREAT, film!', 'unknownword', '', 'café dull']
        expected = reference.predict_log_proba(vectorizer.transform(unseen))
        actual = scratch.predict_log_proba(unseen)
        np.testing.assert_array_equal(scratch.classes_, reference.classes_)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(np.exp(actual).sum(axis=1), 1)
        np.testing.assert_allclose(np.exp(actual[1]), [0.5, 0.5])

    def test_reject_invalid_training_inputs(self):
        for alpha in [0, -1, np.nan]:
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                MultinomialNBScratch(alpha)
        for texts, labels in [([], []), (['text'], []), (['a !'], [1]), ([None], [1])]:
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                MultinomialNBScratch().fit(texts, labels)

    def test_prediction_requires_fit(self):
        with self.assertRaises(RuntimeError):
            MultinomialNBScratch().predict(['hello'])

if __name__ == '__main__':
    unittest.main()
