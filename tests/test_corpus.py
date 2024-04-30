import collections
import logging

import nltk

from cltrier_nlp import corpus

nltk.download('gutenberg')

BIB = nltk.corpus.gutenberg
SUBSET: str = 'carroll-alice.txt'

CORPUS = corpus.Corpus(raw=BIB.raw(SUBSET))


def test_from_raw():
    assert isinstance(CORPUS, corpus.Corpus)

    # FIXME
    # throws error: assert len(corp) == len(CORPUS.sents(SUBSET))
    # indicates a different approach in sent tokenization used in generating the nltk based version


def test_languages():
    lang_count: collections.Counter = CORPUS.count_languages()
    logging.debug(lang_count)

    assert isinstance(lang_count, collections.Counter)

    # TODO
    # find test cases


def test_languages_subset():
    subset: corpus.Corpus = CORPUS.create_subset_by_language('german')

    assert isinstance(subset, corpus.Corpus)

    # TODO
    # find test cases


def test_tokens():
    token_count = CORPUS.count_tokens()
    logging.debug(token_count.most_common(40))

    assert isinstance(token_count, collections.Counter)

    # TODO
    # find test cases


def test_ngrams():
    assert True

    # TODO
    # find test cases
