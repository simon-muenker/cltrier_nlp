import typing

import pandas

from cltrier_nlp import corpus

SAMPLES: typing.List[typing.Dict[str, str]] = [
    {
        "lang": "german",
        "content": "Franz jagt im komplett verwahrlosten Taxi quer durch Bayern.",
    },
    {
        "lang": "english",
        "content": "Ted goes to the gym and exercises three times a week.",
    },
]


def test_languages():
    for sample in SAMPLES:
        sent = corpus.Sentence(raw=sample["content"])

        assert sent.language == sample["lang"]


def test_tokenization():
    sent = corpus.Sentence(raw=SAMPLES[0]["content"])

    assert isinstance(sent.tokens, typing.List)
    assert all([isinstance(tok, str) for tok in sent.tokens])


def test_ngrams():
    sent = corpus.Sentence(raw=SAMPLES[0]["content"])

    for n, grams in enumerate(
        [sent.bigrams, sent.trigrams, sent.tetragram, sent.pentagram], start=2
    ):
        assert isinstance(grams, typing.List)
        assert all([isinstance(gr, typing.Tuple) for gr in grams])
        assert all([isinstance(tok, str) for gr in grams for tok in gr])
        assert all([len(gr) == n for gr in grams])


def test_to_row():
    sent = corpus.Sentence(raw=SAMPLES[0]["content"])

    assert isinstance(sent.to_row(), pandas.Series)
