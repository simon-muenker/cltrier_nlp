import typing

from cltrier_nlp import corpus

SAMPLES: typing.List[typing.Dict[str, str]] = [
    {'lang': 'german', 'content': 'Franz jagt im komplett verwahrlosten Taxi quer durch Bayern.'},
    {'lang': 'english', 'content': 'Ted goes to the gym and exercises three times a week.'},
]


def test_tokenization():
    sent = corpus.Sentence(content=SAMPLES[0]['content'])

    assert isinstance(sent.tokens, typing.List)
    assert all([isinstance(tok, str) for tok in sent.tokens])


def test_languages():

    for sample in SAMPLES:
        sent = corpus.Sentence(content=sample['content'])

        assert sent.language == sample['lang']
