import typing
import re

import collections

from nltk.corpus import stopwords

corpus: str = open('data/voltair-candide.txt').read()


class Sentence:

    def __init__(self, text: str) -> None:
        self.text = text

    @property
    def tokens(self) -> typing.List[str]:
        return self.text.lower().split()

    @property
    def hashtags(self) -> typing.List[str]:
        return re.sub(r'#\S+', '<hashtag>', self.text)


class Corpus:

    def __init__(self, corpus: str) -> None:
        self.corpus = corpus
        self.sentences: typing.List[Sentence] = [
            Sentence(sent) for sent in corpus.split('.')
        ]

    @property
    def tokens(self) -> typing.List[typing.List[str]]:
        return [sent.tokens for sent in self.sentences]

    def calc_freq(self) -> collections.Counter:
        _stopwords = stopwords.words('english')

        return collections.Counter([
            tok
            for sent in self.sentences
            for tok in sent.tokens
            if tok not in _stopwords
        ])
