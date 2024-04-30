import collections
import typing
import string

import nltk
import pydantic

from . import util
from .sentence import Sentence

nltk.download('punkt')
nltk.download('stopwords')


class CorpusArgs(pydantic.BaseModel):
    exclude_from_count: typing.List[str] = ['’', '“', '”']


class Corpus(pydantic.BaseModel):
    raw: str
    sentences: typing.List[Sentence] = []
    args: CorpusArgs = CorpusArgs()

    def model_post_init(self, __context) -> None:
        if not self.sentences:
            self.sentences = [
                Sentence(content=sent)
                for sent in
                nltk.tokenize.sent_tokenize(
                    self.raw,
                    language=util.detect_language(self.raw)
                )
            ]

    @pydantic.computed_field
    @property
    def tokens(self) -> typing.List[str]:
        return [
            tok
            for sent in self.sentences
            for tok in sent.tokens
        ]

    def count_languages(self) -> collections.Counter:
        return collections.Counter([sent.language for sent in self.sentences])

    def count_tokens(self) -> collections.Counter:
        __stopwords = list(set().union(*[
            nltk.corpus.stopwords.words(lang)
            for lang in self.count_languages().keys()
            if lang in nltk.corpus.stopwords.fileids()
        ]))

        return collections.Counter([
            tok for tok in self.tokens
            if tok not in [
                *__stopwords,
                *string.punctuation,
                *self.args.exclude_from_count
            ]
        ])

    def create_subset_by_language(self, language: str) -> 'Corpus':
        return Corpus(
            sentences=(
                sentences := [
                    sent for sent in self.sentences
                    if sent.language == language
                ]
            ),
            raw=' '.join([
                sent.content for sent in sentences
            ])
        )

    @classmethod
    def from_txt(cls, path: str) -> 'Corpus':
        return cls(raw=open(path).read())

    def __len__(self) -> int:
        return len(self.sentences)
