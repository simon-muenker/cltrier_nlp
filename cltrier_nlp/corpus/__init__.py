import collections
import string
import typing

import nltk
import pydantic

from . import util
from .sentence import Sentence

nltk.download("punkt")
nltk.download("stopwords")


class CorpusArgs(pydantic.BaseModel):
    exclude_from_count: typing.List[str] = ["’", "“", "”"]


class Corpus(pydantic.BaseModel):
    raw: str
    sentences: typing.List[Sentence] = []
    args: CorpusArgs = CorpusArgs()

    def model_post_init(self, __context) -> None:
        if not self.sentences:
            self.sentences = [
                Sentence(content=sent)
                for sent in nltk.tokenize.sent_tokenize(
                    self.raw, language=util.detect_language(self.raw)
                )
            ]

    @pydantic.computed_field
    @property
    def tokens(self) -> typing.List[str]:
        return [tok for sent in self.sentences for tok in sent.tokens]

    @pydantic.computed_field
    @property
    def bigrams(self) -> typing.List[typing.Tuple[str, ...]]:
        return self.generate_ngrams(2)

    @pydantic.computed_field
    @property
    def trigrams(self) -> typing.List[typing.Tuple[str, ...]]:
        return self.generate_ngrams(3)

    @pydantic.computed_field
    @property
    def tetragram(self) -> typing.List[typing.Tuple[str, ...]]:
        return self.generate_ngrams(4)

    @pydantic.computed_field
    @property
    def pentagram(self) -> typing.List[typing.Tuple[str, ...]]:
        return self.generate_ngrams(5)

    def count_languages(self) -> collections.Counter:
        return collections.Counter([sent.language for sent in self.sentences])

    def count_tokens(self) -> collections.Counter:
        __stopwords = list(
            set().union(
                *[
                    nltk.corpus.stopwords.words(lang)
                    for lang in self.count_languages().keys()
                    if lang in nltk.corpus.stopwords.fileids()
                ]
            )
        )

        return collections.Counter(
            [
                tok
                for tok in self.tokens
                if tok
                not in [
                    *__stopwords,
                    *string.punctuation,
                    *self.args.exclude_from_count,
                ]
            ]
        )

    def count_ngrams(self, n: int) -> collections.Counter:
        return collections.Counter(self.generate_ngrams(n))

    def create_subset_by_language(self, language: str) -> "Corpus":
        return Corpus(
            sentences=(
                sentences := [
                    sent for sent in self.sentences if sent.language == language
                ]
            ),
            raw=" ".join([sent.content for sent in sentences]),
        )

    def generate_ngrams(self, n: int) -> typing.List[typing.Tuple[str, ...]]:
        return [ngram for sent in self.sentences for ngram in sent.generate_ngrams(n)]

    @classmethod
    def from_txt(cls, path: str) -> "Corpus":
        return cls(raw=open(path).read())

    def __len__(self) -> int:
        return len(self.sentences)
