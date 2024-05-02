"""

"""
import collections
import string
import typing

import pandas
import pydantic

from .sentence import Sentence
from .. import functional

__all__ = ["Sentence"]


class CorpusArgs(pydantic.BaseModel):
    """

    """
    token_count_exclude: typing.List[str] = pydantic.Field(
        default_factory=lambda: ["’", "“", "”", *string.punctuation]
    )


class Corpus(pydantic.BaseModel):
    """

    """
    raw: str
    sentences: typing.List[Sentence] = []
    args: CorpusArgs = CorpusArgs()

    @functional.timeit
    def model_post_init(self, __context) -> None:
        """

        """
        if not self.sentences:
            self.sentences = [
                Sentence(raw=sent) for sent in functional.text.sentenize(self.raw)
            ]

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def tokens(self) -> typing.List[str]:
        """

        """
        return [tok for sent in self.sentences for tok in sent.tokens]

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def bigrams(self) -> typing.List[typing.Tuple[str, ...]]:
        """

        """
        return self.generate_ngrams(2)

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def trigrams(self) -> typing.List[typing.Tuple[str, ...]]:
        """

        """
        return self.generate_ngrams(3)

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def tetragram(self) -> typing.List[typing.Tuple[str, ...]]:
        """

        """
        return self.generate_ngrams(4)

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def pentagram(self) -> typing.List[typing.Tuple[str, ...]]:
        """

        """
        return self.generate_ngrams(5)

    def count_languages(self) -> collections.Counter:
        """

        """
        return collections.Counter([sent.language for sent in self.sentences])

    def count_tokens(self) -> collections.Counter:
        """

        """
        filter_words = [
            *functional.text.load_stopwords(
                list(set([sent.language for sent in self.sentences]))
            ),
            *self.args.token_count_exclude,
        ]

        return collections.Counter([tok for tok in self.tokens if tok not in filter_words])

    def count_ngrams(self, n: int) -> collections.Counter:
        """

        """
        return collections.Counter(self.generate_ngrams(n))

    def create_subset_by_language(self, language: str) -> "Corpus":
        """

        """
        return Corpus(
            sentences=(
                subset := [sent for sent in self.sentences if sent.language == language]
            ),
            raw=" ".join([sent.raw for sent in subset]),
        )

    def generate_ngrams(self, n: int) -> typing.List[typing.Tuple[str, ...]]:
        """

        """
        return [
            ngram for sent in self.sentences for ngram in functional.text.ngrams(sent.tokens, n)
        ]

    @classmethod
    def from_txt(cls, path: str) -> "Corpus":
        """

        """
        return cls(raw=open(path).read())

    def to_df(self) -> pandas.DataFrame:
        """

        """
        return pandas.DataFrame(
            [sent.to_row() for sent in self.sentences],
        )

    def __len__(self) -> int:
        """

        """
        return len(self.sentences)
