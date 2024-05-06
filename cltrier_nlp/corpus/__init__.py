"""

"""
import collections
import string
import typing

import pandas
import pydantic

from .sentence import Sentence
from .. import functional
from .. import utility

__all__ = ["Sentence"]


class CorpusArgs(pydantic.BaseModel):
    """

    """
    token_count_exclude: utility.types.Tokens = pydantic.Field(
        default_factory=lambda: ["’", "“", "”", *string.punctuation]
    )


class Corpus(pydantic.BaseModel):
    """

    """
    raw: str
    sentences: typing.List[Sentence] = []
    args: CorpusArgs = CorpusArgs()

    @functional.timeit
    def model_post_init(self, __context: typing.Any) -> None:
        """

        Args:
            __context (typing.Any): ??

        Returns:

        """
        if not self.sentences:
            self.sentences = [
                Sentence(raw=sent) for sent in functional.text.sentenize(self.raw)
            ]

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def tokens(self) -> utility.types.Tokens:
        """

        Returns:

        """
        return [tok for sent in self.sentences for tok in sent.tokens]

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def bigrams(self) -> utility.types.NGrams:
        """

        Returns:
            utility.types.NGrams:
        """
        return self.generate_ngrams(2)

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def trigrams(self) -> utility.types.NGrams:
        """

        Returns:
            utility.types.NGrams:
        """
        return self.generate_ngrams(3)

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def tetragram(self) -> utility.types.NGrams:
        """

        Returns:
            utility.types.NGrams:
        """
        return self.generate_ngrams(4)

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def pentagram(self) -> utility.types.NGrams:
        """

        Returns:
            utility.types.NGrams:
        """
        return self.generate_ngrams(5)

    def count_languages(self) -> collections.Counter:
        """

        Returns:
            collections.Counter:
        """
        return collections.Counter([sent.language for sent in self.sentences])

    def count_tokens(self) -> collections.Counter:
        """

        Returns:
            collections.Counter:
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

        Args:
            n (int):

        Returns:
            collections.Counter:
        """
        return collections.Counter(self.generate_ngrams(n))

    def create_subset_by_language(self, language: str) -> "Corpus":
        """

        Args:
            language (str):

        Returns:
            Corpus
        """
        return Corpus(
            sentences=(
                subset := [sent for sent in self.sentences if sent.language == language]
            ),
            raw=" ".join([sent.raw for sent in subset]),  # type: ignore[has-type]
        )

    def generate_ngrams(self, n: int) -> utility.types.NGrams:
        """

        Args:
            n (int):

        Returns:
            utility.types.NGrams:
        """
        return [
            ngram for sent in self.sentences for ngram in functional.text.ngrams(sent.tokens, n)
        ]

    @classmethod
    def from_txt(cls, path: str) -> "Corpus":
        """

        Args:
            path (str):

        Returns:
            Corpus
        """
        return cls(raw=open(path).read())

    def to_df(self) -> pandas.DataFrame:
        """

        Returns:
            pandas.DataFrame:
        """
        return pandas.DataFrame(
            [sent.to_row() for sent in self.sentences],
        )

    def __len__(self) -> int:
        """

        Returns:
            int:
        """
        return len(self.sentences)
