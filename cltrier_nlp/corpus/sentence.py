import typing

import pandas
import pydantic

from .. import functional


class Sentence(pydantic.BaseModel):
    """

    """
    raw: str

    language: str = 'unknown'
    tokens: typing.List[str] = pydantic.Field(default_factory=lambda: [])

    def model_post_init(self, __context) -> None:
        """

        """
        self.raw = self.raw.replace("\n", " ").strip()

        if not self.language:
            self.language = functional.text.detect_language(self.raw)

        if not self.tokens:
            self.tokens = functional.text.tokenize(self.raw)

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def bigrams(self) -> typing.List[typing.Tuple[str, ...]]:
        """

        """
        return functional.text.ngrams(self.tokens, 2)

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def trigrams(self) -> typing.List[typing.Tuple[str, ...]]:
        """

        """
        return functional.text.ngrams(self.tokens, 3)

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def tetragram(self) -> typing.List[typing.Tuple[str, ...]]:
        """

        """
        return functional.text.ngrams(self.tokens, 4)

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def pentagram(self) -> typing.List[typing.Tuple[str, ...]]:
        """

        """
        return functional.text.ngrams(self.tokens, 5)

    def to_row(self) -> pandas.Series:
        """

        """
        return pandas.Series(self.model_dump())

    def __len__(self) -> int:
        """

        """
        return len(self.tokens)
