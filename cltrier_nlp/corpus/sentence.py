import typing

import pandas
import pydantic

from .. import functional
from .. import utility


class Sentence(pydantic.BaseModel):
    """

    """
    raw: str

    language: str = functional.text.UNK_LANG
    tokens: utility.types.Tokens = pydantic.Field(default_factory=lambda: [])

    def model_post_init(self, __context: typing.Any) -> None:
        """

        Args:
            __context (typing.Any): ??

        Returns:
            None
        """
        self.raw = self.raw.replace("\n", " ").strip()

        if self.language == functional.text.UNK_LANG:
            self.language = functional.text.detect_language(self.raw)

        if not self.tokens:
            self.tokens = functional.text.tokenize(self.raw)

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def bigrams(self) -> utility.types.NGrams:
        """

        Returns:
            utility.types.NGrams:

        """
        return functional.text.ngrams(self.tokens, 2)

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def trigrams(self) -> utility.types.NGrams:
        """

        Returns:
            utility.types.NGrams:

        """
        return functional.text.ngrams(self.tokens, 3)

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def tetragram(self) -> utility.types.NGrams:
        """

        Returns:
            utility.types.NGrams:

        """
        return functional.text.ngrams(self.tokens, 4)

    @pydantic.computed_field  # type: ignore[misc]
    @property
    def pentagram(self) -> typing.List[typing.Tuple[str, ...]]:
        """

        Returns:

        """
        return functional.text.ngrams(self.tokens, 5)

    def to_row(self) -> pandas.Series:
        """

        Returns:
            pandas.Series:

        """
        return pandas.Series(self.model_dump())

    def __len__(self) -> int:
        """

        Returns:
            int:

        """
        return len(self.tokens)
