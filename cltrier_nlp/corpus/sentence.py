import typing

import nltk
import pydantic

from . import util

nltk.download('punkt')


class Sentence(pydantic.BaseModel):
    content: str

    language: str = None
    tokens: typing.List[str] = pydantic.Field(default_factory=lambda: [])

    def model_post_init(self, __context) -> None:

        if not self.language:
            self.language = util.detect_language(self.content)

        if not self.tokens:
            self.tokens = self.generate_tokens()

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

    def generate_ngrams(self, n: int) -> typing.List[typing.Tuple[str, ...]]:
        return [tuple(self.tokens[i:i + n]) for i in range(len(self.tokens) - n + 1)]

    def generate_tokens(self):
        try:
            return nltk.tokenize.word_tokenize(self.content.lower(), language=self.language)

        except LookupError:
            return nltk.tokenize.word_tokenize(self.content.lower())

    def __len__(self) -> int:
        return len(self.tokens)
