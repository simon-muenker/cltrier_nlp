import typing

import nltk
import pydantic

from . import util

nltk.download('punkt')


class Sentence(pydantic.BaseModel):
    content: str
    language: str = None

    def model_post_init(self, __context) -> None:
        self.language = util.detect_language(self.content)

    @pydantic.computed_field
    @property
    def tokens(self) -> typing.List[str]:
        try:
            return nltk.tokenize.word_tokenize(self.content.lower(), language=self.language)

        except LookupError:
            return nltk.tokenize.word_tokenize(self.content.lower())

    def __len__(self) -> int:
        return len(self.tokens)
