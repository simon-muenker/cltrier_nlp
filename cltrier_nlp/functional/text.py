import typing

import nltk

nltk.download("punkt")
nltk.download("stopwords")


def load_nltk_stopwords(languages: typing.List[str]) -> typing.List[str]:
    return list(
        set().union(
            *[
                nltk.corpus.stopwords.words(lang)
                for lang in languages
                if lang in nltk.corpus.stopwords.fileids()
            ]
        )
    )


def tokenize(text: str, language: str) -> typing.List[str]:
    try:
        return nltk.tokenize.word_tokenize(text.lower(), language=language)

    except LookupError:
        return nltk.tokenize.word_tokenize(text.lower())


def ngrams(tokens: typing.List[str], n: int) -> typing.List[typing.Tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
