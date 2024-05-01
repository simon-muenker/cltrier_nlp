import typing
import warnings

import langcodes
import langdetect
import nltk


def load_stopwords(languages: typing.List[str]) -> typing.List[str]:
    return list(
        set().union(
            *[
                nltk.corpus.stopwords.words(lang)
                for lang in languages
                if lang in nltk.corpus.stopwords.fileids()
            ]
        )
    )


def sentenize(text: str) -> typing.List[str]:
    return nltk.tokenize.sent_tokenize(text, language=detect_language(text))


def tokenize(text: str) -> typing.List[str]:
    try:
        return nltk.tokenize.word_tokenize(text.lower(), language=detect_language(text))

    except LookupError:
        return nltk.tokenize.word_tokenize(text.lower())


def ngrams(tokens: typing.List[str], n: int) -> typing.List[typing.Tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def detect_language(content: str) -> str:
    # Ignore langcodes dependent language data warning
    # DeprecationWarning: pkg_resources is deprecated as an API.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            return langcodes.Language.get(langdetect.detect(content)).display_name().lower()

        except langdetect.lang_detect_exception.LangDetectException:
            return "unknown"
