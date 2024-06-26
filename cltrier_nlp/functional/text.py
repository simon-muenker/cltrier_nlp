import warnings

import langcodes
import langdetect
import nltk

from .. import utility

UNK_LANG: str = "unknown"


def load_stopwords(languages: utility.types.Tokens) -> utility.types.Tokens:
    """

    Args:
        languages:

    Returns:

    """
    return list(
        set().union(
            *[
                nltk.corpus.stopwords.words(lang)
                for lang in languages
                if lang in nltk.corpus.stopwords.fileids()
            ]
        )
    )


def sentenize(text: str) -> utility.types.Batch[str]:
    """

    Args:
        text:

    Returns:

    """
    return nltk.tokenize.sent_tokenize(text, language=detect_language(text))


def tokenize(text: str) -> utility.types.Tokens:
    """

    Args:
        text:

    Returns:

    """
    try:
        return nltk.tokenize.word_tokenize(text.lower(), language=detect_language(text))

    except LookupError:
        return nltk.tokenize.word_tokenize(text.lower())


def ngrams(tokens: utility.types.Tokens, n: int) -> utility.types.NGrams:
    """

    Args:
        tokens:
        n:

    Returns:

    """
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def detect_language(content: str) -> str:
    """

    Args:
        content:

    Returns:

    """

    # Ignore langcodes dependent language data warning
    # DeprecationWarning: pkg_resources is deprecated as an API.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            return langcodes.Language.get(langdetect.detect(content)).display_name().lower()

        except langdetect.lang_detect_exception.LangDetectException:
            return UNK_LANG
