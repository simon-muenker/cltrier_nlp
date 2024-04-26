import warnings

import langcodes
import langdetect


def detect_language(content: str) -> str:
    # Ignore langcodes dependent language data warning
    # DeprecationWarning: pkg_resources is deprecated as an API.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            return (
                langcodes.Language.get(
                    langdetect.detect(content)
                )
                .display_name()
                .lower()
            )

        except langdetect.lang_detect_exception.LangDetectException:
            return 'unknown'
