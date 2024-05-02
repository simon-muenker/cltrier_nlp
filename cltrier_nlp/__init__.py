import os

import nltk
import transformers

from . import corpus
from . import encoder
from . import utility

# preload nltk data used in functional.text
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# configure encoder:model/tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()

__all__ = ["corpus", "encoder", "utility"]
