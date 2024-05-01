import os

import nltk
import transformers

from . import corpus
from . import encoder
from . import utility

# preload nltk data used in functional.text
nltk.download("punkt")
nltk.download("stopwords")

# configure encoder:model/tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()

__all__ = [corpus, encoder, utility]
