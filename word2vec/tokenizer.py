import logging
from pathlib import Path
import unicodedata
from typing import Iterable, Optional

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers import trainers, normalizers, models, pre_tokenizers, processors

logger = logging.getLogger(__name__)


SPECIAL_TOKENS = ["<unk>", "<pad>", "<eos>", "<sos>"]


def add_spaces_around_foreign_characters(text: str) -> str:
    """Add spaces around non-ASCII characters in a text.

    This function modifies the input string by inserting spaces around each character
    categorized as a letter but with an ASCII value above 128. This helps in situations
    where such characters need to be tokenized separately in text processing workflows.

    Args:
        text (str): The text to process.

    Returns:
        str: The processed text with spaces added around non-ASCII letters.
    """
    return "".join(
        f" {char} "
        if unicodedata.category(char).startswith("L") and ord(char) >= 128
        else char
        for char in text
    )


def setup_tokenizer():
    """Configure and return a tokenizer with pre-set configurations.

    This function initializes a WordLevel tokenizer and sets up its normalizers,
    pre-tokenizers, and post-processors, including handling of special tokens.

    Returns:
        Tokenizer: A fully configured tokenizer.
    """
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
            normalizers.Strip(),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.Whitespace(),
        ]
    )
    special_tokens = [(token, i) for i, token in enumerate(SPECIAL_TOKENS)]
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<sos> $A <eos>",
        special_tokens=special_tokens,
    )
    return tokenizer


def get_tokenizer(
    data: Optional[Iterable[str]] = None,
    dataset_name: str = "wikitext",
    subset: str = "wikitext-2-raw-v1",
    vocab_size: int = 1_000_000,
    min_frequency: int = 2,
) -> Optional[Tokenizer]:
    """Retrieve or train a tokenizer for a given dataset.

    Loads a tokenizer from a saved file if available; otherwise, it initializes a tokenizer
    and trains it on the provided or automatically loaded data from a specified dataset.

    Args:
        data (Optional[Iterable[str]]): The dataset to train the tokenizer.
        dataset_name (str): The dataset name to load if data is not provided.
        subset (str): The specific subset of the dataset to use.
        vocab_size (int): The maximum size of the vocabulary.
        min_frequency (int): The minimum frequency a token must have to be included.

    Returns:
        Tokenizer: The loaded or trained tokenizer, or None if training failed.
    """
    tokenizers_dir = Path("tokenizers")
    tokenizers_dir.mkdir(exist_ok=True)
    size_str_mil = (
        f"{vocab_size // 1_000_000}M" if vocab_size >= 1_000_000 else f"{vocab_size}"
    )
    tokenizer_path = (
        tokenizers_dir / f"{subset}-vocab_{min_frequency}_{size_str_mil}.json"
    )

    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        logger.info("Loaded tokenizer from file.")
        return tokenizer

    if data is None:
        try:
            dataset = load_dataset(dataset_name, subset, split="train")
            data = [
                add_spaces_around_foreign_characters(text)
                for text in dataset["text"]
                if text
            ]
        except ValueError as e:
            logger.error(f"Failed to load dataset {dataset_name}/{subset}: {e}")
            return None

    if not data:
        logger.error("Data is empty. Cannot train tokenizer.")
        return None

    tokenizer = setup_tokenizer()
    try:
        tokenizer.train_from_iterator(
            data,
            trainer=trainers.WordLevelTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=SPECIAL_TOKENS,
                show_progress=logging.getLogger().level <= logging.INFO,
            ),
        )
        tokenizer.save(str(tokenizer_path))
        logger.info("Trained tokenizer and saved to file.")
    except Exception as e:
        logger.error(f"Failed to train or save tokenizer: {e}")
        return None

    return tokenizer
