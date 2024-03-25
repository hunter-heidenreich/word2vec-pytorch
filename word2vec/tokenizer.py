import os
import unicodedata
from typing import Iterable, Optional

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers import trainers, normalizers, models, pre_tokenizers, processors


def add_spaces_around_foreign_characters(text: str) -> str:
    """Add spaces around foreign characters in a text.

    Args:
        text: The text.

    Returns:
        The text with spaces around foreign characters.
    """
    new_text = []
    for char in text:
        # Check character category
        cat = unicodedata.category(char)
        # Letter categories start with 'L'
        if cat.startswith("L") and ord(char) >= 128:
            new_text.append(" " + char + " ")
        else:
            new_text.append(char)
    return "".join(new_text)


def get_tokenizer(
    data: Optional[Iterable[str]] = None,
    dataset_name: Optional[str] = "wikitext",
    subset: Optional[str] = "wikitext-2-raw-v1",
    vocab_size: int = 1_000_000,
    min_frequency: int = 2,
) -> Tokenizer:
    """Get a tokenizer for a dataset. If the tokenizer is not found, train a new one.

    Args:
        data: The dataset.
        dataset_name: The name of the dataset.
        subset: The subset of the dataset.
        vocab_size: The size of the vocabulary.
        min_frequency: The minimum frequency of a token to be included in the vocabulary.

    Returns:
        The tokenizer.
    """
    if not os.path.exists("tokenizers"):
        os.makedirs("tokenizers")

    size_str_mil = (
        f"{vocab_size // 1_000_000}M" if vocab_size >= 1_000_000 else f"{vocab_size}"
    )
    try:
        tokenizer = Tokenizer.from_file(
            f"tokenizers/{subset}-vocab_{min_frequency}_{size_str_mil}.json"
        )
        print("Loaded. Vocabulary size:", tokenizer.get_vocab_size())
        return tokenizer
    except Exception:
        pass

    if data is None:
        data = load_dataset(dataset_name, subset, split="train")["text"]
        data = [
            add_spaces_around_foreign_characters(text) for text in data if len(text) > 0
        ]

    # Initialize a tokenizer
    tokenizer = Tokenizer(
        models.WordLevel(unk_token="<unk>"),
    )
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
    special_tokens = [
        ("<unk>", 0),
        ("<pad>", 1),
        ("<eos>", 2),
        ("<sos>", 3),
    ]
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<sos> $A <eos>",
        special_tokens=special_tokens,
    )
    tokenizer.train_from_iterator(
        data,
        trainer=trainers.WordLevelTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<unk>", "<pad>", "<eos>", "<sos>"],
            show_progress=True,
        ),
    )
    tokenizer.save(f"tokenizers/{subset}-vocab_{min_frequency}_{size_str_mil}.json")
    print("Trained. Vocabulary size:", tokenizer.get_vocab_size())
    return tokenizer
