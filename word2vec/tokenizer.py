from typing import Optional

from datasets import load_dataset, Dataset
from tokenizers import Tokenizer 
from tokenizers import trainers, normalizers, models, pre_tokenizers


def get_tokenizer(
    data: Optional[Dataset] = None,
    dataset_name: Optional[str] = 'wikitext',
    subset: Optional[str] = 'wikitext-2-raw-v1',
    vocab_size: int = 1_000_000,
) -> Tokenizer:
    """Get a tokenizer for a dataset. If the tokenizer is not found, train a new one.
    
    Args:
        dataset_name: The name of the dataset.
        subset: The subset of the dataset.
        vocab_size: The size of the vocabulary.
        
    Returns:
        The tokenizer.
    """
    size_str_mil = f'{vocab_size // 1_000_000}M' if vocab_size >= 1_000_000 else f'{vocab_size}'
    try:
        tokenizer = Tokenizer.from_file(f'tokenizers/{subset}-vocab_{size_str_mil}.json')
        print('Loaded. Vocabulary size:', tokenizer.get_vocab_size())
        return tokenizer
    except Exception:
        pass
    
    if data is None:
        data = load_dataset(dataset_name, subset, split='train')
    
    # Initialize a tokenizer
    tokenizer = Tokenizer(
        models.WordLevel(unk_token="<unk>"),
    )
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents(),
        normalizers.Strip(),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.Whitespace(),
    ])
    tokenizer.train_from_iterator(
        data['text'], 
        trainer=trainers.WordLevelTrainer(
            vocab_size=vocab_size,
            special_tokens=['<unk>', '<pad>', '<eos>', '<sos>'],
            show_progress=True,
        ),
    )
    tokenizer.save(f'tokenizers/{subset}-vocab_{size_str_mil}.json')
    print('Trained. Vocabulary size:', tokenizer.get_vocab_size())
    return tokenizer
