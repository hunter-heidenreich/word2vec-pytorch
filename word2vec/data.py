"""Data module for word2vec."""

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from word2vec.tokenizer import get_tokenizer, add_spaces_around_foreign_characters


class WindowedDatasets(Dataset):
    """
    Dataset for word2vec.

    Args:
        dataset_name (str): Name of the dataset to use.
        subset (str): Name of the subset of the dataset to use.
        split (str): Name of the split to use.
        window_size (int): Size of the window to use.
        vocab_size (int): Size of the vocabulary to use.
        min_frequency (int): Minimum frequency of a token to be included in the vocabulary.
    """

    def __init__(
        self,
        dataset_name: str = "wikitext",
        subset: str = "wikitext-2-raw-v1",
        split: str = "train",
        window_size: int = 5,
        vocab_size: int = 1_000_000,
        min_frequency: int = 2,
    ):
        self.window_size = window_size
        self.dataset = load_dataset(dataset_name, subset, split=split)["text"]
        self.dataset = [
            add_spaces_around_foreign_characters(text)
            for text in self.dataset
            if text.strip()
        ]

        # get tokenizer
        self.tokenizer = get_tokenizer(
            data=self.dataset,
            dataset_name=dataset_name,
            subset=subset,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )

        # encode dataset
        self.centers = []
        self.contexts = []
        for text in tqdm(self.dataset, desc="Encoding dataset"):
            encoding = self.tokenizer.encode(text)
            datum = encoding.ids
            # pre- and post-pad with <pad> token
            datum = (
                [self.tokenizer.token_to_id("<pad>")] * self.window_size
                + datum
                + [self.tokenizer.token_to_id("<pad>")] * self.window_size
            )
            for i in range(self.window_size, len(datum) - self.window_size):
                center = datum[i]
                context = (
                    datum[i - self.window_size : i]
                    + datum[i + 1 : i + self.window_size + 1]
                )
                self.centers.append(center)
                self.contexts.append(context)
        print(f"Number of examples: {len(self):,}")

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return self.centers[idx], torch.LongTensor(self.contexts[idx])

    def get_tokenizer(self):
        """
        Get the tokenizer used by the dataset.

        Returns:
            Tokenizer: The tokenizer used by the dataset.
        """
        return self.tokenizer

    def get_vocab_size(self):
        """
        Get the size of the vocabulary used by the dataset.

        Returns:
            int: The size of the vocabulary.
        """
        return len(self.tokenizer)
