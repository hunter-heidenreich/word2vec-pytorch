"""Data module for word2vec."""

import logging
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from word2vec.tokenizer import get_tokenizer, add_spaces_around_foreign_characters
from word2vec.util import num2str

logger = logging.getLogger(__name__)


class WindowedDatasets(Dataset):
    """
    Dataset for word2vec model training with sliding window context sampling.

    Attributes:
        dataset_name (str): Name of the dataset.
        subset (str): Specific subset of the dataset.
        split (str): Dataset split (e.g., 'train').
        window_size (int): Number of words on each side of the center word.
        vocab_size (int): Maximum number of words in the vocabulary.
        min_frequency (int): Minimum occurrence of words to be included in the vocabulary.
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
        logger.info(
            f"Initializing WindowedDatasets with window_size={window_size}, vocab_size={num2str(vocab_size)}, min_frequency={min_frequency}"
        )

        self.window_size = window_size
        dataset = load_dataset(dataset_name, subset, split=split)["text"]
        logger.info(f"Loaded {num2str(len(dataset))} examples from dataset.")

        self.dataset = [
            add_spaces_around_foreign_characters(text)
            for text in dataset
            if text.strip()
        ]

        self.tokenizer = get_tokenizer(
            data=self.dataset,
            dataset_name=dataset_name,
            subset=subset,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )
        logger.info(
            f"Tokenizer loaded with vocabulary size of {num2str(self.tokenizer.get_vocab_size())}"
        )

        self.centers, self.contexts = self._encode_dataset()

    def _encode_dataset(self):
        centers = []
        contexts = []
        pad_id = self.tokenizer.token_to_id("<pad>")
        for text in tqdm(
            self.dataset, desc="Encoding dataset", disable=logger.level >= logging.INFO
        ):
            encoding = self.tokenizer.encode(text)
            datum = (
                [pad_id] * self.window_size + encoding.ids + [pad_id] * self.window_size
            )
            for i in range(self.window_size, len(datum) - self.window_size):
                center = datum[i]
                context = (
                    datum[i - self.window_size : i]
                    + datum[i + 1 : i + self.window_size + 1]
                )
                centers.append(center)
                contexts.append(context)
        logger.info(f"Generated {num2str(len(centers))} training samples.")
        return centers, contexts

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return self.centers[idx], torch.LongTensor(self.contexts[idx])

    def get_tokenizer(self):
        return self.tokenizer

    def get_vocab_size(self):
        return len(self.tokenizer)
