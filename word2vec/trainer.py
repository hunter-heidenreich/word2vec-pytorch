"""Trainer for Word2Vec model."""

import logging
import os
from datetime import datetime
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from word2vec.data import WindowedDatasets
from word2vec.model import CBOW, SkipGram


logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration class for the Trainer."""

    # Training Parameters
    batch_size: int = 128
    num_epochs: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    logging_interval: int = 1

    # Evaluation Parameters
    validation_batch_size: int = 1024

    # WindowedDatasets Parameters
    dataset_name: str = "wikitext"
    subset: str = "wikitext-2-raw-v1"
    window_size: int = 10

    # Model Parameters
    is_skipgram: bool = False
    embedding_dim: int = 128
    vocab_size: int = 1_000_000
    min_frequency: int = 2

    # Saving Parameters
    save_dir: str = "models"
    run_name: str = None  # Automatically set if None


class Trainer:
    """Trainer class for training a Word2Vec model."""

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.run_dir = os.path.join(
            config.save_dir,
            config.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_dataset, self.train_loader = self._init_data("train")
        self.valid_dataset, self.valid_loader = self._init_data("validation")
        self.model = self._init_model()
        self.optimizer = self._init_optimizer()

    def _init_data(self, split):
        """Initialize training or validation data based on split."""
        dataset = WindowedDatasets(
            dataset_name=self.config.dataset_name,
            subset=self.config.subset,
            split=split,
            window_size=self.config.window_size,
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size
            if split == "train"
            else self.config.validation_batch_size,
            shuffle=(split == "train"),
        )
        return dataset, loader

    def _init_model(self):
        """Initialize the Word2Vec model."""
        model_cls = SkipGram if self.config.is_skipgram else CBOW
        model = model_cls(
            vocab_size=self.train_dataset.get_vocab_size(),
            embedding_dim=self.config.embedding_dim,
        ).to(self.device)
        logger.info(
            f"Initialized {model_cls.__name__} with {sum(p.numel() for p in model.parameters())} parameters."
        )
        return model

    def _init_optimizer(self):
        """Initialize the optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def train(self):
        """Trains the Word2Vec model using the configured Trainer."""
        for epoch in range(self.config.num_epochs):
            self._train_epoch(epoch)
            self._validate(epoch)
            self._save_model(epoch)

    def _train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        for i, (centers, contexts) in enumerate(
            tqdm(self.train_loader, desc=f"Epoch {epoch} Training")
        ):
            centers, contexts = centers.to(self.device), contexts.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(centers, contexts)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

            if i % self.config.logging_interval == 0:
                self.writer.add_scalar(
                    "Loss/train", loss.item(), epoch * len(self.train_loader) + i
                )
                self.writer.add_scalar(
                    "LearningRate",
                    self.optimizer.param_groups[0]["lr"],
                    epoch * len(self.train_loader) + i,
                )

    def _validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for centers, contexts in tqdm(
                self.valid_loader, desc=f"Epoch {epoch} Validation"
            ):
                centers, contexts = centers.to(self.device), contexts.to(self.device)
                loss = self.model(centers, contexts).loss
                total_loss += loss.item()
        avg_loss = total_loss / len(self.valid_loader)
        self.writer.add_scalar("Loss/validation", avg_loss, epoch)

    def _save_model(self, epoch):
        """Save the model's state dictionary."""
        model_path = os.path.join(self.run_dir, f"model_epoch_{epoch}.pt")
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
