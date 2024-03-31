"""Word2Vec models."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Output:
    """Output of CBOW model."""

    logits: torch.Tensor
    loss: torch.Tensor


class CBOW(nn.Module):
    """Continuous Bag of Words model.

    Naive implementation of CBOW model,
    - does not use hierarchical softmax or negative sampling.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
    ):
        super(CBOW, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        centers: torch.Tensor,
        contexts: torch.Tensor,
        return_loss: bool = True,
    ) -> Output:
        """Forward pass of CBOW model.

        Args:
            centers (torch.Tensor): Tensor of center word IDs (batch_size).
            contexts (torch.Tensor): Tensor of context word IDs (batch_size, 2 * window_size).
            return_loss (bool): If True, return loss. If False, return only logits. Default: True.

        Returns:
            Output: Output of CBOW model.
        """
        # Embed context words
        e_ctx = self.embedding(contexts)  # (batch_size, 2 * window_size, embedding_dim)

        # Average context word embeddings
        e_ctx = e_ctx.mean(dim=1)  # (batch_size, embedding_dim)

        # Predict center word
        p_ctr = self.linear(e_ctx)  # (batch_size, vocab_size)

        # Compute loss
        if return_loss:
            loss = self.loss_fn(p_ctr, centers)
            return Output(logits=p_ctr, loss=loss)

        return Output(logits=p_ctr, loss=None)


class SkipGram(nn.Module):
    """SkipGram model.

    Naive implementation of SkipGram model,
    - does not use hierarchical softmax or negative sampling."""

    def __init__(self, vocab_size: int, embedding_dim: int):
        super(SkipGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        centers: torch.Tensor,
        contexts: torch.Tensor,
        return_loss: bool = True,
    ) -> Output:
        """Forward pass of SkipGram model.

        Args:
            centers (torch.Tensor): Tensor of center word IDs (batch_size).
            contexts (torch.Tensor): Tensor of context word IDs (batch_size, 2 * window_size).
            return_loss (bool): If True, return loss. If False, return only logits. Default: True.

        Returns:
            Output: Output of SkipGram model."""
        # Embed center words
        e_ctr = self.embedding(centers)  # (batch_size, embedding_dim)

        # Predict context word distribution
        p_ctx = self.linear(e_ctr)  # (batch_size, vocab_size)

        # duplicate center word prediction for each context word
        p_ctx = p_ctx.unsqueeze(1).expand(
            -1, contexts.size(1), -1
        )  # (batch_size, 2 * window_size, vocab_size)

        # Compute loss
        if return_loss:
            loss = self.loss_fn(p_ctx, contexts)
            return Output(logits=p_ctx, loss=loss)

        return Output(logits=p_ctx, loss=None)
