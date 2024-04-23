"""Word2Vec models."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Output:
    """Output of Word2Vec model, containing logits and optionally the loss if computed.

    Attributes:
        logits (torch.Tensor): The logits from the model's output.
        loss (torch.Tensor): The computed loss if requested, otherwise None.
    """

    logits: torch.Tensor
    loss: torch.Tensor


class Word2VecBase(nn.Module):
    """Base class for Word2Vec models encapsulating common components.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimensionality of the embedding space.
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()


class CBOW(Word2VecBase):
    """Continuous Bag of Words model.

    A naive implementation of the CBOW model which simplifies the prediction complexity by
    not utilizing hierarchical softmax or negative sampling.

    Complexity for prediction is O(N x D + D x V), where:
        N is the number of context words (2 * window_size),
        D is the embedding dimension,
        V is the vocabulary size.
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__(vocab_size, embedding_dim)

    def forward(
        self, centers: torch.Tensor, contexts: torch.Tensor, return_loss: bool = True
    ) -> Output:
        """Perform forward pass of the CBOW model.

        Args:
            centers (torch.Tensor): Tensor of center word IDs (batch_size).
            contexts (torch.Tensor): Tensor of context word IDs (batch_size, 2 * window_size).
            return_loss (bool, optional): Flag to determine if loss is returned. Defaults to True.

        Returns:
            Output: An instance containing logits and, optionally, loss.
        """
        e_ctx = self.embedding(contexts).mean(dim=1)  # Average context word embeddings
        p_ctr = self.linear(e_ctx)  # Predict center word
        loss = self.loss_fn(p_ctr, centers) if return_loss else None
        return Output(logits=p_ctr, loss=loss)


class SkipGram(Word2VecBase):
    """SkipGram model.

    A naive implementation of the SkipGram model without the use of hierarchical softmax or negative sampling.

    Complexity for prediction is O(D + D x V), where:
        D is the embedding dimension,
        V is the vocabulary size.
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__(vocab_size, embedding_dim)

    def forward(
        self, centers: torch.Tensor, contexts: torch.Tensor, return_loss: bool = True
    ) -> Output:
        """Perform forward pass of the SkipGram model.

        Args:
            centers (torch.Tensor): Tensor of center word IDs (batch_size).
            contexts (torch.Tensor): Tensor of context word IDs (batch_size, 2 * window_size).
            return_loss (bool, optional): Flag to determine if loss is returned. Defaults to True.

        Returns:
            Output: An instance containing logits and, optionally, loss.
        """
        e_ctr = self.embedding(centers)
        p_ctx = self.linear(e_ctr).unsqueeze(1).expand(-1, contexts.size(1), -1)
        loss = (
            self.loss_fn(p_ctx.reshape(-1, p_ctx.size(-1)), contexts.reshape(-1))
            if return_loss
            else None
        )
        return Output(logits=p_ctx, loss=loss)
