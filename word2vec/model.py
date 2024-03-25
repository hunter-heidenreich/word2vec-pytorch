import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.linear(x)
        return x


class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(SkipGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    import torch

    cbow = CBOW(10, 5)
    print(cbow)
    skip_gram = SkipGram(10, 5)
    print(skip_gram)

    x = torch.randint(0, 10, (2, 3))
    print(x)
    print(cbow(x))
    print(skip_gram(x))
