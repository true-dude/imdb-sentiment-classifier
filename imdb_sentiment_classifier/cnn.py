import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        num_filters: int = 100,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=4, padding=1)
        self.conv3 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5, padding=1)

        self.fc = nn.Linear(num_filters * 3, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x1 = F.relu(self.conv1(x))
        x1 = torch.max(x1, dim=2).values

        x2 = F.relu(self.conv2(x))
        x2 = torch.max(x2, dim=2).values

        x3 = F.relu(self.conv3(x))
        x3 = torch.max(x3, dim=2).values

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
