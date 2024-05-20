from torch.nn import functional as F
import torch.nn as nn, torch

class FeedForwardConfig:
    n_hidden = 2
    n_layer = 1
    input_size = None
    output_size = None

class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()

        # Create a list to hold the layers
        self.layers = nn.ModuleList()

        # Add input layer
        self.layers.append(nn.Linear(FeedForwardConfig.input_size, FeedForwardConfig.n_hidden))
        self.layers.append(nn.ReLU())

        # Add hidden layers
        for _ in range(FeedForwardConfig.n_layer - 1):
            self.layers.append(nn.Linear(FeedForwardConfig.n_hidden, FeedForwardConfig.n_hidden))
            self.layers.append(nn.ReLU())

        # Add output layer
        self.layers.append(nn.Linear(FeedForwardConfig.n_hidden, FeedForwardConfig.output_size))

    def forward(self, x, targets=None):
        out = x
        for layer in self.layers:
            out = layer(out)

        # Apply softmax to obtain probabilities for each class
        out = F.softmax(out, dim=-1)

        if targets is None:
            loss = None

        else:
            loss = F.cross_entropy(out, targets)

        return out, loss

    def predict(self, x, classes):
        out, _ = self(x)
        _, predicted = torch.max(out, dim=1)

        tag = classes[predicted.item()]

        probs = torch.softmax(out, dim=1)
        prob = probs[0][predicted.item()]
        confidence = prob.item()

        return tag, confidence
