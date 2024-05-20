from torch.nn import functional as F
import torch.nn as nn, torch

class RNNConfig:
    n_hidden = 2
    n_layer = 1
    input_size = None
    output_size = None
    device = None

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        #Defining the layers
        self.rnn = nn.RNN(RNNConfig.input_size, RNNConfig.n_hidden, RNNConfig.n_layer, batch_first=True) # RNN Layer
        self.fc = nn.Linear(RNNConfig.n_hidden, RNNConfig.output_size) # Fully connected layer

    def forward(self, x, targets=None):
        # Apply RNN layer
        out, _ = self.rnn(x)

        # Squeeze the dimensions to remove the batch_first dimension
        out = self.fc(out)

        # Calculate loss
        loss = None if targets is None else F.cross_entropy(out, targets)
        return out, loss

    def predict(self, x, classes):
        out, _ = self(x)
        _, predicted = torch.max(out, dim=1)

        tag = classes[predicted.item()]

        probs = torch.softmax(out, dim=1)
        prob = probs[0][predicted.item()]
        confidence = prob.item()

        return tag, confidence
