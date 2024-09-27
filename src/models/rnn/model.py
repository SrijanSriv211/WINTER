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

        # Defining the layers
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

    def predict(self, x):
        out, _ = self(x)
        _, predicted = torch.max(out, dim=1)

        probs = torch.softmax(out, dim=1)
        prob = probs[0][predicted.item()]
        confidence = prob.item()

        return predicted.item(), confidence

class inference:
    def __init__(self, model_data, auto_load=False):
        self.state_dict = model_data["state_dict"]
        RNNConfig.device = model_data["device"]
        RNNConfig.input_size = model_data["config"]["input_size"]
        RNNConfig.output_size = model_data["config"]["output_size"]
        RNNConfig.n_hidden = model_data["config"]["n_hidden"]
        RNNConfig.n_layer = model_data["config"]["n_layer"]

        # automatically load the model
        if auto_load: self.load()

    def load(self):
        self.model = RNN() # create an instance of RNN

        # load the saved model state_dict
        self.model.load_state_dict(self.state_dict)
        self.model.to(RNNConfig.device)
        self.model.eval()  # set the model to evaluation mode

    # use the model for classification or other tasks
    def predict(self, X, classes):
        i, confidence = self.model.predict(self.prepare_context(X))
        tag = classes[i]

        return tag, confidence

    def prepare_context(self, X):
        X = X.reshape(1, X.shape[0])
        return torch.tensor(X).to(RNNConfig.device)
